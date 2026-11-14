"""Isaac-free kinematics shared by trailer_unload.py and tools/arm_clearance.py (needs only pxr + numpy).

  * ChainFK: forward kinematics straight from a robot asset's USD joints, plus the collision-mesh points of every
    link, so arm-vs-trailer clearance can be checked before (or without) running physics.
  * Resolved-rate IK (damped least squares) that mirrors the controller in trailer_unload.py.
  * pick_waypoints(): the tool poses of one pick (approach, slide, lift, pull, lower, carry, place), used by both the
    simulation and the offline clearance check, so they always agree.
"""
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def quat_to_R(q):                                    # (w, x, y, z)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _gf_np(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], float).T     # Gf row-vector -> column


def _T(pos, rot):
    M = np.eye(4)
    M[:3, :3] = quat_to_R([rot.GetReal(), *rot.GetImaginary()])
    M[:3, 3] = [float(v) for v in pos]
    return M


def _rot(axis, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    M = np.eye(4)
    j, k = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    M[j, j], M[j, k], M[k, j], M[k, k] = c, -s_, s_, c
    return M


def rot_between(u, v):
    """Smallest rotation taking unit vector u to unit vector v."""
    u = np.asarray(u, float) / np.linalg.norm(u); v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(np.clip(u @ v, -1, 1)); k = np.cross(u, v); s_ = np.linalg.norm(k)
    if s_ < 1e-9:
        if c > 0:
            return np.eye(3)
        k = np.cross(u, [0, 0, 1] if abs(u[2]) < 0.9 else [1, 0, 0]); k /= np.linalg.norm(k)
        return 2 * np.outer(k, k) - np.eye(3)
    k /= s_
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s_ * K + (1 - c) * K @ K


def orientation_error(R, a_local, a_des, R_ref):
    """Rotation vector from the current tool orientation R to the desired one: the reference orientation R_ref
    (tool along +x, fixed roll) turned the shortest way onto a_des. Constraining the roll too keeps a held box from
    swinging about the tool axis. Without R_ref: axis only."""
    if R_ref is None:
        return np.cross(R @ a_local, a_des)
    R_des = rot_between([1.0, 0.0, 0.0], a_des) @ R_ref
    return _rotvec(R_des @ R.T)


def reference_orientation(R, a_local):
    """R turned so that the tool axis points exactly along +x (keeps its roll)."""
    return rot_between(R @ a_local, [1.0, 0.0, 0.0]) @ R


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (a / (2 * math.sin(a))) if a < math.pi - 1e-6 else v


class ChainFK:
    """Forward kinematics from the USD joints: body1 = body0 * frame0 * R(axis, q) * frame1^-1.
    base_T (4x4) is applied on top of the stage pose (offline use: place the asset in the trailer)."""

    def __init__(self, stage, root, base_T=None, log=print):
        xc = UsdGeom.XformCache()
        js = []
        for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            ex = j.GetExcludeFromArticulationAttr().Get()
            if not b1 or ex:
                continue
            rev = p.IsA(UsdPhysics.RevoluteJoint)
            rj = UsdPhysics.RevoluteJoint(p) if rev else None
            js.append(dict(name=p.GetName(), prim=p, b0=b0[0] if b0 else None, b1=b1[0], rev=rev,
                           axis=str(rj.GetAxisAttr().Get()) if rev else None,
                           lo=rj.GetLowerLimitAttr().Get() if rev else None,
                           hi=rj.GetUpperLimitAttr().Get() if rev else None,
                           F0=_T(j.GetLocalPos0Attr().Get() or (0, 0, 0), j.GetLocalRot0Attr().Get() or Gf.Quatf(1)),
                           F1=_T(j.GetLocalPos1Attr().Get() or (0, 0, 0), j.GetLocalRot1Attr().Get() or Gf.Quatf(1))))
        bodies = {j["b1"] for j in js} | {j["b0"] for j in js if j["b0"]}
        B = np.eye(4) if base_T is None else base_T
        self.W0 = {b: B @ _gf_np(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(b))) for b in bodies}
        self.roots = bodies - {j["b1"] for j in js if j["b0"]}
        order, known = [], set(self.roots)                # topological order: base outwards
        rest = [j for j in js if j["b0"]]
        while rest:
            nxt = [j for j in rest if j["b0"] in known]
            if not nxt:
                break
            for j in nxt:
                order.append(j); known.add(j["b1"]); rest.remove(j)
        self.joints = order
        self.rev = [j for j in order if j["rev"]]
        self.names = [j["name"] for j in self.rev]
        self.bodies = [j["b1"] for j in order]
        self.sign = 1.0
        self.stage, self.log = stage, log

    def fk(self, q):                                       # q: {joint name: rad}
        W = {b: self.W0[b] for b in self.roots}
        for j in self.joints:
            a = self.sign * q.get(j["name"], 0.0) if j["rev"] else 0.0
            R = _rot(j["axis"], a) if j["rev"] else np.eye(4)
            W[j["b1"]] = W[j["b0"]] @ j["F0"] @ R @ np.linalg.inv(j["F1"])
        return W

    def zero_error(self):
        W = self.fk({})
        return max((np.linalg.norm(W[b][:3, 3] - self.W0[b][:3, 3]) for b in self.bodies), default=0.0)

    def limits(self):
        lo = np.radians([j["lo"] if j["lo"] is not None and j["lo"] > -1e3 else -180.0 for j in self.rev])
        hi = np.radians([j["hi"] if j["hi"] is not None and j["hi"] < 1e3 else 180.0 for j in self.rev])
        return lo, hi

    def tool_axis(self, ee):
        """Axis of the end-effector frame pointing away from the wrist (zero pose)."""
        W = self.W0
        p_ee = W[ee][:3, 3]
        for b in reversed(self.bodies):
            if b == ee:
                continue
            v = W[ee][:3, :3].T @ (p_ee - W[b][:3, 3])
            if np.linalg.norm(v) > 0.03:
                v = v / np.linalg.norm(v)
                k = int(np.argmax(np.abs(v)))
                e = np.zeros(3); e[k] = np.sign(v[k])
                return e
        return np.array([0.0, 0.0, 1.0])

    def link_points(self, max_pts=400):
        """Collision-mesh vertices of every body, in the body's frame (subsampled), for clearance checks."""
        xc = UsdGeom.XformCache()
        out = {}
        for b in [*self.roots, *self.bodies]:
            bp = self.stage.GetPrimAtPath(b)
            Wb = _gf_np(xc.GetLocalToWorldTransform(bp))
            pts = []
            for c in Usd.PrimRange(bp):
                if c != bp and c.HasAPI(UsdPhysics.RigidBodyAPI):
                    break
                if c.HasAPI(UsdPhysics.CollisionAPI) and c.IsA(UsdGeom.Mesh):
                    P = np.asarray(UsdGeom.Mesh(c).GetPointsAttr().Get() or [], float)
                    if len(P):
                        M = np.linalg.inv(Wb) @ _gf_np(xc.GetLocalToWorldTransform(c))
                        pts.append(P @ M[:3, :3].T + M[:3, 3])
            if pts:
                P = np.vstack(pts)
                if len(P) > max_pts:
                    P = P[np.random.default_rng(0).choice(len(P), max_pts, replace=False)]
                out[b] = P
        return out

    def search(self, ee, tip_len, home, front, H, W_, step_deg=10.0):
        """Grid over joints 2, 3, 5 (1, 4, 6 at zero): the pose whose tool tip is nearest `home` with the tool along
        +x, keeping every link origin (and the midpoints between them) clear of the roof, side walls and the wall."""
        ax = self.tool_axis(ee)
        if len(self.rev) < 5:
            return None, None
        vary = [self.rev[1], self.rev[2], self.rev[4]]
        lo, hi = self.limits()
        idx = [1, 2, 4]
        grids = [np.arange(lo[k] + 0.035, hi[k] - 0.035 + 1e-6, math.radians(step_deg)) for k in idx]
        best = (1e9, None, None)
        moving = self.bodies[1:]
        for a in grids[0]:
            for b in grids[1]:
                for c in grids[2]:
                    q = {vary[0]["name"]: a, vary[1]["name"]: b, vary[2]["name"]: c}
                    Wd = self.fk(q)
                    pts = [Wd[k][:3, 3] for k in moving]
                    pts += [(u + v) / 2 for u, v in zip(pts[:-1], pts[1:])]
                    P = np.array(pts)
                    a_w = Wd[ee][:3, :3] @ ax
                    tip = Wd[ee][:3, 3] + a_w * tip_len
                    if (P[:, 2].max() > H - 0.40 or P[:, 2].min() < 0.15 or np.abs(P[:, 1]).max() > W_ / 2 - 0.35
                            or P[:, 0].max() > front - 0.40 or tip[0] > front - 0.15 or tip[2] > H - 0.40):
                        continue
                    cost = np.linalg.norm(tip - home) + 0.5 * (1 - a_w[0])
                    if cost < best[0]:
                        best = (cost, q, tip)
        return best[1], best[2]


class IKTracker:
    """Offline twin of Robot.step_towards: DLS resolved-rate steps along straight tool-tip lines, the tool axis
    turning towards a_des at the controller's rate, joints clipped to their limits."""

    def __init__(self, fk, ee, axis_local, tool_off, q0, full_orientation=True):
        self.fk, self.ee, self.ax, self.off = fk, ee, np.asarray(axis_local, float), tool_off
        self.lo, self.hi = fk.limits()
        self.q = np.array([q0.get(n, 0.0) for n in fk.names], float)
        self.R_ref = reference_orientation(self.tool()[2], self.ax) if full_orientation else None

    def qmap(self, q=None):
        return dict(zip(self.fk.names, self.q if q is None else q))

    def tool(self, q=None):
        W = self.fk.fk(self.qmap(q))
        R = W[self.ee][:3, :3]
        a = R @ self.ax
        return W[self.ee][:3, 3] + a * self.off, a, R, W

    def jac(self):
        p, _, R, _ = self.tool()
        J = np.zeros((6, len(self.q)))
        h = 1e-5
        for i in range(len(self.q)):
            dq = self.q.copy(); dq[i] += h
            p2, _, R2, _ = self.tool(dq)
            J[:3, i] = (p2 - p) / h
            J[3:, i] = _rotvec(R2 @ R.T) / h
        return J

    def move(self, target, a_des=(1.0, 0.0, 0.0), step=0.02, iters=3, on_step=None):
        a_des = np.asarray(a_des, float) / np.linalg.norm(a_des)
        p0, _, _, _ = self.tool()
        n = max(1, int(np.linalg.norm(target - p0) / step))
        for k in range(1, n + 1):
            p_des = p0 + (target - p0) * k / n
            for _ in range(iters):
                p, a, R, _ = self.tool()
                v = p_des - p
                w = 0.57 * orientation_error(R, self.ax, a_des, self.R_ref)   # ~ the controller's 6/s rate per step
                J = self.jac()
                dq = J.T @ np.linalg.solve(J @ J.T + 0.03 ** 2 * np.eye(6), np.concatenate([v, w]))
                self.q = np.clip(self.q + dq, self.lo, self.hi)
            if on_step:
                on_step(self)
        p, a, _, _ = self.tool()
        return float(np.linalg.norm(target - p)), float(math.degrees(math.acos(np.clip(a @ a_des, -1, 1))))


def free_boxes(plan, removed):
    """Boxes with nothing on top and nothing in front (towards the door). Only boxes that left the wall
    (picked or fallen) stop blocking: an unreachable box still blocks the ones under and behind it."""
    out = []
    for i, b in enumerate(plan):
        if i in removed:
            continue
        d, w, h = b["size_m"]; cx, cy, cz = b["center_m"]
        blocked = False
        for j, o in enumerate(plan):
            if j == i or j in removed:
                continue
            od, ow, oh = o["size_m"]; ox, oy, oz = o["center_m"]
            y_ov = abs(oy - cy) < (w + ow) / 2 - 0.01
            x_ov = abs(ox - cx) < (d + od) / 2 - 0.01
            z_ov = abs(oz - cz) < (h + oh) / 2 - 0.01
            if y_ov and x_ov and oz > cz:                    # on top
                blocked = True; break
            if y_ov and z_ov and ox < cx:                    # in front
                blocked = True; break
        if not blocked:
            out.append(i)
    return out


def suction_face(center, size, H, roof):
    """Suction point on the front face: its centre, but at most `roof` below the roof (and >= 6 cm above the
    box bottom). center = box centre."""
    d, w, h = size
    bottom = center[2] - h / 2
    gz = max(min(center[2], H - roof), bottom + min(0.06, h / 2))
    return np.array([center[0] - d / 2, center[1], gz])


def pick_waypoints(face, size, top_z, tgt_aabb, col_front, wall_front, H, cup_r, pitch, belt_y, belt_top, pedestal):
    """Tool-tip poses (point, axis) for one pick. pitch=True tilts the tool up to 25 deg for high suction points so
    the wrist and forearm stay under the roof; the box is pulled only until it clears its own column, lowered while
    still tilted, levelled, carried over the belt and put down."""
    d, w, h = size
    theta = math.radians(float(np.clip((face[2] - (H - 0.42)) / 0.12 * 25.0, 0.0, 25.0))) if pitch else 0.0
    a_pick = np.array([math.cos(theta), 0.0, math.sin(theta)])
    x_ = np.array([1.0, 0.0, 0.0])
    f = face - np.array([cup_r * math.sin(theta), 0, 0])          # lowest rim of the tilted cup touches first
    slide = min(0.18, 0.4 * d)
    lift = min(0.01, max(0.0, H - top_z - 0.004))
    x_clear = min(col_front - 0.03 - d - cup_r * math.sin(theta), f[0] - slide)
    off = face[2] - (top_z - h / 2)                               # suction point relative to the box centre
    carry_z = max(belt_top + h / 2 + 0.40, pedestal + h / 2 + 0.12) + off
    tool_x = wall_front - 0.08 - d
    place = np.array([tool_x, belt_y, belt_top + h / 2 + 0.03 + off])
    return dict(theta=theta, a_pick=a_pick, face=f, slide=slide, lift=lift, carry_z=carry_z, legs=[
        ("approach", f - np.array([0.15, 0, 0]), a_pick),
        ("contact", f - np.array([0.008, 0, 0]), a_pick),
        ("slide", f - np.array([slide, 0, 0]), a_pick),
        ("lift", f - np.array([slide, 0, -lift]), a_pick),
        ("pull", np.array([x_clear, f[1], f[2] + lift]), a_pick),
        ("lower", np.array([x_clear, f[1], carry_z]), a_pick),
        ("clear", np.array([tool_x, f[1], carry_z]), x_),
        ("over", np.array([tool_x, belt_y, carry_z]), x_),
        ("place", place, x_),
        ("back", place - np.array([0.10, 0, 0]), x_),
        ("up", place + np.array([-0.10, 0, 0.30]), x_),
    ])
