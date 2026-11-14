"""Trailer unloading with physics: a KUKA KR210 R2700 (L150) -- or, with --robot fanuc, a FANUC CRX-10iA/L -- with a
suction tool pulls cartons off a wall packed floor to ceiling against the trailer's back wall and puts them on a
conveyor that carries them out of the trailer.

  * Physics on: every carton is a rigid body with an exact box collider and box-on-box friction, so pulling one
    out drags the box under it and its neighbours a little (the script measures and logs how far).
  * Picks start at the top: the next box is always one with nothing on top of it and nothing in front of it,
    highest first, so no box drops onto others.
  * The arm is driven from Python by resolved-rate Cartesian control (Jacobian from PhysX, damped least squares):
    every physics step it sends joint targets that move the suction cup in a straight line.
  * Collision detection on the robot's link meshes: contact reports on every link; if the arm or the end
    effector touches the trailer (walls, roof, floor, posts, rails), motion stops at once and the contact is
    logged. --collision-test drives the tool into the side wall at the end to show it.
  * Suction is a fixed joint between the tool flange and the box, created on contact and removed on release.
  * Gripper stoppers (--stoppers, on by default): before a pull the gripper extends three thin plates -- left, right
    and bottom -- just in front of the neighbouring boxes beside and under the chosen one, so friction cannot drag
    those out with it. They are kinematic bodies (immovable), retracted once the box is clear of the wall.
  * The robot starts in a collision-free pose found by forward kinematics on the asset's USD joints (the KR210's
    zero pose would reach into the wall).

    python download_simready_assets.py --out ~/simready --all-cartons
    ./python.sh trailer_unload.py --simready ~/simready --picks 4 --collision-test --out unload_out
    ./python.sh trailer_unload.py --simready ~/simready --gui

Outputs (--out): frames/rgb_*.jpg, unload.gif, wall_start.png, wall_end.png, boxes.json, events.json (picks,
neighbour displacement per pull, collisions).
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--simready", default=os.path.expanduser("~/simready"), help="SimReady download folder")
ap.add_argument("--out", default="unload_out")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--picks", type=int, default=4, help="boxes to unload (0 = all reachable)")
ap.add_argument("--rows", type=int, default=2)
ap.add_argument("--friction", type=float, nargs=2, default=[0.5, 0.4], help="box static / dynamic friction")
ap.add_argument("--robot", default="kuka", choices=["kuka", "fanuc"], help="KUKA KR210 L150 or FANUC CRX-10iA/L")
ap.add_argument("--stoppers", type=int, default=1, help="1: gripper extends left/right/bottom stoppers during a pull")
ap.add_argument("--density", type=float, default=60.0, help="kg/m^3 of a filled carton (mass clipped per robot)")
ap.add_argument("--belt-speed", type=float, default=0.5, help="conveyor speed (m/s), towards the door")
ap.add_argument("--speed", type=float, default=0.35, help="tool speed (m/s)")
ap.add_argument("--tool-len", type=float, default=None, help="suction cup length beyond the flange face (m)")
ap.add_argument("--tool-axis", default="auto", choices=["auto", "x", "y", "z", "-x", "-y", "-z"],
                help="end-effector link axis the suction tool points along (auto: away from the wrist)")
ap.add_argument("--pedestal", type=float, default=None, help="robot pedestal height (m) (default per robot)")
ap.add_argument("--standoff", type=float, default=None, help="robot base distance from the wall face (m)")
ap.add_argument("--collision-test", action="store_true", help="at the end, drive the tool into the side wall")
ap.add_argument("--frame-every", type=int, default=10, help="physics steps between saved frames (120 Hz)")
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
args, _ = ap.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui, "width": args.res[0], "height": args.res[1]})

import numpy as np                                                    # noqa: E402
import omni.replicator.core as rep                                    # noqa: E402
import omni.usd                                                       # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction            # noqa: E402
from omni.physx import get_physx_simulation_interface                 # noqa: E402
try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    from isaacsim.core.utils.nucleus import get_assets_root_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trailer_common as tc                                           # noqa: E402

rng = np.random.default_rng(args.seed)
DT = 1.0 / 120.0
# Per robot: asset, pedestal/plinth, distance to the wall, shoulder geometry for the reach check (wrist centre within
# `reach` of the shoulder, which sits sh_r in front of the base axis at height sh_z), roof margin of the suction point,
# max carton mass, conveyor position (beside the robot, along the right-hand wall).
ROBOTS = {
    "kuka": dict(name="KUKA KR210 R2700 (KR210 L150)",
                 usd=["/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd", "/Isaac/Robots/KUKA/KR210_L150/kr210_l150.usd"],
                 search=["/Isaac/Robots/Kuka", "/Isaac/Robots/KUKA"], key="kr210",
                 pedestal=0.30, plinth=0.90, standoff=1.85, tool_len=0.15, sh_z=0.675, sh_r=0.35, wrist=0.215,
                 reach=2.20,            # A2-A5 = 1.15 + 1.22 m, minus a margin
                 roof=0.28, mass_max=15.0, belt_y=0.88, belt_w=0.45, start_search=True),
    "fanuc": dict(name="FANUC CRX-10iA/L",
                  usd=["/Isaac/Robots/Fanuc/CRX10IAL/crx10ial.usd", "/Isaac/Robots/FANUC/CRX10IAL/crx10ial.usd"],
                  search=["/Isaac/Robots/Fanuc"], key="crx10",
                  pedestal=1.30, plinth=0.45, standoff=0.62, tool_len=0.10, sh_z=0.245, sh_r=0.0, wrist=0.0,
                  reach=1.28,           # 1418 mm reach minus a margin
                  roof=0.22, mass_max=5.0, belt_y=0.70, belt_w=0.55, start_search=False),
}
SPEC = ROBOTS[args.robot]
for k in ("pedestal", "standoff", "tool_len"):
    if getattr(args, k) is None:
        setattr(args, k, SPEC[k])
BELT_Y, BELT_W, BELT_TOP = SPEC["belt_y"], SPEC["belt_w"], 0.90
BELT_X_OUT = -2.5                                    # belt end, outside the trailer (dock side)
AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "-x": [-1, 0, 0], "-y": [0, -1, 0], "-z": [0, 0, -1]}
EVENTS = {"robot": SPEC["name"], "stoppers": bool(args.stoppers), "picks": [], "collisions": [], "skipped": []}


def log(*a):
    print("[unload]", *a, flush=True)


def quat_to_R(q):                                    # (w, x, y, z)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    m = Gf.Matrix3d(*R.T.flatten().tolist())          # Gf is row-vector convention
    q = m.ExtractRotation().GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ------------------------------------------------------------------------------------------- scene
def build_boxes(stage):
    plan = tc.plan_packed_wall(rng, rows=args.rows)
    models = tc.measure_assets(stage, tc.box_assets(args.simready))
    log(f"{len(models)} SimReady carton models" if models else "no SimReady cartons found: plain cardboard boxes")
    card = tc.material(stage, "/World/Looks/Card", (0.66, 0.50, 0.33), 0.88)
    pm = tc.physics_material(stage, "/World/Physics/Cardboard", *args.friction)
    UsdGeom.Xform.Define(stage, "/World/Boxes")
    used = {}
    for i, b in enumerate(plan):
        d, w, h = b["size_m"]
        cx, cy, cz = b["center_m"]
        path = f"/World/Boxes/box_{i:03d}"
        top = UsdGeom.Xform.Define(stage, path)
        xf = UsdGeom.Xformable(top)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz - h / 2 + 0.001))
        xf.AddRotateZOp().Set(b["yaw_deg"])
        # physics: one rigid body, an exact box collider (invisible), cardboard friction
        prim = top.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = float(np.clip(args.density * d * w * h, 1.0, SPEC["mass_max"]))
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        pr = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        pr.CreateSolverPositionIterationCountAttr(16)
        pr.CreateSolverVelocityIterationCountAttr(2)
        pr.CreateMaxDepenetrationVelocityAttr(1.0)          # resolve overlaps gently (no box shoots away)
        col = tc.cube(stage, path + "/collider", (0, 0, h / 2), (d, w, h), collide=True, phys_mat=pm)
        UsdGeom.Imageable(col).MakeInvisible()
        pc = PhysxSchema.PhysxCollisionAPI.Apply(col.GetPrim())
        pc.CreateContactOffsetAttr(0.004); pc.CreateRestOffsetAttr(0.0)
        model = "plain"
        if models:
            url = tc.pick_model(models, (d, w, h), rng, used)
            try:
                tc.visual_box(stage, path, url, (d, w, h))
                model = os.path.basename(url)
            except Exception as e:
                log(f"{url}: {e}")
                stage.RemovePrim(path + "/fit")
        if model == "plain":
            tc.cube(stage, path + "/visual", (0, 0, h / 2), (d, w, h), card)
        tc.add_label(prim, "carton")
        b.update(prim=path, model=model, mass_kg=mass)
    log(f"{len(plan)} boxes packed floor to ceiling in {args.rows} rows "
        f"(front faces {100 * (max(p['front_face_x_m'] for p in plan) - min(p['front_face_x_m'] for p in plan)):.0f} cm apart)")
    return plan


class Conveyor:
    """Kinematic belt slab with PhysX surface velocity (-x, out of the door), tiled SimReady conveyor visuals."""

    def __init__(self, stage, x_start, speed):
        self.x0, self.x1 = BELT_X_OUT, x_start
        length = self.x1 - self.x0
        cx = (self.x0 + self.x1) / 2
        url = None
        import glob
        c = sorted(glob.glob(os.path.join(args.simready, "conveyor_belt_a08", "simready_usd", "*.usd")))
        if c:
            url = c[0]
        n = max(1, int(math.ceil(length / 2.0)))
        seg = length / n
        UsdGeom.Xform.Define(stage, "/World/Conveyor")
        for k in range(n):
            p = f"/World/Conveyor/Seg{k:02d}"
            UsdGeom.Xform.Define(stage, p)
            UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(
                Gf.Vec3d(self.x0 + (k + 0.5) * seg, BELT_Y, 0.0))
            ok = False
            if url:
                try:
                    tc.visual_box(stage, p, url, (seg, BELT_W + 0.12, BELT_TOP))
                    ok = True
                except Exception as e:
                    log("conveyor model failed:", e)
            if not ok:
                frame = tc.material(stage, "/World/Looks/ConveyorFrame", (0.25, 0.27, 0.30), 0.5, 0.5)
                tc.cube(stage, p + "/frame", (0, 0, BELT_TOP / 2 - 0.02), (seg, BELT_W + 0.1, BELT_TOP - 0.04), frame)
        belt_mat = tc.material(stage, "/World/Looks/Belt", (0.08, 0.08, 0.08), 0.7)
        pm = tc.physics_material(stage, "/World/Physics/Belt", 0.9, 0.8)
        self.belt = tc.cube(stage, "/World/Conveyor/Belt", (cx, BELT_Y, BELT_TOP - 0.02), (length, BELT_W, 0.04),
                            belt_mat, "conveyor", collide=True, phys_mat=pm)
        if url:
            UsdGeom.Imageable(self.belt).MakeInvisible()
        bp = self.belt.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(bp).CreateKinematicEnabledAttr(True)
        self.sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(bp)
        self.sv.CreateSurfaceVelocityEnabledAttr(True)
        self.sv.CreateSurfaceVelocityAttr(Gf.Vec3f(-speed, 0.0, 0.0))
        self.speed = speed
        log(f"conveyor x {self.x0:.1f}..{self.x1:.1f} m at {speed} m/s (out of the door at x=0)")

    def add_transfer_plate(self, stage, x_end):
        """Fixed, slippery plate between the belt end and the wall: boxes are put down across it and the belt
        end, and the belt pulls them off; wall boxes nudged forward only ever touch this (non-moving) plate."""
        steel = tc.material(stage, "/World/Looks/Plate", (0.70, 0.71, 0.72), 0.3, 0.8)
        pm = tc.physics_material(stage, "/World/Physics/Plate", 0.05, 0.05)
        L_ = x_end - self.x1
        tc.cube(stage, "/World/Conveyor/TransferPlate", (self.x1 + L_ / 2, BELT_Y, BELT_TOP - 0.01),
                (L_, BELT_W, 0.02), steel, "conveyor", collide=True, phys_mat=pm)
        log(f"transfer plate x {self.x1:.2f}..{x_end:.2f} m")

    def set_speed(self, v):
        self.sv.GetSurfaceVelocityAttr().Set(Gf.Vec3f(-v, 0.0, 0.0))


def resolve_robot_url():
    """First listed asset path that exists on the asset server, else a search of the vendor folders."""
    assets = get_assets_root_path() or ""
    try:
        import omni.client
    except ImportError:
        return assets + SPEC["usd"][0]
    for p in SPEC["usd"]:
        r, _ = omni.client.stat(assets + p)
        if r == omni.client.Result.OK:
            return assets + p
    found = []
    def walk(url, depth):
        r, entries = omni.client.list(url)
        if r != omni.client.Result.OK:
            return
        for e in entries:
            u = url.rstrip("/") + "/" + e.relative_path
            if e.relative_path.lower().endswith((".usd", ".usda")) and SPEC["key"] in e.relative_path.lower():
                found.append(u)
            elif depth > 0 and "." not in e.relative_path:
                walk(u, depth - 1)
    for d in SPEC["search"]:
        walk(assets + d, 2)
    log(f"robot asset not at the listed paths; search found {found}")
    if not found:
        raise RuntimeError(f"no {SPEC['name']} asset under {assets}{SPEC['search']}")
    return sorted(found, key=len)[0]


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


class ChainFK:
    """Forward kinematics straight from the asset's USD joints (body0 * frame0 * R(axis, q) * frame1^-1), used to
    find a start pose before physics runs (no box is ever touched while searching)."""

    def __init__(self, stage, root):
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
        self.W0 = {b: _gf_np(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(b))) for b in bodies}
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
        self.bodies = [j["b1"] for j in order]
        self.sign = 1.0

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

    def tool_axis(self, ee):
        """Axis of the end-effector frame pointing away from the wrist (zero pose), as in Robot.setup."""
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

    def search(self, ee, tip_len, home, front, step_deg=10.0):
        """Grid over joints 2, 3, 5 (1, 4, 6 at zero): the pose whose tool tip is nearest `home` with the tool along
        +x, keeping every link origin (and the midpoints between them) clear of the roof, side walls and the wall."""
        ax = self.tool_axis(ee)
        if len(self.rev) < 5:
            return None, None
        vary = [self.rev[1], self.rev[2], self.rev[4]]
        grids = []
        for j in vary:
            lo = j["lo"] if j["lo"] is not None and j["lo"] > -1e3 else -180.0
            hi = j["hi"] if j["hi"] is not None and j["hi"] < 1e3 else 180.0
            grids.append(np.radians(np.arange(lo + 2, hi - 2 + 1e-6, step_deg)))
        best = (1e9, None, None)
        moving = self.bodies[1:]
        for a in grids[0]:
            for b in grids[1]:
                for c in grids[2]:
                    q = {vary[0]["name"]: a, vary[1]["name"]: b, vary[2]["name"]: c}
                    W = self.fk(q)
                    pts = [W[k][:3, 3] for k in moving]
                    pts += [(u + v) / 2 for u, v in zip(pts[:-1], pts[1:])]
                    P = np.array(pts)
                    R = W[ee][:3, :3]
                    a_w = R @ ax
                    tip = W[ee][:3, 3] + a_w * tip_len
                    if (P[:, 2].max() > tc.H - 0.40 or P[:, 2].min() < 0.15 or np.abs(P[:, 1]).max() > tc.W / 2 - 0.35
                            or P[:, 0].max() > front - 0.40 or tip[0] > front - 0.15 or tip[2] > tc.H - 0.40):
                        continue
                    cost = np.linalg.norm(tip - home) + 0.5 * (1 - a_w[0])
                    if cost < best[0]:
                        best = (cost, q, tip)
        return best[1], best[2]

    def apply_initial(self, q):
        """Start the articulation in q: joint state and drive target on every revolute joint (degrees)."""
        for j in self.rev:
            deg = float(np.degrees(self.sign * q.get(j["name"], 0.0)))
            try:
                PhysxSchema.JointStateAPI.Apply(j["prim"], "angular").CreatePositionAttr(deg)
            except Exception as e:
                log("joint state:", e)
            drv = UsdPhysics.DriveAPI.Get(j["prim"], "angular")
            if drv and drv.GetTargetPositionAttr():
                drv.GetTargetPositionAttr().Set(deg)


class Stoppers:
    """Three extendable stoppers on the gripper (left, right, bottom): thin kinematic plates that, during a pull,
    sit ~3 mm in front of the neighbouring boxes beside and under the chosen box, so friction from the box being
    pulled cannot drag them out. Created once at build time (no prims are added during the simulation), parked
    below the floor when retracted."""
    GAP, T = 0.003, 0.02                                 # clearance to the neighbour face; plate thickness (x)
    SIZES = {"left": (0.02, 0.08, 0.20), "right": (0.02, 0.08, 0.20), "bottom": (0.02, 0.20, 0.06)}

    def __init__(self, stage, park_xy):
        mat = tc.material(stage, "/World/Looks/Stopper", (0.95, 0.80, 0.05), 0.4, 0.5)
        pm = tc.physics_material(stage, "/World/Physics/Stopper", 0.15, 0.1)
        UsdGeom.Xform.Define(stage, "/World/Gripper")
        self.park = {}
        self.ops, self.paths, self.pos = {}, {}, {}
        for k, (sx, sy, sz) in self.SIZES.items():
            p = f"/World/Gripper/Stop_{k}"
            park = np.array([park_xy[0], park_xy[1] + {"left": 0.3, "right": -0.3, "bottom": 0.0}[k], -2.0])
            c = tc.cube(stage, p, park, (sx, sy, sz), mat, "gripper_stopper", collide=True, phys_mat=pm)
            prim = c.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
            pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            pc.CreateContactOffsetAttr(0.002); pc.CreateRestOffsetAttr(0.0)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            self.ops[k] = UsdGeom.Xformable(c).GetOrderedXformOps()[0]
            self.paths[k], self.park[k], self.pos[k] = p, park, park.copy()
        self.touch = set()

    def filter_robot(self, stage, link_paths):
        for p in self.paths.values():
            fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(p))
            for lp in link_paths:
                fp.CreateFilteredPairsRel().AddTarget(lp)

    def _set(self, k, p):
        self.pos[k] = np.asarray(p, float)
        self.ops[k].Set(Gf.Vec3d(*map(float, p)))

    def plan(self, target, others, H_roof):
        """Stopper poses for pulling `target` (dict with 'lo','hi' world AABB), `others` = AABBs of the rest of the
        wall. A stopper is only deployed if there is a neighbour in its region."""
        lo, hi = target["lo"], target["hi"]
        zc = float(np.clip((lo[2] + hi[2]) / 2, lo[2] + 0.11, hi[2] - 0.11)) if hi[2] - lo[2] > 0.22 else \
            (lo[2] + hi[2]) / 2
        yc = (lo[1] + hi[1]) / 2
        out = {}
        regions = {"left": ((hi[1] + 0.004, hi[1] + 0.084), (zc - 0.10, zc + 0.10)),
                   "right": ((lo[1] - 0.084, lo[1] - 0.004), (zc - 0.10, zc + 0.10)),
                   "bottom": ((yc - 0.10, yc + 0.10), (lo[2] - 0.064, lo[2] - 0.004))}
        for k, ((y0, y1), (z0, z1)) in regions.items():
            if z0 < 0.01 or abs(y0) > tc.W / 2 - 0.01 or abs(y1) > tc.W / 2 - 0.01:
                continue
            sy, sz = self.SIZES[k][1] / 2, self.SIZES[k][2] / 2
            yc_, zc_ = (y0 + y1) / 2, (z0 + z1) / 2
            hit = [o for o in others                      # every box the plate's footprint overlaps at all
                   if min(yc_ + sy, o["hi"][1]) - max(yc_ - sy, o["lo"][1]) > 0
                   and min(zc_ + sz, o["hi"][2]) - max(zc_ - sz, o["lo"][2]) > 0]
            if not hit:
                continue
            out[k] = (np.array([min(o["lo"][0] for o in hit) - self.GAP - self.T / 2, yc_, zc_]),
                      [o["name"] for o in hit])
        return out

    def extend(self, poses, sim, robot, steps=25, travel=0.05):
        """Pop out 5 cm short of the neighbour faces, then glide in (no impact)."""
        for k, (p, _) in poses.items():
            self._set(k, p - np.array([travel, 0, 0]))
        sim.step(robot)
        for s in range(1, steps + 1):
            e = 0.5 - 0.5 * math.cos(math.pi * s / steps)
            for k, (p, _) in poses.items():
                self._set(k, p - np.array([travel * (1 - e), 0, 0]))
            robot.hold_cmd()
            sim.step(robot)

    def retract(self):
        for k in self.paths:
            self._set(k, self.park[k])

    def contacts(self, pairs):
        for a0, c0, a1, c1 in pairs:
            for a, o in ((a0, a1), (a1, a0)):
                if a.startswith("/World/Gripper/") and o.startswith("/World/Boxes/"):
                    self.touch.add((a.split("_")[-1], o.split("/")[3]))


def read_contacts():
    headers, _ = get_physx_simulation_interface().get_contact_report()
    out = []
    for h in headers:
        out.append((str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(h.actor1)), str(PhysicsSchemaTools.intToSdfPath(h.collider1))))
    return out


def box_aabb(path, size):
    p, q = box_pose(path)
    R = quat_to_R(q)
    d, w, h = size
    P = np.array([p + R @ np.array([sx * d / 2, sy * w / 2, z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0, h)])
    return {"lo": P.min(0), "hi": P.max(0), "name": path.split("/")[-1]}


class Robot:
    """Industrial arm on a plinth, Cartesian resolved-rate control of the suction tool, contact-based stop."""

    def __init__(self, stage, base_xy, home=None, front=None):
        url = resolve_robot_url()
        self.stage = stage
        self.base = np.array([base_xy[0], base_xy[1], args.pedestal])
        if args.pedestal > 0.005:
            steel = tc.material(stage, "/World/Looks/Pedestal", (0.95, 0.75, 0.10), 0.5, 0.3)
            tc.cube(stage, "/World/Cell/Pedestal", (base_xy[0], base_xy[1], args.pedestal / 2),
                    (SPEC["plinth"], SPEC["plinth"], args.pedestal), steel, "robot_base", collide=True)
        wrap = UsdGeom.Xform.Define(stage, "/World/Robot")
        UsdGeom.Xformable(wrap).AddTranslateOp().Set(Gf.Vec3d(*self.base))
        stage.GetPrimAtPath("/World/Robot").GetReferences().AddReference(url)
        root = stage.GetPrimAtPath("/World/Robot")
        art_root = next((p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)), root)
        self.links = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not self.links:
            raise RuntimeError(f"no rigid links under the robot asset {url}")
        n_col = 0
        for p in self.links:                          # contact reports on every link (collision detection)
            cr = PhysxSchema.PhysxContactReportAPI.Apply(p)
            cr.CreateThresholdAttr(0.0)
        for p in Usd.PrimRange(root):
            n_col += p.HasAPI(UsdPhysics.CollisionAPI)
        self.link_paths = [str(p.GetPath()) for p in self.links]
        self.art_path = str(art_root.GetPath())
        log(f"robot {SPEC['name']} {url}: articulation {self.art_path}, {len(self.links)} links, "
            f"{n_col} collision shapes, base at {np.round(self.base, 2).tolist()}")
        self.stopped = None
        self.held = None
        self.box_touch = set()
        self.q_cmd = None
        self.q0 = None
        self.fk = None
        self.home_target, self.front = home, front
        if SPEC["start_search"] and home is not None:
            self.find_start(home, front)

    def _ee_guess(self, bodies):
        pick = [b for b in bodies if any(k in b.split("/")[-1].lower() for k in ("flange", "tool0", "link_6", "link6", "j6"))]
        return pick[-1] if pick else bodies[-1]

    def find_start(self, home, front):
        try:
            self.fk = ChainFK(self.stage, "/World/Robot")
            ee = self._ee_guess(self.fk.bodies)
            log(f"FK: {len(self.fk.rev)} revolute joints {[j['name'] for j in self.fk.rev]}, zero-pose check "
                f"{100 * self.fk.zero_error():.2f} cm, end effector {ee.split('/')[-1]}")
            q, tip = self.fk.search(ee, args.tool_len + 0.05, home, front)
            if q is None:
                log("FK: no collision-free start pose found; starting from the asset's zero pose")
                return
            self.q0, self.fk_ee = q, ee
            log(f"start pose {[(k, round(math.degrees(v))) for k, v in q.items()]} deg -> tool tip "
                f"{np.round(tip, 2).tolist()} (home {np.round(home, 2).tolist()})")
            self.fk.apply_initial(q)
        except Exception as e:
            import traceback
            traceback.print_exc()
            log("FK start-pose search failed:", e)

    def setup(self, world):
        self.art = world.scene.add(SingleArticulation(self.art_path, name=args.robot))
        world.reset()
        if self.q0 is not None:
            self._apply_start(world)
        self.view = self.art._articulation_view
        names = list(self.view.body_names)
        pick = [n for n in names if any(k in n.lower() for k in ("flange", "tool0", "link_6", "link6", "j6"))]
        self.ee_name = pick[-1] if pick else names[-1]
        self.ee_path = next(p for p in self.link_paths if p.endswith("/" + self.ee_name))
        self.ee = SingleXFormPrim(self.ee_path)
        self.ndof = self.art.num_dof
        self.arm = np.arange(min(6, self.ndof))
        J = self.view.get_jacobians()
        self.jidx = names.index(self.ee_name) - (1 if J.shape[1] == len(names) - 1 else 0)
        try:
            lim = self.view.get_dof_limits()[0]
            self.lo, self.hi = np.asarray(lim[:, 0]), np.asarray(lim[:, 1])
        except Exception:
            self.lo, self.hi = -np.full(self.ndof, np.pi * 2), np.full(self.ndof, np.pi * 2)
        log(f"links {names}; end effector {self.ee_name}; joints {list(self.art.dof_names)}")
        # tool axis: the end-effector axis pointing away from the wrist (or --tool-axis)
        ee_p, ee_q = self.ee.get_world_pose()
        R = quat_to_R(np.asarray(ee_q, float))
        name = args.tool_axis
        if name == "auto":
            name = "z"
            for lp in reversed(self.link_paths):
                if lp == self.ee_path:
                    continue
                pp, _ = SingleXFormPrim(lp).get_world_pose()
                v = R.T @ (np.asarray(ee_p, float) - np.asarray(pp, float))
                if np.linalg.norm(v) > 0.03:
                    v /= np.linalg.norm(v)
                    k = int(np.argmax(np.abs(v)))
                    name = ("" if v[k] > 0 else "-") + "xyz"[k]
                    break
        self.axis_local = np.array(AXES[name], float)
        # flange face: how far the end-effector link's own geometry extends along the tool axis
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        rng_ = cache.ComputeUntransformedBound(self.stage.GetPrimAtPath(self.ee_path)).ComputeAlignedRange()
        corners = [np.array([x, y, z]) for x in (rng_.GetMin()[0], rng_.GetMax()[0])
                   for y in (rng_.GetMin()[1], rng_.GetMax()[1]) for z in (rng_.GetMin()[2], rng_.GetMax()[2])]
        face = max(0.0, max(float(c @ self.axis_local) for c in corners)) if not rng_.IsEmpty() else 0.0
        self.tool_off = face + args.tool_len
        log(f"tool axis {name} of {self.ee_name}; flange face {100 * face:.1f} cm, suction tip "
            f"{100 * self.tool_off:.1f} cm from the link origin")
        cup = UsdGeom.Cylinder.Define(self.stage, self.ee_path + "/suction_cup")
        self.cup_r = 0.05 if args.robot == "kuka" else 0.04
        cup.CreateRadiusAttr(self.cup_r); cup.CreateHeightAttr(args.tool_len)
        cup.CreateAxisAttr(name[-1].upper())
        UsdGeom.Xformable(cup).AddTranslateOp().Set(Gf.Vec3d(*(self.axis_local * (face + args.tool_len / 2))))
        UsdPhysics.CollisionAPI.Apply(cup.GetPrim())        # the cup is a collision shape of the end-effector body
        self.cup_path = self.ee_path + "/suction_cup"
        world.reset()                                       # re-parse physics so the cup collider is included
        if self.q0 is not None:
            self.art.set_joint_positions(self.q0_vec)
            self.art.apply_action(ArticulationAction(joint_positions=self.q0_vec))
        self.q_cmd = np.asarray(self.art.get_joint_positions(), float)
        log(f"tool tip at start {np.round(self.tool()[0], 3).tolist()}, axis {np.round(self.tool()[1], 2).tolist()}")

    def _apply_start(self, world):
        """Put the articulation in the searched start pose; if the measured end-effector position disagrees with
        the FK prediction, the joint sign convention is the other one: flip it and search again."""
        names = list(self.art.dof_names)
        def vec(q):
            v = np.asarray(self.art.get_joint_positions(), float).copy()
            for i, n in enumerate(names):
                if n in q:
                    v[i] = self.fk.sign * q[n]
            return v
        for attempt in range(2):
            v = vec(self.q0)
            self.art.set_joints_default_state(positions=v)
            self.art.set_joint_positions(v)
            self.art.apply_action(ArticulationAction(joint_positions=v))
            world.step(render=False)
            p_meas = np.asarray(SingleXFormPrim(self.fk_ee).get_world_pose()[0], float)
            qa = {n: float(x) * self.fk.sign for n, x in zip(names, self.art.get_joint_positions())}
            p_fk = self.fk.fk(qa)[self.fk_ee][:3, 3]
            err = np.linalg.norm(p_meas - p_fk)
            log(f"start pose applied: end effector at {np.round(p_meas, 3).tolist()}, FK {np.round(p_fk, 3).tolist()} "
                f"({100 * err:.1f} cm)")
            if err < 0.05 or attempt == 1:
                break
            self.fk.sign = -self.fk.sign
            log("FK: joint sign flipped, searching again")
            q, tip = self.fk.search(self.fk_ee, args.tool_len + 0.05, self.home_target, self.front)
            if q is None:
                self.fk.sign = -self.fk.sign
                break
            self.q0 = q
        self.q0_vec = vec(self.q0)

    # ---- state
    def tool(self):
        p, q = self.ee.get_world_pose()
        R = quat_to_R(np.asarray(q, float))
        a = R @ self.axis_local
        return np.asarray(p, float) + a * self.tool_off, a, np.asarray(p, float), R

    def jacobian(self, tool_p, ee_p):
        J = np.asarray(self.view.get_jacobians())[0, self.jidx][:, self.arm]     # 6 x n (lin; ang)
        Jl, Ja = J[:3], J[3:]
        return np.vstack([Jl - skew(tool_p - ee_p) @ Ja, Ja])

    def contacts(self, pairs):
        """Robot-link contacts from the PhysX contact report: returns [(link, other)]."""
        out = []
        for a0, c0, a1, c1 in pairs:
            r0, r1 = a0.startswith("/World/Robot"), a1.startswith("/World/Robot")
            if r0 != r1:
                out.append((a0, c1) if r0 else (a1, c0))
        return out

    # ---- control
    def step_towards(self, p_des, a_des=np.array([1.0, 0.0, 0.0]), v_ff=np.zeros(3)):
        p, a, ee_p, _ = self.tool()
        v = v_ff + 4.0 * (p_des - p)
        n = np.linalg.norm(v)
        vmax = 1.5 * args.speed * (0.6 if self.held else 1.0)
        if n > vmax:
            v *= vmax / n
        w = 6.0 * np.cross(a, a_des)
        J = self.jacobian(p, ee_p)
        lam = 0.03
        dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), np.concatenate([v, w])) * DT
        dq = np.clip(dq, -1.5 * DT, 1.5 * DT)                    # <= 1.5 rad/s per joint
        q = self.q_cmd.copy()
        q[self.arm] = np.clip(q[self.arm] + dq, self.lo[self.arm], self.hi[self.arm])
        self.q_cmd = q
        self.art.apply_action(ArticulationAction(joint_positions=self.q_cmd))
        return np.linalg.norm(p_des - p), float(np.degrees(np.arccos(np.clip(a @ a_des, -1, 1))))

    def hold(self):
        self.q_cmd = np.asarray(self.art.get_joint_positions(), float)
        self.art.apply_action(ArticulationAction(joint_positions=self.q_cmd))

    def hold_cmd(self):                                     # keep the last command (no drift while waiting)
        self.art.apply_action(ArticulationAction(joint_positions=self.q_cmd))

    # ---- suction
    def _qf(self, R):
        q = R_to_quat(R)
        return Gf.Quatf(float(q[0]), *map(float, q[1:]))

    def grip(self, box_path, compliant=False):
        """Suction as a physics joint between the end-effector link and the box, at their current relative pose.
        compliant=True: every direction is locked except world-vertical translation (+/-3 cm), so while the box
        is slid out it rests on the box below with its own weight only (realistic friction, no arm pushing down).
        compliant=False: fully rigid, for lifting and carrying."""
        self.release(keep_filter=True)
        _, _, ee_p, R = self.tool()
        bp, bq = SingleXFormPrim(box_path).get_world_pose()
        bp = np.asarray(bp, float)
        Rb = quat_to_R(np.asarray(bq, float))
        if compliant:
            # joint frame = world axes at the box origin, so the free axis is world z
            j = UsdPhysics.Joint.Define(self.stage, "/World/Grip")
            j.CreateLocalPos0Attr(Gf.Vec3f(*(R.T @ (bp - ee_p))))
            j.CreateLocalRot0Attr(self._qf(R.T))
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            j.CreateLocalRot1Attr(self._qf(Rb.T))
            prim = j.GetPrim()
            for dof in ("transX", "transY", "rotX", "rotY", "rotZ"):
                lim = UsdPhysics.LimitAPI.Apply(prim, dof)
                lim.CreateLowAttr(1.0); lim.CreateHighAttr(-1.0)          # low > high: locked
            lim = UsdPhysics.LimitAPI.Apply(prim, "transZ")
            lim.CreateLowAttr(-0.03); lim.CreateHighAttr(0.03)
        else:
            j = UsdPhysics.FixedJoint.Define(self.stage, "/World/Grip")
            j.CreateLocalPos0Attr(Gf.Vec3f(*(R.T @ (bp - ee_p))))
            j.CreateLocalRot0Attr(self._qf(R.T @ Rb))
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateBody0Rel().SetTargets([self.ee_path])
        j.CreateBody1Rel().SetTargets([box_path])
        j.CreateExcludeFromArticulationAttr(True)
        if self.held != box_path:
            fp = UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath(box_path))
            for lp in self.link_paths:                      # the held box does not collide with the arm
                fp.CreateFilteredPairsRel().AddTarget(lp)
        self.held = box_path
        self.compliant = compliant

    def release(self, keep_filter=False):
        if self.stage.GetPrimAtPath("/World/Grip"):
            self.stage.RemovePrim("/World/Grip")
        if not keep_filter:
            self.held = None


# ------------------------------------------------------------------------------------------- run
class Sim:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
        try:
            pc = self.world.get_physics_context()
            pc.set_solver_type("TGS")
            pc.enable_gpu_dynamics(False)
            pc.enable_stablization(True)
        except Exception as e:
            log("physics context:", e)
        self.stage = omni.usd.get_context().get_stage()
        self.n = 0
        self.frames = 0
        self.stoppers = None
        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    def camera(self, eye, target):
        cam = UsdGeom.Camera.Define(self.stage, "/World/Camera")
        cam.CreateFocalLengthAttr(14.0); cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * args.res[1] / args.res[0])
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 60.0))
        UsdGeom.Xformable(cam).AddTransformOp().Set(tc.look_at_matrix(eye, target))
        rp = rep.create.render_product("/World/Camera", tuple(args.res))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([rp])

    def snap(self, name=None):
        from PIL import Image
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
            a = np.asarray(self.rgb.get_data())
            if a.ndim == 3:
                break
        img = Image.fromarray(a[..., :3])
        if name:
            img.save(os.path.join(args.out, name))
        else:
            img.save(os.path.join(args.out, "frames", f"rgb_{self.frames:04d}.jpg"), quality=85)
            self.frames += 1

    def step(self, robot=None):
        self.world.step(render=False)
        self.n += 1
        pairs = read_contacts() if robot is not None else []
        if self.stoppers is not None:
            self.stoppers.contacts(pairs)
        if robot is not None and robot.stopped is None:
            cs = robot.contacts(pairs)
            for l, o in cs:
                if o.startswith("/World/Boxes/") and not (robot.held and o.startswith(robot.held)):
                    robot.box_touch.add((l.split("/")[-1], "/".join(o.split("/")[:4]).split("/")[-1]))
            hits = [(l, o) for l, o in cs if o.startswith("/World/Trailer")]
            if hits:
                link, other = hits[0]
                robot.stopped = {"link": link, "hit": other, "t_s": round(self.n * DT, 3)}
                robot.hold()
                EVENTS["collisions"].append(robot.stopped)
                log(f"COLLISION: {link.split('/')[-1]} touched {other} at t={self.n * DT:.2f} s -> robot stopped")
        if self.n % args.frame_every == 0:
            self.snap()


def EVENTS_PICKED(plan):
    names = {p["box"] for p in EVENTS["picks"]}
    return [j for j, o in enumerate(plan) if o["prim"] in names]


def box_pose(path):
    p, q = SingleXFormPrim(path).get_world_pose()
    return np.asarray(p, float), np.asarray(q, float)


def free_boxes(plan, removed):
    """Boxes with nothing on top and nothing in front (towards the door). Only boxes that left the wall
    (picked or fallen) stop blocking: an unreachable box still blocks the ones under and behind it, so the
    robot never pulls a box out from under another."""
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


def main():
    sim = Sim()
    st = sim.stage
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    trailer_pm = tc.physics_material(st, "/World/Physics/Trailer", 0.5, 0.45)
    plan = build_boxes(st)
    front = min(b["front_face_x_m"] for b in plan)
    base_xy = (front - args.standoff, 0.0)
    cam_x = front - (4.4 if args.robot == "kuka" else 4.6)
    if args.robot == "kuka":
        home = np.array([front - 0.45, base_xy[1], 1.60])
    else:
        home = np.array([base_xy[0] + 0.35, base_xy[1], args.pedestal + 0.55])
    tc.build_trailer(st, collide=True, phys_mat=trailer_pm)
    tc.build_lights(st, cam_x)
    conveyor = Conveyor(st, front - 0.30, args.belt_speed)          # stops 30 cm short of the wall: a moving belt
                                                                    # drags anything that touches it, even its end
    conveyor.add_transfer_plate(st, front - 0.04)
    robot = Robot(st, base_xy, home, front)
    stoppers = None
    if args.stoppers:
        stoppers = Stoppers(st, (base_xy[0] - 1.0, 0.0))
        stoppers.filter_robot(st, robot.link_paths)
        sim.stoppers = stoppers
    if args.robot == "kuka":
        sim.camera((cam_x, -1.0, 2.3), (front - 0.2, 0.3, 1.3))
    else:
        sim.camera((cam_x, -0.75, 1.9), (front + 0.2, 0.25, 1.25))
    robot.setup(sim.world)
    sim.world.play()

    # ---- settle the wall (gravity, friction) and measure it
    p0 = {b["prim"]: box_pose(b["prim"])[0] for b in plan}
    for _ in range(240):
        sim.step()
        robot.hold()
    moved = max(np.linalg.norm(box_pose(k)[0] - v) for k, v in p0.items())
    log(f"wall settled: largest box movement {100 * moved:.1f} cm")
    sim.snap("wall_start.png")

    def move(target, a_des=np.array([1.0, 0.0, 0.0]), tol=0.01, timeout=8.0):
        """Straight-line tool motion at args.speed; False on collision stop or if the target is not reached."""
        p, _, _, _ = robot.tool()
        dist = np.linalg.norm(target - p)
        n_line = max(1, int(dist / (args.speed * (0.6 if robot.held else 1.0) * DT)))
        for k in range(int(timeout / DT)):
            s = min(1.0, (k + 1) / n_line)
            p_des = p + s * (target - p)
            v_ff = (target - p) / max(n_line * DT, 1e-6) if s < 1 else np.zeros(3)
            err, ang = robot.step_towards(p_des, a_des, v_ff)
            sim.step(robot)
            if robot.stopped:
                return False
            if s >= 1 and err < tol and ang < 3.0:
                return True
        if err < 0.03 and ang < 6.0:                     # close enough (heavy box, joint limits)
            return True
        log(f"target {np.round(target, 3).tolist()} not reached (error {100 * err:.1f} cm, {ang:.0f} deg)")
        return False

    def in_reach(face):
        """Wrist centre within reach of the shoulder (which turns with joint 1 towards the target)."""
        wc = face - np.array([robot.tool_off + SPEC["wrist"], 0, 0])
        dxy = wc[:2] - robot.base[:2]
        r = max(np.linalg.norm(dxy), 1e-6)
        sh = robot.base + np.array([0, 0, SPEC["sh_z"]]) + np.r_[dxy / r * SPEC["sh_r"], 0.0]
        return np.linalg.norm(wc - sh) < SPEC["reach"]

    if not move(home, timeout=12.0):
        log("could not reach the home pose; check --tool-axis / --pedestal")
    removed, unreachable, picks = set(), set(), 0
    while picks < (args.picks or 10 ** 6) and not robot.stopped:
        for j, o in enumerate(plan):                          # live poses; a box that left the wall is removed
            if j in removed:
                continue
            p, _ = box_pose(o["prim"])
            c = p + np.array([0, 0, o["size_m"][2] / 2])
            if (o["center_m"][2] - c[2] > 0.12) or (c[0] < front - 0.35):   # dropped or out of the wall
                removed.add(j)
                log(f"{o['prim']} left the wall (now at {np.round(c, 2).tolist()})")
                EVENTS.setdefault("fell", []).append(o["prim"])
            else:
                o["center_m"] = c.tolist()
        wall_front = min(o["center_m"][0] - o["half_extent_xy_m"][0] for j, o in enumerate(plan) if j not in removed)
        cand = [i for i in free_boxes(plan, removed) if i not in unreachable]
        reach = []
        for i in cand:
            p, _ = box_pose(plan[i]["prim"])
            d, w, h = plan[i]["size_m"]
            # suction point on the front face: its centre, but low enough that the wrist stays 22 cm below the roof
            gz = max(min(p[2] + h / 2, tc.H - SPEC["roof"]), p[2] + min(0.06, h / 2))
            face = np.array([p[0] - d / 2, p[1], gz])                # box origin = bottom centre
            if in_reach(face):
                reach.append((p[2] + h, -abs(face[1] - base_xy[1]), i, face))
            else:
                EVENTS["skipped"].append(plan[i]["prim"]); unreachable.add(i)
                log(f"{plan[i]['prim']} out of reach (face at {np.round(face, 2).tolist()}): left in place")
        if not reach:
            log("no free box within reach (the rest is out of reach or under/behind boxes that are)")
            break
        top_z, _, i, face = max(reach)                                 # highest first, then nearest the robot
        b = plan[i]
        d, w, h = b["size_m"]
        log(f"pick {picks + 1}: {b['prim']} ({b['model']}, {b['mass_kg']:.1f} kg) top at "
            f"{top_z:.2f} m, suction at {face[2]:.2f} m")
        before = {o["prim"]: box_pose(o["prim"])[0] for j, o in enumerate(plan) if j not in removed and j != i}
        # boxes touching the chosen one (beside / under / above, within 2 cm): their pull towards the robot is
        # the drag that the stoppers are meant to prevent
        tgt0 = box_aabb(b["prim"], b["size_m"])
        adjacent = []
        for j, o in enumerate(plan):
            if j in removed or j == i:
                continue
            a = box_aabb(o["prim"], o["size_m"])
            gap = np.maximum(a["lo"] - tgt0["hi"], tgt0["lo"] - a["hi"])     # per axis, < 0 = overlap
            if gap[1] < 0.02 and gap[2] < 0.02 and (gap[1] > -0.005 or gap[2] > -0.005) and gap[0] < 0:
                adjacent.append(o["prim"])
        # top boxes: pitch the suction tool up (up to 25 deg) so the wrist and forearm stay below the roof
        theta = math.radians(float(np.clip((face[2] - (tc.H - 0.42)) / 0.12 * 25.0, 0.0, 25.0))) \
            if args.robot == "kuka" else 0.0
        a_pick = np.array([math.cos(theta), 0.0, math.sin(theta)])
        face = face - np.array([robot.cup_r * math.sin(theta), 0, 0])    # lowest rim of the tilted cup touches first
        def phase(name):
            now = {k: box_pose(k)[0] for k in before}
            top = sorted(((100 * np.linalg.norm(now[k] - before[k]), k) for k in before), reverse=True)[:3]
            log(f"  after {name}: neighbours moved " + ", ".join(f"{k.split('/')[-1]} {c:.1f} cm" for c, k in top)
                + (f"; arm touched boxes {sorted(robot.box_touch)}" if robot.box_touch else "; arm touched no box"))
            robot.box_touch.clear()

        robot.box_touch.clear()
        if theta > 0:
            log(f"  tool pitched up {math.degrees(theta):.0f} deg (roof clearance)")
        approach = face - np.array([0.15, 0, 0])
        if not move(approach, a_pick) or not move(face - np.array([0.008, 0, 0]), a_pick, timeout=4.0):
            if robot.stopped:
                break
            unreachable.add(i); EVENTS["skipped"].append(b["prim"]); move(home); continue
        phase("approach")
        robot.grip(b["prim"], compliant=True)
        for _ in range(10):
            sim.step(robot)
        deployed = {}
        if stoppers is not None:                            # extend left / right / bottom stoppers
            tgt = box_aabb(b["prim"], b["size_m"])
            others = [box_aabb(o["prim"], o["size_m"]) for j, o in enumerate(plan) if j not in removed and j != i]
            deployed = stoppers.plan(tgt, others, tc.H)
            stoppers.touch.clear()
            if deployed:
                stoppers.extend(deployed, sim, robot)
            log("  stoppers: " + (", ".join(f"{k} at x={v[0][0]:.3f} in front of {v[1]}" for k, v in deployed.items())
                                  or "none needed (no neighbour beside or under)"))
        # 1) slide it out by ~40 % of its depth while it still rests on the box below: friction drags the
        #    neighbours a little; 2) lift 1 cm (top boxes have ~1.5 cm to the roof); 3) pull it clear
        slide = min(0.18, 0.4 * d)
        lift = min(0.01, max(0.0, tc.H - top_z - 0.004))
        pulled = move(face - np.array([slide, 0, 0]), a_pick, timeout=6.0)
        phase("slide (compliant grip)")
        robot.grip(b["prim"], compliant=False)              # rigid from here: lift and carry
        for _ in range(5):
            sim.step(robot)
        # pull only until the box's back is 3 cm in front of every box in its own column below it (then it can be
        # lowered straight down): pulling a high box all the way back folds the arm and raises the elbow
        lo_t, hi_t = tgt0["lo"], tgt0["hi"]
        col = [box_aabb(o["prim"], o["size_m"])["lo"][0] for j, o in enumerate(plan) if j not in removed and j != i
               and min(hi_t[1], box_aabb(o["prim"], o["size_m"])["hi"][1]) - max(lo_t[1], box_aabb(o["prim"], o["size_m"])["lo"][1]) > 0
               and box_aabb(o["prim"], o["size_m"])["lo"][2] < lo_t[2] + 0.02]
        col_front = min(col) if col else wall_front
        x_clear = min(col_front - 0.03 - d - robot.cup_r * math.sin(theta), face[0] - slide)   # box stays level
        pulled = pulled and move(face - np.array([slide, 0, -lift]), a_pick, timeout=3.0) and \
            move(np.array([x_clear, face[1], face[2] + lift]), a_pick, timeout=8.0)     # clear of its column
        phase("lift + pull clear")
        after = {k: box_pose(k)[0] for k in before}
        disp = sorted(((100 * np.linalg.norm(after[k] - before[k]), k) for k in before), reverse=True)[:3]
        log("  largest movement anywhere in the wall: " + ", ".join(f"{k.split('/')[-1]} {c:.1f} cm" for c, k in disp))
        adj = sorted(((100 * (before[k][0] - after[k][0]), k) for k in adjacent), reverse=True)
        log("  adjacent boxes pulled towards the robot: " +
            (", ".join(f"{k.split('/')[-1]} {c:.1f} cm" for c, k in adj) or "none adjacent"))
        if stoppers is not None and deployed:
            log(f"  stopper contacts: {sorted(stoppers.touch) or 'none'}")
        if stoppers is not None:
            stoppers.retract()
            sim.step(robot)
        if robot.stopped:                                   # collision: hold the box where it is
            break
        x_after_pull = robot.tool()[0][0]
        if not pulled:
            log("  pull fell short; carrying it to the belt anyway")
        # to the conveyor: belt stopped while placing, tool keeps pointing +x (box hangs in front of it)
        conveyor.set_speed(0.0)
        off = face[2] - (top_z - h / 2)                     # suction point relative to the box centre (<= 0)
        carry_z = max(BELT_TOP + h / 2 + 0.40, args.pedestal + h / 2 + 0.12) + off   # box bottom above the pedestal
        clear_x = wall_front - 0.08 - d
        # put it down in front of the robot, on the transfer plate and the belt end (box front 8 cm from the wall)
        tool_x = wall_front - 0.08 - d
        over = np.array([tool_x, BELT_Y, carry_z])
        place = np.array([tool_x, BELT_Y, BELT_TOP + h / 2 + 0.03 + off])
        # every leg is attempted even if one falls short, so the box is only ever released over the belt
        legs = [move(np.array([x_after_pull, face[1], carry_z]), timeout=8.0) if not robot.stopped else False,  # down, level
                move(np.array([clear_x, face[1], carry_z]), timeout=8.0) if not robot.stopped else False,
                move(over, timeout=8.0) if not robot.stopped else False,
                move(place, timeout=6.0) if not robot.stopped else False]
        ok = all(legs)
        if robot.stopped:                                   # collision while carrying: keep holding, stop
            break
        robot.release()
        for _ in range(30):
            sim.step(robot)
        move(place - np.array([0.10, 0, 0]), timeout=3.0)  # back off the box, then up
        move(place + np.array([-0.10, 0, 0.30]), timeout=4.0)
        conveyor.set_speed(args.belt_speed)
        removed.add(i); picks += 1
        EVENTS["picks"].append({"box": b["prim"], "model": b["model"], "mass_kg": round(b["mass_kg"], 2),
                                "placed": bool(ok), "neighbours_dragged_cm": [[k, round(c, 2)] for c, k in disp],
                                "adjacent_pulled_cm": [[k, round(c, 2)] for c, k in adj],
                                "tool_pitch_deg": round(math.degrees(theta), 1),
                                "stoppers": {k: v[1] for k, v in deployed.items()},
                                "stopper_contacts": sorted(stoppers.touch) if stoppers is not None else []})
        if not move(home, timeout=10.0) and robot.stopped:
            break

    conveyor.set_speed(max(args.belt_speed, 1.5))                     # let the belt carry the boxes out
    fe = args.frame_every
    args.frame_every = 3 * fe
    for _ in range(int(12.0 / DT)):
        sim.step(robot)
        if all(box_pose(plan[j]["prim"])[0][0] < 0 for j in EVENTS_PICKED(plan)):
            break
    args.frame_every = fe
    out_boxes = [o["prim"] for o in plan if box_pose(o["prim"])[0][0] < 0]
    log(f"{picks} boxes picked, {len(out_boxes)} carried out of the trailer by the conveyor")

    if args.collision_test and not robot.stopped:
        log("collision test: driving the tool towards the left trailer wall")
        move(home, timeout=10.0)
        if args.robot == "fanuc":
            start = np.array([base_xy[0] + 0.25, base_xy[1] - 0.45, args.pedestal + 0.25])
            move(start, a_des=np.array([0.0, -1.0, 0.0]), timeout=10.0)
            move(np.array([start[0], -tc.W / 2 - 0.2, start[2]]), a_des=np.array([0.0, -1.0, 0.0]), timeout=10.0)
        else:                                       # tool still pointing at the wall, sideways into the left wall
            move(np.array([home[0], -tc.W / 2 - 0.2, home[2]]), timeout=12.0)
        for _ in range(60):                                            # show that it stays stopped
            sim.step(robot)
        if not robot.stopped:
            log("collision test: no contact detected")

    sim.snap("wall_end.png")
    json.dump(plan, open(os.path.join(args.out, "boxes.json"), "w"), indent=1, default=float)
    json.dump(EVENTS, open(os.path.join(args.out, "events.json"), "w"), indent=1, default=float)
    make_gif()
    log(f"done: {sim.frames} frames, {picks} picks, collisions {len(EVENTS['collisions'])}")
    while args.gui and simulation_app.is_running():
        sim.world.step(render=True)


def make_gif():
    from PIL import Image
    import glob
    f = sorted(glob.glob(os.path.join(args.out, "frames", "rgb_*.jpg")))
    if not f:
        return
    step = max(1, len(f) // 200)
    imgs = [Image.open(p).convert("RGB").resize((480, 270)) for p in f[::step]]
    imgs[0].save(os.path.join(args.out, "unload.gif"), save_all=True, append_images=imgs[1:], duration=80, loop=0)


if __name__ == "__main__":
    import traceback
    ok = False
    try:
        main()
        ok = True
    except Exception:
        traceback.print_exc()
    finally:
        try:
            json.dump(EVENTS, open(os.path.join(args.out, "events.json"), "w"), indent=1, default=float)
        except Exception:
            pass
        sys.stdout.flush()
        simulation_app.close()
        os._exit(0 if ok else 1)
