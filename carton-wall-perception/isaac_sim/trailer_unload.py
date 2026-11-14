"""Trailer unloading with physics: a FANUC CRX-10iA/L with a suction tool pulls cartons off a wall packed floor to
ceiling against the trailer's back wall and puts them on a conveyor that carries them out of the trailer.

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
ap.add_argument("--density", type=float, default=60.0, help="kg/m^3 of a filled carton (mass clipped to 1-5 kg)")
ap.add_argument("--belt-speed", type=float, default=0.5, help="conveyor speed (m/s), towards the door")
ap.add_argument("--speed", type=float, default=0.35, help="tool speed (m/s)")
ap.add_argument("--tool-len", type=float, default=0.10, help="suction cup length beyond the flange face (m)")
ap.add_argument("--tool-axis", default="auto", choices=["auto", "x", "y", "z", "-x", "-y", "-z"],
                help="end-effector link axis the suction tool points along (auto: away from the wrist)")
ap.add_argument("--pedestal", type=float, default=1.05, help="robot pedestal height (m)")
ap.add_argument("--standoff", type=float, default=0.72, help="robot base distance from the wall face (m)")
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
FANUC = ["/Isaac/Robots/Fanuc/CRX10IAL/crx10ial.usd", "/Isaac/Robots/FANUC/CRX10IAL/crx10ial.usd"]
REACH = 1.40                                         # CRX-10iA/L reach 1418 mm
BELT_Y, BELT_W, BELT_TOP = 0.80, 0.55, 0.72          # conveyor along the right-hand wall, towards the door
BELT_X_OUT = -2.5                                    # belt end, outside the trailer (dock side)
AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "-x": [-1, 0, 0], "-y": [0, -1, 0], "-z": [0, 0, -1]}
EVENTS = {"picks": [], "collisions": [], "skipped": []}


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
        mass = float(np.clip(args.density * d * w * h, 1.0, 5.0))
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

    def set_speed(self, v):
        self.sv.GetSurfaceVelocityAttr().Set(Gf.Vec3f(-v, 0.0, 0.0))


class Fanuc:
    """CRX-10iA/L on a pedestal, Cartesian resolved-rate control of the suction tool, contact-based stop."""

    def __init__(self, stage, base_xy):
        assets = get_assets_root_path() or ""
        url = next((assets + p for p in FANUC if assets), None)
        self.stage = stage
        self.base = np.array([base_xy[0], base_xy[1], args.pedestal])
        steel = tc.material(stage, "/World/Looks/Pedestal", (0.95, 0.75, 0.10), 0.5, 0.3)
        tc.cube(stage, "/World/Cell/Pedestal", (base_xy[0], base_xy[1], args.pedestal / 2),
                (0.45, 0.45, args.pedestal), steel, "robot_base", collide=True)
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
        log(f"robot {url}: articulation {self.art_path}, {len(self.links)} links, {n_col} collision shapes")
        self.stopped = None
        self.held = None
        self.box_touch = set()
        self.q_cmd = None

    def setup(self, world):
        self.art = world.scene.add(SingleArticulation(self.art_path, name="fanuc"))
        world.reset()
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
        cup.CreateRadiusAttr(0.04); cup.CreateHeightAttr(args.tool_len)
        cup.CreateAxisAttr(name[-1].upper())
        UsdGeom.Xformable(cup).AddTranslateOp().Set(Gf.Vec3d(*(self.axis_local * (face + args.tool_len / 2))))
        UsdPhysics.CollisionAPI.Apply(cup.GetPrim())        # the cup is a collision shape of the end-effector body
        self.cup_path = self.ee_path + "/suction_cup"
        world.reset()                                       # re-parse physics so the cup collider is included
        self.q_cmd = np.asarray(self.art.get_joint_positions(), float)

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

    def contacts(self):
        """Robot-link contacts from the PhysX contact report: returns [(link, other)]."""
        headers, _ = get_physx_simulation_interface().get_contact_report()
        out = []
        for h in headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            c0 = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
            c1 = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
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
        if robot is not None and robot.stopped is None:
            cs = robot.contacts()
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
    cam_x = front - 4.6
    tc.build_trailer(st, collide=True, phys_mat=trailer_pm)
    tc.build_lights(st, cam_x)
    conveyor = Conveyor(st, base_xy[0] + 0.45, args.belt_speed)
    robot = Fanuc(st, base_xy)
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

    shoulder = robot.base + np.array([0, 0, 0.245])
    home = np.array([base_xy[0] + 0.35, base_xy[1], 1.55])
    if not move(home, timeout=12.0):
        log("could not reach the home pose; check --tool-axis / --pedestal")
    removed, unreachable, picks = set(), set(), 0
    while picks < (args.picks or 10 ** 6) and not robot.stopped:
        for j, o in enumerate(plan):                          # live poses; a box that left the wall is removed
            if j in removed:
                continue
            p, _ = box_pose(o["prim"])
            c = p + np.array([0, 0, o["size_m"][2] / 2])
            if np.linalg.norm(c - np.asarray(o["center_m"])) > 0.15:
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
            face = np.array([p[0] - d / 2, p[1], p[2] + h / 2])      # front-face centre (box origin = bottom centre)
            if np.linalg.norm(face - np.array([robot.tool_off, 0, 0]) - shoulder) < REACH - 0.15:
                reach.append((face[2] + h / 2, -abs(face[1] - base_xy[1]), i, face))
            else:
                EVENTS["skipped"].append(plan[i]["prim"]); unreachable.add(i)
                log(f"{plan[i]['prim']} out of reach (face at {np.round(face, 2).tolist()}): left in place")
        if not reach:
            log("no free box within reach (the rest is out of reach or under/behind boxes that are)")
            break
        _, _, i, face = max(reach)                                     # highest first, then nearest the robot
        b = plan[i]
        d, w, h = b["size_m"]
        log(f"pick {picks + 1}: {b['prim']} ({b['model']}, {b['mass_kg']:.1f} kg) top at "
            f"{face[2] + h / 2:.2f} m")
        before = {o["prim"]: box_pose(o["prim"])[0] for j, o in enumerate(plan) if j not in removed and j != i}
        def phase(name):
            now = {k: box_pose(k)[0] for k in before}
            top = sorted(((100 * np.linalg.norm(now[k] - before[k]), k) for k in before), reverse=True)[:3]
            log(f"  after {name}: neighbours moved " + ", ".join(f"{k.split('/')[-1]} {c:.1f} cm" for c, k in top)
                + (f"; arm touched boxes {sorted(robot.box_touch)}" if robot.box_touch else "; arm touched no box"))
            robot.box_touch.clear()

        robot.box_touch.clear()
        approach = face - np.array([0.15, 0, 0])
        if not move(approach) or not move(face - np.array([0.008, 0, 0]), timeout=4.0):
            if robot.stopped:
                break
            unreachable.add(i); EVENTS["skipped"].append(b["prim"]); move(home); continue
        phase("approach")
        robot.grip(b["prim"], compliant=True)
        for _ in range(10):
            sim.step(robot)
        # 1) slide it out by ~40 % of its depth while it still rests on the box below: friction drags the
        #    neighbours a little; 2) lift 1 cm (top boxes have ~1.5 cm to the roof); 3) pull it clear
        slide = min(0.18, 0.4 * d)
        lift = min(0.01, max(0.0, tc.H - (face[2] + h / 2) - 0.004))
        pulled = move(face - np.array([slide, 0, 0]), timeout=6.0)
        phase("slide (compliant grip)")
        robot.grip(b["prim"], compliant=False)              # rigid from here: lift and carry
        for _ in range(5):
            sim.step(robot)
        pulled = pulled and move(face - np.array([slide, 0, -lift]), timeout=3.0) and \
            move(np.array([wall_front - 0.08 - d, face[1], face[2] + lift]), timeout=8.0)   # clear of the whole wall
        phase("lift + pull clear")
        after = {k: box_pose(k)[0] for k in before}
        disp = sorted(((100 * np.linalg.norm(after[k] - before[k]), k) for k in before), reverse=True)[:3]
        log("  neighbours dragged: " + ", ".join(f"{k.split('/')[-1]} {c:.1f} cm" for c, k in disp))
        if robot.stopped:                                   # collision: hold the box where it is
            break
        if not pulled:
            log("  pull fell short; carrying it to the belt anyway")
        # to the conveyor: belt stopped while placing, tool keeps pointing +x (box hangs in front of it)
        conveyor.set_speed(0.0)
        drop_x = base_xy[0] + 0.1
        carry_z = max(BELT_TOP + h / 2 + 0.40, args.pedestal + h / 2 + 0.12)    # box bottom above the pedestal
        clear_x = wall_front - 0.08 - d
        over = np.array([drop_x - d / 2, BELT_Y, carry_z])
        place = np.array([drop_x - d / 2, BELT_Y, BELT_TOP + h / 2 + 0.03])
        # every leg is attempted even if one falls short, so the box is only ever released over the belt
        legs = [move(np.array([clear_x, face[1], carry_z]), timeout=8.0) if not robot.stopped else False,
                move(over, timeout=8.0) if not robot.stopped else False,
                move(place, timeout=5.0) if not robot.stopped else False]
        ok = all(legs)
        if robot.stopped:                                   # collision while carrying: keep holding, stop
            break
        robot.release()
        for _ in range(30):
            sim.step(robot)
        move(place - np.array([0.10, 0, 0]), timeout=3.0)
        move(place + np.array([-0.10, 0, 0.30]), timeout=4.0)
        conveyor.set_speed(args.belt_speed)
        removed.add(i); picks += 1
        EVENTS["picks"].append({"box": b["prim"], "model": b["model"], "mass_kg": round(b["mass_kg"], 2),
                                "placed": bool(ok), "neighbours_dragged_cm": [[k, round(c, 2)] for c, k in disp]})
        if not move(home, timeout=10.0) and robot.stopped:
            break

    for _ in range(int(4.0 / DT)):                                     # let the belt carry the boxes out
        sim.step(robot)
    out_boxes = [o["prim"] for o in plan if box_pose(o["prim"])[0][0] < 0]
    log(f"{picks} boxes picked, {len(out_boxes)} carried out of the trailer by the conveyor")

    if args.collision_test and not robot.stopped:
        log("collision test: driving the tool towards the left trailer wall")
        start = np.array([base_xy[0] + 0.25, base_xy[1] - 0.45, 1.40])
        move(start, a_des=np.array([0.0, -1.0, 0.0]), timeout=10.0)
        move(np.array([start[0], -tc.W / 2 - 0.2, start[2]]), a_des=np.array([0.0, -1.0, 0.0]), timeout=10.0)
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
