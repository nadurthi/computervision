"""Trailer-unloading cell in Isaac Sim: a UR10 with a suction gripper picks cartons off a stacked box wall inside
a trailer and places them on a conveyor that runs at a fixed speed. Everything is driven from Python:
the joint targets come from Lula inverse kinematics and are interpolated and sent every physics step, and the
conveyor surface speed is a plain number.

    ./python.sh unload_cell.py --gui                                   # watch it
    ./python.sh unload_cell.py --simready ~/simready --gui             # SimReady boxes + conveyor (download first)
    ./python.sh unload_cell.py --belt-speed 0.5 --picks 10 --record out/   # headless, saves camera/lidar frames

Written for the Isaac Sim 4.5 / 5.x Python API (`isaacsim.*`). Untested on a GPU at the time of writing:
expect to adjust asset paths or small API details on the first run.

Assets (NVIDIA first, then fallbacks):
    robot      UR10 from the Isaac assets library (its packaged short suction gripper is used if present)
    boxes      NVIDIA SimReady cardboard boxes (Hugging Face nvidia/simready-assets, CC-BY-4.0; fetch with
               download_simready_assets.py), else Isaac Simple_Warehouse SM_CardBox*, else procedural cartons
    conveyor   SimReady conveyor_belt_a08, else an Isaac Props/Conveyors piece, else a procedural belt
    warehouse  Isaac Simple_Warehouse/full_warehouse.usd around the dock
    trailer    no open trailer USD was found, so a 53 ft dry-van interior is built procedurally
               (floor, corrugated side walls, roof, nose wall); pass --trailer-usd to use your own

Layout (Z up, metres): the trailer interior runs along +x from the dock door (x = 0). The robot stands on a
0.9 m pedestal just inside the door, the box wall starts ~0.6 m in front of it, and the conveyor runs from beside
the robot back out through the door (-x) into the warehouse.

Reach: a UR10 reaches about 1.3 m, so by default the wall is 1.4 m wide, one row deep and 1.5 m high. A full
2.46 m trailer-width wall needs a larger arm or the robot on a rail; boxes out of reach are reported and skipped.
"""
import argparse
import glob
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--simready", default=None, help="folder with downloaded SimReady assets (download_simready_assets.py)")
ap.add_argument("--box-usd-dir", default=None, help="any folder of box USDs (overrides --simready boxes)")
ap.add_argument("--trailer-usd", default=None, help="your own trailer USD (interior along +x, floor at z = 0, door at x = 0)")
ap.add_argument("--no-warehouse", action="store_true")
ap.add_argument("--belt-speed", type=float, default=0.4, help="conveyor surface speed (m/s), set from Python")
ap.add_argument("--belt-length", type=float, default=5.0)
ap.add_argument("--belt-height", type=float, default=0.75)
ap.add_argument("--wall-width", type=float, default=1.4, help="width of the box wall, centred in the trailer (m)")
ap.add_argument("--rows", type=int, default=1, help="rows of boxes behind the front face")
ap.add_argument("--stack-height", type=float, default=1.5, help="max stack height (m)")
ap.add_argument("--picks", type=int, default=0, help="stop after this many picks (0 = until the wall is empty)")
ap.add_argument("--move-time", type=float, default=1.2, help="seconds per arm segment (joint interpolation)")
ap.add_argument("--record", default=None, help="folder: save camera RGB/depth/instance + lidar before every pick")
ap.add_argument("--lidar-config", default="Example_Rotary")
ap.add_argument("--stable-ids", action="store_true", help="RTX stable ids (per-point lidar object ids)")
args, _ = ap.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720,
                                "extra_args": ["--/rtx-transient/stableIds/enabled=true"] if args.stable_ids else []})

import numpy as np                                                    # noqa: E402
import omni.client                                                    # noqa: E402
import omni.kit.commands                                              # noqa: E402
import omni.usd                                                       # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade   # noqa: E402

from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction            # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat      # noqa: E402
from isaacsim.robot_motion.motion_generation import (                # noqa: E402
    ArticulationKinematicsSolver, LulaKinematicsSolver, interface_config_loader)
try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    from isaacsim.core.utils.nucleus import get_assets_root_path

rng = np.random.default_rng(args.seed)
ASSETS = get_assets_root_path() or ""

# ------------------------------------------------------------------------------------------- layout constants
TRAILER_L, TRAILER_W, TRAILER_H = 16.0, 2.46, 2.70     # 53 ft dry van interior
ROBOT_BASE = np.array([0.45, 0.0, 0.0])                # pedestal footprint centre (inside the door)
PEDESTAL_H = 0.9                                        # puts the UR10 shoulder near mid-stack height
WALL_FACE_X = 1.05                                      # front face of the box wall
BELT_Y = 0.75                                           # conveyor centre line (beside the robot)
BELT_X0 = 0.9                                           # belt starts here and runs towards -x
BELT_W = 0.6
DOWN = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))   # UR10 ee_link x-axis pointing down

CANDIDATE_UR10 = ["/Isaac/Robots/UniversalRobots/ur10/ur10.usd", "/Isaac/Robots/UR10/ur10.usd"]
CANDIDATE_ISAAC_BOXES = [f"/Isaac/Environments/Simple_Warehouse/Props/SM_CardBox{c}_{i:02d}.usd"
                         for c in "ABCD" for i in range(1, 6)]
CANDIDATE_ISAAC_CONVEYOR = ["/Isaac/Props/Conveyors/ConveyorBelt_A08.usd", "/Isaac/Props/Conveyors/ConveyorBelt_A02.usd"]
WAREHOUSE = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"


def exists(url):
    return omni.client.stat(url)[0] == omni.client.Result.OK


def first_existing(paths):
    for p in paths:
        if exists(p):
            return p
    return None


# ------------------------------------------------------------------------------------------- USD helpers
def material(stage, path, rgb, rough=0.8, metal=0.0):
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


def cube(stage, path, center, size, mat=None, collide=True, label=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if label:
        add_label(c.GetPrim(), label)
    return c


def add_label(prim, label):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
    except Exception:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, label)


def bounds(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def fit_reference(stage, path, url, size=None, bottom_center=(0, 0, 0), yaw=0.0):
    """Reference `url` under `path`, optionally rescale it to `size` (x, y, z in m), and put its bottom-centre at
    `bottom_center`. The asset keeps its own transforms inside path/fit/asset."""
    top = UsdGeom.Xform.Define(stage, path)
    fit = UsdGeom.Xform.Define(stage, path + "/fit")
    UsdGeom.Xform.Define(stage, path + "/fit/asset").GetPrim().GetReferences().AddReference(url)
    r = bounds(fit.GetPrim())
    ext, ctr = np.array(r.GetSize()), np.array(r.GetMidpoint())
    if np.any(ext <= 1e-6):
        raise RuntimeError(f"empty bounds: {url}")
    s = np.ones(3) if size is None else np.asarray(size) / ext
    if size is None and ext.max() > 50:                      # authored in centimetres
        s = np.full(3, 0.01)
    fx = UsdGeom.Xformable(fit)
    fx.AddTranslateOp().Set(Gf.Vec3d(*(-s[0] * ctr[0], -s[1] * ctr[1], -s[2] * r.GetMin()[2])))
    fx.AddScaleOp().Set(Gf.Vec3f(*s))
    tx = UsdGeom.Xformable(top)
    tx.AddTranslateOp().Set(Gf.Vec3d(*map(float, bottom_center)))
    tx.AddRotateZOp().Set(float(yaw))
    return top, ext * s


def strip_physics(prim):
    """SimReady assets carry their own rigid bodies; the wrapper becomes the single rigid body instead."""
    for p in Usd.PrimRange(prim):
        if p != prim and p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def ensure_colliders(prim):
    has = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim))
    if has:
        return
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
        elif p.IsA(UsdGeom.Cube):
            UsdPhysics.CollisionAPI.Apply(p)


# ------------------------------------------------------------------------------------------- environment
def build_environment(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if not args.no_warehouse and ASSETS and exists(ASSETS + WAREHOUSE):
        wh = UsdGeom.Xform.Define(stage, "/World/Warehouse")
        wh.GetPrim().GetReferences().AddReference(ASSETS + WAREHOUSE)
        r = bounds(wh.GetPrim())
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        # put the warehouse's +x wall at the dock door (x = 0), centred on the trailer
        UsdGeom.Xformable(wh).AddTranslateOp().Set(Gf.Vec3d(-hi[0] + 0.3, -(lo[1] + hi[1]) / 2, 0.0))
        add_label(wh.GetPrim(), "warehouse")
        clear_props(stage, wh.GetPrim(), (-args.belt_length - 2.0, -3.0, 0.02), (0.5, 3.0, 4.0))
    concrete = material(stage, "/World/Looks/Concrete", (0.45, 0.45, 0.43), 0.9)
    ground = cube(stage, "/World/Ground", (0, 0, -0.05), (80, 80, 0.1), concrete, label="floor")
    if not args.no_warehouse:
        UsdGeom.Imageable(ground).MakeInvisible()          # the warehouse floor is the visible one
    build_trailer(stage)
    UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(400.0)
    for k, x in enumerate((0.6, 2.5, -1.5, -3.5)):          # placeholder lights, replace later
        rl = UsdLux.RectLight.Define(stage, f"/World/Lights/L{k}")
        rl.CreateIntensityAttr(20000.0)
        rl.CreateWidthAttr(1.0)
        rl.CreateHeightAttr(0.3)
        UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, TRAILER_H - 0.05 if x > 0 else 4.0))


def clear_props(stage, root, lo, hi):
    """Deactivate warehouse props (anything narrower than 8 m) that overlap the dock / conveyor area."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    it = iter(Usd.PrimRange(root))
    for p in it:
        if p == root or not p.IsA(UsdGeom.Imageable):
            continue
        b = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if b.IsEmpty():
            continue
        mn, mx = np.array(b.GetMin()), np.array(b.GetMax())
        if max(mx[0] - mn[0], mx[1] - mn[1]) > 8.0:
            continue
        if all(mn[i] < hi[i] and mx[i] > lo[i] for i in range(3)):
            p.SetActive(False)
        it.PruneChildren()


def build_trailer(stage):
    if args.trailer_usd:
        t = UsdGeom.Xform.Define(stage, "/World/Trailer")
        t.GetPrim().GetReferences().AddReference(args.trailer_usd)
        ensure_colliders(t.GetPrim())
        add_label(t.GetPrim(), "trailer")
        return
    steel = material(stage, "/World/Looks/TrailerSteel", (0.78, 0.79, 0.80), 0.45, 0.6)
    wood = material(stage, "/World/Looks/TrailerFloor", (0.42, 0.30, 0.20), 0.85)
    L, W, H = TRAILER_L, TRAILER_W, TRAILER_H
    base = "/World/Trailer"
    UsdGeom.Xform.Define(stage, base)
    cube(stage, base + "/Floor", (L / 2, 0, -0.025), (L, W + 0.1, 0.05), wood, label="trailer")
    cube(stage, base + "/Roof", (L / 2, 0, H + 0.03), (L, W + 0.1, 0.06), steel, label="trailer")
    cube(stage, base + "/Nose", (L + 0.03, 0, H / 2), (0.06, W + 0.1, H), steel, label="trailer")
    for side, y in (("L", W / 2 + 0.03), ("R", -W / 2 - 0.03)):
        cube(stage, f"{base}/Wall{side}", (L / 2, y, H / 2), (L, 0.06, H), steel, label="trailer")
        for i, x in enumerate(np.arange(0.3, L, 0.3)):     # corrugation ribs (visual only)
            cube(stage, f"{base}/Wall{side}/Rib{i:02d}", (x, y - np.sign(y) * 0.035, H / 2),
                 (0.04, 0.015, H - 0.1), steel, collide=False)


# ------------------------------------------------------------------------------------------- boxes
def box_assets():
    if args.box_usd_dir:
        f = sorted(glob.glob(os.path.join(args.box_usd_dir, "**", "*.usd*"), recursive=True))
        if f:
            return f
    if args.simready:
        f = sorted(glob.glob(os.path.join(args.simready, "box_*", "simready_usd", "sm_*_01.usd")))
        if f:
            print(f"[cell] {len(f)} SimReady boxes")
            return f
    f = [ASSETS + p for p in CANDIDATE_ISAAC_BOXES if ASSETS and exists(ASSETS + p)]
    print(f"[cell] {len(f)} Isaac warehouse boxes" if f else "[cell] no box USDs found: procedural cartons")
    return f


def build_box_wall(stage):
    assets = box_assets()
    card = [material(stage, f"/World/Looks/Card{i}", c, 0.85) for i, c in
            enumerate([(0.62, 0.47, 0.30), (0.70, 0.55, 0.36), (0.55, 0.41, 0.27)])]
    UsdGeom.Xform.Define(stage, "/World/Boxes")
    boxes, idx = [], 0
    half = min(args.wall_width, TRAILER_W - 0.06) / 2
    y = -half
    while y < half - 0.2:
        w = min(rng.uniform(0.35, 0.55), half - y)
        x_face = WALL_FACE_X
        for _row in range(args.rows):
            d = rng.uniform(0.32, 0.48)
            z, top = 0.0, rng.uniform(0.7 * args.stack_height, args.stack_height)
            while True:
                h = rng.uniform(0.22, 0.40)
                if z + h > top:
                    break
                size = (d * rng.uniform(0.95, 1.0), w * rng.uniform(0.93, 0.99), h)
                pos = (x_face + size[0] / 2 + rng.uniform(0, 0.01), y + w / 2, z + 0.002)
                path = f"/World/Boxes/box_{idx:03d}"
                url = assets[rng.integers(len(assets))] if assets else None
                try:
                    if url is None:
                        raise RuntimeError("procedural")
                    top_prim, _ = fit_reference(stage, path, url, size, pos, rng.normal(0, 1.5))
                    strip_physics(top_prim.GetPrim())
                    ensure_colliders(top_prim.GetPrim())
                except Exception:
                    if stage.GetPrimAtPath(path):
                        stage.RemovePrim(path)
                    top_prim = UsdGeom.Xform.Define(stage, path)
                    UsdGeom.Xformable(top_prim).AddTranslateOp().Set(Gf.Vec3d(*pos))
                    UsdGeom.Xformable(top_prim).AddRotateZOp().Set(float(rng.normal(0, 1.5)))
                    cube(stage, path + "/geom", (0, 0, h / 2), size, card[idx % 3])
                prim = top_prim.GetPrim()
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(1.5 + 30 * np.prod(size)))
                add_label(prim, "carton")
                boxes.append({"path": path, "size": np.array(size), "row": _row})
                idx += 1
                z += h + 0.002
            x_face += d + 0.005
        y += w + rng.uniform(0.0, 0.01)
    print(f"[cell] {len(boxes)} boxes in the trailer")
    return boxes


# ------------------------------------------------------------------------------------------- conveyor
class Conveyor:
    """Belt along -x from (BELT_X0, BELT_Y). The moving surface is a kinematic slab with PhysX surface
    velocity, so anything resting on it is carried at `speed` m/s. If this PhysX build has no surface-velocity
    API, boxes on the belt get their horizontal velocity set every step instead."""

    def __init__(self, stage, speed):
        self.speed = speed
        L, Wb, Hb = args.belt_length, BELT_W, args.belt_height
        self.x0, self.x1 = BELT_X0 - L, BELT_X0
        cx = (self.x0 + self.x1) / 2
        visual = None
        cand = []
        if args.simready:
            cand += sorted(glob.glob(os.path.join(args.simready, "conveyor_belt_a08", "simready_usd", "*.usd")))
        cand += [ASSETS + p for p in CANDIDATE_ISAAC_CONVEYOR if ASSETS and exists(ASSETS + p)]
        for url in cand:
            try:   # stretch the conveyor model to the belt size; the slab below does the physics
                visual, _ = fit_reference(stage, "/World/Conveyor/Model", url, (L, Wb + 0.15, Hb),
                                          (cx, BELT_Y, 0.0))
                for p in Usd.PrimRange(visual.GetPrim()):
                    for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI):
                        if p.HasAPI(api):
                            p.RemoveAPI(api)
                print("[cell] conveyor model:", url)
                break
            except Exception as e:
                print("[cell] conveyor model failed:", url, e)
                if stage.GetPrimAtPath("/World/Conveyor/Model"):
                    stage.RemovePrim("/World/Conveyor/Model")
                visual = None
        frame_mat = material(stage, "/World/Looks/ConveyorFrame", (0.25, 0.27, 0.30), 0.5, 0.5)
        belt_mat = material(stage, "/World/Looks/Belt", (0.08, 0.08, 0.08), 0.7)
        if visual is None:
            cube(stage, "/World/Conveyor/Frame", (cx, BELT_Y, (Hb - 0.04) / 2), (L, Wb + 0.1, Hb - 0.04),
                 frame_mat, collide=False)
        self.belt = cube(stage, "/World/Conveyor/Belt", (cx, BELT_Y, Hb - 0.02), (L, Wb, 0.04), belt_mat,
                         label="conveyor")
        if visual is not None:
            UsdGeom.Imageable(self.belt).MakeInvisible()
        bp = self.belt.GetPrim()
        rb = UsdPhysics.RigidBodyAPI.Apply(bp)
        rb.CreateKinematicEnabledAttr(True)
        self.surface_api = False
        try:
            sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(bp)
            sv.CreateSurfaceVelocityEnabledAttr(True)
            sv.CreateSurfaceVelocityAttr(Gf.Vec3f(-speed, 0.0, 0.0))   # belt-local; slab is axis-aligned
            self.surface_api = True
            print(f"[cell] conveyor: PhysX surface velocity {speed} m/s")
        except Exception as e:
            print(f"[cell] conveyor: no surface-velocity API ({e}); driving boxes on the belt directly")
        self.top = Hb

    def set_speed(self, speed):
        self.speed = speed
        if self.surface_api:
            PhysxSchema.PhysxSurfaceVelocityAPI(self.belt.GetPrim()).GetSurfaceVelocityAttr().Set(
                Gf.Vec3f(-speed, 0.0, 0.0))

    def on_belt(self, p):
        return (self.x0 < p[0] < self.x1 and abs(p[1] - BELT_Y) < BELT_W / 2 + 0.05
                and self.top - 0.05 < p[2] < self.top + 0.6)

    def step(self, bodies):
        """Fallback drive + clean-up: boxes past the end of the belt are removed."""
        for b in list(bodies):
            pos, _ = b["rigid"].get_world_pose()
            if pos[0] < self.x0 - 0.5:
                b["rigid"].prim.SetActive(False)
                bodies.remove(b)
                continue
            if not self.surface_api and self.on_belt(pos) and not b.get("held"):
                v = b["rigid"].get_linear_velocity()
                b["rigid"].set_linear_velocity(np.array([-self.speed, 0.0, v[2]]))


# ------------------------------------------------------------------------------------------- robot
class Robot:
    def __init__(self, stage, world):
        url = first_existing([ASSETS + p for p in CANDIDATE_UR10]) if ASSETS else None
        if url is None:
            raise RuntimeError("UR10 USD not found in the Isaac assets")
        steel = material(stage, "/World/Looks/Pedestal", (0.2, 0.22, 0.25), 0.4, 0.7)
        cube(stage, "/World/Pedestal", (ROBOT_BASE[0], ROBOT_BASE[1], PEDESTAL_H / 2), (0.5, 0.5, PEDESTAL_H), steel)
        prim = UsdGeom.Xform.Define(stage, "/World/UR10")
        prim.GetPrim().GetReferences().AddReference(url)
        self._select_suction_variant(prim.GetPrim())
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(ROBOT_BASE[0], ROBOT_BASE[1], PEDESTAL_H))
        self.arm = world.scene.add(SingleArticulation("/World/UR10", name="ur10"))
        self.gripper_path = self._find_surface_gripper(prim.GetPrim())
        self.tool_len = 0.12
        if self.gripper_path is None:     # visual suction cup on the flange; grasping falls back to attach
            cup = UsdGeom.Cylinder.Define(stage, "/World/UR10/ee_link/SuctionCup")
            cup.CreateRadiusAttr(0.035)
            cup.CreateHeightAttr(self.tool_len)
            cup.CreateAxisAttr("X")
            UsdGeom.Xformable(cup).AddTranslateOp().Set(Gf.Vec3d(self.tool_len / 2, 0, 0))
            print("[cell] no packaged surface gripper: using kinematic attach on suction")
        else:
            print("[cell] surface gripper:", self.gripper_path)
        self.held = None

    @staticmethod
    def _select_suction_variant(prim):
        vsets = prim.GetVariantSets()
        for name in vsets.GetNames():
            vs = vsets.GetVariantSet(name)
            for v in vs.GetVariantNames():
                if "suction" in v.lower() and "short" in v.lower():
                    vs.SetVariantSelection(v)
                    print(f"[cell] UR10 variant {name} = {v}")
                    return

    @staticmethod
    def _find_surface_gripper(prim):
        for p in Usd.PrimRange(prim):
            if "SurfaceGripper" in p.GetTypeName():
                return p.GetPath().pathString
        return None

    def setup(self):
        """After world.reset(): IK solver and gripper handles."""
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("UR10")
        self.lula = LulaKinematicsSolver(**cfg)
        base_pos, base_rot = self.arm.get_world_pose()
        self.lula.set_robot_base_pose(base_pos, base_rot)
        self.ik = ArticulationKinematicsSolver(self.arm, self.lula, "ee_link")
        self.view = None
        if self.gripper_path:
            try:
                from isaacsim.robot.surface_gripper import GripperView
                self.view = GripperView(paths=self.gripper_path)
                # measure the real tool length: flange -> gripper frame
                ee = UsdGeom.Xformable(stage().GetPrimAtPath("/World/UR10/ee_link"))
                gp = UsdGeom.Xformable(stage().GetPrimAtPath(self.gripper_path))
                t = Usd.TimeCode.Default()
                d = (gp.ComputeLocalToWorldTransform(t).ExtractTranslation()
                     - ee.ComputeLocalToWorldTransform(t).ExtractTranslation())
                self.tool_len = max(float(d.GetLength()), 0.05)
            except Exception as e:
                print("[cell] GripperView unavailable, using attach:", e)
                self.gripper_path = None
        self.home = self.solve(np.array([ROBOT_BASE[0] + 0.4, ROBOT_BASE[1] + 0.3, PEDESTAL_H + 0.5]))
        if self.home is None:
            print("[cell] home pose has no IK solution; staying at the current joint positions")
            self.home = self.joints()

    def solve(self, tool_point, orient=DOWN):
        """Joint positions putting the suction cup at `tool_point` (world), pointing down."""
        ee_target = np.asarray(tool_point) + np.array([0.0, 0.0, self.tool_len])
        action, ok = self.ik.compute_inverse_kinematics(ee_target, orient)
        if not ok:
            return None
        return np.asarray(action.joint_positions, dtype=float)

    def joints(self):
        return np.asarray(self.arm.get_joint_positions()[:6], dtype=float)

    def tool_pose(self):
        pos, rot = self.lula.compute_forward_kinematics("ee_link", self.joints())
        return pos - np.array([0, 0, self.tool_len]), rot

    def command(self, q):
        self.arm.apply_action(ArticulationAction(joint_positions=q, joint_indices=np.arange(6)))

    # suction ------------------------------------------------------------------------------------------
    def suction_on(self, bodies):
        if self.view is not None:
            self.view.apply_gripper_action([1.0])
            return
        tool, _ = self.tool_pose()
        best, dmin = None, 0.06
        for b in bodies:                                    # box whose top centre is right under the cup
            pos, _ = b["rigid"].get_world_pose()
            top = pos + np.array([0, 0, b["size"][2]])
            d = np.linalg.norm(top - tool)
            if d < dmin:
                best, dmin = b, d
        if best is not None:
            best["rigid"].prim.GetAttribute("physics:kinematicEnabled").Set(True)
            pos, rot = best["rigid"].get_world_pose()
            best["offset"] = pos - tool
            best["held"] = True
            self.held = best

    def suction_off(self):
        if self.view is not None:
            self.view.apply_gripper_action([-1.0])
            return
        if self.held is not None:
            self.held["rigid"].prim.GetAttribute("physics:kinematicEnabled").Set(False)
            self.held["rigid"].set_linear_velocity(np.zeros(3))
            self.held["held"] = False
            self.held = None

    def carry(self):
        if self.held is not None:                           # kinematic attach: box follows the cup
            tool, _ = self.tool_pose()
            _, rot = self.held["rigid"].get_world_pose()
            self.held["rigid"].set_world_pose(tool + self.held["offset"], rot)


def stage():
    return omni.usd.get_context().get_stage()


# ------------------------------------------------------------------------------------------- sensors
def make_sensors(stage_):
    if not args.record:
        return None
    import omni.replicator.core as rep
    os.makedirs(args.record, exist_ok=True)
    cam = UsdGeom.Camera.Define(stage_, "/World/Sensors/Camera")
    cam.CreateFocalLengthAttr(14.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 720 / 1280)
    eye, tgt = np.array([-0.6, 0.0, 2.1]), np.array([WALL_FACE_X + 0.3, 0.0, 0.9])
    f = (tgt - eye) / np.linalg.norm(tgt - eye)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*r, 0)); m.SetRow(1, Gf.Vec4d(*u, 0)); m.SetRow(2, Gf.Vec4d(*(-f), 0)); m.SetRow(3, Gf.Vec4d(*eye, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    rp = rep.create.render_product(cam.GetPath().pathString, (1280, 720))
    anns = {k: rep.AnnotatorRegistry.get_annotator(n, **({"init_params": {"colorize": False}} if "seg" in n else {}))
            for k, n in (("rgb", "rgb"), ("depth", "distance_to_image_plane"), ("instance", "instance_segmentation_fast"))}
    for a in anns.values():
        a.attach([rp])
    try:
        _, lidar = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="/World/Sensors/Lidar", parent=None,
                                             config=args.lidar_config, translation=Gf.Vec3d(-0.6, 0.0, 2.4),
                                             orientation=Gf.Quatd(1, 0, 0, 0))
        lrp = rep.create.render_product(lidar.GetPath(), [1, 1])
        la = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        la.attach([lrp.path])
        anns["lidar"] = la
    except Exception as e:
        print("[cell] lidar not created:", e)
    return anns


def record(anns, k):
    if anns is None:
        return
    from PIL import Image
    Image.fromarray(np.asarray(anns["rgb"].get_data())[..., :3]).save(os.path.join(args.record, f"rgb_{k:03d}.png"))
    np.save(os.path.join(args.record, f"depth_{k:03d}.npy"), np.asarray(anns["depth"].get_data(), np.float32))
    seg = anns["instance"].get_data()
    Image.fromarray(np.asarray(seg["data"]).astype(np.uint16)).save(os.path.join(args.record, f"instance_{k:03d}.png"))
    if "lidar" in anns:
        np.save(os.path.join(args.record, f"lidar_{k:03d}.npy"),
                np.asarray(anns["lidar"].get_data().get("data", np.zeros((0, 3))), np.float32).reshape(-1, 3))


# ------------------------------------------------------------------------------------------- task logic
def next_box(bodies):
    """Unloading order: highest top surface first, front row before the row behind it."""
    cand = []
    for b in bodies:
        if b.get("held") or b.get("done"):
            continue
        pos, _ = b["rigid"].get_world_pose()
        if pos[0] < WALL_FACE_X - 0.3:                       # already out of the wall
            continue
        cand.append((-(pos[2] + b["size"][2]), b["row"], pos[0], b))
    if not cand:
        return None
    cand.sort(key=lambda c: (round(c[0], 1), c[1], c[2]))
    return cand[0][3]


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
    st = stage()
    build_environment(st)
    boxes = build_box_wall(st)
    conveyor = Conveyor(st, args.belt_speed)
    robot = Robot(st, world)
    anns = make_sensors(st)
    world.reset()
    for b in boxes:
        b["rigid"] = SingleRigidPrim(b["path"])
    robot.setup()
    for _ in range(90):                                      # let the stack settle
        world.step(render=args.gui)

    steps_per_move = max(int(args.move_time * 60), 10)

    def move_to(q_target, carry=True):
        """Linear joint-space interpolation, one target per physics step (the 'joints set from Python')."""
        q0 = robot.joints()
        for i in range(1, steps_per_move + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / steps_per_move)   # smooth start/stop
            robot.command(q0 + s * (q_target - q0))
            if carry:
                robot.carry()
            conveyor.step(boxes)
            world.step(render=True)

    def wait(n):
        for _ in range(n):
            robot.carry()
            conveyor.step(boxes)
            world.step(render=True)

    move_to(robot.home)
    picks = 0
    drop = np.array([BELT_X0 - 0.6, BELT_Y, conveyor.top])
    while simulation_app.is_running():
        b = next_box(boxes)
        if b is None or (args.picks and picks >= args.picks):
            break
        record(anns, picks)
        pos, _ = b["rigid"].get_world_pose()
        top = pos + np.array([0.0, 0.0, b["size"][2]])
        path = [top + [0, 0, 0.20], top + [0, 0, 0.005]]
        qs = [robot.solve(p) for p in path]
        if any(q is None for q in qs):
            print(f"[cell] {b['path']} out of reach, skipped")
            b["done"] = True
            continue
        move_to(qs[0]); move_to(qs[1])
        robot.suction_on(boxes); wait(15)
        lift = top + [-0.15, 0, 0.30]                        # up and back out of the wall
        over_belt = drop + [0, 0, b["size"][2] + 0.25]
        place = drop + [0, 0, b["size"][2] + 0.03]
        for p in (lift, over_belt, place):
            q = robot.solve(p)
            if q is not None:
                move_to(q)
        robot.suction_off(); wait(10)
        b["done"] = True
        q = robot.solve(over_belt)
        if q is not None:
            move_to(q, carry=False)
        picks += 1
        print(f"[cell] pick {picks}: {b['path']}  belt {conveyor.speed} m/s")
    print(f"[cell] finished after {picks} picks")
    while args.gui and simulation_app.is_running():
        conveyor.step(boxes)
        world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
