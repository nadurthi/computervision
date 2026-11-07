"""Stacked cardboard boxes against a wall on a factory floor, captured with an RGB-D camera and an RTX lidar.

Run with Isaac Sim's own Python (standalone workflow):

    ./python.sh box_wall_capture.py                       # headless, writes to ./box_wall_out
    ./python.sh box_wall_capture.py --gui --frames 3      # watch it in the viewer
    ./python.sh box_wall_capture.py --box-usd-dir /path/to/my/cardboard_usds --settle
    ./python.sh box_wall_capture.py --env-offset 0 6 0        # use another stretch of the warehouse wall
    ./python.sh box_wall_capture.py --env slabs               # plain floor + wall, no warehouse

Written against the Isaac Sim 4.5 / 5.x Python API (`isaacsim.*` modules). Older `omni.isaac.*` names are
tried as a fallback where they differ.

What it builds (Z up, metres):
    * the environment: by default NVIDIA's ready-made warehouse (Simple_Warehouse/full_warehouse.usd: concrete floor,
      walls, racks, fixtures), moved so its +x wall is right behind the boxes, with any props in the box-wall /
      sensor area removed and a flat wall panel at x = WALL_X. --env slabs uses a plain floor slab and wall instead.
    * a wall of cardboard boxes: a back row touching the wall and a partial front row, columns stacked
      from the floor. Box models are the open NVIDIA Isaac assets (Simple_Warehouse SM_CardBox*); any USD
      folder can be given instead. If no asset can be found the script falls back to procedural cartons.
    * placeholder lights above the stack (one dome + a grid of rect lights) -- replace them later
    * a pinhole camera and an RTX lidar, both facing the wall

What it writes per frame (in --out):
    rgb_XXXX.png                  colour image
    depth_XXXX.npy                distance to image plane, metres (float32, inf = no hit)
    instance_XXXX.png/.json       instance-id mask + id -> prim path / label mapping
    semantic_XXXX.png/.json       semantic mask (classes: carton, wall, floor, warehouse) + id -> label mapping
    bbox2d_XXXX.json              tight 2D boxes per carton
    lidar_XXXX.npy / .ply         lidar points (world frame, N x 3) and, when available, per-point object ids
    camera.json, lidar.json       intrinsics / extrinsics
    boxes.json                    ground truth: prim path, size (m), pose of every carton
"""
import argparse
import glob
import json
import math
import os
import sys

# --------------------------------------------------------------------------------------------- CLI
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="box_wall_out")
ap.add_argument("--gui", action="store_true", help="show the Isaac Sim window (default: headless)")
ap.add_argument("--frames", type=int, default=1, help="number of captures (scene re-randomised each frame)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--box-usd-dir", default=None, help="folder of cardboard box USDs to use instead of the Isaac assets")
ap.add_argument("--wall-width", type=float, default=3.0, help="width of the box wall (m)")
ap.add_argument("--max-height", type=float, default=1.8, help="max stack height (m)")
ap.add_argument("--front-row", type=float, default=0.35, help="fraction of columns that get a second (front) row")
ap.add_argument("--settle", action="store_true", help="add rigid bodies and let the stack settle under gravity")
ap.add_argument("--cam-dist", type=float, default=2.2, help="camera distance from the wall (m)")
ap.add_argument("--cam-height", type=float, default=1.0)
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--lidar-config", default="Example_Rotary",
                help="RTX lidar config, e.g. Example_Rotary, Example_Solid_State, HESAI_XT32_SD10, OS1_REV6_128ch10hz1024res")
ap.add_argument("--no-lidar", action="store_true")
ap.add_argument("--stable-ids", action="store_true",
                help="RTX stable ids: per-point lidar object ids (crashed the 5.1 container on an L4 in testing)")
ap.add_argument("--env", choices=["warehouse", "slabs"], default="warehouse",
                help="warehouse: NVIDIA's ready-made warehouse (floor, walls, racks, fixtures); slabs: plain floor + wall")
ap.add_argument("--env-usd", default="/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
                help="environment USD, relative to the Isaac assets root or an absolute path/URL")
ap.add_argument("--wall-inset", type=float, default=0.3,
                help="warehouse mode: distance from the building's outer bound (+x side) to its inner wall face (m)")
ap.add_argument("--env-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "YAW"),
                help="warehouse mode: extra shift (m) and yaw (deg) of the building, to pick another wall section")
ap.add_argument("--no-wall-slab", action="store_true",
                help="warehouse mode: do not add the flat wall panel behind the boxes (use the building wall only)")
ap.add_argument("--no-extra-lights", action="store_true", help="do not add the placeholder lights")
args, _unknown = ap.parse_known_args()

# --------------------------------------------------------------------------------------------- Isaac Sim app
try:
    from isaacsim import SimulationApp
except ImportError:                                     # Isaac Sim <= 4.0
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": not args.gui,
    "width": args.res[0],
    "height": args.res[1],
    # stable ids let the lidar return per-point object ids (opt-in: --stable-ids)
    "extra_args": ["--/rtx-transient/stableIds/enabled=true"] if args.stable_ids else [],
})

import numpy as np                                      # noqa: E402  (after SimulationApp)
import omni.client                                      # noqa: E402
import omni.kit.commands                                # noqa: E402
import omni.replicator.core as rep                      # noqa: E402
import omni.usd                                         # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade   # noqa: E402

try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World
try:
    from isaacsim.storage.native import get_assets_root_path
except ImportError:
    try:
        from isaacsim.core.utils.nucleus import get_assets_root_path
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path


def add_label(prim, label):
    """Semantic label used by the segmentation and bbox annotators."""
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
        return
    except Exception:
        pass
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
    except ImportError:
        from omni.isaac.core.utils.semantics import add_update_semantics
    add_update_semantics(prim, label)


# --------------------------------------------------------------------------------------------- constants
WALL_X = 0.0                         # wall face; boxes sit at x < WALL_X, the camera looks along +x
FLOOR_SIZE = 30.0
CARDBOARD_RGB = [(0.62, 0.47, 0.30), (0.70, 0.55, 0.36), (0.55, 0.41, 0.27), (0.76, 0.63, 0.45)]

ISAAC_BOX_CANDIDATES = [f"/Isaac/Environments/Simple_Warehouse/Props/SM_CardBox{l}_{i:02d}.usd"
                        for l in "ABCD" for i in range(1, 6)]


def usd_exists(url):
    result, _ = omni.client.stat(url)
    return result == omni.client.Result.OK


def find_box_assets():
    """Local folder first, then the open Isaac Sim warehouse cardboard boxes."""
    if args.box_usd_dir:
        # a SimReady download: only the top-level box files, not their payloads or the conveyors
        files = sorted(glob.glob(os.path.join(args.box_usd_dir, "box_*", "simready_usd", "sm_*_01.usd")))
        if not files:
            files = sorted(glob.glob(os.path.join(args.box_usd_dir, "**", "*.usd*"), recursive=True))
        if files:
            print(f"[box-wall] using {len(files)} box USDs from {args.box_usd_dir}")
            return files
        print(f"[box-wall] no .usd files in {args.box_usd_dir}")
    root = get_assets_root_path()
    if root is None:
        print("[box-wall] Isaac assets root not found (check the Nucleus / asset settings)")
        return []
    found = [root + p for p in ISAAC_BOX_CANDIDATES if usd_exists(root + p)]
    print(f"[box-wall] Isaac cardboard assets found: {len(found)}")
    for f in found:
        print("   ", f)
    return found


# --------------------------------------------------------------------------------------------- helpers
def preview_material(stage, path, rgb, roughness=0.8):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def slab(stage, path, center, size, mat, label, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(cube.GetPrim(), mat)
    add_label(cube.GetPrim(), label)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def world_range(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """USD camera looks down its -Z with +Y up. Returns the local-to-world matrix (row-vector convention)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*right, 0.0))       # camera +X
    m.SetRow(1, Gf.Vec4d(*cam_up, 0.0))      # camera +Y
    m.SetRow(2, Gf.Vec4d(*(-fwd), 0.0))      # camera +Z (it looks along -Z)
    m.SetRow(3, Gf.Vec4d(*eye, 1.0))
    return m


# --------------------------------------------------------------------------------------------- scene
def clear_zone_box():
    """Volume that must be empty: the box wall plus the corridor the sensors look through."""
    half_w = args.wall_width / 2 + 0.6
    return (WALL_X - args.cam_dist - 0.8, -half_w, 0.02), (WALL_X - 0.02, half_w, args.max_height + 1.5)


def load_warehouse(stage):
    """Reference NVIDIA's warehouse and move it so its +x inner wall sits just behind the box wall
    (x = WALL_X). Props (racks, pallets, cones...) inside the box-wall / sensor corridor are deactivated.
    Structural pieces (floor, walls, ceiling: anything wider than 8 m) are kept."""
    url = args.env_usd
    if not ("://" in url or os.path.isabs(url) and os.path.exists(url)):
        root = get_assets_root_path()
        if root is None:
            raise RuntimeError("Isaac assets root not found")
        url = root + url
    if not usd_exists(url):
        raise RuntimeError(f"environment not found: {url}")
    env = UsdGeom.Xform.Define(stage, "/World/Env/Warehouse")       # our transform lives here ...
    UsdGeom.Xform.Define(stage, "/World/Env/Warehouse/asset").GetPrim().GetReferences().AddReference(url)
    # ... the referenced file keeps its own xformOps (adding ops to it directly fails: 'xformOp:translate already exists')
    add_label(env.GetPrim(), "warehouse")                     # children inherit it in the semantic mask
    r = world_range(env.GetPrim())
    size = np.array(r.GetSize())
    scale = 0.01 if size.max() > 1000 else 1.0              # authored in centimetres
    lo, hi = np.array(r.GetMin()) * scale, np.array(r.GetMax()) * scale
    dx, dy, yaw = args.env_offset
    xf = UsdGeom.Xformable(env)
    # inner face of the +x wall -> just behind our wall panel; centre the building in y
    tx = (WALL_X + (0.0 if args.no_wall_slab else 0.2)) - (hi[0] - args.wall_inset) + dx
    ty = -(lo[1] + hi[1]) / 2 + dy
    xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, 0.0))           # warehouse floor is at z = 0
    xf.AddRotateZOp().Set(float(yaw))
    if scale != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    print(f"[box-wall] warehouse {url}: {size * scale} m, shifted by ({tx:.2f}, {ty:.2f})")

    (x0, y0, z0), (x1, y1, z1) = clear_zone_box()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    it = iter(Usd.PrimRange(env.GetPrim()))
    removed = 0
    for p in it:
        if p == env.GetPrim() or not p.IsA(UsdGeom.Imageable):
            continue
        b = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if b.IsEmpty():
            continue
        bmin, bmax = np.array(b.GetMin()), np.array(b.GetMax())
        if max(bmax[0] - bmin[0], bmax[1] - bmin[1]) > 8.0:  # structural: look inside it instead
            continue
        overlap = bmin[0] < x1 and bmax[0] > x0 and bmin[1] < y1 and bmax[1] > y0 and bmin[2] < z1 and bmax[2] > z0
        if overlap:
            p.SetActive(False)
            removed += 1
        it.PruneChildren()                                   # decided at this level either way
    print(f"[box-wall] cleared {removed} warehouse props from the box-wall area")


def build_static_scene(stage):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    concrete = preview_material(stage, "/World/Looks/Concrete", (0.42, 0.42, 0.40), 0.9)
    wall_mat = preview_material(stage, "/World/Looks/WallPaint", (0.72, 0.73, 0.75), 0.85)
    use_warehouse = args.env == "warehouse"
    if use_warehouse:
        try:
            load_warehouse(stage)
        except Exception as e:
            print(f"[box-wall] warehouse not loaded ({e}); using plain floor and wall")
            use_warehouse = False
    if use_warehouse:
        # invisible collision floor so --settle works whatever colliders the warehouse has
        ground = slab(stage, "/World/Env/CollisionFloor", (0, 0, -0.05), (FLOOR_SIZE, FLOOR_SIZE, 0.1), concrete, "floor")
        UsdGeom.Imageable(ground).MakeInvisible()
        if not args.no_wall_slab:
            slab(stage, "/World/Env/Wall", (WALL_X + 0.1, 0, 2.0), (0.2, args.wall_width + 2.0, 4.0), wall_mat, "wall")
    else:
        slab(stage, "/World/Env/Floor", (0, 0, -0.05), (FLOOR_SIZE, FLOOR_SIZE, 0.1), concrete, "floor")
        slab(stage, "/World/Env/Wall", (WALL_X + 0.1, 0, 2.5), (0.2, 12.0, 5.0), wall_mat, "wall")
    if args.no_extra_lights:
        return

    # placeholder lighting: soft dome + a grid of rect lights 3.5 m up pointing down (change later)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(300.0)
    k = 0
    for x in (-3.0, -1.0):
        for y in (-2.0, 0.0, 2.0):
            rl = UsdLux.RectLight.Define(stage, f"/World/Lights/Ceiling_{k}")
            rl.CreateIntensityAttr(25000.0)
            rl.CreateWidthAttr(1.2)
            rl.CreateHeightAttr(0.3)
            rl.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
            UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(x, y, 3.5))   # RectLight emits along -Z
            k += 1


def add_box(stage, idx, asset, size, base_xy, base_z, yaw_deg, rng):
    """One carton. The wrapper Xform holds the pose; the child holds the asset scaled to `size`
    with its bottom-centre at the wrapper origin, so every box is placed the same way whatever
    units or pivot the source USD uses."""
    root = f"/World/Boxes/box_{idx:03d}"
    wrapper = UsdGeom.Xform.Define(stage, root)
    child_path = root + "/geom"
    if asset is not None:
        child = UsdGeom.Xform.Define(stage, child_path)      # our scale/offset lives here ...
        ref = UsdGeom.Xform.Define(stage, child_path + "/asset")
        ref.GetPrim().GetReferences().AddReference(asset)    # ... so the asset keeps its own xformOps
        r = world_range(child.GetPrim())                     # wrapper is still identity here
        ext = np.array(r.GetSize())
        ctr = np.array(r.GetMidpoint())
        if np.any(ext <= 1e-6):
            raise RuntimeError(f"empty bounds for {asset}")
        s = np.asarray(size) / ext
        xf = UsdGeom.Xformable(child)
        xf.AddTranslateOp().Set(Gf.Vec3d(-s[0] * ctr[0], -s[1] * ctr[1], -s[2] * r.GetMin()[2]))
        xf.AddScaleOp().Set(Gf.Vec3f(*s))
    else:                                                    # procedural carton
        mat = preview_material(stage, f"/World/Looks/Card_{idx:03d}",
                               CARDBOARD_RGB[rng.integers(len(CARDBOARD_RGB))], 0.85)
        cube = UsdGeom.Cube.Define(stage, child_path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, size[2] / 2))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        bind(cube.GetPrim(), mat)

    wxf = UsdGeom.Xformable(wrapper)
    wxf.AddTranslateOp().Set(Gf.Vec3d(base_xy[0], base_xy[1], base_z))
    wxf.AddRotateZOp().Set(float(yaw_deg))
    add_label(wrapper.GetPrim(), "carton")

    if args.settle:
        UsdPhysics.RigidBodyAPI.Apply(wrapper.GetPrim())
        UsdPhysics.MassAPI.Apply(wrapper.GetPrim()).CreateMassAttr(float(2.0 + 8.0 * np.prod(size)))
        for p in Usd.PrimRange(wrapper.GetPrim()):
            if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube):
                UsdPhysics.CollisionAPI.Apply(p)
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
    return {"prim": root, "asset": asset or "procedural", "size_m": [float(v) for v in size],
            "position_m": [float(base_xy[0]), float(base_xy[1]), float(base_z)], "yaw_deg": float(yaw_deg)}


def build_box_wall(stage, assets, rng):
    if stage.GetPrimAtPath("/World/Boxes"):
        stage.RemovePrim("/World/Boxes")
    UsdGeom.Xform.Define(stage, "/World/Boxes")
    boxes, idx = [], 0
    y = -args.wall_width / 2
    while y < args.wall_width / 2 - 0.2:
        col_w = rng.uniform(0.30, 0.60)                       # column width (along the wall)
        col_d = rng.uniform(0.30, 0.50)                       # depth (away from the wall)
        rows = [(WALL_X - 0.005, col_d)]                      # back row touches the wall
        if rng.random() < args.front_row:                     # some columns get a second row in front
            rows.append((WALL_X - 0.005 - col_d - rng.uniform(0.0, 0.01), rng.uniform(0.30, 0.50)))
        for back_x, depth in rows:
            z, limit = 0.0, rng.uniform(0.5, args.max_height)
            while True:
                h = rng.uniform(0.20, 0.45)
                if z + h > limit:
                    break
                w = col_w * rng.uniform(0.93, 1.0)
                d = depth * rng.uniform(0.95, 1.0)
                cx = back_x - d / 2 - rng.uniform(0.0, 0.02)  # pushed-in / pulled-out a little
                cy = y + col_w / 2 + rng.uniform(-0.01, 0.01)
                yaw = rng.normal(0.0, 2.0)
                asset = assets[rng.integers(len(assets))] if assets else None
                try:
                    boxes.append(add_box(stage, idx, asset, (d, w, h), (cx, cy), z + (0.002 if args.settle else 0.0), yaw, rng))
                except Exception as e:                       # bad asset: fall back to a procedural carton
                    print(f"[box-wall] {asset}: {e}; using a procedural box")
                    stage.RemovePrim(f"/World/Boxes/box_{idx:03d}")
                    boxes.append(add_box(stage, idx, None, (d, w, h), (cx, cy), z, yaw, rng))
                idx += 1
                z += h + (0.002 if args.settle else 0.0)
        y += col_w + rng.uniform(0.0, 0.012)                  # small gaps between columns
    print(f"[box-wall] placed {len(boxes)} boxes")
    return boxes


# --------------------------------------------------------------------------------------------- sensors
def make_camera(stage):
    cam = UsdGeom.Camera.Define(stage, "/World/Sensors/Camera")
    focal_mm, h_ap_mm = 18.0, 20.955                         # ~60 deg horizontal FOV
    cam.CreateFocalLengthAttr(focal_mm)
    cam.CreateHorizontalApertureAttr(h_ap_mm)
    cam.CreateVerticalApertureAttr(h_ap_mm * args.res[1] / args.res[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
    eye = (WALL_X - args.cam_dist, 0.0, args.cam_height)
    target = (WALL_X, 0.0, args.cam_height * 0.9)
    UsdGeom.Xformable(cam).AddTransformOp().Set(look_at_matrix(eye, target))
    W, H = args.res
    fx = W * focal_mm / h_ap_mm
    intr = {"width": W, "height": H, "fx": fx, "fy": fx, "cx": W / 2, "cy": H / 2,
            "eye_m": list(eye), "target_m": list(target), "focal_mm": focal_mm, "h_aperture_mm": h_ap_mm}
    return cam.GetPath().pathString, intr


def make_lidar():
    """RTX lidar 0.35 m above the camera, +X (its forward axis) facing the wall."""
    pos = Gf.Vec3d(WALL_X - args.cam_dist, 0.0, args.cam_height + 0.35)
    _, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/World/Sensors/Lidar",
        parent=None,
        config=args.lidar_config,
        translation=pos,
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    )
    rp = rep.create.render_product(sensor.GetPath(), [1, 1], name="LidarRP")
    # Per-frame point cloud. In Isaac Sim 5.1 the accumulating IsaacCreateRTXLidarScanBuffer returned 0 points
    # in a plain standalone loop, while this one returned ~29k (see lidar_test.py). It gives one azimuth
    # slice per frame, so accumulate_lidar() stacks several frames into a full scan (world frame).
    ann = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
    ann.attach([rp.path])
    return ann, {"prim": str(sensor.GetPath()), "config": args.lidar_config, "position_m": list(pos),
                 "orientation_wxyz": [1.0, 0.0, 0.0, 0.0], "points_frame": "world",
                 "scan": "12 per-frame slices stacked (one full rotation or more)"}


def write_ply(path, pts):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.4f")


def accumulate_lidar(ann, step, n_frames=12):
    """A rotating lidar returns one azimuth slice per rendered frame (Example_Rotary: 10 Hz, so 6 frames per turn
    at 60 fps). Step the sim n_frames times and stack the slices into one full scan. Points are in world frame."""
    chunks = []
    for _ in range(n_frames):
        step()
        d = ann.get_data()
        arr = np.asarray(d.get("data", np.zeros((0, 3))) if isinstance(d, dict) else d, dtype=np.float32)
        if arr.size:
            chunks.append(arr.reshape(-1, 3))
    return {"data": np.concatenate(chunks) if chunks else np.zeros((0, 3), np.float32)}


def save_frame(i, anns, lidar_data):
    from PIL import Image
    tag = f"{i:04d}"
    rgb = anns["rgb"].get_data()
    Image.fromarray(np.asarray(rgb)[..., :3]).save(os.path.join(args.out, f"rgb_{tag}.png"))
    np.save(os.path.join(args.out, f"depth_{tag}.npy"),
            np.asarray(anns["depth"].get_data(), dtype=np.float32))
    for name in ("instance", "semantic"):
        d = anns[name].get_data()
        Image.fromarray(np.asarray(d["data"]).astype(np.uint16)).save(os.path.join(args.out, f"{name}_{tag}.png"))
        json.dump(d.get("info", {}), open(os.path.join(args.out, f"{name}_{tag}.json"), "w"), indent=1, default=str)
    bb = anns["bbox2d"].get_data()
    json.dump({"boxes": np.asarray(bb["data"]).tolist(), "info": bb.get("info", {})},
              open(os.path.join(args.out, f"bbox2d_{tag}.json"), "w"), indent=1, default=str)
    if lidar_data is not None:
        d = lidar_data
        pts = np.asarray(d.get("data", np.zeros((0, 3))), dtype=np.float32).reshape(-1, 3)
        np.save(os.path.join(args.out, f"lidar_{tag}.npy"), pts)
        write_ply(os.path.join(args.out, f"lidar_{tag}.ply"), pts)
        for key in ("objectId", "distance", "intensity"):
            if key in d and d[key] is not None and np.size(d[key]):
                np.save(os.path.join(args.out, f"lidar_{key}_{tag}.npy"), np.asarray(d[key]))
        print(f"[box-wall] frame {tag}: {len(pts)} lidar points")


# --------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    build_static_scene(stage)
    assets = find_box_assets()
    if not assets:
        print("[box-wall] no cardboard USDs available -> procedural cartons")

    cam_path, intr = make_camera(stage)
    json.dump(intr, open(os.path.join(args.out, "camera.json"), "w"), indent=1)
    rp = rep.create.render_product(cam_path, tuple(args.res), name="CameraRP")
    anns = {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
        "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
        "instance": rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast", init_params={"colorize": False}),
        "semantic": rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False}),
        "bbox2d": rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
    }
    for a in anns.values():
        a.attach([rp])

    lidar_ann = None
    if not args.no_lidar:
        try:
            lidar_ann, linfo = make_lidar()
            json.dump(linfo, open(os.path.join(args.out, "lidar.json"), "w"), indent=1)
        except Exception as e:
            print("[box-wall] lidar setup failed, continuing with the camera only:", e)

    all_boxes = {}
    for i in range(args.frames):
        boxes = build_box_wall(stage, assets, rng)
        world.reset()
        world.play()                                          # RTX sensor annotators only collect while playing
        n_steps = 240 if args.settle else 60                  # settling time / several full lidar sweeps
        for _ in range(n_steps):
            world.step(render=True)
        if args.settle:                                       # record the settled poses
            for b in boxes:
                m = UsdGeom.Xformable(stage.GetPrimAtPath(b["prim"])).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                b["position_m"] = list(m.ExtractTranslation())
                b["rotation_matrix"] = [list(m.ExtractRotationMatrix().GetRow(r)) for r in range(3)]
        for _ in range(4):
            world.step(render=True)
        # lidar first, while the timeline is still playing (a replicator step pauses it by default)
        lidar_data = (accumulate_lidar(lidar_ann, lambda: world.step(render=True))
                      if lidar_ann is not None else None)
        for _ in range(5):                                    # camera annotators fill on a replicator step
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
            if np.asarray(anns["rgb"].get_data()).ndim == 3:
                break
        save_frame(i, anns, lidar_data)
        all_boxes[f"{i:04d}"] = boxes
    json.dump(all_boxes, open(os.path.join(args.out, "boxes.json"), "w"), indent=1)
    print(f"[box-wall] done -> {os.path.abspath(args.out)}")

    if args.gui:                                              # keep the window open until it is closed
        while simulation_app.is_running():
            world.step(render=True)


if __name__ == "__main__":
    import traceback
    ok = False
    try:
        main()
        ok = True
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        simulation_app.close()
        os._exit(0 if ok else 1)      # never leave Kit running after an error (it can hang on exit)
