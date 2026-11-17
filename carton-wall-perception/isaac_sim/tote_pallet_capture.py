"""Isaac Sim 5.1: 7-layer stack of grey Euro totes (2 x 2 per layer, 28 totes) on a EUR pallet, imaged by an
RTX lidar and an RGB-D camera mounted side by side 1.5 m in front of the stack (arm / AMR mount), for a sequence
of depalletizing states (the top-front totes removed one by one in pick order).

    ./isaac-sim.sh -p tote_pallet_capture.py --out /out/tote_pallet --picks 6
    # or through run_all.sh:  STEPS="tote_pallet" ./run_all.sh

Inputs: assets/totes/{tote.obj, pallet.obj, stack_layout.json} written by
        tote-depalletizing/scripts/export_isaac_assets.py (the layout has every tote pose and the pick order).
Output: <out>/scan_<k>/  -- the same files as the CPU simulator, so tote-depalletizing runs unchanged:
        points.npy  (N,3) lidar points, world frame (m)
        labels.npy  (N,)  tote id / -1 pallet / -2 floor / -3 other, assigned geometrically from the known poses
        rgb.png, depth_mm.png (uint16 z-depth), instances.png (uint16 tote id + 3; 0 bg, 1 floor, 2 pallet)
        gt.json     sensor_position, camera K / pose, per-tote pose / front-face centre / present flag
Removed totes are moved 50 m below the floor (a static scene: no physics needed for a depalletizing snapshot).
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tote_pallet_out")
ap.add_argument("--assets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "totes"))
ap.add_argument("--picks", type=int, default=6, help="pick states to capture (0 = full stack)")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--res", type=int, nargs=2, default=[1600, 1200])
ap.add_argument("--vfov", type=float, default=80.0, help="camera vertical field of view (deg)")
ap.add_argument("--lidar-config", default="auto",
                help="RTX lidar config; auto tries OS0 128-channel configs (+-45 deg) first, then OS1, then Example_Rotary")
ap.add_argument("--lidar-origin-z", type=float, default=None,
                help="height of the lidar's measurement origin above its prim (m). Ouster configs report points from the "
                     "optical centre 36.18 mm above the mounting base; default: 0.03618 for OS* configs, else 0")
ap.add_argument("--lidar-frames", type=int, default=18, help="per-frame lidar slices stacked into one scan")
ap.add_argument("--tote-rgb", type=float, nargs=3, default=[0.42, 0.43, 0.45])
args, _ = ap.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui, "width": args.res[0], "height": args.res[1]})

import numpy as np                                      # noqa: E402
import omni.kit.commands                                # noqa: E402
import omni.replicator.core as rep                      # noqa: E402
import omni.usd                                         # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402
try:
    from isaacsim.core.api import World
except ImportError:
    from omni.isaac.core import World

LIDAR_CONFIGS = ["OS0_REV7_128ch10hz2048res", "OS0_REV6_128ch10hz2048res", "OS0_REV6_128ch10hz1024res",
                 "OS0_128ch10hz1024res", "OS1_REV7_128ch10hz1024res", "OS1_REV6_128ch10hz1024res", "Example_Rotary"]
R_WC = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # OpenCV camera axes in the world (x right, y down)


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    return np.array(V), F


def preview_material(stage, path, rgb, roughness=0.6):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def add_mesh(stage, path, V, F, T, mat, label):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in V]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([len(f) for f in F]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([i for f in F for i in f]))
    m.CreateSubdivisionSchemeAttr("none")
    xf = UsdGeom.Xformable(m)
    op = xf.AddTransformOp()
    op.Set(Gf.Matrix4d(*np.asarray(T, float).T.ravel().tolist()))     # USD is row-vector: pass T^T
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    try:
        from omni.isaac.core.utils.semantics import add_update_semantics
    except ImportError:
        from isaacsim.core.utils.semantics import add_update_semantics
    add_update_semantics(m.GetPrim(), label)
    return op


def look_along_x(eye):
    """USD camera looks down its -Z with +Y up; aim it along world +x."""
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(0, -1, 0, 0))     # camera +X = world -y (image right)
    m.SetRow(1, Gf.Vec4d(0, 0, 1, 0))      # camera +Y = world +z
    m.SetRow(2, Gf.Vec4d(-1, 0, 0, 0))     # camera +Z = world -x (it looks along +x)
    m.SetRow(3, Gf.Vec4d(*eye, 1.0))
    return m


def make_lidar(pos):
    cfgs = LIDAR_CONFIGS if args.lidar_config == "auto" else [args.lidar_config]
    for cfg in cfgs:
        try:
            _, sensor = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="/World/Sensors/Lidar", parent=None,
                                                  config=cfg, translation=Gf.Vec3d(*pos), orientation=Gf.Quatd(1, 0, 0, 0))
            if sensor is None:
                raise RuntimeError("no sensor prim")
            rp = rep.create.render_product(sensor.GetPath(), [1, 1], name="LidarRP")
            ann = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
            ann.attach([rp.path])
            print("[totes] lidar config", cfg)
            return ann, cfg
        except Exception as e:                       # noqa: BLE001
            print("[totes] lidar config", cfg, "failed:", e)
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath("/World/Sensors/Lidar"):
                stage.RemovePrim("/World/Sensors/Lidar")
    raise RuntimeError("no RTX lidar config worked")


def accumulate(ann, step, n_frames=18, az_fov=65.0, max_range=8.0):
    """Stack n_frames per-frame slices of the rotary RTX lidar (one azimuth slice per rendered frame; 10 Hz at
    60 fps is 6 frames per turn, so 18 frames is three turns). Keep the forward sector, drop exact duplicates
    from overlapping turns, and report azimuth coverage. Points are in the sensor frame."""
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()
    if not tl.is_playing():
        tl.play()
    chunks = []
    for _ in range(n_frames):
        step()
        d = ann.get_data()
        a = np.asarray(d.get("data", np.zeros((0, 3))) if isinstance(d, dict) else d, np.float32)
        if a.size:
            chunks.append(a.reshape(-1, 3))
    if not chunks:
        print("[totes] WARNING: lidar returned no points")
        return np.zeros((0, 3), np.float32)
    a = np.concatenate(chunks)
    r = np.linalg.norm(a, axis=1); az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
    a = a[(np.abs(az) < az_fov) & (r > 0.3) & (r < max_range)]
    a = np.unique(np.round(a, 4), axis=0)
    az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
    cov = len(np.unique(np.floor(az + az_fov).astype(int))) / (2 * az_fov)
    print(f"[totes] lidar: {len(chunks)} slices, {len(a)} points in +-{az_fov:.0f} deg, azimuth coverage {cov:.2f}")
    return a


def label_points(P, layout, present, lo, hi, pallet_x):
    lab = np.full(len(P), -3, np.int16)
    lab[P[:, 2] < 0.01] = -2
    pm = (np.abs(P[:, 0] - pallet_x) < 0.41) & (np.abs(P[:, 1]) < 0.61) & (P[:, 2] >= 0.01) & (P[:, 2] < 0.146)
    lab[pm] = -1
    for t in layout["totes"]:
        if t["id"] not in present:
            continue
        T = np.array(t["T"]); q = (P - T[:3, 3]) @ T[:3, :3]
        lab[np.all((q > lo - 0.01) & (q < hi + 0.01), 1)] = t["id"]
    return lab


def main():
    from PIL import Image
    layout = json.load(open(os.path.join(args.assets, "stack_layout.json")))
    Vt_, Ft = load_obj(os.path.join(args.assets, "tote.obj"))
    Vp, Fp = load_obj(os.path.join(args.assets, "pallet.obj"))
    lo, hi = np.array(layout["tote_bounds"][0]), np.array(layout["tote_bounds"][1])
    c_front = np.r_[layout["front_face_center"], 1.0]; n_front = np.array(layout["front_face_normal"])
    os.makedirs(args.out, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    grey = preview_material(stage, "/World/Looks/ToteGrey", args.tote_rgb)
    wood = preview_material(stage, "/World/Looks/Wood", (0.55, 0.42, 0.28), 0.8)
    floor_m = preview_material(stage, "/World/Looks/Concrete", (0.3, 0.3, 0.3), 0.9)
    fl = UsdGeom.Cube.Define(stage, "/World/Floor"); fl.CreateSizeAttr(1.0)
    fx = UsdGeom.Xformable(fl); fx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.01)); fx.AddScaleOp().Set(Gf.Vec3f(20, 20, 0.02))
    UsdShade.MaterialBindingAPI.Apply(fl.GetPrim()).Bind(floor_m)
    Tp = np.eye(4); Tp[0, 3] = layout["pallet_x"]
    add_mesh(stage, "/World/Pallet", Vp, Fp, Tp, wood, "pallet")
    ops = {}
    for t in layout["totes"]:
        ops[t["id"]] = add_mesh(stage, f"/World/Totes/Tote_{t['id']:02d}", Vt_, Ft, np.array(t["T"]), grey, f"tote_{t['id']:02d}")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome"); dome.CreateIntensityAttr(600)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Ceiling"); sun.CreateIntensityAttr(1500); sun.CreateAngleAttr(2.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-20, 25, 0))
    rect = UsdLux.RectLight.Define(stage, "/World/Lights/HeadLamp"); rect.CreateIntensityAttr(20000)
    rect.CreateWidthAttr(0.3); rect.CreateHeightAttr(0.3)
    UsdGeom.Xformable(rect).AddTransformOp().Set(look_along_x((0.05, 0.0, 1.45)))

    # camera beside the lidar, same intrinsics as the CPU simulator (pinhole, square pixels)
    W, H = args.res
    f_px = (H / 2) / np.tan(np.radians(args.vfov) / 2)
    cam = UsdGeom.Camera.Define(stage, "/World/Sensors/Camera")
    focal_mm = 10.0; v_ap = H * focal_mm / f_px; h_ap = W * focal_mm / f_px
    cam.CreateFocalLengthAttr(focal_mm); cam.CreateHorizontalApertureAttr(h_ap); cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 50.0))
    C = np.array(layout["camera"], float)
    UsdGeom.Xformable(cam).AddTransformOp().Set(look_along_x(C))
    K = np.array([[f_px, 0, W / 2], [0, f_px, H / 2], [0, 0, 1.0]])
    rp = rep.create.render_product(cam.GetPath().pathString, (W, H), name="CameraRP")
    anns = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            "instance": rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False})}
    for a in anns.values():
        a.attach([rp])
    sensor = np.array(layout["sensor"], float)
    lidar_ann, cfg = make_lidar(sensor)

    order = layout["pick_order"]
    world.reset()
    world.play()
    for _ in range(30):
        world.step(render=True)
    def set_state(present):
        for t in layout["totes"]:
            T = np.array(t["T"], float)
            if t["id"] not in present:
                T = T.copy(); T[2, 3] -= 50.0                  # park removed totes under the floor
            ops[t["id"]].Set(Gf.Matrix4d(*T.T.ravel().tolist()))

    n = min(args.picks, len(order))
    oz = args.lidar_origin_z if args.lidar_origin_z is not None else (0.03618 if cfg.startswith("OS") else 0.0)
    # pass 1: lidar for every pick state, stepping the world only. In the first run, a replicator step for the
    # camera between lidar captures left the rotary lidar returning partial or empty sweeps afterwards.
    scans = {}
    for k in range(n):
        present = set(order[k:])
        set_state(present)
        for _ in range(8):
            world.step(render=True)
        pts_s = accumulate(lidar_ann, lambda: world.step(render=True), args.lidar_frames)
        scans[k] = pts_s.astype(float) + sensor + np.array([0.0, 0.0, oz])     # sensor frame: identity rotation
    # pass 2: camera images for every pick state
    for k in range(n):
        present = set(order[k:])
        set_state(present)
        for _ in range(3):
            world.step(render=True)
        for _ in range(5):
            rep.orchestrator.step(rt_subframes=8, pause_timeline=False)
            if np.asarray(anns["rgb"].get_data()).ndim == 3:
                break
        P = scans[k]
        d = os.path.join(args.out, f"scan_{k:02d}"); os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "points.npy"), P.astype(np.float32))
        np.save(os.path.join(d, "labels.npy"), label_points(P, layout, present, lo, hi, layout["pallet_x"]))
        Image.fromarray(np.asarray(anns["rgb"].get_data())[..., :3]).save(os.path.join(d, "rgb.png"))
        depth = np.asarray(anns["depth"].get_data(), np.float32)
        Image.fromarray(np.where(np.isfinite(depth), np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(os.path.join(d, "depth_mm.png"))
        inst = anns["instance"].get_data(); ids = np.asarray(inst["data"]).astype(np.int64)
        lut = {}
        for key, path in inst.get("info", {}).get("idToLabels", {}).items():
            p = str(path)
            lut[int(key)] = int(p.split("Tote_")[1][:2]) + 3 if "Tote_" in p else (2 if "Pallet" in p else 1 if "Floor" in p else 0)
        out = np.zeros(ids.shape, np.uint16)
        for key, v in lut.items():
            out[ids == key] = v
        Image.fromarray(out).save(os.path.join(d, "instances.png"))
        totes = []
        for t in layout["totes"]:
            T = np.array(t["T"])
            totes.append(dict(id=t["id"], layer=t["layer"], row=t["row"], col=t["col"], present=t["id"] in present,
                              T=T.tolist(), front_center=(T @ c_front)[:3].tolist(), front_normal=(T[:3, :3] @ n_front).tolist()))
        json.dump(dict(sensor_position=sensor.tolist(), totes=totes, source=f"isaac_sim_rtx_lidar:{cfg}",
                       lidar_origin_z=oz, removed=order[:k], pick_order=order,
                       camera=dict(K=K.tolist(), width=W, height=H, position=C.tolist(), R_world_from_cam=R_WC.tolist(),
                                   model="pinhole, no distortion; x right, y down, z optical axis")),
                  open(os.path.join(d, "gt.json"), "w"), indent=1)
        fl = P[:, 2] < 0.05
        print(f"[totes] floor check: median z of floor returns {np.median(P[fl, 2]) if fl.any() else float('nan'):+.4f} m (should be ~0)")
        hits = np.unique(np.load(os.path.join(d, "labels.npy")), return_counts=True)
        print(f"[totes] scan {k:02d}: {len(P)} lidar points, label counts {dict(zip(hits[0].tolist(), hits[1].tolist()))}")
    print("[totes] done ->", os.path.abspath(args.out))


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
        os._exit(0 if ok else 1)
