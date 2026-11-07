"""Trailer box wall: cardboard boxes of different sizes stacked tightly side by side and upwards against the
back (nose) wall of a 53 ft dry-van trailer, seen from the door. Front faces are deliberately not aligned: each
box has its own depth, a small push in or out, and a small yaw, the way a real hand-loaded wall looks.

The scene is static (no physics), so the wall looks exactly as generated: nothing topples while it settles.

    python download_simready_assets.py --out ~/simready --all-cartons   # ~80 NVIDIA SimReady carton models
    ./python.sh trailer_wall.py --out trailer_out --box-usd-dir ~/simready   # 1 image
    ./python.sh trailer_wall.py --gui                                    # look around in the Isaac Sim window

Outputs (in --out):
    rgb.png             colour image from the door
    depth.npy           distance to the image plane (m, float32)
    instance.png        one id per box (uint16), instance.json maps ids to prims
    boxes.json          size (m), centre, yaw and front-face position of every box
    camera.json         intrinsics and pose

Layout parameters (all in metres or degrees) are command-line options; see --help.
"""
import argparse
import glob
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="trailer_out")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
ap.add_argument("--box-usd-dir", default=os.path.expanduser("~/simready"),
                help="SimReady download folder (download_simready_assets.py --all-cartons); procedural boxes only if empty")
ap.add_argument("--size-min", type=float, default=0.30, help="smallest box edge")
ap.add_argument("--size-max", type=float, default=0.62, help="largest box edge")
ap.add_argument("--cube-ness", type=float, default=0.15,
                help="how far a box may deviate from a cube: each edge = s * (1 +/- this)")
ap.add_argument("--stack-height", type=float, default=2.35, help="wall height (trailer interior is 2.70)")
ap.add_argument("--rows", type=int, default=2, help="rows of boxes from the back wall towards the door")
ap.add_argument("--yaw", type=float, default=5.0, help="max |yaw| of a box (deg)")
ap.add_argument("--push", type=float, default=0.05, help="max front-face offset in/out of a box (m)")
ap.add_argument("--gap", type=float, default=0.006, help="nominal gap between neighbouring boxes (m)")
ap.add_argument("--cam-dist", type=float, default=3.2, help="camera distance from the back wall")
ap.add_argument("--cam-height", type=float, default=1.45)
ap.add_argument("--subframes", type=int, default=32, help="RTX accumulation subframes for the final image")
args, _ = ap.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui, "width": args.res[0], "height": args.res[1]})

import numpy as np                                  # noqa: E402
import omni.replicator.core as rep                  # noqa: E402
import omni.usd                                     # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade   # noqa: E402

# trailer interior (53 ft dry van); x = 0 at the door, back (nose) wall at x = L
L, W, H = 16.0, 2.46, 2.70
rng = np.random.default_rng(args.seed)


# ------------------------------------------------------------------------------------------ helpers
def add_label(prim, label):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
    except Exception:
        try:
            from isaacsim.core.utils.semantics import add_update_semantics
            add_update_semantics(prim, label)
        except Exception:
            pass


def material(stage, path, rgb, rough=0.8, metal=0.0):
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


def cube(stage, path, center, size, mat=None, label=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if label:
        add_label(c.GetPrim(), label)
    return c


def bounds(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*r, 0)); m.SetRow(1, Gf.Vec4d(*u, 0)); m.SetRow(2, Gf.Vec4d(*(-f), 0))
    m.SetRow(3, Gf.Vec4d(*eye, 1))
    return m


# ------------------------------------------------------------------------------------------ trailer
def build_trailer(stage):
    steel = material(stage, "/World/Looks/Steel", (0.80, 0.81, 0.82), 0.40, 0.7)
    wood = material(stage, "/World/Looks/Floor", (0.40, 0.28, 0.18), 0.85)
    liner = material(stage, "/World/Looks/Liner", (0.86, 0.86, 0.84), 0.6)       # white plywood/FRP liner
    b = "/World/Trailer"
    UsdGeom.Xform.Define(stage, b)
    cube(stage, b + "/Floor", (L / 2, 0, -0.025), (L, W + 0.1, 0.05), wood, "trailer")
    cube(stage, b + "/Roof", (L / 2, 0, H + 0.03), (L, W + 0.1, 0.06), liner, "trailer")
    cube(stage, b + "/Nose", (L + 0.03, 0, H / 2), (0.06, W + 0.1, H), liner, "trailer")
    for side, y in (("L", W / 2 + 0.03), ("R", -W / 2 - 0.03)):
        cube(stage, f"{b}/Wall{side}", (L / 2, y, H / 2), (L, 0.06, H), liner, "trailer")
        # posts and rails are siblings of the wall: children of a Cube (a gprim) are not rendered
        UsdGeom.Xform.Define(stage, f"{b}/Trim{side}")
        for i, x in enumerate(np.arange(0.4, L, 0.6)):                 # vertical logistics posts
            cube(stage, f"{b}/Trim{side}/Post{i:02d}", (x, y - np.sign(y) * 0.04, H / 2),
                 (0.05, 0.02, H), steel)
        for k, z in enumerate((0.9, 1.7)):                               # logistics track rails
            cube(stage, f"{b}/Trim{side}/Rail{k}", (L / 2, y - np.sign(y) * 0.045, z), (L, 0.012, 0.12), steel)


def build_lights(stage):
    UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(150.0)
    for k, x in enumerate(np.arange(L - 1.0, L - 9.0, -2.0)):           # ceiling dome lights inside the box
        rl = UsdLux.RectLight.Define(stage, f"/World/Lights/Ceiling{k}")
        rl.CreateIntensityAttr(9000.0); rl.CreateWidthAttr(0.6); rl.CreateHeightAttr(0.25)
        UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(float(x), 0.0, H - 0.02))  # rect light faces -Z
    # dock light behind the camera, aimed into the trailer
    dl = UsdLux.SphereLight.Define(stage, "/World/Lights/Dock")
    dl.CreateIntensityAttr(60000.0); dl.CreateRadiusAttr(0.15)
    UsdGeom.Xformable(dl).AddTranslateOp().Set(Gf.Vec3d(L - args.cam_dist - 1.0, 0.6, 2.2))


# ------------------------------------------------------------------------------------------ boxes
def box_assets():
    """One USD per SimReady carton folder (<name>/simready_usd/sm_<name>*.usd), or any USDs in --box-usd-dir."""
    if not args.box_usd_dir:
        return []
    f = []
    for d in sorted(glob.glob(os.path.join(args.box_usd_dir, "box_*", "simready_usd"))):
        name = os.path.basename(os.path.dirname(d))
        if any(k in name for k in ("_open", "_folded", "_flat", "_long", "_glove", "_paper_office", "composition")):
            continue                                          # not closed shipping cartons
        c = sorted(glob.glob(os.path.join(d, f"sm_{name}*.usd*"))) or sorted(glob.glob(os.path.join(d, "*.usd*")))
        if c:
            f.append(c[0])
    if not f:
        f = sorted(glob.glob(os.path.join(args.box_usd_dir, "**", "*.usd*"), recursive=True))
    return f


def measure_assets(stage, urls):
    """Native bounding-box size of each model (referenced once under a scratch prim, then removed)."""
    out = []
    for k, url in enumerate(urls):
        path = f"/World/_probe{k}"
        try:
            UsdGeom.Xform.Define(stage, path).GetPrim().GetReferences().AddReference(url)
            ext = np.array(bounds(stage.GetPrimAtPath(path)).GetSize(), float)
            if np.all(ext > 1e-6):
                if ext.max() > 20:                            # authored in centimetres
                    ext = ext / 100.0
                out.append((url, ext))
        except Exception as e:
            print(f"[trailer] cannot load {url}: {e}")
        stage.RemovePrim(path)
    return out


def pick_model(models, size, k=4):
    """Random choice among the k models whose proportions are closest to `size` (least stretching of the print).
    Distance = spread of log(target/native) over the three axes, so overall scale does not matter."""
    t = np.log(np.asarray(size, float))
    cost = [np.ptp(t - np.log(ext)) for _, ext in models]
    best = np.argsort(cost)[:k]
    return models[int(rng.choice(best))][0]


def cardboard_materials(stage):
    shades = [(0.63, 0.47, 0.30), (0.70, 0.54, 0.35), (0.56, 0.41, 0.26), (0.74, 0.60, 0.42), (0.60, 0.45, 0.31)]
    return [material(stage, f"/World/Looks/Card{i}", c, 0.88) for i, c in enumerate(shades)]


def procedural_box(stage, path, size, card, tape):
    """A carton: body + a tape strip over the top seam and down the front face."""
    d, w, h = size
    cube(stage, path + "/body", (0, 0, h / 2), (d, w, h), card)
    t = min(0.05, 0.12 * w)
    cube(stage, path + "/tape_top", (0, 0, h + 0.0008), (d + 0.002, t, 0.0016), tape)
    cube(stage, path + "/tape_front", (-d / 2 - 0.0008, 0, h - 0.06), (0.0016, t, 0.12), tape)


def referenced_box(stage, path, url, size):
    """Reference a box USD and scale it so its bounding box is exactly `size`, bottom-centre at the origin."""
    fit = UsdGeom.Xform.Define(stage, path + "/fit")
    asset = UsdGeom.Xform.Define(stage, path + "/fit/asset").GetPrim()
    asset.GetReferences().AddReference(url)
    # bounds in the fit prim's own frame (the box's pose prim above is already placed and rotated)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeRelativeBound(asset, fit.GetPrim()).ComputeAlignedRange()
    ext, ctr = np.array(r.GetSize()), np.array(r.GetMidpoint())
    if np.any(ext <= 1e-6):
        raise RuntimeError("empty bounds")
    s = np.asarray(size) / ext
    fx = UsdGeom.Xformable(fit)
    fx.AddTranslateOp().Set(Gf.Vec3d(-s[0] * ctr[0], -s[1] * ctr[1], -s[2] * r.GetMin()[2]))
    fx.AddScaleOp().Set(Gf.Vec3f(*s))
    # static scene: drop the rigid bodies SimReady assets ship with, so no render step can move a box
    from pxr import UsdPhysics
    for _ in range(4):
        inst = [q for q in Usd.PrimRange(fit.GetPrim()) if q.IsInstance()]
        if not inst:
            break
        for q in inst:
            q.SetInstanceable(False)
    for q in Usd.PrimRange(fit.GetPrim()):
        if q.HasAPI(UsdPhysics.RigidBodyAPI):
            q.RemoveAPI(UsdPhysics.RigidBodyAPI)


def footprint_half_width(d, w, yaw_rad):
    """Half extent along y of a d x w rectangle rotated by yaw."""
    return 0.5 * (w * abs(math.cos(yaw_rad)) + d * abs(math.sin(yaw_rad)))


def plan_wall(rng, a):
    """Pure layout (no USD): a list of boxes with size, centre, yaw. Columns span the trailer width, boxes stack
    upwards in each column, and rows go from the back wall towards the door. Each box gets its own depth, yaw and a
    push towards the door, so the front faces are not aligned; widths are shrunk for the yaw so rotated
    neighbours never intersect."""
    boxes = []
    usable = W - 0.04                                  # small clearance to the side walls
    prev = None                                        # previous row: (y0, y1, front x) per box
    for row in range(a.rows):
        y = -usable / 2
        row_front = []
        while y < usable / 2 - a.size_min * 0.8:
            col_w = min(rng.uniform(a.size_min, a.size_max), usable / 2 - y)
            if prev is None:                           # first row: back faces against the nose wall
                back_x = L - 0.005
            else:                                      # later rows: in front of the boxes directly behind
                behind = [f for (y0, y1, f) in prev if y0 < y + col_w and y1 > y]
                back_x = (min(behind) if behind else min(f for _, _, f in prev)) - a.gap
            z = 0.0
            top = rng.uniform(0.85, 1.0) * a.stack_height
            while True:
                s = rng.uniform(a.size_min, a.size_max)
                h = s * rng.uniform(1 - a.cube_ness, 1 + a.cube_ness)
                if z + h > top:
                    break
                d = s * rng.uniform(1 - a.cube_ness, 1 + a.cube_ness)
                yaw = math.radians(rng.uniform(-a.yaw, a.yaw))
                w_max = (col_w - a.gap - abs(d * math.sin(yaw))) / max(abs(math.cos(yaw)), 1e-3)
                w = max(0.5 * col_w, min(w_max, col_w * rng.uniform(0.92, 1.0)))
                half_y = footprint_half_width(d, w, yaw)
                half_x = 0.5 * (d * abs(math.cos(yaw)) + w * abs(math.sin(yaw)))
                push = rng.uniform(0.0, a.push)                  # pulled towards the door by up to `push`
                cx = back_x - half_x - push
                slack = max(0.0, col_w - a.gap - 2 * half_y)
                cy = y + col_w / 2 + rng.uniform(-0.5, 0.5) * slack
                boxes.append({"row": row, "size_m": [d, w, h], "center_m": [cx, cy, z + h / 2],
                              "yaw_deg": math.degrees(yaw), "front_face_x_m": cx - half_x,
                              "half_extent_xy_m": [half_x, half_y]})
                row_front.append((cy - half_y, cy + half_y, cx - half_x))
                z += h + 0.002                                   # stacked upwards, 2 mm apart
            y += col_w
        prev = row_front or prev
    return boxes


def build_wall(stage):
    models = measure_assets(stage, box_assets())
    if models:
        print(f"[trailer] {len(models)} SimReady carton models")
    else:
        print("[trailer] WARNING: no SimReady cartons found (--box-usd-dir); using procedural cardboard boxes")
    cards = cardboard_materials(stage)
    tape = material(stage, "/World/Looks/Tape", (0.72, 0.58, 0.40), 0.35)
    UsdGeom.Xform.Define(stage, "/World/Boxes")
    boxes = plan_wall(rng, args)
    for idx, b in enumerate(boxes):
        d, w, h = b["size_m"]
        cx, cy, cz = b["center_m"]
        path = f"/World/Boxes/box_{idx:03d}"
        top_prim = UsdGeom.Xform.Define(stage, path)
        xf = UsdGeom.Xformable(top_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz - h / 2 + 0.001))
        xf.AddRotateZOp().Set(b["yaw_deg"])
        used = "procedural"
        if models:
            url = pick_model(models, (d, w, h))
            try:
                referenced_box(stage, path, url, (d, w, h))
                used = os.path.basename(url)
            except Exception as e:
                print(f"[trailer] {url}: {e}; procedural instead")
                stage.RemovePrim(path + "/fit")
        if used == "procedural":
            procedural_box(stage, path, (d, w, h), cards[idx % len(cards)], tape)
        add_label(top_prim.GetPrim(), "carton")
        b["prim"], b["model"] = path, used
    if boxes:                                                   # sanity check: the first box's world bounds
        r = bounds(stage.GetPrimAtPath(boxes[0]["prim"]))
        mid, plan = np.array(r.GetMidpoint()), np.array(boxes[0]["center_m"])
        print(f"[trailer] {boxes[0]['prim']} ({boxes[0]['model']}) centre {np.round(mid, 3).tolist()}, "
              f"planned {np.round(plan, 3).tolist()}, size {np.round(np.array(r.GetSize()), 3).tolist()}")
        if np.linalg.norm(mid - plan) > 0.05:
            print("[trailer] WARNING: first box is not where it was planned")
    fronts = [b["front_face_x_m"] for b in boxes]
    print(f"[trailer] {len(boxes)} boxes in {args.rows} rows; front faces spread over "
          f"{100 * (max(fronts) - min(fronts)):.1f} cm; yaw within +/-{args.yaw} deg")
    return boxes


# ------------------------------------------------------------------------------------------ main
def main():
    os.makedirs(args.out, exist_ok=True)
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    build_trailer(stage)
    build_lights(stage)
    boxes = build_wall(stage)

    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    focal, hap = 14.0, 20.955                                    # ~73 deg horizontal FOV
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(hap)
    cam.CreateVerticalApertureAttr(hap * args.res[1] / args.res[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 40.0))
    eye = (L - args.cam_dist, 0.0, args.cam_height)
    target = (L, 0.0, args.stack_height * 0.5)
    UsdGeom.Xformable(cam).AddTransformOp().Set(look_at_matrix(eye, target))
    fx = args.res[0] * focal / hap
    json.dump({"width": args.res[0], "height": args.res[1], "fx": fx, "fy": fx, "cx": args.res[0] / 2,
               "cy": args.res[1] / 2, "eye_m": eye, "target_m": target}, open(os.path.join(args.out, "camera.json"), "w"),
              indent=1)

    rp = rep.create.render_product("/World/Camera", tuple(args.res))
    anns = {"rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
            "instance": rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast",
                                                            init_params={"colorize": False})}
    for a in anns.values():
        a.attach([rp])

    # Let every referenced SimReady model (geometry, MDL materials, textures) stream into the renderer before the
    # final frame: wait for the stage to finish loading, then keep rendering until the instance annotator sees
    # the boxes. On a fresh machine the first material compile takes minutes.
    import time
    t0 = time.time()
    while time.time() - t0 < 600:
        simulation_app.update()
        _msg, loaded, total = ctx.get_stage_loading_status()
        if total == 0 or loaded >= total:
            break
    for _ in range(60):
        simulation_app.update()
    seen = 0
    for k in range(40):                                         # up to ~40 render passes
        rep.orchestrator.step(rt_subframes=4)
        ins = anns["instance"].get_data()
        labels = (ins or {}).get("info", {}).get("idToLabels", {}) if isinstance(ins, dict) else {}
        seen = sum("/World/Boxes/" in str(v) for v in labels.values())
        if seen >= max(1, len(boxes) // 3):
            break
        time.sleep(2)
    print(f"[trailer] boxes visible to the camera: {seen} (after {k + 1} passes, {time.time() - t0:.0f} s)")
    rep.orchestrator.step(rt_subframes=args.subframes)          # final, well-converged frame

    from PIL import Image
    rgb = np.asarray(anns["rgb"].get_data())[..., :3]
    Image.fromarray(rgb).save(os.path.join(args.out, "rgb.png"))
    np.save(os.path.join(args.out, "depth.npy"), np.asarray(anns["depth"].get_data(), dtype=np.float32))
    ins = anns["instance"].get_data()
    Image.fromarray(np.asarray(ins["data"]).astype(np.uint16)).save(os.path.join(args.out, "instance.png"))
    json.dump(ins.get("info", {}), open(os.path.join(args.out, "instance.json"), "w"), indent=1, default=str)
    json.dump(boxes, open(os.path.join(args.out, "boxes.json"), "w"), indent=1)
    print(f"[trailer] wrote {args.out}/rgb.png ({rgb.shape[1]}x{rgb.shape[0]}, mean {rgb.mean():.0f})")

    if args.gui:
        while simulation_app.is_running():
            simulation_app.update()


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
