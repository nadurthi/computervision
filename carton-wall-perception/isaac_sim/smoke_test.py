"""Minimal Isaac Sim check: headless app, one cube, a light, a camera, rendered frames, one PNG.

    ./python.sh smoke_test.py --out /out/smoke                    # plain
    ./python.sh smoke_test.py --out /out/smoke_ids --stable-ids   # same with RTX stable ids (lidar object ids)

If this crashes, the problem is the container / driver / GPU, not the box-wall scripts.
"""
import argparse
import os
import sys
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="smoke_out")
ap.add_argument("--stable-ids", action="store_true")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp

cfg = {"headless": True, "width": 640, "height": 480}
if args.stable_ids:
    cfg["extra_args"] = ["--/rtx-transient/stableIds/enabled=true"]
app = SimulationApp(cfg)
print("[smoke] app started", flush=True)

ok = False
try:
    import numpy as np                      # noqa: E402
    import omni.replicator.core as rep      # noqa: E402
    import omni.usd                         # noqa: E402
    from pxr import Gf, UsdGeom, UsdLux     # noqa: E402
    from isaacsim.core.api import World     # noqa: E402

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000.0)
    cam = rep.create.camera(position=(3, 3, 2), look_at=(0, 0, 0.5))
    rp = rep.create.render_product(cam, (640, 480))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    print("[smoke] scene built", flush=True)
    world.reset()
    for _ in range(10):
        world.step(render=True)
    img = None
    for attempt in range(5):        # annotators fill on a replicator step; the first ones can be empty
        rep.orchestrator.step(rt_subframes=4)
        img = np.asarray(rgb.get_data())
        if img.ndim == 3 and img.shape[0] > 0:
            break
    print(f"[smoke] rgb data: shape {img.shape} dtype {img.dtype}", flush=True)
    if img.ndim != 3:
        raise RuntimeError("rgb annotator returned no image")
    os.makedirs(args.out, exist_ok=True)
    from PIL import Image                   # noqa: E402
    Image.fromarray(img[..., :3].astype(np.uint8)).save(os.path.join(args.out, "smoke.png"))
    print(f"[smoke] OK image {img.shape} mean={img[..., :3].mean():.1f}", flush=True)
    ok = True
except Exception:
    traceback.print_exc()
finally:
    sys.stdout.flush()
    app.close()
    os._exit(0 if ok else 1)    # never leave Kit running after an error
