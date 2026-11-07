"""Minimal Isaac Sim check: headless app, one cube, a light, a camera, 30 rendered frames, one PNG.

    ./python.sh smoke_test.py --out /out/smoke                 # plain
    ./python.sh smoke_test.py --out /out/smoke_ids --stable-ids    # same with RTX stable ids (needed for lidar object ids)

If this crashes, the problem is the container / driver / GPU, not the box-wall scripts.
"""
import argparse
import os

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
for i in range(30):
    world.step(render=True)
print("[smoke] stepped 30 frames", flush=True)
img = np.asarray(rgb.get_data())
os.makedirs(args.out, exist_ok=True)
from PIL import Image                   # noqa: E402
Image.fromarray(img[..., :3]).save(os.path.join(args.out, "smoke.png"))
print(f"[smoke] OK image {img.shape} mean={img[..., :3].mean():.1f}", flush=True)
app.close()
