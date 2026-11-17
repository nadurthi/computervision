"""CPU stand-in scans of the 7-layer tote stack, in the same format the Isaac Sim script writes.

    python scripts/simulate_scans.py --out data/sim --picks 28 --sigma 0.005

For every pick state k (k totes already removed in pick order) it writes <out>/scan_<k>/:
    points.npy (N,3 world), labels.npy (N,) ground-truth tote id / -1 pallet / -2 floor, gt.json
    rgb.png (camera beside the lidar), depth_mm.png (uint16 z-depth, mm), instances.png (uint16 tote id + 3;
    0 = background, 1 = floor, 2 = pallet) -- camera intrinsics K and pose in gt.json["camera"]
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import trimesh
from totes.cad import ToteSpec, make_tote, make_pallet
from totes.scene import make_stack, pick_order, save_gt
from totes.lidar import build_scene, scan, OS0_128
from totes.camera import render, intrinsics, R_WC
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/sim")
ap.add_argument("--picks", type=int, default=28, help="number of pick states to scan (0..picks-1)")
ap.add_argument("--sigma", type=float, default=0.005, help="range noise std (m)")
ap.add_argument("--sensor", default="0.1,0,1.2", help="lidar position x,y,z (m); stack front is at x=1.6")
ap.add_argument("--camera", default="0.1,-0.15,1.2", help="camera centre x,y,z (m); beside the lidar")
ap.add_argument("--cam-size", default="1600x1200")
ap.add_argument("--cam-vfov", type=float, default=80.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cad", default=None, help="user CAD mesh instead of the stand-in")
a = ap.parse_args()

spec = ToteSpec()
tote = trimesh.load(a.cad, force="mesh") if a.cad else make_tote(spec)
pallet = make_pallet()
totes = make_stack(seed=a.seed, spec=spec)
order = pick_order(totes)
sensor = np.array([float(v) for v in a.sensor.split(",")])
rng = np.random.default_rng(a.seed)
C = np.array([float(v) for v in a.camera.split(",")])
W, H = (int(v) for v in a.cam_size.split("x"))
K = intrinsics(W, H, a.cam_vfov)
for k in range(min(a.picks, len(order))):
    present = set(order[k:])
    sc, labels = build_scene(tote, totes, present, pallet)
    pts, lab, r, inc = scan(sc, labels, sensor, OS0_128, a.sigma, rng)
    d = os.path.join(a.out, f"scan_{k:02d}"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "points.npy"), pts.astype(np.float32))
    np.save(os.path.join(d, "labels.npy"), lab.astype(np.int16))
    img, depth, inst = render(sc, labels, C, K, W, H, rng)
    Image.fromarray(img).save(os.path.join(d, "rgb.png"))
    Image.fromarray(np.where(np.isfinite(depth), np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(os.path.join(d, "depth_mm.png"))
    Image.fromarray((inst.astype(np.int32) + 3).astype(np.uint16)).save(os.path.join(d, "instances.png"))
    cam = dict(K=K.tolist(), width=W, height=H, position=C.tolist(), R_world_from_cam=R_WC.tolist(),
               model="pinhole, no distortion; x right, y down, z optical axis")
    save_gt(os.path.join(d, "gt.json"), totes, present, sensor,
            extra=dict(source="cpu_raycast_os0_128", sigma=a.sigma, removed=order[:k], pick_order=order, camera=cam))
    vis = sorted(set(lab[lab >= 0].tolist()))
    print(f"scan {k:02d}: {len(pts)} points, totes hit: {vis}")
