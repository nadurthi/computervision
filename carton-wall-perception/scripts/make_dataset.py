"""Render the synthetic box-wall dataset.

data/<split>/<id>_rgb.png        8-bit RGB
data/<split>/<id>_depth.png      16-bit depth in millimetres (0 = no return)
data/<split>/<id>_inst.png       16-bit instance ids (0 = trailer / background)
data/<split>/<id>_gt.json        per-box pose, size, row, visible / front-face pixel counts
data/<split>/<id>_lidar.npz      (test splits) lidar points + per-point instance ids
"""
import argparse, json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.scene import make_scene
from boxwall.render import render_rgbd, render_lidar, FX, FY, W, H

SPLITS = {"train": (0, 1200, 1.0, False), "val": (5000, 80, 1.0, False),
          "test": (9000, 100, 1.0, True), "test_hard": (12000, 100, 1.8, True)}


def one(args):
    split, seed, diff, lidar, root = args
    out = os.path.join(root, split)
    sc = make_scene(seed, diff)
    r = render_rgbd(sc)
    sid = f"{seed:05d}"
    cv2.imwrite(os.path.join(out, sid + "_rgb.png"), r["rgb"][:, :, ::-1])
    cv2.imwrite(os.path.join(out, sid + "_depth.png"), np.clip(r["depth"] * 1000, 0, 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(out, sid + "_inst.png"), r["inst"].astype(np.uint16))
    boxes = []
    for i, b in enumerate(sc.boxes, start=1):
        vis = int((r["inst"] == i).sum())
        fr = int(((r["inst"] == i) & r["front"]).sum())
        boxes.append({"id": i, "center": b.center.tolist(), "R": b.R.tolist(), "size": (2 * b.half).tolist(),
                      "row": b.row, "sku": b.sku, "visible_px": vis, "front_px": fr})
    json.dump({"seed": seed, "difficulty": diff, "intrinsics": [FX, FY, W / 2 - 0.5, H / 2 - 0.5, W, H],
               "boxes": boxes}, open(os.path.join(out, sid + "_gt.json"), "w"))
    if lidar:
        L = render_lidar(sc)
        np.savez_compressed(os.path.join(out, sid + "_lidar.npz"), **{k: v for k, v in L.items()})
    return sid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = []
    for s in a.splits:
        start, n, diff, lidar = SPLITS[s]
        os.makedirs(os.path.join(a.root, s), exist_ok=True)
        jobs += [(s, start + k, diff, lidar, a.root) for k in range(n)]
    with Pool(a.workers) as p:
        for k, _ in enumerate(p.imap_unordered(one, jobs, chunksize=4)):
            if k % 100 == 0:
                print(k, "/", len(jobs), flush=True)
    print("done", len(jobs))
