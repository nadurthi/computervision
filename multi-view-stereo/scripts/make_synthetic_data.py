#!/usr/bin/env python3
"""Write a tiny ray-traced dataset (exact depth for every view) in the DTU and MiddEval3 layouts, so the whole
pipeline (scripts/run_benchmark.sh with the `smoke` profile) can run on any machine, CPU only, with no downloads.

  python scripts/make_synthetic_data.py /tmp/mvs_smoke/data
  SKIP_SETUP=1 DEVICE=cpu bash scripts/run_benchmark.sh smoke /tmp/mvs_smoke
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import smoke_test as S                                       # noqa: E402  (scene + cameras)
from mvsbench import geometry as G                           # noqa: E402
from mvsbench.io import write_mvsnet_cam, write_pair, write_pfm  # noqa: E402


def main(root):
    cams = [S.look([0, 0, 0]), S.look([0.6, 0, 0], yaw=-0.03), S.look([-0.6, 0, 0], yaw=0.03),
            S.look([0, -0.5, 0], pitch=0.03), S.look([0, 0.5, 0.1]), S.look([1.2, 0.1, 0], yaw=-0.06)]
    d = os.path.join(root, "dtu", "eval", "scan1")
    gd = os.path.join(root, "dtu", "Depths_raw", "scan1")
    for p in (os.path.join(d, "images"), os.path.join(d, "cams"), gd):
        os.makedirs(p, exist_ok=True)
    renders = [S.render(E) for E in cams]
    pairs = []
    for i, (E, (img, dep)) in enumerate(zip(cams, renders)):
        cv2.imwrite(os.path.join(d, "images", f"{i:08d}.jpg"), img)
        write_mvsnet_cam(os.path.join(d, "cams", f"{i:08d}_cam.txt"), S.K, E, 4.0, 12.0)
        write_pfm(os.path.join(gd, f"depth_map_{i:04d}.pfm"), dep)
        pairs.append((i, [j for j in (1, 2, 3, 4, 5, 0) if j != i]))
    write_pair(os.path.join(d, "pair.txt"), pairs)

    # MiddEval3-style rectified pair (reference + right camera) with GT disparity and a non-occlusion mask
    s = os.path.join(root, "middeval3", "MiddEval3", "trainingH", "Synth")
    os.makedirs(s, exist_ok=True)
    (img0, d0), (img1, d1) = renders[0], renders[1]
    rp = G.RectifiedPair(S.K, cams[0], S.K, cams[1], (S.W, S.H))
    left, right = rp.rectify(img0, img1)
    cv2.imwrite(os.path.join(s, "im0.png"), left)
    cv2.imwrite(os.path.join(s, "im1.png"), right)
    w2, h2 = rp.size
    u, v = G.pixel_grid(h2, w2)
    ray = rp.R1.T @ (np.linalg.inv(rp.P1[:3, :3]) @ np.stack([u.ravel(), v.ravel(), np.ones(h2 * w2)]))
    px = S.K @ (ray / ray[2])
    zr = cv2.remap(d0, px[0].reshape(h2, w2).astype(np.float32), px[1].reshape(h2, w2).astype(np.float32),
                   cv2.INTER_NEAREST, borderValue=0)
    z = zr / ray[2].reshape(h2, w2)
    disp = np.where(z > 0, rp.f * rp.baseline / np.maximum(z, 1e-6), np.inf).astype(np.float32)
    write_pfm(os.path.join(s, "disp0GT.pfm"), disp)
    # occlusion mask from the GT: a left pixel is occluded if the right view sees something closer there
    lab = G.visibility_in_source(d0, S.K, cams[0], d1, S.K, cams[1])
    occ_ref = (lab == G.OCCLUDED).astype(np.uint8)
    occ_rect = cv2.remap(occ_ref, px[0].reshape(h2, w2).astype(np.float32), px[1].reshape(h2, w2).astype(np.float32),
                         cv2.INTER_NEAREST, borderValue=0)
    cv2.imwrite(os.path.join(s, "mask0nocc.png"), np.where(occ_rect > 0, 128, 255).astype(np.uint8))
    open(os.path.join(s, "calib.txt"), "w").write("ndisp=64\n")
    print(f"synthetic DTU scan1 (6 views) + MiddEval3 pair written to {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./data_synthetic")
