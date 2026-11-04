#!/usr/bin/env python3
"""DTU point-cloud evaluation (accuracy / completeness / overall, mm) for one method per process.

Every view of each scan is estimated with its 4 best sources (N=5), all depth maps are fused with the same
geometric-consistency fusion (mvsbench.fusion), and the cloud is scored with DTUeval-python (a faithful port of
the official MATLAB evaluation). Rows go to <out>/pointcloud/<method>.jsonl.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench import geometry as G                  # noqa: E402
from mvsbench.data.mvs import dtu                   # noqa: E402
from mvsbench.fusion import fuse, write_ply         # noqa: E402
from mvsbench.io import read_image_rgb              # noqa: E402
from mvsbench.mvs import CLASSICAL, SCENE_LEVEL, build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nviews", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["mvs"]
    opts = cfg["datasets"]["dtu"]
    run_wh = tuple(opts["work_wh"] if a.method in CLASSICAL else opts["learned_wh"])
    ev_wh = tuple(opts["eval_wh"])
    out = os.path.join(a.out, "pointcloud")
    os.makedirs(out, exist_ok=True)
    jl = os.path.join(out, f"{a.method}.jsonl")
    done = {json.loads(l)["scene"] for l in open(jl)} if os.path.exists(jl) else set()
    method = build(a.method, a.models_root, a.device, units_to_m=1e-3)
    evaldir = os.path.join(a.data_root, "dtu", "dtueval")
    for scene in dtu(a.data_root, scans=opts.get("scans"), work_wh=run_wh, eval_wh=ev_wh):
        if scene.name in done:
            continue
        depths, Ks, Es, imgs = {}, {}, {}, {}
        ddir = os.path.join(out, "depth", a.method, scene.name)
        os.makedirs(ddir, exist_ok=True)
        if a.method in SCENE_LEVEL:                 # one ACMMP / COLMAP run over the whole scan, as they are meant to be used
            ids = sorted(scene.views)
            im, K, E = scene.load(ids, run_wh)
            pairs = [(ids.index(r), [ids.index(s) for s in scene.sources(r, a.nviews - 1)]) for r in ids]
            res = method.run_scene(im, K, E, [scene.depth_range(i) for i in ids], pairs)
            for k, r in enumerate(ids):
                depths[r] = G.resize_depth(res[k], ev_wh)
        for vid, v in scene.views.items():
            p = os.path.join(ddir, f"{vid:04d}.npz")
            if vid not in depths:
                if os.path.exists(p):
                    depths[vid] = np.load(p)["depth"]
                else:
                    ids = [vid] + scene.sources(vid, a.nviews - 1)
                    im, K, E = scene.load(ids, run_wh)
                    depths[vid] = G.resize_depth(method(im, K, E, scene.depth_range(vid)), ev_wh)
            np.savez_compressed(p, depth=depths[vid].astype(np.float32))
            img, K = G.resize_view(read_image_rgb(v.image_path), v.K, ev_wh)
            Ks[vid], Es[vid], imgs[vid] = K, v.E, img
        pts, cols = fuse(depths, Ks, Es, imgs, scene.pairs, min_views=3, pix_thresh=1.0, depth_thresh=0.01)
        ply = os.path.join(out, "ply", a.method, f"{scene.name}.ply")
        os.makedirs(os.path.dirname(ply), exist_ok=True)
        write_ply(ply, pts, cols)
        scan = int(scene.name[4:])
        vis = os.path.join(out, "vis", a.method, scene.name)
        os.makedirs(vis, exist_ok=True)
        r = subprocess.run([sys.executable, os.path.join(a.models_root, "DTUeval-python", "eval.py"), "--data", ply,
                            "--scan", str(scan), "--mode", "pcd", "--dataset_dir", evaldir, "--vis_out_dir", vis],
                           capture_output=True, text=True)
        try:
            acc, comp, overall = [float(x) for x in r.stdout.strip().splitlines()[-1].split()]
        except Exception:
            print(r.stdout[-2000:], r.stderr[-2000:])
            continue
        row = dict(dataset="dtu", scene=scene.name, method=a.method, nviews=a.nviews, points=int(len(pts)),
                   acc_mm=acc, comp_mm=comp, overall_mm=overall)
        with open(jl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{a.method}] {scene.name}: acc {acc:.3f} comp {comp:.3f} overall {overall:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
