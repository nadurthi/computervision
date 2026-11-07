#!/usr/bin/env python3
"""Multi-view depth benchmark with an occlusion breakdown: one method per process.

For every reference view and every N in `nviews`, the method gets the reference plus its N-1 best source views
(the dataset's view selection, best first; source #1 is the "stereo partner"). The prediction is scored against GT
depth on a fixed evaluation grid, in regions defined from GT depth of the reference AND of every source view:

  all             GT available
  vis_partner     the point is visible in source #1
  occ_partner     the point is hidden in source #1 by a closer surface
  oov_partner     the point projects outside source #1
  rescued         not visible in source #1, but visible in at least one of the other N-2 sources
  unseen          visible in none of the N-1 sources (no method can match it; priors only)

Rows go to <out>/mvs/<method>.jsonl. Re-running skips (scene, ref, N) already present.

  python scripts/run_mvs_bench.py --config configs/quick.yaml --method patchmatchnet \
      --data-root /data --models-root /models --out results/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench import geometry as G                     # noqa: E402
from mvsbench.data.mvs import DATASETS                 # noqa: E402
from mvsbench.metrics import depth_metrics             # noqa: E402
from mvsbench.timing import Timed, gpu_name             # noqa: E402
from mvsbench.mvs import CLASSICAL, SCENE_LEVEL, build # noqa: E402

UNITS_TO_M = {"dtu": 1e-3, "eth3d": 1.0, "blendedmvs": 1.0}


def load_scenes(name, opts):
    kw = dict(work_wh=tuple(opts["work_wh"]), eval_wh=tuple(opts["eval_wh"]))
    if name == "dtu":
        return DATASETS[name](ROOT, scans=opts.get("scans"), **kw)
    return DATASETS[name](ROOT, scenes=opts.get("scenes"), **kw)


def pick_refs(scene, opts):
    ids = sorted(r for r, s in scene.pairs if r in scene.views and len(s) > 0)
    if "ref_stride" in opts:
        return ids[::opts["ref_stride"]]
    n = opts.get("refs_per_scene", 6)
    if len(ids) <= n:
        return ids
    return [ids[int(round(i))] for i in np.linspace(0, len(ids) - 1, n)]


class GTCache:
    """GT depth + intrinsics at the evaluation resolution."""

    def __init__(self, scene):
        self.s, self.c = scene, {}

    def __call__(self, vid):
        if vid not in self.c:
            v = self.s.views[vid]
            g = self.s.gt_depth(vid)
            ew, eh = self.s.eval_wh
            K = G.scale_intrinsics(v.K, ew / g.shape[1], eh / g.shape[0])
            self.c[vid] = (G.resize_depth(g, (ew, eh)), K, v.E)
        return self.c[vid]


def regions(gtc, ref, srcs):
    d0, K0, E0 = gtc(ref)
    labels = [G.visibility_in_source(d0, K0, E0, *gtc(s)) for s in srcs]
    valid = d0 > 0
    p = labels[0]
    others = np.zeros_like(valid)
    for lab in labels[1:]:
        others |= lab == G.VISIBLE
    any_vis = others | (p == G.VISIBLE)
    return d0, {
        "all": valid,
        "vis_partner": valid & (p == G.VISIBLE),
        "occ_partner": valid & (p == G.OCCLUDED),
        "oov_partner": valid & (p == G.OUT_OF_VIEW),
        "rescued": valid & (p != G.VISIBLE) & (p != G.UNKNOWN) & others,
        "unseen": valid & ~any_vis & np.all([lab != G.UNKNOWN for lab in labels], axis=0),
    }


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=None)
    a = ap.parse_args()
    ROOT = a.data_root
    global GPU
    GPU = gpu_name()
    cfg = yaml.safe_load(open(a.config))["mvs"]
    out_dir = os.path.join(a.out, "mvs")
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, f"{a.method}.jsonl")
    done = set()
    if os.path.exists(jl):
        for ln in open(jl):
            r = json.loads(ln)
            done.add((r["dataset"], r["scene"], r["ref"], r["nviews"]))
    fout = open(jl, "a")
    is_stereo = a.method.startswith("stereo_")
    is_classical = a.method in CLASSICAL
    models = {}
    for ds, opts in cfg["datasets"].items():
        if a.datasets and ds not in a.datasets:
            continue
        scenes = load_scenes(ds, opts)
        if not scenes:
            print(f"[{a.method}] {ds}: no data found, skipped", flush=True)
            continue
        # rmvd models work in metres and need the dataset units; every other method is unit-agnostic
        key = UNITS_TO_M[ds] if a.method in ("mvsanywhere", "vis_mvsnet", "robust_mvd") else 1.0
        if key not in models:
            models[key] = build(a.method, a.models_root, a.device, units_to_m=key)
        method = models[key]
        run_wh = tuple(opts["work_wh"] if (is_classical or is_stereo) else opts["learned_wh"])
        for scene in scenes:
            gtc = GTCache(scene)
            refs = pick_refs(scene, opts)
            for k_ref, ref in enumerate(refs):
                for n in cfg["nviews"]:
                    if is_stereo and n != 2:
                        continue
                    if (ds, scene.name, ref, n) in done:
                        continue
                    srcs = [s for s in scene.sources(ref, n - 1) if s in scene.views]
                    if len(srcs) < n - 1:
                        continue
                    ids = [ref] + srcs
                    imgs, Ks, Es = scene.load(ids, run_wh)
                    rng = scene.depth_range(ref)
                    try:
                        with Timed(a.device, external=a.method in SCENE_LEVEL) as tm:
                            if a.method in SCENE_LEVEL:
                                pred = method.run_scene(imgs, Ks, Es, [scene.depth_range(i) for i in ids],
                                                        [(0, list(range(1, n)))])[0]
                            else:
                                pred = method(imgs, Ks, Es, rng)
                    except Exception as e:
                        print(f"[{a.method}] {ds}/{scene.name} ref {ref} N={n} failed: {e}", flush=True)
                        if a.device.startswith("cuda"):
                            import torch
                            torch.cuda.empty_cache()
                        continue
                    dt = tm.seconds
                    gt, regs = regions(gtc, ref, srcs)
                    pred_e = G.resize_depth(pred, scene.eval_wh)
                    for reg, m in regs.items():
                        row = dict(dataset=ds, scene=scene.name, ref=int(ref), nviews=n, method=a.method,
                                   region=reg, time_s=dt, peak_mb=tm.peak_mb, gpu=GPU, run_wh=list(run_wh), units=scene.units,
                                   **depth_metrics(pred_e, gt, m, abs_thresholds=scene.abs_thresholds))
                        fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    save = cfg.get("save_depth", "qualitative")
                    if save == "all" or (save == "qualitative" and k_ref < 2):
                        p = os.path.join(out_dir, "depth", a.method, ds, scene.name, f"{ref:04d}_n{n}.npz")
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        np.savez_compressed(p, depth=pred_e.astype(np.float32))
                    print(f"[{a.method}] {ds}/{scene.name} ref {ref} N={n} {dt:.2f}s", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
