#!/usr/bin/env python3
"""Two-view stereo benchmark: one method per process, every configured dataset, metrics per occlusion region.

  python scripts/run_stereo_bench.py --config configs/quick.yaml --method foundation_stereo \
      --data-root /data --models-root /models --out results/

Writes results/stereo/<method>.jsonl (one row per pair and region: all / noc / occ) and, for the first pairs of
each dataset, disparity maps for the figures. Re-running skips pairs already in the jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench.data.stereo import DATASETS            # noqa: E402
from mvsbench.metrics import disparity_metrics       # noqa: E402


def build(method, models_root, device, max_disp):
    if method.startswith("sgbm"):
        from mvsbench.stereo.sgbm import SGBM
        return SGBM(fill=method.endswith("fill"))
    from mvsbench.stereo.deep import build as b
    return b(method, models_root, device, max_disp=max_disp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-first", type=int, default=3, help="save disparity maps of the first N pairs per dataset")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))["stereo"]
    out_dir = os.path.join(a.out, "stereo")
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, f"{a.method}.jsonl")
    done = set()
    if os.path.exists(jl):
        for ln in open(jl):
            done.add(json.loads(ln)["sample"])
    f = open(jl, "a")
    for ds, opts in cfg["datasets"].items():
        samples = list(DATASETS[ds](a.data_root, limit=(opts or {}).get("limit")))
        if not samples:
            print(f"[{a.method}] {ds}: no data found, skipped", flush=True)
            continue
        max_disp = max(s["max_disp"] for s in samples)
        model = build(a.method, a.models_root, a.device, max_disp)
        for i, s in enumerate(samples):
            if s["name"] in done:
                continue
            if a.device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            t0 = time.time()
            try:
                disp = model(s["left"], s["right"], s["max_disp"])
            except Exception as e:                     # an OOM on one huge pair must not kill the whole run
                print(f"[{a.method}] {s['name']} failed: {e}", flush=True)
                if a.device.startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()
                continue
            if a.device.startswith("cuda"):
                import torch
                torch.cuda.synchronize()
            dt = time.time() - t0
            gt = s["gt"]
            regions = {"all": gt > 0, "noc": (gt > 0) & s["nocc"], "occ": (gt > 0) & ~s["nocc"]}
            for reg, m in regions.items():
                row = dict(sample=s["name"], dataset=ds, method=a.method, region=reg, time_s=dt,
                           hw=list(gt.shape), **disparity_metrics(disp, gt, m))
                f.write(json.dumps(row) + "\n")
            f.flush()
            if i < a.save_first:
                p = os.path.join(out_dir, "disp", a.method, s["name"].replace("/", "__") + ".npz")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez_compressed(p, disp=disp.astype(np.float16))
            print(f"[{a.method}] {s['name']} {dt:.2f}s", flush=True)
        del model
    f.close()


if __name__ == "__main__":
    main()
