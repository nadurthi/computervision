#!/usr/bin/env python3
"""Latency benchmark: how long each method takes to produce one depth (or disparity) map, one method per process.

Every method gets the same inputs at the resolution it is run at in the accuracy benchmark (configs: work_wh for
classical methods and stereo-in-MVS, learned_wh for learned MVS; native resolution for stereo datasets). For each
input: 1 untimed warm-up call (CUDA context, cuDNN autotuning, lazy init), then `--repeats` timed calls with CUDA
synchronisation, recording wall time and peak GPU memory. Model loading is timed separately.

  python scripts/run_latency.py --config configs/quick.yaml --method patchmatchnet --kind mvs \
      --data-root /data --models-root /models --out results/

Rows go to <out>/latency/<kind>_<method>.jsonl: one per (dataset, input, N) with median / p90 / min seconds,
megapixels, peak MB, GPU. ACMMP and COLMAP run as external binaries on a temporary workspace, so their time includes
writing the images and reading the result (that is how they are used); their memory is nvidia-smi's used memory.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench.timing import Timed, gpu_name   # noqa: E402

UNITS_TO_M = {"dtu": 1e-3, "eth3d": 1.0, "blendedmvs": 1.0}


def mvs_inputs(cfg, data_root, per_dataset):
    from mvsbench.data.mvs import DATASETS
    for ds, opts in cfg["datasets"].items():
        sel = opts.get("scans") if ds == "dtu" else opts.get("scenes")
        kw = dict(work_wh=tuple(opts["work_wh"]), eval_wh=tuple(opts["eval_wh"]))
        scenes = DATASETS[ds](data_root, sel[:1] if sel else None, **kw)
        if not scenes:
            continue
        sc = scenes[0]
        refs = sorted(r for r, s in sc.pairs if r in sc.views and len(s) >= max(cfg["nviews"]) - 1)[:per_dataset]
        yield ds, opts, sc, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--kind", choices=["mvs", "stereo"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inputs", type=int, default=2, help="inputs per dataset")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out_dir = os.path.join(a.out, "latency")
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, f"{a.kind}_{a.method}.jsonl")
    if os.path.exists(jl) and os.path.getsize(jl) > 0:
        print(f"[latency] {a.method}: already measured ({jl})")
        return
    rows, gpu = [], gpu_name()

    def record(ds, sample, n, mp, run_wh, times, mems, load_s):
        t = np.array(times)
        rows.append(dict(kind=a.kind, method=a.method, dataset=ds, sample=sample, nviews=n, megapixels=mp,
                         run_wh=list(run_wh), median_s=float(np.median(t)), p90_s=float(np.percentile(t, 90)),
                         min_s=float(t.min()), repeats=len(t),
                         peak_mb=float(max(m for m in mems if m is not None)) if any(m is not None for m in mems) else None,
                         load_s=load_s, gpu=gpu))
        print(f"[latency] {a.method} {ds} {sample} N={n}: median {np.median(t) * 1000:.0f} ms, "
              f"peak {rows[-1]['peak_mb']} MB", flush=True)

    if a.kind == "mvs":
        from mvsbench.mvs import CLASSICAL, SCENE_LEVEL, build
        is_lowres = a.method in CLASSICAL or a.method.startswith("stereo_")
        models = {}
        for ds, opts, sc, refs in mvs_inputs(cfg["mvs"], a.data_root, a.inputs):
            key = UNITS_TO_M[ds] if a.method in ("mvsanywhere", "vis_mvsnet", "robust_mvd") else 1.0
            load_s = None
            if key not in models:
                t0 = time.perf_counter()
                models[key] = build(a.method, a.models_root, a.device, units_to_m=key)
                load_s = time.perf_counter() - t0
            m = models[key]
            run_wh = tuple(opts["work_wh"] if is_lowres else opts["learned_wh"])
            for ref in refs:
                for n in (cfg["mvs"].get("method_overrides") or {}).get(a.method, {}).get("nviews", cfg["mvs"]["nviews"]):
                    if a.method.startswith("stereo_") and n != 2:
                        continue
                    ids = [ref] + sc.sources(ref, n - 1)
                    imgs, Ks, Es = sc.load(ids, run_wh)
                    rng = sc.depth_range(ref)

                    def call():
                        if a.method in SCENE_LEVEL:
                            return m.run_scene(imgs, Ks, Es, [sc.depth_range(i) for i in ids], [(0, list(range(1, n)))])
                        return m(imgs, Ks, Es, rng)
                    try:
                        call()                                   # warm-up
                        times, mems = [], []
                        for _ in range(a.repeats):
                            with Timed(a.device, external=a.method in SCENE_LEVEL) as t:
                                call()
                            times.append(t.seconds)
                            mems.append(t.peak_mb)
                    except Exception as e:
                        print(f"[latency] {a.method} {ds} ref {ref} N={n} failed: {e}", flush=True)
                        continue
                    record(ds, f"{sc.name}/{ref}", n, run_wh[0] * run_wh[1] / 1e6, run_wh, times, mems, load_s)
                    load_s = None
    else:
        from mvsbench.data.stereo import DATASETS
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_stereo_bench import build
        for ds, opts in cfg["stereo"]["datasets"].items():
            samples = list(DATASETS[ds](a.data_root, limit=a.inputs))
            if not samples:
                continue
            t0 = time.perf_counter()
            model = build(a.method, a.models_root, a.device, max(s["max_disp"] for s in samples))
            load_s = time.perf_counter() - t0
            for s in samples:
                try:
                    model(s["left"], s["right"], s["max_disp"])
                    times, mems = [], []
                    for _ in range(a.repeats):
                        with Timed(a.device) as t:
                            model(s["left"], s["right"], s["max_disp"])
                        times.append(t.seconds)
                        mems.append(t.peak_mb)
                except Exception as e:
                    print(f"[latency] {a.method} {s['name']} failed: {e}", flush=True)
                    continue
                h, w = s["left"].shape[:2]
                record(ds, s["name"], 2, w * h / 1e6, (w, h), times, mems, load_s)
                load_s = None
            del model
    with open(jl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
