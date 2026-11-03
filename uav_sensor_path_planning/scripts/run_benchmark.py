"""Monte-Carlo benchmark of Seq-UAV vs Seq-Time on the paper's 2021 scenario.

    python scripts/run_benchmark.py --seeds 10

Writes results/benchmark_runs.csv (one row per run), results/benchmark_targets.csv
(per-target metrics) and box plots mirroring the paper's figures.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sensorplan import SimConfig, metrics, run, summary  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 3, 10])
    ap.add_argument("--methods", nargs="+", default=["seq_time", "seq_uav"])
    a = ap.parse_args()

    d = np.load(os.path.join(ROOT, "data", "scenario_2021.npz"))
    truth, t = d["truth"], d["t"]
    runs, targs = [], []
    for method in a.methods:
        for lam in a.lams:
            for seed in range(a.seeds):
                t0 = time.time()
                res = run(truth, t, SimConfig(method=method, lam=lam, seed=seed))
                m = metrics(res)
                s = summary(m)
                s.update(method=method, lam=lam, seed=seed, wall_s=time.time() - t0)
                runs.append(s)
                targs.append(m.assign(method=method, lam=lam, seed=seed))
                print(f"{method:9s} lam={lam:<4} seed={seed}  "
                      f"tracked={s['TIMES_Active_percent']:5.1f}%  rmse={s['RMSE_Active_pos']:.2f}  "
                      f"missed={s['never_found']}  ({s['wall_s']:.1f}s)", flush=True)
                pd.DataFrame(runs).to_csv(os.path.join(ROOT, "results", "benchmark_runs.csv"), index=False)
    pd.concat(targs).to_csv(os.path.join(ROOT, "results", "benchmark_targets.csv"), index_label="target")


if __name__ == "__main__":
    main()
