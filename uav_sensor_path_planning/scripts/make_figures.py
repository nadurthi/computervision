"""Figures for the write-up, from results/benchmark_runs.csv.

    python scripts/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")

INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
C_TIME, C_UAV = "#2a7de1", "#e8772e"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})

CONFIGS = [("seq_time", 0.5), ("seq_time", 3.0), ("seq_time", 10.0),
           ("seq_uav", 0.5), ("seq_uav", 3.0), ("seq_uav", 10.0)]
NAMES = {"seq_time": "Seq-Time", "seq_uav": "Seq-UAV"}

# 2021 saved runs, placed at the configuration each one is consistent with
# (see README: the 'Seq-Time lam=3' run executed with lam=0.5 hard-coded, and
# the folder named 'Time-MIlamda10' matches Seq-UAV lam=0.5, same missed target).
REF2021 = {("seq_uav", 3.0): "seq_uav_lam3", ("seq_time", 0.5): "seq_time_lam3",
           ("seq_uav", 0.5): "seq_time_lam10"}
# original code, re-run on 2026-11-03 on the same scenario (single unseeded run each)
RERUN = {("seq_uav", 3.0): dict(TIMES_Active_percent=82.63, TIMES_Active_percent_after_discovery=100.0,
                                RMSE_Active_pos=3.65, never_found=0),
         ("seq_time", 3.0): dict(TIMES_Active_percent=79.19, TIMES_Active_percent_after_discovery=94.59,
                                 RMSE_Active_pos=4.72, never_found=0),
         ("seq_time", 10.0): dict(TIMES_Active_percent=89.29, TIMES_Active_percent_after_discovery=100.0,
                                  RMSE_Active_pos=4.85, never_found=0)}


def ref2021():
    m = pd.read_csv(os.path.join(ROOT, "data", "original_2021_metrics.csv"))
    out = {}
    for cfg, run in REF2021.items():
        d = m[m.run == run]
        out[cfg] = dict(TIMES_Active_percent=d.TIMES_Active_percent.mean(),
                        TIMES_Active_percent_after_discovery=d.TIMES_Active_percent_after_discovery.mean(),
                        RMSE_Active_pos=d.RMSE_Active_pos.mean(),
                        never_found=int(d.RMSE_Active_pos.isna().sum()))
    return out


def benchmark_figure():
    df = pd.read_csv(os.path.join(R, "benchmark_runs.csv"))
    ref = ref2021()
    panels = [("TIMES_Active_percent", "% of time tracked", (0, 100)),
              ("TIMES_Active_percent_after_discovery", "% tracked after first detection", (60, 101)),
              ("RMSE_Active_pos", "position RMSE while tracked [m]", None),
              ("never_found", "targets never found (of 10)", (-0.3, 3.3))]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
    rng = np.random.default_rng(0)
    for ax, (col, title, ylim) in zip(axs, panels):
        for x, (m, lam) in enumerate(CONFIGS):
            v = df[(df.method == m) & (df.lam == lam)][col].values
            c = C_TIME if m == "seq_time" else C_UAV
            if col != "never_found":
                ax.boxplot([v], positions=[x], widths=0.55, showfliers=False,
                           medianprops=dict(color=c, lw=2), boxprops=dict(color=c),
                           whiskerprops=dict(color=c), capprops=dict(color=c))
            jx = x + rng.uniform(-0.15, 0.15, len(v))
            jy = v + (rng.uniform(-0.12, 0.12, len(v)) if col == "never_found" else 0)
            ax.scatter(jx, jy, s=10, color=c, alpha=0.55, lw=0)
            if (m, lam) in ref:
                ax.scatter([x], [ref[(m, lam)][col]], marker="D", s=46, color="#d62839", zorder=5,
                           label="2021 saved run" if x == 0 else None)
            if (m, lam) in RERUN:
                ax.scatter([x], [RERUN[(m, lam)][col]], marker="x", s=54, color=INK, lw=2, zorder=5)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([f"{NAMES[m]}\nλ={lam:g}" for m, lam in CONFIGS], fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)
    h = [plt.Line2D([], [], marker="o", ls="", color=C_TIME, label="port, 10 seeds (Seq-Time)"),
         plt.Line2D([], [], marker="o", ls="", color=C_UAV, label="port, 10 seeds (Seq-UAV)"),
         plt.Line2D([], [], marker="D", ls="", color="#d62839", label="2021 saved run (at the config it matches)"),
         plt.Line2D([], [], marker="x", ls="", color=INK, mew=2, label="original code, re-run today")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(R, "fig-benchmark.png"), dpi=150, facecolor="white")

    tab = df.groupby(["method", "lam"]).agg(
        tracked=("TIMES_Active_percent", "mean"), tracked_sd=("TIMES_Active_percent", "std"),
        after=("TIMES_Active_percent_after_discovery", "mean"),
        before=("TIME_percent_before_discover", "mean"),
        rmse=("RMSE_Active_pos", "mean"), rmse_sd=("RMSE_Active_pos", "std"),
        vel=("RMSE_Active_vel", "mean"), missed=("never_found", "mean"),
        wall=("wall_s", "mean")).round(2)
    tab.to_csv(os.path.join(R, "benchmark_summary.csv"))
    print(tab.to_string())


if __name__ == "__main__":
    benchmark_figure()
