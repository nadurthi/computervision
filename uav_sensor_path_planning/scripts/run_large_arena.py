"""Large-arena scenario: same two UAVs, much more ground to cover.

    python scripts/run_large_arena.py            # sweep + per-target reports + figures
    python scripts/run_large_arena.py videos     # only the videos (300 m, seed 0)

Arena sizes 100-400 m (the paper's is 100 m), 10 targets, 400 s, way-points
every 10 m, UAV speed 5 m/s, FOVs 20 m / 10 m. Each seed draws a new
target scenario (sensorplan.scenario.generate_truth, the original motion model).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sensorplan import SimConfig, run  # noqa: E402
from sensorplan import animate as A  # noqa: E402
from sensorplan.grid import NodeGrid  # noqa: E402
from sensorplan.scenario import generate_truth, report_summary, track_report  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results", "large_arena")
V = os.path.join(ROOT, "videos")
DUR = 400.0
NT = 10
ARENAS = [100, 200, 300, 400]
SEEDS = range(5)
NAMES = {"seq_time": "Seq-Time", "seq_uav": "Seq-UAV"}


def sim(L, method, lam, seed):
    truth, t = generate_truth(L, NT, DUR, seed=seed)
    cfg = SimConfig(method=method, lam=lam, seed=seed, uav_starts=[(0.7 * L, 0.7 * L), (0.3 * L, 0.3 * L)])
    return run(truth, t, cfg, grid=NodeGrid(xyf=(L + 1, L + 1)))


def sweep():
    rows, reps = [], []
    configs = [(L, m, 3.0) for L in ARENAS for m in ("seq_time", "seq_uav")] + \
              [(300, m, 10.0) for m in ("seq_time", "seq_uav")]
    for L, m, lam in configs:
        for seed in SEEDS:
            t0 = time.time()
            res = sim(L, m, lam, seed)
            rep = track_report(res)
            s = report_summary(rep)
            s.update(arena=L, method=m, lam=lam, seed=seed, wall_s=time.time() - t0)
            rows.append(s)
            reps.append(rep.assign(arena=L, method=m, lam=lam, seed=seed))
            print(f"{L:4d} m {m:9s} lam={lam:<4} seed={seed} found {s['found']}/10  "
                  f"tracked {s['tracked_pct']:5.1f}%  losses {s['losses']:2d}  "
                  f"visit gap {s['median_gap_between_visits_s']:5.1f}s ({s['wall_s']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(f"{R}/sweep_runs.csv", index=False)
    pd.concat(reps).to_csv(f"{R}/sweep_targets.csv", index_label="target")


INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
COL = {"seq_time": "#2a7de1", "seq_uav": "#e8772e"}


def figures():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK})
    df = pd.read_csv(f"{R}/sweep_runs.csv")
    d3 = df[df.lam == 3]
    panels = [("found", "targets found (of 10)", (0, 10.5)),
              ("tracked_pct", "% of time tracked", (0, 100)),
              ("losses", "track losses per run", None),
              ("median_gap_between_visits_s", "median gap between visits to a target [s]", None)]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (c, title, yl) in zip(axs, panels):
        for m in ("seq_time", "seq_uav"):
            g = d3[d3.method == m].groupby("arena")[c]
            mu, lo, hi = g.mean(), g.min(), g.max()
            off = -6 if m == "seq_time" else 6
            ax.errorbar(mu.index + off, mu.values, yerr=[mu - lo, hi - mu], fmt="o-", color=COL[m],
                        capsize=3, lw=2, label=NAMES[m] + " (λ = 3)")
            d10 = df[(df.lam == 10) & (df.method == m)][c]
            if len(d10):
                ax.scatter([300 + off], [d10.mean()], marker="D", s=40, facecolor="white", edgecolor=COL[m],
                           lw=1.8, zorder=5, label=NAMES[m] + " (λ = 10)")
        ax.set_xticks(ARENAS)
        ax.set_xticklabels([f"{L} m" for L in ARENAS])
        ax.set_xlabel("arena side")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if yl:
            ax.set_ylim(*yl)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{R}/fig-arena-sweep.png", dpi=150, facecolor="white")
    plt.close(fig)
    tab = d3.groupby(["arena", "method"])[["found", "tracked_pct", "tracked_after_pct", "losses", "reacquisitions",
                                           "median_first_detect_s", "visits", "median_gap_between_visits_s", "rmse_pos_m",
                                           "outside_3sigma_pct", "wall_s"]].mean().round(1)
    tab10 = df[df.lam == 10].groupby(["arena", "method"])[tab.columns].mean().round(1)
    tab.to_csv(f"{R}/sweep_summary_lam3.csv")
    tab10.to_csv(f"{R}/sweep_summary_lam10_300m.csv")
    print(tab.to_string())
    print(tab10.to_string())


def lifecycle_figure(res, rep, out, title):
    """Static per-target life-cycle chart (light theme) for the write-up."""
    st, t = res["status"], res["t"]
    dt = t[1] - t[0]
    Nt = st.shape[0]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for i in range(Nt):
        c = A.TARGET_C[i % len(A.TARGET_C)]
        for k in range(len(t)):
            if st[i, k] == 1:
                ax.add_patch(plt.Rectangle((t[k] - dt / 2, i - 0.32), dt, 0.64, fc=c, ec="none", alpha=0.85))
            elif st[i, k] == 2:
                ax.add_patch(plt.Rectangle((t[k] - dt / 2, i - 0.32), dt, 0.64, fc="#d1d5db", ec="none"))
    for (k, j, i, _) in res["meas"]:
        ax.plot([t[k]] * 2, [i - 0.42, i + 0.42], color=INK if j == 0 else "#7c3aed", lw=1.1)
    for i in range(Nt):
        for k in np.nonzero((st[i, :-1] == 1) & (st[i, 1:] == 2))[0]:
            ax.scatter([t[k + 1]], [i], marker="x", s=50, c="#d62839", lw=2, zorder=5)
        r = rep.iloc[i]
        txt = ("never found" if np.isnan(r.first_detect_s) else
               f"{int(r.measurements)} fixes · {int(r.losses)} lost · {r.tracked_pct:.0f}% tracked")
        ax.text(t[-1] + 6, i, txt, va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(t[0] - 2, t[-1] + 2)
    ax.set_ylim(Nt - 0.5, -0.5)
    ax.set_yticks(range(Nt))
    ax.set_yticklabels([f"T{i}" for i in range(Nt)])
    ax.set_xlabel("time [s]")
    ax.set_title(title, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, fc="#e8772e", alpha=0.85, label="tracked (1σ < 20 m)"),
          plt.Rectangle((0, 0), 1, 1, fc="#d1d5db", label="lost (frozen)"),
          plt.Line2D([], [], color=INK, lw=1.2, label="fix by UAV 0 (wide)"),
          plt.Line2D([], [], color="#7c3aed", lw=1.2, label="fix by UAV 1 (narrow)"),
          plt.Line2D([], [], marker="x", ls="", color="#d62839", mew=2, label="track lost")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=5, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def showcase():
    out = {}
    for m in ("seq_uav", "seq_time"):
        res = sim(300, m, 3.0, 0)
        rep = track_report(res)
        rep.to_csv(f"{R}/targets_300m_{m}_seed0.csv")
        lifecycle_figure(res, rep, f"{R}/fig-lifecycle-300m-{m}.png",
                         f"{NAMES[m]}, 300 m arena, λ = 3, seed 0: every target's track life-cycle")
        out[m] = (res, rep)
        print(m, report_summary(rep))
        print(rep.round(1).to_string())
    return out


def videos(out):
    os.makedirs(V, exist_ok=True)
    for m in ("seq_uav", "seq_time"):
        res, rep = out[m]
        s = report_summary(rep)
        A.render([(res, f"{NAMES[m]}  ·  λ = 3  ·  300 m × 300 m\n"
                        f"found {s['found']}/10 · tracked {s['tracked_pct']:.0f}% of the time · {s['losses']} track losses")],
                 f"{V}/sensorplan-300m-{m.replace('_', '-')}.mp4", substeps=3,
                 suptitle="Large arena: 2 UAVs, 10 targets, 9× the paper's area")


if __name__ == "__main__":
    os.makedirs(R, exist_ok=True)
    if sys.argv[1:] == ["videos"]:
        videos(showcase())
    else:
        sweep()
        figures()
        showcase()
