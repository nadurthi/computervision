"""Figures for a localization run: map + trajectories, error vs time, particle spread."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lidarloc as ll

plt.rcParams.update({"savefig.dpi": 130, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def trajectory(recs, Xmap, out, title, s2m_marks=True):
    gt = np.array([r["gt"] for r in recs]); es = np.array([r["est"] for r in recs])
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    M = Xmap[(Xmap[:, 2] > np.median(Xmap[:, 2]) + 0.5)]
    pad = 40
    box = (M[:, 0] > gt[:, 0].min() - pad) & (M[:, 0] < gt[:, 0].max() + pad) & (M[:, 1] > gt[:, 1].min() - pad) & (M[:, 1] < gt[:, 1].max() + pad)
    M = M[box]
    if len(M) > 200000:
        M = M[np.random.default_rng(0).choice(len(M), 200000, replace=False)]
    ax.scatter(M[:, 0], M[:, 1], s=0.2, c="#9ca3af", lw=0)
    ax.plot(gt[:, 0], gt[:, 1], c="k", lw=2, label="ground truth")
    ax.plot(es[:, 0], es[:, 1], c="#dc2626", lw=1.4, label="particle filter estimate")
    ax.scatter(*es[0, :2], c="#dc2626", s=30, zorder=5); ax.scatter(*gt[0, :2], c="k", s=30, zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="best"); ax.set_title(title)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def errors(runs, out, title):
    """runs: list of (label, recs, color)."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    for lab, recs, col in runs:
        t = np.array([r["t"] for r in recs]); t = t - t[0]
        ax[0].semilogy(t, np.maximum([r["exy"] for r in recs], 1e-2), c=col, lw=1.5, label=lab)
        ax[1].semilogy(t, np.maximum([r["eyaw"] for r in recs], 1e-2), c=col, lw=1.5)
        ax[2].semilogy(t, np.maximum([r["spread"] for r in recs], 1e-2), c=col, lw=1.5)
    for a, yl in zip(ax, ["horizontal position error [m]", "yaw error [deg]", "particle spread (1σ) [m]"]):
        a.set_xlabel("t [s]"); a.set_ylabel(yl)
    ax[0].axhline(1.0, c="#9ca3af", ls=":", lw=1); ax[0].legend(fontsize=9)
    fig.suptitle(title)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
