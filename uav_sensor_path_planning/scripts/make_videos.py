"""Render the animation videos used on perceptionstack.dev.

    python scripts/make_videos.py            # all videos -> videos/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation, cm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from sensorplan import SimConfig, run, metrics, summary  # noqa: E402
from sensorplan import animate as A  # noqa: E402
from sensorplan.grid import NodeGrid, enumerate_trajectories  # noqa: E402
from sensorplan.models import CVTargetModel, CoverageMap  # noqa: E402
from sensorplan.planners import Belief, evaluate_paths  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "videos")
SEED = 0


def sim(method, lam):
    d = np.load(os.path.join(ROOT, "data", "scenario_2021.npz"))
    res = run(d["truth"], d["t"], SimConfig(method=method, lam=lam, seed=SEED))
    s = summary(metrics(res))
    return res, s


def label(name, lam, s):
    return (f"{name}  ·  λ = {lam:g}\n"
            f"tracked {s['TIMES_Active_percent']:.0f}% of the time · pos RMSE {s['RMSE_Active_pos']:.1f} m")


def explainer(res, k, out, lam=3.0, fps=25):
    """Exhaustive candidate scoring for Seq-UAV at replanning step k."""
    g = NodeGrid()
    tm, cmap = CVTargetModel(), CoverageMap(cells=g.nodes.copy())
    sensors = res["cfg"].sensors
    act = [i for i in range(res["truth"].shape[0])
           if res["status"][i, k] == 1 and np.sqrt(np.linalg.eigvalsh(res["est_P"][i, k][:2, :2]).max()) < 20]
    bel = Belief(x=res["est_x"][act, k], P=res["est_P"][act, k], p=res["coverage"][k])
    starts = [int(u) for u in res["uav"][:, k]]
    n = 5
    stage = []
    plans = [None, None]
    for j in range(2):
        cand = np.array(enumerate_trajectories(g, starts[j], n, 5, 2.0))
        paths = [None if plans[r] is None else np.repeat(plans[r][None], len(cand), 0) for r in range(2)]
        paths[j] = cand
        mt, mc = evaluate_paths(bel, paths, g, sensors, tm, cmap, 2.0, lam)
        J = mt + lam * mc
        b = int(np.argmax(J))
        plans[j] = cand[b]
        stage.append((cand, J, b, mt[b], mc[b]))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=A.BG)
    ax = fig.add_axes([0.02, 0.05, 0.53, 0.86])
    A._style(ax)
    ax.set_xlim(-8, 108)
    ax.set_ylim(-8, 108)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    expl = 1 - A._bin_h(res["coverage"][k]) / np.log(2)
    ax.imshow(expl.reshape(11, 11), origin="lower", extent=(-5, 105, -5, 105), cmap=A.EXPLORED, vmin=0, vmax=1)
    ax.scatter(g.nodes[:, 0], g.nodes[:, 1], s=2, c=A.GRIDC)
    # predicted (no-measurement) target uncertainty at the end of the horizon
    xs, Ps = bel.x, bel.P
    for _ in range(n - 1):
        xs, Ps = tm.predict(xs, Ps, 2.0)
    for q, i in enumerate(act):
        c = A.TARGET_C[i]
        ax.add_patch(A._ellipse(bel.x[q], bel.P[q], 1, fc=c, ec=c, alpha=0.25))
        ax.add_patch(A._ellipse(xs[q], Ps[q], 1, fc="none", ec=c, ls="--", lw=1))
        ax.scatter(*bel.x[q, :2], marker="s", s=14, c=c)
    ax.set_title(f"Seq-UAV planning at t = {res['t'][k]:.0f} s  ·  horizon 4 steps (8 s)  ·  λ = {lam:g}",
                 color=A.FG, fontsize=12, loc="left")
    for j in range(2):
        p = g.nodes[starts[j]]
        ax.add_patch(Circle(p, res["fov"][j], fc=A.UAV_C[j], alpha=0.10, lw=0))
        ax.scatter(*p, marker="^", s=110, c=A.UAV_C[j], edgecolor="white", zorder=10)
    txt = fig.text(0.60, 0.86, "", color=A.FG, fontsize=12, va="top", linespacing=1.6)
    eq = fig.text(0.60, 0.33,
                  r"$J=\sum_{k=1}^{4}\sum_i \frac{1}{2}\ln\frac{|P^i_{k|k-1}|}{|P^i_{k|k}|}"
                  r"\;+\;\lambda\sum_{k=1}^{4}\sum_c\left[H(p^c_{k|k-1})-H(p^c_{k|k})\right]$",
                  color=A.FG, fontsize=13)
    fig.text(0.60, 0.16, "filled ellipse: 1σ now      dashed: 1σ after 8 s with no measurement\n"
             "paths drawn slightly offset so overlapping ones stay visible\n"
             "path colour: score J of that candidate (bright = more information)",
             color=A.MUTED, fontsize=9, linespacing=1.6)
    del eq

    _r = np.random.default_rng(3)
    JIT = [_r.uniform(-1.6, 1.6, (len(s_[0]), 1, 2)) for s_ in stage]  # visual offset only
    lines = [[], []]
    nshow = [len(s[0]) for s in stage]
    seg = [int(3.5 * fps), int(2.0 * fps), int(3.5 * fps), int(3.0 * fps)]
    total = sum(seg)
    best_lines = []

    def draw_cands(j, upto):
        cand, J, b, _, _ = stage[j]
        order = np.argsort(J)
        norm = (J - J.min()) / max(J.max() - J.min(), 1e-9)
        for idx in order[len(lines[j]):upto]:
            pts = g.nodes[cand[idx]] + JIT[j][idx]
            ln, = ax.plot(pts[:, 0], pts[:, 1], "-", color=cm.plasma(0.15 + 0.85 * norm[idx]),
                          lw=0.7, alpha=0.55, zorder=7)
            lines[j].append(ln)

    def show_best(j):
        cand, J, b, mt, mc = stage[j]
        for ln in lines[j]:
            ln.set_alpha(0.12)
        pts = g.nodes[cand[b]]
        ln, = ax.plot(pts[:, 0], pts[:, 1], "-o", color=A.UAV_C[j], lw=3, ms=5, zorder=11)
        best_lines.append(ln)

    def update(f):
        if f < seg[0]:
            draw_cands(0, int(nshow[0] * (f + 1) / seg[0]))
            txt.set_text(f"UAV 0 (FOV 20 m, σ = 1 m)\nscoring {nshow[0]} candidate 4-step paths\n"
                         f"(UAV 1 not planned yet → takes no measurements)")
        elif f < sum(seg[:2]):
            if not best_lines:
                show_best(0)
            _, J, b, mt, mc = stage[0]
            txt.set_text(f"UAV 0 best path\nJ = {J[b]:.2f}  =  target MI {mt:.2f}  +  λ × coverage MI {mc:.2f}\n"
                         f"fixed for the rest of this plan")
        elif f < sum(seg[:3]):
            draw_cands(1, int(nshow[1] * (f - sum(seg[:2]) + 1) / seg[2]))
            txt.set_text(f"UAV 1 (FOV 10 m, σ = 0.2 m)\nscoring {nshow[1]} candidates,\n"
                         f"conditioned on UAV 0's path")
        else:
            if len(best_lines) < 2:
                show_best(1)
            _, J, b, mt, mc = stage[1]
            txt.set_text(f"UAV 1 best path\nJ = {J[b]:.2f}  =  target MI {mt:.2f}  +  λ × coverage MI {mc:.2f}\n"
                         f"{nshow[0] + nshow[1]} evaluations instead of {nshow[0] * nshow[1]:,} joint")
        return []

    anim = animation.FuncAnimation(fig, update, frames=range(total), blit=False)
    anim.save(out, writer=animation.FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "24",
                                              "-movflags", "+faststart"]), savefig_kwargs={"facecolor": A.BG})
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["uav", "time", "compare", "lambda", "explainer"]
    runs = {}

    def get(m, lam):
        if (m, lam) not in runs:
            runs[(m, lam)] = sim(m, lam)
        return runs[(m, lam)]

    if "uav" in which:
        r, s = get("seq_uav", 3)
        A.render([(r, label("Sequential in UAVs", 3, s))], f"{OUT}/sensorplan-seq-uav.mp4",
                 suptitle="Information-theoretic UAV routing: 2 UAVs, 10 moving targets")
    if "time" in which:
        r, s = get("seq_time", 3)
        A.render([(r, label("Sequential in time", 3, s))], f"{OUT}/sensorplan-seq-time.mp4",
                 suptitle="Information-theoretic UAV routing: 2 UAVs, 10 moving targets")
    if "compare" in which:
        (r1, s1), (r2, s2) = get("seq_uav", 3), get("seq_time", 3)
        A.render([(r1, label("Seq-UAV", 3, s1)), (r2, label("Seq-Time", 3, s2))],
                 f"{OUT}/sensorplan-uav-vs-time.mp4", suptitle="Same scenario, same noise seed: Seq-UAV vs Seq-Time")
    if "lambda" in which:
        (r1, s1), (r2, s2) = get("seq_time", 0.5), get("seq_time", 10)
        A.render([(r1, label("Seq-Time", 0.5, s1)), (r2, label("Seq-Time", 10, s2))],
                 f"{OUT}/sensorplan-lambda.mp4", suptitle="Exploit vs explore: the coverage weight λ")
    if "explainer" in which:
        r, _ = get("seq_uav", 3)
        explainer(r, 33, f"{OUT}/sensorplan-candidates.mp4")
    for k, (r, s) in runs.items():
        print(k, {a: round(b, 2) for a, b in s.items()})


if __name__ == "__main__":
    main()
