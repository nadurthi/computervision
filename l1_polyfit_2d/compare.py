"""
Fit the L1 polynomial model with every point set in points.pkl and compare visually.

    python compare.py                          # heavy-tailed banana, degree 6
    python compare.py --surface mixture --degree 5
    python compare.py --frames 36              # quicker, choppier animation

Writes to outputs/:
    contours_filled.png     truth + each fit (shared color levels), error maps, error bar chart
    contours_overlay.png    fitted contour lines drawn over the true contour lines
    surfaces_3d.png         static 3D view of truth + fits
    rotating_surfaces.gif/.mp4   truth + fits, rotating together (sample points at true height)
    rotating_errors.gif/.mp4     fit − truth error surfaces, rotating together
    fits.pkl                coefficients, exponents, chosen C and errors for each method
"""
import argparse
import os
import pickle
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import l1fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
COLORS = {"uniform_5x5": "#1f77b4", "gh_5x5": "#d62728", "cut8": "#2ca02c"}


def rel_rmse(Ff, Ft, mask):
    return 100 * np.sqrt(np.mean((Ff[mask] - Ft[mask]) ** 2)) / Ft.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", default="tbanana", choices=list(l1fit.SURFACES))
    ap.add_argument("--degree", type=int, default=6)
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = l1fit.SURFACES[args.surface]

    with open(os.path.join(HERE, "points.pkl"), "rb") as f:
        sets = pickle.load(f)

    # ---- fit every method --------------------------------------------------------------
    g = np.linspace(-1, 1, 161)
    GX, GY = np.meshgrid(g, g)
    G = np.c_[GX.ravel(), GY.ravel()]
    inside = np.all(np.abs(G) <= 0.85, axis=1)       # region covered by the samples
    Ft = np.exp(logf(G))
    fits = {}
    print(f"surface={args.surface}  degree={args.degree}")
    print(f"{'method':22s} {'N':>3s} {'C':>6s} {'nnz':>7s} {'relRMSE% (|x|<=.85)':>20s}")
    for key, s in sets.items():
        fit = l1fit.fit_l1(s["X"], logf, args.degree)
        Ff = l1fit.evaluate(fit, G)
        fit.update(label=f"{s['label']} ({len(s['X'])} pts)", Ff=Ff,
                   rmse=rel_rmse(Ff, Ft, inside), nnz=int(np.sum(np.abs(fit["lam"]) > 1e-6)))
        fits[key] = fit
        print(f"{fit['label']:22s} {len(s['X']):3d} {fit['C']:6g} "
              f"{fit['nnz']:3d}/{len(fit['lam']):<3d} {fit['rmse']:20.2f}")

    with open(os.path.join(OUT, "fits.pkl"), "wb") as f:
        pickle.dump({"surface": args.surface, "degree": args.degree,
                     "fits": {k: {kk: v[kk] for kk in ("label", "lam", "exps", "C", "rmse", "nnz")}
                              for k, v in fits.items()}}, f)

    sh = GX.shape
    T = Ft.reshape(sh)
    levels = np.linspace(0, T.max(), 15)
    elim = max(np.abs(v["Ff"] - Ft)[inside].max() for v in fits.values())
    elev = np.linspace(-elim, elim, 21)
    title = f"L1 polynomial fit (degree {args.degree}) of '{args.surface}' surface"

    # ---- 1. filled contours + errors + bar chart ----------------------------------------
    fig, ax = plt.subplots(2, 4, figsize=(19, 9.4), constrained_layout=True)
    ax[0, 0].contourf(GX, GY, T, levels, cmap="viridis")
    ax[0, 0].set_title("truth")
    for j, (key, v) in enumerate(fits.items(), start=1):
        a = ax[0, j]
        a.contourf(GX, GY, np.clip(v["Ff"].reshape(sh), 0, None), levels, cmap="viridis", extend="max")
        a.plot(v["X"][:, 0], v["X"][:, 1], "o", ms=5, mfc="w", mec="k")
        a.set_title(f"{v['label']}\nC={v['C']}, {v['nnz']}/{len(v['lam'])} nonzero coeffs")
        b = ax[1, j]
        im = b.contourf(GX, GY, (v["Ff"] - Ft).reshape(sh), elev, cmap="RdBu_r", extend="both")
        b.plot(v["X"][:, 0], v["X"][:, 1], "k.", ms=4)
        b.set_title(f"error (fit − truth)   rel. RMSE = {v['rmse']:.1f}%")
    fig.colorbar(im, ax=ax[1, 1:], shrink=0.8, label="fit − truth (same scale for all)")
    for a in list(ax[0]) + list(ax[1, 1:]):
        a.set_aspect("equal")
        a.add_patch(plt.Rectangle((-0.85, -0.85), 1.7, 1.7, fill=False, ls=":", ec="w" if a in ax[0] else "k"))
    bc = ax[1, 0]
    names = [v["label"].split(" (")[0] for v in fits.values()]
    vals = [v["rmse"] for v in fits.values()]
    bars = bc.bar(names, vals, color=[COLORS.get(k, "gray") for k in fits])
    bc.bar_label(bars, fmt="%.1f%%")
    bc.set_ylabel("relative RMSE inside sampled region (%)")
    bc.set_title("error summary")
    bc.tick_params(axis="x", rotation=15)
    fig.suptitle(title + "  —  dotted box = sampled region |x|,|y| ≤ 0.85", fontsize=14)
    fig.savefig(os.path.join(OUT, "contours_filled.png"), dpi=110)
    plt.close(fig)

    # ---- 2. contour-line overlay ---------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    lv = levels[1:-1:2]
    for a, (key, v) in zip(ax, fits.items()):
        a.contour(GX, GY, T, lv, colors="k", linestyles="--", linewidths=1.1)
        a.contour(GX, GY, v["Ff"].reshape(sh), lv, colors=COLORS.get(key, "C0"), linewidths=1.8)
        a.plot(v["X"][:, 0], v["X"][:, 1], "o", ms=5, mfc="w", mec="k")
        a.plot([], [], "k--", label="truth")
        a.plot([], [], color=COLORS.get(key, "C0"), label="fit")
        a.set_title(f"{v['label']}  —  RMSE {v['rmse']:.1f}%")
        a.set_aspect("equal")
        a.legend(loc="upper center")
    fig.suptitle(title + ": fitted vs true contour lines", fontsize=14)
    fig.savefig(os.path.join(OUT, "contours_overlay.png"), dpi=110)
    plt.close(fig)

    # ---- 3D: coarser grid for rendering ------------------------------------------------------
    g3 = np.linspace(-1, 1, 61)
    X3, Y3 = np.meshgrid(g3, g3)
    G3 = np.c_[X3.ravel(), Y3.ravel()]
    T3 = np.exp(logf(G3)).reshape(X3.shape)
    zmax = 1.1 * T3.max()

    def surf_panels(fig, rows=1):
        axs, arts = [], []
        a = fig.add_subplot(rows, 4, 1, projection="3d")
        a.plot_surface(X3, Y3, T3, cmap="viridis", lw=0, rstride=1, cstride=1, vmin=0, vmax=T3.max())
        a.set_title("truth", fontsize=12)
        axs.append(a)
        for j, (key, v) in enumerate(fits.items(), start=2):
            a = fig.add_subplot(rows, 4, j, projection="3d")
            Z = np.clip(l1fit.evaluate(v, G3).reshape(X3.shape), 0, zmax)
            a.plot_surface(X3, Y3, Z, cmap="viridis", lw=0, rstride=1, cstride=1, vmin=0, vmax=T3.max())
            a.scatter(v["X"][:, 0], v["X"][:, 1], np.exp(logf(v["X"])), c="r", s=14, depthshade=False)
            a.set_title(f"{v['label']}\nRMSE {v['rmse']:.1f}%", fontsize=12)
            axs.append(a)
        for a in axs:
            a.set_zlim(0, zmax)
            a.set_xlim(-1, 1)
            a.set_ylim(-1, 1)
            a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
        return axs

    def error_panels(fig):
        axs = []
        for j, (key, v) in enumerate(fits.items(), start=1):
            a = fig.add_subplot(1, 3, j, projection="3d")
            E = (l1fit.evaluate(v, G3).reshape(X3.shape) - T3)
            E = np.clip(E, -elim, elim)
            a.plot_surface(X3, Y3, E, cmap="RdBu_r", lw=0, rstride=1, cstride=1, vmin=-elim, vmax=elim)
            a.set_zlim(-elim, elim)
            a.set_title(f"error: {v['label']}\nRMSE {v['rmse']:.1f}%", fontsize=12)
            a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
            axs.append(a)
        return axs

    fig = plt.figure(figsize=(20, 6.2))
    for a in surf_panels(fig):
        a.view_init(elev=28, azim=-60)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.05)
    fig.savefig(os.path.join(OUT, "surfaces_3d.png"), dpi=110)
    plt.close(fig)

    # ---- 3. rotating animations ---------------------------------------------------------
    have_ffmpeg = shutil.which("ffmpeg") is not None
    azims = np.linspace(-60, 300, args.frames, endpoint=False)

    def animate(fig, axs, name):
        def upd(i):
            for a in axs:
                a.view_init(elev=28, azim=azims[i])
            return []
        anim = FuncAnimation(fig, upd, frames=len(azims), blit=False)
        anim.save(os.path.join(OUT, name + ".gif"), writer=PillowWriter(fps=15), dpi=72)
        if have_ffmpeg:
            anim.save(os.path.join(OUT, name + ".mp4"), writer=FFMpegWriter(fps=24, bitrate=2400), dpi=100)
        plt.close(fig)
        print("wrote", name + (".gif/.mp4" if have_ffmpeg else ".gif"))

    fig = plt.figure(figsize=(20, 6.2))
    axs = surf_panels(fig)
    fig.suptitle(title + "  (red dots = sample points at true height)", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.05)
    animate(fig, axs, "rotating_surfaces")

    fig = plt.figure(figsize=(16, 6.2))
    axs = error_panels(fig)
    fig.suptitle(title + ": error surfaces (same scale)", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.05)
    animate(fig, axs, "rotating_errors")


if __name__ == "__main__":
    main()
