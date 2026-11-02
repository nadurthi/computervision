"""
Extra analysis figures for the blog post (run after compare.py, or on its own).

    python blog_figures.py            # writes outputs/blog/*.png and prints the numbers used in the post
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import l1fit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "blog")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 130, "font.size": 11,
                     "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})
COL = {"uniform_5x5": "#1f77b4", "gh_5x5": "#d62728", "cut8": "#2ca02c"}
NAME = {"uniform_5x5": "Uniform 5×5", "gh_5x5": "Gauss–Hermite 5×5", "cut8": "CUT-8"}

with open(os.path.join(HERE, "points.pkl"), "rb") as f:
    SETS = pickle.load(f)
logf = l1fit.logf_tbanana
DEG = 6
g = np.linspace(-1, 1, 161)
GX, GY = np.meshgrid(g, g)
G = np.c_[GX.ravel(), GY.ravel()]
INSIDE = np.all(np.abs(G) <= 0.85, axis=1)
FT = np.exp(logf(G))


def rmse(Ff):
    return 100 * np.sqrt(np.mean((Ff[INSIDE] - FT[INSIDE]) ** 2)) / FT.max()


def gauss_moment(k):
    return 0.0 if k % 2 else float(np.prod(np.arange(k - 1, 0, -2))) if k else 1.0


FITS = {k: l1fit.fit_l1(s["X"], logf, DEG) for k, s in SETS.items()}


# ---------- 1. point sets in their own coordinates, weights as marker area -------------------
def fig_points():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for a, (k, s) in zip(ax, SETS.items()):
        X, w = s["raw"], s["w"]
        if k == "uniform_5x5":
            X = X * 2.0                     # show the grid on a comparable ±2 span
        a.scatter(X[:, 0], X[:, 1], s=2500 * w, c=COL[k], alpha=.75, ec="k")
        for r in (1, 2, 3):
            a.add_patch(Circle((0, 0), r, fill=False, ls=":", ec="gray"))
        if k == "cut8":
            for r in sorted(set(np.round(np.hypot(X[:, 0], X[:, 1]), 3)))[1:]:
                a.add_patch(Circle((0, 0), r, fill=False, ec=COL[k], lw=1))
                a.text(r * 0.95, -r * 0.31 - .1, f"r={r:.2f}", color=COL[k], fontsize=9, ha="left", bbox=dict(fc="w", ec="none", pad=1))
        a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_aspect("equal")
        a.set_title(f"{NAME[k]}: {len(X)} points" + ("  (shown on ±2)" if k == "uniform_5x5" else ""))
        a.axhline(0, c="gray", lw=.5); a.axvline(0, c="gray", lw=.5)
    fig.savefig(os.path.join(OUT, "fig_points_raw.png"))
    plt.close(fig)


# ---------- 2. the same sets after scaling to max|x| = 0.85, on top of the test surface -------
def fig_points_scaled():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    T = FT.reshape(GX.shape)
    for a, (k, s) in zip(ax, SETS.items()):
        a.contourf(GX, GY, T, 14, cmap="Greys", alpha=.55)
        X = s["X"]
        r = np.hypot(X[:, 0], X[:, 1])
        a.scatter(X[:, 0], X[:, 1], s=55, c=COL[k], ec="k", zorder=3)
        a.add_patch(Circle((0, 0), 0.3, fill=False, ec="k", ls="--"))
        a.plot(0, -0.35, "k*", ms=13)
        a.set_title(f"{NAME[k]}: {int((r < 0.3).sum())} points within r < 0.3")
        a.set_aspect("equal"); a.set_xlim(-1, 1); a.set_ylim(-1, 1)
    fig.savefig(os.path.join(OUT, "fig_points_scaled.png"))
    plt.close(fig)


# ---------- 3. moment exactness heat maps -----------------------------------------------------
def fig_moments():
    K = 13
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    for a, k in zip(ax, ["gh_5x5", "cut8"]):
        X, w = SETS[k]["raw"], SETS[k]["w"]
        E = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(K):
                ex = gauss_moment(i) * gauss_moment(j)
                q = np.sum(w * X[:, 0] ** i * X[:, 1] ** j)
                E[j, i] = abs(q - ex) / max(1.0, abs(ex))
        im = a.imshow(np.log10(E + 1e-16), origin="lower", cmap="magma_r", vmin=-16, vmax=0)
        a.plot([-.5, 9.5], [9.5, -.5], "c-", lw=2, label="total degree i+j = 9")
        a.add_patch(Rectangle((-.5, -.5), 10, 10, fill=False, ec="b", lw=1.5, ls="--", label="each degree ≤ 9"))
        a.set_xlabel("power of x  (i)"); a.set_ylabel("power of y  (j)")
        a.set_title(f"{NAME[k]} ({len(X)} pts): relative error of E[xⁱ yʲ]")
        a.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, label="log₁₀ relative error  (−16 = exact)")
    fig.savefig(os.path.join(OUT, "fig_moments.png"))
    plt.close(fig)


# ---------- 4. what the LP sees: equality / inequality / guard points -------------------------
def fig_constraints():
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    t = np.linspace(-2.3, 2.3, 201)
    BX, BY = np.meshgrid(t, t)
    L = logf(np.c_[BX.ravel(), BY.ravel()]).reshape(BX.shape)
    a = ax[0]
    cs = a.contourf(BX, BY, L, 20, cmap="viridis")
    fig.colorbar(cs, ax=a, label="log f")
    a.contour(BX, BY, L, [l1fit.CUTOFF], colors="r", linewidths=2)
    X = SETS["cut8"]["X"]
    a.scatter(X[:, 0], X[:, 1], c="w", ec="k", s=45, label="sample points → equality (log f ≥ −9)", zorder=3)
    Gd = l1fit.guard_points()
    a.scatter(Gd[:, 0], Gd[:, 1], marker="s", s=14, c="orange", ec="k", lw=.4, label="guard points → p(x) ≤ log N(x; 0, I)")
    a.add_patch(Rectangle((-1.5, -1.5), 3, 3, fill=False, ec="w", ls="--"))
    a.plot([], [], "r-", label="log f = −9 cutoff (points below → inequality)")
    a.set_aspect("equal"); a.legend(loc="lower left", fontsize=8.5, framealpha=.9)
    a.set_title("Constraint roles for CUT-8 on the test surface (log space)")
    # slice through x = 0
    a = ax[1]
    ys = np.linspace(-2.3, 2.3, 400)
    S = np.c_[np.zeros_like(ys), ys]
    a.plot(ys, logf(S), "k", lw=2.5, label="true log f(0, y)")
    a.plot(ys, -0.5 * ys ** 2 - np.log(2 * np.pi), "orange", ls="--", label="guard bound log N(0,I)")
    for k, fit in FITS.items():
        p = l1fit.design(S, fit["exps"]) @ fit["lam"]
        a.plot(ys, p, c=COL[k], lw=1.8, label=f"fitted p(0, y): {NAME[k]}")
    a.axhline(l1fit.CUTOFF, c="r", lw=1, ls=":")
    a.axvspan(-0.85, 0.85, color="gray", alpha=.12, label="sampled region")
    a.set_ylim(-14, 3); a.set_xlabel("y  (x = 0)"); a.set_ylabel("log value")
    a.legend(fontsize=8.5, loc="lower center")
    a.set_title("The fit happens in log space: slice x = 0, degree 6")
    fig.savefig(os.path.join(OUT, "fig_constraints_logspace.png"))
    plt.close(fig)


# ---------- 5. regularisation path over C ---------------------------------------------------------
def fig_cpath():
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    rows = {}
    for k, s in SETS.items():
        X = s["X"]
        exps = l1fit.basis_exponents(2, DEG)
        lf = logf(X); o = np.argsort(-lf); X, lf = X[o], lf[o]
        eq = lf >= l1fit.CUTOFF
        Aeq, beq = l1fit.design(X[eq], exps), lf[eq]
        Ain = l1fit.design(X[~eq], exps); bin_ = l1fit.CUTOFF * np.ones((~eq).sum())
        Xo = l1fit.guard_points(); Ao = l1fit.design(Xo, exps); bo = -0.5 * np.sum(Xo ** 2, 1) - np.log(2 * np.pi)
        kk = max(1, min(int(np.floor(0.75 * len(exps))), len(beq)))
        Cs, rm, nz, pk = [], [], [], []
        for C in l1fit.CS:
            lam, teq = l1fit.solve_lp(Aeq, beq, Ain, bin_, Ao, bo, C, kk)
            if lam is None:
                continue
            with np.errstate(over="ignore"):
                Ff = np.exp(l1fit.design(G, exps) @ lam)
            Cs.append(C); rm.append(rmse(Ff)); nz.append(int(np.sum(np.abs(lam) > 1e-6)))
            pk.append(100 * abs(teq[0]) / abs(beq[0]))
        rows[k] = (Cs, rm, nz, pk)
        ax[0].plot(Cs, rm, "o-", c=COL[k], label=NAME[k])
        ax[1].plot(Cs, nz, "o-", c=COL[k], label=NAME[k])
        ax[2].plot(Cs, np.maximum(pk, 1e-6), "o-", c=COL[k], label=NAME[k])
        ax[0].plot(FITS[k]["C"], rmse(l1fit.evaluate(FITS[k], G)), "*", ms=18, c=COL[k], mec="k")
    ax[2].axhline(0.01, c="k", ls="--", lw=1)
    ax[2].text(0.012, 0.013, "stop rule: 0.01 %", fontsize=9)
    for a, yl, t in zip(ax, ["rel. RMSE inside sampled region (%)", "non-zero coefficients (of 28)", "peak-point residual |teq₁| / |log f₁| (%)"],
                        ["error vs C (★ = C chosen by the stop rule)", "sparsity vs C", "peak residual vs C"]):
        a.set_xscale("log"); a.set_xlabel("C (weight on ‖λ‖₁)"); a.set_ylabel(yl); a.set_title(t); a.legend(fontsize=9)
    ax[2].set_yscale("log")
    fig.suptitle("Regularisation path: sweeping C in min C/n·‖λ‖₁ + 10·norm_largest(teq, k), degree 6")
    fig.savefig(os.path.join(OUT, "fig_cpath.png"))
    plt.close(fig)
    return rows


# ---------- 6. coefficients ---------------------------------------------------------------------
def fig_coeffs():
    exps = l1fit.basis_exponents(2, DEG)
    lab = []
    for e in exps:
        t = "".join([("x" + ("" if e[0] == 1 else f"{e[0]}")) if e[0] else "",
                     ("y" + ("" if e[1] == 1 else f"{e[1]}")) if e[1] else ""])
        lab.append(t or "1")
    fig, ax = plt.subplots(figsize=(15, 4.6), constrained_layout=True)
    w = 0.27
    for i, (k, fit) in enumerate(FITS.items()):
        ax.bar(np.arange(len(exps)) + (i - 1) * w, fit["lam"], w, color=COL[k], label=f"{NAME[k]} ({int(np.sum(np.abs(fit['lam'])>1e-6))} non-zero)")
    ax.set_xticks(range(len(exps))); ax.set_xticklabels(lab, rotation=60, fontsize=9)
    ax.axhline(0, c="k", lw=.6)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("λ (symlog scale)")
    ax.legend()
    ax.set_title("Fitted coefficients λ of p(x, y) = Σ λᵢ xᵃ yᵇ, degree 6 (28 monomials)")
    fig.savefig(os.path.join(OUT, "fig_coeffs.png"))
    plt.close(fig)


# ---------- 7. L1 LP vs plain least squares ----------------------------------------------------
def fig_l1_vs_ls():
    res = {}
    for k, s in SETS.items():
        exps = l1fit.basis_exponents(2, DEG)
        X = s["X"]; lf = logf(X)
        lam_ls, *_ = np.linalg.lstsq(l1fit.design(X, exps), lf, rcond=None)   # min-norm LS (no constraints)
        with np.errstate(over="ignore"):
            r_ls = rmse(np.exp(l1fit.design(G, exps) @ lam_ls))
        res[k] = (rmse(l1fit.evaluate(FITS[k], G)), r_ls)
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    x = np.arange(3)
    a1 = ax.bar(x - .2, [v[0] for v in res.values()], .4, label="L1 LP (this method)", color=[COL[k] for k in res])
    a2 = ax.bar(x + .2, [min(v[1], 400) for v in res.values()], .4, label="min-norm least squares, no constraints",
                color="lightgray", ec=[COL[k] for k in res], lw=2, hatch="//")
    ax.bar_label(a1, fmt="%.1f%%"); ax.bar_label(a2, labels=[f"{v[1]:.1f}%" for v in res.values()])
    ax.set_xticks(x); ax.set_xticklabels([NAME[k] for k in res])
    ax.set_ylabel("rel. RMSE inside sampled region (%)"); ax.set_yscale("log")
    ax.legend(); ax.set_title("Same points, same degree-6 basis: L1 LP vs least squares")
    fig.savefig(os.path.join(OUT, "fig_l1_vs_ls.png"))
    plt.close(fig)
    return res


# ---------- 8. degree sweep ---------------------------------------------------------------------
def fig_degree():
    degs = list(range(2, 11))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    out = {}
    for k, s in SETS.items():
        r = [rmse(l1fit.evaluate(l1fit.fit_l1(s["X"], logf, d), G)) for d in degs]
        out[k] = r
        ax.plot(degs, r, "o-", c=COL[k], label=f"{NAME[k]} ({len(s['X'])} pts)")
    nco = [len(l1fit.basis_exponents(2, d)) for d in degs]
    ax.set_xticks(degs); ax.set_xticklabels([f"{d}\n({n})" for d, n in zip(degs, nco)])
    ax.axvspan(5.5, 6.5, color="gold", alpha=.2)
    ax.set_xlabel("polynomial degree (number of coefficients)"); ax.set_ylabel("rel. RMSE inside sampled region (%)")
    ax.set_yscale("log"); ax.grid(alpha=.3, which="both"); ax.legend()
    ax.set_title("Error vs degree on the heavy-tailed banana (shaded: degree used in the main comparison)")
    fig.savefig(os.path.join(OUT, "fig_degree.png"))
    plt.close(fig)
    return degs, out


# ---------- 9. conditioning of the design matrix ----------------------------------------------
def fig_svd():
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    exps = l1fit.basis_exponents(2, DEG)
    out = {}
    for k, s in SETS.items():
        sv = np.linalg.svd(l1fit.design(s["X"], exps), compute_uv=False)
        out[k] = sv
        ax.semilogy(np.arange(1, len(sv) + 1), sv, "o-", c=COL[k], label=f"{NAME[k]}: rank {len(sv)} of 28, κ = {sv[0]/sv[-1]:.1e}")
    ax.set_xlabel("index"); ax.set_ylabel("singular value of A (points × monomials)")
    ax.set_title("Singular values of the degree-6 design matrix: every set is under-determined (28 unknowns)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_svd.png"))
    plt.close(fig)
    return out


# ---------- 10. polynomials the tensor grid cannot see -------------------------------------------
def fig_nullspace():
    X = SETS["gh_5x5"]["X"]
    nodes = np.unique(np.round(X[:, 0], 12))
    omega = lambda t: np.prod([t - n for n in nodes], axis=0)       # degree-5 node polynomial
    exps = l1fit.basis_exponents(2, DEG)
    A = l1fit.design(X, exps)
    _, sv, Vt = np.linalg.svd(A)
    nullity = len(exps) - int(np.sum(sv > 1e-10 * sv[0]))
    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    panels = [("ω(x) = x(x²−a²)(x²−b²)", omega(GX)), ("y·ω(x)", GY * omega(GX)), ("x·ω(y) + ω(x)", GX * omega(GY) + omega(GX))]
    for i, (t, Z) in enumerate(panels):
        a = fig.add_subplot(1, 3, i + 1)
        lim = np.abs(Z).max()
        a.contourf(GX, GY, Z, np.linspace(-lim, lim, 21), cmap="PuOr")
        a.contour(GX, GY, Z, [0], colors="k", linewidths=.8)
        a.scatter(X[:, 0], X[:, 1], c="r", s=40, ec="k", zorder=3)
        a.set_aspect("equal"); a.set_title(f"{t}: zero at all 25 nodes")
    fig.suptitle(f"Gauss–Hermite 5×5: {nullity} independent degree-≤6 polynomials vanish at every node (null space of the 25×28 design matrix). "
                 "The data cannot see them — and the grid can't pin p(x) between nodes.", fontsize=11.5)
    fig.savefig(os.path.join(OUT, "fig_nullspace.png"))
    plt.close(fig)
    return nullity


# ---------- 11. where the residual goes -------------------------------------------------------------
def fig_residuals():
    exps = l1fit.basis_exponents(2, DEG)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True, sharey=True)
    out = {}
    for a, (k, s) in zip(ax, SETS.items()):
        fit = FITS[k]
        Xs = fit["X"]; b = logf(Xs)
        r_l1 = l1fit.design(Xs, exps) @ fit["lam"] - b
        lam_ls = np.linalg.lstsq(l1fit.design(Xs, exps), b, rcond=None)[0]
        r_ls = l1fit.design(Xs, exps) @ lam_ls - b
        A = l1fit.design(Xs, exps); rk = np.linalg.matrix_rank(A, tol=1e-10 * np.linalg.norm(A, 2))
        idx = np.arange(len(b))
        a.bar(idx - .2, r_l1, .4, color=COL[k], label="L1 LP residual")
        a.bar(idx + .2, r_ls, .4, color="lightgray", ec="k", lw=.5, label="least-squares residual")
        a.axhline(0, c="k", lw=.6)
        a.set_xlabel("sample point, sorted by true value (0 = highest)")
        a.set_title(f"{NAME[k]}: rank {rk} for {len(b)} points")
        a.legend(fontsize=9)
        out[k] = (rk, r_l1[0], np.abs(r_ls).max())
    ax[0].set_ylabel("p(xᵢ) − log f(xᵢ)")
    fig.suptitle("Where the misfit goes (degree 6). Rank-deficient grids can't interpolate; the L1-type residual term dumps the whole misfit on the peak point.")
    fig.savefig(os.path.join(OUT, "fig_residuals.png"))
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("nullity:", fig_nullspace()); print("residuals:", fig_residuals())
    fig_points(); fig_points_scaled(); fig_moments(); fig_constraints()
    rows = fig_cpath(); fig_coeffs(); res = fig_l1_vs_ls(); degs, dout = fig_degree(); sv = fig_svd()
    print("main fits:", {k: (round(rmse(l1fit.evaluate(f, G)), 2), f["C"], int(np.sum(np.abs(f['lam']) > 1e-6))) for k, f in FITS.items()})
    print("C path:"); [print(" ", k, [(c, round(r, 1), n) for c, r, n in zip(*v[:3])]) for k, v in rows.items()]
    print("L1 vs LS:", {k: (round(a, 1), round(b, 1)) for k, (a, b) in res.items()})
    print("degree sweep", degs); [print(" ", k, [round(x, 1) for x in v]) for k, v in dout.items()]
    print("svd cond:", {k: f"{v[0]/v[-1]:.2e} n={len(v)}" for k, v in sv.items()})
    for k, s in SETS.items():
        r = np.hypot(*s["X"].T); print(k, "pts r<0.3:", int((r < .3).sum()), " nearest to peak:", round(float(np.min(np.hypot(s['X'][:,0], s['X'][:,1]+0.35))), 3))
