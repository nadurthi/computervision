"""Figures: CUT point structure, point counts, and the two interpolation examples."""
import numpy as np
from math import comb
from figs_common import plt, save, C_SPARSE, C_LEAST, C_TRUE
from problems import PTS
from interp_examples import run as run_interp

# ---------- point sets ------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 4.8), constrained_layout=True)
a = fig.add_subplot(1, 3, 1)
U = PTS["cut8_2D_uniform"]["X"]; w = PTS["cut8_2D_uniform"]["w"]
B = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [0, 1], [0, -1], [-1, 0]], float)
a.scatter(*(3 * U).T, s=1800 * w, c=C_SPARSE, ec="k", label="CUT8 (21), area ∝ weight")
a.scatter(*(3 * B).T, s=50, marker="s", c="orange", ec="k", label="boundary (8)")
a.add_patch(plt.Rectangle((-3, -3), 6, 6, fill=False, ls=":", ec="gray"))
a.set_aspect("equal"); a.set_xlim(-3.6, 3.6); a.set_ylim(-3.6, 3.6)
a.set_title("Van der Pol collocation set: 29 points in [−3, 3]²"); a.legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.08), ncol=2)
a.set_xlabel("x₁"); a.set_ylabel("x₂")

a = fig.add_subplot(1, 3, 2, projection="3d")
X3 = PTS["cut8_3D_uniform"]["X"]; w3 = PTS["cut8_3D_uniform"]["w"]
r = np.round(np.abs(X3), 3)
a.scatter(*X3.T, s=900 * w3 + 8, c=np.linalg.norm(X3, axis=1), cmap="viridis", ec="k", lw=.4)
a.set_title("Spin stabilisation: CUT8 in [−1, 1]³ (59 points)")
a.set_xlabel("ω₁"); a.set_ylabel("ω₂"); a.set_zlabel("ω₃")

a = fig.add_subplot(1, 3, 3)
ns = np.arange(2, 7)
cut = [len(PTS[f"cut8_{n}D_gaussian"]["X"]) if f"cut8_{n}D_gaussian" in PTS else len(PTS["cut8_6D_gaussian"]["X"]) for n in ns]
gh = [5 ** n for n in ns]
dic = [comb(n + 8, 8) for n in ns]
a.semilogy(ns, gh, "o-", c=C_LEAST, label="Gauss–Hermite 5ⁿ (same 9th-order accuracy)")
a.semilogy(ns, cut, "o-", c=C_SPARSE, label="CUT8 (9th order)")
a.semilogy(ns, dic, "s--", c="gray", label="degree-8 monomials C(n+8, 8)")
for n, cval, g in zip(ns, cut, gh):
    a.annotate(f"{cval}", (n, cval), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=C_SPARSE)
    a.annotate(f"{g:,}", (n, g), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, color=C_LEAST)
a.set_xlabel("state dimension n"); a.set_ylabel("count"); a.set_xticks(ns)
a.set_title("Collocation points vs dictionary size"); a.legend(fontsize=9)
save(fig, "fig_points.png")

# ---------- interpolation examples --------------------------------------------------------------
out = run_interp(verbose=False)
fig, ax = plt.subplots(2, 3, figsize=(16, 9.4), constrained_layout=True)
for row, (key, title) in enumerate([(("ex1",), "Example 1: degree-6 polynomial, 16 GL points"),
                                    (("ex2", 4), "Example 2: non-polynomial, 16 GL points")]):
    s = out[key + ("sparse",)]; l = out[key + ("leastdeg",)]
    GX, GY = s["grid"]; T = s["true"]
    lv = np.linspace(T.min(), T.max(), 16)
    ax[row, 0].contourf(GX, GY, T, lv, cmap="viridis"); ax[row, 0].plot(*s["X"].T, "wo", mec="k")
    ax[row, 0].set_title(title + "\ntrue function")
    E1 = np.abs(s["fit"] - T); E2 = np.abs(l["fit"] - T); vmax = max(E1.max(), E2.max())
    for a, E, lab, val in [(ax[row, 1], E1, "sparse ℓ1", None), (ax[row, 2], E2, "least degree", None)]:
        im = a.contourf(GX, GY, E, np.linspace(0, vmax, 16), cmap="magma")
        a.plot(*s["X"].T, "wo", mec="k", ms=4)
        if key[0] == "ex1":
            a.set_title(f"|error|, {lab}: max {E.max():.2e}")
        else:
            a.set_title(f"|error|, {lab}: RMSE {np.sqrt(np.mean((E)**2)):.0f}")
    fig.colorbar(im, ax=ax[row, 1:], shrink=.8, label="|fit − true| (shared scale)")
    for a in ax[row]:
        a.set_aspect("equal")
save(fig, "fig_interp.png")
for k, v in out.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else None) for kk, vv in v.items() if kk in ("maxerr", "rmse")})
