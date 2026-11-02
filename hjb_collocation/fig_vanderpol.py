"""Example 1: Van der Pol. Rerun of the paper's setup, the published V, least-degree baseline,
the Pontryagin optimum, and the basis-sensitivity study."""
import numpy as np, pickle, os, sys
import scipy.linalg as sl
from figs_common import plt, save, C_SPARSE, C_LEAST, C_INIT, C_OPT, C_TRUE, HERE
from hjbcoll import Dictionary, policy_iteration, policy_from_coeffs, simulate, pmp_optimum, collocation_rows, hjb_residual
from problems import vanderpol, vanderpol_points, PTS

P = vanderpol(); Q = 50 * np.eye(2)
Xc = vanderpol_points(); dic = Dictionary(2, 7); nm = dic.names()
Plqr = sl.solve_continuous_are(np.array([[0, 1], [-1, 1.]]), np.array([[0], [1.]]), Q, np.array([[5.]]))
PAPER = {'x1x2': 3.047, 'x2^2': 27.881, 'x1^2': 33.926, 'x1^2x2^2': -1.735, 'x1^3x2': 8.09, 'x1x2^3': -1.216, 'x2^4': -1.265,
         'x1^4': 4.349, 'x1^3x2^3': -0.043, 'x1^4x2^2': -0.208, 'x1^2x2^4': 0.049, 'x1^5x2': -0.731, 'x1x2^5': 0.106, 'x2^6': 0.076}
c_pub = np.zeros(dic.m)
for k, v in PAPER.items():
    c_pub[nm.index(k)] = v

# extra configuration found in the sensitivity study (set in run_vanderpol_extra.py output, optional)
EXTRA = os.path.join(HERE, "outputs", "vdp_extra.pkl")

g1 = np.linspace(-3, 3, 7)
tests = np.array([[a, b] for a in g1 for b in g1 if (a, b) != (0, 0)])         # 48 test starts
cache = os.path.join(HERE, "outputs", "vdp_Jstar.pkl")


def costs(u, X0, T=40):
    return np.array([simulate(P, u, x, T=T)["cost"] for x in X0])


# ---------------- solutions ----------------
res = policy_iteration(P, dic, Xc, iters=20, verbose=False)
c_rr = res.c
res_ld = policy_iteration(P, dic, Xc, iters=20, method="leastdeg", verbose=False)
c_ld = res_ld.c
u_pub = policy_from_coeffs(P, dic, c_pub); u_rr = policy_from_coeffs(P, dic, c_rr); u_ld = policy_from_coeffs(P, dic, c_ld)
sols = {"initial u₀": (P.u0, None), "published V (paper)": (u_pub, c_pub), "rerun, paper setup": (u_rr, c_rr),
        "least-degree basis": (u_ld, c_ld)}
extra = None
if os.path.exists(EXTRA):
    extra = pickle.load(open(EXTRA, "rb"))
    dic_x = Dictionary(2, extra["deg"])
    u_x = policy_from_coeffs(P, dic_x, extra["c"])
    sols[extra["label"]] = (u_x, None)

# ---------------- true optimum by Pontryagin (cached) ----------------
if os.path.exists(cache) and pickle.load(open(cache, "rb"))["tests"].shape == tests.shape:
    Jstar = pickle.load(open(cache, "rb"))["Jstar"]
else:
    Jstar = []
    for x0 in tests:
        s = simulate(P, u_pub, x0, T=40)

        def guess(t, s=s):
            Xs = np.array([np.interp(t, s["t"], s["x"][:, k]) for k in range(2)])
            return np.r_[Xs, np.einsum("imk,m->ki", dic.jac(Xs.T), c_pub)]
        o = pmp_optimum(P, Q, x0, Plqr, guess=guess, T=15)
        Jstar.append(np.nan if o is None else o["cost"])
    Jstar = np.array(Jstar)
    pickle.dump(dict(tests=tests, Jstar=Jstar), open(cache, "wb"))
print("PMP solved at", np.sum(np.isfinite(Jstar)), "of", len(tests), "starts")

J = {k: costs(u, tests) for k, (u, _) in sols.items()}
ratio = {k: J[k] / Jstar for k in J}
for k in J:
    r = ratio[k]; fin = np.isfinite(r) & (r < 1e6)
    print(f"{k:28s} J/J*: median {np.nanmedian(r):7.3f}  mean {np.nanmean(r[fin]):7.3f}  worst {np.nanmax(r):10.3g}  diverged {np.sum(~fin)}")

# ---------------- figure 1: rerun of the paper setup ----------------
gg = np.linspace(-3, 3, 61); GX, GY = np.meshgrid(gg, gg); GG = np.c_[GX.ravel(), GY.ravel()]
fig, ax = plt.subplots(2, 3, figsize=(17, 10.6), constrained_layout=True)
starts = np.array([[a, b] for a in np.linspace(-2.7, 2.7, 6) for b in np.linspace(-2.7, 2.7, 6)])
for a, (lab, u, col) in zip(ax[0], [("initial policy u₀ = −1.2(x₁+x₂)", P.u0, C_INIT), ("published V (paper)", u_pub, C_OPT),
                                     ("rerun with the paper's setup", u_rr, C_SPARSE)]):
    for x0 in starts:
        s = simulate(P, u, x0, T=15, n_eval=500)
        a.plot(*s["x"].T, c=col, lw=1)
        a.plot(*x0, "k.", ms=4)
    a.set_xlim(-3.5, 3.5); a.set_ylim(-4, 4); a.set_aspect("equal"); a.set_title(lab); a.set_xlabel("x₁"); a.set_ylabel("x₂")
for a, (lab, c) in zip(ax[1, :2], [("published V", c_pub), ("rerun V (paper setup)", c_rr)]):
    V = (dic.phi(GG) @ c).reshape(GX.shape)
    cs = a.contour(GX, GY, V, np.linspace(10, 900, 14), cmap="viridis")
    a.contour(GX, GY, V, [0], colors="r", linewidths=2)
    a.streamplot(GX, GY, GY, -GX + (1 - GX ** 2) * GY, color="#d1d5db", density=0.8, linewidth=0.7)
    a.plot(*Xc.T, "ko", ms=4)
    a.set_aspect("equal"); a.set_title(f"{lab}: level sets over the unforced flow\n(red = V = 0; black = 29 collocation points)")
a = ax[1, 2]
nrm = np.linalg.norm(tests, axis=1); o = np.argsort(nrm)
a.plot(nrm[o], J["initial u₀"][o], "o", c=C_INIT, ms=4, label="cost of u₀")
a.plot(nrm[o], Jstar[o], "k^", ms=5, label="optimal J* (Pontryagin)")
a.plot(nrm[o], J["published V (paper)"][o], "o", mfc="none", c=C_OPT, label="cost of published-V policy")
a.plot(nrm[o], J["rerun, paper setup"][o], "x", c=C_SPARSE, label="cost of rerun policy")
a.plot(nrm[o], (dic.phi(tests) @ c_rr)[o], ".", c=C_SPARSE, alpha=.5, label="rerun V(x₀)")
a.set_yscale("log"); a.set_xlabel("‖x₀‖"); a.set_ylabel("cost-to-go"); a.legend(fontsize=8.5); a.set_title("cost from 48 initial states")
fig.suptitle("Example 1: Van der Pol, Q = 50 I, R = 5, 29 collocation points, degree-7 dictionary", fontsize=13)
save(fig, "fig_vdp_main.png")

# ---------------- figure 2: closed-loop cost relative to optimum ----------------
fig, a = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
keys = list(J.keys())
cols = [C_INIT, C_OPT, C_SPARSE, C_LEAST, "#7c3aed"][:len(keys)]
data = [np.clip(np.where(np.isfinite(ratio[k]), ratio[k], 1e3), 0.9, 1e3) for k in keys]
bp = a.boxplot(data, vert=False, patch_artist=True, widths=.6, tick_labels=keys, showfliers=True)
for p, cc in zip(bp["boxes"], cols):
    p.set_facecolor(cc); p.set_alpha(.6)
a.set_xscale("log"); a.axvline(1, c="k", lw=1); a.set_xlabel("closed-loop cost / optimal cost J*  (48 initial states; diverged runs and values > 1000 shown at 1000)")
a.set_title("How close each feedback law gets to the true optimum")
save(fig, "fig_vdp_ratio.png")

# ---------------- figure 3: sample trajectory from [3, 3] ----------------
x0 = np.array([3.0, 3.0])
s_pub = simulate(P, u_pub, x0, T=40)
guess = lambda t: np.r_[np.array([np.interp(t, s_pub["t"], s_pub["x"][:, k]) for k in range(2)]),
                        np.einsum("imk,m->ki", dic.jac(np.array([np.interp(t, s_pub["t"], s_pub["x"][:, k]) for k in range(2)]).T), c_pub)]
opt = pmp_optimum(P, Q, x0, Plqr, guess=guess, T=15)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
for lab, (u, _), col in zip(keys, sols.values(), cols):
    s = simulate(P, u, x0, T=8, n_eval=800)
    ax[0].plot(s["t"], s["x"][:, 0], c=col, label=f"{lab} (J = {simulate(P, u, x0, T=40)['cost']:.0f})")
    ax[0].plot(s["t"], s["x"][:, 1], c=col, ls="--")
    ax[1].plot(s["t"], s["u"][:, 0], c=col)
ax[0].plot(opt["t"], opt["x"][:, 0], c=C_TRUE, lw=2.5, alpha=.5, label=f"open-loop optimum (J* = {opt['cost']:.0f})")
ax[0].plot(opt["t"], opt["x"][:, 1], c=C_TRUE, lw=2.5, alpha=.5, ls="--")
ax[1].plot(opt["t"], opt["u"][:, 0], c=C_TRUE, lw=2.5, alpha=.5)
ax[0].set_xlim(0, 8); ax[1].set_xlim(0, 8)
ax[0].set_title("states from x₀ = [3, 3] (solid x₁, dashed x₂)"); ax[0].legend(fontsize=8.5); ax[0].set_xlabel("t [s]")
ax[1].set_title("control u(t)"); ax[1].set_xlabel("t [s]")
save(fig, "fig_vdp_traj.png")

# ---------------- figure 4: which monomial gets dropped ----------------
even = [i for i in range(dic.m) if dic.deg[i] > 0 and dic.deg[i] % 2 == 0]
D, F = collocation_rows(P, dic, Xc, P.u0)
rk_all = np.linalg.matrix_rank(D, tol=1e-8 * np.abs(D).max()); rk_even = np.linalg.matrix_rank(D[:, even], tol=1e-8 * np.abs(D).max())
sub = tests[::3]; Js = Jstar[::3]
rows = []
for drop in even:
    sup = [i for i in even if i != drop]
    u = P.u0
    for _ in range(25):
        D, F = collocation_rows(P, dic, Xc, u)
        c = np.zeros(dic.m); c[sup] = np.linalg.lstsq(D[:, sup], F, rcond=None)[0]
        u = policy_from_coeffs(P, dic, c)
    r = costs(u, sub) / Js
    rows.append((nm[drop], np.nanmedian(r), np.nanmax(r)))
    print(f"drop {nm[drop]:9s}: median J/J* {np.nanmedian(r):8.3f}, worst {np.nanmax(r):10.3g}")
fig, a = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
names = [r[0] for r in rows]; med = np.array([r[1] for r in rows]); worst = np.array([r[2] for r in rows])
xx = np.arange(len(rows))
a.bar(xx - .2, np.clip(med, 0, 50), .4, color=C_SPARSE, label="median over 16 starts")
a.bar(xx + .2, np.clip(worst, 0, 50), .4, color="#93c5fd", label="worst start (clipped at 50)")
for i, n_ in enumerate(names):
    if n_ in ("x1^4", "x1^6"):
        a.annotate("rerun drops this" if n_ == "x1^4" else "paper drops this", (i, min(worst[i], 50)), textcoords="offset points",
                   xytext=(0, 6), ha="center", fontsize=9, color=C_LEAST)
a.axhline(1, c="k", lw=1); a.set_yscale("log")
a.set_xticks(xx); a.set_xticklabels(["drop " + n for n in names], rotation=45, ha="right")
a.set_ylabel("closed-loop cost / J*")
a.set_title(f"14 independent equations, 15 even monomials: the dropped term decides the controller\n(rank D = {rk_all} over all 36 columns, {rk_even} over the even ones)")
a.legend()
save(fig, "fig_vdp_sensitivity.png")

pickle.dump(dict(J=J, Jstar=Jstar, tests=tests, c_rr=c_rr, c_ld=c_ld, c_pub=c_pub, names=nm, sens=rows,
                 iters=len(res.history), iters_ld=len(res_ld.history)), open(os.path.join(HERE, "outputs", "vdp_summary.pkl"), "wb"))
print("rerun coefficients:", {nm[i]: round(c_rr[i], 3) for i in np.where(np.abs(c_rr) > 1e-6)[0]})
