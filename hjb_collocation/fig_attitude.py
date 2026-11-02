"""Example 3: 6-D attitude regulation — evaluate the CUT8 (and, if present, GH3) solutions."""
import numpy as np, pickle, os
import scipy.linalg as sl
from figs_common import plt, save, C_SPARSE, C_LEAST, C_INIT, C_OPT, C_TRUE, HERE
from hjbcoll import Dictionary, policy_from_coeffs, simulate, pmp_optimum, hjb_residual
from problems import attitude, ATT_BOX, I_SPIN

P = attitude(); dic = Dictionary(6, 8); Q = 40 * np.eye(6)
# LQR on the linearisation: ρ̇ = ½ω, ω̇ = I⁻¹u
Alin = np.zeros((6, 6)); Alin[:3, 3:] = 0.5 * np.eye(3)
Blin = np.zeros((6, 3)); Blin[3:, :] = np.diag(1 / I_SPIN)
Plqr = sl.solve_continuous_are(Alin, Blin, Q, np.eye(3))

runs = {}
for kind, lab, col in [("cut8", "CUT8 (745 + 284 points)", C_SPARSE), ("gh3", "Gauss–Hermite 3⁶ (729 + 284 points)", C_LEAST)]:
    f = os.path.join(HERE, "outputs", f"attitude_{kind}.pkl")
    if os.path.exists(f):
        d = pickle.load(open(f, "rb"))
        runs[kind] = dict(c=d["c"], hist=d["history"], times=d["times"], lab=lab, col=col,
                          u=policy_from_coeffs(P, dic, d["c"]))
        print(kind, len(d["history"]), "iterations,", int(np.sum(np.abs(d["c"]) > 1e-8)), "terms,",
              f"mean {np.mean(d['times']):.0f}s per iteration")

rng = np.random.default_rng(3)
tests = rng.uniform(-0.8, 0.8, (30, 6)) * ATT_BOX                 # random starts inside 80 % of the box
tests = np.r_[tests, [[1.5, 1.5, 1.5, 0.8, 0.8, 0.8]]]
J0 = np.array([simulate(P, P.u0, x, T=30)["cost"] for x in tests])
out = {"tests": tests, "J0": J0}
for k, r in runs.items():
    sims = [simulate(P, r["u"], x, T=30, max_norm=100) for x in tests]
    r["J"] = np.array([s["cost"] for s in sims]); r["div"] = np.array([s["diverged"] for s in sims])
    r["V"] = dic.phi(tests) @ r["c"]
    G = rng.uniform(-1, 1, (3000, 6)) * ATT_BOX
    res = hjb_residual(P, dic, r["c"], G)
    r["resid_rel"] = np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(P.l(G) ** 2))
    r["Vneg"] = np.mean(dic.phi(G) @ r["c"] < 0)
    print(f"{k}: diverged {r['div'].sum()}/{len(tests)}, J/J(u0) median {np.median(r['J'][~r['div']] / J0[~r['div']]):.3f}, "
          f"HJB residual rms / l rms {r['resid_rel']:.3f}, V<0 on {100*r['Vneg']:.1f}% of box")
    out[k] = {kk: r[kk] for kk in ("c", "J", "div", "V", "resid_rel", "Vneg", "times")}

# Pontryagin optimum at a few starts (warm-started from the CUT policy)
Jstar = np.full(len(tests), np.nan)
if "cut8" in runs:
    rc = runs["cut8"]
    for i in list(range(0, 30, 3)) + [30]:
        s = simulate(P, rc["u"], tests[i], T=30)
        if s["diverged"]:
            continue
        guess = lambda t, s=s: np.r_[np.array([np.interp(t, s["t"], s["x"][:, k]) for k in range(6)]),
                                     np.einsum("imk,m->ki", dic.jac(np.array([np.interp(t, s["t"], s["x"][:, k]) for k in range(6)]).T), rc["c"])]
        o = pmp_optimum(P, Q, tests[i], Plqr, guess=guess, T=15, nodes=200, tol=1e-4)
        if o is not None:
            Jstar[i] = o["cost"]
    ok = np.isfinite(Jstar)
    print("PMP solved at", ok.sum(), "starts; J(u0)/J* median", np.median(J0[ok] / Jstar[ok]).round(3),
          "; J(CUT policy)/J* median", np.median(rc["J"][ok] / Jstar[ok]).round(3), "max", np.max(rc["J"][ok] / Jstar[ok]).round(3))
out["Jstar"] = Jstar
pickle.dump(out, open(os.path.join(HERE, "outputs", "attitude_summary.pkl"), "wb"))

# ---------------- figure: flows in the ρ1–ρ2 plane ----------------
ncol = 1 + len(runs)
fig, ax = plt.subplots(1, ncol + 1, figsize=(5.2 * (ncol + 1), 5), constrained_layout=True)
grid = [[a, b] for a in np.linspace(-1.6, 1.6, 7) for b in np.linspace(-1.6, 1.6, 7)]
pols = [("initial policy u₀", P.u0, C_INIT)] + [(r["lab"], r["u"], r["col"]) for r in runs.values()]
for a, (lab, u, col) in zip(ax, pols):
    nd = 0
    for g in grid:
        x0 = np.array([g[0], g[1], 1.6, 0.8, 0.8, 0.8])
        s = simulate(P, u, x0, T=20, n_eval=400, max_norm=50)
        if s["diverged"]:
            a.plot(*x0[:2], "o", mfc="none", mec="r"); nd += 1
        else:
            a.plot(s["x"][:, 0], s["x"][:, 1], c=col, lw=1)
        a.plot(*x0[:2], "k.", ms=4)
    a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5); a.set_aspect("equal"); a.set_xlabel("ρ₁"); a.set_ylabel("ρ₂")
    a.set_title(f"{lab}" + (f"\n{nd} of {len(grid)} starts diverge (red)" if nd else ""))
a = ax[-1]
nrm = np.linalg.norm(tests / ATT_BOX, axis=1); o = np.argsort(nrm)
a.plot(nrm[o], J0[o], "o", c=C_INIT, ms=4, label="cost of u₀")
for r in runs.values():
    ok = ~r["div"]
    a.plot(nrm[o][ok[o]], r["J"][o][ok[o]], "x", c=r["col"], label=f"cost, {r['lab'].split(' (')[0]} policy")
    a.plot(nrm[o], r["V"][o], ".", c=r["col"], alpha=.5, label=f"V(x₀), {r['lab'].split(' (')[0]}")
if np.isfinite(Jstar).any():
    a.plot(nrm[np.isfinite(Jstar)], Jstar[np.isfinite(Jstar)], "k^", label="optimal J* (Pontryagin)")
a.set_yscale("symlog", linthresh=10); a.set_xlabel("‖x₀‖ (box-normalised)"); a.set_ylabel("cost-to-go"); a.legend(fontsize=8)
a.set_title("cost from 31 starts")
fig.suptitle("Example 3: attitude regulation (CRP + rates), starts at ρ₃ = 1.6, ω = 0.8; degree-8 dictionary (3003 monomials)", fontsize=12)
save(fig, "fig_att_flows.png")

# ---------------- figure: sample trajectory ----------------
if "cut8" in runs:
    x0 = np.array([1.5, 1.5, 1.5, 0.8, 0.8, 0.8])
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2), constrained_layout=True)
    for lab, u, col, ls in [("u₀", P.u0, C_INIT, "-"), ("CUT8 policy", runs["cut8"]["u"], C_SPARSE, "-")]:
        s = simulate(P, u, x0, T=10, n_eval=1000)
        for k in range(3):
            ax[0].plot(s["t"], s["x"][:, k], c=col, ls=["-", "--", ":"][k], label=f"ρ{k+1} {lab}")
            ax[1].plot(s["t"], s["x"][:, 3 + k], c=col, ls=["-", "--", ":"][k], label=f"ω{k+1} {lab}")
            ax[2].plot(s["t"], s["u"][:, k], c=col, ls=["-", "--", ":"][k], label=f"u{k+1} {lab}")
    for a, t in zip(ax, ["CRP ρ", "body rates ω", "control torque u"]):
        a.set_title(t + " from x₀ = [1.5, 1.5, 1.5, 0.8, 0.8, 0.8]"); a.set_xlabel("t [s]"); a.legend(fontsize=7.5, ncol=2)
    save(fig, "fig_att_traj.png")
