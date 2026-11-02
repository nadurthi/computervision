"""Example 2: spin stabilisation — run both cases, check against the analytic solution, plot."""
import numpy as np, pickle, os
from figs_common import plt, save, C_SPARSE, C_INIT, C_OPT, HERE
from hjbcoll import Dictionary, policy_iteration, policy_from_coeffs, simulate, hjb_residual
from problems import spin, spin_points, I_SPIN

X = spin_points(); dic = Dictionary(3, 6); nm = dic.names("ω")
cases = {"Case 1 (high state cost): q = 50, r = 3, u₀ = −ω": (50, 3, 1.0),
         "Case 2 (high control cost): q = 5, r = 30, u₀ = −5ω": (5, 30, 5.0)}
rng = np.random.default_rng(1)
starts = np.r_[np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]], float) * 0.9,
               rng.uniform(-0.9, 0.9, (40, 3))]
summary = {}
fig, ax = plt.subplots(2, 4, figsize=(19, 8.6), constrained_layout=True)
for row, (title, (q, r, k0)) in enumerate(cases.items()):
    P = spin(q, r, k0)
    res = policy_iteration(P, dic, X, iters=20, verbose=False)
    c = res.c; u = policy_from_coeffs(P, dic, c)
    alpha = np.sqrt(q * r)
    exact = np.zeros(dic.m)
    for k, Ik in enumerate(I_SPIN):
        exact[nm.index(f"ω{k+1}^2")] = alpha * Ik
    G = rng.uniform(-1, 1, (4000, 3))
    resid = hjb_residual(P, dic, c, G)
    Tsim = 60 if q == 50 else 400
    sims0 = [simulate(P, P.u0, x0, T=Tsim, n_eval=2000) for x0 in starts]
    sims1 = [simulate(P, u, x0, T=Tsim, n_eval=2000) for x0 in starts]
    J0 = np.array([s["cost"] for s in sims0]); J1 = np.array([s["cost"] for s in sims1])
    V1 = dic.phi(starts) @ c
    summary[title] = dict(iters=len(res.history), nnz=int(np.sum(np.abs(c) > 1e-8)),
                          coef=c[np.abs(c) > 1e-8], coef_err=np.abs(c - exact).max(), resid_rms=np.sqrt(np.mean(resid ** 2)),
                          J0=J0, J1=J1, V1=V1, k0=k0, alpha=alpha)
    print(title, summary[title]["iters"], "iters,", summary[title]["nnz"], "terms, max|c - exact| =",
          f"{summary[title]['coef_err']:.2e}", " HJB residual rms", f"{summary[title]['resid_rms']:.2e}",
          " V/J(u*) max rel diff", f"{np.max(np.abs(V1 - J1) / J1):.2e}", " J(u*)/J(u0) mean", f"{np.mean(J1 / J0):.3f}")
    # trajectories (3D)
    for col, sims, lab, colr in [(0, sims0, "initial policy u₀", C_INIT), (1, sims1, "converged policy", C_SPARSE)]:
        a = ax[row, col]; a.remove(); a = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d"); ax[row, col] = a
        for s in sims[:8]:
            a.plot(*s["x"].T, c=colr, lw=1.4)
            a.scatter(*s["x"][0], c="k", s=10)
        a.set_title(f"{lab}"); a.set_xlabel("ω₁"); a.set_ylabel("ω₂"); a.set_zlabel("ω₃")
    # value vs norm
    a = ax[row, 2]
    nrm = np.linalg.norm(starts, axis=1)
    a.scatter(nrm, J0, c=C_INIT, s=18, label="cost of u₀ (simulated)")
    a.scatter(nrm, J1, s=40, marker="o", facecolors="none", edgecolors=C_OPT, label="cost of converged policy (simulated)")
    a.scatter(nrm, V1, c=C_SPARSE, s=10, marker="x", label="V(ω₀) from collocation")
    a.set_xlabel("‖ω₀‖"); a.set_ylabel("cost-to-go"); a.legend(fontsize=8.5); a.set_title(title.split(":")[0])
    # sample trajectory
    a = ax[row, 3]
    s0 = simulate(P, P.u0, np.ones(3), T=8 if q == 50 else 20); s1 = simulate(P, u, np.ones(3), T=8 if q == 50 else 20)
    for k, ls in zip(range(3), ["-", "--", ":"]):
        a.plot(s1["t"], s1["x"][:, k], c=C_SPARSE, ls=ls, label=f"ω{k+1} optimal")
        a.plot(s0["t"], s0["x"][:, k], c=C_INIT, ls=ls, label=f"ω{k+1} u₀")
    a2 = a.twinx()
    a2.plot(s1["t"], s1["u"][:, 0], c=C_OPT, lw=1, label="u₁ optimal"); a2.set_ylabel("u₁ (optimal)", color=C_OPT)
    a.set_xlabel("t [s]"); a.set_ylabel("ω"); a.set_title("from ω₀ = [1, 1, 1]"); a.legend(fontsize=8, ncol=2, loc="upper right")
fig.suptitle("Example 2: spin stabilisation, I = diag(14, 10, 8), CUT8 (59 points), 84-term degree-6 dictionary", fontsize=13)
save(fig, "fig_spin.png")

# coefficient bar for case 1
P = spin(50, 3, 1.0); c = policy_iteration(P, dic, X, iters=20, verbose=False).c
fig, a = plt.subplots(figsize=(15, 3.6), constrained_layout=True)
a.bar(range(dic.m), np.abs(c) + 1e-16, color=[C_SPARSE if abs(v) > 1e-8 else "#d1d5db" for v in c])
a.set_yscale("log"); a.set_ylim(1e-14, 1e3)
a.set_xticks(range(dic.m)); a.set_xticklabels(nm, rotation=90, fontsize=7)
for i in np.where(np.abs(c) > 1e-8)[0]:
    a.annotate(f"{c[i]:.3f}", (i, abs(c[i])), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=C_SPARSE)
a.set_ylabel("|cᵢ|"); a.set_title("Converged coefficients over the 84-monomial dictionary (Case 1): only ω₁², ω₂², ω₃² survive; analytic √(qr)·Iᵢ = 171.464, 122.474, 97.980")
save(fig, "fig_spin_coeffs.png")
pickle.dump(summary, open(os.path.join(HERE, "outputs", "spin_summary.pkl"), "wb"))
