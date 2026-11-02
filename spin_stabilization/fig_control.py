"""Detumbling control: rate damping, the Debs–Athans optimal law, saturation, bang-bang, and B-dot."""
import numpy as np, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from spin import (I_DEFAULT, I_CUBESAT, simulate, energy, hmag, rate_damping, debs_athans,
                  saturated, bang_bang, bdot_controller)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
plt.rcParams.update({"savefig.dpi": 130, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
I = I_DEFAULT
W0 = np.array([0.6, -0.7, 0.8])
QC, RC = 50.0, 3.0                       # cost weights used to score every controller
C1, C2, C3 = "#2563eb", "#dc2626", "#16a34a"
stats = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig); print("wrote", name)


def cost(s, q=QC, r=RC):
    L = q * np.sum(s["w"] ** 2, 1) + r * np.sum(s["tau"] ** 2, 1)
    return cumulative_trapezoid(L, s["t"], initial=0)


# ---------------------------------------------------------------- 1. detumble comparison
kopt = np.sqrt(QC / RC)
cases = [("rate damping k = 1", rate_damping(1.0), "#9ca3af", "-"),
         (f"optimal (q={QC:g}, r={RC:g}): k* = {kopt:.2f}", debs_athans(QC, RC), "#111827", "-"),
         ("rate damping k = 12", rate_damping(12.0), "#f59e0b", "-"),
         ("saturated, |uᵢ| ≤ 1 N·m", saturated(kopt, 1.0), "#7c3aed", "--"),
         ("bang-bang, |uᵢ| = 1 N·m", bang_bang(1.0, 2e-3), "#db2777", ":")]
V0 = np.sqrt(QC * RC) * W0 @ (I * W0)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
for lab, ctl, col, ls in cases:
    s = simulate(I, W0, torque=ctl, T=20, n=2000, rtol=1e-8, atol=1e-10)
    J = cost(s)
    ax[0].semilogy(s["t"], np.linalg.norm(s["w"], axis=1), c=col, ls=ls, lw=1.8, label=lab)
    ax[1].plot(s["t"], np.linalg.norm(s["tau"], axis=1), c=col, ls=ls, lw=1.8)
    ax[2].plot(s["t"], J, c=col, ls=ls, lw=1.8)
    t10 = s["t"][np.argmax(np.linalg.norm(s["w"], axis=1) < 0.01 * np.linalg.norm(W0))]
    stats[lab] = dict(J=float(J[-1]), t_1pct=float(t10), umax=float(np.linalg.norm(s["tau"], axis=1).max()))
ax[2].axhline(V0, c="k", lw=.8, ls="--"); ax[2].text(19.5, V0 * 1.03, f"V(ω₀) = √(qr)·ω₀ᵀIω₀ = {V0:.0f}", ha="right", fontsize=9)
ax[0].set_ylim(1e-4, 3); ax[0].set_ylabel("‖ω‖ [rad/s]"); ax[0].legend(fontsize=8.5, loc="lower left")
ax[1].set_ylabel("‖u‖ [N·m]"); ax[2].set_ylabel(f"running cost ∫ q‖ω‖² + r‖u‖² dt")
ax[2].set_ylim(0, 2.2 * V0)
for a, ti in zip(ax, ["body rate (log scale)", "torque magnitude", "accumulated cost (q = 50, r = 3)"]):
    a.set_xlabel("t [s]"); a.set_title(ti)
save(fig, "fig_detumble.png")

# ---------------------------------------------------------------- 2. cost vs gain: the optimum is at k* = √(q/r)
ks = np.geomspace(0.3, 40, 40)
Js = []
for k in ks:
    s = simulate(I, W0, torque=rate_damping(k), T=60, n=4000, rtol=1e-8, atol=1e-10)
    Js.append(cost(s)[-1])
Js = np.array(Js)
# closed form: J(k) = (q + r k²)/(2k) · ω₀ᵀIω₀   (because d/dt(ωᵀIω) = −2k‖ω‖² and ‖u‖² = k²‖ω‖²)
Jth = (QC + RC * ks ** 2) / (2 * ks) * (W0 @ (I * W0))
fig, a = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
a.loglog(ks, Js, "o", c=C1, ms=5, label="simulated")
a.loglog(ks, Jth, c="k", lw=1.2, label="closed form  (q + r k²)/(2k) · ω₀ᵀIω₀")
a.axvline(kopt, c=C2, ls="--"); a.text(kopt * 1.08, Js.max() * 0.6, f"k* = √(q/r) = {kopt:.2f}\nJ* = V(ω₀) = {V0:.0f}", color=C2)
a.set_xlabel("rate-damping gain k"); a.set_ylabel("total cost J"); a.legend(); a.set_title("Every gain detumbles; one gain is optimal")
save(fig, "fig_cost_vs_gain.png")
stats["J_sim_min"] = float(Js.min()); stats["k_at_min"] = float(ks[np.argmin(Js)]); stats["V0"] = float(V0)

# ---------------------------------------------------------------- 3. value-function level sets + trajectories
fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
a = fig.add_subplot(1, 2, 1, projection="3d")
u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
for c, al in [(W0 @ (I * W0), .10), (0.25 * W0 @ (I * W0), .16), (0.04 * W0 @ (I * W0), .25)]:
    rad = np.sqrt(c / I)
    a.plot_wireframe(rad[0] * np.cos(u) * np.sin(v), rad[1] * np.sin(u) * np.sin(v), rad[2] * np.cos(v), color="#9ca3af", alpha=al + .15, lw=.5)
rng = np.random.default_rng(3)
starts = [W0] + [rng.normal(size=3) for _ in range(5)]
starts = [s0 / np.sqrt(s0 @ (I * s0)) * np.sqrt(W0 @ (I * W0)) for s0 in starts]
for j, s0 in enumerate(starts):
    s = simulate(I, s0, torque=debs_athans(5, 30), T=25, n=1500, rtol=1e-8, atol=1e-10)
    a.plot(*s["w"].T, lw=1.5, c=plt.cm.viridis(j / len(starts)))
    a.scatter(*s0, s=15, c="k")
a.set_xlabel("ω₁"); a.set_ylabel("ω₂"); a.set_zlabel("ω₃"); a.set_box_aspect((1, 1, 1)); a.view_init(20, 40)
a.set_title("Level sets of V = √(qr) ωᵀIω (ellipsoids) and closed-loop\ntrajectories, q = 5, r = 30 (slow gain so the spiral shows)")
a = fig.add_subplot(1, 2, 2)
s = simulate(I, W0, torque=debs_athans(5, 30), T=25, n=1500, rtol=1e-8, atol=1e-10)
T = energy(I, s["w"]); k = np.sqrt(5 / 30)
a.semilogy(s["t"], T, c=C1, lw=2, label="kinetic energy T = ½ωᵀIω")
a.semilogy(s["t"], T[0] * np.exp(-2 * k * s["t"] / I.max()), c="k", ls="--", label="bound T₀ exp(−2k t / I_max)")
a.semilogy(s["t"], T[0] * np.exp(-2 * k * s["t"] / I.min()), c="k", ls=":", label="bound T₀ exp(−2k t / I_min)")
a.set_xlabel("t [s]"); a.set_ylabel("T [J]"); a.legend(); a.set_title("Ṫ = −k‖ω‖²  squeezes T between two exponentials")
save(fig, "fig_value.png")

# ---------------------------------------------------------------- 4. B-dot on a 6U CubeSat
ctl = bdot_controller(I_CUBESAT, 0.2)
Torb = 2 * np.pi / ctl.n
w0 = np.deg2rad([10.0, -8.0, 9.0])
s = simulate(I_CUBESAT, w0, torque=ctl, T=4 * Torb, n=6000, rtol=1e-7, atol=1e-10)
m = np.array([ctl.dipole(t, w, q)[0] for t, w, q in zip(s["t"], s["w"], s["q"])])
to = s["t"] / Torb
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
ax[0].semilogy(to, np.rad2deg(np.linalg.norm(s["w"], axis=1)), c="k", lw=1.6, label="‖ω‖")
ax[0].axhline(np.rad2deg(2 * ctl.n), c=C2, ls="--", label="2n (B-dot equilibrium)")
ax[0].set_ylabel("‖ω‖ [deg/s]"); ax[0].legend(); ax[0].set_title("B-dot detumble, 6U CubeSat, 500 km SSO")
for j, col in enumerate([C1, C2, C3]):
    ax[1].plot(to, np.rad2deg(s["w"][:, j]), c=col, lw=1, label=f"ω{j+1}")
    ax[2].plot(to, m[:, j], c=col, lw=.8, label=f"m{j+1}")
ax[1].set_ylabel("ω [deg/s]"); ax[1].legend(); ax[1].set_title("body rates")
ax[2].set_ylabel("dipole [A·m²]"); ax[2].legend(loc="lower right"); ax[2].set_title("magnetorquer command (saturates at ±0.2)")
for a in ax:
    a.set_xlabel("orbits")
save(fig, "fig_bdot.png")
wn = np.rad2deg(np.linalg.norm(s["w"], axis=1))
stats["bdot"] = dict(k=float(ctl.k), n_rad_s=float(ctl.n), Torb_min=float(Torb / 60), w0_deg=float(wn[0]),
                     w_end_deg=float(wn[-1]), t_below_1deg_orbits=float(to[np.argmax(wn < 1.0)]),
                     two_n_deg=float(np.rad2deg(2 * ctl.n)))
json.dump(stats, open(os.path.join(OUT, "control_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
