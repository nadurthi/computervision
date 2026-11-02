"""Torque-free rigid body: polhodes, the intermediate-axis flip, stability, energy dissipation."""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spin import I_DEFAULT, simulate, energy, hmag, internal_damping

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
plt.rcParams.update({"savefig.dpi": 130, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
I = I_DEFAULT
C1, C2, C3 = "#2563eb", "#dc2626", "#16a34a"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- 1. polhodes on the momentum sphere
Hm = 10.0
fig = plt.figure(figsize=(15, 6.4), constrained_layout=True)
ax = fig.add_subplot(1, 2, 1, projection="3d")
u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
ax.plot_surface(Hm * np.cos(u) * np.sin(v), Hm * np.sin(u) * np.sin(v), Hm * np.cos(v), color="#e5e7eb", alpha=.25, lw=0)
Tmin, Tmax, Tsep = Hm ** 2 / (2 * I[0]), Hm ** 2 / (2 * I[2]), Hm ** 2 / (2 * I[1])
rng = np.linspace(0, 1, 11)[1:-1]
for frac in rng:
    for axis_major in (True, False):
        # start points on the sphere with a chosen energy: near the major (x) or minor (z) axis
        if axis_major:
            T0 = Tmin + frac * (Tsep - Tmin)
        else:
            T0 = Tsep + frac * (Tmax - Tsep)
        # find H on sphere with 2T = H1²/I1 + H2²/I2 + H3²/I3, starting from the x-z great circle
        th = np.linspace(0, np.pi / 2, 4000)
        H1, H3 = Hm * np.cos(th), Hm * np.sin(th)
        e = 0.5 * (H1 ** 2 / I[0] + H3 ** 2 / I[2]) - T0
        j = np.argmin(np.abs(e))
        w0 = np.array([H1[j], 0, H3[j]]) / I + np.array([0, 1e-9, 0])
        for sgn in (1, -1):
            s = simulate(I, sgn * w0, T=40, n=1500)
            Hb = I * s["w"]
            col = C1 if axis_major else C3
            ax.plot(*Hb.T, c=col, lw=1.1)
# separatrix
s = simulate(I, np.array([1e-4, Hm / I[1], 1e-4]), T=60, n=3000)
ax.plot(*(I * s["w"]).T, c=C2, lw=2.2)
for k, (lab, col) in enumerate([("major I₁", C1), ("intermediate I₂", C2), ("minor I₃", C3)]):
    e = np.zeros(3); e[k] = Hm * 1.25
    ax.plot(*np.c_[-e, e], c=col, lw=1, ls="--"); ax.text(*e, lab, color=col, fontsize=10)
ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(22, 35)
ax.set_title("Polhodes: body-frame H on the sphere ‖H‖ = const\n(blue: circulate the major axis, green: the minor axis, red: separatrix)")

a = fig.add_subplot(1, 2, 2)
from matplotlib.patches import Ellipse, Circle
a.add_patch(Circle((0, 0), Hm, fill=False, lw=2.2, ec="k"))
for T0, col, lab in [(Tmin + 0.25 * (Tsep - Tmin), C1, "low T: circulates the major axis"),
                     (Tsep, C2, "T = H²/2I₂: separatrix"),
                     (Tsep + 0.6 * (Tmax - Tsep), C3, "high T: circulates the minor axis")]:
    a.add_patch(Ellipse((0, 0), 2 * np.sqrt(2 * T0 * I[0]), 2 * np.sqrt(2 * T0 * I[2]), fill=False, lw=1.8, ec=col, label=lab))
a.set_xlim(-1.45 * Hm, 1.45 * Hm); a.set_ylim(-1.3 * Hm, 1.3 * Hm); a.set_aspect("equal")
a.axhline(0, c="#d1d5db", lw=.6); a.axvline(0, c="#d1d5db", lw=.6)
a.set_xlabel("H₁ (major axis)"); a.set_ylabel("H₃ (minor axis)")
a.legend(loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.32), ncol=1)
a.set_title("Momentum sphere (black) and energy ellipsoids, cut in the H₁–H₃ plane.\nThe motion lives where they intersect; semi-axes √(2T·Iᵢ)")
save(fig, "fig_polhodes.png")

# ---------------------------------------------------------------- 2. spin about each principal axis
fig, ax = plt.subplots(1, 3, figsize=(17, 4.4), constrained_layout=True, sharey=True)
eps = 0.02
for a, k, lab, col in zip(ax, range(3), ["major axis I₁ = 14", "intermediate axis I₂ = 10", "minor axis I₃ = 8"], [C1, C2, C3]):
    w0 = np.full(3, eps); w0[k] = 1.0
    s = simulate(I, w0, T=80, n=4000)
    for j, ls in zip(range(3), ["-", "--", ":"]):
        a.plot(s["t"], s["w"][:, j], ls=ls, c=col if j == k else "gray", lw=1.6 if j == k else 1, label=f"ω{j+1}")
    a.set_title(f"spin about the {lab}"); a.set_xlabel("t [s]"); a.legend(loc="lower left", fontsize=9)
ax[0].set_ylabel("ω [rad/s]  (1 rad/s spin + 0.02 rad/s perturbation)")
save(fig, "fig_three_axes.png")

# ---------------------------------------------------------------- 3. energy dissipation → major-axis spin
s = simulate(I, np.array([0.08, 0.05, 1.0]), torque=internal_damping(I, 0.0015), T=1500, n=8000)
fig = plt.figure(figsize=(16, 5.4), constrained_layout=True)
a = fig.add_subplot(1, 3, 1, projection="3d")
Hb = I * s["w"]; Hm2 = hmag(I, s["w"][0])
a.plot_surface(Hm2 * np.cos(u) * np.sin(v), Hm2 * np.sin(u) * np.sin(v), Hm2 * np.cos(v), color="#e5e7eb", alpha=.2, lw=0)
a.plot(*Hb.T, c="#7c3aed", lw=.8)
a.scatter(*Hb[0], c="k", s=30); a.scatter(*Hb[-1], c=C1, s=40)
a.set_box_aspect((1, 1, 1)); a.set_axis_off(); a.view_init(20, 30)
a.set_title("body-frame H: from the minor axis (top)\nto the major axis, ‖H‖ fixed")
a = fig.add_subplot(1, 3, 2)
for j, col in zip(range(3), [C1, C2, C3]):
    a.plot(s["t"], s["w"][:, j], c=col, lw=1, label=f"ω{j+1}")
a.set_xlabel("t [s]"); a.set_ylabel("ω [rad/s]"); a.legend(); a.set_title("rates: ω₃ (minor) decays, ω₁ (major) takes over")
a = fig.add_subplot(1, 3, 3)
Tn = energy(I, s["w"]); Hn = hmag(I, s["w"])
a.plot(s["t"], Tn, c="#7c3aed", label="kinetic energy T")
a.axhline(Hn[0] ** 2 / (2 * I[2]), c=C3, ls=":", label="H²/2I₃ (minor-axis spin)")
a.axhline(Hn[0] ** 2 / (2 * I[1]), c=C2, ls=":", label="H²/2I₂ (separatrix)")
a.axhline(Hn[0] ** 2 / (2 * I[0]), c=C1, ls=":", label="H²/2I₁ (major-axis spin)")
a2 = a.twinx(); a2.plot(s["t"], Hn / Hn[0], c="k", lw=1); a2.set_ylim(0.99, 1.01); a2.set_ylabel("‖H‖ / ‖H₀‖")
a.set_xlabel("t [s]"); a.set_ylabel("T [J]"); a.legend(fontsize=8.5, loc="lower left"); a.set_title("energy drains, momentum stays")
save(fig, "fig_dissipation.png")
print("dissipation: start ω", s["w"][0].round(3), "end ω", s["w"][-1].round(3))
