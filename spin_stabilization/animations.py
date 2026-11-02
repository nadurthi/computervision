"""Animations: the intermediate-axis (Dzhanibekov) flip, and optimal detumbling of the same body."""
import numpy as np, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from spin import I_DEFAULT, simulate, quat_to_dcm, debs_athans

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
I = I_DEFAULT
# box whose principal inertias are proportional to I (mass 12): a² = 6(I2+I3−I1)/m, …
m = 12.0
half = 0.5 * np.sqrt(6 * np.array([I[1] + I[2] - I[0], I[0] + I[2] - I[1], I[0] + I[1] - I[2]]) / m)
V0 = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
FCOL = ["#bfdbfe", "#bfdbfe", "#fecaca", "#fecaca", "#bbf7d0", "#bbf7d0"]   # faces normal to axis 1, 2, 3
AXC = ["#2563eb", "#dc2626", "#16a34a"]


def make(sim, fname, title, ylabel_ctrl=False, fps=30, every=1):
    t, w, q = sim["t"][::every], sim["w"][::every], sim["q"][::every]
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    ax2 = fig.add_subplot(1, 2, 2)
    L = 2.2
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L); ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title("body in the inertial frame\n(blue: major axis, red: intermediate, green: minor)", fontsize=10)
    coll = Poly3DCollection([], facecolors=FCOL, edgecolors="k", linewidths=.6, alpha=.55, zorder=1)
    ax.add_collection3d(coll)
    lines = [ax.plot([], [], [], c=c, lw=3, zorder=5)[0] for c in AXC]
    hl = ax.plot([], [], [], c="#7c3aed", lw=2.2, ls="--", zorder=6)[0]
    for k, c in enumerate(AXC):
        ax2.plot(t, w[:, k], c=c, lw=1.6, label=f"ω{k+1}")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("body rate [rad/s]"); ax2.legend(loc="upper right")
    if ylabel_ctrl:
        ax2.plot(t, np.linalg.norm(w, axis=1), c="k", lw=1, ls=":", label="‖ω‖")
        ax2.legend(loc="upper right")
    cur = ax2.axvline(0, c="k", lw=1)
    Hin = quat_to_dcm(q[0]) @ (I * w[0]); Hdir = Hin / (np.linalg.norm(Hin) + 1e-12) * 2.0
    fig.suptitle(title, fontsize=12)

    def upd(i):
        C = quat_to_dcm(q[i])
        Vw = V0 @ C.T
        coll.set_verts([[Vw[j] for j in f] for f in FACES])
        for k, ln in enumerate(lines):
            e = C[:, k] * half[k] * 1.9
            ln.set_data([-e[0], e[0]], [-e[1], e[1]]); ln.set_3d_properties([-e[2], e[2]])
        H = C @ (I * w[i]); n = np.linalg.norm(H)
        if n > 1e-6 and not ylabel_ctrl:
            d = H / n * 2.0
            hl.set_data([0, d[0]], [0, d[1]]); hl.set_3d_properties([0, d[2]])
        cur.set_xdata([t[i], t[i]])
        return []
    anim = FuncAnimation(fig, upd, frames=len(t), blit=False)
    anim.save(os.path.join(OUT, fname + ".mp4"), writer=FFMpegWriter(fps=fps, bitrate=2200), dpi=100)
    anim.save(os.path.join(OUT, fname + ".gif"), writer=PillowWriter(fps=15), dpi=60)
    plt.close(fig); print("wrote", fname)


if __name__ == "__main__":
    print("box half-sizes", half.round(3))
    s = simulate(I, np.array([0.01, 1.0, 0.01]), T=75, n=900)
    make(s, "anim_dzhanibekov", "Torque-free spin about the intermediate axis: the body flips over, then flips back (dashed: inertial H, fixed)")
    s = simulate(I, np.array([0.6, -0.7, 0.8]), torque=debs_athans(50, 3), T=12, n=360)
    make(s, "anim_detumble", "Optimal detumbling, u* = −√(q/r) ω  (q = 50, r = 3)", ylabel_ctrl=True)
