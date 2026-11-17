"""Figures for the write-up -> docs/figures/*.png (reads data/sim, results/)."""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
from PIL import Image
from totes.grid_icp import GridICP, preprocess

OUT = "docs/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.25})
C = {"p2mesh": "#1f5fa8", "p2mesh_nocull": "#7aa6d6", "o3d_p2plane": "#d9822b", "o3d_p2point": "#c23b3b", "init_only": "#888888"}
LBL = {"p2mesh": "point-to-mesh, culled", "p2mesh_nocull": "point-to-mesh, no culling", "o3d_p2plane": "Open3D point-to-plane",
       "o3d_p2point": "Open3D point-to-point", "init_only": "detector only"}

# 1. the scene: camera image and lidar points
d = "data/sim/scan_00"; P = np.load(f"{d}/points.npy"); L = np.load(f"{d}/labels.npy")
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax[0].imshow(Image.open(f"{d}/rgb.png")); ax[0].set_axis_off(); ax[0].set_title("camera (1600×1200, 80° vertical FOV), 1.5 m from the stack")
m = (L >= -1)
sc = ax[1].scatter(-P[m, 1], P[m, 2], c=P[m, 0], s=0.15, cmap="viridis", vmin=1.58, vmax=1.75)
ax[1].set_aspect("equal"); ax[1].set_xlabel("lateral −y (m)"); ax[1].set_ylabel("height z (m)")
ax[1].set_title(f"lidar (OS0-128-like, σ = 5 mm): {m.sum():,} points on stack + pallet")
plt.colorbar(sc, ax=ax[1], label="range along x (m)", shrink=0.8); ax[1].grid(False)
plt.tight_layout(); plt.savefig(f"{OUT}/01_scene.png"); plt.close()

# 2. refinement comparison (seam detector + each refinement), front-face centre error
R = json.load(open("results/sim/results.json"))["rows"]
methods = ["p2mesh", "p2mesh_nocull", "o3d_p2plane", "o3d_p2point", "init_only"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
w = 0.16
for k, mth in enumerate(methods):
    E = np.array([[r["e_n"], r["e_l"], r["e_v"]] for r in R if r["method"] == mth]) * 1000
    ax[0].bar(np.arange(3) + (k - 2) * w, np.sqrt((E ** 2).mean(0)), w, color=C[mth], label=LBL[mth])
    e = np.sort(np.linalg.norm(E, axis=1)); ax[1].plot(e, np.linspace(0, 1, len(e)), color=C[mth], lw=2, label=LBL[mth])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["along face normal", "lateral", "vertical"]); ax[0].set_ylabel("RMS error (mm)")
ax[0].set_title("front-face centre error by axis (42 totes, scans 00–02)"); ax[0].legend(fontsize=8, frameon=False)
ax[1].set_xlabel("|front-face centre error| (mm)"); ax[1].set_ylabel("fraction of totes"); ax[1].set_title("error CDF")
plt.tight_layout(); plt.savefig(f"{OUT}/02_refinement_comparison.png"); plt.close()

# 3. why distance alone fails: score vs lateral offset
mesh = trimesh.load("assets/tote_600x400x300.obj", force="mesh"); g = GridICP(mesh)
gt = json.load(open(f"{d}/gt.json")); S = gt["sensor_position"]; Q = preprocess(P.astype(float), S)[0]
T = np.array([t for t in gt["totes"] if t["id"] == 8][0]["T"])
import open3d as o3d
du = np.linspace(-0.45, 0.45, 91); ray, dist = [], []
for x in du:
    T0 = T.copy(); T0[:3, 3] += T[:3, 0] * x
    ray.append(g.score(Q, T0, S))
    mm, q = g._in_box(Q, T0, 0.01); scn = g.p2m._visible_scene(T0[:3, :3].T @ (np.array(S) - T0[:3, 3]))
    r = scn.compute_distance(o3d.core.Tensor(q[mm].astype(np.float32))).numpy(); dist.append(np.mean(np.minimum(r, 0.012) ** 2) * 1e6)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(du * 1000, dist, color="#555"); ax[0].set_title("truncated point-to-mesh cost (mm²)"); ax[0].set_xlabel("lateral offset from true pose (mm)")
ax[1].plot(du * 1000, [s["violations"] for s in ray], color="#c23b3b"); ax[1].set_title("free-space violations (beams through the model)")
ax[1].set_xlabel("lateral offset (mm)")
ax[2].plot(du * 1000, [s["score"] for s in ray], color="#1f5fa8"); ax[2].set_title("ray score = inliers − 5·violations"); ax[2].set_xlabel("lateral offset (mm)")
for a in ax:
    a.axvline(0, color="k", lw=0.8, ls=":"); a.axvline(301.5, color="k", lw=0.6, ls="--"); a.axvline(-301.5, color="k", lw=0.6, ls="--")
plt.tight_layout(); plt.savefig(f"{OUT}/03_straddle_ambiguity.png"); plt.close()

# 4. noise sweep
N = json.load(open("results/studies/noise.json")); gg = collections.defaultdict(list)
for r in N: gg[(r["method"], r["sigma"])].append(r)
sig = [0, 0.002, 0.005, 0.01, 0.02]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for mth in ("p2mesh", "o3d_p2plane", "init_only"):
    for j, k in enumerate(("e_n", "e_l", "e_v")):
        ax[j].plot(np.array(sig) * 1000, [1000 * np.median([r[k] for r in gg[(mth, s)]]) for s in sig], "o-", color=C[mth], label=LBL[mth])
for j, t in enumerate(("along face normal", "lateral", "vertical")):
    ax[j].set_title(f"RMS {t} error (median of 3 seeds)"); ax[j].set_xlabel("lidar range noise σ (mm)"); ax[j].set_ylabel("mm"); ax[j].set_yscale("log")
ax[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/04_noise_sweep.png"); plt.close()

# 5. convergence basin
B = json.load(open("results/studies/basin.json")); gb = collections.defaultdict(list)
for r in B: gb[(r["method"], r["dt"], r["dyaw"])].append(r["e"] < 0.005)
dts = [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]; dys = [0, 2, 5, 10, 15]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, mth in zip(ax, ("p2mesh", "p2mesh_nocull", "o3d_p2plane")):
    M = np.array([[np.mean(gb[(mth, dt, dy)]) for dy in dys] for dt in dts])
    im = a.imshow(M, vmin=0, vmax=1, cmap="Blues", origin="lower", aspect="auto")
    for i in range(len(dts)):
        for j in range(len(dys)):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < 0.6 else "w")
    a.set_xticks(range(len(dys))); a.set_xticklabels(dys); a.set_yticks(range(len(dts))); a.set_yticklabels([int(x * 1000) for x in dts])
    a.set_xlabel("initial yaw error (deg)"); a.set_ylabel("initial in-plane offset (mm)"); a.set_title(f"{LBL[mth]}: P(|e| < 5 mm)"); a.grid(False)
plt.tight_layout(); plt.savefig(f"{OUT}/05_convergence_basin.png"); plt.close()

# 6. sequential depalletizing
Sq = json.load(open("results/studies/sequential.json"))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar([s["step"] for s in Sq], [1000 * s["pick_error"] for s in Sq], color="#1f5fa8")
ax[0].set_xlabel("pick #"); ax[0].set_ylabel("front-face centre error of picked tote (mm)"); ax[0].set_title("28-pick depalletizing loop: every pick on the right tote")
ax[1].plot([s["step"] for s in Sq], [s["n_present"] for s in Sq], "k-", label="totes on pallet")
ax[1].plot([s["step"] for s in Sq], [len({r["gt_id"] for r in s["rows"]}) for s in Sq], "o-", color="#1f5fa8", label="distinct totes detected")
ax[1].plot([s["step"] for s in Sq], [len(s["rows"]) for s in Sq], "x--", color="#c23b3b", label="detections (incl. duplicates)")
ax[1].set_xlabel("pick #"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("seam detector bookkeeping")
plt.tight_layout(); plt.savefig(f"{OUT}/06_sequential.png"); plt.close()
for k in ("scan_00", "scan_01", "scan_03"):
    Image.open(f"results/grid_p2mesh/{k}.png").save(f"{OUT}/07_grid_{k}.png")
print(sorted(os.listdir(OUT)))
