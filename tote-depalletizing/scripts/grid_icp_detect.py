"""Find every tote in a lidar scan by running ICP from a grid of initial poses, then report each tote's
front-face centre (the grasp reference for the arm). Works on CPU-sim and Isaac Sim scans alike.

    python scripts/grid_icp_detect.py --scans data/sim --method p2mesh --out results/grid_p2mesh
    python scripts/grid_icp_detect.py --scans data/sim --method p2plane --out results/grid_p2plane

Writes <out>/<scan>.json with one entry per tote: pose T (tote->world), front_center, front_normal, inliers,
inlier fraction, score; if gt.json is present also the matched ground-truth id and error; plus figures with
the seeds, converged poses and accepted totes over the camera image (if rgb.png exists) and in the face plane.
"""
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import trimesh
from totes.grid_icp import GridICP
from totes.pipeline import load_gt

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="data/sim")
ap.add_argument("--out", default="results/grid_p2mesh")
ap.add_argument("--method", default="p2mesh", choices=["p2mesh", "p2plane"])
ap.add_argument("--dof", type=int, default=4, choices=[4, 6], help="4 = x,y,z,yaw (tote resting flat); 6 = full pose")
ap.add_argument("--du", type=float, default=0.15, help="seed spacing along the face (m)")
ap.add_argument("--dv", type=float, default=0.075, help="seed spacing vertically (m)")
ap.add_argument("--cad", default="assets/tote_600x400x300.obj")
ap.add_argument("--cad-meta", default="assets/tote_cad.json")
ap.add_argument("--no-figs", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
mesh = trimesh.load(a.cad, force="mesh"); meta = json.load(open(a.cad_meta))
c = np.r_[meta["front_face_center"], 1.0]; nf = np.array(meta["front_face_normal"])
g = GridICP(mesh, a.method, dof=a.dof)
summary = []
for d in sorted(glob.glob(os.path.join(a.scans, "scan_*"))):
    name = os.path.basename(d)
    P = np.load(os.path.join(d, "points.npy")).astype(float)
    gt, present = load_gt(os.path.join(d, "gt.json"))
    sensor = gt["sensor_position"]
    t0 = time.time(); acc, info = g.run(P, sensor, a.du, a.dv, return_all=True); dt = time.time() - t0
    totes = []
    for h in acc:
        T = h["T"]; fc = (T @ c)[:3]
        rec = dict(T=T.tolist(), front_center=fc.tolist(), front_normal=(T[:3, :3] @ nf).tolist(),
                   inliers=h["inliers"], frac=h["frac"], score=h["score"])
        if present:
            gid = min(present, key=lambda i: np.linalg.norm(np.array(present[i]["T"])[:3, 3] - T[:3, 3]))
            rec.update(gt_id=gid, err=(fc - np.array(present[gid]["front_center"])).tolist())
        totes.append(rec)
    lab = np.load(os.path.join(d, "labels.npy")) if os.path.exists(os.path.join(d, "labels.npy")) else None
    visible = sorted(int(i) for i, n in zip(*np.unique(lab[lab >= 0], return_counts=True)) if n >= 150) if lab is not None else []
    found = [t.get("gt_id") for t in totes]
    E = np.array([t["err"] for t in totes]) * 1000 if present and totes else np.zeros((0, 3))
    s = dict(scan=name, method=a.method, n_seeds=info["n_seeds"], n_converged=info["n_converged"], n_valid=info["n_valid"],
             n_found=len(totes), visible=visible, missed=sorted(set(visible) - set(found)),
             duplicates=len(found) - len(set(found)), false=[f for f in found if f not in visible],
             err_rms_xyz_mm=np.sqrt((E ** 2).mean(0)).tolist() if len(E) else None,
             err_max_mm=float(np.linalg.norm(E, axis=1).max()) if len(E) else None, time_s=dt)
    summary.append(s)
    json.dump(dict(summary=s, totes=totes), open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
    print(f"{name}: seeds {s['n_seeds']} -> valid {s['n_valid']} -> totes {s['n_found']} (visible {len(visible)}, "
          f"missed {s['missed']}, dup {s['duplicates']}, false {s['false']}), err rms xyz "
          f"{np.round(s['err_rms_xyz_mm'], 2) if s['err_rms_xyz_mm'] else '-'} mm, max {s['err_max_mm'] and round(s['err_max_mm'], 1)} mm, {dt:.1f} s")
    if not a.no_figs:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        Rf = info["Rf"]; Q = P @ Rf
        fig, ax = plt.subplots(1, 2 if "camera" in gt else 1, figsize=(15 if "camera" in gt else 7, 8), squeeze=False)
        ax0 = ax[0, 0]
        fl = P[:, 2] > 0.03; Q = Q[fl]
        ax0.scatter(Q[:, 0], Q[:, 2], s=0.2, c=Q[:, 1], cmap="viridis", vmin=np.percentile(Q[:, 1], 2), vmax=np.percentile(Q[:, 1], 60))
        for h in info["hyps"]:
            p0 = Rf.T @ h["T0"][:3, 3]; p1 = Rf.T @ h["T"][:3, 3]
            ax0.plot([p0[0], p1[0]], [p0[2] + 0.15, p1[2] + 0.15], "-", color="0.6", lw=0.4)
            ax0.plot(p1[0], p1[2] + 0.15, ".", color="tab:red", ms=2)
        for t in totes:
            p = Rf.T @ np.array(t["T"])[:3, 3]; f = Rf.T @ np.array(t["front_center"])
            ax0.add_patch(Rectangle((p[0] - 0.3, p[2]), 0.6, 0.3, fill=False, ec="tab:orange", lw=1.5))
            ax0.plot(f[0], f[2], "+", color="k", ms=10, mew=2)
        ax0.set_aspect("equal"); ax0.set_xlabel("u along face (m)"); ax0.set_ylabel("height (m)")
        ax0.set_title(f"{name}: {info['n_seeds']} seeds (grey: seed -> converged), {len(totes)} totes ({a.method})")
        if "camera" in gt:
            from PIL import Image
            from totes.camera import project
            cam = gt["camera"]; K = np.array(cam["K"]); C = np.array(cam["position"]); Rwc = np.array(cam["R_world_from_cam"])
            ax1 = ax[0, 1]; ax1.imshow(Image.open(os.path.join(d, "rgb.png")))
            V = np.asarray(mesh.vertices)
            E_ = mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.radians(30)]      # crease edges of the CAD
            for t in totes:
                T = np.array(t["T"]); Vw = V @ T[:3, :3].T + T[:3, 3]
                uv, z = project(Vw, C, K, Rwc)
                for i, j in E_:
                    ax1.plot(uv[[i, j], 0], uv[[i, j], 1], "-", color="tab:orange", lw=0.4, alpha=0.8)
                f, _ = project(np.array(t["front_center"]), C, K, Rwc); ax1.plot(f[0], f[1], "+", color="r", ms=12, mew=2)
            ax1.set_axis_off(); ax1.set_title("CAD at the estimated poses, projected into the camera; + = front-face centre")
        plt.tight_layout(); plt.savefig(os.path.join(a.out, f"{name}.png"), dpi=110); plt.close()
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
