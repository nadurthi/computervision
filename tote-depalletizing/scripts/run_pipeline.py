"""Run tote detection + point-to-mesh ICP on scans (CPU stand-in or Isaac Sim output) and score them.

    python scripts/run_pipeline.py --scans data/sim --out results/sim
    python scripts/run_pipeline.py --scans /out/tote_pallet --out results/isaac   # Isaac Sim lidar scans

Each scan dir holds points.npy (N,3 world, metres), gt.json (sensor_position + per-tote pose); labels.npy is only
used to count which totes the sensor actually hit.
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import trimesh
from totes.pipeline import Estimator, load_gt, score, METHODS

ap = argparse.ArgumentParser()
ap.add_argument("--scans", default="data/sim")
ap.add_argument("--out", default="results/sim")
ap.add_argument("--cad", default="assets/tote_600x400x300.obj")
ap.add_argument("--cad-meta", default="assets/tote_cad.json", help="front_center / front_normal of the CAD")
ap.add_argument("--methods", default=",".join(METHODS))
ap.add_argument("--min-hits", type=int, default=150, help="a GT tote counts as visible with this many lidar hits")
a = ap.parse_args()

mesh = trimesh.load(a.cad, force="mesh")
meta = json.load(open(a.cad_meta))
est = Estimator(mesh, meta["front_face_center"], meta["front_face_normal"])
methods = a.methods.split(",")
os.makedirs(a.out, exist_ok=True)
allrows, summary = [], []
for d in sorted(glob.glob(os.path.join(a.scans, "scan_*"))):
    P = np.load(os.path.join(d, "points.npy")).astype(float)
    g, present = load_gt(os.path.join(d, "gt.json"))
    res, t_det = est.estimate(P, g["sensor_position"], methods)
    lab = np.load(os.path.join(d, "labels.npy")) if os.path.exists(os.path.join(d, "labels.npy")) else None
    visible = sorted(int(i) for i, c in zip(*np.unique(lab[lab >= 0], return_counts=True)) if c >= a.min_hits) if lab is not None else []
    for m in methods:
        rows, s = score(res, present, m)
        for r in rows:
            r.update(scan=os.path.basename(d), method=m)
        allrows += rows
    found = sorted({r["gt_id"] for r in rows})
    summary.append(dict(scan=os.path.basename(d), n_points=len(P), visible=visible, detected=found,
                        missed=sorted(set(visible) - set(found)), false=[r["gt_id"] for r in rows if r["gt_id"] not in visible],
                        t_detect=t_det))
    e = np.array([[r["e_n"], r["e_l"], r["e_v"]] for r in allrows if r["scan"] == os.path.basename(d) and r["method"] == "p2mesh"]) * 1000
    print(f"{os.path.basename(d)}: {len(P)} pts, visible {len(visible)}, detected {len(found)}, "
          f"p2mesh rms (n,l,v) mm = {np.round(np.sqrt((e ** 2).mean(0)), 2) if len(e) else '-'}")
json.dump(dict(rows=allrows, scans=summary), open(os.path.join(a.out, "results.json"), "w"), indent=1)
for m in methods:
    E = np.array([[r["e_n"], r["e_l"], r["e_v"], r["e"], abs(r["yaw"])] for r in allrows if r["method"] == m])
    print(f"{m:14s} n={len(E)}  rms e_n {1e3*np.sqrt((E[:,0]**2).mean()):.2f}  e_l {1e3*np.sqrt((E[:,1]**2).mean()):.2f}  "
          f"e_v {1e3*np.sqrt((E[:,2]**2).mean()):.2f}  |e| p95 {1e3*np.percentile(E[:,3],95):.2f}  max {1e3*E[:,3].max():.2f} mm  "
          f"|yaw| p95 {np.percentile(E[:,4],95):.2f} deg")
