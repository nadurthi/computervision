"""Studies for the tote point-to-mesh ICP write-up (CPU ray-cast lidar stand-in).

    python scripts/studies.py --out results/studies [--only noise,basin,seq]

noise : front-face-centre error vs lidar range noise sigma (0..20 mm), 3 seeds, full stack (pick state 0)
basin : convergence basin -- ICP from perturbed starts (in-plane offset 0..10 cm x yaw 0..15 deg), GT-segmented points
seq   : full depalletizing loop, 28 picks: scan -> detect -> ICP -> pick the highest/nearest estimate -> remove the
        true tote under it -> rescan. Also records the Gauss-Newton Hessian for the predicted (Cramer-Rao style)
        covariance of the front-face centre.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import trimesh
from totes.cad import ToteSpec, make_tote, make_pallet
from totes.scene import make_stack, pick_order, front_center_world
from totes.lidar import build_scene, scan, OS0_128
from totes.pipeline import Estimator, score
from totes.icp import skew_exp

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/studies")
ap.add_argument("--only", default="noise,basin,seq")
ap.add_argument("--sensor", default="0.1,0,1.2")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
only = a.only.split(",")
sensor = np.array([float(v) for v in a.sensor.split(",")])
spec = ToteSpec(); tote = make_tote(spec); pallet = make_pallet()
meta = json.load(open(os.path.join(os.path.dirname(__file__), "..", "assets", "tote_cad.json")))
est = Estimator(tote, meta["front_face_center"], meta["front_face_normal"])
totes = make_stack(seed=0, spec=spec); byid = {t["id"]: t for t in totes}


def gt_present(present):
    out = {}
    for t in totes:
        if t["id"] in present:
            c, n = front_center_world(t["T"], spec)
            out[t["id"]] = dict(T=t["T"].tolist(), front_center=c.tolist(), front_normal=n.tolist(), layer=t["layer"], row=t["row"], col=t["col"])
    return out


def rmsd(rows, k):
    return float(np.sqrt(np.mean([r[k] ** 2 for r in rows]))) if rows else float("nan")


if "noise" in only:
    out = []
    sc, labels = build_scene(tote, totes, set(byid), pallet)
    for sigma in (0.0, 0.002, 0.005, 0.01, 0.02):
        for seed in range(3):
            pts, lab, _, _ = scan(sc, labels, sensor, OS0_128, sigma, np.random.default_rng(seed))
            res, _ = est.estimate(pts, sensor, ("p2mesh", "o3d_p2plane", "init_only"))
            for m in ("p2mesh", "o3d_p2plane", "init_only"):
                rows, _ = score(res, gt_present(set(byid)), m)
                out.append(dict(sigma=sigma, seed=seed, method=m, n_det=len(rows), e_n=rmsd(rows, "e_n"), e_l=rmsd(rows, "e_l"),
                                e_v=rmsd(rows, "e_v"), e_max=max(r["e"] for r in rows) if rows else None))
            print("noise", sigma, seed, [(o["method"], o["n_det"], round(1e3 * o["e_l"], 2), round(1e3 * o["e_v"], 2)) for o in out[-3:]])
    json.dump(out, open(os.path.join(a.out, "noise.json"), "w"), indent=1)

if "basin" in only:
    sc, labels = build_scene(tote, totes, set(byid), pallet)
    pts, lab, _, _ = scan(sc, labels, sensor, OS0_128, 0.005, np.random.default_rng(0))
    rng = np.random.default_rng(1); out = []
    c = np.r_[meta["front_face_center"], 1.0]
    for tid in (0, 12, 24):
        P = pts[lab == tid]; Tg = byid[tid]["T"]; cg = (Tg @ c)[:3]
        for dt in (0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1):
            for dyaw in (0, 2, 5, 10, 15):
                for trial in range(8):
                    ang = rng.uniform(0, 2 * np.pi)
                    off = dt * np.array([np.cos(ang), 0.0, np.sin(ang)])          # in the face plane (tote x, z)
                    yaw = np.radians(dyaw) * rng.choice([-1, 1])
                    T0 = Tg.copy(); T0[:3, 3] += Tg[:3, :3] @ off
                    T0[:3, :3] = Tg[:3, :3] @ skew_exp(np.array([0, 0, yaw]))
                    for m in ("p2mesh", "p2mesh_nocull", "o3d_p2plane"):
                        T = est.refine(P, T0, sensor, m)["T"]
                        e = np.linalg.norm((T @ c)[:3] - cg)
                        out.append(dict(tote=tid, dt=dt, dyaw=dyaw, method=m, e=float(e)))
        print("basin tote", tid, "done")
    json.dump(out, open(os.path.join(a.out, "basin.json"), "w"), indent=1)

if "seq" in only:
    present = set(byid); out = []; rng = np.random.default_rng(7)
    c = np.r_[meta["front_face_center"], 1.0]; A_c = np.c_[np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]]), -np.eye(3)]
    step = 0
    while present:
        sc, labels = build_scene(tote, totes, present, pallet)
        pts, lab, _, _ = scan(sc, labels, sensor, OS0_128, 0.005, rng)
        t0 = time.time(); res, t_det = est.estimate(pts, sensor, ("p2mesh",)); t_all = time.time() - t0
        rows, _ = score(res, gt_present(present), "p2mesh")
        for r, rr in zip(rows, res):
            f = rr["fit"]["p2mesh"]
            if f["H"] is not None:                 # predicted covariance of the front-face centre (tote frame)
                Sxi = (f["rms"] ** 2) * np.linalg.inv(f["H"])
                Sc = A_c @ Sxi @ A_c.T
                r["pred_std_tote_xyz"] = np.sqrt(np.diag(Sc)).tolist()
                r["H_eig"] = np.linalg.eigvalsh(f["H"]).tolist()
        hits = {int(i): int(n) for i, n in zip(*np.unique(lab[lab >= 0], return_counts=True))}
        if not res:
            print("no detections with", len(present), "totes left"); break
        # pick: highest top face first, then nearest to the sensor
        k = max(range(len(res)), key=lambda i: (round(res[i]["fit"]["p2mesh"]["T"][2, 3] / 0.1), -res[i]["fit"]["p2mesh"]["T"][0, 3]))
        picked = rows[k]["gt_id"]
        out.append(dict(step=step, n_present=len(present), n_points=len(pts), hits=hits, rows=rows, picked=picked,
                        pick_error=rows[k]["e"], t_detect=t_det, t_total=t_all))
        print(f"step {step:2d}: {len(present)} left, {len(rows)} detected, pick {picked} err {1e3*rows[k]['e']:.1f} mm, {t_all:.2f} s")
        present.discard(picked); step += 1
    json.dump(out, open(os.path.join(a.out, "sequential.json"), "w"), indent=1)
