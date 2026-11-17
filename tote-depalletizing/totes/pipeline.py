"""Detection -> ICP -> front-face centre, and evaluation against ground truth (gt.json)."""
import json, time
import numpy as np
from .detect import detect
from .icp import PointToMeshICP, O3DICP, crop_to_model

METHODS = ("p2mesh", "p2mesh_nocull", "o3d_p2plane", "o3d_p2point", "init_only")


class Estimator:
    def __init__(self, mesh, front_center, front_normal):
        self.mesh = mesh
        self.c = np.r_[front_center, 1.0]; self.n = np.asarray(front_normal, float)
        self.solvers = {"p2mesh": PointToMeshICP(mesh, cull=True), "p2mesh_nocull": PointToMeshICP(mesh, cull=False)}
        self._o3d = {}

    def solver(self, method):
        if method.startswith("o3d"):
            if method not in self._o3d:
                self._o3d[method] = O3DICP(self.mesh, "point_to_plane" if method == "o3d_p2plane" else "point_to_point")
            return self._o3d[method]
        return self.solvers[method]

    def refine(self, pts, T0, sensor, method="p2mesh"):
        if method == "init_only":
            return dict(T=T0, hist=[], H=None, inlier_rms=np.nan)
        s = self.solver(method)
        if method.startswith("p2mesh"):
            r = s.run(pts, T0, sensor, iters=40, d_max=(0.05, 0.015))
            m = crop_to_model(s, pts, r["T"])
            return s.run(pts[m], r["T"], sensor, iters=40, d_max=(0.015, 0.015))
        return s.run(pts, T0, sensor, d_max=0.02)

    def estimate(self, P, sensor, methods=("p2mesh",)):
        t0 = time.time()
        dets = detect(P, self.mesh, sensor)
        t_det = time.time() - t0
        out = []
        for d in dets:
            rec = dict(n_points=len(d["points"]), slab=d["slab"], T0=d["T0"], idx=d["idx"], fit={})
            for m in methods:
                t1 = time.time(); r = self.refine(d["points"], d["T0"], sensor, m)
                T = r["T"]
                rec["fit"][m] = dict(T=T, front_center=(T @ self.c)[:3], front_normal=T[:3, :3] @ self.n,
                                     rms=r["inlier_rms"], iters=len(r["hist"]), H=r["H"], time=time.time() - t1)
            out.append(rec)
        return out, t_det


def load_gt(path):
    g = json.load(open(path))
    return g, {t["id"]: t for t in g["totes"] if t["present"]}


def yaw_err(T, Tg):
    R = Tg[:3, :3].T @ T[:3, :3]
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def score(results, gt_present, method):
    """Match each estimate to the nearest present GT tote (by origin) and express the front-centre error in
    the GT face frame: e_n along the face normal (depth), e_l lateral (horizontal in the face), e_v vertical."""
    rows = []
    for k, r in enumerate(results):
        f = r["fit"][method]
        gid = min(gt_present, key=lambda i: np.linalg.norm(np.array(gt_present[i]["T"])[:3, 3] - f["T"][:3, 3]))
        g = gt_present[gid]; Tg = np.array(g["T"])
        n = np.array(g["front_normal"]); vz = np.array([0, 0, 1.0]); vz = vz - n * (n @ vz); vz /= np.linalg.norm(vz)
        l = np.cross(vz, n)
        e = f["front_center"] - np.array(g["front_center"])
        rows.append(dict(det=k, gt_id=gid, layer=g["layer"], row=g["row"], col=g["col"], n_points=r["n_points"],
                         e_n=float(e @ n), e_l=float(e @ l), e_v=float(e @ vz), e=float(np.linalg.norm(e)),
                         yaw=float(yaw_err(f["T"], Tg)), rms=float(f["rms"]), iters=f["iters"], time=f["time"]))
    ids = [r["gt_id"] for r in rows]
    dup = len(ids) - len(set(ids))
    return rows, dict(n_det=len(rows), n_gt_visible=None, duplicates=dup)
