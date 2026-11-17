"""Detector-free tote finding: ICP seeded from a grid of initial poses over the stack face.

1. Remove the floor and fit the dominant vertical plane facing the sensor -> tote axes R (yaw from the plane).
2. Depth slabs along the plane normal (front row, plus back rows exposed by picks).
3. Seeds: a grid of tote poses over each slab's face extent, spacing (du, dv) a fraction of the tote face.
4. Each seed runs trimmed ICP, then a pattern search over the in-plane offset (u, v) that maximises the
   ray-consistency score below, then ICP again (ICP alone barely slides along a flat face): crop the cloud to the model's bounding box (+margin that shrinks each round)
   and register it to the CAD with point-to-mesh (default) or point-to-plane ICP.
5. Each converged pose is scored by ray consistency (free-space test). For every lidar point near the hypothesis,
   cast the ray sensor -> point against the CAD placed at the pose; where it hits, the model predicts range t_i:
       inlier     |rho_i - t_i| < tau                (the model explains the return)
       violation  rho_i - t_i > tau_v                (the beam went *through* where the model claims a surface)
       score    = inliers - lambda * violations,  accepted if inliers/(hits) >= f_min and inliers >= n_min
   Point-to-surface distance alone cannot separate a true tote from a pose straddling two neighbours: the faces
   are flat and coplanar, and this CAD's ribs repeat with the tote pitch. The beams that pass through the V-gap
   between two tapered totes can: they violate a straddling pose and not the true ones.
6. Selection = a maximum-weight set of non-overlapping poses (face-rectangle IoU <= 0.2). Greedy by score,
   then 1-for-2 swaps: an accepted pose is replaced by two mutually compatible candidates that overlap only it
   when their summed score is higher. A pose straddling two totes can outscore either true pose alone, but
   not the pair. Finally each accepted pose gets a fine pattern search (2.5 mm) on the ray score.
"""
import numpy as np
import open3d as o3d
from scipy import signal
from .detect import ransac_plane, plane_frame, cad_profile
from .icp import PointToMeshICP


def preprocess(P, sensor, min_points=150):
    sensor = np.asarray(sensor, float)
    n_f, d_f, inl_f = ransac_plane(P, 0.02, normal_ok=lambda n: abs(n[2]) > 0.95)
    keep = ~inl_f & ((P @ n_f + d_f) * np.sign(n_f[2]) > 0.02)
    Q = P[keep]
    n, d, _ = ransac_plane(Q, 0.03, normal_ok=lambda n: abs(n[2]) < 0.2)
    if n @ (sensor - Q.mean(0)) < 0:
        n = -n
    Rf = plane_frame(n)
    u, s, v = (Q @ Rf).T
    h, e = np.histogram(s, bins=np.arange(s.min(), s.max() + 0.02, 0.02))
    pk = signal.find_peaks(np.r_[0, h, 0], height=min_points, distance=10)[0] - 1
    slabs = [0.5 * (e[i] + e[i + 1]) for i in pk]
    return Q, np.nonzero(keep)[0], Rf, u, s, v, slabs


class GridICP:
    def __init__(self, mesh, method="p2mesh", tau=0.012, tau_v=0.015, lam=5.0, f_min=0.7, n_min=100, dof=4):
        self.mesh = mesh; self.cad = cad_profile(mesh); self.method = method
        self.p2m = PointToMeshICP(mesh, cull=True)                  # used for scoring
        self.solver = PointToMeshICP(mesh, cull=True, residual=method, dof=dof)
        self.lo = np.asarray(mesh.vertices).min(0); self.hi = np.asarray(mesh.vertices).max(0)
        self.tau, self.tau_v, self.lam, self.f_min, self.n_min = tau, tau_v, lam, f_min, n_min
        self.full = o3d.t.geometry.RaycastingScene()
        m = o3d.t.geometry.TriangleMesh()
        m.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, np.float32))
        m.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, np.int32))
        self.full.add_triangles(m)

    def _in_box(self, P, T, pad):
        q = (P - T[:3, 3]) @ T[:3, :3]
        return np.all((q > self.lo - pad) & (q < self.hi + pad), 1), q

    def fit(self, P, T0, sensor, margins=(0.08, 0.03, 0.01)):
        T = T0
        for k, pad in enumerate(margins):
            m, _ = self._in_box(P, T, pad)
            if m.sum() < 30:
                return None
            T = self.solver.run(P[m], T, sensor, iters=15, d_max=(max(0.05 * 0.5 ** k, 0.015), 0.015))["T"]
        return T

    def score(self, P, T, sensor):
        m, q = self._in_box(P, T, np.array([0.05, 0.3, 0.05]))
        if m.sum() == 0:
            return dict(inliers=0, violations=0, hits=0, frac=0.0, score=-1e9)
        o = T[:3, :3].T @ (np.asarray(sensor, float) - T[:3, 3])
        q = q[m]; rho = np.linalg.norm(q - o, axis=1); dirn = (q - o) / rho[:, None]
        rays = np.concatenate([np.broadcast_to(o, q.shape), dirn], 1).astype(np.float32)
        t = self.full.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(t)
        res = rho[hit] - t[hit]
        inl = int((np.abs(res) < self.tau).sum()); vio = int((res > self.tau_v).sum())
        return dict(inliers=inl, violations=vio, hits=int(hit.sum()), frac=inl / max(hit.sum(), 1),
                    score=inl - self.lam * vio)

    def slide(self, P, T, sensor, steps=(0.04, 0.02, 0.01, 0.005)):
        """Pattern search on the tote's in-plane offset (x along the face, z up) maximising the ray score."""
        best = self.score(P, T, sensor)["score"]
        for st in steps:
            improved = True
            while improved:
                improved = False
                for dx, dz in ((st, 0), (-st, 0), (0, st), (0, -st)):
                    Tn = T.copy(); Tn[:3, 3] = T[:3, 3] + T[:3, 0] * dx + T[:3, 2] * dz
                    sc = self.score(P, Tn, sensor)["score"]
                    if sc > best:
                        best, T, improved = sc, Tn, True
        return T

    def seeds(self, Rf, u, s, v, slabs, du=0.15, dv=0.075, slab_half=0.07):
        out = []
        for sc_ in slabs:
            m = np.abs(s - sc_) < slab_half
            if m.sum() < self.n_min:
                continue
            for uc in np.arange(u[m].min() + 0.5 * self.cad["W"] - 0.05, u[m].max() - 0.5 * self.cad["W"] + 0.05 + 1e-9, du):
                for vc in np.arange(v[m].min() + 0.5 * self.cad["H"] - 0.05, v[m].max() - 0.5 * self.cad["H"] + 0.05 + 1e-9, dv):
                    T0 = np.eye(4); T0[:3, :3] = Rf
                    T0[:3, 3] = Rf @ np.array([uc, np.median(s[m]) - self.cad["y_rim"] - 0.005, vc - 0.5 * self.cad["H"] - self.cad["z0"]])
                    out.append(dict(T0=T0, slab=float(sc_)))
        return out

    def run(self, P, sensor, du=0.15, dv=0.075, iou_max=0.2, return_all=False):
        Q, qidx, Rf, u, s, v, slabs = preprocess(P, sensor)
        seeds = self.seeds(Rf, u, s, v, slabs, du, dv)
        hyps = []
        for sd in seeds:
            T = self.fit(Q, sd["T0"], sensor)
            if T is None:
                continue
            T = self.slide(Q, T, sensor)
            T = self.fit(Q, T, sensor, margins=(0.02, 0.01)) if T is not None else None
            if T is None:
                continue
            sc = self.score(Q, T, sensor)
            hyps.append(dict(T=T, T0=sd["T0"], slab=sd["slab"], **sc))
        ok = [h for h in hyps if h["frac"] >= self.f_min and h["inliers"] >= self.n_min]
        ok.sort(key=lambda h: -h["score"])
        uniq = []                                                       # merge poses that converged together
        for h in ok:
            if all(np.linalg.norm(h["T"][:3, 3] - u_["T"][:3, 3]) > 0.01 for u_ in uniq):
                uniq.append(h)
        ok = uniq
        acc = []
        for h in ok:
            if all(self.face_iou(h["T"], a["T"]) <= iou_max for a in acc):
                acc.append(h)
        acc = self._swap(acc, ok, iou_max)
        for h in acc:                                                   # polish
            h["T"] = self.slide(Q, h["T"], sensor, steps=(0.005, 0.0025))
            h.update(self.score(Q, h["T"], sensor))
        for h in acc:
            h["points"] = Q[self._in_box(Q, h["T"], 0.01)[0]]
        info = dict(n_seeds=len(seeds), n_converged=len(hyps), n_valid=len(ok), slabs=slabs, Rf=Rf)
        if return_all:
            info["hyps"] = hyps
        return acc, info

    def _swap(self, acc, cands, iou_max, rounds=5):
        def compatible(h, S):
            return all(self.face_iou(h["T"], a["T"]) <= iou_max for a in S)
        for _ in range(rounds):
            changed = False
            for a in list(acc):
                rest = [b for b in acc if b is not a]
                pool = [c for c in cands if c is not a and compatible(c, rest)]
                best = None
                for i, c1 in enumerate(pool):
                    for c2 in pool[i + 1:]:
                        if self.face_iou(c1["T"], c2["T"]) <= iou_max and c1["score"] + c2["score"] > a["score"] + 1e-6:
                            if best is None or c1["score"] + c2["score"] > best[0]:
                                best = (c1["score"] + c2["score"], c1, c2)
                if best is not None:
                    acc = rest + [best[1], best[2]]; changed = True
                    break
            if not changed:
                break
        # also add any candidate that is now compatible with everything (gaps opened by swaps)
        for c in cands:
            if all(c is not x for x in acc) and compatible(c, acc):
                acc.append(c)
        return sorted(acc, key=lambda h: -h["score"])

    def face_iou(self, T1, T2):
        """IoU of the two totes' front rectangles (x-z extent) in a common face frame; 0 if depths differ."""
        if abs((T1[:3, 3] - T2[:3, 3]) @ T1[:3, 1]) > 0.2:
            return 0.0
        d = T1[:3, :3].T @ (T2[:3, 3] - T1[:3, 3])
        w = self.hi[0] - self.lo[0]; h = self.hi[2] - self.lo[2]
        ox = max(0.0, w - abs(d[0])); oz = max(0.0, h - abs(d[2]))
        inter = ox * oz
        return inter / (2 * w * h - inter)
