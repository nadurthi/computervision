"""Point-to-mesh ICP against the sensor-facing part of the CAD mesh, plus baselines.

State: T (tote -> world). Data points p_i (world) are moved into the tote frame, q_i = T^-1 p_i, and matched to
their closest point m_i on the mesh. The residual is the point-to-mesh distance:

    face interior:      r_i = n_i . (q_i - m_i)                (point-to-plane on the closest triangle)
    edge / vertex:      r_i = |q_i - m_i|,  direction d_i = (q_i - m_i) / |q_i - m_i|

Linearising a small motion of the points, q_i' = q_i + w x q_i + t, gives J_i = [ (q_i x d_i)^T, d_i^T ].
Each Gauss-Newton step solves (J^T W J + lam I) xi = -J^T W r with Huber weights W, applies the motion to the
points and composes T <- T * dT^-1. Only triangles whose normal faces the sensor are used: a thin-walled
tote has an inner wall 4 mm behind the outer one, and matching to it is a local minimum.
"""
import numpy as np
import open3d as o3d


def _scene(V, F):
    sc = o3d.t.geometry.RaycastingScene()
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(V.astype(np.float32))
    m.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
    sc.add_triangles(m)
    return sc


def skew_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


class PointToMeshICP:
    """residual='p2mesh': closest point on the (visible) triangle mesh, distance along the face normal inside a
    triangle and along the offset at edges/vertices.  residual='p2plane': classical point-to-plane -- nearest
    point of a dense sample of the visible surface and that sample's normal.
    dof=6: full pose; dof=4: x, y, z + yaw only (a tote resting flat on a pallet or another tote)."""
    def __init__(self, mesh, cull=True, residual="p2mesh", dof=6, sample_spacing=0.004):
        self.V = np.asarray(mesh.vertices, float); self.F = np.asarray(mesh.faces)
        self.fn = np.asarray(mesh.face_normals, float)
        self.fc = self.V[self.F].mean(1)
        self.cull = cull; self.residual = residual; self.dof = dof
        self._cache = {}; self._kd = {}
        if residual == "p2plane":
            import trimesh
            n = int(mesh.area / sample_spacing ** 2)
            self.S, fi = trimesh.sample.sample_surface_even(mesh, n, seed=0)
            self.S_f = fi; self.S_n = self.fn[fi]

    def _visible_samples(self, s_tote):
        from scipy.spatial import cKDTree
        mask = ((s_tote - self.fc) * self.fn).sum(1) > 0 if self.cull else np.ones(len(self.F), bool)
        key = mask.tobytes()
        if key not in self._kd:
            keep = mask[self.S_f]
            self._kd[key] = (cKDTree(self.S[keep]), self.S[keep], self.S_n[keep])
        return self._kd[key]

    def _visible_scene(self, s_tote):
        mask = ((s_tote - self.fc) * self.fn).sum(1) > 0 if self.cull else np.ones(len(self.F), bool)
        key = mask.tobytes()
        if key not in self._cache:
            self._cache[key] = _scene(self.V, self.F[mask])
        return self._cache[key]

    def run(self, P, T0, sensor, iters=40, d_max=(0.05, 0.008), huber=1.345, tol=1e-6):
        T = T0.copy(); hist = []
        sensor = np.asarray(sensor, float)
        for it in range(iters):
            R, t = T[:3, :3], T[:3, 3]
            q = (P - t) @ R
            if self.residual == "p2plane":
                kd, S, Sn = self._visible_samples(R.T @ (sensor - t))
                _, j = kd.query(q)
                diff = q - S[j]; d = Sn[j]
                r = (d * diff).sum(1)
                r = np.where(np.linalg.norm(diff, axis=1) > 0.05, 1e3, r)    # no pairing beyond 5 cm
            else:
                sc = self._visible_scene(R.T @ (sensor - t))
                ans = sc.compute_closest_points(o3d.core.Tensor(q.astype(np.float32)))
                m = ans["points"].numpy().astype(float); nm = ans["primitive_normals"].numpy().astype(float)
                diff = q - m; dist = np.linalg.norm(diff, axis=1)
                dirn = np.where(dist[:, None] > 1e-9, diff / np.maximum(dist, 1e-12)[:, None], nm)
                interior = np.abs((dirn * nm).sum(1)) > 0.95
                d = np.where(interior[:, None], nm, dirn)
                r = (d * diff).sum(1)
            gate = max(d_max[0] * 0.85 ** it, d_max[1])
            ok = np.abs(r) < gate
            if ok.sum() < 30:
                break
            rr = r[ok]; sig = 1.4826 * np.median(np.abs(rr)) + 1e-5
            k = huber * sig
            w = np.where(np.abs(rr) <= k, 1.0, k / np.abs(rr))
            J = np.concatenate([np.cross(q[ok], d[ok]), d[ok]], 1)
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ rr
            if self.dof == 4:                                   # yaw about tote z (= world up) + translation
                sel = [2, 3, 4, 5]
                xi = np.zeros(6); xi[sel] = -np.linalg.solve(H[np.ix_(sel, sel)] + 1e-9 * np.eye(4), g[sel])
            else:
                xi = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
            dR = skew_exp(xi[:3]); dT = np.eye(4); dT[:3, :3] = dR; dT[:3, 3] = xi[3:]
            T = T @ np.linalg.inv(dT)
            hist.append(dict(it=it, rms=float(np.sqrt(np.mean(rr ** 2))), n=int(ok.sum()), step=float(np.linalg.norm(xi))))
            if np.linalg.norm(xi) < tol:
                break
        return dict(T=T, hist=hist, H=H if hist else None, inlier_rms=hist[-1]["rms"] if hist else np.nan)


class O3DICP:
    """Baselines: Open3D point-to-point / point-to-plane ICP against points sampled densely on the whole mesh."""
    def __init__(self, mesh, method="point_to_plane", n_samples=60000):
        import trimesh
        pts, fi = trimesh.sample.sample_surface_even(mesh, n_samples, seed=0)
        self.target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        self.target.normals = o3d.utility.Vector3dVector(mesh.face_normals[fi])
        self.method = method

    def run(self, P, T0, sensor=None, iters=40, d_max=0.02):
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        est = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if self.method == "point_to_plane"
               else o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # register world points onto the tote model: X maps world -> tote, so T = X^-1
        res = o3d.pipelines.registration.registration_icp(
            src, self.target, d_max, np.linalg.inv(T0), est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        return dict(T=np.linalg.inv(res.transformation), hist=[], H=None, inlier_rms=res.inlier_rmse)


def truncated_cost(icp, P, T, sensor, g=0.01):
    """Truncated-quadratic point-to-mesh cost: mean(min(r^2, g^2)). Points far from the model cost g^2 each,
    so a pose that explains more points wins; used to pick among ICP hypotheses."""
    R, t = T[:3, :3], T[:3, 3]
    q = (P - t) @ R
    sc = icp._visible_scene(R.T @ (np.asarray(sensor, float) - t))
    d = sc.compute_distance(o3d.core.Tensor(q.astype(np.float32))).numpy()
    return float(np.mean(np.minimum(d ** 2, g ** 2)))


def crop_to_model(icp, P, T, pad=0.01):
    lo = icp.V.min(0) - pad; hi = icp.V.max(0) + pad
    q = (P - T[:3, 3]) @ T[:3, :3]
    return np.all((q > lo) & (q < hi), 1)


def multistart(icp, P, T0, sensor, offsets=None, iters=40):
    """Run ICP from a small grid of starts (in-plane offsets in the tote frame), keep the one with the lowest
    truncated cost, then re-crop the cloud to the fitted model's bounding box and polish."""
    if offsets is None:
        offsets = [(dx, 0.0, dz) for dx in (-0.03, 0.0, 0.03) for dz in (-0.03, -0.015, 0.0, 0.015, 0.03)]
    best = None
    for o in offsets:
        Ts = T0.copy(); Ts[:3, 3] = T0[:3, 3] + T0[:3, :3] @ np.asarray(o)
        r = icp.run(P, Ts, sensor, iters=iters)
        c = truncated_cost(icp, P, r["T"], sensor)
        if best is None or c < best[0]:
            best = (c, r, o)
    c, r, o = best
    m = crop_to_model(icp, P, r["T"])
    r2 = icp.run(P[m], r["T"], sensor, iters=iters, d_max=(0.01, 0.006))
    r2.update(cost=c, start=o, n_cropped=int(m.sum()), n_starts=len(offsets))
    return r2
