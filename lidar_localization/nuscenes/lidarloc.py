"""LIDAR-only localization: particle filter + lookup-table likelihood + multi-resolution BEV scan-to-map matching.

NumPy/SciPy port of the method in N. Adurthi, "Scan-Matching Based Particle Filtering Approach for LIDAR-Only
Localization" (arXiv:2302.06843) and of the original KITTI code in ../drivers/main_kitti_filter_localize.py and
../lidarprocessing/numba_codes. The original needs numba, open3d, PCL and the pyslam C++ module; this port needs only
numpy + scipy. Each function names the original it mirrors.
"""
import heapq
import time
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------------- geometry
def voxel_down(X, voxel):
    """open3d voxel_down_sample equivalent: centroid of the points in each occupied voxel."""
    if len(X) == 0:
        return X
    key = np.floor(X / voxel).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((cnt.size, X.shape[1]))
    np.add.at(out, inv, X)
    return out / cnt[:, None]


def Rzyx(yaw, pitch, roll):
    """R = Rz(yaw) Ry(pitch) Rx(roll), as pose2Rt / gicp.Rz.Ry.Rx in the original. Vectorised over arrays."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    R = np.empty(np.shape(yaw) + (3, 3))
    R[..., 0, 0] = cy * cp; R[..., 0, 1] = cy * sp * sr - sy * cr; R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp; R[..., 1, 1] = sy * sp * sr + cy * cr; R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp;     R[..., 2, 1] = cp * sr;                R[..., 2, 2] = cp * cr
    return R


def R2ypr(R):
    return Rotation.from_matrix(R).as_euler("zyx")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------- likelihood (paper §II.B, pyslam computeLikelihood_lookup)
class DistanceLookup:
    """Pre-computed nearest-distance-to-map tensor L(ix,iy,iz), truncated at dmax (paper's sparse 3-D lookup table).

    Stored densely as uint8 in units of `res` (exact Euclidean distance transform of the map occupancy grid);
    lookups outside the box or beyond dmax return dmax."""

    def __init__(self, Xmap, res=0.25, dmax=10.0, pad=None):
        pad = dmax if pad is None else pad
        self.res, self.dmax = res, dmax
        self.lb = Xmap.min(0) - pad
        ub = Xmap.max(0) + pad
        self.shape = np.ceil((ub - self.lb) / res).astype(int) + 1
        occ = np.ones(self.shape, dtype=bool)
        idx = np.floor((Xmap - self.lb) / res).astype(int)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = False
        d = ndimage.distance_transform_edt(occ, sampling=res, return_indices=False).astype(np.float32)
        # distance from the voxel centre's grid node, stored in cm (uint16); 65535 marks "beyond dmax"
        q = np.minimum(np.round(d * 100), 65534).astype(np.uint16)
        q[d >= dmax] = 65535
        self.L = q
        self.mb = self.L.nbytes / 1e6

    def dist(self, P):
        idx = np.floor((P - self.lb) / self.res).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=-1)
        out = np.full(P.shape[:-1], self.dmax)
        q = self.L[idx[ok][:, 0], idx[ok][:, 1], idx[ok][:, 2]]
        out[ok] = np.where(q == 65535, self.dmax, q * 0.01)
        return out

    def neg_log_lik(self, poses, Xs, sig0):
        """-log p(y|x_i) = sum_j min(d_j,dmax)^2 / sigma^2,  sigma^2 = sig0^2 * N_s  (as pyslam computeLikelihood)."""
        R = Rzyx(poses[:, 3], poses[:, 4], poses[:, 5])                       # (Np,3,3)
        P = np.einsum("nij,mj->nmi", R, Xs) + poses[:, None, :3]              # (Np,Ns,3)
        d = np.minimum(self.dist(P), self.dmax)
        return np.sum(d * d, axis=1) / (sig0 ** 2 * Xs.shape[0])


# ----------------------------------------------------------------------------- dynamics (dynmodelCT3D) and PF
def dynmodelCT3D(X, dt):
    """x=[x,y,z,yaw,pitch,roll,v,wyaw,wpitch,wroll,a]: constant turn-rate / acceleration along the body x axis."""
    R = Rzyx(X[:, 3], X[:, 4], X[:, 5])
    drn = R[:, :, 0]
    X = X.copy()
    X[:, 0:3] += (X[:, 6:7] * dt + 0.5 * X[:, 10:11] * dt ** 2) * drn
    X[:, 6] += X[:, 10] * dt
    X[:, 3:6] += X[:, 7:10] * dt
    return X


Q_CT = np.diag([0.1 ** 2, 0.1 ** 2, 0.1 ** 2,
                np.deg2rad(2) ** 2, np.deg2rad(2) ** 2, np.deg2rad(0.1) ** 2,
                0.2 ** 2,
                np.deg2rad(0.05) ** 2, np.deg2rad(0.05) ** 2, np.deg2rad(0.02) ** 2,
                0.01 ** 2])                                                    # getCTinitialSamples Q


class ParticleFilter:
    """uq.filters.pf.Particles: bootstrap SIS with Neff-triggered systematic resampling."""

    def __init__(self, X, rng):
        self.X = X
        self.logw = np.full(len(X), -np.log(len(X)))
        self.rng = rng

    @property
    def w(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def neff(self):
        return 1.0 / np.sum(self.w ** 2)

    def resample(self):
        n = len(self.X)
        c = np.cumsum(self.w)
        u = (self.rng.random() + np.arange(n)) / n
        self.X = self.X[np.minimum(np.searchsorted(c, u), n - 1)]
        self.logw[:] = -np.log(n)

    def estimate(self):
        w = self.w
        m = w @ self.X
        m[3:6] = np.arctan2(w @ np.sin(self.X[:, 3:6]), w @ np.cos(self.X[:, 3:6]))
        return m


# ----------------------------------------------------------------------------- road segmentation (normals |n_z| < 0.5)
def remove_road(X, radius=1.0, k=20, nz_max=0.5):
    """Keep points whose local surface normal is not vertical (driver: open3d estimate_normals, |n_z|<0.5)."""
    tree = cKDTree(X)
    _, nn = tree.query(X, k=min(k, len(X)), distance_upper_bound=radius)
    nn = np.where(nn >= len(X), np.arange(len(X))[:, None], nn)
    Y = X[nn] - X[nn].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Y, Y)
    _, V = np.linalg.eigh(C)
    nz = np.abs(V[:, 2, 0])          # normal = eigenvector of the smallest eigenvalue
    return X[nz < nz_max]


# ----------------------------------------------------------------------------- multi-resolution BEV matcher (binMatcherAdaptive3)
def _hist2d(X, xedges, yedges):
    H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[xedges, yedges])
    return (H > 0).astype(np.int32)


def _upsample_max(Hup, n=2):
    """UpsampleMax: H[j,k] = max(Hup[2j:2j+n+1, 2k:2k+n+1]) (coarser level is an upper bound)."""
    m = ndimage.maximum_filter(Hup, size=n + 1, origin=-(n // 2), mode="constant")
    return np.ascontiguousarray(m[::2, ::2]).astype(np.int32)


def _point_cost(H, dx, X, O):
    P = np.floor((X + O) / dx).astype(np.int64)
    ok = (P[:, 0] >= 0) & (P[:, 0] < H.shape[0]) & (P[:, 1] >= 0) & (P[:, 1] < H.shape[1])
    return int(H[P[ok, 0], P[ok, 1]].sum())


def _cell_cost(H, P, cnt, off):
    """getPointCost with the scan pre-binned: offsets that are whole cells shift indices instead of re-flooring."""
    i, j = P[:, 0] + off[0], P[:, 1] + off[1]
    ok = (i >= 0) & (i < H.shape[0]) & (j >= 0) & (j < H.shape[1])
    return int(cnt[ok] @ H[i[ok], j[ok]])


def bin_matcher(Xmap2, Xs2, H12, Lmax, thmax, thmin, dxMatch, dxBase=(30.0, 30.0)):
    """Branch-and-bound multi-resolution BEV matching (Olson 2015) of the 2-D scan Xs2 (sensor frame) against map
    points Xmap2, searching absolute positions within +/-Lmax of the guess H12 (3x3, scan->map) and rotations in
    [-thmax, thmax] at step thmin. Boxes are first split down to dxBase, then refined best-first.
    Returns (H12_corrected, cost_at_guess, cost_best). Mirrors binmatchers.binMatcherAdaptive_super."""
    mn_orig = Xmap2.min(0) - dxMatch
    H12mn = H12.copy(); H12mn[0:2, 2] -= mn_orig
    X1 = Xmap2 - mn_orig
    mx = X1.max(0)
    t0 = H12mn[0:2, 2]
    dxs, edges = [], []
    for i in range(100):
        f = 2 ** i
        xe = np.linspace(0, mx[0] + dxMatch[0], f + 1); ye = np.linspace(0, mx[1] + dxMatch[0], f + 1)
        edges.append((xe, ye)); dxs.append(np.array([xe[1] - xe[0], ye[1] - ye[0]]))
        if np.any(dxs[-1] <= dxMatch):
            break
    levels = [_hist2d(X1, *edges[-1])]
    for _ in range(1, len(dxs)):
        levels.append(_upsample_max(levels[-1]))
    levels = levels[::-1]
    top = len(levels) - 1
    bb = (t0[0] - Lmax[0], t0[1] - Lmax[1], t0[0] + Lmax[0], t0[1] + Lmax[1])
    rng = np.random.default_rng(0)
    Xth, heap, cells = {}, [], {}
    for th in np.arange(-thmax, thmax + thmin, thmin):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Xth[th] = (H12mn[0:2, 0:2] @ (R @ Xs2.T)).T                 # rotated, not translated: boxes are absolute
        for lv in range(len(levels)):                                # distinct cells + counts per level (speed-up)
            P, cnt = np.unique(np.floor(Xth[th] / dxs[lv]).astype(np.int64), axis=0, return_counts=True)
            cells[th, lv] = (P, cnt)
        heap.append((-_cell_cost(levels[0], *cells[th, 0], (0, 0)), 0.0, 0.0, dxs[0][0], dxs[0][1], 0, th))
    cost0 = _point_cost(levels[-1], dxs[-1], (H12mn[0:2, 0:2] @ Xs2.T).T + t0, np.zeros(2))

    def children(c, xs0, ys0, d0, d1, lvl, th):
        nl = lvl + 1; dx = dxs[nl]
        for xs in np.arange(xs0, xs0 + d0, dx[0])[:2]:
            for ys in np.arange(ys0, ys0 + d1, dx[1])[:2]:
                if min(bb[2], xs + dx[0]) < max(bb[0], xs) or min(bb[3], ys + dx[1]) < max(bb[1], ys):
                    continue
                c3 = _cell_cost(levels[nl], *cells[th, nl], (int(round(xs / dx[0])), int(round(ys / dx[1]))))
                yield (-c3 - rng.random() / 1e10, xs, ys, dx[0], dx[1], nl, th)

    while any(h[3] > dxBase[0] or h[4] > dxBase[1] for h in heap):      # breadth expansion to dxBase
        heap = [k for h in heap for k in ((children(*h)) if (h[3] > dxBase[0] or h[4] > dxBase[1]) else (h,))]
    heapq.heapify(heap)
    while True:
        h = heapq.heappop(heap)
        if h[5] == top:
            break
        for k in children(*h):
            heapq.heappush(heap, k)
    c, xs0, ys0, d0, d1, lvl, th = h
    HR = np.eye(3); HR[0:2, 0:2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    Ht = np.eye(3); Ht[0:2, 2] = np.array([xs0, ys0]) - t0
    H = Ht @ H12mn @ HR
    H[0:2, 2] += mn_orig
    return H, cost0, float(np.floor(-c))


# ----------------------------------------------------------------------------- ICP (replaces PCL IterativeClosestPoint)
def icp_point_to_plane(src, dst_tree, dst, dst_normals, H0, iters=20, max_corr=2.0):
    H = H0.copy()
    for _ in range(iters):
        P = src @ H[:3, :3].T + H[:3, 3]
        d, j = dst_tree.query(P, distance_upper_bound=max_corr)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        p, q, n = P[ok], dst[j[ok]], dst_normals[j[ok]]
        A = np.hstack([np.cross(p, n), n]); b = np.sum((q - p) * n, 1)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        dH = np.eye(4); dH[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix(); dH[:3, 3] = x[3:]
        H = dH @ H
        if np.linalg.norm(x) < 1e-4:
            break
    return H


def normals(X, k=12):
    _, nn = cKDTree(X).query(X, k=k)
    Y = X[nn] - X[nn].mean(1, keepdims=True)
    _, V = np.linalg.eigh(np.einsum("nki,nkj->nij", Y, Y))
    return V[:, :, 0]


def pose_to_H(x):
    H = np.eye(4); H[:3, :3] = Rzyx(x[3], x[4], x[5]); H[:3, 3] = x[:3]
    return H


def H_to_pose(H):
    y, p, r = R2ypr(H[:3, :3])
    return np.array([*H[:3, 3], y, p, r])
