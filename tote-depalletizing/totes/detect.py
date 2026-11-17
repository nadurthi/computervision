"""Tote detection in a lidar point cloud, using the CAD model as a matched filter.

1. Remove the floor (RANSAC plane with a vertical normal).
2. Fit the dominant vertical plane that faces the sensor; its normal n gives the stack's facing direction.
3. Express points in a plane frame (u along the face, v up, s = depth away from the sensor) and split them into
   depth slabs (front row, or a back row exposed after picks).
4. Per slab, rasterise the front-most depth s(u, v) at 1 cm and correlate it with the CAD's front-face depth
   relief T(qx, qz) (normalised cross-correlation). Peaks are tote hypotheses: (u0, v0) -> tote origin.
5. Each hypothesis gives an initial pose and the points that belong to it; ICP (icp.py) refines it.
"""
import numpy as np
import open3d as o3d
from scipy import ndimage, signal


def ransac_plane(P, thresh, iters=400, normal_ok=None, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_n, best_d = None, None, None
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if normal_ok is not None and not normal_ok(n):
            continue
        d = -n @ P[i[0]]
        inl = np.abs(P @ n + d) < thresh
        if best is None or inl.sum() > best.sum():
            best, best_n, best_d = inl, n, d
    A = P[best]; c = A.mean(0)                                  # least-squares refit on inliers
    n = np.linalg.svd(A - c, full_matrices=False)[2][-1]
    return n, -n @ c, np.abs(P @ n - n @ c) < thresh


def plane_frame(n_toward_sensor):
    """Axes of the tote frame implied by the face normal: y_t = away from sensor (horizontal), z up, x = y x z."""
    y = -np.array([n_toward_sensor[0], n_toward_sensor[1], 0.0]); y /= np.linalg.norm(y)
    z = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)                               # columns: tote x, y, z in world


def cad_template(mesh, res=0.01, margin=0.03, far=0.02):
    """Front-face depth relief of the CAD: T[iz, ix] = front-most y of the mesh at (x, z), tote frame.
    Cells in a margin around the tote get `far` behind the face: that is where a neighbour's tapered wall
    (set back ~2 cm) or empty space shows, so the seams between totes are part of the signature."""
    lo, hi = mesh.bounds
    xs = np.arange(lo[0] - margin, hi[0] + margin + 1e-9, res)
    zs = np.arange(lo[2] - margin, hi[2] + margin + 1e-9, res)
    X, Z = np.meshgrid(xs, zs)
    o = np.stack([X.ravel(), np.full(X.size, lo[1] - 1.0), Z.ravel()], 1)
    d = np.tile([0.0, 1.0, 0.0], (len(o), 1))
    sc = o3d.t.geometry.RaycastingScene()
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, np.float32))
    m.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, np.int32))
    sc.add_triangles(m)
    t = sc.cast_rays(o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32)))["t_hit"].numpy()
    y = (lo[1] - 1.0) + t
    face_y = np.nanmedian(y[np.isfinite(y)])
    y = np.where(np.isfinite(y), y, face_y + far).reshape(X.shape)
    return dict(T=y, x0=xs[0], z0=zs[0], res=res, face_y=face_y)


def rasterise(u, v, s, res, u0, v0, nu, nv, fill_dist=0.03, far=None):
    """Front-most depth per cell; sparse rows filled from neighbours within fill_dist; remaining holes = far."""
    iu = np.floor((u - u0) / res).astype(int); iv = np.floor((v - v0) / res).astype(int)
    ok = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    img = np.full((nv, nu), np.inf)
    np.minimum.at(img, (iv[ok], iu[ok]), s[ok])
    hole = ~np.isfinite(img)
    dist, (ii, jj) = ndimage.distance_transform_edt(hole, return_indices=True)
    filled = img[ii, jj]
    filled[dist * res > fill_dist] = np.nan
    if far is not None:
        filled = np.where(np.isfinite(filled), filled, far)
    return filled, hole


def ncc_map(img, tpl):
    """Zero-mean normalised cross-correlation of tpl over img ('valid' placements), via FFT."""
    t = tpl - tpl.mean(); tn = np.sqrt((t ** 2).sum())
    ones = np.ones_like(tpl)
    s1 = signal.fftconvolve(img, ones[::-1, ::-1], mode="valid")
    s2 = signal.fftconvolve(img ** 2, ones[::-1, ::-1], mode="valid")
    num = signal.fftconvolve(img, t[::-1, ::-1], mode="valid")
    var = np.maximum(s2 - s1 ** 2 / tpl.size, 1e-12)
    return num / (np.sqrt(var) * tn)


def cad_profile(mesh):
    """Numbers the detector needs from the CAD: height H, rim width W, front-most y of the rim (tote frame)."""
    V = np.asarray(mesh.vertices)
    lo, hi = V.min(0), V.max(0)
    top = V[:, 2] > hi[2] - 0.01
    return dict(H=hi[2] - lo[2], W=V[top, 0].max() - V[top, 0].min(), y_rim=V[top, 1].min(), z0=lo[2])


def seam_heights(v, s, res=0.01, jump=0.008, min_sep=0.15):
    """Rim tops of each layer. Every tote tapers inward toward its base, so going up the face the depth grows
    slowly and then jumps back by ~2 cm where the next tote's base starts (or where the stack ends).
    Peaks of the upward depth jump of the per-row median depth are the rim tops."""
    iv = np.floor((v - v.min()) / res).astype(int)
    nv = iv.max() + 1
    med = np.full(nv, np.nan)
    for k in range(nv):
        sk = s[iv == k]
        if len(sk) >= 5:
            med[k] = np.median(sk)
    rows = np.nonzero(np.isfinite(med))[0]
    medf = np.interp(np.arange(nv), rows, med[rows])
    occ = np.isfinite(med)
    g = np.zeros(nv); g[:-2] = medf[2:] - medf[:-2]               # depth(v + res) - depth(v - res)
    tops = list(signal.find_peaks(g, height=jump, distance=int(min_sep / res))[0] + 1)
    top_row = rows.max() + 1                                      # the top of the stack is also a rim top
    tops = [t for t in tops if top_row - t > min_sep / res] + [top_row]
    return v.min() + np.array(tops) * res, dict(med=med, grad=g, v0=v.min(), res=res)


def segments_1d(x, gap=0.03, res=0.005):
    """Contiguous runs of occupied bins along one axis."""
    if len(x) == 0:
        return []
    b = np.unique(np.floor(x / res).astype(int))
    cuts = np.nonzero(np.diff(b) * res > gap)[0]
    starts = np.r_[b[0], b[cuts + 1]]; ends = np.r_[b[cuts], b[-1]]
    out = []
    for a, e in zip(starts, ends):
        xs = x[(x >= a * res) & (x < (e + 1) * res)]
        out.append((np.percentile(xs, 0.5), np.percentile(xs, 99.5), len(xs)))
    return out


def detect(P, mesh, sensor, slab_halfwidth=0.07, min_points=150, debug=None):
    """Returns a list of tote hypotheses: dict(T0 = initial tote->world pose, points = cloud for ICP, idx)."""
    sensor = np.asarray(sensor, float)
    cad = cad_profile(mesh)
    # 1. floor
    n_f, d_f, inl_f = ransac_plane(P, 0.02, normal_ok=lambda n: abs(n[2]) > 0.95)
    keep = ~inl_f & ((P @ n_f + d_f) * np.sign(n_f[2]) > 0.02)
    qidx = np.nonzero(keep)[0]
    Q = P[keep]
    # 2. dominant vertical plane facing the sensor -> tote axes
    n, d, _ = ransac_plane(Q, 0.03, normal_ok=lambda n: abs(n[2]) < 0.2)
    if n @ (sensor - Q.mean(0)) < 0:
        n, d = -n, -d
    Rf = plane_frame(n)
    u, s, v = (Q @ Rf).T
    # 3. depth slabs (front row; back rows exposed by earlier picks)
    h, e = np.histogram(s, bins=np.arange(s.min(), s.max() + 0.02, 0.02))
    pk = signal.find_peaks(np.r_[0, h, 0], height=min_points, distance=int(0.2 / 0.02))[0] - 1
    slabs = [0.5 * (e[i] + e[i + 1]) for i in pk]
    dets, dbg_slabs = [], []
    for sc_ in slabs:
        m = np.abs(s - sc_) < slab_halfwidth
        # lateral connected pieces of this slab first (a missing tote splits a layer)
        for ua, ub, _ in segments_1d(u[m], gap=0.1):
            mm = m & (u >= ua - 0.01) & (u <= ub + 0.01)
            if mm.sum() < min_points // 2:
                continue
            tops, prof = seam_heights(v[mm], s[mm])
            dbg_slabs.append(dict(slab=sc_, u=(ua, ub), tops=tops, prof=prof))
            for vt in tops:
                vo = vt - cad["H"]                                 # tote origin height (base)
                band = mm & (v > vt - 0.03) & (v < vt + 0.005)      # rim band: the front-most part of the tote
                face = mm & (v > vo) & (v < vt + 0.005)
                if band.sum() < 20:
                    continue
                for a, b, cnt in segments_1d(u[face], gap=0.08):
                    k = max(1, int(round((b - a) / (cad["W"] + 0.003))))
                    w = (b - a) / k
                    for j in range(k):
                        uo = a + (j + 0.5) * w
                        rim = band & (np.abs(u - uo) < 0.4 * w)
                        if rim.sum() < 5:
                            continue
                        so = np.median(s[rim]) - cad["y_rim"]
                        sel = mm & (np.abs(u - uo) < 0.5 * w + 0.01) & (v > vo - 0.01) & (v < vt + 0.005)
                        rows = np.unique(np.floor((v[sel] - vo) / 0.02).astype(int))
                        # the face must span most of the tote height, unless a nearer row hides its lower part
                        # (a back-row tote exposed by a pick, seen over the front row below it)
                        occl = np.any((s < sc_ - 0.2) & (np.abs(u - uo) < 0.5 * w) & (v > vo - cad["H"]) & (v < vt))
                        need = (0.25 if occl else 0.7) * cad["H"] / 0.02
                        if sel.sum() < (min_points // 2 if occl else min_points) or len(rows) < need:
                            continue
                        T0 = np.eye(4); T0[:3, :3] = Rf; T0[:3, 3] = Rf @ np.array([uo, so, vo - cad["z0"]])
                        dets.append(dict(T0=T0, points=Q[sel], idx=qidx[sel], slab=float(sc_), n=int(sel.sum())))
    if debug is not None:
        debug.update(dict(n=n, Rf=Rf, slabs=slabs, Q=Q, u=u, v=v, s=s, segs=dbg_slabs, floor=(n_f, d_f)))
    return dets
