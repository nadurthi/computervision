"""Depth-map fusion into a point cloud with the standard geometric-consistency check (MVSNet / PatchmatchNet style),
applied identically to every method so point-cloud scores compare the depth maps and not the fusion recipe.

A reference depth is kept when at least `min_views` source depth maps agree with it:
reprojection error < pix_thresh pixels and relative depth difference < depth_thresh. The kept point is the average
of the reference point and the agreeing back-projected source points.
"""
import cv2
import numpy as np


def _reproject(d_ref, K_ref, E_ref, d_src, K_src, E_src):
    h, w = d_ref.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xyz_ref = np.linalg.inv(K_ref) @ np.vstack([u.ravel(), v.ravel(), np.ones(h * w)]) * d_ref.ravel()
    T = E_src @ np.linalg.inv(E_ref)
    xyz_src = T[:3, :3] @ xyz_ref + T[:3, 3:4]
    p = K_src @ xyz_src
    xs = (p[0] / np.maximum(p[2], 1e-8)).reshape(h, w).astype(np.float32)
    ys = (p[1] / np.maximum(p[2], 1e-8)).reshape(h, w).astype(np.float32)
    ds = cv2.remap(d_src.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderValue=0)
    xyz_s = np.linalg.inv(K_src) @ np.vstack([xs.ravel(), ys.ravel(), np.ones(h * w)]) * ds.ravel()
    Ti = np.linalg.inv(T)
    xyz_back = Ti[:3, :3] @ xyz_s + Ti[:3, 3:4]
    pb = K_ref @ xyz_back
    xb = (pb[0] / np.maximum(pb[2], 1e-8)).reshape(h, w)
    yb = (pb[1] / np.maximum(pb[2], 1e-8)).reshape(h, w)
    return xyz_back[2].reshape(h, w), xb, yb, ds


def fuse(depths, Ks, Es, images, pairs, min_views=3, pix_thresh=1.0, depth_thresh=0.01, n_src=10):
    """depths/Ks/Es/images: dicts keyed by view id (same resolution). Returns (points N,3 world, colors N,3 uint8)."""
    pts, cols = [], []
    for ref, srcs in pairs:
        if ref not in depths:
            continue
        d = depths[ref]
        h, w = d.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        count = np.zeros((h, w), np.int32)
        acc = d.copy()
        for s in [s for s in srcs if s in depths][:n_src]:
            db, xb, yb, _ = _reproject(d, Ks[ref], Es[ref], depths[s], Ks[s], Es[s])
            ok = (d > 0) & (db > 0) & (np.hypot(xb - u, yb - v) < pix_thresh) & (np.abs(db - d) / np.maximum(d, 1e-8) < depth_thresh)
            count += ok
            acc += np.where(ok, db, 0)
        keep = (d > 0) & (count >= min_views)
        z = (acc / (count + 1))[keep]
        rays = np.linalg.inv(Ks[ref]) @ np.vstack([u[keep], v[keep], np.ones(keep.sum())])
        Xc = rays * z
        Ei = np.linalg.inv(Es[ref])
        pts.append((Ei[:3, :3] @ Xc + Ei[:3, 3:4]).T)
        cols.append(images[ref][keep])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    return np.concatenate(pts), np.concatenate(cols)


def write_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        rec.tofile(f)
