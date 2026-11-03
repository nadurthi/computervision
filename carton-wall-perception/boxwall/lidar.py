"""Lidar box segmentation.

L1  range_image   project the scan to its native (ring x column) range image, cut on range jumps and
                  normal creases, label connected regions, map labels back to points.
L2  ransac_dbscan sequential RANSAC planes (Open3D), each plane's inliers split by DBSCAN so coplanar
                  faces separated by a gap become separate boxes.
L3  camera_to_lidar  paint points with an image instance map (e.g. SeamNet) after a depth-consistency check.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .render import FX, FY, W, H

CX, CY = W / 2 - 0.5, H / 2 - 0.5


def _range_image(L):
    rows, cols = L["shape"]
    R = np.zeros((rows, cols), np.float32)
    P = np.zeros((rows, cols, 3), np.float32)
    R[L["ring"], L["col"]] = L["range"]
    P[L["ring"], L["col"]] = L["points"]
    return R, P


def range_image(L, jump_rel=0.015, crease_deg=80, min_pts=30, max_frac=0.25):
    R, P = _range_image(L)
    valid = R > 0
    # fill isolated dropouts from neighbours (keeps far side, like a seam)
    Rf = R.copy(); Pf = P.copy()
    for _ in range(2):
        miss = Rf <= 0
        if not miss.any():
            break
        g = cv2.dilate(Rf, np.ones((3, 3), np.uint8))
        Rf[miss] = g[miss]
        for c in range(3):
            Pf[..., c][miss] = cv2.dilate(Pf[..., c] * (Rf > 0), np.ones((3, 3), np.uint8))[miss]
    # columns are ~5 mm apart but rows ~3 cm: smooth and difference anisotropically
    Ps = cv2.blur(Pf, (9, 3))
    dx = np.zeros_like(Ps); dy = np.zeros_like(Ps)
    dx[:, 4:-4] = Ps[:, 8:] - Ps[:, :-8]; dy[1:-1] = Ps[2:] - Ps[:-2]
    n = np.cross(dx, dy); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    jump = np.zeros(R.shape, bool)
    Rm = cv2.medianBlur(Rf, 3)
    for dyy, dxx in ((0, 1), (1, 0), (0, 2)):
        rr = np.roll(Rm, (-dyy, -dxx), (0, 1))
        jump |= np.abs(rr - Rm) > jump_rel * Rm + 0.01
    cosang = np.ones(R.shape, np.float32)
    for dyy, dxx in ((0, 6), (2, 0)):
        nn = np.roll(n, (-dyy, -dxx), (0, 1))
        cosang = np.minimum(cosang, np.abs((n * nn).sum(-1)))
    bnd = jump | (cosang < np.cos(np.deg2rad(crease_deg))) | (Rf <= 0)
    seeds, nseg = ndi.label(~bnd)
    seeds = cv2.morphologyEx((seeds > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 5), np.uint8)) * seeds
    seeds, _ = ndi.label(seeds > 0)
    lab = watershed(np.zeros(R.shape, np.float32), markers=seeds, mask=Rf > 0)
    ids, cnt = np.unique(lab[valid], return_counts=True)
    total = valid.sum()
    bad = {i for i, c in zip(ids, cnt) if c < min_pts or c > max_frac * total}
    out = lab[L["ring"], L["col"]]
    out[np.isin(out, list(bad))] = 0
    return out


def ransac_dbscan(L, max_planes=80, dist=0.015, eps=0.035, min_pts=30, max_frac=0.25):
    import open3d as o3d
    pts = L["points"].astype(np.float64)
    idx_left = np.arange(len(pts))
    out = np.zeros(len(pts), np.int32)
    k = 1
    for _ in range(max_planes):
        if len(idx_left) < min_pts:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[idx_left]))
        _, inl = pc.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=200)
        inl = np.asarray(inl)
        if len(inl) < min_pts:
            break
        sub = pc.select_by_index(inl)
        cl = np.asarray(sub.cluster_dbscan(eps=eps, min_points=8))
        for c in np.unique(cl[cl >= 0]):
            m = cl == c
            if min_pts <= m.sum() <= max_frac * len(pts):
                out[idx_left[inl[m]]] = k; k += 1
        idx_left = np.delete(idx_left, inl)
    return out


def camera_to_lidar(L, img_lab: np.ndarray, depth: np.ndarray, tol=0.04):
    p = L["points"]
    z = p[:, 2]
    u = np.round(p[:, 0] / np.maximum(z, 1e-3) * FX + CX).astype(int)
    v = np.round(p[:, 1] / np.maximum(z, 1e-3) * FY + CY).astype(int)
    ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.zeros(len(p), np.int32)
    uu, vv = u[ok], v[ok]
    dz = depth[vv, uu]
    vis = (dz > 0) & (np.abs(dz - z[ok]) < tol)          # point is the surface the camera saw
    lab = img_lab[vv, uu]
    tmp = np.zeros(ok.sum(), np.int32); tmp[vis] = lab[vis]
    out[ok] = tmp
    return out
