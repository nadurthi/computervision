"""Depth-image geometry shared by all methods: hole filling, back-projection, normals, face fitting."""
from __future__ import annotations

import cv2
import numpy as np

from .render import FX, FY, W, H

CX, CY = W / 2 - 0.5, H / 2 - 0.5


def fill_depth(depth: np.ndarray) -> np.ndarray:
    """Fill dropouts with the nearest valid neighbour (max of a growing window keeps the far side,
    which is what a seam really is)."""
    d = depth.copy()
    for k in (3, 5, 9, 17):
        miss = d <= 0
        if not miss.any():
            break
        grown = cv2.dilate(d, np.ones((k, k), np.uint8))
        d[miss] = grown[miss]
    return d


def backproject(depth: np.ndarray) -> np.ndarray:
    v, u = np.indices(depth.shape, dtype=np.float32)
    return np.stack([(u - CX) / FX * depth, (v - CY) / FY * depth, depth], -1)


def normals_from_depth(depth: np.ndarray, smooth: int = 5) -> np.ndarray:
    P = backproject(depth)
    if smooth > 1:
        P = cv2.blur(P, (smooth, smooth))
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n[n[..., 2] > 0] *= -1                       # face the camera
    return n.astype(np.float32)


def fit_plane(pts: np.ndarray, iters: int = 60, thr: float = 0.008, rng=None):
    """RANSAC + SVD refit. Returns (normal, d, inlier mask); normal faces the camera."""
    rng = rng or np.random.default_rng(0)
    n_pts = len(pts)
    if n_pts < 10:
        return None
    best = None
    for _ in range(iters):
        s = pts[rng.choice(n_pts, 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        inl = np.abs((pts - s[0]) @ nrm) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 10:
        return None
    q = pts[best]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    nrm = vt[2]
    if nrm @ c > 0:
        nrm = -nrm
    return nrm, -nrm @ c, best


def face_from_mask(depth: np.ndarray, mask: np.ndarray, rng=None):
    """Fit the dominant plane of a mask and a min-area rectangle inside it.
    Returns dict(normal, center, width, height) in metres, or None."""
    m = mask & (depth > 0)
    if m.sum() < 60:
        return None
    pts = backproject(depth)[m]
    if len(pts) > 4000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 4000, replace=False)]
    fit = fit_plane(pts, rng=rng)
    if fit is None:
        return None
    nrm, d, inl = fit
    q = pts[inl]
    # in-plane basis: e1 ~ camera x projected onto the plane
    e1 = np.array([1.0, 0, 0]) - nrm * nrm[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    c0 = q.mean(0)
    uv = np.stack([(q - c0) @ e1, (q - c0) @ e2], 1).astype(np.float32)
    # Orientation from the min-area rectangle, extents from the 1st-99th percentiles along its axes, so a few
    # stray pixels leaked from a neighbouring face don't stretch the rectangle.
    _, _, ang = cv2.minAreaRect(uv)
    a = np.deg2rad(ang)
    if abs(np.cos(a)) < abs(np.sin(a)):        # keep the first axis the more horizontal one
        a += np.pi / 2
    ax1 = np.array([np.cos(a), np.sin(a)]); ax2 = np.array([-np.sin(a), np.cos(a)])
    s1, s2 = uv @ ax1, uv @ ax2
    lo1, hi1 = np.percentile(s1, [1, 99]); lo2, hi2 = np.percentile(s2, [1, 99])
    w, h = hi1 - lo1, hi2 - lo2
    cu, cv = (lo1 + hi1) / 2 * ax1 + (lo2 + hi2) / 2 * ax2
    center = c0 + cu * e1 + cv * e2
    return {"normal": nrm, "center": center, "width": float(w), "height": float(h), "inlier_frac": float(inl.mean()),
            "axes": (e1 * ax1[0] + e2 * ax1[1], e1 * ax2[0] + e2 * ax2[1])}
