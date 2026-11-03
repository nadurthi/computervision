"""Classical (non-learned) box segmentation baselines.

C1  rgb_lines       RGB only: Canny + long straight segments as boundaries, watershed, rectangularity filter.
C2  depth_edges     Depth only: depth-jump + normal-crease + dropout boundaries, connected components,
                    watershed growth, plane check.
C3  rgbd_seams      C2, then split coplanar regions along long RGB lines whose cross-profile is a dark
                    *valley* (a seam) rather than a *step* (tape, print, label edges).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .geometry import fill_depth, normals_from_depth, backproject, fit_plane

_fld = cv2.ximgproc.createFastLineDetector(length_threshold=20, distance_threshold=1.414,
                                           canny_th1=40, canny_th2=90, canny_aperture_size=3,
                                           do_merge=True)


def _lines(gray: np.ndarray, min_len: float = 25):
    ls = _fld.detect(gray)
    if ls is None:
        return np.zeros((0, 4), np.float32)
    ls = np.asarray(ls, np.float32).reshape(-1, 4)
    L = np.hypot(ls[:, 2] - ls[:, 0], ls[:, 3] - ls[:, 1])
    return ls[L >= min_len]


def _rectangularity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return float(mask.sum()) / max(w * h, 1)


def _grow(seeds: np.ndarray, region: np.ndarray, elev: np.ndarray) -> np.ndarray:
    return watershed(elev, markers=seeds, mask=region)


def rgb_lines(rgb: np.ndarray, min_area: int = 500) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    bnd = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in _lines(gray):
        cv2.line(bnd, (int(x0), int(y0)), (int(x1), int(y1)), 1, 2)
    inside = bnd == 0
    dist = ndi.distance_transform_edt(inside)
    seeds, n = ndi.label(dist > 4)
    lab = _grow(seeds, np.ones_like(inside), -dist)
    out = np.zeros_like(lab)
    k = 1
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_area or _rectangularity(m) < 0.7 or m.sum() > 0.25 * m.size:
            continue
        out[m] = k; k += 1
    return out


def _depth_boundaries(depth_raw: np.ndarray, jump_rel=0.012, crease_deg=22):
    d = fill_depth(depth_raw)
    n = normals_from_depth(d, smooth=5)
    z = d
    jump = np.zeros(z.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        zz = np.roll(z, (-dy, -dx), (0, 1))
        jump |= np.abs(zz - z) > jump_rel * z + 0.004
    cosang = np.ones(z.shape, np.float32)
    for dy, dx in ((0, 2), (2, 0)):
        nn = np.roll(n, (-dy, -dx), (0, 1))
        cosang = np.minimum(cosang, (n * nn).sum(-1))
    crease = cosang < np.cos(np.deg2rad(crease_deg))
    bnd = jump | crease | (depth_raw <= 0)
    bnd = cv2.dilate(bnd.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return bnd, d, n


def depth_edges(depth_raw: np.ndarray, min_area: int = 300, max_frac: float = 0.25) -> np.ndarray:
    bnd, d, n = _depth_boundaries(depth_raw)
    seeds, nseg = ndi.label(~bnd)
    # grow seeds back over the boundary band using depth as elevation (stays on the near side)
    gy, gx = np.gradient(d)
    lab = _grow(seeds, np.ones_like(bnd), np.hypot(gx, gy))
    return _filter_planar(lab, d, min_area, max_frac)


def _filter_planar(lab, d, min_area, max_frac, min_inlier=0.6):
    P = backproject(d)
    out = np.zeros_like(lab)
    k = 1
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    for i, c in zip(ids, counts):
        if c < min_area or c > max_frac * lab.size:
            continue
        m = lab == i
        pts = P[m]
        if len(pts) > 1500:
            pts = pts[np.random.default_rng(int(i)).choice(len(pts), 1500, replace=False)]
        fit = fit_plane(pts, iters=30, thr=0.01)
        if fit is None or fit[2].mean() < min_inlier:
            continue
        out[m] = k; k += 1
    return out


def _seam_score(gray: np.ndarray, line, off: int = 4) -> float:
    """Valley depth across a line: mean(sides) - mean(centre). Seams are dark valleys;
    tape and print edges are steps (one side bright, one side dark) and score ~0 or negative."""
    x0, y0, x1, y1 = line
    L = np.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return 0.0
    t = np.linspace(0.1, 0.9, max(int(L / 2), 5))
    xs = x0 + (x1 - x0) * t; ys = y0 + (y1 - y0) * t
    nx, ny = -(y1 - y0) / L, (x1 - x0) / L
    g = gray.astype(np.float32)

    def samp(o):
        return ndi.map_coordinates(g, [ys + ny * o, xs + nx * o], order=1, mode="nearest")
    c = np.minimum(samp(0), np.minimum(samp(-1), samp(1)))
    a = samp(-off); b = samp(off)
    valley = np.minimum(a, b) - c          # both sides brighter than the centre
    step = np.abs(a - b)
    return float(np.median(valley - 0.5 * step))


def seam_lines(rgb: np.ndarray, valley_thr: float = 8.0, min_len: float = 30):
    """Long, thin, dark lines. A morphological black-hat keeps dark structures narrower than the
    kernel (seams, ~1-4 px) and suppresses steps (tape and print edges) and wide dark areas."""
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    bh8 = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    out = []
    for x0, y0, x1, y1 in _lines(bh8, min_len=min_len):
        L = np.hypot(x1 - x0, y1 - y0)
        t = np.linspace(0, 1, max(int(L), 5))
        xs = np.clip(x0 + (x1 - x0) * t, 0, gray.shape[1] - 1); ys = np.clip(y0 + (y1 - y0) * t, 0, gray.shape[0] - 1)
        v = ndi.maximum_filter(bh.astype(np.float32), 3)[ys.astype(int), xs.astype(int)]
        if np.median(v) > valley_thr:
            out.append((x0, y0, x1, y1))
    return out, bh


def rgbd_seams(rgb: np.ndarray, depth_raw: np.ndarray, min_area: int = 300, max_frac: float = 0.25,
               valley_thr: float = 8.0) -> np.ndarray:
    lab = depth_edges(depth_raw, min_area=min_area, max_frac=0.6)
    lines, _ = seam_lines(rgb, valley_thr)
    seam = np.zeros(lab.shape, np.uint8)
    for x0, y0, x1, y1 in lines:
        dx, dy = x1 - x0, y1 - y0; L = np.hypot(dx, dy)
        e = 6 / max(L, 1)
        cv2.line(seam, (int(x0 - dx * e), int(y0 - dy * e)), (int(x1 + dx * e), int(y1 + dy * e)), 1, 2)
    d = fill_depth(depth_raw)
    out = np.zeros_like(lab)
    k = 1
    for i in np.unique(lab[lab > 0]):
        m = lab == i
        sub = m & (seam == 0)
        parts, npart = ndi.label(sub)
        sizes = ndi.sum(sub, parts, range(1, npart + 1)) if npart else []
        good = [j + 1 for j, s in enumerate(sizes) if s >= min_area]
        if len(good) <= 1:
            out[m] = k; k += 1
            continue
        seeds = np.zeros_like(parts)
        for j, g in enumerate(good, start=1):
            seeds[parts == g] = j
        grown = watershed(np.zeros(m.shape, np.float32), markers=seeds, mask=m)
        for j in range(1, len(good) + 1):
            out[grown == j] = k; k += 1
    return _filter_planar(out, d, min_area, max_frac)
