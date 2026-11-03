"""Vectorised ray casting of box-wall scenes into RGB-D images and lidar scans."""
from __future__ import annotations

import numpy as np
import cv2

from .scene import Scene, FLOOR_Y, CEIL_Y, HALF_W

W, H = 480, 360
FX = FY = 380.0
CX, CY = W / 2 - 0.5, H / 2 - 0.5
LIGHT = np.array([0.08, -0.25, 0.0])


def camera_rays(w=W, h=H, fx=FX, fy=FY, cx=None, cy=None):
    cx = (w / 2 - 0.5) if cx is None else cx
    cy = (h / 2 - 0.5) if cy is None else cy
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
    return d  # not normalised: t equals the z depth for camera rays


def _cast(origin, dirs, scene: Scene):
    """Returns t (ray parameter), instance id (0 = trailer structure), local hit info."""
    n = dirs.shape[0]
    t_best = np.full(n, np.inf, np.float32)
    inst = np.zeros(n, np.int32)
    axis = np.full(n, -1, np.int8)
    sign = np.zeros(n, np.int8)
    for i, b in enumerate(scene.boxes, start=1):
        o = (b.R.T @ (origin - b.center)).astype(np.float32)
        d = dirs @ b.R.astype(np.float32)            # == (R^T d^T)^T
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (-b.half.astype(np.float32) - o) * inv
            t2 = (b.half.astype(np.float32) - o) * inv
        tlo = np.minimum(t1, t2)
        thi = np.maximum(t1, t2)
        tmin = tlo.max(1)
        tmax = thi.min(1)
        hit = (tmax >= tmin) & (tmin > 1e-4) & (tmin < t_best)
        if not hit.any():
            continue
        ax = tlo[hit].argmax(1).astype(np.int8)
        t_best[hit] = tmin[hit]
        inst[hit] = i
        axis[hit] = ax
        # face sign: the slab entered from the -half side if t1 was the entry
        entry_from_neg = np.take_along_axis(t1[hit], ax[:, None].astype(np.int64), 1)[:, 0] <= \
            np.take_along_axis(t2[hit], ax[:, None].astype(np.int64), 1)[:, 0]
        sign[hit] = np.where(entry_from_neg, -1, 1)
    # trailer: floor, ceiling, side walls, far bulkhead
    planes = [((0, FLOOR_Y, 0), (0, -1, 0)), ((0, -CEIL_Y, 0), (0, 1, 0)),
              ((-HALF_W, 0, 0), (1, 0, 0)), ((HALF_W, 0, 0), (-1, 0, 0)),
              ((0, 0, scene.meta.get("z_wall", 2.0) + 2.2), (0, 0, -1))]
    struct_n = np.zeros((n, 3), np.float32)
    for p0, nrm in planes:
        p0 = np.array(p0, np.float32); nrm = np.array(nrm, np.float32)
        den = dirs @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p0 - origin) @ nrm) / den
        hit = (t > 1e-4) & (t < t_best) & (den < 0)
        t_best[hit] = t[hit]
        inst[hit] = 0
        axis[hit] = -1
        struct_n[hit] = nrm
    return t_best, inst, axis, sign, struct_n


def _hash(a, b, c):
    x = np.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453
    return x - np.floor(x)


def render_rgbd(scene: Scene, rng: np.random.Generator | None = None, noise=True):
    rng = rng or np.random.default_rng(scene.seed + 7)
    origin = np.zeros(3, np.float32)
    dirs = camera_rays()
    t, inst, axis, sign, struct_n = _cast(origin, dirs, scene)
    pts = dirs * t[:, None]
    depth = pts[:, 2].copy()

    rgb = np.zeros((dirs.shape[0], 3), np.float32)
    normal = np.zeros((dirs.shape[0], 3), np.float32)
    front = np.zeros(dirs.shape[0], bool)
    seam_k = rng.uniform(0.0, 0.45)                 # how dark the box edges look (0 = invisible seams)
    for i, b in enumerate(scene.boxes, start=1):
        m = inst == i
        if not m.any():
            continue
        sku = scene.skus[b.sku]
        loc = (pts[m] - b.center) @ b.R                 # local coords
        ax = axis[m].astype(np.int64); sg = sign[m]
        nl = np.zeros((m.sum(), 3), np.float32)
        nl[np.arange(m.sum()), ax] = sg
        normal[m] = nl @ b.R.T
        # uv on the face
        other = np.array([[1, 2], [0, 2], [0, 1]])[ax]
        a0 = np.take_along_axis(loc, other[:, :1], 1)[:, 0]
        a1 = np.take_along_axis(loc, other[:, 1:], 1)[:, 0]
        h0 = b.half[other[:, 0]]; h1 = b.half[other[:, 1]]
        u = (a0 + h0) / (2 * h0); v = (a1 + h1) / (2 * h1)
        col = np.tile(sku.base, (m.sum(), 1)).astype(np.float32)
        col *= np.where(ax[:, None] == 2, 1.0, 0.93)
        is_front = (ax == 2) & (sg == -1)
        front[np.where(m)[0][is_front]] = True
        # fibre noise
        q = np.floor(loc / 0.002)
        col *= (0.94 + 0.12 * _hash(q[:, 0], q[:, 1], q[:, 2] + i))[:, None]
        # prints, labels, tape on the front face; tape also on the top
        for (u0, v0, u1, v1, c) in sku.prints:
            sel = is_front & (u > u0) & (u < u1) & (v > v0) & (v < v1)
            col[sel] = c
        if sku.label is not None:
            u0, v0, u1, v1 = sku.label
            sel = is_front & (u > u0) & (u < u1) & (v > v0) & (v < v1)
            col[sel] = 0.93
            bars = sel & (v > (v0 + v1) / 2) & (_hash(np.floor((u - u0) * 120), 3.0, 5.0) > 0.5)
            col[bars] = 0.08
        tw = 0.03 / (2 * b.half[0]), 0.03 / (2 * b.half[1])
        if "h" in sku.tape:
            sel = (is_front | ((ax == 1) & (sg == -1))) & (np.abs(v - 0.5) < (tw[1] if True else 0))
            col[sel] = sku.tape_rgb * (0.95 + 0.1 * _hash(u * 50, v, 1.0))[sel, None]
        if "v" in sku.tape:
            sel = is_front & (np.abs(u - 0.5) < tw[0])
            col[sel] = sku.tape_rgb
        # darkening near face borders (rounded/crushed cardboard edges)
        de = np.minimum.reduce([a0 + h0, h0 - a0, a1 + h1, h1 - a1])
        col *= (1 - seam_k * np.exp(-de / 0.006))[:, None]
        rgb[m] = col
    sm = inst == 0
    normal[sm] = struct_n[sm]
    wall_col = np.array([0.55, 0.56, 0.58], np.float32)
    rgb[sm] = wall_col * (0.9 + 0.2 * _hash(np.floor(pts[sm, 0] / 0.05), np.floor(pts[sm, 1] / 0.05), np.floor(pts[sm, 2] / 0.05)))[:, None]

    # point light near the sensor + ambient
    Lv = LIGHT - pts
    r = np.linalg.norm(Lv, axis=1) + 1e-6
    lam = np.clip((normal * (Lv / r[:, None])).sum(1), 0, 1)
    shade = scene.light_gain * lam * (2.0 / r) ** 2 * 0.95 + 0.12
    img = rgb * shade[:, None]

    img = img.reshape(H, W, 3)
    depth = depth.reshape(H, W).astype(np.float32)
    inst = inst.reshape(H, W)
    normal = normal.reshape(H, W, 3)
    front = front.reshape(H, W)
    clean_depth = depth.copy()
    if noise:
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        img = img + rng.normal(0, 0.015, img.shape).astype(np.float32)
        depth = _depth_noise(depth, rng)
    img8 = (np.clip(img, 0, 1) ** (1 / 1.8) * 255).astype(np.uint8)
    return {"rgb": img8, "depth": depth, "depth_gt": clean_depth, "inst": inst,
            "normal": normal, "front": front}


def _depth_noise(depth, rng):
    d = depth + rng.normal(0, 1, depth.shape).astype(np.float32) * (0.0012 * depth ** 2)
    gy, gx = np.gradient(depth)
    disc = np.hypot(gx, gy) > 0.03
    disc = cv2.dilate(disc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    drop = (disc & (rng.random(depth.shape) < 0.5)) | (rng.random(depth.shape) < 0.004)
    d[drop] = 0.0
    return d


def render_lidar(scene: Scene, rng: np.random.Generator | None = None,
                 rows=80, cols=640, az_fov=(-48, 48), el_fov=(-38, 30),
                 origin=(0.0, -0.35, -0.10), noise=0.008):
    """Scanning lidar mounted above the camera. Returns points (camera frame) and labels."""
    rng = rng or np.random.default_rng(scene.seed + 11)
    o = np.array(origin, np.float32)
    az = np.deg2rad(np.linspace(*az_fov, cols, dtype=np.float32))
    el = np.deg2rad(np.linspace(el_fov[1], el_fov[0], rows, dtype=np.float32))  # row 0 looks up
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.sin(A), -np.sin(E), np.cos(E) * np.cos(A)], -1).reshape(-1, 3)
    t, inst, axis, sign, _ = _cast(o, d.astype(np.float32), scene)
    rng_m = t + rng.normal(0, noise, t.shape).astype(np.float32)
    valid = np.isfinite(t) & (t < 8) & (rng.random(t.shape) > 0.02)
    pts = o + d * rng_m[:, None]
    ring, col = np.divmod(np.arange(rows * cols), cols)
    return {"points": pts[valid].astype(np.float32), "inst": inst[valid], "ring": ring[valid],
            "col": col[valid], "range": rng_m[valid], "shape": (rows, cols), "origin": o}
