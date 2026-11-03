"""Procedural box-wall scenes for truck/container unloading.

Coordinates: camera frame, X right, Y down, Z forward (metres). The trailer floor
is at Y = +FLOOR_Y, the ceiling at Y = -CEIL_Y, and the side walls at X = +/-HALF_W.
Boxes are oriented cuboids (centre, rotation, half extents). Every scene comes with
exact ground truth: per-box pose and size, plus per-pixel / per-point instance ids
from the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

FLOOR_Y = 1.25      # camera height above trailer floor (m)
CEIL_Y = 1.45       # ceiling above camera (m)
HALF_W = 1.22       # half trailer width (m)


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


@dataclass
class Sku:
    """Appearance shared by every box of one product, so stacks repeat textures."""
    base: np.ndarray            # RGB in [0,1]
    prints: list                # (u0,v0,u1,v1,rgb) rectangles on the front face, in face-normalised uv
    tape: str                   # 'none' | 'h' | 'v' | 'hv'
    tape_rgb: np.ndarray
    label: tuple | None         # (u0,v0,u1,v1) white shipping label


@dataclass
class Box:
    center: np.ndarray
    R: np.ndarray
    half: np.ndarray            # half extents along local x (width), y (height), z (depth)
    sku: int
    row: int                    # 0 = front row, 1 = back row


@dataclass
class Scene:
    boxes: list
    skus: list
    seed: int
    light_gain: float = 1.0
    meta: dict = field(default_factory=dict)


def _make_sku(rng: np.random.Generator) -> Sku:
    kind = rng.choice(["kraft", "kraft", "kraft", "white", "printed"])
    if kind == "kraft":
        base = np.array([0.62, 0.45, 0.28]) * rng.uniform(0.8, 1.15) + rng.normal(0, 0.025, 3)
    elif kind == "white":
        base = np.array([0.86, 0.84, 0.80]) * rng.uniform(0.85, 1.05)
    else:
        base = rng.uniform(0.25, 0.85, 3)
    prints = []
    for _ in range(int(rng.integers(0, 5))):
        u0, v0 = rng.uniform(0.05, 0.7, 2)
        du, dv = rng.uniform(0.08, 0.45), rng.uniform(0.03, 0.25)
        prints.append((u0, v0, min(u0 + du, 0.95), min(v0 + dv, 0.95), rng.uniform(0.05, 0.9, 3)))
    tape = rng.choice(["none", "h", "v", "hv", "h"])
    tape_rgb = rng.choice([np.array([0.55, 0.38, 0.18]), np.array([0.82, 0.74, 0.58]),
                           np.array([0.75, 0.70, 0.62]), np.array([0.85, 0.85, 0.85])])
    label = None
    if rng.random() < 0.6:
        u0, v0 = rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.6)
        label = (u0, v0, u0 + rng.uniform(0.15, 0.3), v0 + rng.uniform(0.12, 0.25))
    return Sku(np.clip(base, 0, 1), prints, str(tape), tape_rgb, label)


def make_scene(seed: int, difficulty: float = 1.0) -> Scene:
    """Build one wall. `difficulty` scales gaps (smaller = harder), yaw/tilt and recesses."""
    rng = np.random.default_rng(seed)
    n_sku = int(rng.integers(2, 7))
    skus = [_make_sku(rng) for _ in range(n_sku)]
    z_wall = rng.uniform(1.5, 2.3)
    boxes: list[Box] = []

    for row in (0, 1):
        x = -HALF_W + rng.uniform(0.0, 0.04)
        z_row = z_wall + (0 if row == 0 else rng.uniform(0.45, 0.65))
        while x < HALF_W - 0.15:
            col_w = rng.uniform(0.25, 0.62)
            if x + col_w > HALF_W:
                col_w = HALF_W - x - 0.005
            if col_w < 0.15:
                break
            sku_col = int(rng.integers(n_sku))
            y_bottom = FLOOR_Y
            # column height: how far the stack goes up
            top_limit = -CEIL_Y + rng.uniform(0.05, 0.9)
            while True:
                h = rng.uniform(0.18, 0.5)
                if y_bottom - h < top_limit:
                    break
                w = col_w * rng.uniform(0.82, 1.0)
                d = rng.uniform(0.28, 0.6)
                same = rng.random() < 0.65
                sku = sku_col if same else int(rng.integers(n_sku))
                gap = rng.uniform(0.0, 0.012) / max(difficulty, 0.3)
                yaw = rng.normal(0, np.deg2rad(4) * difficulty) if rng.random() < 0.5 else 0.0
                roll = rng.normal(0, np.deg2rad(1.5) * difficulty) if rng.random() < 0.3 else 0.0
                dz = rng.normal(0, 0.02 * difficulty)
                if row == 0 and rng.random() < 0.07 * difficulty:
                    dz += rng.uniform(0.08, 0.2)          # pushed-in box
                cx = x + w / 2 + rng.uniform(0, col_w - w)
                cy = y_bottom - h / 2 - gap
                cz = z_row + d / 2 + dz
                boxes.append(Box(np.array([cx, cy, cz]), rot_xyz(0, yaw, roll),
                                 np.array([w / 2, h / 2, d / 2]), sku, row))
                y_bottom = cy - h / 2
            x += col_w + rng.uniform(0.0, 0.015) / max(difficulty, 0.3)
    # Front-row gaps: delete a few front boxes to reveal the back row (common in partly unloaded trailers)
    keep = []
    for b in boxes:
        if b.row == 0 and rng.random() < 0.06 * difficulty:
            continue
        keep.append(b)
    # Remove front boxes that would float after deletion (simple support check)
    keep = _drop_unsupported(keep)
    return Scene(keep, skus, seed, light_gain=rng.uniform(0.75, 1.25),
                 meta={"z_wall": z_wall, "difficulty": difficulty})


def _drop_unsupported(boxes: list) -> list:
    ok = []
    for b in sorted(boxes, key=lambda b: -b.center[1]):          # bottom first (largest Y)
        bottom = b.center[1] + b.half[1]
        if bottom > FLOOR_Y - 0.03:
            ok.append(b)
            continue
        supported = False
        for s in ok:
            if s.row != b.row:
                continue
            top = s.center[1] - s.half[1]
            overlap_x = min(b.center[0] + b.half[0], s.center[0] + s.half[0]) - max(b.center[0] - b.half[0], s.center[0] - s.half[0])
            if abs(top - bottom) < 0.05 and overlap_x > 0.3 * 2 * b.half[0]:
                supported = True
                break
        if supported:
            ok.append(b)
    return ok
