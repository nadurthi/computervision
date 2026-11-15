"""Unloading-cell scene and a simulated 3D lidar.

World frame = robot base frame (panda_link0), z up, metres.
The scene is a list of axis-aligned boxes: a carton wall in front of the arm,
a conveyor to its right, a vertical post between them and an overhead beam.
The lidar is ray-cast analytically against those boxes and the floor.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Box:
    name: str
    lo: np.ndarray  # (3,) min corner
    hi: np.ndarray  # (3,) max corner
    kind: str = "obstacle"  # carton | conveyor | post | beam

    @property
    def centre(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def size(self):
        return self.hi - self.lo


def carton_wall(rng, x_face=0.62, y_range=(-0.50, 0.50), rows=3, depth=0.30):
    """Stack of cartons, faces roughly at x = x_face (jittered a few cm)."""
    boxes = []
    z = 0.0
    for r in range(rows):
        h = rng.uniform(0.22, 0.28)
        y = y_range[0]
        c = 0
        while y < y_range[1] - 0.05:
            w = min(rng.uniform(0.22, 0.34), y_range[1] - y)
            dx = rng.uniform(-0.03, 0.03)
            lo = np.array([x_face + dx, y + 0.004, z])
            hi = np.array([x_face + dx + depth, y + w - 0.004, z + h])
            boxes.append(Box(f"carton_r{r}_c{c}", lo, hi, "carton"))
            y += w
            c += 1
        z += h
    return boxes


def build_scene(seed=0):
    rng = np.random.default_rng(seed)
    boxes = carton_wall(rng)
    boxes += [
        Box("conveyor", np.array([-0.25, -0.80, 0.0]), np.array([0.35, -0.48, 0.32]), "conveyor"),
        Box("post", np.array([0.33, -0.36, 0.0]), np.array([0.41, -0.28, 1.25]), "post"),
        Box("beam", np.array([0.15, -0.90, 1.12]), np.array([0.95, 0.90, 1.20]), "beam"),
    ]
    return boxes


@dataclass
class Lidar:
    """Spinning multi-beam lidar (Ouster/Velodyne-like), ray-cast analytically."""
    origin: np.ndarray = field(default_factory=lambda: np.array([-0.30, 0.25, 1.00]))
    n_rings: int = 64
    elev_deg: tuple = (-45.0, 20.0)
    n_azimuth: int = 512
    az_deg: tuple = (-100.0, 100.0)  # only the forward half is kept
    max_range: float = 4.0
    range_sigma: float = 0.008
    dropout: float = 0.02

    def directions(self):
        el = np.deg2rad(np.linspace(*self.elev_deg, self.n_rings))
        az = np.deg2rad(np.linspace(*self.az_deg, self.n_azimuth))
        E, A = np.meshgrid(el, az, indexing="ij")
        d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
        return d.reshape(-1, 3), E.reshape(-1)

    def scan(self, boxes, rng, floor=True):
        """Returns hit points (N,3) in world frame and their ring index."""
        d, _ = self.directions()
        o = self.origin
        t_best = np.full(len(d), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            for b in boxes:  # slab test, all rays at once
                t1 = (b.lo - o) * inv
                t2 = (b.hi - o) * inv
                tmin = np.nanmax(np.minimum(t1, t2), axis=1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=1)
                hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
                t_best = np.where(hit & (tmin < t_best), tmin, t_best)
            if floor:
                tf = -o[2] / d[:, 2]
                t_best = np.where((d[:, 2] < 0) & (tf < t_best), tf, t_best)
        ok = np.isfinite(t_best) & (t_best < self.max_range)
        ok &= rng.random(len(d)) > self.dropout
        t = t_best[ok] + rng.normal(0, self.range_sigma, ok.sum())
        ring = np.repeat(np.arange(self.n_rings), self.n_azimuth)[ok]
        return o + d[ok] * t[:, None], ring


def preprocess(points, robot_radius=0.22, ground_z=0.03):
    """Ground removal and robot self-filter (a cylinder around the base).

    Real lidar pipelines remove the arm's own returns with its current pose;
    here the arm is not ray-cast, so only the base column is cleared.
    """
    keep = points[:, 2] > ground_z
    keep &= np.hypot(points[:, 0], points[:, 1]) > robot_radius
    return points[keep]
