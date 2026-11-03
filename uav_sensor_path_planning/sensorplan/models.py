"""Target, sensor and coverage-map models.

Faithful re-implementation of the models used in

    N. Adurthi, P. Singla, M. Majji,
    "Information Theoretic Sensor Path Planning: Applications to UAV Routing"

(original code: nadurthi/ResearchCodes, main_dynSensorTasking_Exhaustive.py, April 2021).
Every model here is written to be batched: leading dimensions are free, so the
planners can push hundreds of candidate trajectories through in one call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Target motion: nearly-constant-velocity with a speed cap
# ---------------------------------------------------------------------------
@dataclass
class CVTargetModel:
    """x = [px, py, vx, vy].

    x_{k+1} = F x_k + nu_k,   nu_k ~ N(0, Q)
    Q = L1 * [[T^3/3 I, T^2/2 I], [T^2/2 I, T I]]   (white-noise acceleration)

    The original `KinematicModel_UM_control.propforward` also clips the mean
    speed to `maxvmag` before every prediction; we keep that behaviour.
    """

    L1: float = 0.0016
    maxvmag: float = 0.3

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def Q(self, dt: float) -> np.ndarray:
        T = dt
        q = np.array([[T**3 / 3, 0, T**2 / 2, 0],
                      [0, T**3 / 3, 0, T**2 / 2],
                      [T**2 / 2, 0, T, 0],
                      [0, T**2 / 2, 0, T]])
        return self.L1 * q

    def predict(self, x: np.ndarray, P: np.ndarray, dt: float):
        """Batched EKF prediction. x: (...,4)  P: (...,4,4)."""
        x = x.copy()
        v = np.linalg.norm(x[..., 2:4], axis=-1, keepdims=True)
        scale = np.where(v > 0, np.clip(v, 0, self.maxvmag) / np.where(v > 0, v, 1), 1.0)
        x[..., 2:4] = x[..., 2:4] * scale
        F = self.F(dt)
        xn = x @ F.T
        Pn = F @ P @ F.T + self.Q(dt)
        return xn, Pn


# ---------------------------------------------------------------------------
# Look-down camera: circular FOV, direct (x, y) position measurement
# ---------------------------------------------------------------------------
@dataclass
class CircularFOVSensor:
    """y = [px, py] + w,  w ~ N(0, R), valid only if ||p - s|| <= r_fov.

    TP/FP are the detection probabilities used by the coverage (search) map.
    """

    fov_radius: float
    sigma: float
    TP: float = 0.9
    FP: float = 0.1
    TN: float = 0.9
    FN: float = 0.1
    H: np.ndarray = field(default_factory=lambda: np.hstack([np.eye(2), np.zeros((2, 2))]))

    @property
    def R(self) -> np.ndarray:
        return self.sigma**2 * np.eye(2)

    def in_fov(self, p: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.linalg.norm(p[..., :2] - s[..., :2], axis=-1) <= self.fov_radius

    def measure(self, x_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return x_true[:2] + self.sigma * rng.standard_normal(2)

    def update(self, x, P, z=None, mask=None):
        """Batched linear Kalman update.

        z=None performs the *covariance-only* (pseudo) update used during
        planning, where the measurement has not been taken yet: the mean is
        left unchanged (innovation = 0) and P <- P - K S K^T.
        `mask` (...,) selects which entries are updated; others are copied.
        """
        H, R = self.H, self.R
        S = H @ P @ H.T + R                      # (...,2,2)
        Pxz = P @ H.T                            # (...,4,2)
        K = Pxz @ np.linalg.inv(S)               # (...,4,2)
        Pu = P - K @ S @ np.swapaxes(K, -1, -2)
        Pu = 0.5 * (Pu + np.swapaxes(Pu, -1, -2))
        if z is None:
            xu = x
        else:
            innov = z - x[..., :2]
            xu = x + (K @ innov[..., None])[..., 0]
        if mask is not None:
            m4 = mask[..., None]
            m44 = mask[..., None, None]
            xu = np.where(m4, xu, x)
            Pu = np.where(m44, Pu, P)
        return xu, Pu


# ---------------------------------------------------------------------------
# 1-sigma ellipse / FOV overlap (planning gate)
# ---------------------------------------------------------------------------
def _sunflower_disk(n: int = 400) -> np.ndarray:
    """Deterministic, near-uniform points on the unit disk."""
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    th = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([r * np.cos(th), r * np.sin(th)], -1)


_DISK = _sunflower_disk(400)


def cov_fov_overlap(m: np.ndarray, P: np.ndarray, s: np.ndarray, radius: float) -> np.ndarray:
    """Fraction of the 1-sigma position ellipse that lies inside the FOV disk.

    Matches `PlanarSensorModel.intersectCovWithFOV` (area(ellipse ∩ FOV) /
    area(ellipse)), computed by mapping a uniform disk through chol(P_pos)
    instead of shapely polygons so it can be batched.
    m: (...,>=2) P: (...,>=2,>=2) s: (...,2) broadcastable.
    """
    Ppos = P[..., :2, :2]
    L = np.linalg.cholesky(Ppos + 1e-12 * np.eye(2))
    pts = m[..., None, :2] + np.einsum("...ij,nj->...ni", L, _DISK)
    d = np.linalg.norm(pts - s[..., None, :2], axis=-1)
    return (d <= radius).mean(-1)


# ---------------------------------------------------------------------------
# Coverage / search map: independent binary Markov chain per grid cell
# ---------------------------------------------------------------------------
@dataclass
class CoverageMap:
    """p_c = probability attached to cell c (the original's "SearchTarget").

    Prediction:  p <- a p + (1-a)(1-p)            (a = 0.99, drifts to 0.5)
    Update for cells inside a FOV, given detection z_c:
        z=1: p <- p TP / (p TP + (1-p) FP)
        z=0: p <- p FN / (p FN + (1-p) TN)
    During planning the detection is assumed (z=1 inside the FOV), so a
    candidate path is rewarded for the binary-entropy drop it causes.
    In effect this is a decaying "where have we looked recently" map; its
    entropy drop is the lambda-weighted exploration term in the objective.
    """

    cells: np.ndarray            # (Nc,2)
    stay: float = 0.99

    def predict(self, p):
        return self.stay * p + (1 - self.stay) * (1 - p)

    def update(self, p, inside, sensor: CircularFOVSensor, z=None):
        """inside: bool (...,Nc). z: None (planning, z=inside) or 0/1 array."""
        if z is None:
            z = np.ones_like(p)
        num1 = np.where(z == 1, p * sensor.TP, p * sensor.FN)
        num0 = np.where(z == 1, (1 - p) * sensor.FP, (1 - p) * sensor.TN)
        pu = num1 / (num1 + num0)
        return np.where(inside, pu, p)

    def random_detections(self, inside, sensor: CircularFOVSensor, rng):
        u = rng.random(inside.shape)
        return np.where(inside, u < sensor.TP, u < sensor.FP).astype(float)


def binary_entropy(p):
    p = np.where(p == 0, 1e-3, p)       # same guard as the original code
    q = np.where(1 - p == 0, 1e-3, 1 - p)
    return -(p * np.log(p) + q * np.log(q))
