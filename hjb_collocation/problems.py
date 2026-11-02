"""The three control problems of the paper, as vectorised Problem objects."""
import os
import pickle
import itertools
import numpy as np
from hjbcoll import Problem

HERE = os.path.dirname(os.path.abspath(__file__))
PTS = pickle.load(open(os.path.join(HERE, "points", "points.pkl"), "rb"))


def quad_cost(Q):
    Q = np.asarray(Q, float)
    return lambda X: np.einsum("ik,kl,il->i", X, Q, X)


# ---------------- Example 1: Van der Pol oscillator ------------------------------------------
def vanderpol(q=50.0, r=5.0, K=(-1.2, -1.2)):
    K = np.asarray(K, float)
    f = lambda X: np.c_[X[:, 1], -X[:, 0] + (1 - X[:, 0] ** 2) * X[:, 1]]
    g = lambda X: np.broadcast_to(np.array([[0.0], [1.0]]), (X.shape[0], 2, 1))
    return Problem(n=2, f=f, g=g, l=quad_cost(q * np.eye(2)), R=np.array([[r]]),
                   u0=lambda X: (X @ K)[:, None], name="vanderpol")


def vanderpol_points(half=3.0):
    """CUT8 (uniform, 21 pts) scaled to [-3,3]^2 plus the 8 boundary points → 29 points."""
    X = PTS["cut8_2D_uniform"]["X"] * half
    B = half * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [0, 1], [0, -1], [-1, 0]], float)
    return np.r_[X, B]


# ---------------- Example 2: spin stabilisation (Euler's equations) ---------------------------
I_SPIN = np.array([14.0, 10.0, 8.0])


def spin(q, r, k0):
    I1, I2, I3 = I_SPIN

    def f(X):
        w1, w2, w3 = X.T
        return np.c_[(I2 - I3) / I1 * w2 * w3, (I3 - I1) / I2 * w3 * w1, (I1 - I2) / I3 * w1 * w2]
    g = lambda X: np.broadcast_to(np.diag(1 / I_SPIN), (X.shape[0], 3, 3))
    return Problem(n=3, f=f, g=g, l=quad_cost(q * np.eye(3)), R=r * np.eye(3),
                   u0=lambda X: -k0 * X, name=f"spin_q{q}_r{r}")


def spin_points():
    return PTS["cut8_3D_uniform"]["X"].copy()          # 59 points in [-1,1]^3


# ---------------- Example 3: attitude regulation (CRP + Euler) ------------------------------
def attitude(q=40.0, r=1.0, Kp=15.0, Kd=5.0):
    I1, I2, I3 = I_SPIN

    def f(X):
        rho, w = X[:, :3], X[:, 3:]
        # ρ̇ = ½ (I + [ρ×] + ρρᵀ) ω   (paper: ½(I − S(ρ) + ρρᵀ)ω with S(ρ) = −[ρ×])
        rhodot = 0.5 * (w + np.cross(rho, w) + rho * np.sum(rho * w, 1, keepdims=True))
        w1, w2, w3 = w.T
        wdot = np.c_[(I2 - I3) / I1 * w2 * w3, (I3 - I1) / I2 * w3 * w1, (I1 - I2) / I3 * w1 * w2]
        return np.c_[rhodot, wdot]

    G = np.zeros((6, 3)); G[3:, :] = np.diag(1 / I_SPIN)
    g = lambda X: np.broadcast_to(G, (X.shape[0], 6, 3))
    K = np.c_[Kp * np.eye(3), Kd * np.eye(3)]
    return Problem(n=6, f=f, g=g, l=quad_cost(q * np.eye(6)), R=r * np.eye(3),
                   u0=lambda X: -X @ K.T, name="attitude")


ATT_SIGMA2 = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.0])     # covariance used to scale Gaussian points
ATT_BOX = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])        # ρ ∈ [-2,2]^3, ω ∈ [-1,1]^3


def attitude_boundary():
    """[±1]^6 corners, [±1,±1,0,0,0,0] and [±1,±1,±1,0,0,0] patterns (all placements) → 284 pts,
    scaled to the domain box."""
    pts = set()
    for k in (6, 2, 3):
        for idx in itertools.combinations(range(6), k):
            for s in itertools.product((-1, 1), repeat=k):
                v = [0] * 6
                for i, sg in zip(idx, s):
                    v[i] = sg
                pts.add(tuple(v))
    return np.array(sorted(pts), float) * ATT_BOX


def attitude_points(kind="cut8"):
    key = {"cut8": "cut8_6D_gaussian", "gh3": "gh3_6D_gaussian"}[kind]
    X = PTS[key]["X"] * np.sqrt(ATT_SIGMA2)
    return np.r_[X, attitude_boundary()]
