"""Franka Panda kinematics (modified DH, as published by Franka) and a sphere model.

The joint values used here are the same as in the MoveIt Panda URDF
(moveit_resources_panda_description), so a path planned here can be replayed in
MoveIt and vice versa. plan_moveit.py checks the TCP pose against MoveIt's own
forward kinematics and stores the error in the results.
"""
import numpy as np

# (a_{i-1}, d_i, alpha_{i-1}) for joints 1..7, then the flange (link8)
DH = [
    (0.0,     0.333,  0.0),
    (0.0,     0.0,   -np.pi / 2),
    (0.0,     0.316,  np.pi / 2),
    (0.0825,  0.0,    np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0,     0.0,    np.pi / 2),
    (0.088,   0.0,    np.pi / 2),
]
FLANGE_D = 0.107
TCP_D = 0.1034          # panda_hand_tcp is 10.34 cm beyond the flange
HAND_YAW = -np.pi / 4   # panda_hand is rotated -45 deg about the flange z

LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0, 0, 0, 1],
    ])


# modified DH: T_i = RotX(alpha) TransX(a) * RotZ(theta) TransZ(d); the first factor is constant
_A = []
for a, d, al in DH:
    ca, sa = np.cos(al), np.sin(al)
    _A.append(np.array([[1, 0, 0, a], [0, ca, -sa, -d * sa], [0, sa, ca, d * ca], [0, 0, 0, 1.0]]))
_FLANGE = _mdh(0, FLANGE_D, 0, 0)
_HAND_T = _mdh(0, 0, 0, HAND_YAW)
_TCP = _mdh(0, TCP_D, 0, 0)


def fk_frames(q):
    """4x4 transforms of frames 1..7, flange, hand and TCP (10 frames)."""
    T = np.eye(4)
    out = []
    Rz = np.eye(4)
    for A, th in zip(_A, q):
        c, s = np.cos(th), np.sin(th)
        Rz[0, 0] = c; Rz[0, 1] = -s; Rz[1, 0] = s; Rz[1, 1] = c
        T = T @ (A @ Rz)
        out.append(T)
    T = T @ _FLANGE
    out.append(T)
    Th = T @ _HAND_T
    out.append(Th)
    out.append(Th @ _TCP)
    return out


def tcp(q):
    return fk_frames(q)[-1]


def skeleton(q):
    """Points for drawing the arm: base, joint origins, flange, TCP."""
    F = fk_frames(q)
    pts = [np.zeros(3)] + [F[i][:3, 3] for i in (0, 2, 3, 4, 6, 7, 9)]
    return np.array(pts)


# Collision spheres per link, in the link's own frame: [x, y, z, radius]. Fitted to the URDF
# collision meshes of moveit_resources_panda_description by fit_spheres.py (k-means on surface
# samples; radius = farthest assigned sample + 4 mm, so every mesh sample is inside a sphere).
# panda_link0 is left out: it never moves and the lidar pre-filter clears the base column.
import json as _json
from pathlib import Path as _Path
_SPH = _json.loads((_Path(__file__).with_name("panda_spheres.json")).read_text())
_FRAME = {"link1": 0, "link2": 1, "link3": 2, "link4": 3, "link5": 4, "link6": 5, "link7": 6, "hand": 8}
_GROUPS = [(_FRAME[k], np.c_[np.array(v)[:, :3], np.ones(len(v))].T) for k, v in _SPH.items()]
RADII = np.concatenate([np.array(v)[:, 3] for v in _SPH.values()])
SPHERE_LINK = sum(([k] * len(v) for k, v in _SPH.items()), [])


def spheres(q):
    """Sphere centres (M,3) and radii (M,) approximating the arm at q."""
    F = fk_frames(q)
    return np.vstack([(F[f] @ L)[:3].T for f, L in _GROUPS]), RADII


# Self-collision: link pairs MoveIt checks for the Panda, i.e. every pair not disabled in
# moveit_resources_panda_moveit_config/config/panda.srdf (fingers counted with the hand).
SELF_PAIRS = [("link1", "link5"), ("link1", "link6"), ("link1", "link7"), ("link1", "hand"),
              ("link2", "link5"), ("link2", "link7"), ("link2", "hand"), ("link5", "link7"), ("link5", "hand")]
_IDX = {k: np.flatnonzero(np.array(SPHERE_LINK) == k) for k in _SPH}
# link5 against link7/hand: only the forearm body (local z < -0.08 m), not the casing at the wrist
# joint, which the fat spheres make touch link7 in most configurations
_IDX["link5_body"] = _IDX["link5"][np.array(_SPH["link5"])[:, 2] < -0.08]
SELF_PAIRS = [("link5_body" if (a, b) in (("link5", "link7"), ("link5", "hand")) else a, b)
              for a, b in SELF_PAIRS]
_PI = np.concatenate([np.repeat(_IDX[a], len(_IDX[b])) for a, b in SELF_PAIRS])
_PJ = np.concatenate([np.tile(_IDX[b], len(_IDX[a])) for a, b in SELF_PAIRS])
_PR2 = (RADII[_PI] + RADII[_PJ]) ** 2


def _prune_joint_neighbours(n=2000, max_freq=0.25):
    """The fitted spheres are fat, so spheres either side of the wrist joints overlap in many
    configurations where the meshes do not touch (link5/link7, link5/hand). Drop sphere pairs
    that overlap in more than max_freq of random configurations, as MoveIt's setup assistant
    drops link pairs that are 'always' in contact."""
    global _PI, _PJ, _PR2
    qs = np.random.default_rng(1).uniform(LOWER, UPPER, (n, 7))
    C = np.array([spheres(q)[0] for q in qs])
    d = C[:, _PI] - C[:, _PJ]
    keep = (np.einsum("nij,nij->ni", d, d) < _PR2).mean(0) <= max_freq
    _PI, _PJ, _PR2 = _PI[keep], _PJ[keep], _PR2[keep]


_prune_joint_neighbours()


def self_collision(c):
    """True if any sphere pair from a checked link pair overlaps."""
    d = c[_PI] - c[_PJ]
    return bool(np.any(np.einsum("ij,ij->i", d, d) < _PR2))


N_BASE_SPHERES = len(_SPH["link1"])  # link1 sits on the base: skip it for the floor test


def rot_err(R, Rd):
    """Axis-angle error vector taking R to Rd."""
    E = Rd @ R.T
    v = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    s = np.linalg.norm(v); cth = (np.trace(E) - 1) / 2
    ang = np.arctan2(s, cth)
    return v * (ang / s) if s > 1e-9 else v


def jacobian(q, eps=1e-6):
    T0 = tcp(q)
    J = np.zeros((6, 7))
    for i in range(7):
        dq = q.copy(); dq[i] += eps
        T1 = tcp(dq)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        J[3:, i] = rot_err(T0[:3, :3], T1[:3, :3]) / eps
    return J


def ik(p_des, R_des, seeds, iters=300, tol=1e-4, rng=None, n_random=40):
    """Damped least-squares IK. Returns all distinct solutions within limits."""
    rng = rng or np.random.default_rng(0)
    seeds = list(seeds) + [rng.uniform(LOWER, UPPER) for _ in range(n_random)]
    sols = []
    for q in seeds:
        q = np.clip(np.array(q, float), LOWER, UPPER)
        for _ in range(iters):
            T = tcp(q)
            e = np.concatenate([p_des - T[:3, 3], rot_err(T[:3, :3], R_des)])
            if np.linalg.norm(e) < tol:
                break
            J = jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), LOWER, UPPER)
        if np.linalg.norm(e) < tol and not any(np.linalg.norm(q - s) < 1e-2 for s in sols):
            sols.append(q)
    return sols


def tool_pointing(z_axis, x_axis):
    z = np.asarray(z_axis, float); z /= np.linalg.norm(z)
    x = np.asarray(x_axis, float); x -= z * (x @ z); x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])
