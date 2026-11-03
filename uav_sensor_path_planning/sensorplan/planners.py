"""Information-theoretic UAV path planners (Sec. IV of the paper).

Objective for a candidate set of way-points over the horizon k = 1..T

    J = sum_k sum_i  1/2 ln( |P^i_{k|k-1}| / |P^i_{k|k}| )          (target MI)
      + lambda * sum_k sum_c [ H(p^c_{k|k-1}) - H(p^c_{k|k}) ]       (coverage MI)

The covariance update does not depend on the actual measurement value, so J
can be evaluated before any measurement is taken. A target is assumed
measurable by UAV j at step k when >30% of its 1-sigma position ellipse lies
inside UAV j's FOV (the original planning gate).

Two greedy approximations to the joint search over S^1 x ... x S^Ns:
  * seq_uav  : "Greedy/Sequential in UAVs"  - optimise UAV 1's whole T-step path,
               fix it, then UAV 2 conditioned on UAV 1, ...  cost O(Ns |U|^T)
  * seq_time : "Greedy/Sequential in time"  - optimise all UAVs jointly for one
               step, commit, move to the next step.          cost O(T |U|^Ns)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import NodeGrid, enumerate_trajectories, one_step_moves
from .models import CoverageMap, CVTargetModel, CircularFOVSensor, binary_entropy, cov_fov_overlap

OVERLAP_GATE = 0.3


@dataclass
class Belief:
    """Snapshot of everything the planner needs at the planning time."""
    x: np.ndarray          # (Na,4) means of ACTIVE targets
    P: np.ndarray          # (Na,4,4)
    p: np.ndarray          # (Nc,) coverage map


def gaussian_mi(Pprior, Ppost):
    """I(x;y) = 1/2 ln(|P_prior| / |P_post|)  (eq. MIforms). Batched."""
    _, ld1 = np.linalg.slogdet(Pprior)
    _, ld2 = np.linalg.slogdet(Ppost)
    return 0.5 * (ld1 - ld2)


def _measure_step(x, P, p, uav_pos, sensors, cmap: CoverageMap):
    """Apply planning-time (covariance-only) updates from a list of UAVs.
    x,P: (C,Na,4[,4])  p: (C,Nc)  uav_pos: list of (C,2) or None (UAV idle)."""
    for pos, sen in zip(uav_pos, sensors):
        if pos is None:
            continue
        if x.shape[1]:
            ov = cov_fov_overlap(x, P, pos[:, None, :], sen.fov_radius)
            x, P = sen.update(x, P, z=None, mask=ov > OVERLAP_GATE)
        inside = np.linalg.norm(cmap.cells[None] - pos[:, None, :], axis=-1) <= sen.fov_radius
        p = cmap.update(p, inside, sen)
    return x, P, p


def evaluate_paths(belief: Belief, paths, grid, sensors, tmodel, cmap, dt, lam):
    """Score C candidate joint paths in one batched pass.

    paths: list over UAVs of (C, n_nodes) int arrays, or None if that UAV has
    no plan yet (it then takes no measurements - as in the original seq-UAV).
    Returns (MI_targets, MI_coverage) each shape (C,).
    """
    C = next(p.shape[0] for p in paths if p is not None)
    n_nodes = next(p.shape[1] for p in paths if p is not None)
    x = np.broadcast_to(belief.x, (C,) + belief.x.shape).copy()
    P = np.broadcast_to(belief.P, (C,) + belief.P.shape).copy()
    p = np.broadcast_to(belief.p, (C,) + belief.p.shape).copy()
    mi_t = np.zeros(C)
    mi_c = np.zeros(C)
    for k in range(1, n_nodes):
        if x.shape[1]:
            x, P = tmodel.predict(x, P, dt)
        p = cmap.predict(p)
        Pprior, pprior = P, p
        pos = [None if pth is None else grid.nodes[pth[:, k]] for pth in paths]
        x, P, p = _measure_step(x, P, p, pos, sensors, cmap)
        if x.shape[1]:
            mi_t += gaussian_mi(Pprior, P).sum(-1)
        mi_c += (binary_entropy(pprior) - binary_entropy(p)).sum(-1)
    return mi_t, mi_c


def _argmax_first(score):
    """Index of the first maximum (original uses strict '>' in generation order)."""
    return int(np.argmax(score))


def seq_uav(belief, starts, grid, sensors, tmodel, cmap, n_nodes, dt, lam, vmax=5.0,
            faithful=True):
    """Sequential-in-UAV exhaustive search. Returns list of node paths."""
    plans: list = [None] * len(starts)
    for j, s0 in enumerate(starts):
        cand = np.array(enumerate_trajectories(grid, s0, n_nodes, vmax, dt, faithful))
        paths = [None if plans[r] is None else np.repeat(plans[r][None], len(cand), 0)
                 for r in range(len(starts))]
        paths[j] = cand
        mi_t, mi_c = evaluate_paths(belief, paths, grid, sensors, tmodel, cmap, dt, lam)
        plans[j] = cand[_argmax_first(mi_t + lam * mi_c)]
    return plans


def seq_time(belief, starts, grid, sensors, tmodel, cmap, n_nodes, dt, lam, vmax=5.0):
    """Sequential-in-time: joint one-step search over U^1 x ... x U^Ns."""
    x, P, p = belief.x.copy(), belief.P.copy(), belief.p.copy()
    cur = list(starts)
    plans = [[s] for s in starts]
    for _ in range(1, n_nodes):
        if len(x):
            x, P = tmodel.predict(x, P, dt)
        p = cmap.predict(p)
        moves = [one_step_moves(grid, c, vmax, dt) for c in cur]
        grids = np.meshgrid(*moves, indexing="ij")          # UAV0 outer loop
        combos = np.stack([g.ravel() for g in grids], -1)   # (C,Ns)
        C = len(combos)
        xb = np.broadcast_to(x, (C,) + x.shape).copy()
        Pb = np.broadcast_to(P, (C,) + P.shape).copy()
        pb = np.broadcast_to(p, (C,) + p.shape).copy()
        pos = [grid.nodes[combos[:, j]] for j in range(len(cur))]
        xu, Pu, pu = _measure_step(xb, Pb, pb, pos, sensors, cmap)
        mi_t = gaussian_mi(Pb, Pu).sum(-1) if len(x) else np.zeros(C)
        mi_c = (binary_entropy(pb) - binary_entropy(pu)).sum(-1)
        b = _argmax_first(mi_t + lam * mi_c)
        x, P, p = xu[b], Pu[b], pu[b]
        cur = [int(c) for c in combos[b]]
        for j in range(len(cur)):
            plans[j].append(cur[j])
    return [np.array(pl) for pl in plans]


def joint_exhaustive(belief, starts, grid, sensors, tmodel, cmap, n_nodes, dt, lam, vmax=5.0):
    """Full joint search over S^1 x S^2 (only tractable for tiny horizons) -
    the reference that seq_uav/seq_time approximate."""
    cands = [np.array(enumerate_trajectories(grid, s, n_nodes, vmax, dt, faithful=False))
             for s in starts]
    idx = np.meshgrid(*[np.arange(len(c)) for c in cands], indexing="ij")
    idx = [i.ravel() for i in idx]
    paths = [cands[j][idx[j]] for j in range(len(starts))]
    mi_t, mi_c = evaluate_paths(belief, paths, grid, sensors, tmodel, cmap, dt, lam)
    b = _argmax_first(mi_t + lam * mi_c)
    return [pth[b] for pth in paths]
