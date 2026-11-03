"""Spatial discretisation, reachable sets and trajectory enumeration."""
from __future__ import annotations

import numpy as np


class NodeGrid:
    """Regular 2-D node grid. Node index = iy * nx + ix (x varies fastest),
    the same ordering as the original `Regular2DNodeGrid`, so enumeration
    order (and therefore tie-breaking between equal-information paths) matches.
    """

    def __init__(self, xy0=(0, 0), xyf=(101, 101), d=(10, 10)):
        self.xy0 = np.asarray(xy0, float)
        self.xyf = np.asarray(xyf, float)
        self.xg = np.arange(xy0[0], xyf[0], d[0])
        self.yg = np.arange(xy0[1], xyf[1], d[1])
        X, Y = np.meshgrid(self.xg, self.yg)
        self.nodes = np.stack([X.ravel(), Y.ravel()], -1)
        self.N = len(self.nodes)

    def snap(self, p) -> int:
        return int(np.argmin(np.linalg.norm(self.nodes - np.asarray(p)[:2], axis=1)))

    def in_radius(self, idx: int, radius: float) -> list[int]:
        d = np.linalg.norm(self.nodes - self.nodes[idx], axis=1)
        return [int(i) for i in np.nonzero(d < radius)[0]]

    def in_map(self, p) -> bool:
        return bool(np.all(p[:2] >= self.xy0) and np.all(p[:2] <= self.xyf))

    def middle(self):
        return 0.5 * (self.xy0 + self.xyf)


def one_step_moves(grid: NodeGrid, idx: int, vmax: float, dt: float) -> list[int]:
    """Reachable set U(s): nodes within vmax*1.1*dt, excluding s itself
    (a UAV must move every step; with vmax=5, dt=2, d=10 this is the
    4-neighbourhood)."""
    nb = grid.in_radius(idx, vmax * 1.1 * dt)
    nb.remove(idx)
    return nb


def dependent_seq_generator(seqL, ind0, Xset, Xconst):
    """Verbatim port of utils/math/gen_seq.dependent_seq_generator.

    Enumerates node sequences [ind0, n1, ..., n_{seqL-1}] with n_{i+1} in
    Xconst[n_i]. Kept byte-for-byte equivalent because its odometer shares
    counters per (node, depth), which determines exactly which candidate
    paths the original planner scored.
    """
    Xconst_ctr = {s: {} for s in Xset}
    brkflg = 0
    while True:
        Xseq = [ind0]
        for i in range(1, seqL):
            prevnode = Xseq[i - 1]
            nextset = Xconst[prevnode]
            if i not in Xconst_ctr[prevnode]:
                Xconst_ctr[prevnode][i] = 0
            cnode = Xconst_ctr[prevnode][i]
            Xseq.append(nextset[cnode])
            if i == seqL - 1:
                Xconst_ctr[prevnode][i] = Xconst_ctr[prevnode][i] + 1
        for i in range(seqL - 1, 0, -1):
            prevnode = Xseq[i - 1]
            nextset = Xconst[prevnode]
            if Xconst_ctr[prevnode][i] == len(nextset):
                Xconst_ctr[prevnode][i] = 0
                j = i - 1
                if j == 0:
                    brkflg = 1
                else:
                    prevnode = Xseq[j - 1]
                    Xconst_ctr[prevnode][j] = Xconst_ctr[prevnode][j] + 1
        yield Xseq
        if brkflg:
            break


def exact_set_filter(seqs):
    """Original `traj_filter(..., 'ExactSetNodes')`: keep only the first
    trajectory for each *set* of visited nodes."""
    seen, out = [], []
    for s in seqs:
        ss = set(s)
        if ss not in seen:
            seen.append(ss)
            out.append(s)
    return out


def enumerate_trajectories(grid: NodeGrid, start: int, n_nodes: int, vmax: float, dt: float,
                           faithful: bool = True):
    """All T-step way-point trajectories S^j_{1:T} from `start`.

    faithful=True reproduces the original generator + set filter;
    faithful=False returns the full Cartesian set |U|^T (for comparison).
    """
    T = (n_nodes - 1) * dt
    Xset = grid.in_radius(start, vmax * (T + 0.1 * dt))
    Xconst = {i: one_step_moves(grid, i, vmax, dt) for i in Xset}
    if faithful:
        seqs = list(dependent_seq_generator(n_nodes, start, Xset, Xconst))
        return exact_set_filter(seqs)
    out = [[start]]
    for _ in range(n_nodes - 1):
        out = [s + [n] for s in out for n in Xconst[s[-1]]]
    return out
