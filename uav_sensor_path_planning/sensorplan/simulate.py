"""Closed-loop simulation: plan -> fly -> measure -> filter, receding horizon.

Mirrors the main loop of the original `main_dynSensorTasking_Exhaustive.py`:
  1. Detection: an untracked (or lost) target whose true position is inside
     any UAV's FOV at time t is (re)initialised at the measured position.
  2. Every `replan_every` steps, plan `horizon` way-points (seq-UAV / seq-time).
  3. Predict all ACTIVE target tracks (lost tracks are frozen) and the
     coverage map to t+dt; move UAVs to their next way-point.
  4. EKF update for every active target inside each UAV's FOV (true check);
     coverage map update with random TP/FP detections.
  5. A track is declared lost when any 1-sigma position axis exceeds the
     largest FOV radius.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .grid import NodeGrid
from .models import CircularFOVSensor, CoverageMap, CVTargetModel
from .planners import Belief, seq_time, seq_uav, joint_exhaustive

P0 = np.diag([5.0**2, 5.0**2, 0.2**2, 0.2**2])


@dataclass
class SimConfig:
    method: str = "seq_uav"            # seq_uav | seq_time | joint
    lam: float = 3.0                   # coverage-MI weight lambda
    dt: float = 2.0
    horizon: int = 5                   # way-points incl. current (TT=5)
    replan_every: int = 3              # TTrecomp
    lost_factor: float = 1.0
    vmax: float = 5.0
    faithful_enum: bool = True
    seed: int = 0
    sensors: list = field(default_factory=lambda: [
        CircularFOVSensor(fov_radius=20, sigma=1.0),     # UAV:0 wide, coarse
        CircularFOVSensor(fov_radius=10, sigma=0.2),     # UAV:1 narrow, precise
    ])
    uav_starts: list = field(default_factory=lambda: [(70, 70), (30, 30)])


@dataclass
class Track:
    x: np.ndarray
    P: np.ndarray
    active: bool = True
    zk: list = field(default_factory=list)   # (t, z) kept for velocity re-init


def initialise(track: Track | None, t, z, dt):
    """Original RE_Initialize: velocity 0.1,0.1 unless two fixes <= 2dt apart."""
    x0 = np.hstack([z, 0.1, 0.1])
    zk = [] if track is None else track.zk
    if zk:
        zk = zk + [(t, z)]
        ddt = zk[1][0] - zk[0][0]
        vf = (zk[1][1] - zk[0][1]) / ddt
        if ddt <= 2 * dt:
            x0 = np.hstack([z, vf])
        zk = []
    else:
        zk = [(t, z)]
    return Track(x=x0, P=P0.copy(), active=True, zk=zk)


def run(truth: np.ndarray, tvec: np.ndarray, cfg: SimConfig, grid: NodeGrid | None = None,
        progress: bool = False):
    """truth: (Nt, K, 4) ground-truth target states at tvec (K steps).

    Returns a dict with everything needed for metrics and animation.
    """
    rng = np.random.default_rng(cfg.seed)
    grid = grid or NodeGrid()
    tmodel = CVTargetModel()
    cmap = CoverageMap(cells=grid.nodes.copy())
    sensors = cfg.sensors
    Nt, K = truth.shape[:2]
    Nu = len(sensors)
    dt = cfg.dt

    uav = [grid.snap(s) for s in cfg.uav_starts]
    tracks: list[Track | None] = [None] * Nt
    p_cov = 0.5 * np.ones(grid.N)

    # logs (posterior at each time index)
    est_x = np.full((Nt, K, 4), np.nan)
    est_P = np.full((Nt, K, 4, 4), np.nan)
    status = np.zeros((Nt, K), dtype=np.int8)        # 0 untracked, 1 active, 2 lost
    uav_log = np.zeros((Nu, K), dtype=int)
    plan_log = {}                                    # k -> list of node paths
    cov_log = np.zeros((K, grid.N))
    meas_log = []                                    # (k, uav, target, z)
    uav_log[:, 0] = uav
    cov_log[0] = p_cov

    plan = None
    plan_k0 = 0
    for k in range(K - 1):
        t = tvec[k]
        # ---- 1. detection / (re)initialisation at time t -------------------
        for i in range(Nt):
            if tracks[i] is not None and tracks[i].active:
                continue
            for j in range(Nu):
                if sensors[j].in_fov(truth[i, k], grid.nodes[uav[j]]):
                    z = sensors[j].measure(truth[i, k], rng)
                    tracks[i] = initialise(tracks[i], t, z, dt)
                    est_x[i, k], est_P[i, k], status[i, k] = tracks[i].x, tracks[i].P, 1
                    meas_log.append((k, j, i, z))
                    break

        # ---- 2. plan --------------------------------------------------------
        if plan is None or k - plan_k0 == cfg.replan_every:
            n_nodes = min(cfg.horizon, K - k)
            act = [i for i in range(Nt) if tracks[i] is not None and tracks[i].active]
            belief = Belief(
                x=np.array([tracks[i].x for i in act]).reshape(-1, 4),
                P=np.array([tracks[i].P for i in act]).reshape(-1, 4, 4),
                p=p_cov.copy())
            args = (belief, uav, grid, sensors, tmodel, cmap, n_nodes, dt, cfg.lam)
            if cfg.method == "seq_uav":
                plan = seq_uav(*args, vmax=cfg.vmax, faithful=cfg.faithful_enum)
            elif cfg.method == "seq_time":
                plan = seq_time(*args, vmax=cfg.vmax)
            elif cfg.method == "joint":
                plan = joint_exhaustive(*args, vmax=cfg.vmax)
            else:
                raise ValueError(cfg.method)
            plan_k0 = k
            plan_log[k] = [np.asarray(pl) for pl in plan]

        # ---- 3. predict + move ---------------------------------------------
        for i in range(Nt):
            tr = tracks[i]
            if tr is not None and tr.active:
                tr.x, tr.P = tmodel.predict(tr.x, tr.P, dt)
        p_cov = cmap.predict(p_cov)
        uav = [int(plan[j][k - plan_k0 + 1]) for j in range(Nu)]
        uav_log[:, k + 1] = uav

        # ---- 4. measure + update -------------------------------------------
        for j in range(Nu):
            s = grid.nodes[uav[j]]
            inside_c = sensors[j].in_fov(grid.nodes, s)
            det = cmap.random_detections(inside_c, sensors[j], rng)
            p_cov = cmap.update(p_cov, inside_c, sensors[j], z=det)
        for i in range(Nt):
            tr = tracks[i]
            if tr is None:
                continue
            if tr.active:
                for j in range(Nu):
                    z = sensors[j].measure(truth[i, k + 1], rng)
                    if sensors[j].in_fov(truth[i, k + 1], grid.nodes[uav[j]]):
                        tr.x, tr.P = sensors[j].update(tr.x, tr.P, z=z)
                        meas_log.append((k + 1, j, i, z))
            est_x[i, k + 1], est_P[i, k + 1] = tr.x, tr.P
            status[i, k + 1] = 1 if tr.active else 2
        cov_log[k + 1] = p_cov

        # ---- 5. lost-track test --------------------------------------------
        rmax = max(s.fov_radius for s in sensors)
        for tr in tracks:
            if tr is not None and tr.active:
                ev = np.linalg.eigvalsh(tr.P[:2, :2])
                if np.any(np.sqrt(np.maximum(ev, 0)) >= cfg.lost_factor * rmax):
                    tr.active = False
        if progress and k % 10 == 0:
            print(f"  t={t:5.0f}  active={sum(1 for tr in tracks if tr and tr.active)}/{Nt}")

    return dict(truth=truth, t=tvec, est_x=est_x, est_P=est_P, status=status,
                uav=uav_log, plans=plan_log, coverage=cov_log, meas=meas_log,
                grid_nodes=grid.nodes, fov=[s.fov_radius for s in sensors], cfg=cfg)


# ---------------------------------------------------------------------------
# Metrics (utils/metrics/tracking.multiTarget_tracking_metrics)
# ---------------------------------------------------------------------------
def metrics(res) -> pd.DataFrame:
    truth, ex, eP, st, t = res["truth"], res["est_x"], res["est_P"], res["status"], res["t"]
    rows = {}
    for i in range(truth.shape[0]):
        act = st[i] == 1
        r = {}
        if not act.any():
            r.update(RMSE_Active_pos=np.nan, RMSE_Active_vel=np.nan, TIMES_Active_count=0,
                     TIMES_Active_percent=0.0, TIMES_Active_percent_after_discovery=0.0,
                     TIME_percent_before_discover=100.0, pos_in_2sigma_percent=np.nan)
        else:
            e = truth[i, act] - ex[i, act]
            k0 = int(np.argmax(act))
            d2 = np.array([e_[:2] @ np.linalg.solve(P_[:2, :2], e_[:2])
                           for e_, P_ in zip(e, eP[i, act])])
            r.update(RMSE_Active_pos=np.sqrt(np.mean(np.sum(e[:, :2]**2, 1))),
                     RMSE_Active_vel=np.sqrt(np.mean(np.sum(e[:, 2:]**2, 1))),
                     TIMES_Active_count=int(act.sum()),
                     TIMES_Active_percent=100 * act.mean(),
                     TIMES_Active_percent_after_discovery=100 * act[k0:].mean(),
                     TIME_percent_before_discover=100 * t[k0] / t[-1],
                     pos_in_2sigma_percent=100 * np.mean(d2 <= 4.0))
        rows[f"Trgt:{i}"] = r
    return pd.DataFrame(rows).T


def prob_ratio_2sigma(res):
    """p(x_true)/p(2-sigma boundary) for the position block, per target/time.
    > 1  <=> truth lies inside the 2-sigma ellipse (paper's last figure)."""
    truth, ex, eP, st = res["truth"], res["est_x"], res["est_P"], res["status"]
    out = np.full(st.shape, np.nan)
    for i, k in zip(*np.nonzero(st == 1)):
        e = truth[i, k, :2] - ex[i, k, :2]
        d2 = e @ np.linalg.solve(eP[i, k, :2, :2], e)
        out[i, k] = np.exp(-0.5 * (d2 - 4.0))
    return out


SUMMARY_COLS = ["RMSE_Active_pos", "RMSE_Active_vel", "TIMES_Active_percent",
                "TIMES_Active_percent_after_discovery", "TIME_percent_before_discover"]


def summary(m: pd.DataFrame) -> dict:
    m = m.astype(float)
    d = {c: float(np.nanmean(m[c])) for c in SUMMARY_COLS}
    d["never_found"] = int(m["TIMES_Active_count"].eq(0).sum())
    return d
