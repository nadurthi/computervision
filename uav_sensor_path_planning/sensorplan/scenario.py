"""Ground-truth scenario generation and detailed per-target track accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_truth(arena=100.0, n_targets=10, duration=200.0, dt=2.0, speed=0.3,
                   change_rate=0.7, seed=0):
    """Ground truth with the original script's target motion
    (main_dynSensorTasking_Exhaustive.py, 'Generate truth').

    Targets start uniformly in the arena with heading drawn at random and fixed
    speed. Each step, with probability 1-change_rate the heading is nudged
    (v <- normalise(0.9 v + 0.1 n), n ~ N(0, I)); a target outside the arena
    is steered half-way toward the arena centre. Speed is capped at `speed`.
    Returns truth (Nt, K, 4) and t (K,).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)
    K = len(t)
    X = np.zeros((n_targets, K, 4))
    mid = np.array([arena / 2, arena / 2])
    for i in range(n_targets):
        x = np.zeros(4)
        x[:2] = rng.random(2) * arena
        v = rng.standard_normal(2)
        x[2:] = speed * v / np.linalg.norm(v)
        X[i, 0] = x
        for k in range(K - 1):
            vmag = np.linalg.norm(x[2:])
            vdir = x[2:] / vmag if vmag > 0 else np.zeros(2)
            if np.any(x[:2] < 0) or np.any(x[:2] > arena):
                dm = mid - x[:2]
                vdir = 0.5 * (vdir + dm / np.linalg.norm(dm))
                vdir /= np.linalg.norm(vdir)
                x[2:] = vmag * vdir
            elif rng.random() > change_rate:
                vdir = 0.9 * vdir + 0.1 * rng.standard_normal(2)
                vdir /= np.linalg.norm(vdir)
                x[2:] = vmag * vdir
            vm = np.linalg.norm(x[2:])
            if vm > speed:
                x[2:] *= speed / vm
            x = x.copy()
            x[:2] = x[:2] + dt * x[2:]
            X[i, k + 1] = x
    return X, t


CHI2_2D_3SIGMA = 11.829   # P(chi2_2 <= 11.829) = 0.9973, the 2-D analogue of 3 sigma


def track_report(res) -> pd.DataFrame:
    """One row per target: detection, track life-cycle and consistency.

    first_detect_s    time of the first detection (NaN if never found)
    measurements      number of position fixes that went into the track
    acquisitions      track (re)initialisations: first detection + re-acquisitions
    losses            times the track was declared lost (1-sigma axis > largest FOV)
    longest_lost_s    longest stretch between a loss and the re-acquisition (or end)
    longest_blind_s   longest stretch with an active track but no measurement
    visits            bursts of fixes (a UAV passing over the target); fixes more than
                      one step apart start a new visit
    median_gap_between_visits_s  median time between the end of one visit and the next
    tracked_pct       % of the whole run with an active track
    tracked_after_pct % of the time after first detection with an active track
    rmse_pos_m        position RMSE while active
    outside_3sigma_pct  % of active time the truth lies outside the 3-sigma
                      position ellipse - the filter is confidently wrong
                      ("track failure" in the sense of inconsistency)
    """
    st, t, truth = res["status"], res["t"], res["truth"]
    ex, eP = res["est_x"], res["est_P"]
    dt = t[1] - t[0]
    meas = res["meas"]
    rows = {}
    for i in range(truth.shape[0]):
        s = st[i]
        act = s == 1
        r = {}
        mk = sorted({k for (k, j, ii, _) in meas if ii == i})
        r["measurements"] = len([m for m in meas if m[2] == i])
        if not act.any():
            r.update(first_detect_s=np.nan, visits=0, median_gap_between_visits_s=np.nan, acquisitions=0, losses=0, longest_lost_s=np.nan,
                     longest_blind_s=np.nan, tracked_pct=0.0, tracked_after_pct=0.0,
                     rmse_pos_m=np.nan, outside_3sigma_pct=np.nan)
            rows[f"T{i}"] = r
            continue
        k0 = int(np.argmax(act))
        r["first_detect_s"] = float(t[k0])
        trans = np.diff(s.astype(int))
        r["losses"] = int(np.sum((s[:-1] == 1) & (s[1:] == 2)))
        r["acquisitions"] = 1 + int(np.sum((s[:-1] == 2) & (s[1:] == 1)))
        # longest lost run
        runs, cur = [], 0
        for v in s[k0:]:
            cur = cur + 1 if v == 2 else 0
            runs.append(cur)
        r["longest_lost_s"] = float(max(runs) * dt)
        # longest active stretch without a fix
        mset = set(mk)
        blind, cur = 0, 0
        for k in range(k0, len(t)):
            if s[k] == 1 and k not in mset:
                cur += 1
                blind = max(blind, cur)
            else:
                cur = 0
        r["longest_blind_s"] = float(blind * dt)
        # a "visit" is a burst of fixes on consecutive (or every other) steps
        gaps = np.diff(mk) if len(mk) > 1 else np.array([])
        r["visits"] = 1 + int(np.sum(gaps > 2)) if len(mk) else 0
        long_gaps = gaps[gaps > 2] * dt
        r["median_gap_between_visits_s"] = float(np.median(long_gaps)) if len(long_gaps) else np.nan
        r["tracked_pct"] = 100 * act.mean()
        r["tracked_after_pct"] = 100 * act[k0:].mean()
        e = truth[i, act, :2] - ex[i, act, :2]
        r["rmse_pos_m"] = float(np.sqrt(np.mean(np.sum(e**2, 1))))
        d2 = np.array([ee @ np.linalg.solve(P[:2, :2], ee) for ee, P in zip(e, eP[i, act])])
        r["outside_3sigma_pct"] = 100 * float(np.mean(d2 > CHI2_2D_3SIGMA))
        rows[f"T{i}"] = r
        del trans
    cols = ["first_detect_s", "measurements", "acquisitions", "losses", "longest_lost_s",
            "longest_blind_s", "visits", "median_gap_between_visits_s", "tracked_pct", "tracked_after_pct", "rmse_pos_m", "outside_3sigma_pct"]
    return pd.DataFrame(rows).T[cols]


def report_summary(rep: pd.DataFrame) -> dict:
    found = rep["first_detect_s"].notna()
    return dict(found=int(found.sum()), n=len(rep),
                median_first_detect_s=float(rep["first_detect_s"].median()),
                losses=int(rep["losses"].sum()), reacquisitions=int((rep["acquisitions"] - 1).clip(lower=0).sum()),
                tracked_pct=float(rep["tracked_pct"].mean()),
                tracked_after_pct=float(rep.loc[found, "tracked_after_pct"].mean()) if found.any() else 0.0,
                rmse_pos_m=float(rep["rmse_pos_m"].mean()),
                visits=int(rep["visits"].sum()),
                median_gap_between_visits_s=float(rep["median_gap_between_visits_s"].median()),
                measurements=int(rep["measurements"].sum()),
                outside_3sigma_pct=float(rep["outside_3sigma_pct"].mean()))
