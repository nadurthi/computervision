"""Error metrics for disparity (stereo) and depth (MVS), always reported per region so occlusion effects show up.

A missing prediction (0 / NaN / inf) is never silently dropped: errors are computed on predicted pixels, and the
inlier / bad-pixel rates are computed over *all* GT pixels of the region with missing predictions counted as failures.
`density` reports the fraction of GT pixels that got a prediction.
"""
import numpy as np


def _valid_pred(x):
    return np.isfinite(x) & (x > 0)


def disparity_metrics(pred, gt, mask, thresholds=(0.5, 1.0, 2.0, 3.0)):
    m = mask & (gt > 0) & np.isfinite(gt)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    has = m & _valid_pred(pred)
    err = np.abs(np.where(has, pred, 0) - gt)
    out = {"n": n, "density": float(has.sum() / n)}
    out["epe"] = float(err[has].mean()) if has.any() else float("nan")
    for t in thresholds:
        bad = m & (~has | (err > t))
        out[f"bad{t:g}"] = float(100.0 * bad.sum() / n)
    d1 = m & (~has | ((err > 3.0) & (err > 0.05 * gt)))       # KITTI D1
    out["d1"] = float(100.0 * d1.sum() / n)
    return out


def depth_metrics(pred, gt, mask, abs_thresholds=(), rel_thresholds=(1.03, 1.10)):
    """abs_thresholds in GT units (e.g. mm for DTU); rel_thresholds are max(pred/gt, gt/pred) < t (inliers)."""
    m = mask & (gt > 0) & np.isfinite(gt)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    has = m & _valid_pred(pred)
    p = np.where(has, pred, 1.0)
    g = np.where(m, gt, 1.0)
    abs_err = np.abs(p - g)
    ratio = np.maximum(p / g, g / p)
    out = {"n": n, "density": float(has.sum() / n)}
    if has.any():
        out["abs_rel"] = float(100.0 * (abs_err[has] / g[has]).mean())     # percent
        out["mae"] = float(abs_err[has].mean())
        out["median_ae"] = float(np.median(abs_err[has]))
    else:
        out["abs_rel"] = out["mae"] = out["median_ae"] = float("nan")
    for t in rel_thresholds:
        out[f"inlier{t:g}"] = float(100.0 * (has & (ratio < t)).sum() / n)
    for t in abs_thresholds:
        out[f"within{t:g}"] = float(100.0 * (has & (abs_err < t)).sum() / n)
    return out


def aggregate(rows, key_fields):
    """Pixel-weighted mean of metric dicts grouped by key_fields. rows: list of dict(**keys, **metrics, n=..)."""
    groups = {}
    for r in rows:
        if not r.get("n"):
            continue
        k = tuple(r[f] for f in key_fields)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rs in groups.items():
        tot = sum(r["n"] for r in rs)
        agg = dict(zip(key_fields, k))
        agg["n"] = tot
        agg["samples"] = len(rs)
        metric_keys = [m for m in rs[0] if m not in key_fields and m not in ("n",) and isinstance(rs[0][m], (int, float))]
        for m in metric_keys:
            vals = [(r[m], r["n"]) for r in rs if m in r and np.isfinite(r[m])]
            if vals:
                agg[m] = float(sum(v * w for v, w in vals) / sum(w for _, w in vals))
        out.append(agg)
    return out
