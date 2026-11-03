"""Run the scan-matching particle filter (paper Fig. 'Block diagram') over a sequence and score it against ground truth.

Blocks per time step t_k:
  PF  : propagate particles with the constant-turn-rate model, weight with the lookup-table likelihood
  RPE : scan-to-scan ICP -> speed and yaw rate, injected into the particles
  RS  : remove road points (|n_z| >= 0.5)
  S2M : multi-resolution BEV match of the road-free scan for the highest-weight particle j_max, then 3-D ICP
  SU  : overwrite particle j_max with the corrected pose when the match improves the bin cost
"""
import json, time, argparse
import numpy as np
from scipy.spatial import cKDTree
import lidarloc as ll


DEFAULTS = dict(
    Npf=1000, n_meas=1500, voxel_scan=0.5, voxel_map=0.25, lookup_res=0.5, dmax=3.0, sig0=0.2,
    z_below=2.0, z_above=12.0,       # map height band kept around the ground level (bounds the lookup tensor)
    noise_scale=2.0,                 # driver: X += 2 * N(0, Q)
    resample_frac=0.25,
    s2m_every=1, Lmax=20.0, thmax_deg=30.0, thmin_deg=1.0, dxMatch=0.5, dxBase=30.0,
    init="tracking",                 # "tracking": N(gt0, 2 m, 10 deg) ; "global": uniform over the map
    init_pos_sig=2.0, init_yaw_sig_deg=10.0,
    use_rpe=True, max_range=60.0, seed=0,
    adaptive=False, recheck_every=0, wide_when_spread_m=5.0, Lmax_wide=80.0, thmax_wide_deg=180.0, thmin_wide_deg=2.0,
)


def init_particles(cfg, H0, map_lb, map_ub, rng):
    N = cfg["Npf"]
    X = np.zeros((N, 11))
    p0 = ll.H_to_pose(H0)
    if cfg["init"] == "global":
        X[:, 0:2] = rng.uniform(map_lb[:2], map_ub[:2], (N, 2))
        X[:, 2] = p0[2] + rng.normal(0, 0.5, N)
        X[:, 3] = rng.uniform(-np.pi, np.pi, N)
    else:
        X[:, 0:2] = p0[:2] + rng.normal(0, cfg["init_pos_sig"], (N, 2))
        X[:, 2] = p0[2] + rng.normal(0, 0.2, N)
        X[:, 3] = p0[3] + rng.normal(0, np.deg2rad(cfg["init_yaw_sig_deg"]), N)
    X[:, 4] = p0[4] + rng.normal(0, np.deg2rad(1), N)
    X[:, 5] = p0[5] + rng.normal(0, np.deg2rad(0.5), N)
    X[:, 6] = np.abs(rng.normal(0, 5, N))
    X[:, 7:10] = rng.normal(0, 1e-3, (N, 3))
    X[:, 10] = 0.01 * np.abs(rng.normal(size=N))
    return X


def run(frames, Xmap, cfg, log=print):
    """frames: list of dict(t, X (sensor-frame points Nx3), H (4x4 sensor->map ground truth)).
    Returns per-frame records and the timing breakdown."""
    cfg = {**DEFAULTS, **cfg}
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    zg = np.percentile(Xmap[:, 2], 5)                                   # ground level from the map itself
    Xmap = Xmap[(Xmap[:, 2] > zg - cfg["z_below"]) & (Xmap[:, 2] < zg + cfg["z_above"])]
    lut = ll.DistanceLookup(ll.voxel_down(Xmap, cfg["voxel_map"]), res=cfg["lookup_res"], dmax=cfg["dmax"])
    Xmap_icp = ll.voxel_down(Xmap, 0.4)
    tree_map, nrm_map = cKDTree(Xmap_icp), ll.normals(Xmap_icp)
    map_rr = ll.remove_road(ll.voxel_down(Xmap, 0.3))
    Xmap2 = ll.voxel_down(map_rr[:, :2], cfg["dxMatch"])
    t_setup = time.time() - t0
    log(f"map: {len(Xmap):,} pts -> lookup {lut.mb:.0f} MB, BEV {len(Xmap2):,} cells, setup {t_setup:.1f}s")

    pf = ll.ParticleFilter(init_particles(cfg, frames[0]["H"], Xmap.min(0), Xmap.max(0), rng), rng)
    recs, prev, prevH = [], None, None
    T = dict(read=0, rpe=0, pf=0, s2m=0)
    for k, fr in enumerate(frames):
        s = time.time()
        X = fr["X"][np.linalg.norm(fr["X"][:, :2], axis=1) < cfg["max_range"]]
        Xd = ll.voxel_down(X, cfg["voxel_scan"])
        Xs = Xd[rng.choice(len(Xd), min(cfg["n_meas"], len(Xd)), replace=False)]
        T["read"] += time.time() - s
        dt = fr["t"] - frames[k - 1]["t"] if k else 0.0

        # --- RPE: scan-to-scan ICP for speed / yaw rate
        s = time.time()
        rpe = None
        if cfg["use_rpe"] and prev is not None and dt > 0:
            tree_prev, nrm_prev = cKDTree(prev), ll.normals(prev)
            Hrel = ll.icp_point_to_plane(Xd, tree_prev, prev, nrm_prev, np.eye(4), iters=15, max_corr=1.5)
            rpe = (Hrel[0, 3] / dt, ll.R2ypr(Hrel[:3, :3])[0] / dt)       # body-frame forward speed, yaw rate
        T["rpe"] += time.time() - s

        # --- PF: propagate + weight
        s = time.time()
        if k:
            if rpe is not None:
                pf.X[:, 6] = rpe[0] + rng.normal(0, 0.3, len(pf.X))
                pf.X[:, 7] = rpe[1] + rng.normal(0, np.deg2rad(1), len(pf.X))
            pf.X = ll.dynmodelCT3D(pf.X, dt)
            pf.X += cfg["noise_scale"] * rng.multivariate_normal(np.zeros(11), ll.Q_CT, len(pf.X))
        nll = lut.neg_log_lik(pf.X, Xs, cfg["sig0"])
        pf.logw += -nll
        T["pf"] += time.time() - s

        # --- RS + S2M + SU on the best particle
        s = time.time()
        s2m = None
        spread_wide = bool(cfg.get("adaptive") and recs and recs[-1]["spread"] > cfg["wide_when_spread_m"])
        recheck = bool(cfg.get("recheck_every") and k and k % cfg["recheck_every"] == 0)   # periodic wide re-localization
        every = 1 if spread_wide else cfg["s2m_every"]
        if (every and k % every == 0) or recheck:
            j = int(np.argmax(pf.logw))
            xj = pf.X[j]
            Xrr = ll.remove_road(ll.voxel_down(X, 0.2))
            X2 = ll.voxel_down(Xrr[:, :2], cfg["dxMatch"])
            Hg = np.eye(3); c, sn = np.cos(xj[3]), np.sin(xj[3])
            Hg[:2, :2] = [[c, -sn], [sn, c]]; Hg[:2, 2] = xj[:2]
            wide = spread_wide or recheck
            Lmax = cfg["Lmax_wide"] if wide else cfg["Lmax"]
            thmax = cfg["thmax_wide_deg"] if wide else cfg["thmax_deg"]
            thmin = cfg["thmin_wide_deg"] if wide else cfg["thmin_deg"]
            win = np.all(np.abs(Xmap2 - xj[:2]) < Lmax + cfg["max_range"] + 5, axis=1)
            if win.sum() > 50 and len(X2) > 30:
                H2, c0, c1 = ll.bin_matcher(Xmap2[win], X2, Hg, np.full(2, Lmax), np.deg2rad(thmax),
                                            np.deg2rad(thmin), np.full(2, cfg["dxMatch"]), np.full(2, cfg["dxBase"]))
                s2m = dict(cost0=c0, cost=c1, accepted=bool(c1 > c0), wide=bool(wide))
                if c1 > c0:
                    xn = xj.copy(); xn[:2] = H2[:2, 2]; xn[3] = np.arctan2(H2[1, 0], H2[0, 0])
                    Hn = ll.icp_point_to_plane(Xd, tree_map, Xmap_icp, nrm_map, ll.pose_to_H(xn), iters=20, max_corr=1.0)
                    pf.X[j, :6] = ll.H_to_pose(Hn)
                    pf.logw[j] = pf.logw.max() - lut.neg_log_lik(pf.X[j:j + 1], Xs, cfg["sig0"])[0] + nll[j]
        T["s2m"] += time.time() - s

        if pf.neff() / len(pf.X) < cfg["resample_frac"]:
            pf.resample()
        est = pf.estimate()
        gt = ll.H_to_pose(fr["H"])
        jb = int(np.argmax(pf.logw)); best = pf.X[jb]
        recs.append(dict(k=k, t=fr["t"], est=est[:6].tolist(), gt=gt.tolist(), best=best[:6].tolist(),
                         exy_best=float(np.linalg.norm(best[:2] - gt[:2])), eyaw_best=float(np.rad2deg(abs(ll.wrap(best[3] - gt[3])))),
                         exy=float(np.linalg.norm(est[:2] - gt[:2])), eyaw=float(np.rad2deg(abs(ll.wrap(est[3] - gt[3])))),
                         spread=float(np.sqrt(np.trace(np.cov(pf.X[:, :2].T, aweights=pf.w)))), neff=float(pf.neff()),
                         rpe=rpe, s2m=s2m))
        prev = Xd
        if k % 10 == 0:
            log(f"k={k:4d} xy err {recs[-1]['exy']:6.2f} m  yaw err {recs[-1]['eyaw']:6.2f} deg  spread {recs[-1]['spread']:6.2f} m"
                + (f"  s2m {s2m['cost0']:.0f}->{s2m['cost']:.0f}" if s2m else ""))
    n = len(frames)
    timing = {k: v / n for k, v in T.items()}
    timing["setup_s"] = t_setup
    return recs, timing


def summarize(recs, settle_m=1.0):
    e = np.array([r["exy"] for r in recs]); y = np.array([r["eyaw"] for r in recs])
    ok = e < settle_m
    conv = next((i for i in range(len(e)) if ok[i] and ok[i:].mean() >= 0.98), None)  # within 1 m from here on (98%)
    post = slice(conv, None) if conv is not None else slice(len(e), None)
    tt = np.array([r["t"] for r in recs]) - recs[0]["t"]
    return dict(n=len(e), converged_at=conv, converged_after_s=float(tt[conv]) if conv is not None else None,
                rmse_xy=float(np.sqrt(np.mean(e[post] ** 2))) if conv is not None else None,
                median_xy=float(np.median(e[post])) if conv is not None else None,
                p95_xy=float(np.percentile(e[post], 95)) if conv is not None else None,
                median_yaw=float(np.median(y[post])) if conv is not None else None,
                frac_within_1m=float(ok.mean()), final_xy=float(e[-1]),
                path_m=float(np.sum(np.linalg.norm(np.diff(np.array([r["gt"][:2] for r in recs]), axis=0), axis=1))),
                mean_speed=float(np.sum(np.linalg.norm(np.diff(np.array([r["gt"][:2] for r in recs]), axis=0), axis=1)) / max(tt[-1], 1e-6)))
