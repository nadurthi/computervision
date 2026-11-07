#!/usr/bin/env python3
"""CPU smoke test on a ray-traced synthetic scene with exact depth: geometry, occlusion labels, rectification for
stereo networks (source to the right, left, above, below), SGBM, plane sweep, PatchMatch, fusion, metrics.

  python scripts/smoke_test.py            # ~1-2 minutes on a laptop CPU; exits non-zero on failure
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench import geometry as G      # noqa: E402
from mvsbench.metrics import depth_metrics  # noqa: E402

W, H = 320, 240
K = np.array([[300.0, 0, W / 2 - 0.5], [0, 300.0, H / 2 - 0.5], [0, 0, 1]])
rng = np.random.default_rng(0)
TEX = rng.random((512, 512))
TEX = (TEX + np.roll(TEX, 1, 0) + np.roll(TEX, 1, 1)) / 3


def texture(a, b, freq):
    i = (np.floor(a * freq) % 512).astype(int)
    j = (np.floor(b * freq) % 512).astype(int)
    return TEX[i, j]


def look(center, yaw=0.0, pitch=0.0):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    R_wc = Ry @ Rx                      # camera -> world rotation
    E = np.eye(4)
    E[:3, :3] = R_wc.T
    E[:3, 3] = -R_wc.T @ np.asarray(center, float)
    return E


def render(E):
    """Scene: textured back wall z=10, slanted floor-ish plane, and a textured box face at z=6 in front."""
    u, v = G.pixel_grid(H, W)
    rays_c = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(H * W)])
    Rcw = E[:3, :3].T
    C = -Rcw @ E[:3, 3]
    d = Rcw @ rays_c                                             # world directions (z_cam = 1 scaled)
    best_t = np.full(H * W, np.inf)
    col = np.zeros(H * W)
    # back wall z = 10
    t = (10 - C[2]) / d[2]
    P = C[:, None] + t * d
    ok = (t > 0) & (t < best_t)
    best_t[ok], col[ok] = t[ok], texture(P[0][ok] + 20, P[1][ok] + 20, 30)[...]
    # occluder: square patch at z = 6, x in [-0.8, 0.8], y in [-0.6, 0.9]
    t = (6 - C[2]) / d[2]
    P = C[:, None] + t * d
    ok = (t > 0) & (t < best_t) & (np.abs(P[0]) < 0.8) & (P[1] > -0.6) & (P[1] < 0.9)
    best_t[ok], col[ok] = t[ok], 0.3 + 0.7 * texture(P[0][ok] + 7, P[1][ok] + 3, 45)
    # slanted plane: y = 1.4 + 0.15 (z - 6)  (a floor tilted towards the camera)
    t = (1.4 - 0.15 * 6 + 0.15 * C[2] - C[1]) / (d[1] - 0.15 * d[2])
    P = C[:, None] + t * d
    ok = (t > 0) & (t < best_t) & (P[2] < 10)
    best_t[ok], col[ok] = t[ok], texture(P[0][ok] + 3, P[2][ok] + 3, 25)
    depth = np.where(np.isfinite(best_t), best_t, 0).reshape(H, W)   # rays have z_cam = 1, so t is depth
    img = np.clip(col.reshape(H, W) * 255, 0, 255).astype(np.uint8)
    return np.dstack([img] * 3), depth.astype(np.float32)


def check(name, cond, msg=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {msg}" if msg else ""))
    if not cond:
        check.failed = True


check.failed = False


def main():
    E0 = look([0, 0, 0])
    sources = {"right": look([0.6, 0, 0], yaw=-0.03), "left": look([-0.6, 0, 0], yaw=0.03),
               "up": look([0, -0.5, 0], pitch=0.03), "down": look([0, 0.5, 0.1]),
               "far_right": look([1.2, 0.1, 0], yaw=-0.06)}
    img0, d0 = render(E0)
    views = {k: render(E) for k, E in sources.items()}

    # 1) reprojection consistency and occlusion labels
    us, vs, zs = G.reproject(d0, K, E0, K, sources["right"])
    lab = G.visibility_in_source(d0, K, E0, views["right"][1], K, sources["right"])
    frac = {n: float((lab == c).mean()) for n, c in (("vis", G.VISIBLE), ("occ", G.OCCLUDED), ("oov", G.OUT_OF_VIEW))}
    check("visibility labels", frac["vis"] > 0.6 and 0.005 < frac["occ"] < 0.2, str(frac))
    cnt, _ = G.visibility_count(d0, K, E0, [(views[k][1], K, sources[k]) for k in ("right", "left", "up")])
    occ_r = lab == G.OCCLUDED
    check("occluded-in-right pixels are mostly seen by another view", (cnt[occ_r] >= 1).mean() > 0.8,
          f"{(cnt[occ_r] >= 1).mean():.2f}")

    # 2) rotation <-> quaternion (COLMAP export)
    from mvsbench.io import qvec2rotmat
    from mvsbench.mvs.colmap import rotmat2qvec
    R = sources["far_right"][:3, :3] @ look([0, 0, 0], 0.4, -0.3)[:3, :3]
    check("rotmat2qvec round trip", np.allclose(qvec2rotmat(rotmat2qvec(R)), R, atol=1e-8))

    # 2b) MVSNet cam files: "min interval" (MVSNet) and "min max" (PatchmatchNet/IterMVS DTU cams_1)
    import tempfile
    from mvsbench.io import read_mvsnet_cam
    for line, want in (("425.0 2.5", (425.0, 425.0 + 2.5 * 191)), ("425.0 935.0", (425.0, 935.0)),
                       ("425.0 2.5 192 935.0", (425.0, 935.0))):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("extrinsic\n" + "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n\nintrinsic\n1 0 0\n0 1 0\n0 0 1\n\n" + line + "\n")
        _, _, dmin, dmax, _ = read_mvsnet_cam(f.name)
        os.unlink(f.name)
        check(f"cam depth range '{line}'", abs(dmin - want[0]) < 1e-6 and abs(dmax - want[1]) < 1e-6, f"{dmin}, {dmax}")

    # 3) rectification + SGBM -> reference depth, for every baseline direction
    from mvsbench.stereo.sgbm import SGBM
    from mvsbench.mvs.stereo_as_mvs import StereoAsMVS
    sgbm = StereoAsMVS(SGBM(fill=False))
    for k in ("right", "left", "up", "down"):
        pred = sgbm([img0, views[k][0]], [K, K], [E0, sources[k]], (4.0, 12.0))
        lab_k = G.visibility_in_source(d0, K, E0, views[k][1], K, sources[k])
        m = depth_metrics(pred, d0, lab_k == G.VISIBLE, rel_thresholds=(1.03,))
        check(f"rectified SGBM, source {k}", m["inlier1.03"] > 45 and m["density"] > 0.55,
              f"inlier1.03={m['inlier1.03']:.1f}% density={m['density']:.2f} absrel={m['abs_rel']:.2f}%")

    # 4) GT-disparity round trip through RectifiedPair for every baseline direction (no matcher involved):
    #    render the GT depth into the rectified frame, convert to disparity, map back, compare with GT
    import cv2
    for k in ("right", "left", "up", "down"):
        rp = G.RectifiedPair(K, E0, K, sources[k], (W, H))
        w2, h2 = rp.size
        u, v = G.pixel_grid(h2, w2)
        rays_ref = rp.R1.T @ (np.linalg.inv(rp.P1[:3, :3]) @ np.stack([u.ravel(), v.ravel(), np.ones(h2 * w2)]))
        px = K @ (rays_ref / rays_ref[2])
        zr = cv2.remap(d0, px[0].reshape(h2, w2).astype(np.float32), px[1].reshape(h2, w2).astype(np.float32),
                       cv2.INTER_NEAREST, borderValue=0)
        z_rect = zr / rays_ref[2].reshape(h2, w2)
        disp_rect = np.where(z_rect > 0, rp.f * rp.baseline / np.maximum(z_rect, 1e-6), 0).astype(np.float32)
        back = rp.disparity_to_ref_depth(rp._to_net(disp_rect))
        m = depth_metrics(back, d0, d0 > 0, rel_thresholds=(1.01,))
        check(f"rectification round trip, source {k} (vertical={rp.vertical}, mirrored={rp.flip})",
              m["inlier1.01"] > 97, f"{m['inlier1.01']:.2f}%")

    # 5) classical MVS (torch, CPU)
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"SKIP torch-based checks ({e})")
        return
    from mvsbench.mvs.planesweep import PlaneSweep
    from mvsbench.mvs.patchmatch import PatchMatchMVS
    imgs = [img0] + [views[k][0] for k in ("right", "left", "up", "down")]
    Es = [E0] + [sources[k] for k in ("right", "left", "up", "down")]
    Ks = [K] * len(imgs)
    occ_right = lab == G.OCCLUDED
    res = {}
    for name, meth, n in (("planesweep_mean", PlaneSweep(96, aggregation="mean", device="cpu"), 5),
                          ("planesweep_bestk", PlaneSweep(96, aggregation="bestk", device="cpu"), 5),
                          ("planesweep_bestk_sgm", PlaneSweep(96, aggregation="bestk", use_sgm=True, device="cpu"), 5),
                          ("planesweep_2view", PlaneSweep(96, aggregation="mean", device="cpu"), 2),
                          ("patchmatch", PatchMatchMVS(iterations=3, device="cpu"), 5)):
        t0 = time.time()
        pred = meth(imgs[:n], Ks[:n], Es[:n], (4.0, 12.0))
        a = depth_metrics(pred, d0, d0 > 0, rel_thresholds=(1.03,))
        o = depth_metrics(pred, d0, occ_right, rel_thresholds=(1.03,))
        res[name] = (a["inlier1.03"], o["inlier1.03"])
        print(f"     {name:22s} all {a['inlier1.03']:5.1f}%  occluded-in-partner {o['inlier1.03']:5.1f}%  {time.time() - t0:.1f}s")
    check("plane sweep works", res["planesweep_bestk"][0] > 60, str(res["planesweep_bestk"]))
    check("N views beat 2 views on pixels occluded in the partner",
          res["planesweep_bestk"][1] > res["planesweep_2view"][1] + 10, f"{res['planesweep_bestk'][1]:.1f} vs {res['planesweep_2view'][1]:.1f}")
    check("PatchMatch works", res["patchmatch"][0] > 60, str(res["patchmatch"]))

    # 6) fusion runs and produces points near the true surfaces
    from mvsbench.fusion import fuse
    depths = {0: d0, 1: views["right"][1], 2: views["left"][1], 3: views["up"][1]}
    pts, cols = fuse(depths, {i: K for i in depths}, {0: E0, 1: sources["right"], 2: sources["left"], 3: sources["up"]},
                     {0: img0, 1: views["right"][0], 2: views["left"][0], 3: views["up"][0]},
                     [(0, [1, 2, 3]), (1, [0, 2, 3])], min_views=2)
    check("fusion of GT depth keeps most points", len(pts) > 0.5 * (d0 > 0).sum(), f"{len(pts)} points")


if __name__ == "__main__":
    main()
    sys.exit(1 if check.failed else 0)
