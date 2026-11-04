"""Calibrated multi-view datasets with per-view GT depth (needed to label occlusions in every source view).

Each loader returns Scene objects. A Scene knows its views (image path, K, world->camera E), a view-selection
list (ref -> source ids ranked best first), a depth range per reference view, and loads GT depth per view.

Layout under DATA_ROOT (written by setup/download_datasets.sh):
  dtu/eval/scan<N>/{images/%08d.jpg, cams/%08d_cam.txt, pair.txt}     PatchmatchNet's DTU evaluation set (1600x1200)
  dtu/Depths/scan<N>/depth_map_%04d.pfm  (or dtu/Depths_raw/...)      MVSNet / CasMVSNet GT depth, mm, 1600x1200
  eth3d_mvs/<scene>/images/dslr_images_undistorted/*.JPG
  eth3d_mvs/<scene>/dslr_calibration_undistorted/{cameras,images,points3D}.txt
  eth3d_mvs/<scene>/ground_truth_depth/dslr_images_undistorted_by_rmvd/*.JPG   (raw float32, from rmvd's script)
  blendedmvs/<PID>/{blended_images/%08d.jpg, cams/%08d_cam.txt, cams/pair.txt, rendered_depth_maps/%08d.pfm}
"""
import glob
import math
import os
from dataclasses import dataclass, field

import numpy as np

from ..geometry import resize_view
from ..io import (colmap_pinhole_K, read_colmap_cameras_txt, read_colmap_images_txt, read_eth3d_depth,
                  read_image_rgb, read_mvsnet_cam, read_pair, read_pfm)


@dataclass
class View:
    id: int
    image_path: str
    K: np.ndarray
    E: np.ndarray
    hw: tuple


@dataclass
class Scene:
    dataset: str
    name: str
    views: dict
    pairs: list                    # [(ref_id, [src ids, best first])]
    units: str
    abs_thresholds: tuple          # in `units`, for "within" metrics
    work_wh: tuple                 # resolution every method is run at (methods may pick their own)
    eval_wh: tuple                 # resolution every prediction is scored at
    gt_loader: object = None
    range_fn: object = None
    meta: dict = field(default_factory=dict)

    def gt_depth(self, vid):
        return self.gt_loader(vid)

    def depth_range(self, vid):
        return self.range_fn(vid)

    def sources(self, ref, n_src):
        for r, s in self.pairs:
            if r == ref:
                return s[:n_src]
        raise KeyError(ref)

    def load(self, ids, wh=None):
        """-> images (list HxWx3 uint8), Ks (list 3x3 at that size), Es (list 4x4)."""
        wh = wh or self.work_wh
        imgs, Ks, Es = [], [], []
        for i in ids:
            v = self.views[i]
            img, K = resize_view(read_image_rgb(v.image_path), v.K, wh)
            imgs.append(img)
            Ks.append(K)
            Es.append(v.E)
        return imgs, Ks, Es


# ------------------------------------------------------------------------------------------------ DTU
DTU_TEST_SCANS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]


def _dtu_gt_path(root, scan, vid):
    for sub in ("Depths", "Depths_raw"):
        p = os.path.join(root, "dtu", sub, f"scan{scan}", f"depth_map_{vid:04d}.pfm")
        if os.path.exists(p):
            return p
    return None


def dtu(root, scans=None, work_wh=(800, 600), eval_wh=(800, 600)):
    scans = scans or DTU_TEST_SCANS
    out = []
    for scan in scans:
        d = os.path.join(root, "dtu", "eval", f"scan{scan}")
        if not os.path.isdir(d) or _dtu_gt_path(root, scan, 0) is None:
            print(f"[dtu] skip scan{scan}: images or GT depth missing")
            continue
        cam_dir = os.path.join(d, "cams_1") if os.path.isdir(os.path.join(d, "cams_1")) else os.path.join(d, "cams")
        views, ranges = {}, {}
        for p in sorted(glob.glob(os.path.join(d, "images", "*.jpg"))):
            vid = int(os.path.basename(p)[:8])
            K, E, dmin, dmax, _ = read_mvsnet_cam(os.path.join(cam_dir, f"{vid:08d}_cam.txt"))
            views[vid] = View(vid, p, K, E, (1200, 1600))
            ranges[vid] = (dmin or 425.0, dmax or 935.0)

        def gt(vid, scan=scan):
            g = read_pfm(_dtu_gt_path(root, scan, vid))
            g[~np.isfinite(g)] = 0
            return g

        out.append(Scene("dtu", f"scan{scan}", views, read_pair(os.path.join(d, "pair.txt")), "mm", (1.0, 2.0, 4.0),
                         work_wh, eval_wh, gt_loader=gt, range_fn=lambda vid, r=ranges: r[vid]))
    return out


# ------------------------------------------------------------------------------------------------ ETH3D
ETH3D_SCENES = ["courtyard", "delivery_area", "electro", "facade", "kicker", "meadow", "office", "pipes",
                "playground", "relief", "relief_2", "terrace", "terrains"]


def _read_points3d(path):
    """COLMAP points3D.txt -> xyz (P,3), tracks list of image-id arrays."""
    xyz, tracks = [], []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        xyz.append([float(t[1]), float(t[2]), float(t[3])])
        tracks.append(np.array(t[8::2], dtype=np.int64))
    return np.array(xyz), tracks


def mvsnet_view_scores(centers, xyz, tracks, image_ids, theta0=5.0, sigma1=1.0, sigma2=10.0):
    """MVSNet view-selection score from sparse SfM points: sum over shared points of a Gaussian on the
    triangulation angle (peak at theta0 degrees)."""
    idx = {iid: k for k, iid in enumerate(image_ids)}
    n = len(image_ids)
    score = np.zeros((n, n))
    for X, tr in zip(xyz, tracks):
        ks = [idx[i] for i in tr if i in idx]
        if len(ks) < 2:
            continue
        rays = centers[ks] - X
        rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
        cos = np.clip(rays @ rays.T, -1, 1)
        theta = np.degrees(np.arccos(cos))
        s = np.where(theta <= theta0, np.exp(-(theta - theta0) ** 2 / (2 * sigma1 ** 2)),
                     np.exp(-(theta - theta0) ** 2 / (2 * sigma2 ** 2)))
        np.fill_diagonal(s, 0)
        score[np.ix_(ks, ks)] += s
    return score


def eth3d_mvs(root, scenes=None, work_wh=(1152, 768), eval_wh=(1152, 768), max_src=10):
    scenes = scenes or ETH3D_SCENES
    out = []
    for name in scenes:
        d = os.path.join(root, "eth3d_mvs", name)
        calib = os.path.join(d, "dslr_calibration_undistorted")
        gt_dir = os.path.join(d, "ground_truth_depth", "dslr_images_undistorted_by_rmvd")
        if not (os.path.isdir(calib) and os.path.isdir(gt_dir)):
            print(f"[eth3d] skip {name}: calibration or undistorted GT depth missing")
            continue
        cams = read_colmap_cameras_txt(os.path.join(calib, "cameras.txt"))
        imgs = read_colmap_images_txt(os.path.join(calib, "images.txt"))
        # COLMAP image ids are the 1-based order in images.txt only by convention; read them explicitly
        image_ids = []
        for ln in open(os.path.join(calib, "images.txt")):
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            if len(t) >= 10 and t[-1].lower().endswith((".jpg", ".png")):
                image_ids.append(int(t[0]))
        views, centers = {}, []
        for k, im in enumerate(imgs):
            cam = cams[im["camera_id"]]
            path = os.path.join(d, "images", "dslr_images_undistorted", os.path.basename(im["name"]))
            views[k] = View(k, path, colmap_pinhole_K(cam), im["E"], (cam["h"], cam["w"]))
            R, t = im["E"][:3, :3], im["E"][:3, 3]
            centers.append(-R.T @ t)
        centers = np.array(centers)
        xyz, tracks = _read_points3d(os.path.join(calib, "points3D.txt"))
        score = mvsnet_view_scores(centers, xyz, tracks, image_ids)
        pairs = [(k, [int(j) for j in np.argsort(-score[k]) if j != k and score[k, j] > 0][:max_src]) for k in views]
        ranges = {}
        seen_by = {}
        for p_i, tr in enumerate(tracks):
            for iid in set(tr.tolist()):
                seen_by.setdefault(iid, []).append(p_i)
        for k, v in views.items():
            Xc = (v.E[:3, :3] @ xyz.T + v.E[:3, 3:4])[2]
            idx = np.array(seen_by.get(image_ids[k], []), dtype=np.int64)
            z = Xc[idx][Xc[idx] > 0] if idx.size else Xc[Xc > 0]
            ranges[k] = (float(np.percentile(z, 1) * 0.75), float(np.percentile(z, 99) * 1.25)) if z.size else (0.3, 50.0)

        def gt(vid, views=views, gt_dir=gt_dir):
            v = views[vid]
            return read_eth3d_depth(os.path.join(gt_dir, os.path.basename(v.image_path)), *v.hw)

        out.append(Scene("eth3d", name, views, pairs, "m", (0.02, 0.05, 0.10), work_wh, eval_wh, gt_loader=gt,
                         range_fn=lambda vid, r=ranges: r[vid]))
    return out


# ------------------------------------------------------------------------------------------------ BlendedMVS
def blendedmvs(root, scenes=None, work_wh=(768, 576), eval_wh=(768, 576)):
    base = os.path.join(root, "blendedmvs")
    if scenes is None:
        lst = os.path.join(base, "validation_list.txt")
        scenes = [s.strip() for s in open(lst)] if os.path.exists(lst) else []
    out = []
    for pid in scenes:
        d = os.path.join(base, pid)
        if not os.path.isdir(d):
            continue
        views, ranges = {}, {}
        for p in sorted(glob.glob(os.path.join(d, "blended_images", "*.jpg"))):
            if "_masked" in p:
                continue
            vid = int(os.path.basename(p)[:8])
            K, E, dmin, dmax, _ = read_mvsnet_cam(os.path.join(d, "cams", f"{vid:08d}_cam.txt"))
            views[vid] = View(vid, p, K, E, (576, 768))
            ranges[vid] = (dmin, dmax)

        def gt(vid, d=d):
            g = read_pfm(os.path.join(d, "rendered_depth_maps", f"{vid:08d}.pfm"))
            g[~np.isfinite(g)] = 0
            return g

        rng = [r[1] - r[0] for r in ranges.values()]
        scale = float(np.median(rng)) if rng else 1.0
        out.append(Scene("blendedmvs", pid, views, read_pair(os.path.join(d, "cams", "pair.txt")), "units",
                         tuple(scale * f for f in (0.005, 0.01, 0.02)), work_wh, eval_wh, gt_loader=gt,
                         range_fn=lambda vid, r=ranges: r[vid]))
    return out


DATASETS = {"dtu": dtu, "eth3d": eth3d_mvs, "blendedmvs": blendedmvs}
