"""Two-view stereo datasets with GT disparity and an occlusion (non-occluded) mask.

Every loader yields dicts: name, left, right (H,W,3 uint8 RGB), gt (H,W float32 disparity, 0 = no GT),
nocc (bool, True where the left pixel is also visible in the right image), max_disp (int hint).

Layout under DATA_ROOT (written by setup/download_datasets.sh):
  middeval3/MiddEval3/trainingH/<scene>/{im0.png,im1.png,calib.txt,disp0GT.pfm,mask0nocc.png}
  eth3d_two_view/<scene>/{im0.png,im1.png,disp0GT.pfm,mask0nocc.png}
  kitti2015/training/{image_2,image_3,disp_occ_0,disp_noc_0}/<id>_10.png
  middlebury2014/<scene>-perfect/{im0.png,im1.png,disp0.pfm,disp1.pfm,calib.txt}
"""
import glob
import os

import cv2
import numpy as np

from ..io import read_image_rgb, read_kitti_disp, read_pfm


def _calib_ndisp(path, default=256):
    if os.path.exists(path):
        for ln in open(path):
            if ln.startswith("ndisp="):
                return int(ln.split("=")[1])
    return default


def _middlebury_style(scene_dir, name):
    gt = read_pfm(os.path.join(scene_dir, "disp0GT.pfm"))
    gt[~np.isfinite(gt)] = 0
    mask = cv2.imread(os.path.join(scene_dir, "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    nocc = mask == 255 if mask is not None else gt > 0
    return dict(name=name, left=read_image_rgb(os.path.join(scene_dir, "im0.png")),
                right=read_image_rgb(os.path.join(scene_dir, "im1.png")), gt=gt, nocc=nocc,
                max_disp=_calib_ndisp(os.path.join(scene_dir, "calib.txt")))


def middeval3(root, limit=None):
    scenes = sorted(glob.glob(os.path.join(root, "middeval3", "MiddEval3", "trainingH", "*")))
    for d in scenes[:limit]:
        if os.path.exists(os.path.join(d, "disp0GT.pfm")):
            yield _middlebury_style(d, "middeval3/" + os.path.basename(d))


def eth3d_two_view(root, limit=None):
    scenes = sorted(glob.glob(os.path.join(root, "eth3d_two_view", "*")))
    for d in scenes[:limit]:
        if os.path.exists(os.path.join(d, "disp0GT.pfm")):
            yield _middlebury_style(d, "eth3d_two_view/" + os.path.basename(d))


def kitti2015(root, limit=None):
    base = os.path.join(root, "kitti2015", "training")
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(base, "disp_occ_0", "*_10.png")))
    for i in ids[:limit]:
        occ = read_kitti_disp(os.path.join(base, "disp_occ_0", i + ".png"))
        noc = read_kitti_disp(os.path.join(base, "disp_noc_0", i + ".png"))
        yield dict(name="kitti2015/" + i, left=read_image_rgb(os.path.join(base, "image_2", i + ".png")),
                   right=read_image_rgb(os.path.join(base, "image_3", i + ".png")), gt=occ, nocc=noc > 0,
                   max_disp=256)


def middlebury2014(root, limit=None, scale=0.5):
    """Full-resolution 2014 scenes, downscaled by `scale` (disparities scaled too). Occlusion mask from a
    left-right consistency check on the two GT disparity maps."""
    scenes = sorted(glob.glob(os.path.join(root, "middlebury2014", "*-perfect")))
    for d in scenes[:limit]:
        d0 = read_pfm(os.path.join(d, "disp0.pfm"))
        d1 = read_pfm(os.path.join(d, "disp1.pfm"))
        d0[~np.isfinite(d0)] = 0
        d1[~np.isfinite(d1)] = 0
        h, w = d0.shape
        u = np.arange(w)[None].repeat(h, 0)
        x1 = np.round(u - d0).astype(int)
        inb = (x1 >= 0) & (x1 < w) & (d0 > 0)
        back = np.zeros_like(d0)
        rows = np.arange(h)[:, None].repeat(w, 1)
        back[inb] = d1[rows[inb], x1[inb]]
        nocc = inb & (np.abs(back - d0) <= 1.0)
        left = read_image_rgb(os.path.join(d, "im0.png"))
        right = read_image_rgb(os.path.join(d, "im1.png"))
        if scale != 1:
            sz = (int(w * scale), int(h * scale))
            left = cv2.resize(left, sz, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, sz, interpolation=cv2.INTER_AREA)
            d0 = cv2.resize(d0, sz, interpolation=cv2.INTER_NEAREST) * scale
            nocc = cv2.resize(nocc.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0
        yield dict(name="middlebury2014/" + os.path.basename(d).replace("-perfect", ""), left=left, right=right,
                   gt=d0, nocc=nocc, max_disp=int(_calib_ndisp(os.path.join(d, "calib.txt")) * scale))


DATASETS = {"middeval3": middeval3, "eth3d_two_view": eth3d_two_view, "kitti2015": kitti2015,
            "middlebury2014": middlebury2014}
