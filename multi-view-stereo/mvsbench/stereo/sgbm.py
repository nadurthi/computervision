"""OpenCV semi-global block matching (Hirschmuller 2008, SGBM variant) as the classical two-view baseline."""
import cv2
import numpy as np


class SGBM:
    def __init__(self, block=5, fill=False, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):
        self.block, self.fill, self.mode = block, fill, mode
        self.name = "sgbm_fill" if fill else "sgbm"

    def __call__(self, left, right, max_disp=256):
        nd = int(np.ceil(max(max_disp, 16) / 16.0) * 16)
        b = self.block
        sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=b, P1=8 * 3 * b * b,
                                     P2=32 * 3 * b * b, disp12MaxDiff=1, uniquenessRatio=10,
                                     speckleWindowSize=100, speckleRange=2, preFilterCap=63, mode=self.mode)
        disp = sgbm.compute(left, right).astype(np.float32) / 16.0
        disp[disp <= 0] = 0
        if self.fill:
            disp = fill_background(disp)
        return disp


def fill_background(disp):
    """Fill holes along each scanline with the smaller (farther) of the nearest valid neighbours, the usual
    Middlebury-style occlusion fill: occluded pixels belong to the background surface."""
    out = disp.copy()
    h, w = disp.shape
    idx = np.arange(w)
    for y in range(h):
        row = disp[y]
        valid = row > 0
        if not valid.any() or valid.all():
            continue
        vi = idx[valid]
        left_i = np.maximum.accumulate(np.where(valid, idx, -1))
        right_i = np.minimum.accumulate(np.where(valid, idx, w)[::-1])[::-1]
        lv = np.where(left_i >= 0, row[np.clip(left_i, 0, w - 1)], np.inf)
        rv = np.where(right_i < w, row[np.clip(right_i, 0, w - 1)], np.inf)
        fillv = np.minimum(lv, rv)
        fillv[~np.isfinite(fillv)] = row[vi[0]]
        out[y, ~valid] = fillv[~valid]
    return out
