"""Two-view stereo used as a depth estimator inside the MVS benchmark.

The reference and its best source view (the same "partner" every N-view method gets first) are rectified with the
known calibration, a stereo matcher runs on the rectified pair, and its disparity is converted back to depth on the
reference grid. This puts stereo and MVS on the same images, cameras and GT, so their occlusion behaviour can be
compared pixel for pixel.
"""
import numpy as np

from ..geometry import RectifiedPair


class StereoAsMVS:
    def __init__(self, stereo, max_disp_cap=768):
        self.stereo, self.cap = stereo, max_disp_cap
        self.name = "stereo_" + stereo.name

    def __call__(self, images, Ks, Es, depth_range):
        h, w = images[0].shape[:2]
        rp = RectifiedPair(Ks[0], Es[0], Ks[1], Es[1], (w, h))
        left, right = rp.rectify(images[0], images[1])
        max_disp = int(min(self.cap, left.shape[1] - 1, np.ceil(rp.f * rp.baseline / depth_range[0] * 1.1)))
        disp = self.stereo(left, right, max(max_disp, 32))
        return rp.disparity_to_ref_depth(disp)
