"""SeamNet: a small RGB-D U-Net that predicts {background, box interior, box seam}.

Instances come from connected components of the interior class, grown over the seam band with a
watershed. This is the 'segment the seams, not the boxes' idea in its simplest learned form, and it
is small enough to train on a CPU and to run in a few ms on a Jetson.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .geometry import fill_depth, normals_from_depth

IN_H, IN_W = 240, 320


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SeamNet(nn.Module):
    def __init__(self, in_ch: int = 8, base: int = 24, n_cls: int = 3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.e4 = _block(in_ch, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])
        self.d3, self.d2, self.d1 = _block(c[3] + c[2], c[2]), _block(c[2] + c[1], c[1]), _block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], n_cls, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(e4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.head(d1)


def make_input(rgb: np.ndarray, depth_raw: np.ndarray, use_depth: bool = True) -> np.ndarray:
    """(C,H,W) float32 at the network resolution. Channels: RGB, [inverse depth, normal xyz]."""
    rgb_s = cv2.resize(rgb, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    chans = [rgb_s.transpose(2, 0, 1) - 0.5]
    if use_depth:
        d = fill_depth(depth_raw)
        n = normals_from_depth(d, smooth=5)
        d_s = cv2.resize(d, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        n_s = cv2.resize(n, (IN_W, IN_H), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize((depth_raw > 0).astype(np.float32), (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        chans += [(1.0 / np.maximum(d_s, 0.3))[None] - 0.5, n_s.transpose(2, 0, 1), valid[None] - 0.5]
    return np.concatenate(chans, 0).astype(np.float32)


def make_target(inst: np.ndarray, band: int = 2) -> np.ndarray:
    """0 = background/trailer, 1 = interior, 2 = seam (within `band` px of another instance)."""
    lab = cv2.resize(inst.astype(np.int32).astype(np.float32), (IN_W, IN_H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    mx = ndi.maximum_filter(lab, size=2 * band + 1)
    mn = ndi.minimum_filter(lab, size=2 * band + 1)
    edge = (mx != mn) & (lab > 0)
    t = np.zeros(lab.shape, np.uint8)
    t[lab > 0] = 1
    t[edge] = 2
    return t


def decode(prob: np.ndarray, out_shape=(360, 480), min_px: int = 25) -> np.ndarray:
    """prob: (3,h,w) softmax. Returns an instance label map at out_shape."""
    interior = prob[1] > 0.5
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    seeds, n = ndi.label(interior)
    if n:
        sizes = ndi.sum(interior, seeds, range(1, n + 1))
        for i, s in enumerate(sizes, start=1):
            if s < min_px:
                seeds[seeds == i] = 0
    fg = prob[0] < 0.5
    lab = watershed(prob[2], markers=seeds, mask=fg | (seeds > 0))
    return cv2.resize(lab.astype(np.float32), (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)


@torch.no_grad()
def predict(model: SeamNet, rgb, depth_raw, use_depth=True):
    x = torch.from_numpy(make_input(rgb, depth_raw, use_depth))[None]
    p = torch.softmax(model(x), 1)[0].numpy()
    return decode(p), p
