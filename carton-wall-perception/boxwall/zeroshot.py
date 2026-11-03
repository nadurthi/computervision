"""Zero-shot 'segment everything' models (FastSAM, MobileSAM, SAM 2.1-tiny) turned into box instances.

Raw everything-masks mix whole boxes, sub-parts (labels, tape, prints) and multi-box groups.
Two generic post-processors pick one mask per pixel:
  * rgb    : area limits + rectangularity, larger masks first (no depth).
  * rgbd   : additionally requires a planar face whose fitted rectangle has a physically plausible
             carton size (both sides 8-70 cm). This is the 'foundation model + geometry' recipe.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import fill_depth, face_from_mask

_models = {}


def _get(name: str):
    if name not in _models:
        import os
        from ultralytics import FastSAM
        from ultralytics.models.sam import Predictor, SAM2Predictor
        path = {"fastsam": "FastSAM-s.pt", "mobilesam": "mobile_sam.pt", "sam2t": "sam2.1_t.pt"}[name]
        path = os.path.join(os.environ.get("WEIGHTS", "/home/claude/weights"), path)
        if name == "fastsam":
            _models[name] = FastSAM(path)
        else:
            cls = SAM2Predictor if name == "sam2t" else Predictor
            _models[name] = cls(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                                               model=path, save=False, verbose=False))
    return _models[name]


def raw_masks(name: str, rgb: np.ndarray) -> list:
    m = _get(name)
    bgr = rgb[:, :, ::-1].copy()
    if name == "fastsam":
        r = m(bgr, retina_masks=True, imgsz=480, conf=0.25, iou=0.7, verbose=False)[0]
    else:
        h, w = rgb.shape[:2]
        xs, ys = np.meshgrid(np.linspace(w / 32, w - w / 32, 16), np.linspace(h / 24, h - h / 24, 12))
        pts = np.stack([xs.ravel(), ys.ravel()], 1)                     # 192 single-point prompts
        m.set_image(bgr)                                                 # encode once
        chunks = []
        for k in range(0, len(pts), 16):
            p = pts[k:k + 16][:, None, :].tolist()                      # 16 objects x 1 point
            rr = m(points=p, labels=[[1]] * len(p))[0]
            if rr.masks is not None:
                chunks.append(rr.masks.data.cpu().numpy() > 0.5)
        m.reset_image()
        if not chunks:
            return []
        ms = np.concatenate(chunks)
        return _resize(ms, rgb)
    if r.masks is None:
        return []
    ms = r.masks.data.cpu().numpy() > 0.5
    return _resize(ms, rgb)


def _resize(ms, rgb):
    if ms.shape[1:] != rgb.shape[:2]:
        ms = np.stack([cv2.resize(x.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0 for x in ms])
    return list(ms)


def _rect(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return float(mask.sum()) / max(w * h, 1)


def select(masks: list, depth_raw: np.ndarray | None = None, min_px=300, max_frac=0.2,
           min_rect=0.75, size_range=(0.08, 0.70)) -> np.ndarray:
    H, W = masks[0].shape if masks else (360, 480)
    keep = []
    d = fill_depth(depth_raw) if depth_raw is not None else None
    for m in masks:
        a = int(m.sum())
        if a < min_px or a > max_frac * H * W:
            continue
        if _rect(m) < min_rect:
            continue
        if d is not None:
            f = face_from_mask(d, m)
            if f is None or f["inlier_frac"] < 0.6:
                continue
            if not (size_range[0] <= f["width"] <= size_range[1] and size_range[0] <= f["height"] <= size_range[1]):
                continue
        keep.append((a, m))
    keep.sort(key=lambda t: -t[0])
    lab = np.zeros((H, W), np.int32)
    k = 1
    for a, m in keep:
        new = m & (lab == 0)
        if new.sum() < 0.7 * a:
            continue
        lab[new] = k; k += 1
    return lab
