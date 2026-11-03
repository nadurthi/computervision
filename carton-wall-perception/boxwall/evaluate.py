"""Instance-level metrics for box segmentation (image masks or lidar point labels) and face pose error."""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import face_from_mask

MIN_GT_PX = 300          # boxes with fewer visible pixels are 'ignore' regions
MIN_PRED_PX = 150


def masks_from_labels(lab: np.ndarray, min_px: int = MIN_PRED_PX):
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    return [lab == i for i, c in zip(ids, counts) if c >= min_px]


def match(gt_inst: np.ndarray, preds: list, gt_ids: list, ignore: np.ndarray):
    """Hungarian matching on IoU. A prediction lying >=50% on ignored pixels (trailer structure or
    tiny boxes) is dropped rather than counted as a false positive. Returns (pairs, ious, n_fp)."""
    keep = []
    for p in preds:
        a = p.sum()
        if a == 0:
            continue
        if (p & ignore).sum() / a >= 0.5:
            continue
        keep.append(p)
    if not keep or not gt_ids:
        return [], [], len(keep), keep
    G = np.stack([gt_inst == g for g in gt_ids]).reshape(len(gt_ids), -1)
    P = np.stack(keep).reshape(len(keep), -1)
    inter = P.astype(np.float32) @ G.T.astype(np.float32)
    union = P.sum(1)[:, None] + G.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    r, c = linear_sum_assignment(-iou)
    pairs, ious = [], []
    for i, j in zip(r, c):
        if iou[i, j] >= 0.5:
            pairs.append((i, j)); ious.append(float(iou[i, j]))
    return pairs, ious, len(keep) - len(pairs), keep


class Accumulator:
    """Accumulates TP/FP/FN, IoU sums and face-pose errors over a split."""

    def __init__(self, name: str):
        self.name = name
        self.tp = self.fp = self.fn = 0
        self.tp75 = 0
        self.iou_sum = 0.0
        self.ang, self.ctr, self.size = [], [], []
        self.times = []

    def add_image(self, gt: dict, gt_inst: np.ndarray, preds: list, depth: np.ndarray | None = None,
                  t: float | None = None):
        ids = [b["id"] for b in gt["boxes"] if b["visible_px"] >= MIN_GT_PX]
        ignore = np.isin(gt_inst, ids, invert=True)
        pairs, ious, nfp, keep = match(gt_inst, preds, ids, ignore)
        self.tp += len(pairs); self.fp += nfp; self.fn += len(ids) - len(pairs)
        self.tp75 += sum(i >= 0.75 for i in ious)
        self.iou_sum += sum(ious)
        if t is not None:
            self.times.append(t)
        if depth is None:
            return
        byid = {b["id"]: b for b in gt["boxes"]}
        for (i, j), iou in zip(pairs, ious):
            b = byid[ids[j]]
            if b["visible_px"] < 1500 or b["front_px"] < 0.7 * b["visible_px"]:
                continue
            f = face_from_mask(depth, keep[i])
            if f is None:
                continue
            R = np.array(b["R"]); c = np.array(b["center"]); s = np.array(b["size"])
            n_gt = -R[:, 2]
            fc = c + n_gt * s[2] / 2
            self.ang.append(np.degrees(np.arccos(np.clip(abs(f["normal"] @ n_gt), -1, 1))))
            self.ctr.append(float(np.linalg.norm(f["center"] - fc)))
            self.size.append(float(np.mean([abs(f["width"] - s[0]), abs(f["height"] - s[1])])))

    def summary(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1_75 = 2 * self.tp75 / max(2 * self.tp75 + (tp + fp - self.tp75) + (tp + fn - self.tp75), 1)
        sq = self.iou_sum / max(tp, 1)
        pq = self.iou_sum / max(tp + 0.5 * fp + 0.5 * fn, 1)
        out = {"method": self.name, "precision": prec, "recall": rec, "F1@0.5": f1, "F1@0.75": f1_75,
               "SQ": sq, "PQ": pq, "TP": tp, "FP": fp, "FN": fn}
        if self.ang:
            out.update({"normal_err_deg_median": float(np.median(self.ang)),
                        "center_err_cm_median": float(np.median(self.ctr) * 100),
                        "size_err_cm_median": float(np.median(self.size) * 100),
                        "n_pose": len(self.ang)})
        if self.times:
            out["sec_per_frame"] = float(np.mean(self.times))
        return out


def point_metrics(gt_lab: np.ndarray, pred_lab: np.ndarray, min_pts: int = 30):
    """Instance metrics on lidar points. gt_lab 0 = trailer (ignored), pred_lab 0 = unassigned."""
    gids = [g for g, c in zip(*np.unique(gt_lab[gt_lab > 0], return_counts=True)) if c >= min_pts]
    ignore = ~np.isin(gt_lab, gids)
    preds = [pred_lab == p for p, c in zip(*np.unique(pred_lab[pred_lab > 0], return_counts=True)) if c >= 10]
    return match(gt_lab, preds, gids, ignore), len(gids)


def save(rows: list, path: str):
    json.dump(rows, open(path, "w"), indent=1)
