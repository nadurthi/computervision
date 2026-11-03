"""Face-pose error on a common subset: boxes that every compared method matched (IoU >= 0.5).

Per-method pose medians in RESULTS.md are computed over each method's own matches, so a method that only finds
the easy boxes looks more accurate. This script removes that selection effect.
"""
import glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.evaluate import masks_from_labels, MIN_GT_PX
from boxwall.geometry import face_from_mask


def best_match(gt_mask, labs):
    ids, cnt = np.unique(labs[gt_mask], return_counts=True)
    best, biou = None, 0.0
    for i, c in zip(ids, cnt):
        if i == 0:
            continue
        m = labs == i
        iou = c / (gt_mask.sum() + m.sum() - c)
        if iou > biou:
            best, biou = m, iou
    return best if biou >= 0.5 else None


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    methods = sys.argv[2:] or ["rgbd_seams", "seamnet_rgbd", "yolo", "fastsam_rgbd", "sam2t_rgbd"]
    errs = {m: {"ang": [], "ctr": [], "size": []} for m in methods}
    n_boxes = 0
    for f in sorted(glob.glob(f"data/{split}/*_gt.json")):
        p = f[:-8]; sid = os.path.basename(p)
        paths = [f"results/preds/{m}/{split}/{sid}.png" for m in methods]
        if not all(os.path.exists(x) for x in paths):
            continue
        gt = json.load(open(f))
        inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
        d = cv2.imread(p + "_depth.png", -1).astype(np.float32) / 1000
        labs = [cv2.imread(x, -1).astype(np.int32) for x in paths]
        for b in gt["boxes"]:
            if b["visible_px"] < 1500 or b["front_px"] < 0.7 * b["visible_px"]:
                continue
            g = inst == b["id"]
            ms = [best_match(g, L) for L in labs]
            if any(m is None for m in ms):
                continue
            fits = [face_from_mask(d, m) for m in ms]
            if any(f_ is None for f_ in fits):
                continue
            n_boxes += 1
            R = np.array(b["R"]); c = np.array(b["center"]); s = np.array(b["size"])
            n_gt = -R[:, 2]; fc = c + n_gt * s[2] / 2
            for m, fit in zip(methods, fits):
                errs[m]["ang"].append(np.degrees(np.arccos(np.clip(abs(fit["normal"] @ n_gt), -1, 1))))
                errs[m]["ctr"].append(np.linalg.norm(fit["center"] - fc) * 100)
                errs[m]["size"].append(np.mean([abs(fit["width"] - s[0]), abs(fit["height"] - s[1])]) * 100)
    out = {"split": split, "n_boxes": n_boxes}
    for m in methods:
        out[m] = {k: float(np.median(v)) for k, v in errs[m].items()}
        out[m].update({k + "_p90": float(np.percentile(v, 90)) for k, v in errs[m].items()})
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"results/pose_common_{split}.json", "w"), indent=1)
