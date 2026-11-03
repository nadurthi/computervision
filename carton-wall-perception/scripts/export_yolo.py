"""Export the synthetic walls to an Ultralytics YOLO-seg dataset (one polygon per visible box)."""
import glob, os, shutil, sys

import cv2
import numpy as np

ROOT = "data/yolo"


def export(split, n_max=None):
    ims = os.path.join(ROOT, "images", split); lbs = os.path.join(ROOT, "labels", split)
    os.makedirs(ims, exist_ok=True); os.makedirs(lbs, exist_ok=True)
    ps = sorted(f[:-8] for f in glob.glob(f"data/{split}/*_gt.json"))[:n_max]
    for p in ps:
        inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
        H, W = inst.shape
        lines = []
        for i in np.unique(inst[inst > 0]):
            m = (inst == i).astype(np.uint8)
            if m.sum() < 300:
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 1.0, True)[:, 0]
            if len(c) < 3:
                continue
            xy = (c / np.array([W, H])).reshape(-1)
            lines.append("0 " + " ".join(f"{v:.5f}" for v in xy))
        sid = os.path.basename(p)
        shutil.copy(p + "_rgb.png", os.path.join(ims, sid + ".png"))
        open(os.path.join(lbs, sid + ".txt"), "w").write("\n".join(lines))
    print(split, len(ps))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    export("train", n); export("val")
    open(os.path.join(ROOT, "boxes.yaml"), "w").write(
        f"path: {os.path.abspath(ROOT)}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n")
