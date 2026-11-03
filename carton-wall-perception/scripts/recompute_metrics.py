"""Recompute every metric in results/metrics_<split>.json from the saved label maps (results/preds/...).

Used after changing the face-fitting code so that every method's pose numbers come from the same version.
Timing fields are kept from the original run (see benchmark_speed.py for clean timings).
"""
import glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.evaluate import Accumulator, masks_from_labels

if __name__ == "__main__":
    for split in ("test", "test_hard"):
        path = f"results/metrics_{split}.json"
        M = json.load(open(path))
        for m in list(M):
            files = sorted(glob.glob(f"results/preds/{m}/{split}/*.png"))
            acc = Accumulator(m)
            for f in files:
                sid = os.path.basename(f)[:-4]
                p = f"data/{split}/{sid}"
                gt = json.load(open(p + "_gt.json"))
                inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
                d = cv2.imread(p + "_depth.png", -1).astype(np.float32) / 1000
                lab = cv2.imread(f, -1).astype(np.int32)
                acc.add_image(gt, inst, masks_from_labels(lab), d)
            s = acc.summary()
            s["n_images"] = len(files)
            if "sec_per_frame" in M[m]:
                s["sec_per_frame"] = M[m]["sec_per_frame"]
            M[m] = s
            print(split, m, len(files), round(s["PQ"], 3), round(s.get("center_err_cm_median", -1), 2), flush=True)
        json.dump(M, open(path, "w"), indent=1)
