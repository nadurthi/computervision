"""Run the RGB-only methods on real photos (no depth, no ground truth): a sim-to-real sanity check."""
import glob, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import method_fn

METHODS = sys.argv[1:] or ["rgb_lines", "seamnet_rgb", "yolo", "fastsam_rgb", "sam2t_rgb"]

if __name__ == "__main__":
    fns = {m: method_fn(m) for m in METHODS}
    for t in sorted(glob.glob("data/real/scd_tile*.png")):
        rgb = cv2.imread(t)[:, :, ::-1].copy()
        dummy = np.zeros(rgb.shape[:2], np.float32)
        for m, fn in fns.items():
            lab = fn(rgb, dummy)
            od = f"results/preds/{m}/real"; os.makedirs(od, exist_ok=True)
            cv2.imwrite(f"{od}/{os.path.basename(t)}", lab.astype(np.uint16))
        print(t, flush=True)
