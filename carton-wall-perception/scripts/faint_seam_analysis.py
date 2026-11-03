"""Where does depth help? Split test scenes by how dark the renderer drew the box edges (seam_k) and compare PQ.

seam_k is the first random draw in render_rgbd (rng seeded with scene seed + 7), so it can be recovered exactly.
"""
import glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.evaluate import Accumulator, masks_from_labels

METHODS = ["depth_edges", "rgbd_seams", "seamnet_rgb", "seamnet_rgbd", "yolo"]

if __name__ == "__main__":
    out = {}
    for split in ("test", "test_hard"):
        for name, keep in (("faint", lambda k: k < 0.15), ("visible", lambda k: k >= 0.15)):
            accs = {m: Accumulator(m) for m in METHODS if os.path.isdir(f"results/preds/{m}/{split}")}
            n = 0
            for f in sorted(glob.glob(f"data/{split}/*_gt.json")):
                gt = json.load(open(f))
                k = np.random.default_rng(gt["seed"] + 7).uniform(0.0, 0.45)
                if not keep(k):
                    continue
                n += 1
                p = f[:-8]; sid = os.path.basename(p)
                inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
                for m, a in accs.items():
                    lab = cv2.imread(f"results/preds/{m}/{split}/{sid}.png", -1).astype(np.int32)
                    a.add_image(gt, inst, masks_from_labels(lab))
            out[f"{split}/{name}"] = {"n_scenes": n, **{m: a.summary()["PQ"] for m, a in accs.items()}}
            print(split, name, out[f"{split}/{name}"])
    json.dump(out, open("results/faint_seam_analysis.json", "w"), indent=1)
