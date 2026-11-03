"""PQ of every method on exactly the walls the SAM models were scored on (first 30 of each split)."""
import glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.evaluate import Accumulator, masks_from_labels

out = {}
for split in ("test", "test_hard"):
    sids = [os.path.basename(f)[:-8] for f in sorted(glob.glob(f"data/{split}/*_gt.json"))[:30]]
    out[split] = {}
    for m in sorted(os.listdir("results/preds")):
        if not all(os.path.exists(f"results/preds/{m}/{split}/{s}.png") for s in sids):
            continue
        acc = Accumulator(m)
        for s in sids:
            p = f"data/{split}/{s}"
            acc.add_image(json.load(open(p + "_gt.json")), cv2.imread(p + "_inst.png", -1).astype(np.int32),
                          masks_from_labels(cv2.imread(f"results/preds/{m}/{split}/{s}.png", -1).astype(np.int32)))
        r = acc.summary(); out[split][m] = {"PQ": r["PQ"], "F1@0.5": r["F1@0.5"]}
        print(split, m, round(r["PQ"], 3), round(r["F1@0.5"], 3))
json.dump(out, open("results/metrics_first30.json", "w"), indent=1)
