"""Run every method on a test split, save label maps and metrics.

python scripts/run_eval.py --split test --methods rgb_lines depth_edges rgbd_seams seamnet_rgbd seamnet_rgb yolo fastsam_rgb fastsam_rgbd
python scripts/run_eval.py --split test --methods sam2t_rgb sam2t_rgbd mobilesam_rgb mobilesam_rgbd --limit 30
"""
import argparse, glob, json, os, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.evaluate import Accumulator, masks_from_labels

_cache = {}


def method_fn(name):
    if name == "rgb_lines":
        from boxwall.classical import rgb_lines
        return lambda rgb, d: rgb_lines(rgb)
    if name == "depth_edges":
        from boxwall.classical import depth_edges
        return lambda rgb, d: depth_edges(d)
    if name == "rgbd_seams":
        from boxwall.classical import rgbd_seams
        return lambda rgb, d: rgbd_seams(rgb, d)
    if name.startswith("seamnet"):
        import torch
        from boxwall.seamnet import SeamNet, predict
        mode = name.split("_")[1]
        net = SeamNet(in_ch=8 if mode == "rgbd" else 3)
        net.load_state_dict(torch.load(f"results/seamnet_{mode}.pt", map_location="cpu"))
        net.eval()
        return lambda rgb, d: predict(net, rgb, d, use_depth=(mode == "rgbd"))[0]
    if name == "yolo":
        from ultralytics import YOLO
        y = YOLO("results/yolo11n_seg_boxes.pt")

        def f(rgb, d):
            r = y(rgb[:, :, ::-1].copy(), imgsz=480, conf=0.3, retina_masks=True, verbose=False)[0]
            lab = np.zeros(rgb.shape[:2], np.int32)
            if r.masks is None:
                return lab
            ms = r.masks.data.cpu().numpy() > 0.5
            order = np.argsort(r.boxes.conf.cpu().numpy())                  # paint low->high confidence
            for k, i in enumerate(order, start=1):
                lab[ms[i]] = k
            return lab
        return f
    for base in ("fastsam", "mobilesam", "sam2t"):
        if name.startswith(base):
            from boxwall import zeroshot as Z
            use_d = name.endswith("rgbd")

            def f(rgb, d, base=base, use_d=use_d, key=None):
                key = (base, rgb.tobytes()[:4096], rgb.sum())
                if key not in _cache:
                    _cache.clear()
                    _cache[key] = Z.raw_masks(base, rgb)
                ms = _cache[key]
                if not ms:
                    return np.zeros(rgb.shape[:2], np.int32)
                return Z.select(ms, d if use_d else None)
            return f
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(f"data/{a.split}/*_gt.json"))[:a.limit]
    fns = {m: method_fn(m) for m in a.methods}
    accs = {m: Accumulator(m) for m in a.methods}
    for k, f in enumerate(files):
        p = f[:-8]; sid = os.path.basename(p)
        gt = json.load(open(f))
        rgb = cv2.imread(p + "_rgb.png")[:, :, ::-1].copy()
        d = cv2.imread(p + "_depth.png", -1).astype(np.float32) / 1000
        inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
        for m, fn in fns.items():
            t = time.time(); lab = fn(rgb, d); dt = time.time() - t
            accs[m].add_image(gt, inst, masks_from_labels(lab), d, dt)
            od = f"results/preds/{m}/{a.split}"; os.makedirs(od, exist_ok=True)
            cv2.imwrite(f"{od}/{sid}.png", lab.astype(np.uint16))
        if k % 10 == 0:
            print(k, len(files), flush=True)
    out = f"results/metrics_{a.split}.json"
    allm = json.load(open(out)) if os.path.exists(out) else {}
    for m, acc in accs.items():
        s = acc.summary(); s["n_images"] = len(files)
        allm[m] = s
        print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in s.items()}, flush=True)
    json.dump(allm, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
