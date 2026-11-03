"""Lidar point-level instance metrics (points inside the camera field of view): range-image cuts, RANSAC+DBSCAN, and camera->lidar label transfer."""
import argparse, glob, json, os, sys, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall import lidar as LL
from boxwall.evaluate import match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--camera_methods", nargs="*", default=["seamnet_rgbd", "rgbd_seams"])
    a = ap.parse_args()
    files = sorted(glob.glob(f"data/{a.split}/*_lidar.npz"))
    methods = ["lidar_range_image", "lidar_ransac_dbscan"] + [f"cam2lidar_{m}" for m in a.camera_methods]
    st = {m: dict(tp=0, fp=0, fn=0, iou=0.0, t=[]) for m in methods}
    for k, f in enumerate(files):
        sid = os.path.basename(f)[:5]
        L = dict(np.load(f))
        # score only points inside the camera frustum, so camera-based and lidar-only methods see the same boxes
        P = L["points"]; z = np.maximum(P[:, 2], 1e-3)
        u = P[:, 0] / z * LL.FX + LL.CX; v = P[:, 1] / z * LL.FY + LL.CY
        inview = (P[:, 2] > 0.1) & (u >= 0) & (u < LL.W) & (v >= 0) & (v < LL.H)
        g = L["inst"].copy(); g[~inview] = 0
        gids = [i for i, c in zip(*np.unique(g[g > 0], return_counts=True)) if c >= 30]
        ign = ~np.isin(g, gids)
        d = cv2.imread(f"data/{a.split}/{sid}_depth.png", -1).astype(np.float32) / 1000
        for m in methods:
            t = time.time()
            if m == "lidar_range_image":
                lab = LL.range_image(L)
            elif m == "lidar_ransac_dbscan":
                lab = LL.ransac_dbscan(L)
            else:
                img = cv2.imread(f"results/preds/{m[len('cam2lidar_'):]}/{a.split}/{sid}.png", -1).astype(np.int32)
                lab = LL.camera_to_lidar(L, img, d)
            dt = time.time() - t
            preds = [(lab == p) & inview for p in np.unique(lab[lab > 0]) if ((lab == p) & inview).sum() >= 10]
            pairs, ious, nfp, _ = match(g, preds, gids, ign)
            s = st[m]; s["tp"] += len(pairs); s["fp"] += nfp; s["fn"] += len(gids) - len(pairs); s["iou"] += sum(ious); s["t"].append(dt)
            if k == 0:
                np.save(f"results/lidar_example_{m}.npy", lab)
        if k % 20 == 0:
            print(k, len(files), flush=True)
    out = {}
    for m, s in st.items():
        P = s["tp"] / max(s["tp"] + s["fp"], 1); R = s["tp"] / max(s["tp"] + s["fn"], 1)
        out[m] = {"method": m, "precision": P, "recall": R, "F1@0.5": 2 * P * R / max(P + R, 1e-9),
                  "PQ": s["iou"] / max(s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"], 1), "TP": s["tp"], "FP": s["fp"],
                  "FN": s["fn"], "sec_per_frame": float(np.mean(s["t"])), "n_scans": len(files)}
        print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in out[m].items()})
    json.dump(out, open(f"results/lidar_metrics_{a.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
