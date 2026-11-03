"""Wall-clock time per frame for every camera method on an otherwise idle CPU (2 cores, PyTorch CPU, FP32).

The s/frame column in metrics_*.json was recorded while other jobs shared the CPU; this script gives clean numbers.
Each method gets one warm-up frame, then the mean of the next N frames. SAM times include encoding the image.
"""
import glob, json, os, sys, time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import method_fn, _cache

METHODS = ["rgb_lines", "depth_edges", "rgbd_seams", "seamnet_rgb", "seamnet_rgbd", "yolo", "fastsam_rgbd", "mobilesam_rgbd", "sam2t_rgbd"]

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    torch.set_num_threads(os.cpu_count())
    files = sorted(glob.glob("data/test/*_gt.json"))[: n + 1]
    frames = []
    for f in files:
        p = f[:-8]
        frames.append((cv2.imread(p + "_rgb.png")[:, :, ::-1].copy(), cv2.imread(p + "_depth.png", -1).astype(np.float32) / 1000))
    out = {}
    for m in METHODS:
        fn = method_fn(m)
        ts = []
        for k, (rgb, d) in enumerate(frames):
            _cache.clear()                              # SAM: re-encode every frame
            t = time.perf_counter(); fn(rgb, d); dt = time.perf_counter() - t
            if k:
                ts.append(dt)
        out[m] = {"mean_s": float(np.mean(ts)), "frames": len(ts)}
        print(m, round(out[m]["mean_s"], 3), flush=True)
    out["_hardware"] = f"{os.cpu_count()} CPU cores, torch {torch.__version__}, FP32"
    json.dump(out, open("results/speed_cpu.json", "w"), indent=1)
