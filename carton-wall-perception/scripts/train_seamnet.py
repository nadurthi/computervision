"""Train SeamNet (RGB-D or RGB-only) on the synthetic walls. CPU-friendly.

python scripts/train_seamnet.py --cache            # build data/cache_*.npy once
python scripts/train_seamnet.py --mode rgbd --epochs 20
python scripts/train_seamnet.py --mode rgb  --epochs 20
"""
import argparse, glob, os, sys, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.seamnet import SeamNet, make_input, make_target


def load(p):
    rgb = cv2.imread(p + "_rgb.png")[:, :, ::-1].copy()
    d = cv2.imread(p + "_depth.png", -1).astype(np.float32) / 1000
    inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
    return rgb, d, inst


def build_cache(split):
    ps = sorted(f[:-8] for f in glob.glob(f"data/{split}/*_gt.json"))
    X = np.zeros((len(ps), 8, 240, 320), np.float16)
    Y = np.zeros((len(ps), 240, 320), np.uint8)
    for i, p in enumerate(ps):
        rgb, d, inst = load(p)
        X[i] = make_input(rgb, d, True)
        Y[i] = make_target(inst)
        if i % 200 == 0:
            print(split, i, flush=True)
    np.save(f"data/cache_{split}_X.npy", X); np.save(f"data/cache_{split}_Y.npy", Y)


def augment(x, y, rng):
    if rng.random() < 0.5:
        x = x.flip(-1); y = y.flip(-1)
        if x.shape[1] > 4:
            x[:, 4] = -x[:, 4]                                   # normal x flips sign with the image
    b = x.shape[0]
    gain = torch.from_numpy(rng.uniform(0.7, 1.3, (b, 1, 1, 1)).astype(np.float32))
    bias = torch.from_numpy(rng.uniform(-0.12, 0.12, (b, 1, 1, 1)).astype(np.float32))
    tint = torch.from_numpy(rng.uniform(-0.05, 0.05, (b, 3, 1, 1)).astype(np.float32))
    x[:, :3] = (x[:, :3] + 0.5) * gain + bias + tint - 0.5
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--mode", default="rgbd", choices=["rgbd", "rgb"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    if a.cache:
        for s in ("val", "train"):
            build_cache(s)
        return
    X = np.load("data/cache_train_X.npy", mmap_mode="r"); Y = np.load("data/cache_train_Y.npy")
    Xv = np.load("data/cache_val_X.npy"); Yv = np.load("data/cache_val_Y.npy")
    ch = slice(0, 8) if a.mode == "rgbd" else slice(0, 3)
    model = SeamNet(in_ch=8 if a.mode == "rgbd" else 3)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(X) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    w = torch.tensor([1.0, 1.0, 3.0])
    rng = np.random.default_rng(0)
    log = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for k in range(len(X) // a.bs):
            idx = np.sort(perm[k * a.bs:(k + 1) * a.bs])
            x = torch.from_numpy(X[idx][:, ch].astype(np.float32))
            y = torch.from_numpy(Y[idx].astype(np.int64))
            x, y = augment(x, y, rng)
            loss = F.cross_entropy(model(x), y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = 0.0; correct = 0; seam_tp = seam_p = seam_t = 0
            for k in range(0, len(Xv), 16):
                x = torch.from_numpy(Xv[k:k + 16][:, ch].astype(np.float32)); y = torch.from_numpy(Yv[k:k + 16].astype(np.int64))
                o = model(x); vl += F.cross_entropy(o, y, weight=w).item() * len(x)
                pr = o.argmax(1)
                seam_tp += ((pr == 2) & (y == 2)).sum().item(); seam_p += (pr == 2).sum().item(); seam_t += (y == 2).sum().item()
        f1 = 2 * seam_tp / max(seam_p + seam_t, 1)
        log.append({"epoch": ep, "train_loss": tot / (len(X) // a.bs), "val_loss": vl / len(Xv), "val_seam_f1": f1,
                    "minutes": (time.time() - t0) / 60})
        print(log[-1], flush=True)
        torch.save(model.state_dict(), f"results/seamnet_{a.mode}.pt")
    import json
    json.dump(log, open(f"results/seamnet_{a.mode}_trainlog.json", "w"), indent=1)


if __name__ == "__main__":
    main()
