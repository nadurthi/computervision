#!/usr/bin/env python3
"""Qualitative panels from the saved predictions: for the first saved reference view of each MVS dataset,
the image, GT depth, the occlusion-region map for its stereo partner, and per-method error maps at N=2 and N=5;
for stereo, disparity maps of every method on the first pair of each dataset with the occlusion mask.
"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench import geometry as G                 # noqa: E402
from mvsbench.data.mvs import DATASETS             # noqa: E402
from mvsbench.data.stereo import DATASETS as SDS   # noqa: E402
from mvsbench.io import read_image_rgb             # noqa: E402

REGION_COLORS = {"vis_partner": (0.75, 0.75, 0.75), "occ_partner": (0.85, 0.2, 0.2), "oov_partner": (0.95, 0.65, 0.1),
                 "rescued": (0.2, 0.55, 0.9), "unseen": (0.1, 0.1, 0.1)}


def mvs_panel(plt, ds, opts, data_root, res_dir, fig_dir):
    kw = dict(work_wh=tuple(opts["work_wh"]), eval_wh=tuple(opts["eval_wh"]))
    sel = opts.get("scans") if ds == "dtu" else opts.get("scenes")
    scenes = DATASETS[ds](data_root, sel[:1] if sel else None, **kw)
    if not scenes:
        return
    sc = scenes[0]
    files = sorted(glob.glob(os.path.join(res_dir, "mvs", "depth", "*", ds, sc.name, "*_n*.npz")))
    if not files:
        return
    ref = int(os.path.basename(files[0])[:4])
    methods = sorted({p.split(os.sep)[-4] for p in files})
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_mvs_bench import GTCache, regions
    gtc = GTCache(sc)
    srcs = sc.sources(ref, 4)
    gt, regs = regions(gtc, ref, srcs)
    img = G.resize_view(read_image_rgb(sc.views[ref].image_path), sc.views[ref].K, sc.eval_wh)[0]
    rmap = np.ones(gt.shape + (3,))
    for k, c in REGION_COLORS.items():
        rmap[regs[k]] = c
    rmap[gt <= 0] = 1
    vmin, vmax = np.percentile(gt[gt > 0], [2, 98])
    cols = 2
    rows = 1 + len(methods)
    fig, ax = plt.subplots(rows, 3, figsize=(10, 2.6 * rows))
    ax[0, 0].imshow(img); ax[0, 0].set_title(f"{ds} {sc.name} view {ref}")
    ax[0, 1].imshow(np.where(gt > 0, gt, np.nan), cmap="turbo", vmin=vmin, vmax=vmax); ax[0, 1].set_title("GT depth")
    ax[0, 2].imshow(rmap); ax[0, 2].set_title("grey visible / red occluded / orange outside\n(partner) blue rescued / black unseen", fontsize=7)
    tol = sc.abs_thresholds[1]
    for i, m in enumerate(methods, start=1):
        for j, n in enumerate((2, 5)):
            p = os.path.join(res_dir, "mvs", "depth", m, ds, sc.name, f"{ref:04d}_n{n}.npz")
            a = ax[i, j]
            if os.path.exists(p):
                d = np.load(p)["depth"].astype(np.float32)
                err = np.where((gt > 0) & (d > 0), np.abs(d - gt), np.nan)
                a.imshow(err, cmap="magma", vmin=0, vmax=4 * tol)
                miss = (gt > 0) & (d <= 0)
                a.imshow(np.dstack([np.zeros_like(gt)] * 3 + [miss * 0.6]))
            a.set_title(f"{m} N={n} |error| (0..{4 * tol:g} {sc.units})", fontsize=7)
        p = os.path.join(res_dir, "mvs", "depth", m, ds, sc.name, f"{ref:04d}_n5.npz")
        if not os.path.exists(p):
            p = os.path.join(res_dir, "mvs", "depth", m, ds, sc.name, f"{ref:04d}_n2.npz")
        if os.path.exists(p):
            d = np.load(p)["depth"].astype(np.float32)
            ax[i, 2].imshow(np.where(d > 0, d, np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
        ax[i, 2].set_title(f"{m} depth", fontsize=7)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"qualitative_{ds}.png"), dpi=110)
    plt.close(fig)


def stereo_panel(plt, data_root, res_dir, fig_dir):
    for ds, fn in SDS.items():
        files = sorted(glob.glob(os.path.join(res_dir, "stereo", "disp", "*", f"{ds}__*.npz")))
        if not files:
            continue
        first = sorted({os.path.basename(f) for f in files})[0]
        name = first[:-4].replace("__", "/")
        sample = next((s for s in fn(data_root, limit=5) if s["name"] == name), None)
        if sample is None:
            continue
        methods = sorted({f.split(os.sep)[-2] for f in files if os.path.basename(f) == first})
        n = 2 + len(methods)
        c = 3
        r = int(np.ceil(n / c))
        fig, ax = plt.subplots(r, c, figsize=(12, 3 * r))
        ax = ax.ravel()
        gt = sample["gt"]
        vmax = np.percentile(gt[gt > 0], 99)
        ax[0].imshow(sample["left"]); ax[0].set_title(name)
        occ = (gt > 0) & ~sample["nocc"]
        ax[1].imshow(np.where(gt > 0, gt, np.nan), cmap="turbo", vmin=0, vmax=vmax)
        ax[1].imshow(np.dstack([occ, np.zeros_like(occ), np.zeros_like(occ), occ * 0.7]).astype(float))
        ax[1].set_title("GT, occluded pixels in red")
        for k, m in enumerate(methods, start=2):
            d = np.load(os.path.join(res_dir, "stereo", "disp", m, first))["disp"].astype(np.float32)
            ax[k].imshow(np.where(d > 0, d, np.nan), cmap="turbo", vmin=0, vmax=vmax)
            ax[k].set_title(m, fontsize=8)
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"qualitative_stereo_{ds}.png"), dpi=110)
        plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    fig_dir = os.path.join(a.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    for ds, opts in cfg["mvs"]["datasets"].items():
        try:
            mvs_panel(plt, ds, opts, a.data_root, a.results, fig_dir)
        except Exception as e:
            print(f"[figures] {ds}: {e}")
    stereo_panel(plt, a.data_root, a.results, fig_dir)


if __name__ == "__main__":
    main()
