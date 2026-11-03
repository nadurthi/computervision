"""Figures for the README / blog post. Writes docs/figures/*.png."""
import glob, json, os, sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxwall.geometry import fill_depth, normals_from_depth

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# Families get fixed categorical slots (reference palette, light mode)
FAMILY = {"classical": "#2a78d6", "learned": "#eb6834", "zero-shot": "#1baf7a"}
NAMES = {
    "rgb_lines": ("Classical: RGB straight lines", "classical"),
    "depth_edges": ("Classical: depth jumps + creases", "classical"),
    "rgbd_seams": ("Classical: depth + RGB seam valleys", "classical"),
    "seamnet_rgb": ("SeamNet U-Net (RGB)", "learned"),
    "seamnet_rgbd": ("SeamNet U-Net (RGB-D)", "learned"),
    "yolo": ("YOLO11n-seg fine-tuned (RGB)", "learned"),
    "fastsam_rgb": ("FastSAM-s (RGB)", "zero-shot"),
    "fastsam_rgbd": ("FastSAM-s + depth priors", "zero-shot"),
    "mobilesam_rgb": ("MobileSAM grid (RGB)", "zero-shot"),
    "mobilesam_rgbd": ("MobileSAM + depth priors", "zero-shot"),
    "sam2t_rgb": ("SAM 2.1-tiny grid (RGB)", "zero-shot"),
    "sam2t_rgbd": ("SAM 2.1-tiny + depth priors", "zero-shot"),
}
ORDER = list(NAMES)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def palette(n, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.integers(60, 255, (n + 1, 3)).astype(np.uint8)
    p[0] = 0
    return p


def overlay(rgb, lab, alpha=0.55):
    p = palette(int(lab.max()) + 1)
    col = p[lab]
    out = rgb.copy().astype(np.float32)
    m = lab > 0
    out[m] = (1 - alpha) * out[m] + alpha * col[m]
    edges = cv2.morphologyEx((lab > 0).astype(np.uint8) * 0 + lab.astype(np.uint16), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edges] = (255, 255, 255)
    return out.astype(np.uint8)


def load(split, sid):
    p = f"data/{split}/{sid}"
    rgb = cv2.imread(p + "_rgb.png")[:, :, ::-1].copy()
    d = cv2.imread(p + "_depth.png", -1).astype(np.float32) / 1000
    inst = cv2.imread(p + "_inst.png", -1).astype(np.int32)
    return rgb, d, inst


def fig_dataset(split="test", sid="09003"):
    rgb, d, inst = load(split, sid)
    n = normals_from_depth(fill_depth(d))
    L = np.load(f"data/{split}/{sid}_lidar.npz")
    rows, cols = L["shape"]
    R = np.zeros((rows, cols), np.float32); R[L["ring"], L["col"]] = L["range"]
    G = np.zeros((rows, cols), np.int32); G[L["ring"], L["col"]] = L["inst"]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.6))
    dv = d.copy(); dv[dv == 0] = np.nan
    panels = [(rgb, "RGB (480x360)"), (dv, "Depth with sensor dropouts (black)"),
              (((n + 1) / 2), "Normals from depth"), (palette(inst.max())[inst], "Ground-truth instances"),
              (R, "Lidar range image (80 x 640)"), (palette(G.max())[G], "Lidar ground-truth instances")]
    for a, (im, t) in zip(ax.ravel(), panels):
        if im.ndim == 2:
            cm = plt.get_cmap("viridis").copy(); cm.set_bad("black")
            a.imshow(im, cmap=cm, aspect="auto" if im.shape[0] == rows else None)
        else:
            a.imshow(im, aspect="auto" if im.shape[0] == rows else None)
        a.set_title(t, fontsize=10, color=INK); a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/dataset_example.png", dpi=110); plt.close(fig)


def fig_qualitative(split, sid, methods, fname):
    rgb, d, inst = load(split, sid)
    n = len(methods) + 1
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.9 * rows))
    ax = ax.ravel()
    from boxwall.evaluate import match, masks_from_labels, MIN_GT_PX
    gt = json.load(open(f"data/{split}/{sid}_gt.json"))
    gids = [b["id"] for b in gt["boxes"] if b["visible_px"] >= MIN_GT_PX]
    ignore = np.isin(inst, gids, invert=True)
    ax[0].imshow(overlay(rgb, inst)); ax[0].set_title(f"Ground truth ({len(gids)} boxes)", fontsize=10)
    for a, m in zip(ax[1:], methods):
        f = f"results/preds/{m}/{split}/{sid}.png"
        lab = cv2.imread(f, -1).astype(np.int32) if os.path.exists(f) else np.zeros_like(inst)
        pairs, ious, nfp, keep = match(inst, masks_from_labels(lab), gids, ignore)
        img = overlay(rgb, lab)
        matched_g = {j for _, j in pairs}; matched_p = {i for i, _ in pairs}
        for j, g in enumerate(gids):                         # missed boxes: red outline
            if j not in matched_g:
                cs, _ = cv2.findContours((inst == g).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (230, 30, 30), 3)
        for i, p in enumerate(keep):                         # false detections: yellow outline
            if i not in matched_p:
                cs, _ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (255, 210, 0), 2)
        tp = len(pairs); fn = len(gids) - tp
        pq_img = sum(ious) / max(tp + 0.5 * nfp + 0.5 * fn, 1)
        a.imshow(img)
        a.set_title(f"{NAMES[m][0]}\nPQ {pq_img:.2f} · found {tp} · missed {fn} · false {nfp}", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=100); plt.close(fig)


def fig_bars():
    mt = json.load(open("results/metrics_test.json")); mh = json.load(open("results/metrics_test_hard.json"))
    ms = [m for m in ORDER if m in mt]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 0.42 * len(ms) + 1.6), sharey=True)
    for a, (title, M) in zip(axs, [("Standard walls (gaps 0-12 mm)", mt), ("Hard walls (tighter gaps, more yaw)", mh)]):
        y = np.arange(len(ms))[::-1]
        vals = [M.get(m, {}).get("PQ", np.nan) for m in ms]
        cols = [FAMILY[NAMES[m][1]] for m in ms]
        a.barh(y, vals, color=cols, height=0.62, edgecolor="white", linewidth=2)
        for yy, v, m in zip(y, vals, ms):
            if np.isfinite(v):
                ntag = M[m].get("n_images", 100)
                a.text(v + 0.01, yy, f"{v:.2f}" + ("" if ntag >= 100 else f"  (n={ntag})"), va="center", fontsize=9, color=INK)
        a.set_xlim(0, 1.0); a.set_title(title, fontsize=11, color=INK, loc="left")
        a.set_xlabel("Panoptic quality (PQ, higher is better)", fontsize=9, color=MUTED)
        a.grid(axis="x", color=GRID, linewidth=0.8); a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9, length=0)
    axs[0].set_yticks(np.arange(len(ms))[::-1]); axs[0].set_yticklabels([NAMES[m][0] for m in ms], fontsize=9, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY.values()]
    fig.legend(handles, ["Classical", "Trained on synthetic walls", "Zero-shot foundation model"], loc="lower center",
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT}/pq_bars.png", dpi=110); plt.close(fig)


def fig_lidar(split="test"):
    f = sorted(glob.glob(f"data/{split}/*_lidar.npz"))[0]
    L = np.load(f)
    rows, cols = L["shape"]

    def img(lab):
        I = np.zeros((rows, cols), np.int32); I[L["ring"], L["col"]] = lab
        return palette(int(I.max()) + 1, 5)[I]
    names = [("Ground truth", L["inst"])]
    for m, t in [("lidar_range_image", "Range-image cuts (lidar only)"), ("lidar_ransac_dbscan", "RANSAC planes + DBSCAN (lidar only)"),
                 ("cam2lidar_seamnet_rgbd", "Camera SeamNet labels projected to lidar")]:
        p = f"results/lidar_example_{m}.npy"
        if os.path.exists(p):
            names.append((t, np.load(p)))
    fig, ax = plt.subplots(len(names), 1, figsize=(10, 2.2 * len(names)))
    for a, (t, lab) in zip(ax, names):
        a.imshow(img(lab), aspect="auto", interpolation="nearest"); a.set_title(t, fontsize=10, loc="left"); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/lidar_example.png", dpi=100); plt.close(fig)


def fig_training():
    fig, a = plt.subplots(figsize=(6.5, 3.4))
    for mode, c in (("rgbd", FAMILY["learned"]), ("rgb", FAMILY["classical"])):
        p = f"results/seamnet_{mode}_trainlog.json"
        if not os.path.exists(p):
            continue
        log = json.load(open(p))
        a.plot([r["epoch"] + 1 for r in log], [r["val_seam_f1"] for r in log], color=c, linewidth=2, marker="o", markersize=4,
               label=f"SeamNet {mode.upper()}")
        a.text(log[-1]["epoch"] + 1.2, log[-1]["val_seam_f1"], f"{mode.upper()} {log[-1]['val_seam_f1']:.2f}", va="center", fontsize=9, color=INK)
    a.set_xlabel("Epoch", color=MUTED, fontsize=9); a.set_ylabel("Seam-pixel F1 (validation)", color=MUTED, fontsize=9)
    a.grid(color=GRID); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/seamnet_training.png", dpi=110); plt.close(fig)


def pick_plan(lab, depth):
    """From an instance map: fit each face, keep faces looking at the robot, order top-down then front-first.
    Returns list of dicts with the face fit and the 2D polygon of its fitted rectangle."""
    from boxwall.geometry import face_from_mask
    from boxwall.render import FX, FY, W, H
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    faces = []
    for i in np.unique(lab[lab > 0]):
        m = lab == i
        if m.sum() < 600:
            continue
        f = face_from_mask(depth, m)
        if f is None or f["normal"][2] > -0.85:          # within ~30 deg of facing the sensor
            continue
        e1, e2 = f["axes"]
        corners = [f["center"] + sx * f["width"] / 2 * e1 + sy * f["height"] / 2 * e2 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        poly = np.array([[c[0] / c[2] * FX + cx, c[1] / c[2] * FY + cy] for c in corners])
        f["poly"] = poly
        faces.append(f)
    # top of the wall first (smallest y = highest), ties broken by nearest face
    faces.sort(key=lambda f: (round((f["center"][1] - f["height"] / 2) / 0.1), f["center"][2]))
    return faces


def fig_picks(split="test", sid="09003", method="seamnet_rgbd"):
    rgb, d, inst = load(split, sid)
    lab = cv2.imread(f"results/preds/{method}/{split}/{sid}.png", -1).astype(np.int32)
    faces = pick_plan(lab, d)
    fig, a = plt.subplots(figsize=(7.5, 5.8))
    a.imshow(rgb)
    for k, f in enumerate(faces[:12], start=1):
        p = np.vstack([f["poly"], f["poly"][:1]])
        a.plot(p[:, 0], p[:, 1], color="#ffffff", linewidth=1.6)
        c = f["poly"].mean(0)
        a.add_patch(plt.Circle(c, 9, color="#eb6834"))
        a.text(c[0], c[1], str(k), color="white", fontsize=8, ha="center", va="center", fontweight="bold")
        if k <= 4 and np.ptp(f["poly"][:, 1]) > 40:           # label the first picks that have room for it
            a.text(f["poly"][:, 0].min() + 3, f["poly"][:, 1].min() + 12,
                   f"{f['width']*100:.0f}x{f['height']*100:.0f} cm, z={f['center'][2]:.2f} m", color="white", fontsize=7,
                   bbox=dict(facecolor="#0b0b0b", alpha=0.6, pad=1, edgecolor="none"))
    a.axis("off")
    a.set_title("Pick plan from SeamNet RGB-D: fitted faces, first 12 picks (top-down, nearest first)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/pick_plan.png", dpi=110); plt.close(fig)


def fig_real(methods):
    tiles = sorted(glob.glob("data/real/scd_tile*.png"))
    fig, ax = plt.subplots(len(tiles), len(methods) + 1, figsize=(3.0 * (len(methods) + 1), 2.3 * len(tiles)))
    for r, t in enumerate(tiles):
        rgb = cv2.imread(t)[:, :, ::-1].copy()
        ax[r, 0].imshow(rgb)
        if r == 0:
            ax[r, 0].set_title("Real photo (SCD)", fontsize=9)
        for c, m in enumerate(methods, start=1):
            f = f"results/preds/{m}/real/{os.path.basename(t)}"
            lab = cv2.imread(f, -1).astype(np.int32) if os.path.exists(f) else np.zeros(rgb.shape[:2], np.int32)
            ax[r, c].imshow(overlay(rgb, lab))
            if r == 0:
                ax[r, c].set_title(NAMES[m][0], fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(f"{OUT}/real_photos.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["dataset", "bars", "qual", "lidar", "training"]
    if "dataset" in which:
        fig_dataset()
    if "bars" in which:
        fig_bars()
    if "qual" in which:
        ms = ["rgb_lines", "depth_edges", "rgbd_seams", "seamnet_rgb", "seamnet_rgbd", "yolo", "fastsam_rgbd", "sam2t_rgbd"]
        fig_qualitative("test", "09003", ms, "qualitative_test.png")
        fig_qualitative("test_hard", "12003", ms, "qualitative_hard.png")
    if "lidar" in which:
        fig_lidar()
    if "training" in which:
        fig_training()
    if "picks" in which:
        fig_picks()
    if "real" in which:
        fig_real(["rgb_lines", "seamnet_rgb", "yolo", "fastsam_rgb", "sam2t_rgb"])
