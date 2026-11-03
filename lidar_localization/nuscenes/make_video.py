"""Render a localization run as an MP4: BEV map + current scan placed at the filter's estimate, ground-truth and
estimated paths, and the error traces of all three modes drawn as the drive plays.

usage: python make_video.py <nuscenes_root> <results_dir> <out_dir> [scene ...]
Reads <results_dir>/<scene>_{tracking,tracking_paper,global}.json written by run_nuscenes.py.
"""
import json, os, sys, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nusc_data as nd
import lidarloc as ll

W, H, FPS = 1280, 720, 10
MODES = [("tracking", "#2563eb", "tracking"), ("tracking_paper", "#9ca3af", "tracking, paper params"),
         ("global", "#dc2626", "global (unknown start)")]


def render(ns, res, out, sc, show="global"):
    runs = {m: json.load(open(os.path.join(res, f"{sc}_{m}.json"))) for m, _, _ in MODES
            if os.path.exists(os.path.join(res, f"{sc}_{m}.json"))}
    recs = runs[show]
    frames = nd.sweep_frames(ns, sc, every=2)
    assert len(frames) == len(recs)
    Xmap, _ = nd.static_keyframe_map(ns, sc)
    Xm = ll.voxel_down(Xmap, 0.4)
    zg = np.percentile(Xm[:, 2], 5)
    Xm = Xm[(Xm[:, 2] > zg + 0.3) & (Xm[:, 2] < zg + 12)]           # drop the ground for a readable BEV
    gt = np.array([r["gt"] for r in recs]); est = np.array([r["est"] for r in recs])
    t = np.array([r["t"] for r in recs]) - recs[0]["t"]
    loc = ns.location(sc)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    axm = fig.add_axes([0.03, 0.06, 0.52, 0.84]); axe = fig.add_axes([0.62, 0.40, 0.35, 0.46])
    axt = fig.add_axes([0.62, 0.06, 0.35, 0.26]); axt.axis("off")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    lo, hi = Xm[:, :2].min(0), Xm[:, :2].max(0)
    for k, fr in enumerate(frames):
        axm.clear(); axe.clear(); axt.clear(); axt.axis("off")
        # camera: wide when the estimate is far from the truth, close-in once locked
        c = 0.5 * (gt[k, :2] + est[k, :2])
        half = max(35.0, 0.6 * np.linalg.norm(gt[k, :2] - est[k, :2]) + 20)
        half = min(half, 0.55 * max(hi - lo))
        axm.scatter(Xm[:, 0], Xm[:, 1], s=0.3, c="#9ca3af", lw=0, rasterized=True)
        X = ll.voxel_down(fr["X"], 0.3)
        Hk = ll.pose_to_H(np.r_[est[k], np.zeros(5)])
        Y = X @ Hk[:3, :3].T + Hk[:3, 3]
        Y = Y[Y[:, 2] > est[k, 2] + 0.4]                                   # hide the scan's ground rings too
        axm.scatter(Y[:, 0], Y[:, 1], s=0.8, c="#0ea5e9", lw=0, rasterized=True)
        axm.plot(gt[:k + 1, 0], gt[:k + 1, 1], "k-", lw=1.8, label="ground truth")
        E = est[:k + 1, :2].copy()
        jump = np.r_[False, np.linalg.norm(np.diff(E, axis=0), axis=1) > 5.0]   # don't draw re-localization jumps
        E[jump] = np.nan
        axm.plot(E[:, 0], E[:, 1], "-", color="#dc2626", lw=1.5, label="filter estimate")
        for P, col in [(gt[k], "k"), (est[k], "#dc2626")]:
            axm.arrow(P[0], P[1], 4 * np.cos(P[3]), 4 * np.sin(P[3]), width=0.6, color=col, length_includes_head=True)
        axm.set_xlim(c[0] - half, c[0] + half); axm.set_ylim(c[1] - half, c[1] + half); axm.set_aspect("equal")
        axm.set_xticks([]); axm.set_yticks([])
        axm.legend(loc="upper left", fontsize=9, framealpha=0.85)
        axm.text(0.99, 0.01, f"{2 * half:.0f} m across", transform=axm.transAxes, ha="right", va="bottom", fontsize=8, color="#4b5563")
        for m, col, lab in MODES:
            if m in runs:
                e = np.array([r["exy"] for r in runs[m][:k + 1]])
                axe.semilogy(t[:k + 1], np.maximum(e, 1e-2), color=col, lw=1.4, label=lab)
        axe.axhline(1.0, color="#6b7280", ls=":", lw=0.8)
        axe.set_xlim(0, t[-1]); axe.set_ylim(1e-2, 300); axe.set_xlabel("time [s]"); axe.set_ylabel("position error [m]")
        axe.legend(fontsize=8, loc="upper right"); axe.spines[["top", "right"]].set_visible(False)
        r = recs[k]
        axt.text(0, 0.95, f"t = {t[k]:5.1f} s    frame {k + 1}/{len(recs)}", fontsize=11, va="top", family="monospace")
        axt.text(0, 0.65, f"{show} run:  position error {r['exy']:7.2f} m\n               yaw error {r['eyaw']:8.2f} deg",
                 fontsize=11, va="top", family="monospace")
        axt.text(0, 0.12, "blue: current lidar sweep placed at the estimate\ngrey: map (keyframes, objects removed, ground hidden)",
                 fontsize=8.5, color="#4b5563", va="bottom")
        fig.suptitle(f"LIDAR-only localization on nuScenes  ·  {sc} ({loc})", fontsize=13, x=0.03, ha="left", y=0.975)
        fig.canvas.draw()
        p.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    p.stdin.close(); p.wait(); plt.close(fig)


if __name__ == "__main__":
    root, res, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    ns = nd.NuScenes(root, "v1.0-mini")
    for sc in sys.argv[4:] or sorted(ns.scenes):
        render(ns, res, os.path.join(out, f"{sc}.mp4"), sc)
        print(sc, os.path.getsize(os.path.join(out, f"{sc}.mp4")) // 1024, "KB", flush=True)
