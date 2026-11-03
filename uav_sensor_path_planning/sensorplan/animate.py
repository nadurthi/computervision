"""MP4 animations of simulation runs (matplotlib + ffmpeg)."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Rectangle  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BG = "#0f1420"
PANEL = "#151b2b"
FG = "#e6e9ef"
MUTED = "#8a93a6"
GRIDC = "#2a3247"
UAV_C = ["#4ea1ff", "#ff9f43", "#7bed9f"]
TARGET_C = ["#ff6b81", "#ffd166", "#06d6a0", "#a78bfa", "#f78c6b",
            "#4cc9f0", "#f15bb5", "#c0eb75", "#ffa69e", "#9bf6ff"]
EXPLORED = LinearSegmentedColormap.from_list("expl", [PANEL, "#1f4d5a"])


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _ellipse(m, P, nsig=1.0, **kw):
    w, v = np.linalg.eigh(P[:2, :2])
    w = np.sqrt(np.maximum(w, 1e-9))
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    return Ellipse(m[:2], 2 * nsig * w[1], 2 * nsig * w[0], angle=ang, **kw)


class MapView:
    """Draws one run on one axes; `draw(f)` updates to fractional step f."""

    def __init__(self, ax, res, title, show_plan=True, trail=12):
        self.ax, self.res, self.trail, self.show_plan = ax, res, trail, show_plan
        _style(ax)
        nodes = res["grid_nodes"]
        xs, ys = np.unique(nodes[:, 0]), np.unique(nodes[:, 1])
        self.shape = (len(ys), len(xs))
        d = xs[1] - xs[0]
        ext = (xs[0] - d / 2, xs[-1] + d / 2, ys[0] - d / 2, ys[-1] + d / 2)
        pad = 0.08 * (xs[-1] - xs[0])
        ax.set_xlim(xs[0] - pad, xs[-1] + pad)
        ax.set_ylim(ys[0] - pad, ys[-1] + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.scatter(nodes[:, 0], nodes[:, 1], s=2 if len(nodes) < 300 else 0.6, c=GRIDC, zorder=1)
        H = res["coverage"]
        self.expl = 1 - _bin_h(H) / np.log(2)
        self.img = ax.imshow(self.expl[0].reshape(self.shape), origin="lower", extent=ext,
                             cmap=EXPLORED, vmin=0, vmax=1, zorder=0, interpolation="nearest")
        ax.set_title(title, color=FG, fontsize=12, loc="left", pad=6)
        self.hud = ax.text(0.01, 0.01, "", transform=ax.transAxes, color=FG, fontsize=9,
                           va="bottom", family="monospace", zorder=20,
                           bbox=dict(fc=BG, ec="none", alpha=0.75, pad=3))
        self.dyn = []

    def draw(self, f):
        for a in self.dyn:
            a.remove()
        self.dyn = []
        ax, r = self.ax, self.res
        K = r["t"].shape[0]
        k = min(int(np.floor(f)), K - 1)
        a = f - k if k < K - 1 else 0.0
        k1 = min(k + 1, K - 1)
        self.img.set_data(self.expl[k].reshape(self.shape))
        nodes = r["grid_nodes"]
        # --- targets
        truth, st = r["truth"], r["status"]
        for i in range(truth.shape[0]):
            c = TARGET_C[i % len(TARGET_C)]
            tp = (1 - a) * truth[i, k] + a * truth[i, k1]
            k0 = max(0, k - self.trail * 2)
            tr = np.vstack([truth[i, k0:k + 1, :2], tp[None, :2]])
            found = (st[i, :k + 1] > 0).any()
            self.dyn += ax.plot(tr[:, 0], tr[:, 1], "-", color=c if found else MUTED, lw=0.8, alpha=0.45, zorder=3)
            self.dyn.append(ax.scatter([tp[0]], [tp[1]], marker="*", s=70,
                                       facecolor=c if found else "none",
                                       edgecolor=c if found else MUTED, lw=0.8, zorder=6))
            s = st[i, k]
            if s == 1:
                m = (1 - a) * r["est_x"][i, k] + a * r["est_x"][i, k1] if st[i, k1] == 1 else r["est_x"][i, k]
                e = _ellipse(m, r["est_P"][i, k], 1.0, fc=c, ec=c, alpha=0.18, lw=1.2, zorder=4)
                ax.add_patch(e)
                e2 = _ellipse(m, r["est_P"][i, k], 1.0, fc="none", ec=c, lw=1.2, zorder=4)
                ax.add_patch(e2)
                self.dyn += [e, e2]
                self.dyn.append(ax.scatter([m[0]], [m[1]], marker="s", s=14, c=c, zorder=7))
            elif s == 2:
                e = _ellipse(r["est_x"][i, k], r["est_P"][i, k], 1.0, fc="none", ec=MUTED, ls=":", lw=1, zorder=4)
                ax.add_patch(e)
                self.dyn.append(e)
        # --- UAVs
        for j in range(r["uav"].shape[0]):
            c = UAV_C[j]
            p0, p1 = nodes[r["uav"][j, k]], nodes[r["uav"][j, k1]]
            p = (1 - a) * p0 + a * p1
            k0 = max(0, k - self.trail)
            tr = np.vstack([nodes[r["uav"][j, k0:k + 1]], p[None]])
            self.dyn += ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=2, alpha=0.7, zorder=8)
            fov = Circle(p, r["fov"][j], fc=c, ec=c, alpha=0.12, lw=0, zorder=2)
            fov2 = Circle(p, r["fov"][j], fc="none", ec=c, lw=1.2, alpha=0.9, zorder=9)
            ax.add_patch(fov)
            ax.add_patch(fov2)
            self.dyn += [fov, fov2]
            self.dyn.append(ax.scatter([p[0]], [p[1]], marker="^", s=90, c=c, edgecolor="white", lw=0.8, zorder=10))
            if self.show_plan:
                kp = max(kk for kk in r["plans"] if kk <= k)
                path = r["plans"][kp][j]
                rest = path[k - kp + 1:]
                if len(rest):
                    pp = np.vstack([p[None], nodes[rest]])
                    self.dyn += ax.plot(pp[:, 0], pp[:, 1], "--", color=c, lw=1.2, alpha=0.9, zorder=8)
                    self.dyn.append(ax.scatter(pp[1:, 0], pp[1:, 1], s=12, c=c, zorder=8))
        n_act = int((st[:, k] == 1).sum())
        n_found = int((st[:, :k + 1] > 0).any(1).sum())
        self.hud.set_text(f"t = {r['t'][k] + a * (r['t'][k1] - r['t'][k]):5.1f} s   "
                          f"tracked {n_act}/{truth.shape[0]}   found {n_found}/{truth.shape[0]}")
        return self.dyn


def _bin_h(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _timeline(ax, res, title=None):
    _style(ax)
    st, t = res["status"], res["t"]
    Nt = st.shape[0]
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.6, Nt - 0.4)
    ax.set_yticks(range(Nt))
    ax.set_yticklabels([f"T{i}" for i in range(Nt)])
    ax.invert_yaxis()
    ax.set_xlabel("time [s]", fontsize=9)
    if title:
        ax.set_title(title, color=FG, fontsize=10, loc="left")
    dt = t[1] - t[0]
    for i in range(Nt):
        for k in range(len(t)):
            if st[i, k]:
                c = TARGET_C[i % len(TARGET_C)] if st[i, k] == 1 else "#3a4258"
                ax.add_patch(Rectangle((t[k] - dt / 2, i - 0.35), dt, 0.7, fc=c, ec="none", alpha=0.9))
    # measurement fixes (white ticks) and track losses (red x)
    for (k, j, i, _) in res["meas"]:
        ax.plot([t[k], t[k]], [i - 0.35, i + 0.35], color="white", lw=0.8, alpha=0.9, zorder=3)
    for i in range(Nt):
        for k in np.nonzero((st[i, :-1] == 1) & (st[i, 1:] == 2))[0]:
            ax.scatter([t[k + 1]], [i], marker="x", s=36, c="#ff4d4d", lw=1.6, zorder=4)
    cover = ax.add_patch(Rectangle((t[0], -0.6), t[-1] - t[0], Nt, fc=PANEL, ec="none", zorder=5))
    cursor = ax.axvline(t[0], color=FG, lw=1, zorder=6)
    return cover, cursor


def render(runs, out, fps=25, substeps=5, dpi=100, size=(12.8, 7.2), suptitle=None):
    """runs: list of (res, title). One run -> map + status timeline; two runs ->
    side-by-side maps with a shared '% tracked' chart."""
    fig = plt.figure(figsize=size, dpi=dpi, facecolor=BG)
    K = runs[0][0]["t"].shape[0]
    views, extra = [], []
    if len(runs) == 1:
        res, title = runs[0]
        ax = fig.add_axes([0.03, 0.04, 0.50, 0.78])
        views.append(MapView(ax, res, title))
        axt = fig.add_axes([0.62, 0.10, 0.36, 0.70])
        cover, cursor = _timeline(axt, res, "target status · colour = tracked, grey = lost, | = position fix, × = track lost")
        t = res["t"]

        def upd_extra(f):
            k = min(int(f), K - 1)
            x = t[k]
            cover.set_x(x + (t[1] - t[0]) / 2)
            cover.set_width(t[-1] - x + 10)
            cursor.set_xdata([x, x])
        extra.append(upd_extra)
    else:
        n = len(runs)
        w = 0.96 / n
        for q, (res, title) in enumerate(runs):
            ax = fig.add_axes([0.02 + q * w, 0.30, w - 0.01, 0.53])
            views.append(MapView(ax, res, title))
        axc = fig.add_axes([0.07, 0.07, 0.88, 0.17])
        _style(axc)
        t = runs[0][0]["t"]
        axc.set_xlim(t[0], t[-1])
        axc.set_ylim(0, 105)
        axc.set_ylabel("% tracked", fontsize=9)
        axc.set_xlabel("time [s]", fontsize=9)
        lines = []
        for q, (res, title) in enumerate(runs):
            y = 100 * (res["status"] == 1).mean(0)
            ln, = axc.plot([], [], color=["#4ea1ff", "#ff9f43", "#c0eb75"][q], lw=2, label=title.split("\n")[0])
            lines.append((ln, y))
        axc.legend(loc="lower right", fontsize=8, facecolor=PANEL, edgecolor=GRIDC, labelcolor=FG)

        def upd_extra(f):
            k = min(int(f), K - 1)
            for ln, y in lines:
                ln.set_data(t[:k + 1], y[:k + 1])
        extra.append(upd_extra)
    if suptitle:
        fig.text(0.02, 0.975, suptitle, color=FG, fontsize=13, weight="bold", va="top")
    fig.text(0.985, 0.975, "▲ UAV + FOV   - - planned way-points   ★ true target (hollow = not yet found)\n"
             "■ estimate + 1σ   dotted = lost track   teal cells = recently searched",
             color=MUTED, fontsize=8, ha="right", va="top", linespacing=1.5)
    frames = np.arange(0, (K - 1) * substeps + 1) / substeps
    frames = np.concatenate([frames, np.full(fps, frames[-1])])  # 1 s hold at the end

    def update(f):
        for v in views:
            v.draw(f)
        for e in extra:
            e(f)
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                                    extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast",
                                                "-crf", "24", "-movflags", "+faststart"])
    anim.save(out, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
