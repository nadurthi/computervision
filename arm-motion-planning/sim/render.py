"""Videos and figures from the saved results (no planning here).

  video_fcl_ompl.mp4  lidar scan -> OctoMap -> per move: direct line or Informed RRT* tree + path
  video_moveit.mp4    the same OctoMap in MoveIt; MoveIt's RRT* / Informed RRT* trajectories
  cost_vs_time.png    benchmark: best path cost vs planning time, RRT* vs Informed RRT*
  scene.png           lidar points and occupied voxels
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from robot import fk_frames, skeleton  # noqa: E402

W, H, DPI, FPS = 1280, 720, 100, 30
LIM = dict(x=(-0.45, 1.0), y=(-0.9, 0.75), z=(0.0, 1.3))


def resample(Q, n):
    """n configurations evenly spaced along the joint-space path."""
    Q = np.asarray(Q, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    if s[-1] < 1e-9:
        return np.repeat(Q[:1], n, 0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, Q[:, j]) for j in range(Q.shape[1])], 1)


class Stage:
    def __init__(self, scene, title):
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
        self.ax = self.fig.add_axes([0.0, 0.0, 0.72, 0.92], projection="3d")
        self.title = self.fig.text(0.02, 0.95, title, fontsize=15, weight="bold")
        self.status = self.fig.text(0.73, 0.86, "", fontsize=11, va="top", family="monospace", wrap=True)
        self.sc = scene
        self.fig.text(0.73, 0.08, "orange: Panda arm (joint model)\ngreen: tool path\nred dots: planner tree (tool positions)\n"
                      "squares: OctoMap occupied voxels\nmagenta star: pre-grasp target", fontsize=9, color="#444444")
        self.az = -60.0
        self._setup()

    def _setup(self):
        ax = self.ax
        ax.set_xlim(*LIM["x"]); ax.set_ylim(*LIM["y"]); ax.set_zlim(*LIM["z"])
        ax.set_box_aspect([LIM["x"][1] - LIM["x"][0], LIM["y"][1] - LIM["y"][0], LIM["z"][1] - LIM["z"][0]])
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.view_init(elev=24, azim=self.az)
        o = self.sc["lidar_origin"]
        ax.scatter(*o, s=60, c="crimson", marker="^", depthshade=False)
        ax.text(o[0], o[1], o[2] + 0.06, "lidar", color="crimson", fontsize=9)
        self.pts = ax.scatter([], [], [], s=1, c="tab:blue", alpha=0.5, depthshade=False)
        self.vox = None
        self.tree = None
        self.arm, = ax.plot([], [], [], "-o", lw=9, ms=7, color="#e8590c", mfc="#222222", mec="#222222", zorder=10)
        self.hand, = ax.plot([], [], [], "-", lw=4, color="#111111", zorder=10)
        self.trail, = ax.plot([], [], [], "-", lw=3, color="#2b8a3e", zorder=9)
        self.goal = ax.scatter([], [], [], s=60, c="magenta", marker="*", depthshade=False)

    def show_points(self, P):
        self.pts._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

    def show_voxels(self, alpha=0.6):
        occ = self.sc["occupied"]
        if self.vox is not None:
            self.vox.remove()
        self.vox = self.ax.scatter(occ[:, 0], occ[:, 1], occ[:, 2], s=14, marker="s", c=occ[:, 2],
                                   cmap="viridis", vmin=0, vmax=1.2, alpha=alpha, depthshade=True, linewidths=0)

    def show_tree(self, P):
        if self.tree is not None:
            self.tree.remove(); self.tree = None
        if P is not None and len(P):
            self.tree = self.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4, c="#c92a2a", alpha=0.5, depthshade=False)

    def show_arm(self, q, trail=None, goal=None):
        S = skeleton(q)
        self.arm.set_data_3d(S[:-1, 0], S[:-1, 1], S[:-1, 2])
        F = fk_frames(q)
        h = F[8]
        a = (h @ np.array([0, -0.06, 0.06, 1]))[:3]; b = (h @ np.array([0, 0.06, 0.06, 1]))[:3]
        fa = (h @ np.array([0, -0.03, 0.105, 1]))[:3]; fb = (h @ np.array([0, 0.03, 0.105, 1]))[:3]
        f = F[7][:3, 3]
        hp = np.array([f, 0.5 * (a + b), a, fa, a, b, fb])
        self.hand.set_data_3d(hp[:, 0], hp[:, 1], hp[:, 2])
        if trail is not None and len(trail):
            T = np.asarray(trail)
            self.trail.set_data_3d(T[:, 0], T[:, 1], T[:, 2])
        else:
            self.trail.set_data_3d([], [], [])
        if goal is not None:
            self.goal._offsets3d = ([goal[0]], [goal[1]], [goal[2]])

    def turn(self, d=0.15):
        self.az += d
        self.ax.view_init(elev=24, azim=self.az)


def tcp_path(Q):
    return np.array([fk_frames(q)[9][:3, 3] for q in Q])


def video_fcl(out, sc, plans, R):
    st = Stage(sc, "Lidar → OctoMap → FCL → Informed RRT* (OMPL), Franka Panda")
    w = FFMpegWriter(fps=FPS, bitrate=4000)
    P, ring = sc["points_all"], sc["ring"]
    occ = sc["occupied"]
    with w.saving(st.fig, str(out / "video_fcl_ompl.mp4"), DPI):
        q0 = np.array(plans[0]["q_start"])
        st.show_arm(q0)
        rings = np.unique(ring)
        for k in range(0, len(rings), 2):  # 1. lidar scan, ring by ring
            m = ring <= rings[k]
            st.show_points(P[m])
            st.status.set_text(f"1  Simulated 3D lidar\n   {len(rings)} rings x 512 columns\n"
                               f"   returns so far: {m.sum()}\n   (8 mm range noise, 2% dropout)")
            st.turn(); w.grab_frame()
        lid = R["lidar"]; om = R["octomap"]
        for k in range(30):  # 2. octomap
            if k == 0:
                st.show_points(sc["points"])
                st.show_voxels()
            st.status.set_text(f"2  OctoMap (ray casting)\n   resolution {om['resolution'] * 100:.0f} cm\n"
                               f"   {lid['after_ground_and_self_filter']} points after ground\n"
                               f"   and robot-base filtering\n   {om['occupied_leaves']} occupied voxels\n"
                               f"   built in {om['insert_s'] * 1000:.0f} ms")
            st.turn(); w.grab_frame()
        st.show_points(np.zeros((0, 3)))
        f = R["fcl"]
        for p in plans:  # 3. moves
            if p.get("status") not in ("direct", "planned"):
                continue
            Q = resample(p["path"], 75 if p["status"] == "planned" else 30)
            goal = p["target_xyz"]
            if p["status"] == "planned":
                tf = out / f"tree_{p['name']}.npz"
                if tf.exists():
                    tq = np.load(tf)["q"]
                    tq = tq[np.random.default_rng(0).permutation(len(tq))[:4000]]
                    TP = tcp_path(tq)
                    for k in range(25):
                        st.show_tree(TP[: int(len(TP) * (k + 1) / 25)])
                        st.show_arm(Q[0], goal=goal)
                        st.status.set_text(
                            f"3  {p['name'].replace('_', ' ')}\n   straight joint line: COLLIDES\n"
                            f"   -> Informed RRT* ({p['planner']})\n   tree (tool positions of\n   {len(tq)} sampled vertices)\n"
                            f"   first path {p['t_first']:.1f} s\n   cost {p['cost_raw']:.2f} -> {p['cost']:.2f} rad\n"
                            f"   (simplified)\n   FCL: {f['spheres']} spheres vs octree\n   {f['mean_check_us']:.0f} us/state")
                        st.turn(); w.grab_frame()
            else:
                st.show_tree(None)
                st.status.set_text(f"3  {p['name'].replace('_', ' ')}\n   straight joint line:\n"
                                   f"   collision-free (FCL)\n   -> no planning needed\n"
                                   f"   {p['cost']:.2f} rad")
            trail = []
            for q in Q:
                trail.append(fk_frames(q)[9][:3, 3])
                st.show_arm(q, trail, goal)
                st.turn(); w.grab_frame()
            st.show_tree(None)
            for _ in range(8):
                w.grab_frame()
    plt.close(st.fig)


def video_moveit(out, sc, mplans, fplans, RM):
    st = Stage(sc, "MoveIt 2: same OctoMap in the planning scene, OMPL + FCL")
    w = FFMpegWriter(fps=FPS, bitrate=4000)
    by = {(p["name"], p["planner"]): p for p in mplans}
    st.show_voxels()
    with w.saving(st.fig, str(out / "video_moveit.mp4"), DPI):
        for fp in fplans:
            if fp.get("status") != "planned":
                continue
            for planner in ("RRTstar", "InformedRRTstar"):
                mp = by.get((fp["name"], planner))
                if mp is None:
                    continue
                rec = next(r for r in RM["picks"] if r["name"] == fp["name"])
                runs = [r for r in rec["runs"] if r["planner"] == planner]
                ok = [r for r in runs if r["success"]]
                L = np.mean([r["joint_length"] for r in ok]) if ok else float("nan")
                Q = resample(mp["path"], 75)
                trail = []
                for q in Q:
                    trail.append(fk_frames(q)[9][:3, 3])
                    st.show_arm(q, trail, fp["target_xyz"])
                    st.status.set_text(
                        f"{fp['name'].replace('_', ' ')}\n\nMoveIt planner:\n  {mp['planner']}\n"
                        f"  (OMPL geometric::RRTstar{', informed' if 'Informed' in planner else ''})\n"
                        f"  success {len(ok)}/{len(runs)}\n  mean length {L:.2f} rad\n"
                        f"  trajectory {mp['time_from_start'][-1]:.1f} s (TOTG)\n\n"
                        f"standalone path in MoveIt:\n  {rec['standalone_path_states_in_collision_moveit']} of "
                        f"{rec['standalone_path_states_checked']}\n  states in collision")
                    st.turn(); w.grab_frame()
                for _ in range(10):
                    w.grab_frame()
    plt.close(st.fig)


def cost_figure(out, R):
    b = R.get("benchmark")
    if not b:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=120)
    colors = {"RRTstar": "tab:orange", "InformedRRTstar": "tab:blue"}
    for planner in ("RRTstar", "InformedRRTstar"):
        runs = [r for r in b["runs"] if r["planner"] == planner]
        for i, r in enumerate(runs):
            t = [c[0] for c in r["curve"] if c[1] is not None]
            c = [c[1] for c in r["curve"] if c[1] is not None]
            ax.step(t, c, where="post", color=colors[planner], alpha=0.8,
                    label=f"{planner} ({sum(1 for x in runs if x['t_first'] is not None)}/{len(runs)} solved)" if i == 0 else None)
    ax.set_xlabel("planning time [s]"); ax.set_ylabel("best path length [rad, joint space]")
    ax.set_title(f"{b['pick'].replace('_', ' ')}: best cost vs time")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "cost_vs_time.png"); plt.close(fig)


def scene_figure(out, sc):
    st = Stage(sc, "Lidar returns (blue) and OctoMap occupied voxels")
    st.show_points(sc["points"]); st.show_voxels(0.5)
    st.fig.savefig(out / "scene.png"); plt.close(st.fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    sc = dict(np.load(out / "scene.npz"))
    plans = json.loads((out / "plans.json").read_text())["plans"]
    R = json.loads((out / "results_fcl_ompl.json").read_text())
    scene_figure(out, sc)
    cost_figure(out, R)
    video_fcl(out, sc, plans, R)
    print("[render] video_fcl_ompl.mp4")
    if (out / "moveit_plans.json").exists():
        mplans = json.loads((out / "moveit_plans.json").read_text())["plans"]
        RM = json.loads((out / "results_moveit.json").read_text())
        video_moveit(out, sc, mplans, plans, RM)
        print("[render] video_moveit.mp4")


if __name__ == "__main__":
    main()
