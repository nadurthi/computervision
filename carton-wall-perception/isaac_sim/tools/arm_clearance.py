"""Offline arm-vs-trailer clearance check for trailer_unload.py (no Isaac Sim: pxr + numpy only).

Places the robot asset in the trailer exactly as trailer_unload.py does, finds the same start pose, then follows the
same pick waypoints (arm_kin.pick_waypoints) for the boxes the simulation would pick (top first, nearest the robot),
with a DLS IK twin of the controller, and reports for every leg how close any collision-mesh vertex of any link comes
to the roof, the side walls/posts and the floor.

    curl -O https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd
    python tools/arm_clearance.py --usd kr210_l150.usd --standoff 1.85 2.2 2.5 --pedestal 0.3 --picks 6
"""
import argparse
import math
import os
import sys

import numpy as np
from pxr import Usd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import arm_kin as ak                                                 # noqa: E402
import trailer_common as tc                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--root", default=None, help="robot root prim (default: the stage's default prim)")
ap.add_argument("--standoff", type=float, nargs="+", default=[1.85])
ap.add_argument("--pedestal", type=float, nargs="+", default=[0.30])
ap.add_argument("--roof-margin", type=float, nargs="+", default=[0.28], help="suction point max height below roof")
ap.add_argument("--reach", type=float, default=2.20)
ap.add_argument("--picks", type=int, default=6)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--rows", type=int, default=2)
ap.add_argument("--tool-len", type=float, default=0.15)
ap.add_argument("--no-pitch", action="store_true")
ap.add_argument("--belt-y", type=float, default=0.88)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

SH_Z, SH_R, WRIST, CUP_R, BELT_TOP = 0.75, 0.35, 0.1925, 0.05, 0.90
SIDE = tc.W / 2 - 0.05                                               # posts / rails stick 5 cm into the box


def aabb(b):
    d, w, h = b["size_m"]; c = np.array(b["center_m"]); yaw = math.radians(b["yaw_deg"])
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    P = np.array([c + R @ np.array([sx * d / 2, sy * w / 2, sz * h / 2])
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return {"lo": P.min(0), "hi": P.max(0)}


def run(standoff, pedestal, roof_margin):
    rng = np.random.default_rng(args.seed)
    plan = tc.plan_packed_wall(rng, rows=args.rows)
    front = min(b["front_face_x_m"] for b in plan)
    base = np.array([front - standoff, 0.0, pedestal])
    st = Usd.Stage.Open(args.usd)
    root = args.root or str(st.GetDefaultPrim().GetPath())
    B = np.eye(4); B[:3, 3] = base
    fk = ak.ChainFK(st, root, base_T=B)
    ee = next(b for b in reversed(fk.bodies) if b.split("/")[-1] in ("tool0", "flange", "link_6"))
    pts = fk.link_points()
    home = np.array([front - 0.45, 0.0, 1.60])
    q0, tip = fk.search(ee, args.tool_len + 0.05, home, front, tc.H, tc.W)
    if q0 is None:
        return {"standoff": standoff, "pedestal": pedestal, "error": "no start pose"}
    ik = ak.IKTracker(fk, ee, fk.tool_axis(ee), args.tool_len, q0)
    moving = [b for b in pts if b not in fk.roots and b != fk.bodies[0]]
    worst = {"roof": 9.0, "side": 9.0, "floor": 9.0}
    legs_out = []

    def clearance(t):
        _, _, _, W = t.tool()
        out = {"roof": 9.0, "side": 9.0, "floor": 9.0, "who": ""}
        for b in moving:
            P = pts[b] @ W[b][:3, :3].T + W[b][:3, 3]
            r = tc.H - P[:, 2].max()
            if r < out["roof"]:
                out["roof"], out["who"] = r, b.split("/")[-1]
            out["side"] = min(out["side"], SIDE - np.abs(P[:, 1]).max())
            out["floor"] = min(out["floor"], P[:, 2].min())
        return out

    def leg(name, target, a, box=""):
        acc = {"roof": 9.0, "side": 9.0, "floor": 9.0, "who": ""}
        def on(t):
            c = clearance(t)
            if c["roof"] < acc["roof"]:
                acc["roof"], acc["who"] = c["roof"], c["who"]
            acc["side"] = min(acc["side"], c["side"]); acc["floor"] = min(acc["floor"], c["floor"])
        err, ang = ik.move(np.asarray(target, float), a, on_step=on)
        for k in worst:
            worst[k] = min(worst[k], acc[k])
        legs_out.append((box, name, acc["roof"], acc["who"], acc["side"], err, ang))
        return err, ang

    leg("home", home, (1, 0, 0))
    removed, unreachable, picks = set(), set(), 0
    while picks < args.picks:
        wall_front = min(o["center_m"][0] - o["half_extent_xy_m"][0] for j, o in enumerate(plan) if j not in removed)
        reach = []
        for i in ak.free_boxes(plan, removed):
            if i in unreachable:
                continue
            b = plan[i]
            face = ak.suction_face(np.array(b["center_m"]), b["size_m"], tc.H, roof_margin)
            wc = face - np.array([args.tool_len + WRIST, 0, 0])
            dxy = wc[:2] - base[:2]; r = max(np.linalg.norm(dxy), 1e-6)
            sh = base + np.array([0, 0, SH_Z]) + np.r_[dxy / r * SH_R, 0.0]
            if np.linalg.norm(wc - sh) < args.reach:
                reach.append((b["center_m"][2] + b["size_m"][2] / 2, -abs(face[1]), i, face))
            else:
                unreachable.add(i)
        if not reach:
            break
        top_z, _, i, face = max(reach)
        b = plan[i]
        t = aabb(b)
        col = [aabb(o)["lo"][0] for j, o in enumerate(plan) if j not in removed and j != i
               and min(t["hi"][1], aabb(o)["hi"][1]) - max(t["lo"][1], aabb(o)["lo"][1]) > 0
               and aabb(o)["lo"][2] < t["lo"][2] + 0.02]
        wp = ak.pick_waypoints(face, b["size_m"], top_z, t, min(col) if col else wall_front, wall_front, tc.H, CUP_R,
                               not args.no_pitch, args.belt_y, BELT_TOP, pedestal)
        name = f"box_{i:03d}@{face[2]:.2f}m/{math.degrees(wp['theta']):.0f}deg"
        for lname, p, a in wp["legs"]:
            leg(lname, p, a, name)
        leg("home", home, (1, 0, 0), name)
        removed.add(i); picks += 1
    return {"standoff": standoff, "pedestal": pedestal, "roof_margin": roof_margin, "picks": picks,
            "unreachable": len(unreachable), "worst": worst, "legs": legs_out, "start_tip": tip}


for so in args.standoff:
    for pd in args.pedestal:
        for rm in args.roof_margin:
            r = run(so, pd, rm)
            if "error" in r:
                print(r); continue
            w = r["worst"]
            bad = [l for l in r["legs"] if l[2] < 0.03 or l[4] < 0.03 or l[5] > 0.03]
            print(f"standoff {so:.2f} pedestal {pd:.2f} roof-margin {rm:.2f}: {r['picks']} picks, "
                  f"{r['unreachable']} unreachable so far; min clearance roof {100 * w['roof']:.1f} cm, "
                  f"side {100 * w['side']:.1f} cm, floor {100 * w['floor']:.1f} cm; {len(bad)} bad legs")
            for l in (r["legs"] if args.verbose else bad):
                print(f"   {l[0]:28s} {l[1]:9s} roof {100 * l[2]:6.1f} cm ({l[3]}) side {100 * l[4]:6.1f} cm "
                      f"tracking err {100 * l[5]:5.1f} cm {l[6]:4.0f} deg")
