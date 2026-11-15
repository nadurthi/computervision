"""Standalone pipeline: simulated lidar -> OctoMap -> FCL -> (if needed) OMPL Informed RRT*.

For every pick:
  1. IK for a pre-grasp pose in front of the carton face (tool pointing into the wall)
  2. check the straight joint-space line start -> goal with FCL against the OctoMap
  3. only if that line collides, plan with OMPL Informed RRT* (path length cost)
Then a small benchmark on the hardest pick: RRT* vs Informed RRT*, several seeds,
cost of the best path versus planning time.

Writes to --out: scene.npz, octomap.bt, octomap_raw.bin, plans.json, results_fcl_ompl.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene import build_scene, Lidar, preprocess  # noqa: E402
from robot import ik, tool_pointing, tcp, LOWER, UPPER, JOINT_NAMES  # noqa: E402
from collision import build_octomap, octomap_bytes, ArmChecker  # noqa: E402

from ompl import base as ob, geometric as og, util as ou  # noqa: E402

ou.setLogLevel(ou.LogLevel.LOG_WARN)
ou.RNG.setSeed(42)  # OMPL allows one global seed per process; runs after the first differ by design

START_POSE = dict(p=[0.15, -0.58, 0.48], z=[0, 0, -1], x=[1, 0, 0])  # over the conveyor
STANDOFF = 0.10  # pre-grasp: 10 cm in front of the face
CONTACT = 0.06   # end of the approach move


def pick_targets(boxes):
    """Three cartons: top centre, middle left and bottom left."""
    cartons = [b for b in boxes if b.kind == "carton"]
    rows = sorted({round(b.lo[2], 3) for b in cartons})

    def nearest(row, y):
        cand = [b for b in cartons if round(b.lo[2], 3) == rows[row]]
        return min(cand, key=lambda b: abs(b.centre[1] - y))

    return [("top_centre", nearest(len(rows) - 1, 0.0)),
            ("middle_left", nearest(1, 0.30)),
            ("bottom_left", nearest(0, 0.40))]


def pregrasp(box):
    p = np.array([box.lo[0] - STANDOFF, box.centre[1], box.centre[2]])
    return p, tool_pointing([1, 0, 0], [0, 0, -1])


def roll(R, a):
    """Rotate the tool frame about its own z (the approach axis)."""
    c, s = np.cos(a), np.sin(a)
    return R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def choose_ik(p, R, chk, near, rng, free_roll=False, margin=0.2):
    """Collision-free IK solution closest to `near` in joint space, at least `margin` rad
    inside the joint limits. With free_roll, any rotation about the approach axis is
    accepted (a suction cup is round), tried in 30 deg steps."""
    rolls = np.deg2rad(np.arange(0, 360, 30)) if free_roll else [0.0]
    best = None
    for a in rolls:
        for q in ik(p, roll(R, a), seeds=[near], rng=rng, n_random=12 if free_roll else 40):
            if np.minimum(q - LOWER, UPPER - q).min() < margin or not chk.valid(q):
                continue
            if best is None or np.linalg.norm(q - near) < np.linalg.norm(best - near):
                best = q
    return best


def make_setup(chk, q_start, q_goal, planner_name, rng_range=1.2):
    space = ob.RealVectorStateSpace(7)
    b = ob.RealVectorBounds(7)
    for i in range(7):
        b.setLow(i, float(LOWER[i])); b.setHigh(i, float(UPPER[i]))
    space.setBounds(b)
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(lambda s: chk.valid([s[i] for i in range(7)]))
    si = ss.getSpaceInformation()
    # extent of the joint box is ~11 rad, so 0.002 -> one check per ~0.022 rad (~1.5 cm at the tool,
    # below the 4 cm voxel size)
    si.setStateValidityCheckingResolution(0.002)
    s, g = space.allocState(), space.allocState()
    for i in range(7):
        s[i] = float(q_start[i]); g[i] = float(q_goal[i])
    ss.setStartAndGoalStates(s, g)
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    planner = {"InformedRRTstar": og.InformedRRTstar, "RRTstar": og.RRTstar,
               "RRTConnect": og.RRTConnect}[planner_name](si)
    if hasattr(planner, "setRange"):
        planner.setRange(rng_range)
    if hasattr(planner, "setGoalBias"):
        planner.setGoalBias(0.2)
    ss.setPlanner(planner)
    ss.setup()
    return ss, si, planner


def path_to_array(path):
    return np.array([[path.getState(k)[i] for i in range(7)] for k in range(path.getStateCount())])


def joint_length(Q):
    return float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))


def plan(chk, q_start, q_goal, planner_name, total_time, step, seed):
    """Run an anytime planner in slices; record (time, best cost) after each slice."""
    ss, si, planner = make_setup(chk, q_start, q_goal, planner_name)
    curve, t, t_first = [], 0.0, None
    while t < total_time - 1e-9:
        t0 = time.perf_counter()
        ss.solve(step)
        t += time.perf_counter() - t0
        if ss.haveExactSolutionPath():
            c = ss.getSolutionPath().length()
            if t_first is None:
                t_first = t
            curve.append((round(t, 3), float(c)))
        else:
            curve.append((round(t, 3), None))
    out = dict(planner=planner_name, seed=int(seed), curve=curve, t_first=t_first)
    if ss.haveExactSolutionPath():
        raw = path_to_array(ss.getSolutionPath())
        simp = og.PathSimplifier(si)
        p = og.PathGeometric(ss.getSolutionPath())
        simp.simplifyMax(p)
        out.update(raw=raw.tolist(), simplified=path_to_array(p).tolist(),
                   cost_raw=joint_length(raw), cost_simplified=joint_length(path_to_array(p)))
        # tree, for the video: TCP positions of every vertex and each edge
        pd = ob.PlannerData(si)
        planner.getPlannerData(pd)
        V = np.array([[pd.getVertex(k).getState()[i] for i in range(7)] for k in range(pd.numVertices())])
        out["tree_q"] = V
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--scene-seed", type=int, default=3)
    ap.add_argument("--res", type=float, default=0.04, help="OctoMap resolution [m]")
    ap.add_argument("--plan-time", type=float, default=15.0, help="Informed RRT* time per move [s]")
    ap.add_argument("--bench-seeds", type=int, default=5)
    ap.add_argument("--bench-time", type=float, default=20.0)
    ap.add_argument("--quick", action="store_true", help="short run for testing")
    a = ap.parse_args()
    if a.quick:
        a.plan_time, a.bench_seeds, a.bench_time = 3.0, 2, 4.0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    R = {"config": vars(a)}

    # 1. scene + lidar
    boxes = build_scene(a.scene_seed)
    lidar = Lidar()
    t0 = time.perf_counter()
    pts_all, ring = lidar.scan(boxes, rng)
    pts = preprocess(pts_all)
    R["lidar"] = dict(rays=lidar.n_rings * lidar.n_azimuth, returns=int(len(pts_all)),
                      after_ground_and_self_filter=int(len(pts)),
                      scan_sim_s=round(time.perf_counter() - t0, 3))
    print(f"[lidar] {len(pts_all)} returns, {len(pts)} after filtering")

    # 2. OctoMap
    tree, occ, free, dt = build_octomap(pts, lidar.origin, a.res, lidar.max_range)
    raw = octomap_bytes(tree, out / "octomap.bt")
    (out / "octomap_raw.bin").write_bytes(raw)
    R["octomap"] = dict(resolution=a.res, insert_s=round(dt, 3), occupied_leaves=int(len(occ)),
                        free_leaves=int(len(free)), binary_bytes=len(raw))
    print(f"[octomap] {len(occ)} occupied leaves at {a.res} m in {dt:.3f} s")

    # 3. FCL checker
    chk = ArmChecker(raw, a.res)
    q_home = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])
    q_start = choose_ik(np.array(START_POSE["p"]), tool_pointing(START_POSE["z"], START_POSE["x"]),
                        chk, q_home, rng, free_roll=True)
    assert q_start is not None, "no collision-free start configuration"
    qs = np.random.default_rng(1).uniform(LOWER, UPPER, (2000, 7))
    chk.n_checks, chk.t_checks = 0, 0.0
    frac = np.mean([chk.valid(q) for q in qs])
    R["fcl"] = dict(spheres=len(chk.radii), mean_check_us=round(1e6 * chk.t_checks / chk.n_checks, 1),
                    random_states_valid=float(frac))
    print(f"[fcl] {1e6 * chk.t_checks / chk.n_checks:.0f} us per state check, {frac:.0%} of random states valid")

    # 4. picks. Each pick is two moves:
    #    transit  start -> pre-grasp (10 cm from the face)
    #    approach pre-grasp -> contact standoff (6 cm; the suction cup's compliance closes the rest)
    #    For each move: straight joint line if FCL says it is free, otherwise Informed RRT*.
    plans, hardest = [], None
    for name, box in pick_targets(boxes):
        p, Rg = pregrasp(box)
        q_goal = choose_ik(p, Rg, chk, q_start, rng, free_roll=True)
        if q_goal is None:
            plans.append(dict(name=name, box=box.name, status="no_ik")); print(f"[{name}] no IK"); continue
        p_contact = p + np.array([STANDOFF - CONTACT, 0, 0])
        Rq = tcp(q_goal)[:3, :3]  # keep the roll chosen for the pre-grasp
        q_contact = choose_ik(p_contact, Rq, chk, q_goal, rng, margin=0.0)
        moves = [("transit", q_start, q_goal, p)]
        if q_contact is not None:
            moves.append(("approach", q_goal, q_contact, p_contact))
        for kind, qa, qb, target in moves:
            rec = dict(name=f"{name}_{kind}", pick=name, move=kind, box=box.name, target_xyz=target.tolist(),
                       q_start=qa.tolist(), q_goal=qb.tolist(),
                       tcp_err_m=float(np.linalg.norm(tcp(qb)[:3, 3] - target)),
                       straight_line_cost=joint_length(np.array([qa, qb])))
            t0 = time.perf_counter()
            direct = chk.segment_valid(qa, qb)
            rec["direct_check_s"] = round(time.perf_counter() - t0, 3)
            if direct:
                rec.update(status="direct", path=[qa.tolist(), qb.tolist()], cost=rec["straight_line_cost"])
                print(f"[{rec['name']}] straight joint line is collision-free: no planning needed")
            else:
                r = plan(chk, qa, qb, "InformedRRTstar", a.plan_time, 0.5, seed=0)
                if "simplified" in r:
                    rec.update(status="planned", planner="InformedRRTstar", t_first=r["t_first"],
                               cost_raw=r["cost_raw"], cost=r["cost_simplified"], path=r["simplified"],
                               raw_path=r["raw"], curve=r["curve"])
                    np.savez_compressed(out / f"tree_{rec['name']}.npz", q=r["tree_q"])
                    print(f"[{rec['name']}] line collides -> Informed RRT*: first path {r['t_first']:.2f} s, "
                          f"cost {r['cost_raw']:.2f} rad, simplified {r['cost_simplified']:.2f} rad "
                          f"(straight line {rec['straight_line_cost']:.2f})")
                    if hardest is None or r["t_first"] > hardest[1]:
                        hardest = (rec["name"], r["t_first"], qa, qb)
                else:
                    rec["status"] = "failed"; print(f"[{rec['name']}] no path in {a.plan_time} s")
            plans.append(rec)
    R["moves"] = [{k: v for k, v in p.items() if k not in ("path", "raw_path", "q_start", "q_goal", "curve")}
                  for p in plans]

    # 5. benchmark RRT* vs Informed RRT* on the hardest pick
    if hardest is not None:
        name, _, q_start_b, q_goal = hardest
        bench = []
        for planner in ("RRTstar", "InformedRRTstar"):
            for s in range(a.bench_seeds):
                r = plan(chk, q_start_b, q_goal, planner, a.bench_time, 1.0, seed=100 + s)
                bench.append(dict(planner=planner, seed=s, t_first=r["t_first"], curve=r["curve"]))
                last = r["curve"][-1][1]
                print(f"[bench {name}] {planner:16s} seed {s}: first {r['t_first']}, final cost {last}")
        R["benchmark"] = dict(pick=name, time_s=a.bench_time, runs=bench)

    (out / "plans.json").write_text(json.dumps(dict(joint_names=JOINT_NAMES, plans=plans), indent=1))
    np.savez_compressed(out / "scene.npz", box_lo=np.array([b.lo for b in boxes]),
                        box_hi=np.array([b.hi for b in boxes]), box_kind=np.array([b.kind for b in boxes]),
                        box_name=np.array([b.name for b in boxes]), lidar_origin=lidar.origin,
                        points=pts, points_all=pts_all, ring=ring, occupied=occ, res=a.res)
    (out / "results_fcl_ompl.json").write_text(json.dumps(R, indent=1))
    print("[done] wrote", out)
    sys.stdout.flush()
    os._exit(0)  # skip nanobind leak warnings at interpreter shutdown


if __name__ == "__main__":
    main()
