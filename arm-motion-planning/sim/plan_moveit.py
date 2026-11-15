"""MoveIt 2 on the same scene: the lidar OctoMap goes into the planning scene, and
MoveIt's OMPL plugin (collision checking with FCL) plans the same transit moves.

MoveIt 2.12 does not register OMPL's InformedRRTstar class by name. OMPL's RRTstar
becomes Informed RRT* through parameters, so a planner config
"InformedRRTstarkConfigDefault" of type geometric::RRTstar is added with tree pruning
and informed *rejection* sampling. The direct informed sampler (what the
InformedRRTstar class uses) needs explicit goal states when it is created, and
MoveIt's goal is a lazily sampled goal region, so OMPL throws; the rejection sampler
only needs the cost-to-go heuristic, which MoveIt's goal provides.

Needs results/plans.json and results/octomap_raw.bin from plan_fcl_ompl.py.
Writes results/moveit_plans.json and results/results_moveit.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from robot import fk_frames, JOINT_NAMES, LOWER, UPPER  # noqa: E402
from collision import ArmChecker  # noqa: E402

import rclpy  # noqa: E402
from moveit.planning import MoveItPy, PlanRequestParameters  # noqa: E402
from moveit.core.robot_state import RobotState  # noqa: E402
from moveit_configs_utils import MoveItConfigsBuilder  # noqa: E402
from octomap_msgs.msg import Octomap  # noqa: E402

HERE = Path(__file__).resolve().parent
PLANNERS = {
    "RRTstar": "RRTstarkConfigDefault",
    "InformedRRTstar": "InformedRRTstarkConfigDefault",
}


def moveit_config(res):
    cfg = (MoveItConfigsBuilder(robot_name="panda", package_name="moveit_resources_panda_moveit_config")
           .robot_description(file_path="config/panda.urdf.xacro")
           .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
           .planning_pipelines(pipelines=["ompl"])
           .moveit_cpp(file_path=str(HERE / "config" / "moveit_py.yaml"))
           .to_moveit_configs()).to_dict()
    cfg["octomap_resolution"] = float(res)  # the occupancy map monitor reads this
    ompl = cfg["ompl"]
    ompl["planner_configs"]["InformedRRTstarkConfigDefault"] = dict(
        type="geometric::RRTstar", range=1.2, goal_bias=0.2, informed_sampling=0, sample_rejection=1,
        tree_pruning=1, new_state_rejection=0, use_admissible_heuristic=1)
    ompl["planner_configs"]["RRTstarkConfigDefault"].update(range=1.2, goal_bias=0.2)
    # check motions at the same resolution as the standalone planner (fraction of the joint-space extent)
    ompl["panda_arm"]["longest_valid_segment_fraction"] = 0.002
    grp = ompl["panda_arm"]["planner_configs"]
    if "InformedRRTstarkConfigDefault" not in grp:
        grp.append("InformedRRTstarkConfigDefault")
    return cfg


def joint_length(Q):
    Q = np.asarray(Q)
    return float(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)))


def state(model, q):
    rs = RobotState(model)
    rs.set_to_default_values()
    rs.set_joint_group_positions("panda_arm", np.asarray(q, float))
    rs.update()
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--plan-time", type=float, default=10.0)
    ap.add_argument("--runs", type=int, default=3, help="runs per planner and pick")
    ap.add_argument("--agree-n", type=int, default=3000, help="random states for the checker cross-check")
    a = ap.parse_args()
    out = Path(a.out)
    plans = json.loads((out / "plans.json").read_text())["plans"]
    raw = (out / "octomap_raw.bin").read_bytes()
    res = json.loads((out / "results_fcl_ompl.json").read_text())["octomap"]["resolution"]

    rclpy.init()
    t0 = time.perf_counter()
    moveit = MoveItPy(node_name="moveit_py", config_dict=moveit_config(res))
    t_init = time.perf_counter() - t0
    model = moveit.get_robot_model()
    psm = moveit.get_planning_scene_monitor()
    arm = moveit.get_planning_component("panda_arm")

    # random states, robot only (before the OctoMap goes in): MoveIt's self-collision verdict
    Qr = np.random.default_rng(7).uniform(LOWER, UPPER, (a.agree_n, 7))
    with psm.read_only() as scene:
        self_coll = np.array([scene.is_state_colliding(state(model, q), "panda_arm", False) for q in Qr])

    msg = Octomap()
    msg.header.frame_id = model.model_frame
    msg.binary, msg.id, msg.resolution = True, "OcTree", float(res)
    msg.data = np.frombuffer(raw, np.int8).tolist()
    with psm.read_write() as scene:
        scene.process_octomap(msg)
        scene.current_state.update()
    R = dict(init_s=round(t_init, 2), octomap_frame=model.model_frame, resolution=res, picks=[])

    # agreement between the standalone checker (spheres + FCL octree + floor) and MoveIt (meshes + FCL)
    chk = ArmChecker(raw, res)
    mine_free = np.array([chk.valid(q) for q in Qr])
    with psm.read_only() as scene:
        mv_coll = np.array([scene.is_state_colliding(state(model, q), "panda_arm", False) for q in Qr])
    world_coll = mv_coll & ~self_coll
    R["checker_agreement"] = dict(
        states=int(a.agree_n),
        moveit_collision=int(mv_coll.sum()), moveit_self_collision=int(self_coll.sum()),
        standalone_invalid=int((~mine_free).sum()),
        standalone_free_but_moveit_world_collision=int((mine_free & world_coll).sum()),
        standalone_free_but_moveit_self_collision=int((mine_free & self_coll).sum()),
        standalone_invalid_but_moveit_free=int((~mine_free & ~mv_coll).sum()))
    print("[moveit] checker agreement:", R["checker_agreement"])

    # forward-kinematics agreement between robot.py and MoveIt's URDF model
    errs = []
    for q in [np.zeros(7), np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])] + \
             [np.array(p["q_goal"]) for p in plans if "q_goal" in p]:
        T = state(model, q).get_global_link_transform("panda_hand")
        errs.append(float(np.linalg.norm(np.asarray(T)[:3, 3] - fk_frames(q)[8][:3, 3])))
    R["fk_max_error_m"] = max(errs)
    print(f"[moveit] up in {t_init:.1f} s; FK max difference to robot.py: {max(errs) * 1000:.3f} mm")

    out_plans = []
    for p in plans:
        if p.get("status") != "planned":
            continue
        qs, qg = np.array(p["q_start"]), np.array(p["q_goal"])
        with psm.read_only() as scene:
            # does MoveIt (link meshes + FCL) accept the standalone path?
            dense = []
            P = np.array(p["path"])
            for i in range(len(P) - 1):
                n = int(np.ceil(np.max(np.abs(P[i + 1] - P[i])) / 0.02)) + 1
                dense += [P[i] + t * (P[i + 1] - P[i]) for t in np.linspace(0, 1, n)]
            coll = sum(scene.is_state_colliding(state(model, q), "panda_arm", False) for q in dense)
            line = [qs + t * (qg - qs) for t in np.linspace(0, 1, 200)]
            line_coll = sum(scene.is_state_colliding(state(model, q), "panda_arm", False) for q in line)
        rec = dict(name=p["name"], standalone_path_states_in_collision_moveit=int(coll),
                   standalone_path_states_checked=len(dense),
                   straight_line_states_in_collision_moveit=int(line_coll), runs=[])
        for label, pid in PLANNERS.items():
            for k in range(a.runs):
                arm.set_start_state(robot_state=state(model, qs))
                arm.set_goal_state(robot_state=state(model, qg))
                prm = PlanRequestParameters(moveit, "")
                prm.planner_id, prm.planning_pipeline = pid, "ompl"
                prm.planning_time, prm.planning_attempts = a.plan_time, 1
                t0 = time.perf_counter()
                result = arm.plan(single_plan_parameters=prm)
                dt = time.perf_counter() - t0
                run = dict(planner=label, planner_id=pid, run=k, wall_s=round(dt, 2), success=bool(result))
                if result:
                    traj = result.trajectory
                    m = traj.get_robot_trajectory_msg().joint_trajectory
                    idx = [m.joint_names.index(j) for j in JOINT_NAMES]
                    Q = np.array([[pt.positions[i] for i in idx] for pt in m.points])
                    run.update(waypoints=len(Q), joint_length=joint_length(Q),
                               duration_s=float(traj.duration))
                    if k == 0:
                        out_plans.append(dict(name=p["name"], planner=label, path=Q.tolist(),
                                              time_from_start=[pt.time_from_start.sec + 1e-9 * pt.time_from_start.nanosec
                                                               for pt in m.points]))
                print(f"[moveit {p['name']}] {label:16s} run {k}: success={run['success']} "
                      f"length={run.get('joint_length', float('nan')):.2f} rad wall={dt:.1f} s")
                rec["runs"].append(run)
        R["picks"].append(rec)

    (out / "moveit_plans.json").write_text(json.dumps(dict(joint_names=JOINT_NAMES, plans=out_plans), indent=1))
    (out / "results_moveit.json").write_text(json.dumps(R, indent=1))
    print("[done] MoveIt results in", out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
