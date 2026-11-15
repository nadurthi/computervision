# Arm motion planning through a lidar OctoMap: FCL, Informed RRT* and MoveIt 2

A Franka Panda picks cartons from a box wall. Its world model comes from a simulated 3D lidar:

```
simulated lidar ─▶ OctoMap (ray casting, 4 cm) ─▶ FCL: arm spheres vs octree
                                                   │
             per move: straight joint line collision-free? ── yes ─▶ execute it (no planning)
                                                   └─ no ─▶ OMPL Informed RRT* (path length)
same OctoMap ─▶ MoveIt 2 planning scene ─▶ MoveIt's OMPL plugin (RRT*, informed RRT*), FCL collision checking
```

Write-up with the videos: https://perceptionstack.dev/arm-motion-planning-octomap-fcl-informed-rrt-moveit/
Background survey: https://perceptionstack.dev/neural-informed-rrt-star-robot-arm-survey/

Everything runs in one Docker image (Ubuntu 24.04, ROS 2 Jazzy + MoveIt 2 from RoboStack, and the
`octomap-python`, `python-fcl` and `ompl` wheels). The same image and the same script run on your
machine and on AWS.

## Run locally

Needs Docker (about 7 GB of disk for the image) and nothing else.

```bash
git clone https://github.com/nadurthi/computervision
cd computervision/arm-motion-planning
./run_local.sh                 # build the image, run everything, write results/<UTC time>/
```

Stages (arguments after the stage go to the Python script):

```bash
./run_local.sh fcl --quick     # standalone pipeline only, short planning times (~1.5 min)
./run_local.sh fcl --plan-time 20 --bench-seeds 10 --res 0.03
./run_local.sh moveit          # MoveIt only (needs a previous fcl run in the same results folder: set RUN=...)
./run_local.sh render          # videos and figures only
./run_local.sh shell           # a shell inside the environment
```

Or by hand:

```bash
docker build -t arm-motion-planning .
docker run --rm -v "$PWD/results:/work/results" arm-motion-planning all
```

The first build downloads about 1.5 GB of packages and takes 5 to 10 minutes. A full run then takes about
12 minutes on 2 CPU cores, most of it the planner benchmark and MoveIt's 18 planning runs.
Behind a proxy, pass it to the build: `DOCKER_BUILD_ARGS="--build-arg HTTPS_PROXY=$HTTPS_PROXY" ./run_local.sh`.

## Run on AWS

`aws/launch_ec2.py` (boto3) starts a CPU instance (c7i.2xlarge, Ubuntu 24.04). The instance installs
Docker, clones this repo, runs **the same `./run_local.sh`**, copies `results/<run>/` to
`s3://arm-motion-planning-<account>-<region>/runs/<run>/` and shuts down. Shutdown terminates the
instance and deletes its disk, so S3 is the only thing left. A `shutdown` after `MAX_MIN` (90 min) is
scheduled first as a safety net.

```bash
pip install boto3                       # and AWS credentials (AWS_PROFILE etc.)
python aws/launch_ec2.py launch         # bucket, role, security group (no inbound) created if missing
python aws/launch_ec2.py status         # instance state and runs in S3
python aws/launch_ec2.py wait           # returns when the instance has terminated itself
python aws/launch_ec2.py fetch results  # download the newest run
python aws/launch_ec2.py terminate      # stop it early
python aws/launch_ec2.py cleanup        # terminate + delete role, instance profile, security group (keeps the bucket)
```

Settings: `AWS_REGION` (us-east-1), `INSTANCE_TYPE` (c7i.2xlarge), `MAX_MIN` (90), `SPOT=1`,
`RUN_ARGS` (default `all`, e.g. `'fcl --quick'`), `BRANCH` (master). No GPU is needed.

## What is in `results/<run>/`

| file | content |
|---|---|
| `video_fcl_ompl.mp4` | lidar scan, OctoMap, then each move: direct line or Informed RRT* tree and path |
| `video_moveit.mp4` | MoveIt's RRT* and informed RRT* trajectories in the same OctoMap |
| `cost_vs_time.png` | benchmark: best path length vs planning time, RRT* vs Informed RRT* |
| `scene.png` | lidar returns and occupied voxels |
| `results_fcl_ompl.json` | lidar, OctoMap, FCL timings; every move; benchmark curves |
| `results_moveit.json` | MoveIt success, path length, trajectory duration per run; cross-checks |
| `plans.json`, `moveit_plans.json` | joint-space paths |
| `octomap.bt` | the OctoMap (open with `octovis`) |

## Reference run (in this repo)

`docs/media/` has the two videos, `cost_vs_time.png` and `scene.png` from a run in this container;
`docs/results_*.json` hold all its numbers.

## Code

| file | |
|---|---|
| `sim/scene.py` | carton wall, conveyor, post, overhead beam; ray-cast multi-beam lidar with noise and dropout |
| `sim/robot.py` | Panda forward kinematics (Franka's modified DH), 60-sphere collision model + self-collision pairs, damped-least-squares IK |
| `sim/fit_spheres.py`, `sim/panda_spheres.json` | spheres fitted to the URDF collision meshes (k-means, every surface sample covered) |
| `sim/collision.py` | OctoMap from the scan; state checker: joint limits, floor, self-collision, FCL spheres vs `OcTree` |
| `sim/plan_fcl_ompl.py` | IK for each pick, direct-line check, OMPL Informed RRT*, RRT* vs Informed RRT* benchmark |
| `sim/plan_moveit.py` | MoveItPy: OctoMap into the planning scene, OMPL planning, FK and collision cross-checks |
| `sim/render.py` | videos and figures (matplotlib + ffmpeg) |
| `Dockerfile`, `environment.yml`, `entrypoint.sh` | the image |
| `run_local.sh` | build + run; the AWS instance runs this same file |
| `aws/launch_ec2.py` | EC2 runner, results to S3, self-terminating |

## Notes and limits

* The lidar is simulated by analytic ray casting against boxes (64 rings, 512 columns, 8 mm range noise,
  2 % dropout). Ground returns and the robot base column are removed before the OctoMap insert. The
  arm is not ray-cast, so no arm self-filter is needed here; a real cell needs one.
* Space behind the carton faces is *unknown* in the OctoMap and FCL only collides with *occupied*
  cells. The approach moves stay in front of the faces, so this does not matter here; for deeper
  reaches, treat unknown as occupied.
* The standalone checker uses 60 spheres fitted to the URDF collision meshes; MoveIt uses the meshes
  themselves. `plan_moveit.py` cross-checks the two on 3000 random states (in the reference run: 0
  collisions missed by the spheres, world or self; 758 states rejected that MoveIt accepts, partly
  because only the standalone checker keeps the arm above the floor) and checks every
  standalone path state in MoveIt's planning scene.
* MoveIt 2.12's OMPL plugin does not register OMPL's `InformedRRTstar` class. The `InformedRRTstarkConfigDefault`
  added here is `geometric::RRTstar` with tree pruning and informed *rejection* sampling: the direct
  informed sampler needs explicit goal states when it is created, which MoveIt's lazily sampled goal
  region does not provide.
* Self-collision in the standalone checker: sphere pairs for every link pair the Panda SRDF does not
  disable. Around the wrist the fitted spheres are fat, so link5 is represented by its forearm body
  only against link7 and the hand, and sphere pairs that overlap in more than 25 % of random
  configurations are dropped.
* At Python speed (about 155 µs per state check) the planners find a first path in 0.5 to 4 s, but
  neither RRT* nor Informed RRT* improves it much within 20 s in 7-D; shortcutting does most of the
  work. A vectorised checker (VAMP) or MoveIt's C++ loop is what makes the anytime part pay off.
* OMPL lets you seed its random generator once per process, so benchmark runs after the first are not
  individually reproducible.
