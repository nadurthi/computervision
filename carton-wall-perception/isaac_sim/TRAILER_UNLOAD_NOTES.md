# trailer_unload.py: issues, failure cases and fixes

Engineering log for the trailer-unloading simulation (`trailer_unload.py`, Isaac Sim 5.1, AWS g6/g5). It records
what went wrong, how each failure was diagnosed, what fixed it, and what is still open.

**Current setup (works):**
- KUKA KR210 R2700 (L150) on an AMR (deck 0.25 m), standing 2.4 m from the wall face.
- Suction tool with extendable left/right/bottom stoppers.
- On-board conveyor plus a fixed boom conveyor out of the dock door.

**Last good run** (run 8, `runs/run-20261114T090726Z/`):
- The AMR drives in 1.0 m.
- 6 picks, 5 boxes carried out of the trailer.
- Neighbouring boxes are dragged at most 1.6 cm towards the robot.
- The AMR backs out 0.8 m.
- The collision test stops the arm on contact with the side rail.

Related files:
- `arm_kin.py`: Isaac-free kinematics and the pick waypoints, shared by the simulation and the checker.
- `tools/arm_clearance.py`: offline arm/tool/box clearance check against the trailer and conveyor, on the robot's
  real collision meshes (pxr + numpy only).

---

## 1. How to debug this simulation (lessons)

1. **Check kinematics offline before spending a GPU run.** Every KUKA failure from run 1 to run 6 was a geometry
   problem: roof, posts, belt. `tools/arm_clearance.py` reproduces those failures in about 10 s on a laptop, while an
   AWS run takes about 20 min.
   ```
   curl -O https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Kuka/KR210_L150/kr210_l150.usd
   python tools/arm_clearance.py --usd kr210_l150.usd --standoff 2.4 --pedestal 0.25 --reach 2.6 \
       --roof-margin 0.25 --min-release 0.27 --amr --picks 40
   ```
2. **Log per motion leg, not per pick.** The `    leg ...` lines record tool tip and axis, the six joint angles, and
   the held box's position and tilt. Before they existed, failures could only be guessed from frames. After, the
   cause was usually visible in one line (for example `tilt 25 deg`, or `axis [0.8, 0.56, ...]`).
3. **Use the real asset geometry.** Assumed KR210 link lengths (1.15 / 1.22 m) were wrong; the L150 is
   1.25 / 1.50 m. That one mistake caused three failed runs.
4. **Make the checker model everything the arm can hit.** First it missed the suction cup and the held box (so a
   wrong box model hid a problem), then the conveyor slab (so it could not see run 6's failure). If the checker
   says "clear" but the simulation collides, something is missing from the checker.

---

## 2. Failure cases, root causes and fixes

### FANUC CRX-10iA/L phase (runs 1–11, before the switch to KUKA)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| F1 | Moves time out near the target | Position tolerance too tight for a box-loaded arm | 1 cm tolerance; 3 cm accepted at timeout |
| F2 | Cup passes through boxes | Suction cup had no collider | Cylinder collider on the cup |
| F3 | Boxes pulled from under others | Unreachable boxes were marked "done", so the boxes under them looked free | Unreachable boxes keep blocking (`free_boxes`) |
| F4 | Carry path sweeps protruding columns | Box carried sideways while still inside the wall envelope | Pull clear of the wall first |
| F5 | Large drag of the box below | Rigid grip pressed down while sliding | Compliant D6 grip (free only in world z ±3 cm) for the slide, then rigid |
| F6 | Top boxes out of reach | CRX reach 1.418 m | 1.30 m pedestal, 0.62 m standoff |
| F7 | `link_4` hits the roof | Suction point too high | Suction point at least 22 cm below the roof |
| F8 | Belt drags wall boxes off | A moving belt drags anything touching it, including its end face | Belt ends 30 cm short; static low-friction transfer plate in the gap |
| F9 | Arm jams when placing with a rotated wrist | Wrist singularity / joint limits | Place straight ahead onto the transfer plate |
| F10 | Box under the first pick "explodes"; placements stall above the belt | Drag up to 13.5 cm, then the solver went unstable; CRX reach is the root limit | Switched to KUKA, added stoppers (below) |

### KUKA KR210 phase (runs 1–8, 2026-11-14)

| Run | Symptom (log) | Root cause | Fix |
|---|---|---|---|
| K1 | `link_3 touched /World/Trailer/Roof` on pick 2 | Short top box forced the suction point to 2.42 m; the forearm housing sits about 0.27 m above its axis | Tool pitched up to 25° for high suction points; pull only clear of the box's own column, then lower |
| K2 | `tool0 touched Roof` after pick 2 | Tool levelled (25° → 0°) faster than it descended, so the flange rose | Lower first while still pitched, level afterwards |
| K3 | `link_3 touched Roof` while carrying | Wrong link lengths assumed (1.15 / 1.22 m; real L150 1.25 / 1.50 m). On a 0.3 m plinth 1.85 m from the wall the arm folds tightly to reach the belt and raises the A3 housing | Downloaded the real USD, wrote the offline checker, swept placements: robot on the floor 2.2 m from the wall (≥ 19.9 cm roof clearance over 25 picks) |
| K4 | `tool0 touched TrimL/Post23`; frames show the wrist twisted, tool pointing at the side wall | Unknown at the time: frames alone did not explain it | Added per-leg pose/joint/box logging; full-orientation control (roll held, not only the tool axis) |
| K5 | Same post hit; log: box `tilt 25 deg` after the `clear` leg | Rigid grip made at 25° pitch, so levelling the tool tilted the box 25°. Its low edge hit the belt/plate when placing and shoved the arm 30 cm sideways | Keep the pick pitch until release, so the box stays level |
| K6 | Same post hit; box level until `place`, then tilt 27° | Forearm (`link_4`) dipped about 2 cm into the belt slab. Reaching sideways and down over the belt puts the forearm 15–30 cm below the tip, and the kinematic belt pushed the arm | Checker models belt and plate; tool level (no pitch); tool tip at least 27 cm above the belt at release, so forearm clearance is at least 8 cm; boxes low behind the conveyor end are skipped |
| K7 | Success: 4 picks, 3 out, collision test OK | – | – |
| K8 (AMR) | Success: drive in 1.0 m, 6 picks, 5 out, back out 0.8 m, collision test OK | – | AMR added (see §3) |

Found during the AMR work, all in the offline checker:

| Issue | Root cause | Fix |
|---|---|---|
| Wall boxes overlapped the side posts by 3 cm | `plan_packed_wall` packed to `W − 0.03`, but posts and rails stick 5 cm into the box each side | Pack to `W − 0.12` (inside the posts, 1 cm play) |
| On a 0.25 m deck, arm hits the roof at a 2.0–2.2 m standoff | Higher shoulder | Standoff 2.4 m (24 picks clean) |
| Short top-row boxes (suction at 2.54–2.58 m) block their whole column | A level tool can't grip that high, and unreachable boxes still block the boxes beneath them | Pitch up to 25° only for suction points above the 25 cm roof margin (`roof_ok(..., pitch=True)`, `pitch_from = H − roof`); the higher release keeps the pitched wrist off the belt |

### Infrastructure issues

| Issue | Fix |
|---|---|
| Runs slow to start (image pull plus shader/asset cache) | S3 cache of the container image and data cache (`aws/s3_cache.sh`); restore takes about 2.5 min |
| `InsufficientInstanceCapacity` for g6.2xlarge in every zone | Fallback chain g6 on-demand → g6 spot → g5 on-demand → g5 spot, across all zones |
| `VcpuLimitExceeded` while the previous instance is still shutting down | Use spot, or wait; the on-demand G quota is 8 vCPU (one instance) |
| Leftover cost | Every instance uses `InstanceInitiatedShutdownBehavior=terminate`, `DeleteOnTermination` volumes, an idle watchdog and a 3 h `shutdown` safety net; only S3 persists. After each run, check that there are no instances, volumes, snapshots, EIPs or open spot requests |
| No `report.html` | Host Python lacked numpy/PIL; falls back to Isaac's Python |

---

## 3. Design choices that matter

- **Stoppers.** Three kinematic plates (left, right, bottom) are placed 3 mm in front of the boxes beside and under
  the chosen one, and glide in over 25 steps. Kinematic bodies are immovable, so neighbours cannot be dragged out.
  The plates are created once at build time and parked below the floor (no prims added mid-simulation). Measured
  drag of adjacent boxes towards the robot: at most 1.6 cm, about 0 cm on most picks, against 2–14 cm without
  stoppers.
- **Start pose.** The KR210 zero pose puts the tool about 6 cm inside the wall. A forward-kinematics grid search on
  the asset's USD joints (no physics) picks a collision-free start; it is checked against the measured end-effector
  position (0.0 cm error).
- **AMR.** The chassis, on-board belt and transfer plate are kinematic and move by their translate ops. The robot
  root is teleported in small steps (`set_world_pose`). A fixed boom conveyor runs 2 cm lower under the on-board
  belt's rear end, long enough for the whole travel, so boxes always transfer onto it. The AMR drives forward only
  when no free box is in reach, and only as far as keeps its plate and chassis clear of every box still in the
  wall.
- **Same waypoints in the simulation and the checker** (`arm_kin.pick_waypoints`), so an offline "clear" means the
  simulation follows that path.

---

## 4. Open limitations and suggested fixes

| Limitation | Effect | Suggested fix |
|---|---|---|
| Low boxes behind the on-board conveyor end (in the belt's width, below about 1.05 m) can't be reached without passing through the belt | They are skipped, and they also stop the AMR advancing. Offline, the sequence stops at 37 of 71 boxes after one 0.18 m advance | Retractable or telescoping nose conveyor that pulls back while those boxes are picked; or a low (floor-level) nose with a lift; or an AMR that also moves laterally |
| Release height | A short box gripped near its bottom edge drops up to about 20 cm onto the belt (needed for forearm clearance) | Lower on-board belt near the robot, or the robot placing onto an inclined chute |
| Tool pitch for short top-row boxes | Up to 25°; the box stays level only because the pitch is held until release | A suction gripper with a passive wrist or tilt joint, or a top-face (vertical) grip for those boxes |
| Fixed-base reach | Back-row boxes deeper than about 2.6 m from the shoulder wait for the AMR to advance | Already handled by the AMR once front boxes clear; see the first row |
| Camera | Sits behind the robot, so the arm hides the stoppers | Side camera near the wall, or a second camera |
| Only the arm, cup and held box are checked offline | Box-vs-wall rubbing and box-vs-box contacts are physics-only | Add the remaining wall boxes as obstacles in `tools/arm_clearance.py` |
| Robot teleported with the AMR | No wheel or drive dynamics | Model the AMR as a wheeled articulation if the drive itself matters |
