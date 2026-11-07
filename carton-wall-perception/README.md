# Carton-wall perception: classical vs. learned vs. foundation models

Detecting and segmenting stacked cardboard boxes for robotic truck and container unloading, with a camera
(RGB-D) and a lidar. Every method in this folder runs end to end on a CPU and is scored against exact ground
truth from a procedural box-wall simulator, so the comparisons are apples to apples.

**Full write-up:** [perceptionstack.dev — Unloading a box wall](https://perceptionstack.dev/tutorials/applications/carton-wall-perception/)
(results, annotated comparisons and a layer-by-layer architecture deep dive)
· **Architecture notes:** [`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md)
· **All numbers:** [`results/RESULTS.md`](results/RESULTS.md)

![Panoptic quality of every method](docs/figures/pq_bars.png)

## What is here

| Family | Methods | Code |
|---|---|---|
| Classical, camera | RGB straight lines · depth jumps + creases · depth + RGB seam valleys | `boxwall/classical.py` |
| Classical, lidar | range-image cuts · RANSAC planes + DBSCAN · camera-to-lidar label transfer | `boxwall/lidar.py` |
| Trained here | **SeamNet** (1.1 M-param RGB-D U-Net that segments seams) · YOLO11n-seg fine-tuned | `boxwall/seamnet.py`, `scripts/` |
| Zero-shot | FastSAM-s · MobileSAM · SAM 2.1-tiny, each with and without a depth-prior selector | `boxwall/zeroshot.py` |
| Shared | depth hole filling, normals, RANSAC plane fit, face rectangle, pick planner, metrics | `boxwall/geometry.py`, `boxwall/evaluate.py` |

### Code map

| File | What it does |
|---|---|
| `boxwall/scene.py` | procedural box walls: columns of cartons, SKUs, gaps, yaw/tilt, pushed-in and missing boxes |
| `boxwall/render.py` | vectorised ray caster: RGB, noisy depth with dropouts, normals, instance ids, scanning lidar |
| `boxwall/geometry.py` | depth hole filling, back-projection, normals, RANSAC plane fit, robust face rectangle |
| `boxwall/classical.py` | C1 RGB straight lines, C2 depth jumps + creases, C3 depth + RGB seam valleys |
| `boxwall/lidar.py` | L1 range-image cuts, L2 RANSAC planes + DBSCAN, L3 camera-to-lidar label transfer |
| `boxwall/seamnet.py` | SeamNet U-Net, 8-channel input encoding, seam targets, watershed decoding |
| `boxwall/zeroshot.py` | FastSAM / MobileSAM / SAM 2.1-tiny runners and the depth-prior selector |
| `boxwall/evaluate.py` | Hungarian matching, F1, panoptic quality, face-pose error |
| `scripts/make_dataset.py` | renders train / val / test / test_hard splits |
| `scripts/train_seamnet.py` | caches inputs and trains SeamNet (RGB-D or RGB) |
| `scripts/export_yolo.py` | converts walls to a YOLO-seg dataset |
| `scripts/run_eval.py` | runs any set of camera methods on a split, saves label maps and metrics |
| `scripts/run_lidar_eval.py` | lidar point-level evaluation inside the camera view |
| `scripts/run_real.py` | RGB-only methods on the real photos in `data/real/` |
| `scripts/benchmark_speed.py` | clean CPU timing of every method |
| `scripts/faint_seam_analysis.py`, `pose_common_subset.py`, `first30.py`, `recompute_metrics.py` | the extra analyses on the page |
| `scripts/make_figures.py`, `results_table.py`, `build_post.py` | figures, tables and the web page / teaser HTML |

## The simulator

`boxwall/scene.py` + `boxwall/render.py` build walls the way trailers are loaded: columns of cartons of random
size (25–62 cm wide, 18–50 cm tall), stacked from the floor, with a second row behind. The difficulty knobs:

- gaps of 0–12 mm between boxes (halved in the hard split);
- yaw up to ~±8°, roll ~±3°, pushed-in boxes, and missing front boxes that reveal the row behind;
- 2–6 SKUs per wall, so neighbouring boxes share print, tape and labels (the repeated-texture problem);
- random "seam darkness", so in some walls the seams between flush boxes are almost invisible;
- a point light on the robot with inverse-square falloff, so recessed boxes and gaps render darker (no cast shadows).

A vectorised ray caster renders RGB (480×360), depth with distance-dependent noise and dropouts at
discontinuities, normals, instance ids and per-box 6-DoF ground truth. The same scene is scanned by a
simulated 80-ring lidar mounted above the camera.

| Split | Scenes | Difficulty | Lidar |
|---|---:|---|---|
| train | 1,200 | standard | – |
| val | 80 | standard | – |
| test | 100 | standard | ✓ |
| test_hard | 100 | 1.8× (tighter gaps, more yaw/tilt) | ✓ |

![Example scene: RGB, depth, normals, labels, lidar](docs/figures/dataset_example.png)

## Results

Headline numbers: panoptic quality on 100 standard / 100 hard walls (SAM models scored on 30 each).

| Method | Standard | Hard |
|---|---:|---:|
| YOLO11n-seg fine-tuned on synthetic walls (RGB) | 0.92 | 0.91 |
| SeamNet U-Net, RGB-D (1.1 M params) | 0.88 | 0.87 |
| SeamNet U-Net, RGB | 0.87 | 0.85 |
| Classical: depth edges + RGB seam valleys | 0.67 | 0.62 |
| SAM 2.1-tiny, 192-point grid (zero-shot) | 0.64 | 0.51 |
| FastSAM-s + depth priors (zero-shot) | 0.45 | 0.38 |
| Classical: RGB straight lines | 0.21 | 0.11 |

On the lidar, lidar-only methods reach F1 0.09–0.16, against 0.92 for camera labels projected onto the
points. **On real photos the ranking flips**: the synthetic-trained networks fragment real boxes, while FastSAM
and SAM 2.1 segment them cleanly (see the last figure). The practical recipe is to label real data with SAM
and distil it into the small network.

See [`results/RESULTS.md`](results/RESULTS.md) for the full tables (F1 at IoU 0.5 and 0.75, panoptic quality,
face-normal / centre / size error against ground truth); clean CPU timings are in `results/speed_cpu.json`. Qualitative comparisons:

![Standard wall](docs/figures/qualitative_test.png)
![Hard wall](docs/figures/qualitative_hard.png)
![Lidar](docs/figures/lidar_example.png)
![Pick plan](docs/figures/pick_plan.png)

Sim-to-real check on real warehouse photos from the Stacked Carton Dataset (RGB only, no ground truth):

![Real photos](docs/figures/real_photos.png)

## Reproduce

```bash
pip install -r requirements.txt
# weights: yolo11n-seg.pt, FastSAM-s.pt, mobile_sam.pt, sam2.1_t.pt from github.com/ultralytics/assets releases
export WEIGHTS=/path/to/weights

python scripts/make_dataset.py                       # ~30 min on 2 cores, ~2 GB
python scripts/train_seamnet.py --cache
python scripts/train_seamnet.py --mode rgbd --epochs 6
python scripts/train_seamnet.py --mode rgb  --epochs 6
python scripts/export_yolo.py 700
yolo segment train model=$WEIGHTS/yolo11n-seg.pt data=data/yolo/boxes.yaml imgsz=480 epochs=10 batch=8 device=cpu
cp runs/segment/train/weights/best.pt results/yolo11n_seg_boxes.pt

for s in test test_hard; do
  python scripts/run_eval.py --split $s --methods rgb_lines depth_edges rgbd_seams seamnet_rgb seamnet_rgbd yolo fastsam_rgb fastsam_rgbd
  python scripts/run_eval.py --split $s --methods sam2t_rgb sam2t_rgbd --limit 30
done
python scripts/run_eval.py --split test --methods mobilesam_rgb mobilesam_rgbd --limit 30
python scripts/run_lidar_eval.py --split test
python scripts/run_real.py
python scripts/make_figures.py dataset bars qual lidar training picks real
python scripts/results_table.py
```

## Limits

- **Synthetic test data.** The walls are procedural and the boxes are perfect cuboids. Real cartons bulge, sag,
  are shrink-wrapped and carry glossy tape that breaks depth sensors. The numbers rank methods under controlled
  conditions; they are not field accuracy.
- **CPU budget.** SeamNet trained for 6 epochs and YOLO for 10, on 2 CPU cores. The SAM models were
  scored on the first 30 scenes of each split because each frame takes ~40 s on this CPU.
- **Sim-to-real** is checked only qualitatively, on six real photos without depth.

## Licences

The real example photos in `data/real/`, and the figure made from them, come from the
[Stacked Carton Dataset](https://github.com/yancie-yjr/scd.github.io) (Yang et al., *Sensors* 2022). That
dataset is licensed CC BY-NC-SA 4.0, and the photos are used here for non-commercial illustration with
attribution.

The YOLO and SAM runners use the Ultralytics package, which is AGPL-3.0. Check that before reusing this code in a
closed-source product; the classical methods and SeamNet do not depend on it. Model weights carry their
authors' licences (MobileSAM and SAM 2.1 are Apache-2.0).

## Isaac Sim scene (next step)

[`isaac_sim/box_wall_capture.py`](isaac_sim/box_wall_capture.py) builds a more realistic version of the scene in
NVIDIA Isaac Sim (4.5 / 5.x): open-source cardboard box USDs (Isaac `Simple_Warehouse` `SM_CardBox*`, or any folder of
USDs via `--box-usd-dir`) stacked against a wall inside NVIDIA's ready-made warehouse (`--env slabs` for a plain floor and wall), placeholder ceiling lights, an RGB-D camera and an
RTX lidar. Each capture writes RGB, depth, instance and semantic masks (carton / wall / floor), tight 2D boxes, the
lidar point cloud (plus per-point object ids where supported) and per-box ground-truth poses.

```bash
./python.sh isaac_sim/box_wall_capture.py --frames 5 --out box_wall_out          # headless
./python.sh isaac_sim/box_wall_capture.py --gui --settle                          # watch it, let boxes settle under gravity
```

### Run everything on your own machine

[`isaac_sim/run_all.sh`](isaac_sim/run_all.sh) runs all Isaac Sim scripts and builds an HTML report; it needs only
a Linux machine with an NVIDIA RTX GPU (driver 580 for Isaac Sim 5.1) and Docker with the NVIDIA Container Toolkit,
or a local Isaac Sim install (`MODE=native ISAAC_SIM_PATH=...`). The AWS scripts in `isaac_sim/aws/` call this same file.

```bash
bash isaac_sim/run_all.sh                       # -> ./isaac_runs/run-<time>/report.html
RUN_SCRIPTS=0 bash isaac_sim/run_all.sh         # smoke + lidar tests only
python3 isaac_sim/make_report.py <run folder>   # rebuild the report for any run folder
```

### Unloading cell (robot + conveyor)

[`isaac_sim/unload_cell.py`](isaac_sim/unload_cell.py): a UR10 with a suction gripper on a pedestal inside a trailer
unloads the box wall onto a conveyor that runs out into an Isaac warehouse. Joint targets come from Lula inverse
kinematics and are interpolated and sent every physics step from Python; the conveyor surface speed is a Python
number (`--belt-speed`, PhysX surface velocity on a kinematic belt, with a direct-velocity fallback).

Assets: NVIDIA SimReady cardboard boxes and conveyor ([`nvidia/simready-assets`](https://huggingface.co/datasets/nvidia/simready-assets),
CC-BY-4.0; fetch with [`isaac_sim/download_simready_assets.py`](isaac_sim/download_simready_assets.py)), the UR10 and
`Simple_Warehouse` from the Isaac assets library, and a procedural 53 ft dry-van interior (no open trailer USD found;
`--trailer-usd` takes your own).

```bash
python isaac_sim/download_simready_assets.py --out ~/simready
./python.sh isaac_sim/unload_cell.py --simready ~/simready --gui --belt-speed 0.4
./python.sh isaac_sim/unload_cell.py --simready ~/simready --picks 10 --record cell_out/   # headless, camera + lidar per pick
```
