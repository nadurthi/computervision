# Tote depalletizing: lidar + camera, CAD point-to-mesh ICP, front-face centre

A pallet holds a 7-layer stack of identical grey Euro totes (2 × 2 per layer, 28 totes). A lidar and an RGB
camera, side by side on the arm/AMR 1.5 m in front of the stack, image its front. For every visible tote we
find its pose by registering the lidar points to the tote's CAD mesh, and report the centre of the tote's front
face (the face toward the sensor). The arm uses it as the grasp reference. Each pick removes a tote, so the
scene is rescanned until the pallet is empty.

The tote CAD here is a stand-in (600 × 400 × 300 mm tapered Euro container with rim flange and two ribs). To use
your own CAD, pass `--cad your_tote.obj|stl|ply` (mm or m; front = the −y face of the model) to the scripts.

## Layout

```
totes/cad.py          stand-in tote CAD (trimesh + manifold booleans), EUR pallet, CAD loader
totes/scene.py        7-layer stack (2x2 per layer, ±2 mm / ±0.8° placement jitter), pick order, ground truth
totes/lidar.py        CPU stand-in lidar: Ouster-OS0-128-like pattern ray-cast with Open3D, range noise, grazing dropout
totes/camera.py       CPU stand-in camera beside the lidar: pinhole render (RGB, z-depth, instance ids)
totes/detect.py       baseline only: seam-profile detector (layer seams + tote split), used to compare ICP variants
totes/icp.py          point-to-mesh ICP (sensor-facing triangles, Huber Gauss-Newton, 6 or 4 DOF), point-to-plane, Open3D baselines
totes/grid_icp.py     MAIN DETECTOR: point-to-mesh ICP from a grid of initial poses over the front-facing points,
                      ray-consistency (free-space) score, max-weight non-overlapping selection
totes/pipeline.py     detect -> ICP -> front-face centre, scoring against gt.json
scripts/make_cad.py             writes assets/ (tote OBJ/STL, pallet, tote_cad.json with the front-face centre/normal)
scripts/simulate_scans.py       lidar + camera captures for each pick state -> data/sim/scan_XX/
scripts/run_pipeline.py         baseline: seam detector + ICP variants (culled / unculled / Open3D) on a folder of scans
scripts/grid_icp_detect.py      MAIN: grid-seeded ICP (point-to-mesh or point-to-plane) on a folder of scans, figures
scripts/studies.py              noise sweep, convergence basin, full 28-pick depalletizing loop
scripts/export_isaac_assets.py  copies tote/pallet OBJ + stack layout to carton-wall-perception/isaac_sim/assets/totes
data/sim/scan_00..03            committed sample captures (see below)
```

Each scan folder (CPU simulator or Isaac Sim) holds the same files:

| file | content |
|---|---|
| `points.npy` | (N, 3) float32 lidar points, world frame, metres |
| `labels.npy` | (N,) int16 ground truth: tote id, −1 pallet, −2 floor |
| `rgb.png` | camera image, 1600 × 1200, vertical FOV 80° |
| `depth_mm.png` | uint16 z-depth along the optical axis, mm (0 = no hit) |
| `instances.png` | uint16 per-pixel tote id + 3 (0 background, 1 floor, 2 pallet) |
| `gt.json` | sensor position, camera `K` / position / `R_world_from_cam`, per tote: pose `T`, front-face centre and normal, present flag |

World frame: z up, the lidar at (0.1, 0, 1.2) looks along +x; the camera is 0.15 m to its right at
(0.1, −0.15, 1.2); the stack front is at x ≈ 1.6 m.

## Run (Docker, CPU)

```bash
docker build -t tote-depalletizing .
docker run --rm -v $PWD/data:/app/data -v $PWD/results:/app/results tote-depalletizing bash -c "
  python scripts/make_cad.py &&
  python scripts/simulate_scans.py --out data/sim --picks 4 &&
  python scripts/run_pipeline.py --scans data/sim --out results/sim &&
  python scripts/grid_icp_detect.py --scans data/sim --method p2mesh  --out results/grid_p2mesh &&
  python scripts/grid_icp_detect.py --scans data/sim --method p2plane --out results/grid_p2plane &&
  python scripts/studies.py --out results/studies"
```

Without Docker: `pip install -r requirements.txt` and run the same commands.

## Isaac Sim (RTX lidar + camera)

`carton-wall-perception/isaac_sim/tote_pallet_capture.py` builds the same pallet and stack in Isaac Sim 5.1. It
places an RTX lidar (an OS0 128-channel config when available) and a camera with the same intrinsics. Then it
writes the same scan folders for each pick state. It runs in the NVIDIA container through `run_all.sh`:

```bash
cd ../carton-wall-perception/isaac_sim
python ../../tote-depalletizing/scripts/export_isaac_assets.py      # once, or after changing the CAD
STEPS="smoke tote_pallet tote_icp" TOTE_PICKS=6 bash run_all.sh       # local RTX GPU
# AWS g6 (L4), result synced to S3, instance terminates itself:
STEPS="smoke tote_pallet tote_icp" python ../aws/launch_isaac_ec2.py launch
```

`tote_icp` builds this folder's Dockerfile on the same machine and runs `run_pipeline.py` and
`grid_icp_detect.py` on `/out/tote_pallet`.

## Method in one paragraph

Points `p_i` are moved into the tote frame, `q_i = Tᵀ p_i`, and each is matched to its closest point `m_i` on the
**sensor-facing** triangles of the CAD mesh. The residual is `r_i = d_iᵀ (q_i − m_i)`, where `d_i` is the triangle
normal inside a face and the unit offset at an edge or vertex. A small motion of the points `q ← q + ω×q + t`
gives `J_i = [(q_i × d_i)ᵀ, d_iᵀ]`. Each Gauss-Newton step solves `(JᵀWJ) ξ = −JᵀW r` with Huber weights
`W` (scale from the MAD) and a correspondence gate that shrinks from 5 cm to 1.5 cm. For totes resting flat the
solve keeps only yaw and translation (4 DOF). The front-face centre is `T c_front`.

Culling back-facing triangles matters: the tote wall is 4 mm thick, and without culling the fit snaps to the
inner wall.

The grid detector seeds this ICP on a lattice over the stack face (15 cm × 7.5 cm). It scores every converged
pose by ray consistency: a beam that passes through a surface the model says is there counts as a violation.
It then keeps the maximum-weight set of non-overlapping poses. Point-to-surface distance alone cannot reject a
pose that straddles two neighbouring totes, because the faces are flat and coplanar and this CAD's ribs repeat
with the tote pitch. The beams through the V-gap between two tapered totes can reject it.

## Results on the CPU simulator (σ = 5 mm range noise; full numbers in `results/`)

Front-face centre error, RMS in mm along the face normal / lateral / vertical. Totes are found by the seam
detector (scans 00–02, 42 totes):

| refinement | normal | lateral | vertical | p95 \|e\| |
|---|---|---|---|---|
| point-to-mesh, culled (ours) | 0.55 | 2.38 | 2.37 | 5.97 |
| point-to-mesh, no culling | 2.63 | 3.48 | 2.78 | 6.89 |
| Open3D point-to-plane | 3.10 | 5.34 | 3.68 | 8.87 |
| Open3D point-to-point | 3.40 | 8.28 | 4.63 | 11.88 |
| detector only (no ICP) | 1.45 | 9.48 | 5.30 | 13.55 |

Grid-seeded ICP, scans 00–03 (55 visible totes, including back-row totes exposed by picks):

| residual | found / visible | duplicates | straddling poses | RMS world x / y / z (mm) |
|---|---|---|---|---|
| point-to-mesh | 55 / 55 | 0 | 0 | 0.3 / 1.0 / 3.4 (max 15 mm: back-row tote, top 11 cm visible) |
| point-to-plane | 55 / 55 | 1 | 2 | straddles are 0.2–0.3 m off; otherwise 0.6 / 0.9 / 1.4–7 |

## Isaac Sim results (RTX lidar OS0-128 + camera, g6.2xlarge, run-20261117T040645Z)

Six pick states (0-5 totes removed), grid-seeded point-to-mesh ICP (4 DOF) with the ray-consistency score:
**79 of 79 visible totes found, no duplicates, no false detections.** Front-face centre error, RMS world
x / y / z = 0.49 / 1.21 / 5.7 mm (mean z bias -1.7 mm), p95 |e| 7.8 mm, max 33.6 mm (a back-row tote seen only
over the front row). Per-scan JSON and figures: `results/isaac_grid_p2mesh/`; two of the scans: `data/isaac/`.

What the Isaac runs needed (all in `tote_pallet_capture.py`):
* lidar for all pick states first, camera second (a replicator camera step left the rotary lidar returning
  partial or empty sweeps);
* captures at sensor yaw 0/90/180/270 merged, because the per-frame slices left a fixed azimuth gap
  (+5..+55 deg) on half of the stack;
* the Ouster optical-origin offset (36.18 mm above the mount): without it the floor came back at z = -0.036 m;
* inverse-density point weights in the grid score (overlapping sweeps scan some strips several times).
