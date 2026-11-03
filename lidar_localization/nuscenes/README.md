# LIDAR-only localization — NumPy port and nuScenes runs

Port of the method in *Scan-Matching Based Particle Filtering Approach for LIDAR-Only Localization*
([arXiv:2302.06843](https://arxiv.org/abs/2302.06843)) and of the original KITTI
code in [`../drivers/main_kitti_filter_localize.py`](../drivers/main_kitti_filter_localize.py). The original needs
numba, open3d, scikit-learn, PCL and the `pyslam` C++ module; this port needs only `numpy` and `scipy`.

| File | What it is | Mirrors |
|---|---|---|
| `lidarloc.py` | lookup-table likelihood, constant-turn-rate model, particle filter, road removal, multi-resolution BEV matcher, point-to-plane ICP | `pyslam` `computeLikelihood_lookup`, `dynmodelCT3D`, `uq.filters.pf`, open3d normals, `binmatchers.binMatcherAdaptive_super`, PCL ICP |
| `localize.py` | the per-frame loop: PF → RPE → RS → S2M → SU, plus scoring against ground truth | paper block diagram |
| `nusc_data.py` | nuScenes reader (no devkit): LIDAR_TOP sweeps, lidar→global poses, map building from other drives through the same area | — |
| `synthetic.py` | ray-cast 32-beam LIDAR in a synthetic city, for an end-to-end check without data | — |
| `check_against_original.py` | runs the original `binMatcherAdaptive_super` (numba bodies, no JIT, via `tools/stubs`) and the port on the same scene | — |

```bash
cd lidar_localization/nuscenes
python check_against_original.py tools/stubs     # original vs port: identical pose and costs
python -c "import synthetic, localize; f,M,_=synthetic.make_sequence(0,120); r,T=localize.run(f,M,dict(Npf=500)); print(localize.summarize(r))"
```

**Note on the earlier matcher.** `binMatcherAdaptive3` in the KITTI driver starts its translation boxes at offset
(0, 0) and only splits into non-negative offsets, so it can move the guess in +x/+y only. `binMatcherAdaptive_super`
(the one the driver calls) searches absolute positions within ±Lmax of the guess; the port follows `_super`.

## nuScenes v1.0-mini results

```bash
./get_nuscenes_mini.sh data/nuscenes                       # tables + LIDAR_TOP only (CC BY-NC-SA 4.0)
python run_nuscenes.py data/nuscenes results                # 10 scenes x {tracking, tracking_paper, global}
python make_video.py data/nuscenes results videos           # one MP4 per scene
```

Map = each scene's own 2 Hz keyframes with every annotated object cut out; the localized scans are the non-keyframe
sweeps (~10 Hz). The mini split has no second drive over the same streets, so this is a same-drive test, optimistic
compared with the paper's KITTI setup. Horizontal error after convergence, median / 95th percentile:

| scene | place | path | tracking | tracking, paper σ0=0.5, d_max=10 | global: time to < 1 m | global median |
|---|---|---|---|---|---|---|
| scene-0061 | Singapore One North | 91 m | 0.20 / 0.28 m | 0.20 / 0.37 m | 0.1 s | 0.19 m |
| scene-0103 | Boston Seaport | 117 m | 0.06 / 0.20 m | 0.11 / 0.28 m | 0.6 s | 0.06 m |
| scene-0553 | Boston Seaport (parked) | 0 m | 0.08 / 0.17 m | 0.28 / 0.38 m | 0.0 s | 0.08 m |
| scene-0655 | Boston Seaport | 162 m | 0.15 / 0.43 m | 0.31 / 0.66 m | 2.3 s | 0.16 m |
| scene-0757 | Boston Seaport | 27 m | 0.08 / 0.15 m | 0.12 / 0.20 m | 0.0 s | 0.09 m |
| scene-0796 | Singapore Queenstown | 235 m | 0.12 / 0.23 m | 0.16 / 0.33 m | 0.1 s | 0.11 m |
| scene-0916 | Singapore Queenstown | 93 m | 0.12 / 0.25 m | 0.15 / 0.30 m | 0.1 s | 0.11 m |
| scene-1077 | Singapore Holland Village, night | 251 m | 0.10 / 0.20 m | 0.16 / 0.30 m | 0.1 s | 0.08 m |
| scene-1094 | Singapore Holland Village, night | 126 m | 0.11 / 0.24 m | 0.14 / 0.30 m | 2.2 s | 0.12 m |
| scene-1100 | Singapore Holland Village, night | 1 m | 0.09 / 0.20 m | 0.16 / 0.30 m | 0.6 s | 0.08 m |

Global mode: 2000 particles uniform over the map with random heading; the matcher window widens to ±80 m / ±180°
while the cloud is spread out, and every 20 frames a coarse wide re-check replaces the best particle if it finds a pose
scoring 1.25x better. Without the re-check, scene-0655 (a long, nearly symmetric boulevard) locked onto a pose facing
the wrong way and never recovered. Summary: [`../results/nuscenes_mini_summary.json`](../results/nuscenes_mini_summary.json);
videos: [`../media/videos/`](../media/videos/).
