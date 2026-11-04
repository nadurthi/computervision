# Datasets for stereo and calibrated multi-view depth

Everything below has known intrinsics and extrinsics. "Used" marks what `setup/download_datasets.sh` fetches
(quick / full profile). Sizes are approximate.

## Two-view stereo (rectified pairs, GT disparity)

| Dataset | What | GT / occlusion info | Size | Access | Used |
|---|---|---|---|---|---|
| Middlebury 2014 – MiddEval3 | 15 training pairs, indoor, structured-light GT; H = half resolution | dense disparity + `mask0nocc` (non-occluded / occluded) | 0.5 GB (H) | direct | quick, full |
| Middlebury 2014 scenes | 23 full-res scenes, "perfect" calibration, GT for **both** views | dense, occlusion from left-right check | ~3 GB | direct | full |
| Middlebury 2021 | 24 mobile-robot scenes | dense | ~3 GB | direct | – |
| ETH3D low-res two-view | 27 grayscale pairs, indoor + outdoor, laser-scan GT | sparse-dense + `mask0nocc` | 0.1 GB | direct | quick, full |
| KITTI 2015 stereo | 200 driving pairs, LiDAR + CAD car models | `disp_noc` vs `disp_occ` | 2 GB | direct (S3) | quick (40), full |
| KITTI 2012 stereo | 194 pairs, static scenes | `disp_noc` vs `disp_occ` | 2 GB | registration | – |
| Scene Flow (FlyingThings3D, Driving, Monkaa) | synthetic, the standard *training* set | dense, both views | 1 TB+ | direct / torrent | – (training only) |
| Booster | indoor, specular / transparent surfaces | dense | 30 GB | registration | – |
| DrivingStereo | 180k driving pairs | LiDAR | 100 GB+ | Google Drive | – |
| FoundationStereo FSD | 1M synthetic pairs | dense | >1 TB | Google Drive | – |

## Multi-view stereo (calibrated images, GT depth per view or GT 3-D)

| Dataset | What | GT | Size | Access | Used |
|---|---|---|---|---|---|
| DTU MVS 2014 | 124 tabletop objects, 49 robot-arm views, 7 lights, 1600x1200 | structured-light point clouds; per-view depth maps rendered from them (MVSNet / CasMVSNet) | 6.3 GB points + 6.3 GB masks; images 123 GB official, ~3 GB preprocessed test set | direct + Google Drive | quick (8 scans), full (22 test scans, point clouds) |
| ETH3D high-res multi-view | 13 training scenes (indoor/outdoor), DSLR 6048x4032 | laser-scan depth per image + occlusion maps | ~1-3 GB per scene | direct | quick (6), full (13) |
| BlendedMVS | 113 scenes, 768x576, rendered from textured meshes | rendered depth for every view | 27 GB (low-res) | Google Drive | full |
| Tanks and Temples | large indoor/outdoor scenes | laser point clouds (training set public; test via submission) | 10+ GB | direct; depth maps via robustmvd | – |
| ETH3D low-res many-view | 10 video sequences, 4 grayscale rig cameras | laser scan | small | direct | – |
| ScanNet / ScanNet++ | indoor RGB-D video, known poses | sensor depth / laser | TB | ToS agreement | – |
| KITTI raw + depth completion | driving, multi-frame | accumulated LiDAR | 180 GB | registration | – |
| Replica, TartanAir, Virtual KITTI 2 | synthetic | dense | large | direct | – |
| Robust MVD benchmark (lmb-freiburg/robustmvd) | zero-shot protocol over KITTI, ScanNet, ETH3D, DTU, T&T | depth per key view | - | scripts | (MVSAnywhere, Vis-MVSNet, RobustMVD are run through its wrappers) |

## Why per-view GT depth matters here

The occlusion analysis needs, for every reference pixel, whether the same 3-D point is visible in each source view.
That is decided by projecting the reference GT point into the source and comparing its depth with the source's own
GT depth (`mvsbench/geometry.py: visibility_in_source`). DTU (rendered depth maps), ETH3D (laser depth per image)
and BlendedMVS (rendered depth) all provide GT depth for every view, so the labels are exact up to GT holes.

## Licences

DTU, ETH3D, Middlebury and KITTI are free for research with citation; KITTI and BlendedMVS are CC BY-NC-SA.
Nothing in this repository redistributes data: the scripts download from the original hosts and mirror archives only
to your own private S3 bucket.
