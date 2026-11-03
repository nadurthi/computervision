### Standard walls (test, 100 scenes)

| Method | F1@0.5 | F1@0.75 | PQ | normal err (deg) | centre err (cm) | size err (cm) | walls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical: RGB straight lines | 0.29 | 0.15 | 0.21 | 0.30 | 3.4 | 4.2 | 100 |
| Classical: depth jumps + creases | 0.74 | 0.60 | 0.63 | 0.34 | 0.7 | 1.5 | 100 |
| Classical: depth + RGB seam valleys | 0.77 | 0.66 | 0.67 | 0.31 | 0.5 | 1.4 | 100 |
| SeamNet U-Net (RGB) | 0.96 | 0.92 | 0.87 | 0.32 | 0.6 | 1.5 | 100 |
| SeamNet U-Net (RGB-D) | 0.97 | 0.94 | 0.88 | 0.31 | 0.7 | 1.6 | 100 |
| YOLO11n-seg fine-tuned (RGB) | 0.98 | 0.97 | 0.92 | 0.30 | 0.6 | 1.7 | 100 |
| FastSAM-s (RGB) | 0.48 | 0.37 | 0.40 | 0.33 | 0.6 | 1.7 | 100 |
| FastSAM-s + depth priors | 0.52 | 0.42 | 0.45 | 0.28 | 0.5 | 1.5 | 100 |
| MobileSAM grid (RGB) | 0.56 | 0.40 | 0.46 | 0.34 | 0.5 | 1.5 | 30 |
| MobileSAM + depth priors | 0.60 | 0.46 | 0.50 | 0.31 | 0.4 | 1.4 | 30 |
| SAM 2.1-tiny grid (RGB) | 0.74 | 0.62 | 0.64 | 0.33 | 0.5 | 1.4 | 30 |
| SAM 2.1-tiny + depth priors | 0.72 | 0.63 | 0.63 | 0.29 | 0.5 | 1.4 | 30 |

### Hard walls (test_hard, 100 scenes: gaps ~half as wide, more yaw and tilt)

| Method | F1@0.5 | F1@0.75 | PQ | normal err (deg) | centre err (cm) | size err (cm) | walls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classical: RGB straight lines | 0.15 | 0.06 | 0.11 | 0.46 | 4.4 | 5.7 | 100 |
| Classical: depth jumps + creases | 0.71 | 0.57 | 0.61 | 0.39 | 0.9 | 1.7 | 100 |
| Classical: depth + RGB seam valleys | 0.72 | 0.59 | 0.62 | 0.38 | 0.8 | 1.6 | 100 |
| SeamNet U-Net (RGB) | 0.95 | 0.89 | 0.85 | 0.39 | 0.7 | 1.7 | 100 |
| SeamNet U-Net (RGB-D) | 0.97 | 0.92 | 0.87 | 0.38 | 0.8 | 1.8 | 100 |
| YOLO11n-seg fine-tuned (RGB) | 0.98 | 0.95 | 0.91 | 0.37 | 0.7 | 1.9 | 100 |
| FastSAM-s (RGB) | 0.41 | 0.29 | 0.34 | 0.44 | 0.9 | 2.1 | 100 |
| FastSAM-s + depth priors | 0.45 | 0.34 | 0.38 | 0.34 | 0.6 | 1.7 | 100 |
| SAM 2.1-tiny grid (RGB) | 0.60 | 0.48 | 0.51 | 0.38 | 0.8 | 1.9 | 30 |
| SAM 2.1-tiny + depth priors | 0.58 | 0.49 | 0.51 | 0.35 | 0.7 | 1.8 | 30 |

### Lidar (point-level instances inside the camera view, 100 scans)

| Method | precision | recall | F1@0.5 | PQ | s/scan |
|---|---:|---:|---:|---:|---:|
| Lidar only: range-image cuts | 0.37 | 0.10 | 0.16 | 0.11 | 0.01 |
| Lidar only: RANSAC planes + DBSCAN | 0.07 | 0.12 | 0.09 | 0.06 | 0.51 |
| Camera SeamNet RGB-D labels → lidar | 0.96 | 0.89 | 0.92 | 0.80 | 0.00 |
| Camera classical RGB-D labels → lidar | 0.80 | 0.66 | 0.72 | 0.60 | 0.00 |

