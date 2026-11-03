# LIDAR-based localization

Scan matching, particle-filter localization, map building and lidar-inertial EKF on KITTI and turtlebot data.

## Papers

- N. Adurthi. **Scan-Matching Based Particle Filtering Approach for LIDAR-Only Localization**. *arXiv:2302.06843*, 2023. [https://arxiv.org/abs/2302.06843](https://arxiv.org/abs/2302.06843)
  - Particle filter whose likelihood comes from fast binned scan matching against a prebuilt map; KITTI experiments.
  - Reproduces / validates with: [`lidarprocessing`](lidarprocessing), [`drivers/main_kitti_filter_localize.py`](drivers/main_kitti_filter_localize.py), [`drivers/main_kitti_global_2Dlocalize.py`](drivers/main_kitti_global_2Dlocalize.py), [`drivers/main_kitti_map_build.py`](drivers/main_kitti_map_build.py)
  - Paper figures: [`media/paper/`](media/paper/); paper on [arXiv:2302.06843](https://arxiv.org/abs/2302.06843)

## Entry-point scripts

Scripts (not functions) found in this folder. Most simulation drivers that regenerate paper figures are here.

- [`lidarprocessing/__init__.py`](lidarprocessing/__init__.py)
- [`lidarprocessing/pclregistration.py`](lidarprocessing/pclregistration.py)
- [`lidarprocessing/pklmsgtopython.py`](lidarprocessing/pklmsgtopython.py)
- [`lidarprocessing/3DScanMatching/readKittiOdom.py`](lidarprocessing/3DScanMatching/readKittiOdom.py)
- [`lidarprocessing/core/__init__.py`](lidarprocessing/core/__init__.py)
- [`lidarprocessing/core/non_max_suppression/__init__.py`](lidarprocessing/core/non_max_suppression/__init__.py)
- [`lidarprocessing/core/point_cloud/__init__.py`](lidarprocessing/core/point_cloud/__init__.py)
- [`lidarprocessing/numba_codes/__init__.py`](lidarprocessing/numba_codes/__init__.py)
- [`lidarprocessing/point2d/train.py`](lidarprocessing/point2d/train.py)
- [`odometry/__init__.py`](odometry/__init__.py)
- [`odometry/open3DegVIZ2.py`](odometry/open3DegVIZ2.py)
- [`odometry/visualodo.py`](odometry/visualodo.py)

Full function list: [`llm/08-lidar-localization.md`](../llm/08-lidar-localization.md). Original repositories: see [SOURCES.md](../SOURCES.md).
