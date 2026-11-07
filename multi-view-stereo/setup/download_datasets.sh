#!/usr/bin/env bash
# Download and lay out every dataset the benchmark uses (see docs/DATASETS.md for the full catalogue).
#
#   bash setup/download_datasets.sh /opt/mvs/data quick      # what configs/quick.yaml needs  (~45 GB download incl. 28 GB DTU depth maps)
#   bash setup/download_datasets.sh /opt/mvs/data full       # everything in configs/full.yaml (~160 GB)
#
# Idempotent. With CACHE_S3=s3://bucket/cache set, each archive is mirrored to S3 after its first download.
# Needs: wget/curl, unzip, 7z, aws cli; gdown is installed on demand. A python with numpy/opencv/tqdm is needed for
# the ETH3D undistortion step (PYTHON, default: the main env of install_models.sh).
set -uo pipefail
DATA=${1:-/opt/mvs/data}
PROFILE=${2:-quick}
MODELS_ROOT=${MODELS_ROOT:-/opt/mvs/models}
PYTHON=${PYTHON:-$MODELS_ROOT/envs/main/bin/python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/fetch.sh"
DL="$DATA/_downloads"
mkdir -p "$DATA" "$DL"
cd "$DATA"
log() { echo "[datasets] $*"; }
FAILED=()

if [ "$PROFILE" = "full" ]; then
  ETH3D_SCENES="courtyard delivery_area electro facade kicker meadow office pipes playground relief relief_2 terrace terrains"
  DTU_POINTCLOUD=1; MIDDLEBURY2014=1; BLENDEDMVS=1
else
  ETH3D_SCENES="courtyard delivery_area electro office pipes terrains"
  DTU_POINTCLOUD=0; MIDDLEBURY2014=0; BLENDEDMVS=0
fi

# ================================================================================================ two-view stereo
# Middlebury 2014 "MiddEval3" training set, half resolution, with GT + non-occlusion masks
if [ ! -d middeval3/MiddEval3/trainingH ]; then
  for z in MiddEval3-data-H.zip MiddEval3-GT0-H.zip; do
    fetch_url "$z" "https://vision.middlebury.edu/stereo/submit3/zip/$z" "$DL/$z" && unzip -qo "$DL/$z" -d middeval3 \
      || FAILED+=("middeval3:$z")
  done
fi

# ETH3D low-res two-view training set (27 grayscale pairs) + GT with non-occlusion masks
if [ -z "$(ls eth3d_two_view 2>/dev/null)" ]; then
  mkdir -p eth3d_two_view
  for z in two_view_training.7z two_view_training_gt.7z; do
    fetch_url "eth3d_$z" "https://www.eth3d.net/data/$z" "$DL/$z" && 7z x -y -o"eth3d_two_view" "$DL/$z" >/dev/null \
      || FAILED+=("eth3d_two_view:$z")
  done
  # flatten: <scene>/ must sit directly under eth3d_two_view/
  find eth3d_two_view -mindepth 2 -name im0.png -printf '%h\n' 2>/dev/null | while read -r d; do
    [ "$(dirname "$d")" = "eth3d_two_view" ] || { mkdir -p "eth3d_two_view/$(basename "$d")"; cp -rn "$d"/. "eth3d_two_view/$(basename "$d")/"; }
  done
  find eth3d_two_view -mindepth 2 -name disp0GT.pfm -printf '%h\n' 2>/dev/null | while read -r d; do
    [ "$(dirname "$d")" = "eth3d_two_view" ] || cp -n "$d"/* "eth3d_two_view/$(basename "$d")/" 2>/dev/null
  done
fi

# KITTI 2015 stereo (200 training pairs, sparse LiDAR GT with occluded / non-occluded maps)
if [ ! -d kitti2015/training/disp_occ_0 ]; then
  fetch_url kitti_data_scene_flow.zip "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip" \
    "$DL/data_scene_flow.zip" && unzip -qo "$DL/data_scene_flow.zip" -d kitti2015 || FAILED+=("kitti2015")
fi

# Middlebury 2014 full-resolution scenes with perfect calibration (both GT disparity maps -> occlusion masks)
if [ "$MIDDLEBURY2014" = 1 ]; then
  for s in Adirondack Backpack Bicycle1 Cable Classroom1 Couch Flowers Jadeplant Mask Motorcycle Piano Pipes \
           Playroom Playtable Recycle Shelves Shopvac Sticks Storage Sword1 Sword2 Umbrella Vintage; do
    [ -d "middlebury2014/$s-perfect" ] && continue
    fetch_url "mb2014_$s.zip" "https://vision.middlebury.edu/stereo/data/scenes2014/zip/$s-perfect.zip" "$DL/mb2014_$s.zip" \
      && unzip -qo "$DL/mb2014_$s.zip" -d middlebury2014 || FAILED+=("middlebury2014:$s")
  done
fi

# ================================================================================================ multi-view
# DTU: PatchmatchNet's preprocessed evaluation set (22 test scans: 1600x1200 images, cams, pair.txt)
mkdir -p dtu/eval
if [ -z "$(ls dtu/eval 2>/dev/null)" ]; then
  if fetch_gdrive dtu_eval_patchmatchnet.zip 1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_ "$DL/dtu_eval.zip"; then
    tmp=$(mktemp -d -p "$DL"); unzip -qo "$DL/dtu_eval.zip" -d "$tmp"
    find "$tmp" -name pair.txt -printf '%h\n' | while read -r d; do mv "$d" dtu/eval/; done
    rm -rf "$tmp"
  else
    FAILED+=("dtu_eval")
  fi
fi
# DTU GT depth maps (1600x1200, mm). CasMVSNet's Depths_raw is a direct download; MVSNet's Depths (Google Drive,
# used by robustmvd) is fetched only if some evaluation scan is missing from Depths_raw.
if [ ! -d dtu/Depths_raw ]; then
  fetch_url dtu_Depths_raw.zip \
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip" \
    "$DL/Depths_raw.zip" && unzip -qo "$DL/Depths_raw.zip" -d dtu || FAILED+=("dtu_Depths_raw")
fi
missing=0
for d in dtu/eval/scan*; do
  [ -d "$d" ] || continue
  s=$(basename "$d"); [ -f "dtu/Depths_raw/$s/depth_map_0000.pfm" ] || [ -f "dtu/Depths/$s/depth_map_0000.pfm" ] || missing=$((missing + 1))
done
if [ "$missing" -gt 0 ] && [ ! -d dtu/Depths ]; then
  log "$missing DTU evaluation scans have no depth in Depths_raw; fetching MVSNet Depths"
  fetch_gdrive dtu_Depths_mvsnet.zip 1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS- "$DL/Depths.zip" \
    && unzip -qo "$DL/Depths.zip" -d dtu || FAILED+=("dtu_Depths")
fi
# DTU official point clouds + observability masks for the point-cloud evaluation
if [ "$DTU_POINTCLOUD" = 1 ] && [ ! -e dtu/dtueval/Points ]; then
  for z in Points SampleSet; do
    fetch_url "dtu_$z.zip" "http://roboimagedata2.compute.dtu.dk/data/MVS/$z.zip" "$DL/dtu_$z.zip" \
      && unzip -qo "$DL/dtu_$z.zip" -d dtu/official || FAILED+=("dtu_$z")
  done
  mkdir -p dtu/dtueval
  p=$(find dtu/official -type d -name stl | head -1); o=$(find dtu/official -type d -name ObsMask | head -1)
  [ -n "$p" ] && ln -sfn "$(realpath "$(dirname "$p")")" dtu/dtueval/Points
  [ -n "$o" ] && ln -sfn "$(realpath "$o")" dtu/dtueval/ObsMask
fi

# ETH3D high-res multi-view: distorted JPGs + calibration (for the GT undistortion), undistorted images +
# calibration, and laser-scan GT depth; then undistort the GT depth with robustmvd's exact COLMAP-model script.
mkdir -p eth3d_mvs
for s in $ETH3D_SCENES; do
  [ -d "eth3d_mvs/$s/ground_truth_depth/dslr_images_undistorted_by_rmvd" ] && continue
  for part in dslr_jpg dslr_undistorted dslr_depth; do
    f="${s}_${part}.7z"
    fetch_url "eth3d_$f" "https://www.eth3d.net/data/$f" "$DL/$f" && 7z x -y -o"eth3d_mvs" "$DL/$f" >/dev/null \
      || FAILED+=("eth3d:$f")
  done
done
if ls eth3d_mvs/*/ground_truth_depth/dslr_images >/dev/null 2>&1; then
  fetch_url rmvd_compute_eth3d_undistorted.py \
    "https://raw.githubusercontent.com/lmb-freiburg/robustmvd/master/rmvd/data/scripts/compute_eth3d_undistorted.py" \
    "$DL/compute_eth3d_undistorted.py"
  todo=""
  for s in $ETH3D_SCENES; do
    [ -d "eth3d_mvs/$s/ground_truth_depth/dslr_images" ] && [ ! -d "eth3d_mvs/$s/ground_truth_depth/dslr_images_undistorted_by_rmvd" ] && todo="$todo $s"
  done
  [ -n "$todo" ] && "$PYTHON" "$DL/compute_eth3d_undistorted.py" eth3d_mvs --scenes $todo --num_workers 4
fi

# BlendedMVS low-res (768x576, rendered GT depth for every view)
if [ "$BLENDEDMVS" = 1 ] && [ -z "$(ls blendedmvs 2>/dev/null)" ]; then
  if fetch_gdrive blendedmvs_lowres.zip 1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb "$DL/BlendedMVS.zip"; then
    tmp=$(mktemp -d -p "$DL"); unzip -qo "$DL/BlendedMVS.zip" -d "$tmp"; mkdir -p blendedmvs
    find "$tmp" -type d -name blended_images -printf '%h\n' | while read -r d; do mv "$d" blendedmvs/; done
    find "$tmp" -name validation_list.txt -exec cp {} blendedmvs/ \; ; rm -rf "$tmp"
  else
    FAILED+=("blendedmvs")
  fi
fi

du -sh "$DATA"/* 2>/dev/null | grep -v _downloads
if [ ${#FAILED[@]} -gt 0 ]; then
  log "FAILED (the benchmark skips what is missing): ${FAILED[*]}"
fi
[ "${KEEP_ARCHIVES:-0}" = 1 ] || rm -rf "$DL"
log "done"
