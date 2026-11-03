#!/usr/bin/env bash
# Download nuScenes v1.0-mini and keep only what the localizer needs (tables + LIDAR_TOP), streaming so the
# 4 GB archive never touches disk.  Usage: ./get_nuscenes_mini.sh <out_dir>
# nuScenes is licensed CC BY-NC-SA 4.0 (non-commercial); see https://www.nuscenes.org/terms-of-use
set -euo pipefail
OUT=${1:-data/nuscenes}
mkdir -p "$OUT"
URL=https://www.nuscenes.org/data/v1.0-mini.tgz
curl -sSfL --retry 3 "$URL" | tar -xz -C "$OUT" --wildcards --no-anchored \
  'v1.0-mini/*.json' 'samples/LIDAR_TOP/*' 'sweeps/LIDAR_TOP/*' 'maps/*.png' 2>/dev/null || \
curl -sSfL --retry 3 "$URL" | tar -xz -C "$OUT" --wildcards --no-anchored 'v1.0-mini/*.json' 'samples/LIDAR_TOP/*' 'sweeps/LIDAR_TOP/*'
du -sh "$OUT"; ls "$OUT"
