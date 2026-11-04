#!/usr/bin/env bash
# Run the whole benchmark for one profile: stereo, multi-view depth, (DTU point clouds), report.
# Each method runs in its own process and Python environment; results are appended to jsonl files, so an
# interrupted run resumes where it stopped. Results are synced to S3 after every method if S3_OUT is set.
#
#   bash scripts/run_all.sh quick /opt/mvs/data /opt/mvs/models ./results [s3://bucket/runs/<id>]
#   ONLY=stereo|mvs|pointcloud  METHODS="patchmatchnet acmmp"  to run a subset
set -uo pipefail
PROFILE=${1:-quick}; DATA=${2:-/opt/mvs/data}; MODELS=${3:-/opt/mvs/models}; OUT=${4:-./results}; S3_OUT=${5:-}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$HERE/configs/$PROFILE.yaml"
ONLY=${ONLY:-all}
METHOD_TIMEOUT=${METHOD_TIMEOUT:-3h}
mkdir -p "$OUT/logs"
export PYTHONUNBUFFERED=1

env_for() {          # python interpreter for a method
  case "$1" in
    monster*|stereo_monster*) echo "$MODELS/envs/monster/bin/python" ;;
    mvsanywhere|vis_mvsnet|robust_mvd|colmap) echo "$MODELS/envs/mvsa/bin/python" ;;
    *) echo "$MODELS/envs/main/bin/python" ;;
  esac
}

methods_of() { "$MODELS/envs/main/bin/python" -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(' '.join(c['$1'].get('${2:-methods}') or []))"; }

sync() { [ -n "$S3_OUT" ] && aws s3 sync "$OUT" "$S3_OUT" --only-show-errors --exclude "*/depth/*" --exclude "*/ply/*" --exclude "*/vis/*" || true; }

run() {             # run <kind> <method> <script>
  local kind=$1 m=$2 script=$3 py
  py=$(env_for "$m")
  if [ -n "${METHODS:-}" ] && ! [[ " $METHODS " == *" $m "* ]]; then return; fi
  if [ ! -x "$py" ]; then echo "[run_all] skip $m: env missing ($py)"; return; fi
  echo "[run_all] $(date -u +%H:%M:%S) $kind / $m"
  local extra=()
  [ "$m" = colmap ] && extra=(env PATH="$MODELS/envs/mvsa/bin:$PATH")
  timeout --kill-after=2m "$METHOD_TIMEOUT" "${extra[@]}" "$py" "$HERE/scripts/$script" --config "$CFG" --method "$m" \
    --data-root "$DATA" --models-root "$MODELS" --out "$OUT" >> "$OUT/logs/${kind}_$m.log" 2>&1 \
    || echo "[run_all] $m exited with $? (see $OUT/logs/${kind}_$m.log)"
  sync
}

if [ "$ONLY" = all ] || [ "$ONLY" = mvs ]; then
  for m in $(methods_of mvs); do run mvs "$m" run_mvs_bench.py; done
fi
if [ "$ONLY" = all ] || [ "$ONLY" = stereo ]; then
  for m in $(methods_of stereo); do run stereo "$m" run_stereo_bench.py; done
fi
if { [ "$ONLY" = all ] || [ "$ONLY" = pointcloud ]; } && [ "$("$MODELS/envs/main/bin/python" -c "import yaml; print(bool(yaml.safe_load(open('$CFG'))['mvs'].get('pointcloud')))")" = True ]; then
  for m in $(methods_of mvs pointcloud_methods); do run pointcloud "$m" run_dtu_pointcloud.py; done
fi
"$MODELS/envs/main/bin/python" "$HERE/scripts/make_report.py" --results "$OUT" > "$OUT/logs/report.log" 2>&1
"$MODELS/envs/main/bin/python" "$HERE/scripts/make_figures.py" --results "$OUT" --data-root "$DATA" --config "$CFG" \
  >> "$OUT/logs/report.log" 2>&1 || true
sync
echo "[run_all] done: $OUT/RESULTS.md"
