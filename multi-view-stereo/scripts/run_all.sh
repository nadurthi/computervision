#!/usr/bin/env bash
# Run the whole benchmark for one profile: stereo, multi-view depth, (DTU point clouds), report.
# Each method runs in its own process and Python environment; results are appended to jsonl files, so an
# interrupted run resumes where it stopped. Pure local: reads DATA and MODELS, writes only under OUT.
#
#   bash scripts/run_all.sh quick /opt/mvs/data /opt/mvs/models ./results
#   DEVICE=cpu  MODELS_PYTHON=python3  ONLY=stereo|mvs|pointcloud|latency  METHODS="patchmatchnet acmmp"  to run a subset
set -uo pipefail
PROFILE=${1:-quick}; DATA=${2:-/opt/mvs/data}; MODELS=${3:-/opt/mvs/models}; OUT=${4:-./results}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$HERE/configs/$PROFILE.yaml"
ONLY=${ONLY:-all}
METHOD_TIMEOUT=${METHOD_TIMEOUT:-3h}
mkdir -p "$OUT/logs"
export PYTHONUNBUFFERED=1

# MODELS_PYTHON=python3 uses one interpreter for everything (e.g. a local venv with only the methods you run);
# otherwise each method uses its environment from setup/install_models.sh. DEVICE=cpu runs on the CPU.
DEVICE=${DEVICE:-cuda}
PY_MAIN=${MODELS_PYTHON:-$MODELS/envs/main/bin/python}

env_for() {          # python interpreter for a method
  [ -n "${MODELS_PYTHON:-}" ] && { echo "$MODELS_PYTHON"; return; }
  case "$1" in
    monster*|stereo_monster*) echo "$MODELS/envs/monster/bin/python" ;;
    mvsanywhere|vis_mvsnet|robust_mvd|colmap) echo "$MODELS/envs/mvsa/bin/python" ;;
    *) echo "$MODELS/envs/main/bin/python" ;;
  esac
}

methods_of() { "$PY_MAIN" -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(' '.join(c['$1'].get('${2:-methods}') or []))"; }

run() {             # run <kind> <method> <script> [extra args]
  local kind=$1 m=$2 script=$3 py
  shift 3
  py=$(env_for "$m")
  if [ -n "${METHODS:-}" ] && ! [[ " $METHODS " == *" $m "* ]]; then return; fi
  if ! command -v "$py" >/dev/null 2>&1; then echo "[run_all] skip $m: env missing ($py)"; return; fi
  echo "[run_all] $(date -u +%H:%M:%S) $kind / $m"
  local extra=()
  [ "$m" = colmap ] && extra=(env PATH="$MODELS/envs/mvsa/bin:$PATH")
  timeout --kill-after=2m "$METHOD_TIMEOUT" "${extra[@]}" "$py" "$HERE/scripts/$script" --config "$CFG" --method "$m" \
    --data-root "$DATA" --models-root "$MODELS" --out "$OUT" --device "$DEVICE" "$@" >> "$OUT/logs/${kind}_$m.log" 2>&1 \
    || echo "[run_all] $m exited with $? (see $OUT/logs/${kind}_$m.log)"
}

if [ "$ONLY" = all ] || [ "$ONLY" = mvs ]; then
  for m in $(methods_of mvs); do run mvs "$m" run_mvs_bench.py; done
fi
if [ "$ONLY" = all ] || [ "$ONLY" = stereo ]; then
  for m in $(methods_of stereo); do run stereo "$m" run_stereo_bench.py; done
fi
if { [ "$ONLY" = all ] || [ "$ONLY" = pointcloud ]; } && [ "$("$PY_MAIN" -c "import yaml; print(bool(yaml.safe_load(open('$CFG'))['mvs'].get('pointcloud')))")" = True ]; then
  for m in $(methods_of mvs pointcloud_methods); do run pointcloud "$m" run_dtu_pointcloud.py; done
fi
# latency: 1 warm-up + 5 timed calls per input, same resolution as the accuracy runs (after them, so it never
# delays accuracy results; skipped for methods already measured)
if [ "$ONLY" = all ] || [ "$ONLY" = latency ]; then
  for m in $(methods_of mvs); do run latency_mvs "$m" run_latency.py --kind mvs; done
  for m in $(methods_of stereo); do run latency_stereo "$m" run_latency.py --kind stereo; done
fi
"$PY_MAIN" "$HERE/scripts/make_report.py" --results "$OUT" > "$OUT/logs/report.log" 2>&1
"$PY_MAIN" "$HERE/scripts/make_figures.py" --results "$OUT" --data-root "$DATA" --config "$CFG" \
  >> "$OUT/logs/report.log" 2>&1 || true
"$PY_MAIN" "$HERE/scripts/make_html_report.py" --results "$OUT" >> "$OUT/logs/report.log" 2>&1 || true
echo "[run_all] done: $OUT/report.html  $OUT/RESULTS.md"
