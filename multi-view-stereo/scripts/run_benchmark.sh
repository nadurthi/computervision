#!/usr/bin/env bash
# The whole benchmark in one command, on any Ubuntu 22.04 machine with an NVIDIA GPU and the CUDA toolkit:
# environments + models, datasets, stereo / MVS / point-cloud / latency benchmarks, RESULTS.md + report.html.
# Nothing here knows about AWS; aws/run_on_ec2.sh runs exactly this script and only syncs folders with S3.
#
#   bash scripts/run_benchmark.sh quick /opt/mvs          # or: full
#
# Layout under ROOT (default /opt/mvs):
#   models/  envs, model repos, weights, ACMMP build       data/  datasets
#   cache/   downloaded archives and weights (reused by later installs; delete to free space)
#   results/<profile>/  jsonl results, RESULTS.md, report.html, figures/, logs/
#
# Every step is idempotent: re-running skips installed environments, extracted datasets and finished results.
# Environment variables:
#   METHODS="a b"   run only these methods          ONLY=mvs|stereo|pointcloud|latency   run one stage
#   RESET="a b"     discard these methods' results first (RESET=all: every result of this profile)
#   SKIP_SETUP=1    skip install + downloads (everything already in place)
#   CACHE_DIR=...   download cache (default ROOT/cache; set empty to disable)
#   METHOD_TIMEOUT=3h   time limit per method and stage
set -uo pipefail
PROFILE=${1:-quick}
ROOT=${2:-/opt/mvs}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CACHE_DIR=${CACHE_DIR-$ROOT/cache}
export METHOD_TIMEOUT=${METHOD_TIMEOUT:-3h}
OUT="$ROOT/results/$PROFILE"
mkdir -p "$ROOT"/{models,data,results} "$OUT/logs"
stamp() { echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "no NVIDIA GPU visible"
if [ "${SKIP_SETUP:-0}" != 1 ]; then
  stamp "install models + environments"
  bash "$HERE/setup/install_models.sh" "$ROOT/models"
  stamp "download datasets ($PROFILE)"
  MODELS_ROOT="$ROOT/models" bash "$HERE/setup/download_datasets.sh" "$ROOT/data" "$PROFILE"
  df -h "$ROOT" | tail -1
fi

for m in ${RESET:-}; do
  if [ "$m" = all ]; then rm -rf "$OUT"; mkdir -p "$OUT/logs"; echo "[reset] all results of $PROFILE"; break; fi
  for kind in mvs stereo pointcloud; do
    rm -rf "$OUT/$kind/$m.jsonl" "$OUT/$kind/depth/$m" "$OUT/$kind/disp/$m" "$OUT/logs/${kind}_$m.log"
  done
  rm -f "$OUT/latency/mvs_$m.jsonl" "$OUT/latency/stereo_$m.jsonl"
  echo "[reset] $m"
done

stamp "benchmark ($PROFILE)"
bash "$HERE/scripts/run_all.sh" "$PROFILE" "$ROOT/data" "$ROOT/models" "$OUT"
stamp "done: $OUT/report.html"
