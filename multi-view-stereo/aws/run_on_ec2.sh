#!/usr/bin/env bash
# The benchmark job on the GPU instance. Everything lives on the instance disk under /opt/mvs and survives
# stop/start, so a rerun skips installed environments, downloaded data and finished methods.
#
#   /opt/mvs/computervision   this repository (branch $BRANCH)
#   /opt/mvs/models           envs/, model repos, weights, ACMMP build
#   /opt/mvs/data             datasets
#   /opt/mvs/results/<profile>  jsonl results, RESULTS.md, figures  -> s3://$BUCKET/results/<profile>/
#
# Archives and Google-Drive weights are mirrored to s3://$BUCKET/cache/ after their first download.
set -uo pipefail
ROOT=/opt/mvs
PROFILE=${PROFILE:-quick}
BUCKET=${BUCKET:?BUCKET not set}
CODE="$ROOT/computervision/multi-view-stereo"
export CACHE_S3="s3://$BUCKET/cache"
export METHOD_TIMEOUT=${METHOD_TIMEOUT:-3h}
export HOME=${HOME:-/root}

touch /run/mvs-busy                     # the idle watchdog treats the instance as busy while this exists
trap 'rm -f /run/mvs-busy' EXIT
mkdir -p "$ROOT"/{models,data,results}
stamp() { echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

stamp "GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
stamp "install models + environments"
bash "$CODE/setup/install_models.sh" "$ROOT/models"
stamp "download datasets ($PROFILE)"
MODELS_ROOT="$ROOT/models" bash "$CODE/setup/download_datasets.sh" "$ROOT/data" "$PROFILE"
df -h "$ROOT" | tail -1

stamp "benchmark ($PROFILE)"
OUT="$ROOT/results/$PROFILE"
# continue from results synced by an earlier instance (e.g. after a terminate) before resuming
aws s3 sync "s3://$BUCKET/results/$PROFILE" "$OUT" --only-show-errors || true
bash "$CODE/scripts/run_all.sh" "$PROFILE" "$ROOT/data" "$ROOT/models" "$OUT" "s3://$BUCKET/results/$PROFILE"
cp /var/log/mvs-run.log "$OUT/logs/instance.log" 2>/dev/null || true
aws s3 sync "$OUT" "s3://$BUCKET/results/$PROFILE" --only-show-errors --exclude "*/depth/*" --exclude "*/ply/*"
stamp "done"
