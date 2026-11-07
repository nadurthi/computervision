#!/usr/bin/env bash
# AWS wrapper around ../run_all.sh. Everything the run does lives in run_all.sh (which also works on a local
# machine); this file only adds the AWS parts:
#   * the busy marker read by the auto-stop watchdog (isaac-watchdog.sh),
#   * fixed folders on the instance (/opt/isaac), and
#   * copying the run folder, report.html included, to S3, and
#   * an S3 cache of the Isaac image, SimReady models and shader caches (s3_cache.sh; NO_CACHE=1 skips it).
#
#   sudo bash run_on_ec2.sh s3://my-bucket/isaac-runs
#   env: STEP_TIMEOUT, RUN_SCRIPTS, EXTRA_ARGS, FRAMES, PICKS, ISAAC_IMAGE are passed through to run_all.sh
set -uo pipefail
S3_DEST="${1:-}"
ROOT=/opt/isaac
RUN="run-$(date -u +%Y%m%dT%H%M%SZ)"

# The idle watchdog treats the instance as busy while this marker exists.
touch /run/isaac-busy
trap 'rm -f /run/isaac-busy' EXIT

# latest code (the repo is chowned to the container uid on earlier runs, hence safe.directory)
if [ ! -d "$ROOT/computervision" ]; then
  git clone --depth 1 https://github.com/nadurthi/computervision "$ROOT/computervision"
else
  git -c safe.directory="*" -C "$ROOT/computervision" pull --ff-only
fi

IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
CACHE="${S3_DEST%/runs}/cache"                      # s3://<bucket>/cache, next to the runs
AWS_DIR="$ROOT/computervision/carton-wall-perception/isaac_sim/aws"
# fresh instances keep nothing between sessions: reload the Isaac image, SimReady models and shader caches from S3
[ -n "$S3_DEST" ] && [ "${NO_CACHE:-0}" != "1" ] && bash "$AWS_DIR/s3_cache.sh" restore "$CACHE" "$ROOT/data-cache" "$IMAGE"

OUT="$ROOT/out/$RUN" DATA="$ROOT/data-cache" HOME=/root \
  bash "$ROOT/computervision/carton-wall-perception/isaac_sim/run_all.sh"

if [ -n "$S3_DEST" ]; then
  aws s3 sync "$ROOT/out/$RUN" "$S3_DEST/$RUN/" --only-show-errors
  echo "synced to $S3_DEST/$RUN/ (open $RUN/report.html)"
  # results first, then refresh the cache (image once per tag; data when it grew)
  [ "${NO_CACHE:-0}" != "1" ] && bash "$AWS_DIR/s3_cache.sh" save "$CACHE" "$ROOT/data-cache" "$IMAGE"
fi
echo "DONE $RUN"
