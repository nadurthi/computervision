#!/usr/bin/env bash
# AWS wrapper around ../run_all.sh. Everything the run does lives in run_all.sh (which also works on a local
# machine); this file only adds the AWS parts:
#   * the busy marker read by the auto-stop watchdog (isaac-watchdog.sh),
#   * fixed folders on the instance (/opt/isaac), and
#   * copying the run folder, report.html included, to S3.
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

OUT="$ROOT/out/$RUN" DATA="$ROOT/data-cache" HOME=/root \
  bash "$ROOT/computervision/carton-wall-perception/isaac_sim/run_all.sh"

if [ -n "$S3_DEST" ]; then
  aws s3 sync "$ROOT/out/$RUN" "$S3_DEST/$RUN/" --only-show-errors
  echo "synced to $S3_DEST/$RUN/ (open $RUN/report.html)"
fi
echo "DONE $RUN"
