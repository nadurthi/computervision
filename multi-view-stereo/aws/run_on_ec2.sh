#!/usr/bin/env bash
# The benchmark job on the EC2 instance = scripts/run_benchmark.sh (the same script you run locally) wrapped in S3
# syncing. Nothing benchmark-specific lives here.
#
#   before:  s3://$BUCKET/cache/              -> /opt/mvs/cache/             (archives + weights from earlier runs)
#            s3://$BUCKET/results/<profile>/  -> /opt/mvs/results/<profile>/ (resume from the previous instance)
#            RESET methods are also deleted from S3, so the sync cannot bring them back
#   during:  results/<profile>/ -> S3 every 5 minutes, new cache downloads every 15 minutes
#   after:   results and cache   -> S3
#
# S3 is the only persistent storage: /opt/mvs (models, data, cache, results) is on the instance disk, which is
# deleted when the instance shuts down (shutdown behaviour = terminate).
set -uo pipefail
ROOT=/opt/mvs
PROFILE=${PROFILE:-quick}
BUCKET=${BUCKET:?BUCKET not set}
CODE="$ROOT/computervision/multi-view-stereo"
S3R="s3://$BUCKET/results/$PROFILE"
S3C="s3://$BUCKET/cache"
OUT="$ROOT/results/$PROFILE"
export HOME=${HOME:-/root} CACHE_DIR="$ROOT/cache"
EXCL=(--exclude "*/depth/*" --exclude "*/disp/*" --exclude "*/ply/*" --exclude "*/vis/*")

touch /run/mvs-busy                     # the idle watchdog treats the instance as busy while this exists
mkdir -p "$ROOT" "$CACHE_DIR" "$OUT"

aws s3 sync "$S3C" "$CACHE_DIR" --only-show-errors || true
for m in ${RESET:-}; do
  if [ "$m" = all ]; then aws s3 rm "$S3R" --recursive --only-show-errors || true; break; fi
  for k in mvs stereo pointcloud; do
    aws s3 rm "$S3R/$k/$m.jsonl" --only-show-errors 2>/dev/null || true
    aws s3 rm "$S3R/logs/${k}_$m.log" --only-show-errors 2>/dev/null || true
  done
  for k in mvs stereo; do aws s3 rm "$S3R/latency/${k}_$m.jsonl" --only-show-errors 2>/dev/null || true; done
done
aws s3 sync "$S3R" "$OUT" --only-show-errors || true

( n=0; while sleep 300; do aws s3 sync "$OUT" "$S3R" --only-show-errors "${EXCL[@]}" >/dev/null 2>&1
    n=$((n + 1)); [ $((n % 3)) -eq 0 ] && aws s3 sync "$CACHE_DIR" "$S3C" --only-show-errors >/dev/null 2>&1; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null; rm -f /run/mvs-busy' EXIT

bash "$CODE/scripts/run_benchmark.sh" "$PROFILE" "$ROOT"
rc=$?

kill $SYNC_PID 2>/dev/null
cp /var/log/mvs-run.log "$OUT/logs/instance.log" 2>/dev/null || true
aws s3 sync "$OUT" "$S3R" --only-show-errors "${EXCL[@]}"
aws s3 sync "$CACHE_DIR" "$S3C" --only-show-errors
echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_on_ec2 done (benchmark exit $rc)"
exit $rc
