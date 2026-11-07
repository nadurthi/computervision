#!/bin/bash
# Shuts the instance down (shutdown behaviour = terminate, so the disk is deleted) when it is idle or has run too
# long, after a last sync of results and cache to S3. bootstrap.sh installs it as the systemd service mvs-watchdog.
#   idle  = no benchmark run in progress (/run/mvs-busy absent) AND GPU utilisation < 5 % AND 1-min load < 1.0,
#           for IDLE_MIN consecutive minutes
#   limit = uptime above MAX_HOURS, whatever is running
IDLE_MIN=${IDLE_MIN:-20}
MAX_HOURS=${MAX_HOURS:-4}
BUCKET=${BUCKET:-}
PROFILE=${PROFILE:-quick}
idle=0
while true; do
  sleep 60
  up_h=$(awk '{print int($1/3600)}' /proc/uptime)
  if [ "$up_h" -ge "$MAX_HOURS" ]; then reason="uptime ${up_h}h >= ${MAX_HOURS}h"; break; fi
  gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  gpu=${gpu:-0}
  load=$(awk '{print $1}' /proc/loadavg)
  if [ ! -e /run/mvs-busy ] && [ "$gpu" -lt 5 ] && awk "BEGIN{exit !($load < 1.0)}"; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  if [ "$idle" -ge "$IDLE_MIN" ]; then reason="idle for ${IDLE_MIN} min"; break; fi
done
echo "$(date -u) mvs-watchdog: shutting down ($reason)" | tee -a /var/log/mvs-run.log
if [ -n "$BUCKET" ]; then
  # S3 is the only persistent storage: save what the run has produced before the disk goes away
  [ -d /opt/mvs/results/$PROFILE ] && timeout 15m aws s3 sync /opt/mvs/results/$PROFILE "s3://$BUCKET/results/$PROFILE" --only-show-errors \
    --exclude "*/depth/*" --exclude "*/disp/*" --exclude "*/ply/*" --exclude "*/vis/*" || true
  [ -d /opt/mvs/cache ] && timeout 30m aws s3 sync /opt/mvs/cache "s3://$BUCKET/cache" --only-show-errors || true
  aws s3 cp /var/log/mvs-run.log "s3://$BUCKET/logs/watchdog-$(date -u +%Y%m%dT%H%M%SZ).log" || true
fi
shutdown -h now
