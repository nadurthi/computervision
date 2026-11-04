#!/bin/bash
# Stops the instance when it is idle or has run too long. bootstrap.sh installs it as the systemd service
# mvs-watchdog, so it runs on every boot (first launch, restarts, reruns).
#   idle  = no benchmark run in progress (/run/mvs-busy absent) AND GPU utilisation < 5 % AND 1-min load < 1.0,
#           for IDLE_MIN consecutive minutes
#   limit = uptime above MAX_HOURS, whatever is running
IDLE_MIN=${IDLE_MIN:-20}
MAX_HOURS=${MAX_HOURS:-4}
BUCKET=${BUCKET:-}
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
echo "$(date -u) mvs-watchdog: stopping instance ($reason)" | tee -a /var/log/mvs-run.log
if [ -n "$BUCKET" ]; then
  aws s3 cp /var/log/mvs-run.log "s3://$BUCKET/logs/watchdog-$(date -u +%Y%m%dT%H%M%SZ).log" || true
fi
shutdown -h now
