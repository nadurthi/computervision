#!/bin/bash
# Runs on the EC2 instance at first boot (called from the user data written by launch_isaac_ec2.sh / .py):
#   1. installs the isaac-watchdog systemd service (idle / max-uptime stop, active on every boot)
#   2. waits for the NVIDIA driver
#   3. runs run_on_ec2.sh (container pull, assets, both Isaac scripts, S3 sync)
#   4. uploads the boot log and stops the instance, whether the run succeeded or failed (unless AUTO_STOP=0)
#
# Settings come from the environment: BUCKET, IDLE_MIN, MAX_HOURS, STEP_TIMEOUT, ISAAC_IMAGE, AUTO_STOP.
set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET=${BUCKET:?BUCKET not set}
IDLE_MIN=${IDLE_MIN:-20}; MAX_HOURS=${MAX_HOURS:-4}; STEP_TIMEOUT=${STEP_TIMEOUT:-90m}
ISAAC_IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}; AUTO_STOP=${AUTO_STOP:-1}

install -m 755 "$HERE/isaac-watchdog.sh" /usr/local/bin/isaac-watchdog.sh
cat > /etc/systemd/system/isaac-watchdog.service <<UNIT
[Unit]
Description=Stop the instance when Isaac runs are finished, idle or too long
After=network-online.target
[Service]
Environment=IDLE_MIN=$IDLE_MIN MAX_HOURS=$MAX_HOURS BUCKET=$BUCKET
ExecStart=/usr/local/bin/isaac-watchdog.sh
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now isaac-watchdog.service

for i in $(seq 1 30); do nvidia-smi && break; sleep 10; done

ISAAC_IMAGE="$ISAAC_IMAGE" STEP_TIMEOUT="$STEP_TIMEOUT" bash "$HERE/run_on_ec2.sh" "s3://$BUCKET/runs"
echo "run exit code: $?"
aws s3 cp /var/log/isaac-run.log "s3://$BUCKET/logs/boot-$(date -u +%Y%m%dT%H%M%SZ).log" || true
if [ "$AUTO_STOP" = "1" ]; then
  shutdown -h now
else
  echo "AUTO_STOP=0: not stopping now; the watchdog stops the instance after $IDLE_MIN idle minutes"
fi
