#!/bin/bash
# Runs on the EC2 instance (first boot via user data; every rerun via SSM):
#   1. installs the mvs-watchdog systemd service (idle / max-uptime shutdown)
#   2. waits for the NVIDIA driver
#   3. runs run_on_ec2.sh (environments, models, datasets, benchmark, report, S3 sync)
#   4. uploads the log and shuts down = terminates the instance (disk deleted; S3 keeps results + cache), whether the run succeeded or failed (unless AUTO_STOP=0)
#
# Settings come from the environment: BUCKET, PROFILE, IDLE_MIN, MAX_HOURS, METHOD_TIMEOUT, AUTO_STOP, METHODS, ONLY.
set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUCKET=${BUCKET:?BUCKET not set}
IDLE_MIN=${IDLE_MIN:-20}; MAX_HOURS=${MAX_HOURS:-8}; AUTO_STOP=${AUTO_STOP:-1}

install -m 755 "$HERE/mvs-watchdog.sh" /usr/local/bin/mvs-watchdog.sh
cat > /etc/systemd/system/mvs-watchdog.service <<UNIT
[Unit]
Description=Shut down (terminate) the instance when benchmark runs are finished, idle or too long
After=network-online.target
[Service]
Environment=IDLE_MIN=$IDLE_MIN MAX_HOURS=$MAX_HOURS BUCKET=$BUCKET PROFILE=${PROFILE:-quick}
ExecStart=/usr/local/bin/mvs-watchdog.sh
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now mvs-watchdog.service
systemctl restart mvs-watchdog.service          # pick up new IDLE_MIN / MAX_HOURS on reruns

for i in $(seq 1 30); do nvidia-smi && break; sleep 10; done

bash "$HERE/run_on_ec2.sh"
echo "run exit code: $?"
aws s3 cp /var/log/mvs-run.log "s3://$BUCKET/logs/boot-$(date -u +%Y%m%dT%H%M%SZ).log" || true
if [ "$AUTO_STOP" = "1" ]; then
  shutdown -h now
else
  echo "AUTO_STOP=0: not shutting down now; the watchdog terminates the instance after $IDLE_MIN idle minutes"
fi
