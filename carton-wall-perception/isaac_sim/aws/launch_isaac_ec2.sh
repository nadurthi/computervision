#!/usr/bin/env bash
# AWS CLI version of launch_isaac_ec2.py: run the Isaac Sim box-wall scripts on a g6.2xlarge in us-east-1.
#
#   ./launch_isaac_ec2.sh launch              # create role/bucket/security group if needed, start the instance
#   ./launch_isaac_ec2.sh status              # instance state, runs in S3, last log lines (via SSM)
#   ./launch_isaac_ec2.sh rerun               # later session: start the stopped instance, pull the latest code, run again
#   ./launch_isaac_ec2.sh fetch [dir]         # download the latest run from S3
#   ./launch_isaac_ec2.sh stop                # stop now (disk kept)
#   ./launch_isaac_ec2.sh terminate           # delete the instance and its disk
#   ./launch_isaac_ec2.sh cleanup             # terminate + delete role, instance profile, security group (keeps bucket)
#
# Settings (environment variables):
#   IDLE_MIN=20        stop after this many idle minutes (no run in progress, GPU < 5 %, low CPU load)
#   MAX_HOURS=4        stop after this many hours of uptime per boot, whatever is running
#   STEP_TIMEOUT=90m   time limit for each Isaac script
#   AUTO_STOP=1        stop right after the run finishes, success or failure (0 = leave it to the watchdog)
#   SPOT=0             1 = spot instance (cheaper, can be interrupted)
#   ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
#   AMI_NAME=...20260619  DLAMI build with NVIDIA driver 580 (Isaac Sim 5.1 crashes on 595); 'latest' = newest
#   AWS_PROFILE        the usual AWS CLI profile selection
#
# The automatic stop is always on. On the instance, bootstrap.sh installs isaac-watchdog.sh as a systemd service
# (idle + max-uptime stop on every boot), each Isaac script runs under a timeout, and the instance shuts itself
# down when the run ends. The user data also schedules a hard shutdown after MAX_HOURS before doing anything else,
# so even a failed git clone cannot leave the GPU running.
#
# Needs: AWS CLI v2 and the EC2 quota "Running On-Demand G and VT instances" >= 8 vCPUs in us-east-1.
set -euo pipefail

REGION=us-east-1
NAME=isaac-boxwall
INSTANCE_TYPE=g6.2xlarge
AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
# Isaac Sim 5.1 is validated on the NVIDIA 580 production driver; DLAMI builds from autumn 2026 ship 595, on which
# the 5.1 RTX renderer segfaults at startup (librtx.scenedb). So the AMI is pinned by name to a 580 build:
# 20260619 ships driver 580.159.04. Set AMI_NAME=latest to use the newest DLAMI instead.
AMI_NAME=${AMI_NAME:-"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260619"}
REPO=https://github.com/nadurthi/computervision
CODE_DIR=/opt/isaac/computervision/carton-wall-perception/isaac_sim/aws
IDLE_MIN=${IDLE_MIN:-20}; MAX_HOURS=${MAX_HOURS:-4}; STEP_TIMEOUT=${STEP_TIMEOUT:-90m}
AUTO_STOP=${AUTO_STOP:-1}; SPOT=${SPOT:-0}; ISAAC_IMAGE=${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}

export AWS_DEFAULT_REGION=$REGION
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="$NAME-$ACCOUNT-$REGION"
TAGS="Key=Project,Value=$NAME"

log() { echo "[$NAME] $*" >&2; }

instance_id() {   # instance_id [states] -> id of our instance or empty
  local states="${1:-pending,running,stopping,stopped}"
  aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=$NAME" "Name=instance-state-name,Values=$states" \
    --query 'Reservations[].Instances[].InstanceId' --output text | awk '{print $1}'
}

env_line() {      # settings passed to bootstrap.sh on the instance
  echo "BUCKET=$BUCKET IDLE_MIN=$IDLE_MIN MAX_HOURS=$MAX_HOURS STEP_TIMEOUT=$STEP_TIMEOUT ISAAC_IMAGE=$ISAAC_IMAGE AUTO_STOP=$AUTO_STOP"
}

# ------------------------------------------------------------------------------------------- resources
ensure_bucket() {
  if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null
    aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "TagSet=[{$TAGS}]"
    log "created bucket $BUCKET"
  fi
}

ensure_role() {
  if ! aws iam get-role --role-name "$NAME" >/dev/null 2>&1; then
    aws iam create-role --role-name "$NAME" --tags "$TAGS" \
      --description "Isaac Sim box-wall runner: SSM + results bucket" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam attach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    log "created role $NAME"
  fi
  aws iam put-role-policy --role-name "$NAME" --policy-name results-bucket --policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:PutObject\",\"s3:GetObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}"
  if ! aws iam get-instance-profile --instance-profile-name "$NAME" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$NAME" --tags "$TAGS" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$NAME" --role-name "$NAME"
    log "created instance profile $NAME; waiting for IAM to propagate"
    sleep 15
  fi
}

ensure_sg() {
  local vpc sg
  vpc=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$NAME" "Name=vpc-id,Values=$vpc" \
       --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$sg" = "None" ] || [ -z "$sg" ]; then
    sg=$(aws ec2 create-security-group --group-name "$NAME" --vpc-id "$vpc" \
         --description "Isaac Sim runner: no inbound, all outbound" \
         --tag-specifications "ResourceType=security-group,Tags=[{$TAGS}]" --query GroupId --output text)
    log "created security group $sg (no inbound rules)"
  fi
  echo "$sg"
}

user_data_file() {
  local f; f=$(mktemp)
  cat > "$f" <<EOF
#!/bin/bash
exec > >(tee -a /var/log/isaac-run.log) 2>&1
set -x
export HOME=/root
shutdown -h +$((MAX_HOURS * 60)) "isaac: max uptime safety net"   # in case anything below fails early
if ! git clone --depth 1 $REPO /opt/isaac/computervision; then
  aws s3 cp /var/log/isaac-run.log s3://$BUCKET/logs/boot-failed-\$(date -u +%Y%m%dT%H%M%SZ).log || true
  shutdown -h now
fi
$(env_line) bash $CODE_DIR/bootstrap.sh
EOF
  echo "$f"
}

wait_ssm() {      # wait until the instance is registered with SSM
  local id=$1
  for _ in $(seq 1 40); do
    if [ "$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$id" \
            --query 'length(InstanceInformationList)' --output text)" = "1" ]; then return 0; fi
    sleep 15
  done
  log "instance $id did not register with SSM"; return 1
}

# ------------------------------------------------------------------------------------------- commands
cmd_launch() {
  local id; id=$(instance_id)
  if [ -n "$id" ]; then log "instance already exists: $id (use rerun, stop or terminate)"; return; fi
  ensure_bucket; ensure_role
  local sg ami ud market=()
  sg=$(ensure_sg)
  if [ "$AMI_NAME" = "latest" ]; then
    ami=$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
  else
    ami=$(aws ec2 describe-images --owners amazon --filters "Name=name,Values=$AMI_NAME" \
          --query 'Images[0].ImageId' --output text)
    [ "$ami" = "None" ] && { log "AMI '$AMI_NAME' not found in $REGION"; exit 1; }
  fi
  ud=$(user_data_file)
  # Spot: one-time request, so a shutdown terminates (results are already in S3); `rerun` then means `launch` again.
  local shutdown_behavior=stop
  if [ "$SPOT" = "1" ]; then
    market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
    shutdown_behavior=terminate
  fi
  # GPU capacity varies by availability zone: try each default subnet that offers the instance type.
  local vpc subnets
  vpc=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  subnets=$(for az in $(aws ec2 describe-instance-type-offerings --location-type availability-zone \
                 --filters "Name=instance-type,Values=$INSTANCE_TYPE" --query 'InstanceTypeOfferings[].Location' --output text); do
              aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" "Name=availability-zone,Values=$az" \
                "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text
            done | grep -v None | sort)
  log "launching $INSTANCE_TYPE$([ "$SPOT" = "1" ] && echo ' (spot)') from $ami"
  id=""
  for subnet in $subnets; do
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
      if id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$INSTANCE_TYPE" --count 1 \
          --subnet-id "$subnet" --iam-instance-profile "Name=$NAME" --security-group-ids "$sg" \
          --user-data "file://$ud" --instance-initiated-shutdown-behavior "$shutdown_behavior" \
          --metadata-options HttpTokens=required \
          --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
          --tag-specifications "ResourceType=instance,Tags=[{$TAGS},{Key=Name,Value=$NAME}]" "ResourceType=volume,Tags=[{$TAGS}]" \
          "${market[@]}" --query 'Instances[0].InstanceId' --output text 2>/tmp/isaac-run-err); then
        break 2
      fi
      id=""
      if grep -q "Invalid IAM Instance Profile" /tmp/isaac-run-err && [ $attempt -lt 10 ]; then sleep 10; continue; fi
      # a just-terminated spot instance keeps counting against the spot quota for a minute or two
      if grep -q "MaxSpotInstanceCountExceeded" /tmp/isaac-run-err && [ $attempt -lt 10 ]; then log "spot quota still held by a terminating instance; retrying in 30 s"; sleep 30; continue; fi
      if grep -q "InsufficientInstanceCapacity" /tmp/isaac-run-err; then log "no capacity in $subnet, trying the next zone"; break; fi
      grep -q VcpuLimitExceeded /tmp/isaac-run-err && log "EC2 GPU quota (G and VT, $([ "$SPOT" = "1" ] && echo spot || echo on-demand)) too low in $REGION: other GPU instances may be using it"
      cat /tmp/isaac-run-err >&2; rm -f "$ud"; exit 1
    done
  done
  rm -f "$ud"
  [ -z "$id" ] && { log "no zone had capacity for $INSTANCE_TYPE; try again later or with/without SPOT=1"; exit 1; }
  log "launched $id. Auto-stop: after the run, after ${IDLE_MIN} idle min, or after ${MAX_HOURS} h."
  log "progress: $0 status    shell: aws ssm start-session --target $id"
}

cmd_rerun() {     # a later session: same instance, latest code from GitHub, same watchdog rules
  local id; id=$(instance_id)
  [ -z "$id" ] && { log "no instance; run: $0 launch"; exit 1; }
  if [ "$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)" != "running" ]; then
    aws ec2 start-instances --instance-ids "$id" >/dev/null
    aws ec2 wait instance-running --instance-ids "$id"
    log "started $id"
  fi
  wait_ssm "$id"
  local script="cd /opt/isaac/computervision && git -c safe.directory=* pull --ff-only; $(env_line) bash $CODE_DIR/bootstrap.sh >> /var/log/isaac-run.log 2>&1"
  local cid
  cid=$(aws ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
        --parameters "{\"commands\":[\"$script\"],\"executionTimeout\":[\"$((MAX_HOURS * 3600))\"]}" \
        --query Command.CommandId --output text)
  log "rerun started on $id (command $cid). The watchdog service is active on every boot."
}

cmd_status() {
  local id; id=$(instance_id pending,running,stopping,stopped,shutting-down)
  if [ -n "$id" ]; then
    aws ec2 describe-instances --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text
  else
    echo "no instance"
  fi
  echo "runs in s3://$BUCKET/runs/:"
  aws s3 ls "s3://$BUCKET/runs/" 2>/dev/null | tail -5 || true
  if [ -n "$id" ] && [ "$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)" = "running" ]; then
    local cid
    cid=$(aws ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
          --parameters '{"commands":["systemctl is-active isaac-watchdog; tail -n 25 /var/log/isaac-run.log"]}' \
          --query Command.CommandId --output text 2>/dev/null) || { echo "SSM not ready yet"; return; }
    sleep 5
    aws ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query StandardOutputContent --output text || true
  fi
}

cmd_fetch() {
  local dest="${1:-./isaac_results}" last
  last=$(aws s3 ls "s3://$BUCKET/runs/" | awk '{print $2}' | sort | tail -1)
  [ -z "$last" ] && { log "no runs yet"; exit 1; }
  aws s3 sync "s3://$BUCKET/runs/$last" "$dest/$last" --only-show-errors
  log "downloaded s3://$BUCKET/runs/$last to $dest/$last"
}

cmd_stop() {
  local id; id=$(instance_id pending,running)
  [ -n "$id" ] && aws ec2 stop-instances --instance-ids "$id" >/dev/null && log "stopping $id" || log "nothing running"
}

cmd_terminate() {
  local id; id=$(instance_id)
  [ -z "$id" ] && { log "no instance"; return; }
  aws ec2 terminate-instances --instance-ids "$id" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$id"
  log "terminated $id"
}

cmd_cleanup() {
  cmd_terminate
  local sg; sg=$(ensure_sg); aws ec2 delete-security-group --group-id "$sg" && log "deleted security group $sg" || true
  aws iam remove-role-from-instance-profile --instance-profile-name "$NAME" --role-name "$NAME" 2>/dev/null || true
  aws iam delete-instance-profile --instance-profile-name "$NAME" 2>/dev/null || true
  aws iam delete-role-policy --role-name "$NAME" --policy-name results-bucket 2>/dev/null || true
  aws iam detach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
  aws iam delete-role --role-name "$NAME" 2>/dev/null && log "deleted role and instance profile $NAME" || true
  log "kept bucket s3://$BUCKET (delete it when you no longer need the results)"
}

case "${1:-}" in
  launch) cmd_launch ;;
  rerun) cmd_rerun ;;
  status) cmd_status ;;
  fetch) shift; cmd_fetch "${1:-}" ;;
  stop) cmd_stop ;;
  terminate) cmd_terminate ;;
  cleanup) cmd_cleanup ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
