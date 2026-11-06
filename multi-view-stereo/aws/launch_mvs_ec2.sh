#!/usr/bin/env bash
# Run the stereo / multi-view-stereo benchmark on a GPU EC2 instance in us-east-1 (g6.2xlarge L4, else g5.2xlarge A10G).
#
#   ./launch_mvs_ec2.sh quota               # is the G-instance vCPU quota approved yet?
#   ./launch_mvs_ec2.sh launch              # create role/bucket/security group if needed, start the instance + run
#   ./launch_mvs_ec2.sh status              # instance state, results in S3, last log lines (via SSM, no SSH)
#   ./launch_mvs_ec2.sh rerun               # later session: start the stopped instance, pull the latest code, resume
#   ./launch_mvs_ec2.sh fetch [dir]         # download results (RESULTS.md, jsonl, figures) from S3
#   ./launch_mvs_ec2.sh shell               # SSM shell on the instance
#   ./launch_mvs_ec2.sh stop                # stop now (disk with envs + datasets is kept)
#   ./launch_mvs_ec2.sh terminate           # delete the instance and its disk
#   ./launch_mvs_ec2.sh cleanup             # terminate + delete role, instance profile, security group (keeps bucket)
#
# Settings (environment variables):
#   PROFILE=quick      quick (~3-5 h incl. setup) or full (~10-14 h, resumes across sessions)
#   INSTANCE_TYPES="g6.2xlarge g5.2xlarge"   tried in order, in every us-east-1 zone that offers them, until one has
#                      capacity (g6 = L4 24 GB ~$0.98/h, g5 = A10G 24 GB ~$1.21/h; both need 8 G/VT vCPUs)
#   IDLE_MIN=20        stop after this many idle minutes (no run in progress, GPU < 5 %, low CPU load)
#   MAX_HOURS=8        stop after this many hours of uptime per boot, whatever is running
#   METHOD_TIMEOUT=3h  time limit per method and benchmark
#   AUTO_STOP=1        stop right after the run finishes, success or failure (0 = leave it to the watchdog)
#   SPOT=0             1 = spot instance (cheaper, can be interrupted; the run resumes on rerun)
#   BRANCH=multi-view-stereo   branch of github.com/nadurthi/computervision to run
#   DISK_GB=500
#   AWS_PROFILE        the usual AWS CLI profile selection
#
# The automatic stop is always on: bootstrap.sh installs mvs-watchdog.sh as a systemd service (idle + max-uptime
# stop on every boot), each method runs under a timeout, the instance shuts itself down when the run ends, and the
# user data schedules a hard shutdown after MAX_HOURS before doing anything else.
set -euo pipefail

REGION=us-east-1
NAME=mvs-bench
INSTANCE_TYPES=${INSTANCE_TYPES:-${INSTANCE_TYPE:-g6.2xlarge g5.2xlarge}}
AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
REPO=https://github.com/nadurthi/computervision
BRANCH=${BRANCH:-multi-view-stereo}
ROOT=/opt/mvs
CODE_DIR=$ROOT/computervision/multi-view-stereo
PROFILE=${PROFILE:-quick}
IDLE_MIN=${IDLE_MIN:-20}; MAX_HOURS=${MAX_HOURS:-8}; METHOD_TIMEOUT=${METHOD_TIMEOUT:-3h}
AUTO_STOP=${AUTO_STOP:-1}; SPOT=${SPOT:-0}; DISK_GB=${DISK_GB:-500}

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
  echo "BUCKET=$BUCKET PROFILE=$PROFILE IDLE_MIN=$IDLE_MIN MAX_HOURS=$MAX_HOURS METHOD_TIMEOUT=$METHOD_TIMEOUT AUTO_STOP=$AUTO_STOP BRANCH=$BRANCH ${METHODS:+METHODS=\"$METHODS\"} ${ONLY:+ONLY=$ONLY}"
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
      --description "MVS benchmark runner: SSM + benchmark bucket" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    aws iam attach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    log "created role $NAME"
  fi
  aws iam put-role-policy --role-name "$NAME" --policy-name bench-bucket --policy-document \
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
         --description "MVS benchmark runner: no inbound, all outbound" \
         --tag-specifications "ResourceType=security-group,Tags=[{$TAGS}]" --query GroupId --output text)
    log "created security group $sg (no inbound rules)"
  fi
  echo "$sg"
}

user_data_file() {
  local f; f=$(mktemp)
  cat > "$f" <<EOF
#!/bin/bash
exec > >(tee -a /var/log/mvs-run.log) 2>&1
set -x
export HOME=/root
shutdown -h +$((MAX_HOURS * 60)) "mvs: max uptime safety net"   # in case anything below fails early
mkdir -p $ROOT
if ! git clone --depth 1 --branch $BRANCH $REPO $ROOT/computervision; then
  aws s3 cp /var/log/mvs-run.log s3://$BUCKET/logs/boot-failed-\$(date -u +%Y%m%dT%H%M%SZ).log || true
  shutdown -h now
fi
$(env_line) bash $CODE_DIR/aws/bootstrap.sh
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
cmd_quota() {
  local code=L-DB2E81BA
  [ "$SPOT" = "1" ] && code=L-3819A6DF
  aws service-quotas get-service-quota --service-code ec2 --quota-code $code \
    --query 'Quota.[QuotaName,Value]' --output text
  aws service-quotas list-requested-service-quota-change-history --service-code ec2 \
    --query "RequestedQuotas[?QuotaCode=='$code'].[QuotaName,DesiredValue,Status,Created]" --output text
}

cmd_launch() {
  local id; id=$(instance_id)
  if [ -n "$id" ]; then log "instance already exists: $id (use rerun, stop or terminate)"; return; fi
  ensure_bucket; ensure_role
  local sg ami ud market=()
  sg=$(ensure_sg)
  ami=$(aws ssm get-parameter --name "$AMI_PARAM" --query Parameter.Value --output text)
  ud=$(user_data_file)
  [ "$SPOT" = "1" ] && market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}')
  local vpc; vpc=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  id=""
  for itype in $INSTANCE_TYPES; do
    for az in $(aws ec2 describe-instance-type-offerings --location-type availability-zone \
                  --filters "Name=instance-type,Values=$itype" --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort); do
      local subnet
      subnet=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc" "Name=availability-zone,Values=$az" \
               "Name=default-for-az,Values=true" --query 'Subnets[0].SubnetId' --output text)
      [ "$subnet" = "None" ] && continue
      log "trying $itype in $az (profile $PROFILE, branch $BRANCH, $ami)"
      for attempt in 1 2 3 4; do
        if id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$itype" --count 1 \
            --iam-instance-profile "Name=$NAME" \
            --network-interfaces "DeviceIndex=0,SubnetId=$subnet,Groups=$sg,AssociatePublicIpAddress=true" \
            --user-data "file://$ud" --instance-initiated-shutdown-behavior stop \
            --metadata-options HttpTokens=required \
            --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,Throughput=500,Iops=6000,DeleteOnTermination=true}" \
            --tag-specifications "ResourceType=instance,Tags=[{$TAGS},{Key=Name,Value=$NAME}]" "ResourceType=volume,Tags=[{$TAGS}]" \
            "${market[@]}" --query 'Instances[0].InstanceId' --output text 2>/tmp/mvs-run-err); then
          log "launched $itype in $az"
          break 3
        fi
        id=""
        if grep -q "Invalid IAM Instance Profile" /tmp/mvs-run-err && [ $attempt -lt 4 ]; then sleep 10; continue; fi
        break
      done
      if grep -qE "InsufficientInstanceCapacity|Unsupported" /tmp/mvs-run-err; then
        log "no $itype capacity in $az"; continue
      fi
      grep -q VcpuLimitExceeded /tmp/mvs-run-err && { log "G/VT vCPU quota in $REGION is too low for $itype:"; cmd_quota >&2; }
      cat /tmp/mvs-run-err >&2; rm -f "$ud"; exit 1
    done
  done
  if [ -z "$id" ]; then log "no capacity for any of: $INSTANCE_TYPES (try later, or SPOT=1)"; rm -f "$ud"; exit 1; fi
  rm -f "$ud"
  log "launched $id. Auto-stop: after the run, after ${IDLE_MIN} idle min, or after ${MAX_HOURS} h."
  log "progress: $0 status    shell: $0 shell"
}

cmd_rerun() {     # a later session: same instance + disk (envs, datasets, partial results), latest code
  local id; id=$(instance_id)
  [ -z "$id" ] && { log "no instance; run: $0 launch"; exit 1; }
  if [ "$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)" != "running" ]; then
    aws ec2 start-instances --instance-ids "$id" >/dev/null
    aws ec2 wait instance-running --instance-ids "$id"
    log "started $id"
  fi
  wait_ssm "$id"
  local script="cd $ROOT/computervision && git fetch --depth 1 origin $BRANCH && git checkout -q -B $BRANCH FETCH_HEAD; $(env_line) bash $CODE_DIR/aws/bootstrap.sh >> /var/log/mvs-run.log 2>&1"
  script=${script//\"/\\\"}
  local cid
  cid=$(aws ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
        --parameters "{\"commands\":[\"$script\"],\"executionTimeout\":[\"$((MAX_HOURS * 3600))\"]}" \
        --query Command.CommandId --output text)
  log "rerun started on $id (command $cid, profile $PROFILE). Finished methods are skipped automatically."
}

cmd_status() {
  local id; id=$(instance_id pending,running,stopping,stopped,shutting-down)
  if [ -n "$id" ]; then
    aws ec2 describe-instances --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text
  else
    echo "no instance"
  fi
  echo "results in s3://$BUCKET/results/:"
  aws s3 ls "s3://$BUCKET/results/" 2>/dev/null || true
  if [ -n "$id" ] && [ "$(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)" = "running" ]; then
    local cid
    cid=$(aws ssm send-command --instance-ids "$id" --document-name AWS-RunShellScript \
          --parameters '{"commands":["systemctl is-active mvs-watchdog; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; tail -n 30 /var/log/mvs-run.log"]}' \
          --query Command.CommandId --output text 2>/dev/null) || { echo "SSM not ready yet"; return; }
    sleep 5
    aws ssm get-command-invocation --command-id "$cid" --instance-id "$id" --query StandardOutputContent --output text || true
  fi
}

cmd_fetch() {
  local dest="${1:-./mvs_results}"
  aws s3 sync "s3://$BUCKET/results/" "$dest" --only-show-errors
  log "downloaded s3://$BUCKET/results/ to $dest (see $dest/<profile>/RESULTS.md)"
}

cmd_shell() {
  local id; id=$(instance_id running)
  [ -z "$id" ] && { log "not running"; exit 1; }
  aws ssm start-session --target "$id"
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
  aws iam delete-role-policy --role-name "$NAME" --policy-name bench-bucket 2>/dev/null || true
  aws iam detach-role-policy --role-name "$NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
  aws iam delete-role --role-name "$NAME" 2>/dev/null && log "deleted role and instance profile $NAME" || true
  log "kept bucket s3://$BUCKET (dataset/weight cache + results; delete it when no longer needed)"
}

case "${1:-}" in
  quota) cmd_quota ;;
  launch) cmd_launch ;;
  rerun) cmd_rerun ;;
  status) cmd_status ;;
  fetch) shift; cmd_fetch "${1:-}" ;;
  shell) cmd_shell ;;
  stop) cmd_stop ;;
  terminate) cmd_terminate ;;
  cleanup) cmd_cleanup ;;
  *) sed -n '2,30p' "$0"; exit 1 ;;
esac
