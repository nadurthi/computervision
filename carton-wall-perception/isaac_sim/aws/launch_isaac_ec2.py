"""Launch a g6.2xlarge in us-east-1 that runs the Isaac Sim container on the box-wall scripts from GitHub.

    pip install boto3
    python launch_isaac_ec2.py launch             # create role/bucket/SG if needed, start the instance
    python launch_isaac_ec2.py rerun              # later session: start it again, pull latest code, run
    python launch_isaac_ec2.py status             # instance state, run progress, last log lines
    python launch_isaac_ec2.py fetch ./results    # download the latest run from S3
    python launch_isaac_ec2.py stop               # stop the instance (disk kept, compute billing stops)
    python launch_isaac_ec2.py terminate          # delete the instance and its disk
    python launch_isaac_ec2.py cleanup            # terminate + remove the role, profile, security group (keeps the bucket)

Uses your normal AWS credentials (profile/env). What `launch` creates, all tagged Project=isaac-boxwall:
  * IAM role + instance profile `isaac-boxwall`: SSM managed-instance access + write to the results bucket
  * S3 bucket `isaac-boxwall-<account>-us-east-1` for outputs and logs
  * security group `isaac-boxwall` with NO inbound rules (no SSH needed; use SSM Session Manager to get a shell)
  * g6.2xlarge (NVIDIA L4 24 GB, 8 vCPU, 32 GB RAM) from the AWS Deep Learning Base OSS NVIDIA Driver GPU AMI
    (Ubuntu 22.04, resolved from its public SSM parameter), 150 GB gp3 root volume

The instance's user data clones github.com/nadurthi/computervision and runs
carton-wall-perception/isaac_sim/aws/run_on_ec2.sh, which pulls nvcr.io/nvidia/isaac-sim:5.1.0, downloads the
SimReady assets, runs box_wall_capture.py and unload_cell.py headless and syncs the outputs to the bucket.
Automatic stop (always on): the instance stops itself (1) when the run finishes, whether it succeeded or failed,
(2) after 20 idle minutes (no run in progress, GPU < 5 %, low CPU load), and (3) after 4 hours of uptime per
boot, whatever is running. Each Isaac script also has a 90-minute time limit. (2) and (3) are
isaac-watchdog.sh, installed by bootstrap.sh as a systemd service, so they also cover restarts and `rerun`. Tune with --idle-minutes, --max-hours, --step-timeout;
--no-auto-stop only skips (1).

Cost (us-east-1, Nov 2026): g6.2xlarge on-demand $0.9776/h; 150 GB gp3 about $12/month while the disk exists.
Needs the EC2 quota "Running On-Demand G and VT instances" >= 8 vCPUs (new accounts have 0; request it in
Service Quotas).
"""
import argparse
import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
NAME = "isaac-boxwall"
INSTANCE_TYPE = "g6.2xlarge"
AMI_PARAM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
REPO = "https://github.com/nadurthi/computervision"
TAGS = [{"Key": "Project", "Value": NAME}]

session = boto3.Session(region_name=REGION)
ec2, iam, s3, ssm, sts = (session.client(s) for s in ("ec2", "iam", "s3", "ssm", "sts"))
ACCOUNT = sts.get_caller_identity()["Account"]
BUCKET = f"{NAME}-{ACCOUNT}-{REGION}"


# ------------------------------------------------------------------------------------------- resources
def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)      # us-east-1: no LocationConstraint
        s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        s3.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": TAGS})
        print("created bucket", BUCKET)
    return BUCKET


def ensure_role():
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    try:
        iam.get_role(RoleName=NAME)
    except iam.exceptions.NoSuchEntityException:
        iam.create_role(RoleName=NAME, AssumeRolePolicyDocument=json.dumps(trust), Tags=TAGS,
                        Description="Isaac Sim box-wall runner: SSM + results bucket")
        iam.attach_role_policy(RoleName=NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        print("created role", NAME)
    iam.put_role_policy(RoleName=NAME, PolicyName="results-bucket", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [
            {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
             "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"]}]}))
    try:
        iam.get_instance_profile(InstanceProfileName=NAME)
    except iam.exceptions.NoSuchEntityException:
        iam.create_instance_profile(InstanceProfileName=NAME, Tags=TAGS)
        iam.add_role_to_instance_profile(InstanceProfileName=NAME, RoleName=NAME)
        print("created instance profile", NAME, "- waiting for IAM to propagate")
        time.sleep(15)
    return NAME


def ensure_security_group():
    vpc = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    sgs = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [NAME]},
                                                {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if sgs:
        return sgs[0]["GroupId"]
    sg = ec2.create_security_group(GroupName=NAME, VpcId=vpc, Description="Isaac Sim runner: no inbound, all outbound",
                                   TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS}])["GroupId"]
    print("created security group", sg, "(no inbound rules)")
    return sg


def find_instance(states=("pending", "running", "stopping", "stopped")):
    r = ec2.describe_instances(Filters=[{"Name": "tag:Project", "Values": [NAME]},
                                        {"Name": "instance-state-name", "Values": list(states)}])
    inst = [i for res in r["Reservations"] for i in res["Instances"]]
    return inst[0] if inst else None


CODE_DIR = "/opt/isaac/computervision/carton-wall-perception/isaac_sim/aws"


def env_line(a):
    """Settings passed to bootstrap.sh on the instance (same as launch_isaac_ec2.sh)."""
    return (f"BUCKET={BUCKET} IDLE_MIN={a.idle_minutes} MAX_HOURS={a.max_hours} STEP_TIMEOUT={a.step_timeout} "
            f"ISAAC_IMAGE={a.image} AUTO_STOP={1 if a.auto_stop else 0}")


def user_data(a):
    """Boot script. A hard shutdown is scheduled first so even a failed clone cannot leave the GPU running; then
    bootstrap.sh (from the repo) installs the isaac-watchdog service, runs the job and stops the instance."""
    return f"""#!/bin/bash
exec > >(tee -a /var/log/isaac-run.log) 2>&1
set -x
export HOME=/root
shutdown -h +{a.max_hours * 60} "isaac: max uptime safety net"
if ! git clone --depth 1 {REPO} /opt/isaac/computervision; then
  aws s3 cp /var/log/isaac-run.log s3://{BUCKET}/logs/boot-failed-$(date -u +%Y%m%dT%H%M%SZ).log || true
  shutdown -h now
fi
{env_line(a)} bash {CODE_DIR}/bootstrap.sh
"""


# ------------------------------------------------------------------------------------------- commands
def launch(a):
    inst = find_instance()
    if inst:
        if inst["State"]["Name"] == "stopped":
            print("instance", inst["InstanceId"], "is stopped: use `rerun` to start it and run again")
        else:
            print("already exists:", inst["InstanceId"], inst["State"]["Name"])
        return
    ensure_bucket()
    profile = ensure_role()
    sg = ensure_security_group()
    ami = ssm.get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    print("AMI", ami)
    market = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time",
                                                    "InstanceInterruptionBehavior": "terminate"}} if a.spot else None
    kw = dict(ImageId=ami, InstanceType=INSTANCE_TYPE, MinCount=1, MaxCount=1,
              IamInstanceProfile={"Name": profile}, SecurityGroupIds=[sg],
              UserData=user_data(a),
              InstanceInitiatedShutdownBehavior="stop",
              BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                                    "Ebs": {"VolumeSize": 150, "VolumeType": "gp3", "DeleteOnTermination": True}}],
              MetadataOptions={"HttpTokens": "required"},
              TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS + [{"Key": "Name", "Value": NAME}]},
                                 {"ResourceType": "volume", "Tags": TAGS}])
    if market:
        kw["InstanceMarketOptions"] = market
    for attempt in range(4):
        try:
            iid = ec2.run_instances(**kw)["Instances"][0]["InstanceId"]
            break
        except ClientError as e:
            if "Invalid IAM Instance Profile" in str(e) and attempt < 3:
                time.sleep(10)
                continue
            if "VcpuLimitExceeded" in str(e):
                sys.exit("EC2 quota: 'Running On-Demand G and VT instances' must be >= 8 vCPUs in us-east-1 "
                         "(Service Quotas console). " + str(e))
            raise
    print(f"launched {iid} ({INSTANCE_TYPE}{', spot' if a.spot else ''}). The run takes ~30-60 min the first time.")
    print(f"progress:  python {sys.argv[0]} status        shell:  aws ssm start-session --target {iid}")


def rerun(a):
    """A later session: start the stopped instance, pull the latest code, run again under the same auto-stop rules."""
    inst = find_instance(("pending", "running", "stopping", "stopped"))
    if not inst:
        sys.exit("no instance; run `launch` first")
    iid = inst["InstanceId"]
    if inst["State"]["Name"] != "running":
        if inst["State"]["Name"] == "stopping":
            ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
        ec2.start_instances(InstanceIds=[iid])
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        print("started", iid)
    for _ in range(40):                                       # wait for the SSM agent
        info = ssm.describe_instance_information(Filters=[{"Key": "InstanceIds", "Values": [iid]}])
        if info["InstanceInformationList"]:
            break
        time.sleep(15)
    script = (f"cd /opt/isaac/computervision && git pull --ff-only; "
              f"{env_line(a)} bash {CODE_DIR}/bootstrap.sh >> /var/log/isaac-run.log 2>&1")
    cmd = ssm.send_command(InstanceIds=[iid], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": [script], "executionTimeout": [str(a.max_hours * 3600)]})
    print("rerun started:", cmd["Command"]["CommandId"])


def status(a):
    inst = find_instance(("pending", "running", "stopping", "stopped", "shutting-down"))
    if not inst:
        print("no instance")
    else:
        print(inst["InstanceId"], inst["State"]["Name"], inst.get("LaunchTime"))
    runs = s3.list_objects_v2(Bucket=BUCKET, Prefix="runs/", Delimiter="/").get("CommonPrefixes", [])
    print("runs in s3:", [p["Prefix"] for p in runs][-5:])
    if inst and inst["State"]["Name"] == "running":
        try:
            c = ssm.send_command(InstanceIds=[inst["InstanceId"]], DocumentName="AWS-RunShellScript",
                                 Parameters={"commands": ["tail -n 25 /var/log/isaac-run.log"]})
            time.sleep(4)
            out = ssm.get_command_invocation(CommandId=c["Command"]["CommandId"], InstanceId=inst["InstanceId"])
            print(out.get("StandardOutputContent", ""))
        except ClientError as e:
            print("SSM not ready yet:", e)


def fetch(a):
    runs = sorted(p["Prefix"] for p in s3.list_objects_v2(Bucket=BUCKET, Prefix="runs/", Delimiter="/").get("CommonPrefixes", []))
    if not runs:
        sys.exit("no runs yet")
    import os
    prefix = runs[-1]
    pag = s3.get_paginator("list_objects_v2")
    n = 0
    for page in pag.paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            dst = os.path.join(a.dest, o["Key"][len("runs/"):])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            s3.download_file(BUCKET, o["Key"], dst)
            n += 1
    print(f"downloaded {n} files from s3://{BUCKET}/{prefix} to {a.dest}")


def stop(a):
    inst = find_instance(("pending", "running"))
    if inst:
        ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
        print("stopping", inst["InstanceId"])
    else:
        print("nothing running")


def terminate(a):
    inst = find_instance()
    if inst:
        ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[inst["InstanceId"]])
        print("terminated", inst["InstanceId"])
    else:
        print("no instance")


def cleanup(a):
    terminate(a)
    try:
        sg = ensure_security_group()
        ec2.delete_security_group(GroupId=sg)
        print("deleted security group", sg)
    except ClientError as e:
        print("security group:", e)
    try:
        iam.remove_role_from_instance_profile(InstanceProfileName=NAME, RoleName=NAME)
        iam.delete_instance_profile(InstanceProfileName=NAME)
        iam.delete_role_policy(RoleName=NAME, PolicyName="results-bucket")
        iam.detach_role_policy(RoleName=NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        iam.delete_role(RoleName=NAME)
        print("deleted role and instance profile", NAME)
    except ClientError as e:
        print("iam:", e)
    print(f"kept bucket s3://{BUCKET} (delete it in the console when you no longer need the results)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch")
    p.add_argument("--spot", action="store_true", help="spot instance (~20%% cheaper, can be interrupted)")
    p.add_argument("--no-auto-stop", dest="auto_stop", action="store_false",
                   help="do not stop right after the run (the idle/uptime watchdog still applies)")
    p.add_argument("--image", default="nvcr.io/nvidia/isaac-sim:5.1.0")
    p.add_argument("--idle-minutes", type=int, default=20, help="stop after this many idle minutes (default 20)")
    p.add_argument("--max-hours", type=int, default=4, help="hard limit on uptime per boot (default 4)")
    p.add_argument("--step-timeout", default="90m", help="time limit for each Isaac script (default 90m)")
    r = sub.add_parser("rerun")
    for q in (p, r):
        if q is r:
            q.add_argument("--no-auto-stop", dest="auto_stop", action="store_false")
            q.add_argument("--image", default="nvcr.io/nvidia/isaac-sim:5.1.0")
            q.add_argument("--idle-minutes", type=int, default=20)
            q.add_argument("--max-hours", type=int, default=4)
            q.add_argument("--step-timeout", default="90m")
    sub.add_parser("status"); sub.add_parser("stop"); sub.add_parser("terminate")
    sub.add_parser("cleanup")
    f = sub.add_parser("fetch"); f.add_argument("dest", nargs="?", default="./isaac_results")
    a = ap.parse_args()
    {"launch": launch, "rerun": rerun, "status": status, "fetch": fetch, "stop": stop,
     "terminate": terminate, "cleanup": cleanup}[a.cmd](a)
