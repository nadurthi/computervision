#!/usr/bin/env python3
"""Run the arm-motion-planning container on an EC2 instance; results go to S3.

    python aws/launch_ec2.py launch      # create bucket/role/security group if needed, start the instance
    python aws/launch_ec2.py status      # instance state and the runs in S3
    python aws/launch_ec2.py wait        # block until the instance is gone (the run ends with a shutdown)
    python aws/launch_ec2.py fetch [dir] # download the newest run from S3 (default ./results)
    python aws/launch_ec2.py terminate   # kill the instance now
    python aws/launch_ec2.py cleanup     # terminate + delete role, instance profile, security group (keeps bucket)

What the instance does (user data): install Docker + AWS CLI, clone the repo, run ./run_local.sh,
which is exactly what you run on your own machine, copy results/<run>/ to s3://<bucket>/runs/<run>/,
upload the boot log, and shut down. Shutdown = terminate, and the root disk is deleted on
termination, so S3 is the only thing that persists. A `shutdown -h +MAX_MIN` is scheduled before
anything else as a safety net.

Settings (environment variables):
    AWS_REGION=us-east-1   INSTANCE_TYPE=c7i.2xlarge   MAX_MIN=90   SPOT=0
    REPO=https://github.com/nadurthi/computervision   BRANCH=master
    RUN_ARGS='all'         arguments passed to run_local.sh (e.g. 'fcl --quick')
Needs: boto3 and AWS credentials (the usual AWS_PROFILE / env vars).
CPU only: no GPU is used, so the default vCPU quota is enough.
"""
import os
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

NAME = "arm-motion-planning"
REGION = os.environ.get("AWS_REGION", "us-east-1")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "c7i.2xlarge")
MAX_MIN = int(os.environ.get("MAX_MIN", "90"))
SPOT = os.environ.get("SPOT", "0") == "1"
REPO = os.environ.get("REPO", "https://github.com/nadurthi/computervision")
BRANCH = os.environ.get("BRANCH", "master")
RUN_ARGS = os.environ.get("RUN_ARGS", "all")
AMI_PARAM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
TAGS = [{"Key": "Project", "Value": NAME}]

session = boto3.Session(region_name=REGION)
ec2, iam, s3, ssm, sts = (session.client(c) for c in ("ec2", "iam", "s3", "ssm", "sts"))
ACCOUNT = sts.get_caller_identity()["Account"]
BUCKET = f"{NAME}-{ACCOUNT}-{REGION}"


def log(*a):
    print(f"[{NAME}]", *a, file=sys.stderr)


def user_data(run):
    return f"""#!/bin/bash
exec > >(tee -a /var/log/{NAME}.log) 2>&1
set -x
export HOME=/root DEBIAN_FRONTEND=noninteractive
shutdown -h +{MAX_MIN} "{NAME}: max uptime safety net"
upload_log() {{ aws s3 cp /var/log/{NAME}.log s3://{BUCKET}/logs/{run}.log || true; }}
apt-get update -q && apt-get install -y -q docker.io git
snap install aws-cli --classic
export PATH=/snap/bin:$PATH
if git clone --depth 1 -b {BRANCH} {REPO} /opt/code; then
  cd /opt/code/{NAME}
  RUN={run} ./run_local.sh {RUN_ARGS}
  echo "run exit code: $?"
  aws s3 sync results/{run} s3://{BUCKET}/runs/{run} --only-show-errors
fi
upload_log
shutdown -h now
"""


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        kw = {} if REGION == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": REGION}}
        s3.create_bucket(Bucket=BUCKET, **kw)
        s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration=dict(
            BlockPublicAcls=True, IgnorePublicAcls=True, BlockPublicPolicy=True, RestrictPublicBuckets=True))
        s3.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": TAGS})
        log("created bucket", BUCKET)


def ensure_role():
    import json
    try:
        iam.get_role(RoleName=NAME)
    except ClientError:
        iam.create_role(RoleName=NAME, Tags=TAGS, Description="arm-motion-planning runner: SSM + results bucket",
                        AssumeRolePolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [{
                            "Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                            "Action": "sts:AssumeRole"}]}))
        iam.attach_role_policy(RoleName=NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
        log("created role", NAME)
    iam.put_role_policy(RoleName=NAME, PolicyName="results-bucket", PolicyDocument=json.dumps({
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                                               "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                                               "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"]}]}))
    try:
        iam.get_instance_profile(InstanceProfileName=NAME)
    except ClientError:
        iam.create_instance_profile(InstanceProfileName=NAME, Tags=TAGS)
        iam.add_role_to_instance_profile(InstanceProfileName=NAME, RoleName=NAME)
        log("created instance profile; waiting for IAM to propagate")
        time.sleep(15)


def default_vpc():
    return ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]


def ensure_sg():
    vpc = default_vpc()
    g = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [NAME]},
                                              {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    if g:
        return g[0]["GroupId"]
    sg = ec2.create_security_group(GroupName=NAME, VpcId=vpc, Description=f"{NAME}: no inbound, all outbound",
                                   TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS}])["GroupId"]
    log("created security group", sg, "(no inbound rules)")
    return sg


def instances(states=("pending", "running", "stopping", "stopped", "shutting-down")):
    r = ec2.describe_instances(Filters=[{"Name": "tag:Project", "Values": [NAME]},
                                        {"Name": "instance-state-name", "Values": list(states)}])
    return [i for res in r["Reservations"] for i in res["Instances"]]


def cmd_launch():
    live = instances(("pending", "running"))
    if live:
        log("instance already running:", live[0]["InstanceId"]); return
    ensure_bucket(); ensure_role()
    sg = ensure_sg()
    ami = ssm.get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kw = dict(ImageId=ami, InstanceType=INSTANCE_TYPE, MinCount=1, MaxCount=1,
              IamInstanceProfile={"Name": NAME}, SecurityGroupIds=[sg], UserData=user_data(run),
              InstanceInitiatedShutdownBehavior="terminate", MetadataOptions={"HttpTokens": "required"},
              BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {
                  "VolumeSize": 40, "VolumeType": "gp3", "DeleteOnTermination": True}}],
              TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS + [{"Key": "Name", "Value": NAME}]},
                                 {"ResourceType": "volume", "Tags": TAGS}])
    if SPOT:
        kw["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
            "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
    for attempt in range(10):
        try:
            iid = ec2.run_instances(**kw)["Instances"][0]["InstanceId"]
            break
        except ClientError as e:
            if "Invalid IAM Instance Profile" in str(e) and attempt < 9:
                time.sleep(10); continue
            raise
    log(f"launched {iid} ({INSTANCE_TYPE}, {ami}); run {run} -> s3://{BUCKET}/runs/{run}/")
    log(f"it shuts itself down (= terminates) when the run ends, or after {MAX_MIN} min at the latest")


def cmd_status():
    for i in instances():
        print(i["InstanceId"], i["InstanceType"], i["State"]["Name"], i["LaunchTime"])
    if not instances():
        print("no instance")
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix="runs/", Delimiter="/")
    print("runs:", [p["Prefix"] for p in r.get("CommonPrefixes", [])])


def cmd_wait():
    while instances():
        time.sleep(30)
    log("no instance left running")


def cmd_fetch(dest="results"):
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix="runs/", Delimiter="/")
    runs = sorted(p["Prefix"] for p in r.get("CommonPrefixes", []))
    if not runs:
        log("no runs yet"); return
    last = runs[-1]
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=last):
        for o in page.get("Contents", []):
            p = os.path.join(dest, o["Key"][len("runs/"):])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            s3.download_file(BUCKET, o["Key"], p)
    log(f"downloaded s3://{BUCKET}/{last} to {dest}/{last[len('runs/'):]}")


def cmd_terminate():
    ids = [i["InstanceId"] for i in instances(("pending", "running", "stopping", "stopped"))]
    if not ids:
        log("no instance"); return
    ec2.terminate_instances(InstanceIds=ids)
    ec2.get_waiter("instance_terminated").wait(InstanceIds=ids)
    log("terminated", ids)


def cmd_cleanup():
    cmd_terminate()
    try:
        ec2.delete_security_group(GroupId=ensure_sg()); log("deleted security group")
    except ClientError as e:
        log("security group:", e)
    for f in (lambda: iam.remove_role_from_instance_profile(InstanceProfileName=NAME, RoleName=NAME),
              lambda: iam.delete_instance_profile(InstanceProfileName=NAME),
              lambda: iam.delete_role_policy(RoleName=NAME, PolicyName="results-bucket"),
              lambda: iam.detach_role_policy(RoleName=NAME,
                                             PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"),
              lambda: iam.delete_role(RoleName=NAME)):
        try:
            f()
        except ClientError:
            pass
    log(f"deleted role/instance profile; kept s3://{BUCKET} (delete it when you no longer need the results)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"launch": cmd_launch, "status": cmd_status, "wait": cmd_wait, "terminate": cmd_terminate,
     "stop": cmd_terminate, "cleanup": cmd_cleanup,
     "fetch": lambda: cmd_fetch(*(sys.argv[2:3] or ["results"]))}.get(cmd, lambda: print(__doc__))()
