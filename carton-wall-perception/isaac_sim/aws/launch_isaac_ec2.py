"""Launch a g6.2xlarge in us-east-1 that runs the Isaac Sim container on the box-wall scripts from GitHub.

    pip install boto3
    python launch_isaac_ec2.py launch             # create role/bucket/SG if needed, start the instance
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
With --auto-stop (default) the instance shuts itself down when the run finishes, so it only bills while working.

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


def user_data(auto_stop, args_extra):
    stop = "shutdown -h now" if auto_stop else "true"
    return f"""#!/bin/bash
exec > >(tee /var/log/isaac-run.log) 2>&1
set -x
export HOME=/root
for i in $(seq 1 30); do nvidia-smi && break; sleep 10; done          # driver ready
apt-get install -y -q awscli >/dev/null 2>&1 || true
git clone --depth 1 {REPO} /opt/isaac/computervision || git -C /opt/isaac/computervision pull --ff-only
ISAAC_IMAGE={args_extra.get('image', 'nvcr.io/nvidia/isaac-sim:5.1.0')} \\
  bash /opt/isaac/computervision/carton-wall-perception/isaac_sim/aws/run_on_ec2.sh s3://{BUCKET}/runs
aws s3 cp /var/log/isaac-run.log s3://{BUCKET}/logs/user-data-$(date -u +%Y%m%dT%H%M%SZ).log || true
{stop}
"""


# ------------------------------------------------------------------------------------------- commands
def launch(a):
    inst = find_instance()
    if inst:
        if inst["State"]["Name"] == "stopped":
            ec2.start_instances(InstanceIds=[inst["InstanceId"]])
            print("restarted", inst["InstanceId"], "- user data does not re-run on restart; use `rerun`")
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
              UserData=user_data(a.auto_stop, {"image": a.image}),
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
    inst = find_instance(("running",))
    if not inst:
        sys.exit("no running instance")
    cmd = ssm.send_command(InstanceIds=[inst["InstanceId"]], DocumentName="AWS-RunShellScript",
                           Parameters={"commands": [f"bash /opt/isaac/computervision/carton-wall-perception/isaac_sim/aws/run_on_ec2.sh s3://{BUCKET}/runs"],
                                       "executionTimeout": ["10800"]})
    print("command", cmd["Command"]["CommandId"])


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
    p.add_argument("--no-auto-stop", dest="auto_stop", action="store_false", help="keep running after the run")
    p.add_argument("--image", default="nvcr.io/nvidia/isaac-sim:5.1.0")
    sub.add_parser("rerun"); sub.add_parser("status"); sub.add_parser("stop"); sub.add_parser("terminate")
    sub.add_parser("cleanup")
    f = sub.add_parser("fetch"); f.add_argument("dest", nargs="?", default="./isaac_results")
    a = ap.parse_args()
    {"launch": launch, "rerun": rerun, "status": status, "fetch": fetch, "stop": stop,
     "terminate": terminate, "cleanup": cleanup}[a.cmd](a)
