# Step 02 - a CodeBuild project that downloads public datasets straight into the bucket (us-east-1).
#
# No SageMaker. CodeBuild starts a container per build, runs scripts/datasets_to_s3.py from the public
# repo branch, and stops; it is billed only while a build runs. Which datasets a build fetches is chosen
# at start time (step 05) through the DATASETS / EXTRA_ARGS / DEST_PREFIX environment variables.
#
# Creates:
#   IAM role  vb-dataset-ingest-codebuild  (trusted by codebuild.amazonaws.com)
#     inline policy vb-dataset-ingest: read/write objects in vb-data-493280347481 only, and write its
#     own CloudWatch log group /aws/codebuild/vb-dataset-ingest
#   CodeBuild project  vb-dataset-ingest  (no source, inline buildspec, BUILD_GENERAL1_MEDIUM, 8 h timeout)
#
# CLI equivalent, for reference:
#   aws iam create-role --role-name vb-dataset-ingest-codebuild --assume-role-policy-document file://trust.json
#   aws iam put-role-policy --role-name vb-dataset-ingest-codebuild --policy-name vb-dataset-ingest --policy-document file://policy.json
#   aws codebuild create-project --cli-input-json file://project.json --region us-east-1
ACCOUNT, REGION, B = "493280347481", "us-east-1", "vb-data-493280347481"
ROLE, PROJECT = "vb-dataset-ingest-codebuild", "vb-dataset-ingest"
LOG_GROUP = f"/aws/codebuild/{PROJECT}"

trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "codebuild.amazonaws.com"},
          "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}}}]}
policy = {"Version": "2012-10-17", "Statement": [
    {"Sid": "BucketList", "Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": f"arn:aws:s3:::{B}"},
    {"Sid": "BucketObjects", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
     "Resource": f"arn:aws:s3:::{B}/*"},
    {"Sid": "Logs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
     "Resource": [f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{LOG_GROUP}", f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{LOG_GROUP}:*"]}]}

buildspec = """version: 0.2
env:
  variables:
    DATASETS: "ovdata pydatavolley_dvw"
    DEST_PREFIX: "datasets"
    EXTRA_ARGS: ""
    REPO_BRANCH: "volleyball-pipeline"
phases:
  install:
    commands:
      - python3 --version
      - python3 -m pip install --quiet --upgrade pip
      - python3 -m pip install --quiet boto3 gdown SoccerNet
  build:
    commands:
      - git clone --depth 1 -b "$REPO_BRANCH" https://github.com/nadurthi/computervision /tmp/cv
      - echo "repo commit $(git -C /tmp/cv rev-parse HEAD)"
      - df -h /tmp /codebuild/output || true
      - cd /tmp/cv/volleyball && python3 scripts/datasets_to_s3.py --dest "s3://$BUCKET/$DEST_PREFIX" --scratch /codebuild/output $EXTRA_ARGS $DATASETS
  post_build:
    commands:
      - df -h /codebuild/output || true
"""

created = {}
try:
    r = await call_boto3(service_name="iam", operation_name="CreateRole", params={
        "RoleName": ROLE, "AssumeRolePolicyDocument": json.dumps(trust),
        "Description": "CodeBuild: download public datasets into s3://vb-data-493280347481 (volleyball project)"})
    created["role"] = r["Role"]["Arn"]
except Exception as e:                      # ClientError is not importable in the MCP sandbox
    if "EntityAlreadyExists" not in str(e):
        raise
    created["role"] = f"arn:aws:iam::{ACCOUNT}:role/{ROLE} (already existed)"
await call_boto3(service_name="iam", operation_name="PutRolePolicy", params={
    "RoleName": ROLE, "PolicyName": "vb-dataset-ingest", "PolicyDocument": json.dumps(policy)})

project = {
    "name": PROJECT,
    "description": "Download public volleyball/jersey datasets into s3://vb-data-493280347481/datasets",
    "source": {"type": "NO_SOURCE", "buildspec": buildspec},
    "artifacts": {"type": "NO_ARTIFACTS"},
    "environment": {"type": "LINUX_CONTAINER", "image": "aws/codebuild/amazonlinux-x86_64-standard:5.0",
                    "computeType": "BUILD_GENERAL1_MEDIUM",
                    "environmentVariables": [{"name": "BUCKET", "value": B, "type": "PLAINTEXT"}]},
    "serviceRole": f"arn:aws:iam::{ACCOUNT}:role/{ROLE}",
    "timeoutInMinutes": 480,
    "logsConfig": {"cloudWatchLogs": {"status": "ENABLED", "groupName": LOG_GROUP}},
    "tags": [{"key": "project", "value": "volleyball"}],
}
# a new role takes a few seconds to become assumable by CodeBuild: retry
last = None
for attempt in range(8):
    try:
        p = await call_boto3(service_name="codebuild", operation_name="CreateProject", region_name=REGION, params=project)
        created["project"] = p["project"]["arn"]
        break
    except Exception as e:
        if "ResourceAlreadyExistsException" in str(e):
            upd = {k: v for k, v in project.items() if k != "tags"}
            p = await call_boto3(service_name="codebuild", operation_name="UpdateProject", region_name=REGION, params=upd)
            created["project"] = p["project"]["arn"] + " (updated)"
            break
        if "not authorized to perform: sts:AssumeRole" not in str(e):
            raise
        last = str(e)
        await asyncio.sleep(5 * (attempt + 1))
else:
    raise RuntimeError(f"CreateProject kept failing: {last}")

result = {"created": created, "attempts": attempt + 1}
result
