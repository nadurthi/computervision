# Step 04 - fetch the small open datasets into S3 with a short-lived Lambda function (no SageMaker, no CodeBuild).
#
# Why Lambda: CodeBuild's concurrent-build quota is 0 in this account until step 03's request is approved.
# Lambda allows 15 minutes and has internet access, which is enough for the small sets (openvolley: ~6 MB).
# The big Google Drive / SoccerNet sets go through CodeBuild (step 05) once the quota is raised.
#
# The function downloads scripts/datasets_to_s3.py from the repo at a pinned commit and calls its run().
# It is deployed as a CloudFormation stack with the handler inline (Code.ZipFile), because the MCP
# bridge sends parameters as text and corrupts a binary zip (first attempt: "Could not unzip uploaded file").
#
# Creates / uses:
#   IAM role  vb-dataset-ingest-lambda  (made by the first attempt; trusted by lambda.amazonaws.com;
#     inline policy vb-dataset-ingest: objects in vb-data-493280347481 only + its own log group)
#   CloudFormation stack vb-dataset-ingest-lambda -> Lambda vb-dataset-ingest
#     (python3.12, 1024 MB, 2048 MB /tmp, 900 s timeout)
# Then invokes it once, synchronously, and returns the per-dataset report.
#
# CLI equivalent, for reference:
#   aws iam create-role --role-name vb-dataset-ingest-lambda --assume-role-policy-document file://trust.json
#   aws iam put-role-policy --role-name vb-dataset-ingest-lambda --policy-name vb-dataset-ingest --policy-document file://policy.json
#   aws cloudformation deploy --stack-name vb-dataset-ingest-lambda --template-file template.yaml --region us-east-1
#   aws lambda invoke --function-name vb-dataset-ingest --cli-read-timeout 900 \
#     --payload '{"datasets":["ovdata","pydatavolley_dvw"],"ref":"<commit>"}' --cli-binary-format raw-in-base64-out out.json
ACCOUNT, REGION, B = "493280347481", "us-east-1", "vb-data-493280347481"
ROLE, FN, STACK = "vb-dataset-ingest-lambda", "vb-dataset-ingest", "vb-dataset-ingest-lambda"
REPO_REF = "04f823f6dfea73f7f17a60a37b4f6b4e19d90693"  # pinned commit of nadurthi/computervision
DATASETS = ["ovdata", "pydatavolley_dvw"]             # small open sets for the first run

trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
          "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}}}]}
policy = {"Version": "2012-10-17", "Statement": [
    {"Sid": "BucketList", "Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"], "Resource": f"arn:aws:s3:::{B}"},
    {"Sid": "BucketObjects", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
     "Resource": f"arn:aws:s3:::{B}/*"},
    {"Sid": "Logs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
     "Resource": [f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/lambda/{FN}", f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/lambda/{FN}:*"]}]}

TEMPLATE = f"""AWSTemplateFormatVersion: '2010-09-09'
Description: Lambda that downloads public volleyball datasets into s3://{B}/datasets (volleyball project)
Resources:
  IngestFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: {FN}
      Runtime: python3.12
      Handler: index.handler
      Role: arn:aws:iam::{ACCOUNT}:role/{ROLE}
      Timeout: 900
      MemorySize: 1024
      EphemeralStorage:
        Size: 2048
      Environment:
        Variables:
          BUCKET: {B}
      Tags:
        - Key: project
          Value: volleyball
      Code:
        ZipFile: |
          import hashlib, importlib.util, os, sys, urllib.request

          def handler(event, context):
              ref = event.get("ref", "volleyball-pipeline")
              url = ("https://raw.githubusercontent.com/nadurthi/computervision/"
                     + ref + "/volleyball/scripts/datasets_to_s3.py")
              src = urllib.request.urlopen(url, timeout=60).read()
              path = "/tmp/datasets_to_s3.py"
              with open(path, "wb") as f:
                  f.write(src)
              spec = importlib.util.spec_from_file_location("datasets_to_s3", path)
              mod = importlib.util.module_from_spec(spec)
              sys.modules["datasets_to_s3"] = mod
              spec.loader.exec_module(mod)
              dest = "s3://" + os.environ["BUCKET"] + "/" + event.get("prefix", "datasets")
              report = mod.run(event["datasets"], dest, force=event.get("force", False), scratch="/tmp")
              return {{"report": report, "dest": dest, "ref": ref,
                      "script_sha256": hashlib.sha256(src).hexdigest()}}
Outputs:
  FunctionArn:
    Value: !GetAtt IngestFunction.Arn
"""

steps = {}
try:
    r = await call_boto3(service_name="iam", operation_name="CreateRole", params={
        "RoleName": ROLE, "AssumeRolePolicyDocument": json.dumps(trust),
        "Description": "Lambda: download small public datasets into s3://vb-data-493280347481 (volleyball project)"})
    steps["role"] = r["Role"]["Arn"]
except Exception as e:                      # ClientError is not importable in the MCP sandbox
    if "EntityAlreadyExists" not in str(e):
        raise
    steps["role"] = "already existed"
await call_boto3(service_name="iam", operation_name="PutRolePolicy", params={
    "RoleName": ROLE, "PolicyName": "vb-dataset-ingest", "PolicyDocument": json.dumps(policy)})

try:
    s = await call_boto3(service_name="cloudformation", operation_name="CreateStack", region_name=REGION, params={
        "StackName": STACK, "TemplateBody": TEMPLATE, "Tags": [{"Key": "project", "Value": "volleyball"}]})
    steps["stack"] = s["StackId"]
except Exception as e:
    if "AlreadyExistsException" not in str(e):
        raise
    steps["stack"] = "already existed"

status = None
for _ in range(60):                          # wait for CREATE_COMPLETE (a role created seconds ago can need a retry inside CFN)
    d = await call_boto3(service_name="cloudformation", operation_name="DescribeStacks", region_name=REGION,
                         params={"StackName": STACK})
    status = d["Stacks"][0]["StackStatus"]
    if not status.endswith("IN_PROGRESS"):
        break
    await asyncio.sleep(5)
steps["stack_status"] = status
if status not in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
    ev = await call_boto3(service_name="cloudformation", operation_name="DescribeStackEvents", region_name=REGION,
                          params={"StackName": STACK})
    steps["stack_events"] = [(e["LogicalResourceId"], e["ResourceStatus"], e.get("ResourceStatusReason"))
                             for e in ev["StackEvents"][:8]]
    result = steps
else:
    inv = await call_boto3(service_name="lambda", operation_name="Invoke", region_name=REGION, params={
        "FunctionName": FN, "InvocationType": "RequestResponse",
        "Payload": json.dumps({"datasets": DATASETS, "ref": REPO_REF})})
    payload = inv["Payload"]
    payload = payload.read() if hasattr(payload, "read") else payload
    payload = payload.decode() if isinstance(payload, (bytes, bytearray)) else payload
    steps["status_code"] = inv.get("StatusCode")
    steps["function_error"] = inv.get("FunctionError")
    steps["result"] = json.loads(payload) if isinstance(payload, str) and payload.startswith(("{", "[")) else payload
    result = steps
result
