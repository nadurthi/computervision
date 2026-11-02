# Step 07 - fetch the single-file Google Drive datasets with the Lambda function from step 04.
#
# Each dataset is its own asynchronous invocation, so each gets the full 15 minutes. Files stream
# from drive.usercontent.google.com straight into S3 (datasets_to_s3.py --direct path; the Lambda
# runtime has no gdown, so the script picks it automatically). Nothing is extracted: archives are
# stored as downloaded. Check the result with step 06 and the log group /aws/lambda/vb-dataset-ingest.
#
#   volleyball_ibrahim_ball    ball positions on the Volleyball dataset frames (Perez et al.)
#   volleyball_ibrahim_tracks  manual player trajectories (Sendo and Ukita)
#   jersey_str_lmdb            weakly labelled jersey-number crops, LMDB (non-commercial)
#   parseq_jersey_weights      PARSeq weights: original, hockey, SoccerNet (non-commercial) -> models/
#
# Async invocations are retried by Lambda up to twice on error; re-runs skip files already in S3.
# CLI equivalent, for reference:
#   aws lambda invoke --function-name vb-dataset-ingest --invocation-type Event --region us-east-1 \
#     --cli-binary-format raw-in-base64-out --payload '{"datasets":["volleyball_ibrahim_ball"],"ref":"<commit>"}' /dev/null
REGION, FN = "us-east-1", "vb-dataset-ingest"
REPO_REF = "55ccaa59e3070dca14cd27b17af7363837cdf09e"  # commit with the --direct Drive path
JOBS = [
    {"datasets": ["volleyball_ibrahim_ball"], "prefix": "datasets"},
    {"datasets": ["volleyball_ibrahim_tracks"], "prefix": "datasets"},
    {"datasets": ["jersey_str_lmdb"], "prefix": "datasets"},
    {"datasets": ["parseq_jersey_weights"], "prefix": "models"},
]
out = []
for job in JOBS:
    r = await call_boto3(service_name="lambda", operation_name="Invoke", region_name=REGION, params={
        "FunctionName": FN, "InvocationType": "Event", "Payload": json.dumps({**job, "ref": REPO_REF})})
    out.append({"datasets": job["datasets"], "prefix": job["prefix"], "status_code": r.get("StatusCode")})
result = {"queued": out, "ref": REPO_REF}
result
