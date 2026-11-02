# Step 05 - start one CodeBuild run of vb-dataset-ingest for a chosen set of datasets.
#
# Edit the three settings below, then run. Each run is one build; it stops by itself when done.
# Dataset names: python scripts/datasets_to_s3.py --list
# CLI equivalent, for reference:
#   aws codebuild start-build --project-name vb-dataset-ingest --region us-east-1 \
#     --compute-type-override BUILD_GENERAL1_MEDIUM \
#     --environment-variables-override name=DATASETS,value="ovdata pydatavolley_dvw",type=PLAINTEXT
DATASETS = "ovdata pydatavolley_dvw"     # run 1: the small open sets (~6 MB)
EXTRA_ARGS = ""                          # e.g. "--extract"
COMPUTE = "BUILD_GENERAL1_MEDIUM"        # BUILD_GENERAL1_LARGE for the big Google Drive sets

b = await call_boto3(service_name="codebuild", operation_name="StartBuild", region_name="us-east-1", params={
    "projectName": "vb-dataset-ingest",
    "computeTypeOverride": COMPUTE,
    "environmentVariablesOverride": [
        {"name": "DATASETS", "value": DATASETS, "type": "PLAINTEXT"},
        {"name": "EXTRA_ARGS", "value": EXTRA_ARGS, "type": "PLAINTEXT"}]})
build = b["build"]
result = {"build_id": build["id"], "status": build["buildStatus"], "datasets": DATASETS, "compute": COMPUTE,
          "start": str(build.get("startTime"))}
result
