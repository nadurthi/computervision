# AWS run log — volleyball datasets

Every AWS change for this project is a script in this folder, run as-is through the AWS MCP server's
`run_script` (an async boto3 sandbox). Each entry: when, which script, what it did, the result.
Account `493280347481`, region `us-east-1` for all data and compute. Signed in as the account root user.

| # | When (UTC) | Script | Result |
|---|---|---|---|
| 0 | 2026-11-02 02:25 | read-only check (no script) | no SageMaker domains in us-east-1/us-east-2/us-west-2; existing buckets `sagemaker-studio-493280347481-8r8pgd2yo7o`, `sagemaker-us-east-1-493280347481` (both us-east-1); GPU Studio quotas 0 in us-east-1 and us-east-2 |
| 1 | 2026-11-02 02:26 | `01_create_bucket.py` | bucket `vb-data-493280347481` created in us-east-1: public access blocked, versioning on, lifecycle (abort partial uploads 7 d, old versions 30 d), SSE-S3 (AES256) |
| 2a | 2026-11-02 02:33 | `02_codebuild_ingest_project.py` | role `vb-dataset-ingest-codebuild` + inline policy created; CreateProject failed (new role not yet assumable) and the retry loop crashed: `ClientError` is not defined in the sandbox. Script fixed to catch `Exception` |
| 2b | 2026-11-02 02:35 | `02_codebuild_ingest_project.py` (re-run) | CodeBuild project `vb-dataset-ingest` created (MEDIUM, 8 h timeout, logs `/aws/codebuild/vb-dataset-ingest`) |
| 2c | 2026-11-02 02:36 | `05_start_codebuild_ingest.py` (then numbered 03) | StartBuild refused: `AccountLimitExceededException: Cannot have more than 0 builds in queue for the account`. Every CodeBuild concurrency quota in this account is 0 (AWS default 1). No build ran |
| 3 | 2026-11-02 02:38 | `03_request_codebuild_quota.py` | quota increase requested to 1: Linux/Medium (`L-2DC20C30`, request `62012f2598654dc5af12f80257660ea6y14w3VY7`) and Linux/Large (`L-4DDC4A99`, request `e0b32a0dd12a409493e8bbee37f525b8KNNnrpM7`); both PENDING |
| 4 | 2026-11-02 02:45 | `04_lambda_small_ingest.py` | (running) role `vb-dataset-ingest-lambda`, function `vb-dataset-ingest`, invoke with `ovdata pydatavolley_dvw` at repo commit `04f823f` |

## Resources created

| Resource | Name | Notes |
|---|---|---|
| S3 bucket | `vb-data-493280347481` | us-east-1, private, versioned |
| IAM role | `vb-dataset-ingest-codebuild` | trusted by CodeBuild; S3 objects in the bucket + its log group only |
| CodeBuild project | `vb-dataset-ingest` | idle; billed only while a build runs |
| Service Quotas requests | CodeBuild Linux/Medium, Linux/Large → 1 | pending |
| IAM role | `vb-dataset-ingest-lambda` | trusted by Lambda; S3 objects in the bucket + its log group only |
| Lambda function | `vb-dataset-ingest` | idle; billed per invocation |

## To remove everything later

Delete the Lambda function and CodeBuild project, delete the two roles (their inline policies first), and
empty then delete the bucket (all versions). None of these resources cost anything while idle except the
data stored in S3.
