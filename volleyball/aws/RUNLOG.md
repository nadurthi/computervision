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
| 4a | 2026-11-02 02:45 | `04_lambda_small_ingest.py` (first version) | role `vb-dataset-ingest-lambda` + inline policy created; CreateFunction failed 8 times: "Could not unzip uploaded file". The MCP bridge sends parameters as text, so the hand-built zip's bytes above 0x7F were replaced. No function created. Script rewritten to deploy through CloudFormation with the handler inline |
| 4b | 2026-11-02 02:51 | `04_lambda_small_ingest.py` | stack `vb-dataset-ingest-lambda` CREATE_COMPLETE → function `vb-dataset-ingest`; invoked with `ovdata pydatavolley_dvw` at commit `04f823f`: **ovdata 8 files 5.2 MB, pydatavolley_dvw 7 files 0.65 MB**, 3.3 s |
| 6a | 2026-11-02 02:53 | `06_verify_datasets.py` | 17 objects, every manifest entry present with the right size; the 15 files' sha256 match the GitHub originals byte for byte (checked against local clones) |
| 3b | 2026-11-02 02:54 | read-only quota check (no script) | both CodeBuild requests moved to `CASE_OPENED` (support cases `179358750000706`, `179358750000049`); applied quota still 0 |
| 7 | 2026-11-02 02:56 | `07_lambda_drive_ingest.py` | 4 async invocations at commit `55ccaa5` (direct Drive streaming): **volleyball_ibrahim_ball 1.7 MB, volleyball_ibrahim_tracks 13.3 MB, jersey_str_lmdb 19.3 MB, parseq_jersey_weights 858.6 MB (→ models/)**, each under 25 s |
| 6b | 2026-11-02 03:00 | `06_verify_datasets.py` (now covers models/ and zip headers) | 27 objects, 898.8 MB; no missing files, no size mismatches; every .zip/.ckpt starts with a zip header |

## Resources created

| Resource | Name | Notes |
|---|---|---|
| S3 bucket | `vb-data-493280347481` | us-east-1, private, versioned |
| IAM role | `vb-dataset-ingest-codebuild` | trusted by CodeBuild; S3 objects in the bucket + its log group only |
| CodeBuild project | `vb-dataset-ingest` | idle; billed only while a build runs |
| Service Quotas requests | CodeBuild Linux/Medium, Linux/Large → 1 | pending |
| IAM role | `vb-dataset-ingest-lambda` | trusted by Lambda; S3 objects in the bucket + its log group only |
| CloudFormation stack | `vb-dataset-ingest-lambda` | holds the Lambda function |
| Lambda function | `vb-dataset-ingest` | idle; billed per invocation |

## In the bucket now

| Prefix | Files | Size | Licence |
|---|---|---|---|
| `datasets/ovdata/` | 8 | 5.2 MB | MIT |
| `datasets/pydatavolley_dvw/` | 7 | 0.65 MB | MIT |
| `datasets/volleyball_ibrahim_ball/` | 1 zip | 1.7 MB | not stated; cite Perez et al. |
| `datasets/volleyball_ibrahim_tracks/` | 1 zip | 13.3 MB | not stated; cite Sendo and Ukita |
| `datasets/jersey_str_lmdb/` | 1 zip | 19.3 MB | CC BY-NC (non-commercial) |
| `models/parseq_jersey_weights/` | 3 ckpt | 858.6 MB | CC BY-NC (non-commercial) |

## Still to fetch

| Dataset | Needs | How |
|---|---|---|
| `volleyball_ibrahim` (the Volleyball dataset frames) | a Drive *folder* (gdown) and tens of GB of disk: too big for Lambda | `05_start_codebuild_ingest.py` with `BUILD_GENERAL1_LARGE` once the CodeBuild quota is approved |
| `soccernet_jersey` | the SoccerNet downloader package | `05_start_codebuild_ingest.py` once the quota is approved |
| `roboflow_*` | a Roboflow API key | add the key, then Lambda or CodeBuild |
| `sportsmot` | the Codalab "Get Data" link | pass it as `--url` |

## To remove everything later

Delete the CloudFormation stack (removes the Lambda function) and the CodeBuild project, delete the two roles (their inline policies first), and
empty then delete the bucket (all versions). None of these resources cost anything while idle except the
data stored in S3.
