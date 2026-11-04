# Run the benchmark on AWS (us-east-1, g6.2xlarge)

```bash
cd multi-view-stereo/aws
./launch_mvs_ec2.sh quota            # quota must be >= 8 vCPUs ("Running On-Demand G and VT instances")
./launch_mvs_ec2.sh launch           # PROFILE=quick by default
./launch_mvs_ec2.sh status           # state, GPU use, last 30 log lines (SSM, no SSH)
./launch_mvs_ec2.sh fetch ./mvs_results    # -> ./mvs_results/quick/RESULTS.md, figures/, jsonl
PROFILE=full ./launch_mvs_ec2.sh rerun     # later: same disk (envs + data kept), resumes unfinished methods
./launch_mvs_ec2.sh cleanup          # terminate + delete role/profile/SG; the bucket (cache + results) is kept
```

| File | Runs where | What it does |
|---|---|---|
| `launch_mvs_ec2.sh` | your machine / CloudShell (AWS CLI v2) | creates the role, bucket, security group; launches, reruns, reports, fetches, cleans up |
| `bootstrap.sh` | instance | installs the auto-stop watchdog, waits for the GPU, runs `run_on_ec2.sh`, stops the instance |
| `mvs-watchdog.sh` | instance (systemd) | stops the instance when idle or after `MAX_HOURS`, on every boot |
| `run_on_ec2.sh` | instance | `setup/install_models.sh` → `setup/download_datasets.sh` → `scripts/run_all.sh` → S3 |

What `launch` creates (tagged `Project=mvs-bench`): IAM role + instance profile `mvs-bench` (SSM + read/write on the
bucket only), private bucket `mvs-bench-<account>-us-east-1` (`cache/` = mirrored datasets and weights,
`results/<profile>/`, `logs/`), a security group with **no inbound rules**, and a **g6.2xlarge** (L4 24 GB, 8 vCPU,
32 GB RAM) from the Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 22.04) with a 500 GB gp3 disk.

## Automatic stop (always on)

1. when the run finishes, success or failure;
2. after 20 idle minutes (no run in progress, GPU < 5 %, CPU load < 1);
3. after `MAX_HOURS` (default 8) of uptime per boot, whatever is running.

Each method also has a `METHOD_TIMEOUT` (default 3 h). (2) and (3) are a systemd service, so they also cover
restarts and reruns; the user data schedules a hard shutdown before doing anything else.

## Timing and cost (estimate, us-east-1, Nov 2026)

| Phase | quick | full |
|---|---|---|
| environments + model repos + ACMMP build | ~35 min (first time only) | same |
| datasets | ~25 GB, ~20-40 min | ~110 GB, ~1.5-2 h |
| benchmark | ~2-3 h | ~10 h (split over two `rerun`s with MAX_HOURS=8) |

g6.2xlarge on-demand is about $0.98/h, so a quick run is roughly $4-5; the 500 GB disk is ~$40/month while it exists
(`terminate` removes it; the S3 cache makes the next setup fast).

## Google Drive

Some weights (IGEV, FoundationStereo) and datasets (DTU test set, BlendedMVS) are only on Google Drive. They go through
`gdown` with retries, and every successful download is mirrored to `s3://mvs-bench-<account>-us-east-1/cache/`, so it
only has to work once. If Drive refuses (quota), the log says which file to download in a browser and where to put it
in the bucket; the benchmark skips whatever is missing and the next `rerun` picks it up.

## Subsets

```bash
METHODS="patchmatchnet acmmp stereo_foundation_stereo" ONLY=mvs ./launch_mvs_ec2.sh rerun
```
