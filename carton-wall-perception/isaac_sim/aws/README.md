# Run the Isaac Sim scripts on AWS (g6.2xlarge, us-east-1)

The actual run is `../run_all.sh`, which works on any Linux machine with an NVIDIA RTX GPU (see the main README).
These scripts only start a GPU instance, call that same script there, and copy its output folder to S3.

Two equivalent launchers. Use whichever you prefer; both create the same resources and use the same on-instance
scripts, so you can launch with one and rerun with the other.

| File | Runs where | What it does |
|---|---|---|
| `launch_isaac_ec2.sh` | your machine or AWS CloudShell (AWS CLI v2) | AWS CLI launcher: `launch`, `rerun`, `status`, `fetch`, `terminate`, `cleanup` |
| `launch_isaac_ec2.py` | same (`pip install boto3`) | the same commands in Python |
| `bootstrap.sh` | on the instance | installs the auto-stop watchdog service, waits for the GPU driver, runs `run_on_ec2.sh`, shuts the instance down (which terminates it) |
| `isaac-watchdog.sh` | on the instance (systemd service) | stops the instance when idle or after the uptime limit, on every boot |
| `run_on_ec2.sh` | on the instance | thin wrapper: pulls the latest code, runs `../run_all.sh` (the same script you run locally: assets, Docker image, smoke/lidar tests, both Isaac scripts, `report.html`), syncs the run folder to S3 |

```bash
./launch_isaac_ec2.sh launch              # first session (~30-60 min: image pull + shader cache)
./launch_isaac_ec2.sh status              # state, runs in S3, watchdog state + last log lines (SSM, no SSH)
./launch_isaac_ec2.sh fetch ./results     # latest run from s3://isaac-boxwall-<account>-us-east-1/runs/

./launch_isaac_ec2.sh rerun               # any later session: a fresh instance with the latest code from GitHub
                                          # (same as launch; nothing is kept on EC2 between sessions)

./launch_isaac_ec2.sh cleanup             # terminate + delete role/profile/security group (bucket kept)
```

What `launch` creates (all tagged `Project=isaac-boxwall`): an IAM role and instance profile `isaac-boxwall`
(SSM access + write to the results bucket only), a private S3 bucket `isaac-boxwall-<account>-us-east-1`, a
security group with **no inbound rules** (a shell is `aws ssm start-session --target <instance-id>`), and a
**g6.2xlarge** (NVIDIA L4 24 GB, 8 vCPU, 32 GB RAM) from the AWS Deep Learning Base OSS NVIDIA Driver GPU AMI
(Ubuntu 22.04) with a 150 GB gp3 disk.

## Storage: S3 only

S3 is the only persistent storage. Every shutdown, whatever the cause, **terminates** the instance and deletes
its disk, so no EBS volume or stopped instance is ever left behind. Results, logs and `report.html` are in
`s3://isaac-boxwall-<account>-us-east-1/runs/<run>/`. The price is that each session pulls the Isaac Sim image
and compiles shaders again (about 10 extra minutes).

## Automatic stop (always on, every session)

The instance shuts itself down, and with it is terminated:

1. **when the run finishes**, whether it succeeded or failed (first launch and every `rerun`);
2. **when idle for 20 minutes**: no run in progress, GPU below 5 %, CPU load below 1;
3. **after 4 hours of uptime per boot**, whatever is running.

Each Isaac script also has a **90-minute** limit, after which its container is killed. (2) and (3) are
`isaac-watchdog.sh`, installed by `bootstrap.sh` as a systemd service, so they apply on every boot, including
every boot. On boot the user data also schedules a hard shutdown after 4 hours before doing
anything else, so even a failed `git clone` cannot leave the GPU running.

Settings, as environment variables for the `.sh` launcher or flags for the `.py` one:

| `.sh` | `.py` | default | meaning |
|---|---|---|---|
| `IDLE_MIN` | `--idle-minutes` | 20 | idle minutes before stopping |
| `MAX_HOURS` | `--max-hours` | 4 | max uptime per boot |
| `STEP_TIMEOUT` | `--step-timeout` | 90m | limit per Isaac script |
| `AUTO_STOP=0` | `--no-auto-stop` | on | skip (1) only; (2) and (3) still apply |
| `SPOT=1` | `--spot` | off | spot instance, cheaper but interruptible |

**Driver.** Isaac Sim 5.1 is validated on the NVIDIA **580** production driver. Newer DLAMI builds ship 595, on
which the 5.1 renderer segfaults at startup, so both launchers pin the AMI by name to the 20260619 build
(driver 580.159.04). Override with `AMI_NAME=latest` / `--ami-name latest`. Every run also starts with
`smoke_test.py` (one cube, one camera, one PNG) so a broken GPU stack shows up before the real scripts.

**Before the first launch:** the EC2 quota *Running On-Demand G and VT instances* in us-east-1 must be at least
8 vCPUs (new accounts have 0; request it in the Service Quotas console).

**Cost** (us-east-1, Nov 2026): g6.2xlarge $0.9776/h on-demand (spot about $0.76-0.89/h). The 150 GB gp3 disk
exists only while the instance runs. S3 storage for the runs is a few cents per month.
