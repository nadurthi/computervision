# Run the Isaac Sim scripts on AWS (g6.2xlarge, us-east-1)

Two equivalent launchers. Use whichever you prefer; both create the same resources and use the same on-instance
scripts, so you can launch with one and rerun with the other.

| File | Runs where | What it does |
|---|---|---|
| `launch_isaac_ec2.sh` | your machine or AWS CloudShell (AWS CLI v2) | AWS CLI launcher: `launch`, `rerun`, `status`, `fetch`, `stop`, `terminate`, `cleanup` |
| `launch_isaac_ec2.py` | same (`pip install boto3`) | the same commands in Python |
| `bootstrap.sh` | on the instance | installs the auto-stop watchdog service, waits for the GPU driver, runs `run_on_ec2.sh`, stops the instance |
| `isaac-watchdog.sh` | on the instance (systemd service) | stops the instance when idle or after the uptime limit, on every boot |
| `run_on_ec2.sh` | on the instance | downloads the NVIDIA SimReady assets, pulls `nvcr.io/nvidia/isaac-sim:5.1.0`, runs `box_wall_capture.py` and `unload_cell.py` headless in the container, syncs outputs and logs to S3 |

```bash
./launch_isaac_ec2.sh launch              # first session (~30-60 min: image pull + shader cache)
./launch_isaac_ec2.sh status              # state, runs in S3, watchdog state + last log lines (SSM, no SSH)
./launch_isaac_ec2.sh fetch ./results     # latest run from s3://isaac-boxwall-<account>-us-east-1/runs/

./launch_isaac_ec2.sh rerun               # any later session: starts the stopped instance, pulls the latest
                                          # code from GitHub, runs again, stops again

./launch_isaac_ec2.sh cleanup             # terminate + delete role/profile/security group (bucket kept)
```

What `launch` creates (all tagged `Project=isaac-boxwall`): an IAM role and instance profile `isaac-boxwall`
(SSM access + write to the results bucket only), a private S3 bucket `isaac-boxwall-<account>-us-east-1`, a
security group with **no inbound rules** (a shell is `aws ssm start-session --target <instance-id>`), and a
**g6.2xlarge** (NVIDIA L4 24 GB, 8 vCPU, 32 GB RAM) from the AWS Deep Learning Base OSS NVIDIA Driver GPU AMI
(Ubuntu 22.04) with a 150 GB gp3 disk.

## Automatic stop (always on, every session)

The instance stops itself:

1. **when the run finishes**, whether it succeeded or failed (first launch and every `rerun`);
2. **when idle for 20 minutes**: no run in progress, GPU below 5 %, CPU load below 1;
3. **after 4 hours of uptime per boot**, whatever is running.

Each Isaac script also has a **90-minute** limit, after which its container is killed. (2) and (3) are
`isaac-watchdog.sh`, installed by `bootstrap.sh` as a systemd service, so they apply on every boot, including
restarts and reruns. On the first boot the user data also schedules a hard shutdown after 4 hours before doing
anything else, so even a failed `git clone` cannot leave the GPU running.

Settings, as environment variables for the `.sh` launcher or flags for the `.py` one:

| `.sh` | `.py` | default | meaning |
|---|---|---|---|
| `IDLE_MIN` | `--idle-minutes` | 20 | idle minutes before stopping |
| `MAX_HOURS` | `--max-hours` | 4 | max uptime per boot |
| `STEP_TIMEOUT` | `--step-timeout` | 90m | limit per Isaac script |
| `AUTO_STOP=0` | `--no-auto-stop` | on | skip (1) only; (2) and (3) still apply |
| `SPOT=1` | `--spot` | off | spot instance, cheaper but interruptible |

**Before the first launch:** the EC2 quota *Running On-Demand G and VT instances* in us-east-1 must be at least
8 vCPUs (new accounts have 0; request it in the Service Quotas console).

**Cost** (us-east-1, Nov 2026): g6.2xlarge $0.9776/h on-demand (spot about $0.76-0.89/h); 150 GB gp3 about
$12/month while the disk exists (a stopped instance keeps its disk; `terminate` or `cleanup` removes it).
