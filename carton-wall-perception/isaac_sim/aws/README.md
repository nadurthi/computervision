# Run the Isaac Sim scripts on AWS (g6.2xlarge, us-east-1)

Two files:

| File | Runs where | What it does |
|---|---|---|
| `launch_isaac_ec2.py` | your machine or AWS CloudShell (`pip install boto3`) | creates an IAM role, a private results bucket and a no-inbound security group, then launches a **g6.2xlarge** (NVIDIA L4 24 GB, 8 vCPU, 32 GB RAM) from the AWS Deep Learning Base GPU AMI (Ubuntu 22.04) with a 150 GB disk. Also `status`, `fetch`, `stop`, `terminate`, `cleanup`. |
| `run_on_ec2.sh` | on the instance (started by the user data) | clones this repo, downloads the NVIDIA SimReady boxes/conveyor, pulls `nvcr.io/nvidia/isaac-sim:5.1.0`, runs `box_wall_capture.py` and `unload_cell.py` headless in the container, syncs outputs + logs to S3 |

```bash
pip install boto3
python launch_isaac_ec2.py launch            # ~30-60 min for the first run (image pull + shader cache)
python launch_isaac_ec2.py status            # state + last log lines (via SSM, no SSH)
python launch_isaac_ec2.py fetch ./results   # latest run from s3://isaac-boxwall-<account>-us-east-1/runs/
python launch_isaac_ec2.py cleanup           # terminate + delete role/profile/security group (bucket kept)
```

The instance shuts itself down when the run finishes (`--no-auto-stop` to keep it up); it can be restarted and
re-run with `python launch_isaac_ec2.py launch` then `rerun`. A shell on it: `aws ssm start-session --target <instance-id>`.

**Before the first launch:** the EC2 quota *Running On-Demand G and VT instances* in us-east-1 must be at least
8 vCPUs (new accounts have 0; request it in the Service Quotas console).

**Cost** (us-east-1, Nov 2026): g6.2xlarge $0.9776/h on-demand (spot about $0.76-0.89/h with `--spot`);
150 GB gp3 about $12/month while the disk exists (stopped instances keep their disk; `terminate` removes it).
