#!/usr/bin/env bash
# Run the Isaac Sim box-wall scripts headless in NVIDIA's container on a GPU EC2 instance.
#
# Host: AWS "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)" on g6.2xlarge (L4 24 GB, 32 GB RAM),
# 150 GB gp3 root volume. That AMI already has the NVIDIA driver, Docker and the NVIDIA container toolkit.
#
#   sudo bash run_on_ec2.sh s3://my-bucket/isaac-runs        # outputs and logs are synced there at the end
set -euo pipefail
S3_DEST="${1:-}"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ROOT=/opt/isaac
RUN="run-$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/out/$RUN"

mkdir -p "$ROOT"/cache/{main/ov,main/warp,computecache} "$ROOT"/{config,data/documents,data/Kit,logs,pkg,simready} "$OUT"
nvidia-smi | tee "$OUT/nvidia-smi.txt"

# code + assets
if [ ! -d "$ROOT/computervision" ]; then
  git clone --depth 1 https://github.com/nadurthi/computervision "$ROOT/computervision"
else
  git -C "$ROOT/computervision" pull --ff-only
fi
CODE="$ROOT/computervision/carton-wall-perception/isaac_sim"
python3 -m pip install -q huggingface_hub
python3 "$CODE/download_simready_assets.py" --out "$ROOT/simready" | tee "$OUT/simready_download.txt"

docker pull "$IMAGE"
chown -R 1234:1234 "$ROOT"     # the container runs as uid 1234

isaac() {   # isaac <log-name> <python args...>
  local name="$1"; shift
  docker run --rm --gpus all --network=host --entrypoint bash \
    -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=N" -u 1234:1234 \
    -v "$ROOT/cache/main:/isaac-sim/.cache:rw" \
    -v "$ROOT/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
    -v "$ROOT/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
    -v "$ROOT/config:/isaac-sim/.nvidia-omniverse/config:rw" \
    -v "$ROOT/data:/isaac-sim/.local/share/ov/data:rw" \
    -v "$ROOT/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
    -v "$CODE:/work:ro" -v "$ROOT/simready:/simready:ro" -v "$OUT:/out:rw" \
    "$IMAGE" -c "cd /isaac-sim && ./python.sh $*" 2>&1 | tee "$OUT/$name.log" || true
}

# 1) box wall in the warehouse: RGB, depth, masks, 2D boxes, lidar
isaac box_wall /work/box_wall_capture.py --frames 3 --out /out/box_wall --box-usd-dir /simready
# 2) unloading cell: UR10 + suction + conveyor, camera + lidar before each pick
isaac unload_cell /work/unload_cell.py --simready /simready --picks 5 --record /out/unload_cell

ls -laR "$OUT" > "$OUT/listing.txt"
if [ -n "$S3_DEST" ]; then
  aws s3 sync "$OUT" "$S3_DEST/$RUN/" --only-show-errors
  echo "synced to $S3_DEST/$RUN/"
fi
echo "DONE $RUN"
