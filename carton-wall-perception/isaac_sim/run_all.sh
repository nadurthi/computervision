#!/usr/bin/env bash
# Run every Isaac Sim script for the box-wall project and build an HTML report. Works on any Linux machine with
# an NVIDIA RTX GPU: your workstation, a lab server or a cloud VM. Nothing here knows about AWS; the AWS scripts
# in aws/ only call this file and copy its output folder to S3.
#
#   bash run_all.sh                         # everything, output in ./isaac_runs/run-<UTC time>/
#   OUT=~/runs/test1 bash run_all.sh        # choose the output folder
#   RUN_SCRIPTS=0 bash run_all.sh           # only the smoke and lidar tests
#   MODE=native ISAAC_SIM_PATH=~/isaacsim bash run_all.sh   # a local Isaac Sim install instead of Docker
#
# Requirements
#   MODE=docker (default): Docker + NVIDIA Container Toolkit, NVIDIA driver 580 for Isaac Sim 5.1
#                          (the 595 driver crashes the 5.1 renderer at startup), ~25 GB for the image.
#   MODE=native:           an Isaac Sim 5.x install; ISAAC_SIM_PATH points at the folder with python.sh.
#   Both: python3 with numpy + Pillow (matplotlib optional) for the report; huggingface_hub to fetch assets.
#
# Settings (environment variables)
#   OUT=./isaac_runs/run-<time>   output folder           ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
#   DATA=~/.cache/isaac-boxwall   assets + Isaac caches    STEP_TIMEOUT=90m  limit per script (smoke: 10m)
#   RUN_SCRIPTS=1                 0 = tests only           EXTRA_ARGS=''     extra args for both main scripts
#   FRAMES=3  PICKS=5             workload size            SKIP_ASSETS=0     1 = do not download SimReady assets
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-docker}"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
DATA="${DATA:-$HOME/.cache/isaac-boxwall}"
OUT="${OUT:-$PWD/isaac_runs/run-$(date -u +%Y%m%dT%H%M%SZ)}"
STEP_TIMEOUT="${STEP_TIMEOUT:-90m}"
FRAMES="${FRAMES:-3}"; PICKS="${PICKS:-5}"
mkdir -p "$OUT" "$DATA"/{simready,kitdata,cache/main/ov,cache/main/warp,cache/computecache,config,data/documents,data/Kit,logs,pkg}
OUT="$(cd "$OUT" && pwd)"
echo "[run_all] mode=$MODE out=$OUT data=$DATA"
{ nvidia-smi || echo "nvidia-smi not available"; } > "$OUT/nvidia-smi.txt" 2>&1
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
case "$IMAGE:$DRIVER" in
  *isaac-sim:5.*:580.*|*:) ;;
  *isaac-sim:5.*) echo "[run_all] WARNING: Isaac Sim 5.x is validated on the 580 driver; $DRIVER may crash the renderer" ;;
esac

# ---- assets (NVIDIA SimReady boxes + conveyor from Hugging Face)
if [ "${SKIP_ASSETS:-0}" != "1" ]; then
  python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install -q --user huggingface_hub 2>/dev/null \
    || python3 -m pip install -q --break-system-packages huggingface_hub
  python3 "$HERE/download_simready_assets.py" --out "$DATA/simready" > "$OUT/simready_download.txt" 2>&1 \
    || echo "[run_all] SimReady download failed; the scripts fall back to Isaac or procedural boxes"
fi

if [ "$MODE" = "docker" ]; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
  chmod -R a+rwX "$DATA" "$OUT" 2>/dev/null || true     # the container runs as its own uid (1234)
fi

isaac() {   # isaac <step-name> <script> <args...>   (runs one Isaac Python script, log in $OUT/<step>.log)
  local name="$1" script="$2"; shift 2
  local t0; t0=$(date +%s)
  echo "[run_all] step $name: $script $*"
  if [ "$MODE" = "docker" ]; then
    timeout --kill-after=2m "$STEP_TIMEOUT" \
    docker run --rm --name "isaac-$name" --gpus all --network=host --entrypoint bash \
      -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=N" -e "PYTHONUNBUFFERED=1" -u 1234:1234 \
      -v "$DATA/cache/main:/isaac-sim/.cache:rw" \
      -v "$DATA/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
      -v "$DATA/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
      -v "$DATA/config:/isaac-sim/.nvidia-omniverse/config:rw" \
      -v "$DATA/data:/isaac-sim/.local/share/ov/data:rw" \
      -v "$DATA/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
      -v "$DATA/kitdata:/isaac-sim/kit/data:rw" \
      -v "$HERE:/work:ro" -v "$DATA/simready:/simready:ro" -v "$OUT:/out:rw" \
      "$IMAGE" -c "cd /isaac-sim && ./python.sh /work/$script $*" > "$OUT/$name.log" 2>&1
    docker rm -f "isaac-$name" >/dev/null 2>&1 || true   # timeout only kills the client
    local crash_dir="$DATA/kitdata"
  else
    local py="${ISAAC_SIM_PATH:?set ISAAC_SIM_PATH to your Isaac Sim folder}/python.sh"
    # same paths as in the container: /work, /out and /simready become the real folders
    local args="${*//\/out/$OUT}"; args="${args//\/simready/$DATA/simready}"
    PYTHONUNBUFFERED=1 timeout --kill-after=2m "$STEP_TIMEOUT" "$py" "$HERE/$script" $args > "$OUT/$name.log" 2>&1
    local crash_dir="${ISAAC_SIM_PATH}/kit/data"
  fi
  local rc=$?
  echo "$name exit=$rc seconds=$(( $(date +%s) - t0 ))" >> "$OUT/steps.txt"
  mkdir -p "$OUT/crash_$name"    # crash reports (py-spy stack + native backtrace) written during this step
  find "$crash_dir" -name '*.txt' -newermt "@$t0" -exec cp {} "$OUT/crash_$name/" \; 2>/dev/null || true
  rmdir "$OUT/crash_$name" 2>/dev/null || true
  echo "[run_all] step $name done (exit $rc)"
}

# ---- 0) environment checks: rendering with and without RTX stable ids, and the ways of reading the lidar
STEP_TIMEOUT=10m isaac smoke smoke_test.py --out /out/smoke
STEP_TIMEOUT=10m isaac smoke_ids smoke_test.py --out /out/smoke_ids --stable-ids
STEP_TIMEOUT=10m isaac lidar_test lidar_test.py --out /out/lidar_test

# ---- 1) box wall in the warehouse (RGB, depth, masks, 2D boxes, lidar)  2) unloading cell (UR10 + conveyor)
if [ "${RUN_SCRIPTS:-1}" = "1" ]; then
  isaac box_wall box_wall_capture.py --frames "$FRAMES" --out /out/box_wall --box-usd-dir /simready ${EXTRA_ARGS:-}
  isaac unload_cell unload_cell.py --simready /simready --picks "$PICKS" --record /out/unload_cell ${EXTRA_ARGS:-}
fi

ls -laR "$OUT" > "$OUT/listing.txt"
python3 "$HERE/make_report.py" "$OUT" || echo "[run_all] report failed"
echo "[run_all] DONE $OUT  (report: $OUT/report.html)"
