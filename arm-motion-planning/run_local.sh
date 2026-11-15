#!/usr/bin/env bash
# Build the image and run the whole simulation; results go to ./results/<UTC timestamp>/.
#   ./run_local.sh                 # everything: OctoMap + FCL + Informed RRT*, MoveIt 2, videos
#   ./run_local.sh fcl --quick     # one stage, extra arguments are passed to the Python script
#   STAGE args are: all | fcl | moveit | render | shell   (see entrypoint.sh)
# Extra docker build arguments (e.g. a proxy) can be passed with DOCKER_BUILD_ARGS.
# The same script runs on AWS (aws/launch_ec2.py runs it on the instance).
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=${IMAGE:-arm-motion-planning}
RUN=${RUN:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT="$PWD/results/$RUN"
mkdir -p "$OUT"
docker build ${DOCKER_BUILD_ARGS:-} -t "$IMAGE" .
docker run --rm ${DOCKER_RUN_ARGS:-} -v "$OUT:/work/results" "$IMAGE" "${@:-all}" 2>&1 | tee "$OUT/run.log"
echo "results in $OUT"
