#!/usr/bin/env bash
# Activates the ROS/conda env and runs the requested stage(s).
#   all       standalone (OctoMap + FCL + Informed RRT*) then MoveIt, then videos
#   fcl       standalone pipeline only
#   moveit    MoveIt 2 pipeline only (needs results/scene.npz from 'fcl')
#   render    re-render videos from saved results
#   shell     interactive shell in the env
set -eo pipefail
eval "$(micromamba shell hook -s bash)"
micromamba activate ros
export ROS_LOG_DIR=/tmp/roslog
OUT=${OUT_DIR:-/work/results}
mkdir -p "$OUT"
stage=${1:-all}; shift || true
case "$stage" in
  fcl)    python -u /work/sim/plan_fcl_ompl.py --out "$OUT" "$@" ;;
  moveit) python -u /work/sim/plan_moveit.py   --out "$OUT" "$@" ;;
  render) python -u /work/sim/render.py        --out "$OUT" "$@" ;;
  all)    python -u /work/sim/plan_fcl_ompl.py --out "$OUT" "$@"
          python -u /work/sim/plan_moveit.py   --out "$OUT"
          python -u /work/sim/render.py        --out "$OUT" ;;
  shell)  exec bash ;;
  *)      exec "$stage" "$@" ;;
esac
