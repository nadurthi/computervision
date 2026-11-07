# Interim results — quick profile, 2026-11-06

GPU: NVIDIA A10G (AWS g5.2xlarge). Produced by `bash scripts/run_benchmark.sh quick` (via `aws/launch_mvs_ec2.sh`).
Full outputs (jsonl per method, `summary.json`, `report.html`) are in `s3://mvs-bench-<account>-us-east-1/results/quick/`.

Done: classical MVS (plane sweep mean / best-K / best-K+SGM, our PatchMatch, ACMMP) and learned MVS (PatchmatchNet,
IterMVS DTU/BlendedMVS, MVSAnywhere; Vis-MVSNet on DTU only) on DTU and ETH3D at N = 2, 3, 5, 10.

Not yet run: COLMAP, Robust MVD, Vis-MVSNet on ETH3D, stereo-inside-MVS, the two-view stereo benchmark, the latency
stage, DTU point clouds. Resume with `./aws/launch_mvs_ec2.sh rerun` (or `run_benchmark.sh quick` locally).

Resolutions: DTU classical 800x600, DTU learned 1600x1200, ETH3D 1152x768. `time s` = wall time per reference view
including CUDA sync. Qualitative panels were re-encoded as JPEG to keep the repo small.
