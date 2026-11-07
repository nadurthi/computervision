# Multi-view stereo vs. two-view stereo: classical vs. learned, with an occlusion breakdown

Depth from **calibrated** cameras (intrinsics + extrinsics known) on real benchmarks, with the same images, cameras and
GT for every method:

* **two-view stereo**: OpenCV SGBM, RAFT-Stereo, IGEV-Stereo, FoundationStereo, MonSter on Middlebury, ETH3D, KITTI 2015
* **classical MVS**: plane sweep (mean / best-K occlusion-robust aggregation, + SGM), our PatchMatch MVS,
  COLMAP PatchMatch, ACMMP
* **learned MVS**: PatchmatchNet, IterMVS (DTU / BlendedMVS weights), Vis-MVSNet, Robust MVD, MVSAnywhere
* **stereo inside MVS**: every stereo matcher also runs on the reference + its best source view (rectified from the
  known calibration), so stereo and MVS are scored on the same pixels

The key measurement: GT depth exists for **every** view, so each reference pixel is labelled *visible / occluded /
outside* in the stereo partner, and *rescued* if another source sees it. Accuracy per region vs. number of views N
shows exactly where extra views help.

**Write-up:** [perceptionstack.dev — Multi-view stereo](https://perceptionstack.dev/tutorials/applications/multi-view-stereo/)
(theory, algorithms, published numbers, our results) · **Datasets:** [`docs/DATASETS.md`](docs/DATASETS.md) ·
**AWS:** [`aws/README.md`](aws/README.md)

## Run it

One script runs everything (environments + models, datasets, all benchmarks, `RESULTS.md` + `report.html`). It is
the same script locally and on AWS; the AWS runner only syncs folders with S3 around it.

**Locally**, on Ubuntu 22.04 with an NVIDIA GPU and the CUDA toolkit:

```bash
bash scripts/run_benchmark.sh quick /opt/mvs        # or: full   (idempotent; re-run to resume)
xdg-open /opt/mvs/results/quick/report.html
```

Useful variables: `METHODS="patchmatchnet acmmp"` (subset), `ONLY=mvs|stereo|pointcloud|latency` (one stage),
`RESET="colmap"` (recompute a method), `SKIP_SETUP=1` (data and models already in place),
`CACHE_DIR=/big/disk/cache` (keep downloaded archives and weights), `METHOD_TIMEOUT=3h`. To rebuild only the report:
`python scripts/make_report.py --results R && python scripts/make_figures.py --results R --data-root D --config C && python scripts/make_html_report.py --results R`.

**No GPU, no downloads** (checks the whole pipeline in ~5 minutes on a laptop CPU):

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                                   # unit checks: geometry, rectification, methods
python scripts/make_synthetic_data.py /tmp/mvs_smoke/data
SKIP_SETUP=1 DEVICE=cpu MODELS_PYTHON=python3 bash scripts/run_benchmark.sh smoke /tmp/mvs_smoke
```

**On AWS** (disposable GPU instance that terminates itself; S3 is the only persistent storage: dataset/weight cache and results): see [`aws/README.md`](aws/README.md).

```bash
cd aws && ./launch_mvs_ec2.sh launch      # then: status, fetch ./mvs_results, rerun, terminate, cleanup
```

## Layout

| Path | What |
|---|---|
| `mvsbench/geometry.py` | projection, per-source visibility labels, `RectifiedPair` (any baseline direction → left/right for a stereo net, disparity → reference depth) |
| `mvsbench/metrics.py` | disparity (EPE, bad-τ, D1) and depth (AbsRel, inliers, within-τ) metrics; missing predictions count as errors |
| `mvsbench/data/stereo.py` | MiddEval3, ETH3D two-view, KITTI 2015, Middlebury 2014 |
| `mvsbench/data/mvs.py` | DTU, ETH3D high-res (MVSNet view selection from SfM points), BlendedMVS |
| `mvsbench/mvs/planesweep.py` | ZNCC plane sweep, mean vs best-K aggregation, 4-path SGM |
| `mvsbench/mvs/patchmatch.py` | slanted-plane PatchMatch MVS, bilateral NCC, top-K view selection, red/black propagation |
| `mvsbench/mvs/colmap.py`, `acmmp.py` | official COLMAP / ACMMP binaries on a temporary workspace |
| `mvsbench/mvs/learned.py` | PatchmatchNet, IterMVS, rmvd-wrapped MVSAnywhere / Vis-MVSNet / Robust MVD |
| `mvsbench/stereo/` | SGBM and the four deep stereo networks |
| `mvsbench/fusion.py` | geometric-consistency fusion (same for all methods) |
| `scripts/run_benchmark.sh` | **the entry point**: setup → data → `run_all.sh` → report (no cloud code) |
| `scripts/run_all.sh` | every configured method in its own process and environment, then the report |
| `scripts/run_*_bench.py`, `run_latency.py` | one method per process; resumable jsonl output; latency = warm-up + timed repeats |
| `scripts/run_dtu_pointcloud.py` | DTU accuracy / completeness with DTUeval-python |
| `scripts/make_report.py`, `make_figures.py`, `make_html_report.py` | `RESULTS.md`, `summary.json`, figures, self-contained `report.html` |
| `scripts/make_synthetic_data.py`, `configs/smoke.yaml` | ray-traced test data + CPU profile for a local end-to-end check |
| `configs/quick.yaml`, `full.yaml` | datasets, methods, N, resolutions; `method_overrides` for slow methods |
| `aws/` | EC2 launcher, bootstrap, watchdog; `run_on_ec2.sh` = S3 sync around `run_benchmark.sh` |
| `setup/` | environments + models, datasets, downloads with an optional local cache |

## Credits

Code and weights of the compared methods belong to their authors and are fetched from their official repositories at
setup time: RAFT-Stereo (Princeton VL), IGEV (Xu et al.), FoundationStereo (NVIDIA, NVIDIA licence), MonSter (Cheng et
al.), PatchmatchNet and IterMVS (Wang et al.), ACMMP (Xu et al.), COLMAP (Schönberger), Vis-MVSNet (Zhang et al.),
MVSAnywhere (Niantic), robustmvd (Schröppel et al., Apache-2.0, also used for the ETH3D GT undistortion),
DTUeval-python (Zhang). Datasets: DTU, ETH3D, Middlebury, KITTI, BlendedMVS — see `docs/DATASETS.md`.
