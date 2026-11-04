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

On AWS (recommended; GPU, everything automated, auto-stop):

```bash
cd aws && ./launch_mvs_ec2.sh launch          # PROFILE=quick (default) or full
./launch_mvs_ec2.sh status; ./launch_mvs_ec2.sh fetch ./mvs_results
```

On any Ubuntu 22.04 machine with an NVIDIA GPU + CUDA toolkit:

```bash
bash setup/install_models.sh /opt/mvs/models            # 3 envs, 11 repos, weights, ACMMP build
bash setup/download_datasets.sh /opt/mvs/data quick      # or full
bash scripts/run_all.sh quick /opt/mvs/data /opt/mvs/models ./results
cat results/RESULTS.md
```

CPU sanity check (synthetic ray-traced scene; geometry, rectification in all 4 baseline directions, SGBM, plane sweep,
PatchMatch, fusion): `pip install -r requirements.txt && python scripts/smoke_test.py`

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
| `scripts/run_*_bench.py` | one method per process; resumable jsonl output |
| `scripts/run_dtu_pointcloud.py` | DTU accuracy / completeness with DTUeval-python |
| `scripts/make_report.py`, `make_figures.py` | `RESULTS.md`, `summary.json`, figures |
| `configs/quick.yaml`, `full.yaml` | what runs where |
| `aws/` | EC2 launcher, bootstrap, watchdog, on-instance job |
| `setup/` | environments + models, datasets, S3-mirrored downloads |

## Credits

Code and weights of the compared methods belong to their authors and are fetched from their official repositories at
setup time: RAFT-Stereo (Princeton VL), IGEV (Xu et al.), FoundationStereo (NVIDIA, NVIDIA licence), MonSter (Cheng et
al.), PatchmatchNet and IterMVS (Wang et al.), ACMMP (Xu et al.), COLMAP (Schönberger), Vis-MVSNet (Zhang et al.),
MVSAnywhere (Niantic), robustmvd (Schröppel et al., Apache-2.0, also used for the ETH3D GT undistortion),
DTUeval-python (Zhang). Datasets: DTU, ETH3D, Middlebury, KITTI, BlendedMVS — see `docs/DATASETS.md`.
