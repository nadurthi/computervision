# LIDAR-only localization — NumPy port and nuScenes runs

Port of the method in *Scan-Matching Based Particle Filtering Approach for LIDAR-Only Localization*
([arXiv:2302.06843](https://arxiv.org/abs/2302.06843)) and of the original KITTI
code in [`../drivers/main_kitti_filter_localize.py`](../drivers/main_kitti_filter_localize.py). The original needs
numba, open3d, scikit-learn, PCL and the `pyslam` C++ module; this port needs only `numpy` and `scipy`.

| File | What it is | Mirrors |
|---|---|---|
| `lidarloc.py` | lookup-table likelihood, constant-turn-rate model, particle filter, road removal, multi-resolution BEV matcher, point-to-plane ICP | `pyslam` `computeLikelihood_lookup`, `dynmodelCT3D`, `uq.filters.pf`, open3d normals, `binmatchers.binMatcherAdaptive_super`, PCL ICP |
| `localize.py` | the per-frame loop: PF → RPE → RS → S2M → SU, plus scoring against ground truth | paper block diagram |
| `nusc_data.py` | nuScenes reader (no devkit): LIDAR_TOP sweeps, lidar→global poses, map building from other drives through the same area | — |
| `synthetic.py` | ray-cast 32-beam LIDAR in a synthetic city, for an end-to-end check without data | — |
| `check_against_original.py` | runs the original `binMatcherAdaptive_super` (numba bodies, no JIT, via `tools/stubs`) and the port on the same scene | — |

```bash
cd 08-lidar-localization/nuscenes
python check_against_original.py tools/stubs     # original vs port: identical pose and costs
python -c "import synthetic, localize; f,M,_=synthetic.make_sequence(0,120); r,T=localize.run(f,M,dict(Npf=500)); print(localize.summarize(r))"
```

**Note on the earlier matcher.** `binMatcherAdaptive3` in the KITTI driver starts its translation boxes at offset
(0, 0) and only splits into non-negative offsets, so it can move the guess in +x/+y only. `binMatcherAdaptive_super`
(the one the driver calls) searches absolute positions within ±Lmax of the guess; the port follows `_super`.
