# Volleyball analysis from one broadcast camera

PyTorch code for the pipeline described on perceptionstack.dev
([part 1](https://perceptionstack.dev/tutorials/applications/volleyball-pipeline/),
[part 2](https://perceptionstack.dev/tutorials/applications/volleyball-identity-live/)):

```
video (S3 / file / SRT) ─► NVDEC ─► RF-DETR (players, ball) ─► tracker ─► RTMPose (GPU crops)
      ─► court homography H_t ─► touches ─► rallies ─► skills from touch order
      ─► jersey identity (PARSeq + DINOv2 + colour, slot voting) ─► DataVolley .dvw + render
```

Runs on a SageMaker Studio GPU notebook (g5/g6). Optional TensorRT engines with a pinned timing cache
make the output byte-identical across machines.

## Quick start (SageMaker Studio)

```bash
git clone https://github.com/nadurthi/computervision
cd computervision/volleyball
pip install -r requirements.txt
```

Then open the notebooks in order:

| Notebook | What it does |
|---|---|
| `00_setup_and_selfcheck` | installs, runs the tests on the GPU, loads every model once, re-measures CPU vs GPU pose crops |
| `01_run_pipeline` | clip from S3 → pick court points → overlay check → run → events, slots, `.dvw`, render → upload to S3 |
| `02_identity` | numbers at serve time, touches with a number, near vs far side |
| `03_tensorrt_and_determinism` | ONNX export, TensorRT engines with a pinned timing cache, speed table, byte-identical check |
| `04_evaluate` | the skill rule on expert `.dvw` files; pipeline events against your hand labels |
| `05_live_stream` | one-pass streaming mode with per-rally latency |

The notebooks are generated from `scripts/build_notebooks.py`; edit that file and re-run it rather than the JSON.

## Layout

```
vb/config.py     every setting, loaded from configs/*.yaml
vb/io.py         S3, NVDEC (torchcodec) or CPU decoding, never mixed within a run
vb/detect.py     RF-DETR (rfdetr package or a TensorRT engine)
vb/track.py      players (IoU + Hungarian), ball (gating, gap bridging, static-ball rejection)
vb/pose.py       RTMPose ONNX/TensorRT, batched affine_grid crops on the GPU, SimCC decode
vb/court.py      DLT/RANSAC homography, frame-to-frame tracking of camera pan/zoom
vb/touches.py    gravity-compensated kink detection, hand attribution, raised-hand rule
vb/rally.py      rally segmentation from serve position and ball gaps
vb/skills.py     skill from touch order (S R E A B D F) + scoring against expert .dvw
vb/identity.py   back-to-camera test, PARSeq digits, DINOv2 embeddings, colour class, slot voting
vb/dvw.py        DataVolley writer (checked with py-datavolley)
vb/trt.py        TensorRT build with a pinned timing cache, runner, fingerprint
vb/pipeline.py   run_offline / run_stream
vb/evaluate.py   events vs hand labels, rule vs expert files
```

## Tests

```bash
python -m pytest -q tests
```

`tests/test_pipeline_fake.py` runs the whole pipeline on a generated clip with stand-in models, so the
plumbing (decode → tracking → homography → touches → skills → identity → `.dvw` → render) is tested without
a GPU or downloads. The GPU crop test runs only where PyTorch is installed (notebook 00).

## Measured so far

Skill from touch order on the 7 expert-scouted matches bundled with py-datavolley
(5,671 touches, 926 rallies; perfect input, so every error is the rule's):

| | accuracy |
|---|---|
| all touches | 96.1 % |
| excluding free balls | **97.4 %** |
| free balls | 44.7 % (141) |
| rallies fully right | 802 / 926 |
| with no net information (blocks invisible) | 84.9 % excl. free balls |

Reproduce with `python scripts/eval_rule_on_dvw.py`. Everything that needs video (detection, identity,
speed, latency) is measured by the notebooks on your own clips.

## Known limits

* Court seed points are picked by hand on the first frame; the homography then follows the camera with
  background features. A court-keypoint model can replace the seed.
* Front digits are not read; the team facing the camera gets numbers only through slots and appearance.
* Point winners in the `.dvw` are a heuristic (last touch); evaluations other than `+`/`#` are not judged yet.
* Free balls sent on a third touch look like attacks in the touch sequence (the main remaining rule error).
