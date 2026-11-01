"""Generate the notebooks in ../notebooks from the cell lists below (no nbformat needed).

    python scripts/build_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


M, C = "markdown", "code"

COMMON = r'''
import os, sys, json, time
from pathlib import Path
REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))
os.chdir(REPO)
print("working in", REPO)
'''

NB = {}

NB["00_setup_and_selfcheck.ipynb"] = [
    (M, """# 00 · Setup and self-check (SageMaker Studio, GPU)

Run this once per instance. Recommended: **ml.g5.xlarge** (A10G) or **ml.g6.xlarge** (L4), image **PyTorch 2.x GPU**.

1. installs the requirements
2. runs the unit tests (includes the GPU crop check that the CPU-only CI skips)
3. loads every model once: RF-DETR, RTMPose, PARSeq, DINOv2, and NVDEC decoding
4. re-measures the blog's profiling surprise: building pose crops on the CPU vs the GPU"""),
    (C, COMMON),
    (C, """!nvidia-smi
!pip install -q -r requirements.txt
# TensorRT for notebook 03 (match the image's CUDA major version)
!pip install -q tensorrt-cu12 || echo "TensorRT install failed: notebook 03 needs it, the rest does not\""""),
    (C, """import torch, cv2, numpy as np
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "pick a GPU instance"
"""),
    (M, "## Unit tests\nThe torch crop test runs here (it is skipped on machines without PyTorch)."),
    (C, "!python -m pytest -q tests"),
    (M, "## Models load and run"),
    (C, """from vb.config import load_config
from vb.detect import Detector
cfg = load_config("configs/default.yaml")
det = Detector(cfg.detector, "cuda")
img = np.full((720, 1280, 3), 120, np.uint8)
cv2.circle(img, (640, 300), 12, (0, 220, 255), -1)          # a ball-like blob, players come from your clip in 01
x = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float().div(255).cuda()[None]
d = det(x)[0]
print("RF-DETR ok:", len(d.player_boxes), "players,", len(d.ball_xy), "balls on a synthetic frame")"""),
    (C, """from vb.pose import PoseEstimator
pose = PoseEstimator(cfg.pose, "cuda")
frame = torch.from_numpy(img).permute(2, 0, 1).float().cuda()     # BGR 0..255
kp, sc = pose(frame, img, np.float32([[600, 200, 680, 420]]))
print("RTMPose ok:", kp.shape, sc.shape)"""),
    (C, """from vb.identity import JerseyReader, AppearanceEmbedder, letterbox_text
reader = JerseyReader(cfg.identity.parseq_hub, cfg.identity.parseq_model, "cuda")
shirt = np.full((64, 64, 3), (40, 40, 200), np.uint8)
cv2.putText(shirt, "12", (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
print("PARSeq read:", reader([letterbox_text(shirt)]))   # expect ('12', high confidence)
emb = AppearanceEmbedder(cfg.identity.dino_model, "cuda")
print("DINOv2 ok:", emb([shirt]).shape)"""),
    (C, """# NVDEC: put the path or s3:// URI of any clip here
from vb.io import fetch, VideoSource
clip = fetch(cfg.video.source, cfg.video.local_cache) if cfg.video.source.startswith("s3://") else None
if clip:
    src = VideoSource(clip, "auto", "cuda")
    print("decoder:", src.decoder, "fps", src.fps, "size", src.size, "frames", src.n_frames)
else:
    print("set video.source in configs/default.yaml to test NVDEC on your clip")"""),
    (M, """## The profiling surprise, re-measured
RTMPose is top-down: every person box becomes a 256×192 crop. Building those crops one by one on the CPU
(what most code does) against one batched `affine_grid` + `grid_sample` on the GPU, for 22 people:"""),
    (C, """from vb.pose import boxes_to_center_scale, batched_crops_cpu, batched_crops_torch
H, W = 1080, 1920
frame_np = (np.random.rand(H, W, 3) * 255).astype(np.float32)
frame_gpu = torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().cuda()
rng = np.random.default_rng(0)
xy = rng.uniform([100, 100], [W - 200, H - 400], (22, 2))
boxes = np.float32(np.c_[xy, xy + rng.uniform([40, 120], [90, 300], (22, 2))])
c, s = boxes_to_center_scale(boxes)
def bench(fn, n=50):
    fn(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return 1000 * (time.perf_counter() - t) / n
cpu_ms = bench(lambda: batched_crops_cpu(frame_np, c, s), 20)
gpu_ms = bench(lambda: batched_crops_torch(frame_gpu, c, s))
diff = np.abs(batched_crops_torch(frame_gpu, c, s).cpu().numpy() - batched_crops_cpu(frame_np, c, s))[:, :, 2:-2, 2:-2].max()
print(f"22 crops: CPU {cpu_ms:.1f} ms/frame, GPU {gpu_ms:.2f} ms/frame, max diff {diff:.4f} (std units)")"""),
]

NB["01_run_pipeline.ipynb"] = [
    (M, """# 01 · Run the pipeline on a clip from S3

Steps: point the config at your clip → pick ≥ 4 court points on the first frame → check the overlay →
run a short test → run the whole clip → look at the events → upload results to S3."""),
    (C, COMMON),
    (C, """from vb.config import load_config
cfg = load_config("configs/default.yaml")
cfg.video.source = "s3://YOUR-BUCKET/volleyball/clip.mp4"      # <- your clip
cfg.output.s3_prefix = "s3://YOUR-BUCKET/volleyball/runs"       # <- where results go (or None)
cfg.output.home_team, cfg.output.away_team = "Home", "Away\""""),
    (M, """## Pick court points
Show the first frame with a pixel grid, read off the pixel positions of court corners / line intersections you can see,
and pair each with its court position in metres. Named points are in `vb.court.COURT_POINTS`
(near = the end closest to the camera: Y = 0; far end Y = 18; net Y = 9; attack lines Y = 6 and 12)."""),
    (C, """import cv2, numpy as np, matplotlib.pyplot as plt
from vb.io import fetch
from vb.court import COURT_POINTS
clip = fetch(cfg.video.source, cfg.video.local_cache)
cap = cv2.VideoCapture(clip); cap.set(cv2.CAP_PROP_POS_MSEC, cfg.video.start_s * 1000); ok, first = cap.read(); cap.release()
plt.figure(figsize=(16, 9)); plt.imshow(first[..., ::-1]); plt.grid(True, alpha=.4)
plt.xticks(range(0, first.shape[1], 100)); plt.yticks(range(0, first.shape[0], 50)); plt.title("read pixel coords of visible court points")
print(COURT_POINTS)"""),
    (C, """# fill in: (u, v) pixels  ->  name of the court point
picked = {
    # "near_left":  (412, 1010),
    # "near_right": (1530, 1004),
    # "far_left":   (760, 352),
    # "far_right":  (1172, 350),
    # "centre_left": (655, 520),
    # "centre_right": (1262, 516),
}
cfg.court.seed_pixels = [list(map(float, uv)) for uv in picked.values()]
cfg.court.seed_court = [list(COURT_POINTS[k]) for k in picked]
len(picked)"""),
    (C, """from vb.court import court_homography, draw_court
H0, inl = court_homography(np.float32(cfg.court.seed_court), np.float32(cfg.court.seed_pixels))
over = draw_court(first.copy(), H0, (0, 255, 255), 3)
plt.figure(figsize=(16, 9)); plt.imshow(over[..., ::-1]); plt.title(f"court overlay, {inl.sum()}/{len(inl)} inliers")"""),
    (M, "## Short test run (first 20 s), then the whole clip"),
    (C, """from vb.pipeline import run_offline
res = run_offline(cfg, run_name="test20s", max_frames=int(20 * 25))
print(json.dumps(res.summary(), indent=1))
print(json.dumps(res.timings, indent=1))"""),
    (C, """res = run_offline(cfg, run_name=Path(cfg.video.source).stem)
print(json.dumps(res.summary(), indent=1))"""),
    (C, """import pandas as pd
ev = pd.DataFrame(json.loads((res.out_dir / "events.json").read_text()))
print("rallies:")
for r in res.rallies: print(f"  {r.idx:3d}  {r.start_t:7.1f}s  {r.labels}")
ev.head(30)"""),
    (C, """pd.DataFrame(res.slots)"""),
    (C, """from vb.dvw import check_with_datavolley
plays, problems = check_with_datavolley(res.dvw_path)
print(len(plays), "rows read by py-datavolley;", len(problems), "problems")
pd.DataFrame(plays)[["code", "team", "player_number", "skill", "evaluation_code", "start_zone"]].head(20)"""),
    (C, """# a few rendered frames around the first touches
cap = cv2.VideoCapture(str(res.out_dir / "render.mp4"))
fig, axs = plt.subplots(2, 3, figsize=(18, 7))
for ax, t in zip(axs.ravel(), res.touches[:6]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, t.frame - int(cfg.video.start_s * res.fps_video)); ok, im = cap.read()
    if ok: ax.imshow(im[..., ::-1]); ax.set_title(f"{t.t:.1f}s {t.skill} #{t.number}"); ax.axis("off")"""),
    (C, """if cfg.output.s3_prefix:
    from vb.io import upload_dir
    uris = upload_dir(str(res.out_dir), f"{cfg.output.s3_prefix}/{res.out_dir.name}")
    print("\\n".join(uris))"""),
]

NB["02_identity.ipynb"] = [
    (M, """# 02 · Jersey identity: what was read, and where it fails

The blog's two identity metrics, computed on your run:
* players with a known number **at serve time** (all 12 on court + libero, every serve)
* touches with a number

plus the breakdown by court side, which shows the weak spot: the team facing the camera rarely shows its back."""),
    (C, COMMON),
    (C, """from vb.config import load_config
from vb.pipeline import run_offline
import pandas as pd, numpy as np
cfg = load_config("configs/default.yaml")
cfg.video.source = "s3://YOUR-BUCKET/volleyball/clip.mp4"
cfg.court.seed_pixels, cfg.court.seed_court = [], []     # <- paste from notebook 01
cfg.output.render = False
res = run_offline(cfg, run_name="identity")"""),
    (C, """s = res.summary()
print("players with a known number at serve time:", s["numbers_at_serve"])
print("touches with a number:", s["touches_with_number"], "/", s["touches_in_rallies"])"""),
    (C, """tt = [t for r in res.rallies for t in r.touches]
by_side = pd.DataFrame([{"side": "near" if t.side == 0 else "far", "has_number": t.number is not None} for t in tt])
by_side.groupby("side").has_number.agg(["sum", "count", "mean"])"""),
    (C, "pd.DataFrame(res.slots).sort_values(['side', 'slot'])"),
]

NB["03_tensorrt_and_determinism.ipynb"] = [
    (M, """# 03 · TensorRT engines, speed, and byte-identical output

1. export RF-DETR and RTMPose to ONNX
2. build TensorRT engines **with a pinned timing cache** (the first machine creates it; commit it or keep it in S3;
   every other machine builds with `strict_cache=True`)
3. compare TensorRT against PyTorch/onnxruntime
4. time the pipeline with each back-end
5. determinism: run twice and compare the scout section of the .dvw byte for byte; do the same on a second instance"""),
    (C, COMMON),
    (C, """from vb.config import load_config
from vb.detect import export_rfdetr_onnx
from vb.pose import download_rtmpose
from vb.trt import build_engine
cfg = load_config("configs/default.yaml")
Path("engines").mkdir(exist_ok=True)
det_onnx = export_rfdetr_onnx(cfg.detector.variant, "engines/onnx", cfg.detector.weights, batch_size=cfg.video.batch)
pose_onnx = download_rtmpose(cfg.pose.model, cfg.pose.onnx_path)
print(det_onnx, pose_onnx)"""),
    (C, """# timing caches live next to the engines; copy them to S3 and reuse them everywhere
CACHE_S3 = None   # e.g. "s3://YOUR-BUCKET/volleyball/trt-cache"
strict = False    # set True on every machine except the one that creates the caches
build_engine(det_onnx, cfg.detector.engine_path, "engines/rfdetr.timing.cache", fp16=True, strict_cache=strict)
build_engine(pose_onnx, cfg.pose.engine_path, "engines/rtmpose.timing.cache", fp16=True, strict_cache=strict)
print(open(Path(cfg.detector.engine_path).with_suffix(".json")).read())"""),
    (C, """# TensorRT vs PyTorch on real frames from your clip
import torch, numpy as np
from vb.detect import Detector
from vb.io import fetch, VideoSource, to_gpu_rgb
cfg.video.source = "s3://YOUR-BUCKET/volleyball/clip.mp4"
src = VideoSource(fetch(cfg.video.source, cfg.video.local_cache), "auto", "cuda")
frames = [f for _, f in zip(range(cfg.video.batch), src)]
rgb, _ = to_gpu_rgb(frames)
cfg.detector.backend = "torch"; d_t = Detector(cfg.detector)(rgb)
cfg.detector.backend = "tensorrt"; d_r = Detector(cfg.detector)(rgb)
for a, b in zip(d_t, d_r):
    print(len(a.player_boxes), len(b.player_boxes), "players |", len(a.ball_xy), len(b.ball_xy), "balls")"""),
    (C, """from vb.pipeline import run_offline
cfg.court.seed_pixels, cfg.court.seed_court = [], []     # <- paste from notebook 01
cfg.output.render = False
rows = []
for det_b, pose_b, gpu_crops in [("torch", "onnxruntime", False), ("torch", "onnxruntime", True), ("tensorrt", "tensorrt", True)]:
    cfg.detector.backend, cfg.pose.backend, cfg.pose.crops_on_gpu = det_b, pose_b, gpu_crops
    r = run_offline(cfg, run_name=f"bench-{det_b}-{pose_b}-{'gpu' if gpu_crops else 'cpu'}crops")
    rows.append({"detector": det_b, "pose": pose_b, "crops": "gpu" if gpu_crops else "cpu",
                 "fps": r.summary()["processing_fps"], **{k: v["ms_per_frame"] for k, v in r.timings.items() if k in ("detect", "pose", "homography", "track_ball")}})
import pandas as pd; pd.DataFrame(rows)"""),
    (M, "## Determinism\nRun the same clip twice here, then run this cell on a second instance with the same caches and compare the printed hash."),
    (C, """from vb.trt import events_fingerprint
cfg.detector.backend, cfg.pose.backend = "tensorrt", "tensorrt"
cfg.video.decoder = "nvdec"; cfg.video.allow_decoder_fallback = False
h = [events_fingerprint(run_offline(cfg, run_name=f"det{i}").dvw_path) for i in range(2)]
print(h[0]); print("identical across runs:", h[0] == h[1])"""),
]

NB["04_evaluate.ipynb"] = [
    (M, """# 04 · Evaluate

**A. The skill rule on expert DataVolley files.** Perfect touch sequences in, so every error is the rule's.
Runs on the 7 matches bundled with py-datavolley, and on your own .dvw files if you point it at them.

**B. Pipeline events against hand labels.** Label a few rallies (CSV: `rally,t,team,number,skill`) and score
event recall/precision, skill accuracy, numbers, and rallies fully right."""),
    (C, COMMON),
    (C, """import datavolley as dv
from vb.evaluate import score_rule_on_dvw_files
r = score_rule_on_dvw_files(dv.example_file_list())
print(json.dumps({k: v for k, v in r.items() if k != "per_skill"}, indent=1))
pd_rows = [{"skill": k, **v} for k, v in r["per_skill"].items()]
import pandas as pd; pd.DataFrame(pd_rows)"""),
    (C, """# your own expert files (download from S3 first, or list local paths)
my_files = []   # e.g. ["data/dvw/match1.dvw", ...]
if my_files:
    print(json.dumps(score_rule_on_dvw_files(my_files), indent=1))"""),
    (M, "### B. Hand labels\nTemplate written to `data/labels_template.csv`. Times are video seconds at the contact."),
    (C, """Path("data").mkdir(exist_ok=True)
Path("data/labels_template.csv").write_text("rally,t,team,number,skill\\n0,12.40,*,7,S\\n0,13.52,a,3,R\\n")
from vb.evaluate import load_labels, score_events
labels_path = "data/labels.csv"          # <- your labels
run_dir = Path("outputs") / "clip"        # <- a run from notebook 01
if Path(labels_path).exists():
    pred = json.loads((run_dir / "events.json").read_text())
    print(json.dumps(score_events(pred, load_labels(labels_path)).as_dict(), indent=1))"""),
]

NB["05_live_stream.ipynb"] = [
    (M, """# 05 · Streaming: one decode pass, rally data as soon as the rally ends

`run_stream` emits each rally when the ball has been gone for `rally.end_gap_s` after the last touch, and reports
the latency from the rally's last decoded frame to the moment its events are ready. Identities are provisional
per rally and re-voted at the end (the set-end correction). Use a file to simulate live, or an `srt://` URL."""),
    (C, COMMON),
    (C, """from vb.config import load_config
from vb.pipeline import run_stream
import numpy as np
cfg = load_config("configs/default.yaml")
cfg.video.source = "s3://YOUR-BUCKET/volleyball/clip.mp4"   # or "srt://host:port" (decoder must be cpu for streams)
cfg.court.seed_pixels, cfg.court.seed_court = [], []        # <- paste from notebook 01
cfg.detector.backend, cfg.pose.backend = "tensorrt", "tensorrt"   # after notebook 03
cfg.output.render = False
def on_rally(r, lat):
    print(f"rally {r.idx:3d} ready {lat:5.1f}s after its last touch: {r.labels}")
res = run_stream(cfg, on_rally=on_rally)
lat = np.array(res.latencies_s)
print(f"median {np.median(lat):.1f}s, p90 {np.percentile(lat, 90):.1f}s over {len(lat)} rallies")"""),
]

NB["06_datasets_to_s3.ipynb"] = [
    (M, """# 06 · Public datasets straight into S3

Run on a Studio space with a large disk (100 GB): Google Drive, SoccerNet and Roboflow downloads are staged on the
space's disk, uploaded, and deleted; GitHub files stream straight into S3. Each dataset lands in
`s3://<bucket>/datasets/<name>/raw/` with a `MANIFEST.json` (source, licence, date, sizes, sha256).

**Licences.** SportsMOT and the Koshkina jersey weights/crops are **non-commercial**; the Volleyball dataset states
no licence for the data. `--list` prints each one. Check before using them in a product."""),
    (C, COMMON),
    (C, """!pip install -q gdown SoccerNet roboflow boto3
BUCKET = os.environ.get("VB_BUCKET", "vb-data-<account-id>")      # <- your bucket
DEST = f"s3://{BUCKET}/datasets"
!python scripts/datasets_to_s3.py --list"""),
    (M, "## Small and open first (seconds): the public video + .dvw pair, and the expert .dvw files"),
    (C, "!python scripts/datasets_to_s3.py --dest {DEST} ovdata pydatavolley_dvw"),
    (M, "## The Volleyball dataset (Ibrahim et al.) and its ball and trajectory annotations\nLarge. If Google Drive refuses with a quota error, retry later or pass the old SFU zip with `--url`."),
    (C, "!python scripts/datasets_to_s3.py --dest {DEST} --extract volleyball_ibrahim volleyball_ibrahim_ball volleyball_ibrahim_tracks"),
    (M, "## Jersey numbers: SoccerNet jersey-2023, and the PARSeq weights fine-tuned on jerseys (non-commercial)"),
    (C, """!python scripts/datasets_to_s3.py --dest {DEST} --extract soccernet_jersey
!python scripts/datasets_to_s3.py --dest s3://{BUCKET}/models parseq_jersey_weights"""),
    (M, "## Roboflow Universe sets (ball, court)\nNeeds a free Roboflow API key: app.roboflow.com → Settings → API Keys."),
    (C, """os.environ["ROBOFLOW_API_KEY"] = ""   # <- paste, or set it in the terminal
!python scripts/datasets_to_s3.py --dest {DEST} roboflow_ball_daniel roboflow_ball_tracker roboflow_volleyball01 roboflow_court"""),
    (M, "## SportsMOT (non-commercial)\nSign up on Codalab, join the SportsMOT competition, copy the link under *Participate → Get Data*."),
    (C, """SPORTSMOT_URL = ""   # <- the Codalab link
if SPORTSMOT_URL:
    !python scripts/datasets_to_s3.py --dest {DEST} sportsmot --url "{SPORTSMOT_URL}\""""),
    (M, "## What is in the bucket"),
    (C, """import boto3, json
s3 = boto3.client("s3")
rows = []
for p in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="datasets/", Delimiter="/"):
    for cp in p.get("CommonPrefixes", []):
        name = cp["Prefix"].split("/")[1]
        try:
            m = json.loads(s3.get_object(Bucket=BUCKET, Key=f"datasets/{name}/MANIFEST.json")["Body"].read())
            rows.append({"dataset": name, "files": len(m["files"]), "GB": round(m["total_bytes"] / 1e9, 2),
                         "commercial_ok": m["commercial_ok"], "licence": m["licence"]})
        except s3.exceptions.NoSuchKey:
            rows.append({"dataset": name, "files": None})
import pandas as pd; pd.DataFrame(rows)"""),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, cells in NB.items():
        (OUT / name).write_text(json.dumps(nb(cells), indent=1) + "\n")
        print("wrote", OUT / name)
