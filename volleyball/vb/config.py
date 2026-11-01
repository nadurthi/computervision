"""Configuration for the volleyball pipeline.

Everything that changes between clips lives here: where the video is, which models to use,
the court seed points for the homography, and the thresholds of the rules. Load it from YAML
with :func:`load_config` and override fields in a notebook cell.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VideoConfig:
    source: str = ""                 # local path, s3://bucket/key.mp4, or srt://host:port for live
    local_cache: str = "/tmp/vb_cache"
    decoder: str = "auto"            # auto | nvdec | cpu  (auto = nvdec if available, else cpu)
    allow_decoder_fallback: bool = False  # never mix decoders inside one run (determinism)
    start_s: float = 0.0
    end_s: float | None = None
    stride: int = 1                  # process every n-th frame
    batch: int = 8


@dataclass
class DetectorConfig:
    variant: str = "medium"          # nano | small | medium | base | large  (rfdetr variants)
    weights: str | None = None       # fine-tuned checkpoint; None = COCO pretrained
    backend: str = "torch"           # torch | tensorrt
    engine_path: str = "engines/rfdetr.engine"
    player_class_id: int = 1         # COCO "person"
    ball_class_id: int = 37          # COCO "sports ball"
    player_thresh: float = 0.35
    ball_thresh: float = 0.15
    max_ball_dets: int = 5


@dataclass
class PoseConfig:
    model: str = "rtmpose-m"         # rtmpose-s | rtmpose-m  (COCO-17 body7 models)
    onnx_path: str = "models/rtmpose-m.onnx"
    backend: str = "onnxruntime"     # onnxruntime | tensorrt
    engine_path: str = "engines/rtmpose.engine"
    input_wh: tuple[int, int] = (192, 256)
    padding: float = 1.25
    crops_on_gpu: bool = True        # the 11.5 -> 29.2 fps change from the blog
    max_people: int = 32


@dataclass
class CourtConfig:
    # Seed correspondences for the first processed frame: pixel (u, v) <-> court metres (X, Y).
    # X across the court in [0, 9], Y along it in [0, 18], net at Y = 9.
    # At least 4 points. Pick them once per clip in notebook 01 (there is a helper).
    seed_pixels: list[list[float]] = field(default_factory=list)
    seed_court: list[list[float]] = field(default_factory=list)
    propagate: bool = True           # follow pan/zoom with frame-to-frame feature homographies
    reseed_every_s: float = 0.0      # 0 = never; >0 re-anchors on seed frame features periodically
    ransac_px: float = 3.0
    near_side_is_home: bool = True   # which team is on Y < 9 at the start of set 1


@dataclass
class BallConfig:
    gate_px: float = 120.0           # max jump from prediction to accept a detection
    max_gap: int = 6                 # frames bridged by the motion model
    lost_after: int = 15             # frames without the ball -> track ends
    ball_diameter_px: float = 14.0   # used by the static-recovery filter
    static_check_steps: int = 4      # the "spare ball on the scorer's table" fix


@dataclass
class TouchConfig:
    angle_deg: float = 25.0          # min direction change at a kink
    lag: int = 3                     # k in cos(theta) = v_t . v_{t+k}
    min_speed_px: float = 2.0
    hand_dist: float = 0.35          # nearest hand distance / player height
    raised_hand_rule: bool = True
    refractory: int = 6              # frames between two touches
    net_band_m: float = 1.2          # |Y - 9| below this counts as "at the net"
    block_window_s: float = 0.6


@dataclass
class RallyConfig:
    end_gap_s: float = 1.5           # ball unseen this long -> rally over
    behind_line_m: float = 0.3       # server's feet this far past an end line -> serve


@dataclass
class IdentityConfig:
    enabled: bool = True
    parseq_hub: str = "baudm/parseq"
    parseq_model: str = "parseq"     # parseq | parseq_tiny
    min_read_conf: float = 0.6
    min_shoulder_conf: float = 0.5
    min_shoulder_px: float = 8.0
    read_every: int = 2              # run the reader on every n-th back-facing frame per track
    dino_model: str = "dinov2_vits14"
    accept_ratio: float = 0.6        # winner share of vote mass
    accept_mass: float = 1.5         # winner absolute vote mass
    libero_colour_min_share: float = 0.5
    home_roster: list[int] = field(default_factory=list)   # optional, constrains reads
    away_roster: list[int] = field(default_factory=list)


@dataclass
class OutputConfig:
    out_dir: str = "outputs"
    s3_prefix: str | None = None     # e.g. s3://my-bucket/volleyball/runs
    render: bool = True
    home_team: str = "Home"
    away_team: str = "Away"
    match_date: str = "01/01/2026"


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    touch: TouchConfig = field(default_factory=TouchConfig)
    rally: RallyConfig = field(default_factory=RallyConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "cuda"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(dc: Any, d: dict[str, Any]) -> Any:
    for k, v in d.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key {type(dc).__name__}.{k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Read a YAML file into a :class:`Config`; keyword overrides use dotted section dicts."""
    cfg = Config()
    if path is not None:
        import yaml

        with open(path) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg
