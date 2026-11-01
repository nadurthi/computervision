"""Video input: S3 download, NVDEC decoding (torchcodec) or CPU decoding (OpenCV/FFmpeg), SRT streams.

One rule from the blog's determinism hunt: never mix decoders inside a run. NVDEC and the CPU
decoder differ by up to 2 grey levels, enough to flip a borderline detection. The decoder is chosen
once at open time; a frame that NVDEC cannot decode is an error, not a silent CPU fallback,
unless VideoConfig.allow_decoder_fallback is set (and then the whole run is re-opened on the CPU).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def fetch(source: str, cache_dir: str) -> str:
    """s3://bucket/key -> local file (cached); local paths and stream URLs are returned unchanged."""
    if not source.startswith("s3://"):
        return source
    import boto3

    bucket, key = source[5:].split("/", 1)
    dst = Path(cache_dir) / bucket / key
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(dst))
    return str(dst)


def upload_dir(local_dir: str, s3_prefix: str) -> list[str]:
    import boto3

    bucket, prefix = s3_prefix[5:].split("/", 1)
    s3 = boto3.client("s3")
    out = []
    for p in sorted(Path(local_dir).rglob("*")):
        if p.is_file():
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir).as_posix()}"
            s3.upload_file(str(p), bucket, key)
            out.append(f"s3://{bucket}/{key}")
    return out


@dataclass
class Frame:
    idx: int
    t: float                       # seconds in the video (stream time)
    wall: float                    # time.perf_counter() when the frame became available
    rgb_gpu: object | None         # (3,H,W) uint8 CUDA tensor, NVDEC path only
    bgr: np.ndarray                # (H,W,3) uint8 host copy (homography, identity crops, render)


class VideoSource:
    def __init__(self, path: str, decoder: str = "auto", device: str = "cuda", start_s: float = 0.0,
                 end_s: float | None = None, stride: int = 1):
        self.path = path
        self.start_s, self.end_s, self.stride = start_s, end_s, stride
        self.is_stream = path.startswith(("srt://", "rtmp://", "rtsp://", "udp://"))
        self.decoder = self._choose(decoder, device)
        if self.decoder == "nvdec":
            from torchcodec.decoders import VideoDecoder

            self._dec = VideoDecoder(path, device=device)
            md = self._dec.metadata
            self.fps = float(md.average_fps)
            self.n_frames = int(md.num_frames)
            self.size = (int(md.width), int(md.height))
        else:
            import cv2

            self._cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                raise IOError(f"cannot open {path}")
            self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)
            self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _choose(self, decoder: str, device: str) -> str:
        if decoder in ("nvdec", "cpu"):
            if decoder == "nvdec" and self.is_stream:
                raise ValueError("NVDEC via torchcodec reads files; use decoder='cpu' for live streams")
            return decoder
        if self.is_stream or not device.startswith("cuda"):
            return "cpu"
        try:
            import torchcodec  # noqa: F401
            from torchcodec.decoders import VideoDecoder

            VideoDecoder(self.path, device=device)
            return "nvdec"
        except Exception:
            return "cpu"

    def __iter__(self) -> Iterator[Frame]:
        first = int(round(self.start_s * self.fps))
        last = int(round(self.end_s * self.fps)) if self.end_s else None
        if self.decoder == "nvdec":
            stop = min(last or self.n_frames, self.n_frames)
            chunk = 32 * self.stride
            for a in range(first, stop, chunk):
                b = min(a + chunk, stop)
                batch = self._dec.get_frames_in_range(a, b, step=self.stride)
                host = batch.data.permute(0, 2, 3, 1).flip(-1).cpu().numpy()   # RGB -> BGR on host
                now = time.perf_counter()
                for k in range(batch.data.shape[0]):
                    i = a + k * self.stride
                    yield Frame(i, i / self.fps, now, batch.data[k], np.ascontiguousarray(host[k]))
        else:
            import cv2

            if first and not self.is_stream:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            i = first
            while True:
                ok, img = self._cap.read()
                if not ok:
                    break
                if last is not None and i >= last:
                    break
                if (i - first) % self.stride == 0:
                    yield Frame(i, i / self.fps, time.perf_counter(), None, img)
                i += 1


def to_gpu_rgb(frames: list[Frame], device: str = "cuda"):
    """Stack frames as (B,3,H,W) float RGB in [0,1] on the GPU, plus (B,3,H,W) BGR 0..255 for pose."""
    import torch

    if frames[0].rgb_gpu is not None:
        rgb = torch.stack([f.rgb_gpu for f in frames]).float()
    else:
        bgr = torch.from_numpy(np.stack([f.bgr for f in frames])).to(device, non_blocking=True)
        rgb = bgr.permute(0, 3, 1, 2).flip(1).float()
    return rgb / 255.0, rgb.flip(1)
