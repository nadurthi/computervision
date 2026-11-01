"""RTMPose (top-down, SimCC heads) with all person crops built on the GPU in one batch.

The crop for each person is the same affine warp rtmlib/mmpose use: box -> centre and scale,
scale padded by 1.25 and widened to the model's 3:4 aspect ratio, then mapped onto a 192x256
input. The blog's profiling surprise was that building those sampling grids in NumPy cost 59 ms
per frame for ~22 people; here they are one batched affine_grid + grid_sample.

Back-ends: onnxruntime (CUDA EP) for the downloaded ONNX model, or a TensorRT engine (vb.trt).
Input convention follows rtmlib's ONNX models: BGR image, mean (123.675, 116.28, 103.53),
std (58.395, 57.12, 57.375), NCHW float32.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

RTMPOSE_URLS = {
    "rtmpose-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    "rtmpose-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
}
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
SIMCC_SPLIT = 2.0


def download_rtmpose(name: str, dst: str | Path) -> Path:
    """Fetch the official RTMPose ONNX SDK zip and extract end2end.onnx to dst."""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RTMPOSE_URLS[name], timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        onnx_name = next(n for n in z.namelist() if n.endswith(".onnx"))
        dst.write_bytes(z.read(onnx_name))
    return dst


def boxes_to_center_scale(boxes: np.ndarray, input_wh=(192, 256), padding: float = 1.25):
    """xyxy (N,4) -> centre (N,2), scale (N,2) with rtmlib's padding and aspect-ratio fix."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    c = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    s = (boxes[:, 2:] - boxes[:, :2]) * padding
    ar = input_wh[0] / input_wh[1]
    w, h = s[:, 0], s[:, 1]
    s = np.where((w > h * ar)[:, None], np.stack([w, w / ar], 1), np.stack([h * ar, h], 1))
    return c.astype(np.float32), s.astype(np.float32)


def crop_affines(centers: np.ndarray, scales: np.ndarray, out_wh=(192, 256)) -> np.ndarray:
    """(N,3,3) maps from output pixel (x_o, y_o, 1) to source pixel, rotation 0.

    x_src = (x_o - W_o/2) * s_w / W_o + c_x   (the inverse of rtmlib's get_warp_matrix).
    """
    Wo, Ho = out_wh
    N = len(centers)
    A = np.zeros((N, 3, 3), np.float64)
    A[:, 0, 0] = scales[:, 0] / Wo
    A[:, 1, 1] = scales[:, 1] / Ho
    A[:, 0, 2] = centers[:, 0] - scales[:, 0] / 2
    A[:, 1, 2] = centers[:, 1] - scales[:, 1] / 2
    A[:, 2, 2] = 1
    return A


def norm_mat(W: int, H: int) -> np.ndarray:
    """Pixel index -> grid_sample normalised coords with align_corners=False."""
    return np.array([[2 / W, 0, 1 / W - 1], [0, 2 / H, 1 / H - 1], [0, 0, 1]], np.float64)


def affine_theta(A: np.ndarray, src_wh: tuple[int, int], out_wh: tuple[int, int]) -> np.ndarray:
    """theta for torch affine_grid: N_src . A . N_out^-1, top two rows, shape (N,2,3)."""
    Ns = norm_mat(*src_wh)
    No_inv = np.linalg.inv(norm_mat(*out_wh))
    return (Ns[None] @ A @ No_inv[None])[:, :2, :].astype(np.float32)


def batched_crops_torch(frame_bgr, centers: np.ndarray, scales: np.ndarray, out_wh=(192, 256)):
    """frame_bgr: (3,H,W) float32 CUDA tensor in 0..255 BGR -> normalised crops (N,3,Ho,Wo)."""
    import torch
    import torch.nn.functional as F

    _, Hs, Ws = frame_bgr.shape
    Wo, Ho = out_wh
    A = crop_affines(centers, scales, out_wh)
    theta = torch.from_numpy(affine_theta(A, (Ws, Hs), out_wh)).to(frame_bgr.device)
    N = theta.shape[0]
    grid = F.affine_grid(theta, (N, 3, Ho, Wo), align_corners=False)
    crops = F.grid_sample(frame_bgr.unsqueeze(0).expand(N, -1, -1, -1), grid,
                          mode="bilinear", padding_mode="zeros", align_corners=False)
    mean = torch.tensor(MEAN, device=crops.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=crops.device).view(1, 3, 1, 1)
    return (crops - mean) / std


def batched_crops_cpu(frame_bgr: np.ndarray, centers: np.ndarray, scales: np.ndarray, out_wh=(192, 256)) -> np.ndarray:
    """Reference path with cv2.warpAffine (what rtmlib does), for checking the GPU path."""
    import cv2

    A = crop_affines(centers, scales, out_wh)
    out = []
    for a in A:
        M = np.linalg.inv(a)[:2]                       # source -> output
        img = cv2.warpAffine(frame_bgr, M, out_wh, flags=cv2.INTER_LINEAR)
        out.append(((img - np.array(MEAN)) / np.array(STD)).transpose(2, 0, 1))
    return np.asarray(out, np.float32).reshape(-1, 3, out_wh[1], out_wh[0])


def simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, centers: np.ndarray, scales: np.ndarray,
                 out_wh=(192, 256)) -> tuple[np.ndarray, np.ndarray]:
    """(N,K,Wx), (N,K,Wy) -> keypoints (N,K,2) in source pixels, scores (N,K)."""
    xl = simcc_x.argmax(-1).astype(np.float32)
    yl = simcc_y.argmax(-1).astype(np.float32)
    score = 0.5 * (simcc_x.max(-1) + simcc_y.max(-1))
    kp = np.stack([xl, yl], -1) / SIMCC_SPLIT                 # output pixels
    kp = kp / np.array(out_wh, np.float32) * scales[:, None, :] + centers[:, None, :] - scales[:, None, :] / 2
    kp[score <= 0] = -1
    return kp, score


class PoseEstimator:
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.out_wh = tuple(cfg.input_wh)
        self.device = device
        self.backend = cfg.backend
        if self.backend == "tensorrt":
            from .trt import TRTModule

            self.trt = TRTModule(cfg.engine_path)
        else:
            import onnxruntime as ort

            path = download_rtmpose(cfg.model, cfg.onnx_path)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") \
                else ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(str(path), so, providers=providers)
            self.in_name = self.sess.get_inputs()[0].name
            b = self.sess.get_inputs()[0].shape[0]
            self.fixed_batch = b if isinstance(b, int) else None   # some exports are batch-1 only

    def _ort(self, crops: np.ndarray):
        if self.fixed_batch in (None, len(crops)):
            return self.sess.run(None, {self.in_name: crops})
        outs = [self.sess.run(None, {self.in_name: crops[i:i + 1]}) for i in range(len(crops))]
        return [np.concatenate([o[k] for o in outs]) for k in range(2)]

    def __call__(self, frame_bgr_gpu, frame_bgr_np: np.ndarray | None, boxes: np.ndarray):
        """frame_bgr_gpu: (3,H,W) float CUDA tensor (0..255) or None; boxes (N,4) -> kp, scores."""
        if len(boxes) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        boxes = boxes[: self.cfg.max_people]
        c, s = boxes_to_center_scale(boxes, self.out_wh, self.cfg.padding)
        if self.cfg.crops_on_gpu and frame_bgr_gpu is not None:
            crops = batched_crops_torch(frame_bgr_gpu, c, s, self.out_wh)
            if self.backend == "tensorrt":
                sx, sy = self.trt(crops.contiguous())
                sx, sy = sx.float().cpu().numpy(), sy.float().cpu().numpy()
            else:
                sx, sy = self._ort(crops.cpu().numpy())
        else:
            crops = batched_crops_cpu(frame_bgr_np, c, s, self.out_wh)
            if self.backend == "tensorrt":
                import torch

                sx, sy = self.trt(torch.from_numpy(crops).to(self.device))
                sx, sy = sx.float().cpu().numpy(), sy.float().cpu().numpy()
            else:
                sx, sy = self._ort(crops)
        return simcc_decode(sx, sy, c, s, self.out_wh)
