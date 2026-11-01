"""Players and ball with RF-DETR (COCO-pretrained by default: class 1 person, 37 sports ball).

Two back-ends:
  torch     - the rfdetr package's model.predict on a batch of RGB tensors
  tensorrt  - an engine built from rfdetr's ONNX export (outputs "dets" cxcywh-normalised and
              "labels" logits); pre/post-processing done here in torch so frames stay on the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
             "base": "RFDETRBase", "large": "RFDETRLarge"}


@dataclass
class FrameDets:
    player_boxes: np.ndarray     # (N,4) xyxy
    player_scores: np.ndarray    # (N,)
    ball_xy: np.ndarray          # (M,2) centres
    ball_scores: np.ndarray      # (M,)
    ball_boxes: np.ndarray       # (M,4)


def load_rfdetr(variant: str, weights: str | None = None):
    import rfdetr

    cls = getattr(rfdetr, _VARIANTS[variant])
    model = cls(pretrain_weights=weights) if weights else cls()
    try:
        model.optimize_for_inference()
    except Exception:                       # older/newer versions differ; inference still works
        pass
    return model


def _split(boxes, scores, labels, cfg) -> FrameDets:
    boxes, scores, labels = np.asarray(boxes).reshape(-1, 4), np.asarray(scores).reshape(-1), np.asarray(labels).reshape(-1)
    p = (labels == cfg.player_class_id) & (scores >= cfg.player_thresh)
    b = (labels == cfg.ball_class_id) & (scores >= cfg.ball_thresh)
    bi = np.where(b)[0]
    bi = bi[np.argsort(-scores[bi], kind="stable")][: cfg.max_ball_dets]
    bb = boxes[bi]
    return FrameDets(boxes[p].astype(np.float32), scores[p].astype(np.float32),
                     (0.5 * (bb[:, :2] + bb[:, 2:])).astype(np.float32), scores[bi].astype(np.float32),
                     bb.astype(np.float32))


class Detector:
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        if cfg.backend == "tensorrt":
            from .trt import TRTModule

            self.trt = TRTModule(cfg.engine_path)
            shp = self.trt.engine.get_tensor_shape(self.trt.inputs[0])
            self.res = (int(shp[2]), int(shp[3]))
        else:
            self.model = load_rfdetr(cfg.variant, cfg.weights)

    def __call__(self, frames_rgb) -> list[FrameDets]:
        """frames_rgb: (B,3,H,W) float tensor in [0,1] on the GPU."""
        if self.cfg.backend == "tensorrt":
            return self._trt(frames_rgb)
        thr = min(self.cfg.player_thresh, self.cfg.ball_thresh)
        preds = self.model.predict(list(frames_rgb), threshold=thr)
        if not isinstance(preds, list):
            preds = [preds]
        return [_split(d.xyxy, d.confidence, d.class_id, self.cfg) for d in preds]

    def _trt(self, frames_rgb) -> list[FrameDets]:
        import torch
        import torch.nn.functional as F

        B, _, H, W = frames_rgb.shape
        x = F.interpolate(frames_rgb, size=self.res, mode="bilinear", align_corners=False, antialias=False)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        outs = self.trt((x - mean) / std)
        names = self.trt.outputs
        dets = outs[names.index("dets")].float()            # (B,Q,4) cx,cy,w,h in [0,1]
        logits = outs[names.index("labels")].float()        # (B,Q,C)
        prob = logits.sigmoid()
        C = prob.shape[2]
        flat = prob.view(B, -1)
        k = min(300, flat.shape[1])
        order = torch.argsort(flat, dim=1, descending=True, stable=True)[:, :k]   # stable = deterministic ties
        scores = flat.gather(1, order)
        q, labels = order // C, order % C
        boxes = dets.gather(1, q.unsqueeze(-1).expand(-1, -1, 4))
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1)
        xyxy, scores, labels = xyxy.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
        return [_split(xyxy[i], scores[i], labels[i], self.cfg) for i in range(B)]


def export_rfdetr_onnx(variant: str, out_dir: str, weights: str | None = None, batch_size: int = 8) -> str:
    """ONNX export through rfdetr's own exporter (fixed batch for a simple, fast engine)."""
    from pathlib import Path

    model = load_rfdetr(variant, weights)
    try:
        path = model.export(output_dir=out_dir, batch_size=batch_size, dynamic_batch=False, format="onnx")
    except TypeError:                          # older rfdetr: no dynamic_batch / format arguments
        path = model.export(output_dir=out_dir, batch_size=batch_size)
    if not path or not str(path).endswith(".onnx"):
        cands = sorted(Path(out_dir).glob("**/*.onnx"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(f"no ONNX file written to {out_dir}")
        path = cands[-1]
    return str(path)
