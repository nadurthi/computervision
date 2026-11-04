"""Learned MVS networks with their official code and weights (setup/install_models.sh fetches both).

  patchmatchnet   Wang et al., CVPR 2021   DTU-trained, learned adaptive PatchMatch, coarse-to-fine
  itermvs_dtu     Wang et al., CVPR 2022   GRU iterative depth update on a probability-encoded hidden state
  itermvs_blended same network, BlendedMVS weights
  mvsanywhere     Izquierdo et al., CVPR 2025   zero-shot, any scale (via the robustmvd wrapper)
  vis_mvsnet      Zhang et al., BMVC 2020  pixel-wise visibility (occlusion) aware cost aggregation (robustmvd)
  robust_mvd      Schroeppel et al., ICRA 2022   Robust MVD baseline (robustmvd native)

PatchmatchNet and IterMVS share module names (`models`, `datasets`), so load one per Python process.
Interface: method(images, Ks, Es, depth_range) -> depth (H,W) on the reference (images[0]) grid, 0 = none.
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _purge(prefixes=("models", "datasets", "utils")):
    for m in list(sys.modules):
        if m.split(".")[0] in prefixes:
            del sys.modules[m]


def _strip(sd):
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def _fit(images, Ks, mult):
    """Resize all views to the largest size <= input that is a multiple of `mult`."""
    h, w = images[0].shape[:2]
    H, W = h // mult * mult, w // mult * mult
    out_i, out_k = [], []
    for img, K in zip(images, Ks):
        sx, sy = W / img.shape[1], H / img.shape[0]
        out_i.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if (H, W) != img.shape[:2] else img)
        K = K.copy()
        K[0] *= sx
        K[1] *= sy
        out_k.append(K)
    return out_i, out_k, (h, w)


def _back(depth, hw):
    if depth.shape != hw:
        depth = cv2.resize(depth, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    depth[~np.isfinite(depth) | (depth < 0)] = 0
    return depth.astype(np.float32)


class PatchmatchNet:
    name = "patchmatchnet"

    def __init__(self, models_root, device="cuda"):
        repo = os.path.join(models_root, "PatchmatchNet")
        _purge()
        sys.path.insert(0, repo)
        from models.net import PatchmatchNet as Net
        self.model = Net(patchmatch_interval_scale=[0.005, 0.0125, 0.025], propagation_range=[6, 4, 2],
                         patchmatch_iteration=[1, 2, 2], patchmatch_num_sample=[8, 8, 16],
                         propagate_neighbors=[0, 8, 16], evaluate_neighbors=[9, 9, 9])
        self.model.load_state_dict(_strip(_load(os.path.join(repo, "checkpoints", "params_000007.ckpt"))["model"]),
                                   strict=False)
        self.model.to(device).eval()
        self.dev = device

    @torch.no_grad()
    def __call__(self, images, Ks, Es, depth_range):
        imgs, Ks2, hw = _fit(images, Ks, 8)
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(self.dev)
        ims = [t(i).permute(2, 0, 1)[None] / 255.0 for i in imgs]
        K = t(np.stack(Ks2))[None]
        E = t(np.stack(Es))[None]
        depth, conf, _ = self.model(ims, K, E, t([depth_range[0]]), t([depth_range[1]]))
        return _back(depth.squeeze().cpu().numpy(), hw)


class IterMVS:
    def __init__(self, models_root, device="cuda", weights="dtu", iterations=4):
        repo = os.path.join(models_root, "IterMVS")
        _purge()
        sys.path.insert(0, repo)
        from models.net import Pipeline
        self.model = Pipeline(iteration=iterations, test=True)
        sd = _load(os.path.join(repo, "checkpoints", weights, "model_000015.ckpt"))["model"]
        self.model.load_state_dict(_strip(sd))
        self.model.to(device).eval()
        self.dev, self.name = device, f"itermvs_{weights}"

    @torch.no_grad()
    def __call__(self, images, Ks, Es, depth_range):
        imgs, Ks2, hw = _fit(images, Ks, 32)
        H, W = imgs[0].shape[:2]
        x = np.stack([2 * i.astype(np.float32) / 255.0 - 1 for i in imgs]).transpose(0, 3, 1, 2)   # N,3,H,W
        x = torch.from_numpy(x).to(self.dev)
        imgs_ms, proj = {}, {}
        for lvl in range(4):
            s = 2 ** lvl
            imgs_ms[f"level_{lvl}"] = (x if lvl == 0 else F.interpolate(x, (H // s, W // s), mode="bilinear",
                                                                       align_corners=False))[None]
            P = []
            for K, E in zip(Ks2, Es):
                Kl = K.copy()
                Kl[:2] /= s
                M = E.copy()
                M[:3, :4] = Kl @ E[:3, :4]
                P.append(M)
            proj[f"level_{lvl}"] = torch.from_numpy(np.stack(P).astype(np.float32))[None].to(self.dev)
        dmin = torch.tensor([depth_range[0]], dtype=torch.float32, device=self.dev)
        dmax = torch.tensor([depth_range[1]], dtype=torch.float32, device=self.dev)
        out = self.model(imgs_ms, proj, dmin, dmax)
        return _back(out["depths_upsampled"].squeeze().cpu().numpy(), hw)


class RMVDModel:
    """Any model registered in lmb-freiburg/robustmvd. rmvd poses map key-view coordinates into each view
    (p_view = pose @ p_key) and use metres, so `units_to_m` converts the dataset units (DTU: mm -> 1e-3)."""

    def __init__(self, rmvd_name, device="cuda", units_to_m=1.0, use_depth_range=False):
        import rmvd
        self.model = rmvd.create_model(rmvd_name, num_gpus=1 if device.startswith("cuda") else 0)
        self.s, self.use_range = units_to_m, use_depth_range
        self.name = rmvd_name.replace("_wrapped", "")

    def __call__(self, images, Ks, Es, depth_range, mult=32):
        imgs, Ks2, hw = _fit(images, Ks, mult)
        E0inv = np.linalg.inv(Es[0])
        poses = []
        for E in Es:
            T = (E @ E0inv).astype(np.float32)
            T[:3, 3] *= self.s
            poses.append(T[None])
        sample = dict(images=[i.transpose(2, 0, 1)[None].astype(np.float32) for i in imgs],
                      keyview_idx=np.array([0], dtype=np.int64), poses=poses,
                      intrinsics=[k.astype(np.float32)[None] for k in Ks2])
        if self.use_range:
            sample["depth_range"] = [np.array([depth_range[0] * self.s], np.float32),
                                     np.array([depth_range[1] * self.s], np.float32)]
        pred, _ = self.model.run(**sample)
        d = np.asarray(pred["depth"])[0, 0] / self.s
        return _back(d.astype(np.float32), hw)


def build(name, models_root, device="cuda", units_to_m=1.0):
    if name == "patchmatchnet":
        return PatchmatchNet(models_root, device)
    if name.startswith("itermvs_"):
        return IterMVS(models_root, device, name.split("_", 1)[1])
    if name in ("mvsanywhere", "vis_mvsnet", "robust_mvd"):
        rname = name if name == "robust_mvd" else name + "_wrapped"
        return RMVDModel(rname, device, units_to_m, use_depth_range=(name == "vis_mvsnet"))
    raise ValueError(name)


ALL = ["patchmatchnet", "itermvs_dtu", "itermvs_blendedmvs", "mvsanywhere", "vis_mvsnet", "robust_mvd"]
