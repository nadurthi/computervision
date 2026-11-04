"""Learned two-view stereo networks, wrapped around their official repositories and checkpoints.

setup/install_models.sh clones each repo into $MODELS_ROOT/<repo> and downloads the weights. All four repos ship a
top-level `core` / `utils` package with the same names, so load exactly ONE of these per Python process
(the run scripts start a fresh process per method).

Every wrapper: predict(left, right, max_disp) -> disparity (H,W) float32 for the left image, 0 = none.
Inputs are RGB uint8 HxWx3, already rectified.
"""
import contextlib
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def _purge(prefixes=("core", "utils", "depth_anything_v2", "Utils")):
    for m in list(sys.modules):
        if m.split(".")[0] in prefixes:
            del sys.modules[m]


@contextlib.contextmanager
def _repo(path, extra=()):
    _purge()
    old_cwd, old_path = os.getcwd(), list(sys.path)
    sys.path[:0] = [path] + [os.path.join(path, e) for e in extra]
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _to_tensor(img, device):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(device)


def _pad(x, mult=32):
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="replicate"), (h, w, ph // 2, pw // 2)


def _unpad(d, info):
    h, w, t, l = info
    return d[..., t:t + h, l:l + w]


def _strip_module(sd):
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class _Net:
    amp = False

    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def __call__(self, left, right, max_disp=256):
        l, info = _pad(_to_tensor(left, self.device))
        r, _ = _pad(_to_tensor(right, self.device))
        with torch.autocast("cuda", enabled=self.amp and self.device.startswith("cuda")):
            d = self.forward(l, r)
        d = _unpad(d.float(), info)[0, 0].cpu().numpy()
        d[~np.isfinite(d) | (d < 0)] = 0
        return d.astype(np.float32)


class RAFTStereo(_Net):
    def __init__(self, models_root, device="cuda", weights="sceneflow", iters=32):
        super().__init__(device)
        repo = os.path.join(models_root, "RAFT-Stereo")
        args = SimpleNamespace(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False,
                               corr_levels=4, corr_radius=4, n_downsample=2, context_norm="batch",
                               slow_fast_gru=False, n_gru_layers=3, mixed_precision=False)
        with _repo(repo, ["core"]):
            from raft_stereo import RAFTStereo as Net
            self.model = Net(args)
        sd = _load(os.path.join(repo, "models", f"raftstereo-{weights}.pth"))
        self.model.load_state_dict(_strip_module(sd))
        self.model.to(device).eval()
        self.iters, self.name = iters, f"raft_stereo_{weights}"

    def forward(self, l, r):
        _, flow = self.model(l, r, iters=self.iters, test_mode=True)
        return -flow


class IGEVStereo(_Net):
    def __init__(self, models_root, device="cuda", weights="sceneflow", iters=32, max_disp=192):
        super().__init__(device)
        repo = os.path.join(models_root, "IGEV", "IGEV-Stereo")
        self.args = SimpleNamespace(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False,
                                    corr_levels=2, corr_radius=4, n_downsample=2, slow_fast_gru=False,
                                    n_gru_layers=3, max_disp=max_disp, mixed_precision=False,
                                    precision_dtype="float32")
        with _repo(repo, ["core"]):
            from igev_stereo import IGEVStereo as Net
            self.model = Net(self.args)
        ck = sorted(glob.glob(os.path.join(models_root, "weights", "igev", "**", f"*{weights}*.pth"), recursive=True))
        if not ck:
            raise FileNotFoundError(f"IGEV {weights} checkpoint not found under {models_root}/weights/igev")
        self.model.load_state_dict(_strip_module(_load(ck[0])))
        self.model.to(device).eval()
        self.iters, self.name = iters, f"igev_stereo_{weights}"

    def forward(self, l, r):
        return self.model(l, r, iters=self.iters, test_mode=True)


class FoundationStereo(_Net):
    amp = True

    def __init__(self, models_root, device="cuda", variant="23-51-11", iters=32, hiera=None):
        super().__init__(device)
        repo = os.path.join(models_root, "FoundationStereo")
        wdir = os.path.join(models_root, "weights", "foundationstereo", variant)
        ckpt = sorted(glob.glob(os.path.join(wdir, "*.pth")))[0]
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(os.path.join(wdir, "cfg.yaml"))
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = iters
        with _repo(repo):
            from core.foundation_stereo import FoundationStereo as Net
            self.model = Net(cfg)
        self.model.load_state_dict(_load(ckpt)["model"])
        self.model.to(device).eval()
        self.iters, self.hiera, self.name = iters, hiera, "foundation_stereo" + ("" if variant == "23-51-11" else "_" + variant)

    def forward(self, l, r):
        big = max(l.shape[-2:]) > 1100 if self.hiera is None else self.hiera
        if big:
            return self.model.run_hierachical(l, r, iters=self.iters, test_mode=True, small_ratio=0.5)
        return self.model.forward(l, r, iters=self.iters, test_mode=True)


class MonSter(_Net):
    def __init__(self, models_root, device="cuda", weights="sceneflow", iters=32, max_disp=192):
        super().__init__(device)
        repo = os.path.join(models_root, "MonSter")
        args = SimpleNamespace(encoder="vitl", hidden_dims=[128] * 3, corr_implementation="reg",
                               shared_backbone=False, corr_levels=2, corr_radius=4, n_downsample=2,
                               slow_fast_gru=False, n_gru_layers=3, max_disp=max_disp, mixed_precision=False,
                               restore_ckpt=None)
        with _repo(repo, ["core"]):                     # Monster() reads ./pretrained/depth_anything_v2_vitl.pth
            from monster import Monster as Net
            self.model = Net(args)
        sd = _load(os.path.join(models_root, "weights", "monster", f"{weights}.pth"))
        sd = sd.get("state_dict", sd)
        self.model.load_state_dict(_strip_module(sd), strict=True)
        self.model.to(device).eval()
        self.iters, self.name = iters, f"monster_{weights}"

    def forward(self, l, r):
        return self.model(l, r, iters=self.iters, test_mode=True)


def build(name, models_root, device="cuda", max_disp=192):
    """name: raft_stereo_sceneflow | raft_stereo_middlebury | igev_stereo_sceneflow | foundation_stereo |
    monster_sceneflow | monster_mix_all"""
    md = int(max(192, np.ceil(max_disp / 32) * 32))
    if name.startswith("raft_stereo_"):
        return RAFTStereo(models_root, device, name[len("raft_stereo_"):])
    if name.startswith("igev_stereo_"):
        return IGEVStereo(models_root, device, name[len("igev_stereo_"):], max_disp=md)
    if name.startswith("foundation_stereo"):
        return FoundationStereo(models_root, device)
    if name.startswith("monster_"):
        return MonSter(models_root, device, name[len("monster_"):], max_disp=md)
    raise ValueError(name)


ALL = ["raft_stereo_sceneflow", "raft_stereo_middlebury", "igev_stereo_sceneflow", "foundation_stereo",
       "monster_sceneflow", "monster_mix_all"]
