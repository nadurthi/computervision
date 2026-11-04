"""Classical plane-sweep MVS (Collins 1996; Gallup et al. 2007) with ZNCC photo-consistency, in PyTorch.

For each fronto-parallel depth plane z_k (uniform in inverse depth) every source image is warped into the reference
through the plane-induced homography, a windowed ZNCC cost c_s(p, k) = 1 - ZNCC is computed per source, and the
costs are aggregated across sources:
  * "mean"  : average over all sources that see the pixel (the textbook N-view cost)
  * "bestk" : average of the K lowest costs (Kang, Szeliski & Chai 2001) - an occlusion-robust aggregation: a source
              that is occluded at p produces a high cost and simply gets ignored
Depth is the winner-take-all minimum with parabolic sub-plane refinement, optionally after semi-global
aggregation (Hirschmuller 2008) along 4 scanline directions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def _gray(img, device):
    g = torch.from_numpy(np.ascontiguousarray(img)).float().to(device)
    g = (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]) / 255.0
    return g[None, None]                                         # 1,1,H,W


def _box(x, r):
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def zncc_cost(ref, warped, valid, r=3, eps=1e-4):
    """ref, warped: (B,1,H,W); valid: (B,1,H,W) bool. Returns 1-ZNCC in [0,2]; 2 where invalid/textureless."""
    mr, ms = _box(ref, r), _box(warped, r)
    vr = _box(ref * ref, r) - mr * mr
    vs = _box(warped * warped, r) - ms * ms
    cov = _box(ref * warped, r) - mr * ms
    ncc = cov / torch.sqrt(torch.clamp(vr, min=0) * torch.clamp(vs, min=0) + eps)
    cost = 1.0 - ncc.clamp(-1, 1)
    vfrac = _box(valid.float(), r)
    return torch.where(vfrac > 0.99, cost, torch.full_like(cost, 2.0))


def sgm(cost, p1=0.05, p2=0.3):
    """Semi-global aggregation over 4 directions. cost: (D,H,W). Returns summed path costs (D,H,W)."""
    D, H, W = cost.shape
    total = torch.zeros_like(cost)
    for dim, rev in ((2, False), (2, True), (1, False), (1, True)):
        c = cost.flip(dim) if rev else cost
        n = c.shape[dim]
        L = torch.empty_like(c)
        prev = c.select(dim, 0).clone()
        L.select(dim, 0).copy_(prev)
        for i in range(1, n):
            cur = c.select(dim, i)
            pmin = prev.min(0, keepdim=True).values
            up = F.pad(prev[1:], (0, 0, 0, 1), value=float("inf"))
            dn = F.pad(prev[:-1], (0, 0, 1, 0), value=float("inf"))
            m = torch.minimum(torch.minimum(prev, torch.minimum(up, dn) + p1), pmin + p2)
            prev = cur + m - pmin
            L.select(dim, i).copy_(prev)
        total += L.flip(dim) if rev else L
    return total


class PlaneSweep:
    def __init__(self, num_planes=192, window_radius=3, aggregation="bestk", k=2, use_sgm=False, device="cuda",
                 p1=0.05, p2=0.3):
        self.D, self.r, self.agg, self.k, self.use_sgm, self.device = num_planes, window_radius, aggregation, k, use_sgm, device
        self.p1, self.p2 = p1, p2
        self.name = f"planesweep_{aggregation}" + ("_sgm" if use_sgm else "")

    @torch.no_grad()
    def __call__(self, images, Ks, Es, depth_range):
        dev = self.device
        h, w = images[0].shape[:2]
        ref = _gray(images[0], dev)
        dmin, dmax = depth_range
        inv = torch.linspace(1.0 / dmax, 1.0 / dmin, self.D, device=dev)          # far -> near
        depths = 1.0 / inv
        u, v = torch.meshgrid(torch.arange(w, device=dev, dtype=torch.float32),
                              torch.arange(h, device=dev, dtype=torch.float32), indexing="xy")
        pix = torch.stack([u.reshape(-1), v.reshape(-1), torch.ones(h * w, device=dev)])  # 3,P
        Kr_inv = torch.from_numpy(np.linalg.inv(Ks[0])).float().to(dev)
        rays = Kr_inv @ pix
        costs = []
        for img, K, E in zip(images[1:], Ks[1:], Es[1:]):
            T = torch.from_numpy(E @ np.linalg.inv(Es[0])).float().to(dev)
            Kt = torch.from_numpy(K).float().to(dev)
            A = Kt @ (T[:3, :3] @ rays)                                         # 3,P
            b = Kt @ T[:3, 3:4]                                                 # 3,1
            src = _gray(img, dev)
            hs, ws = src.shape[-2:]
            cs = torch.empty(self.D, h, w, device=dev)
            for k0 in range(0, self.D, 32):
                z = depths[k0:k0 + 32]
                p = z[:, None, None] * A[None] + b[None]                        # B,3,P
                zz = p[:, 2]
                x = p[:, 0] / zz.clamp(min=1e-6)
                y = p[:, 1] / zz.clamp(min=1e-6)
                grid = torch.stack([2 * x / (ws - 1) - 1, 2 * y / (hs - 1) - 1], -1).view(-1, h, w, 2)
                warped = F.grid_sample(src.expand(len(z), -1, -1, -1), grid, align_corners=True, padding_mode="zeros")
                valid = ((zz > 0) & (x >= 0) & (x <= ws - 1) & (y >= 0) & (y <= hs - 1)).view(-1, 1, h, w)
                cs[k0:k0 + len(z)] = zncc_cost(ref.expand(len(z), -1, -1, -1), warped, valid, self.r)[:, 0]
            costs.append(cs)
        C = torch.stack(costs)                                                  # S,D,H,W
        if self.agg == "mean" or C.shape[0] == 1:
            seen = (C < 2.0).float()
            vol = (C * seen).sum(0) / seen.sum(0).clamp(min=1)
            vol = torch.where(seen.sum(0) > 0, vol, torch.full_like(vol, 2.0))
        else:
            k = min(self.k, C.shape[0])
            vol = torch.topk(C, k, dim=0, largest=False).values.mean(0)
        if self.use_sgm:
            vol = sgm(vol, self.p1, self.p2)
        idx = vol.argmin(0)
        # parabolic sub-plane refinement in inverse depth
        i0 = idx.clamp(1, self.D - 2)
        c0 = vol.gather(0, (i0 - 1)[None])[0]
        c1 = vol.gather(0, i0[None])[0]
        c2 = vol.gather(0, (i0 + 1)[None])[0]
        den = c0 - 2 * c1 + c2
        off = torch.where(den > 1e-6, 0.5 * (c0 - c2) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        fidx = torch.where((idx > 0) & (idx < self.D - 1), i0.float() + off, idx.float())
        step = (inv[-1] - inv[0]) / (self.D - 1)
        depth = 1.0 / (inv[0] + fidx * step)
        best = vol.min(0).values
        depth = torch.where(best < (1.95 if not self.use_sgm else math.inf), depth, torch.zeros_like(depth))
        return depth.cpu().numpy().astype(np.float32)
