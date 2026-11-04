"""Classical PatchMatch MVS with slanted planes and occlusion-robust view selection, in PyTorch.

Follows the Gipuma / ACMH family (Galliani et al. 2015; Xu & Tao 2019; Schoenberger et al. 2016):
  * every reference pixel p carries a plane hypothesis (depth d_p, normal n_p) in the reference camera frame
  * the cost of a hypothesis for source s is 1 - bilateral-weighted NCC over a sparse window, where each window pixel
    q is mapped into s by the plane-induced homography  H = K_s (R + t n^T / rho) K_r^-1,  rho = n^T X_p
  * multi-view cost = mean of the K best source costs (occluded sources drop out)
  * iterations alternate propagation (neighbour planes re-evaluated at p) and random refinement (perturbed planes)
Updates are applied in red/black checkerboard order so half of the pixels always propagate fresh hypotheses.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def _gray(img, dev):
    g = torch.from_numpy(np.ascontiguousarray(img)).float().to(dev)
    return ((0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]) / 255.0)[None, None]


def _sample(img, x, y):
    """img (1,1,H,W); x,y (...) pixel coords -> values (...), zero outside."""
    h, w = img.shape[-2:]
    g = torch.stack([2 * x / (w - 1) - 1, 2 * y / (h - 1) - 1], -1).reshape(1, -1, 1, 2)
    return F.grid_sample(img, g, align_corners=True, padding_mode="border").reshape(x.shape)


class PatchMatchMVS:
    NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-5, 0), (5, 0), (0, -5), (0, 5)]

    def __init__(self, iterations=6, window=11, step=2, k_best=2, sigma_color=0.2, sigma_space=3.0,
                 device="cuda", seed=0):
        self.iters, self.k, self.dev = iterations, k_best, device
        half = (window // 2) // step * step                 # symmetric grid that contains the centre pixel
        offs =[(dx, dy) for dy in range(-half, half + 1, step) for dx in range(-half, half + 1, step)]
        self.offs = torch.tensor(offs, dtype=torch.float32, device=device)            # S,2
        self.sc, self.ss = sigma_color, sigma_space
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.name = f"patchmatch_k{k_best}"

    # -------------------------------------------------------------------------------------------- helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.dev)

    def _random_normals(self, rays):
        n = torch.randn(rays.shape, generator=self.gen, device=self.dev)
        n = n / n.norm(dim=-1, keepdim=True)
        flip = (n * rays).sum(-1, keepdim=True) > 0
        n = torch.where(flip, -n, n)
        return n

    def _setup(self, images, Ks, Es):
        dev = self.dev
        self.h, self.w = images[0].shape[:2]
        h, w = self.h, self.w
        self.ref = _gray(images[0], dev)
        u, v = torch.meshgrid(torch.arange(w, device=dev, dtype=torch.float32),
                              torch.arange(h, device=dev, dtype=torch.float32), indexing="xy")
        self.px = torch.stack([u.reshape(-1), v.reshape(-1)], -1)                       # P,2
        Kinv = torch.from_numpy(np.linalg.inv(Ks[0])).float().to(dev)
        self.Kinv = Kinv
        self.rays = (torch.cat([self.px, torch.ones_like(self.px[:, :1])], 1) @ Kinv.T)  # P,3 (z=1)
        # window samples in the reference + bilateral weights (fixed for the whole run)
        q = self.px[:, None, :] + self.offs[None]                                       # P,S,2
        self.q = q
        self.ray_q = torch.cat([q, torch.ones_like(q[..., :1])], -1) @ Kinv.T           # P,S,3
        rq = _sample(self.ref, q[..., 0], q[..., 1])                                    # P,S
        rc = _sample(self.ref, self.px[:, 0], self.px[:, 1])[:, None]
        wgt = torch.exp(-(rq - rc).abs() / self.sc - self.offs.norm(dim=-1)[None] / self.ss)
        self.wgt = wgt / wgt.sum(1, keepdim=True)
        m = (self.wgt * rq).sum(1, keepdim=True)
        self.rq_c = rq - m
        self.rq_var = (self.wgt * self.rq_c ** 2).sum(1)
        self.srcs = []
        for img, K, E in zip(images[1:], Ks[1:], Es[1:]):
            T = torch.from_numpy(E @ np.linalg.inv(Es[0])).float().to(dev)
            Kt = torch.from_numpy(K).float().to(dev)
            self.srcs.append(dict(img=_gray(img, dev), M=Kt @ T[:3, :3], b=Kt @ T[:3, 3]))

    def _cost(self, d, n, idx=None):
        """Multi-view cost of planes (d: P', n: P',3) for pixels idx (or all). Returns (P',)."""
        rays = self.rays if idx is None else self.rays[idx]
        ray_q = self.ray_q if idx is None else self.ray_q[idx]
        wgt = self.wgt if idx is None else self.wgt[idx]
        rq_c = self.rq_c if idx is None else self.rq_c[idx]
        rq_var = self.rq_var if idx is None else self.rq_var[idx]
        rho = d * (n * rays).sum(-1)                                                   # n.X_p
        nq = (ray_q * n[:, None]).sum(-1) / rho[:, None]                                # (n.ray_q)/rho
        costs = []
        for s in self.srcs:
            hs, ws = s["img"].shape[-2:]
            ph = ray_q @ s["M"].T + nq[..., None] * s["b"]                              # P',S,3
            z = ph[..., 2]
            x, y = ph[..., 0] / z.clamp(min=1e-6), ph[..., 1] / z.clamp(min=1e-6)
            sv = _sample(s["img"], x, y)
            m = (wgt * sv).sum(1, keepdim=True)
            sc = sv - m
            cov = (wgt * rq_c * sc).sum(1)
            var = (wgt * sc ** 2).sum(1)
            ncc = cov / torch.sqrt(rq_var * var + 1e-8)
            c = 1 - ncc.clamp(-1, 1)
            xc, yc, zc = x[:, x.shape[1] // 2], y[:, y.shape[1] // 2], z[:, z.shape[1] // 2]
            inside = (zc > 0) & (xc >= 0) & (xc <= ws - 1) & (yc >= 0) & (yc <= hs - 1)
            costs.append(torch.where(inside & torch.isfinite(c), c, torch.full_like(c, 2.0)))
        C = torch.stack(costs)                                                         # V,P'
        k = min(self.k, C.shape[0])
        return torch.topk(C, k, dim=0, largest=False).values.mean(0)

    def _plane_depth_at(self, d_q, n_q, ray_q_center, rays_p):
        """Depth at ray p of the plane through X_q = d_q*ray_q with normal n_q."""
        rho = d_q * (n_q * ray_q_center).sum(-1)
        den = (n_q * rays_p).sum(-1)
        return torch.where(den.abs() > 1e-6, rho / den, torch.zeros_like(den))

    # -------------------------------------------------------------------------------------------- main
    @torch.no_grad()
    def __call__(self, images, Ks, Es, depth_range, return_normals=False):
        self._setup(images, Ks, Es)
        dmin, dmax = depth_range
        P = self.h * self.w
        inv = 1 / dmax + self._rand(P) * (1 / dmin - 1 / dmax)
        d = 1 / inv
        n = self._random_normals(self.rays)
        cost = self._cost(d, n)
        yy = (self.px[:, 1].long() + self.px[:, 0].long()) % 2
        colours = [torch.nonzero(yy == c).squeeze(1) for c in (0, 1)]
        for it in range(self.iters):
            for idx in colours:
                # propagation: evaluate the planes of 8 neighbours at the pixel
                px = self.px[idx]
                for dx, dy in self.NEIGHBOURS:
                    qx = (px[:, 0] + dx).clamp(0, self.w - 1).long()
                    qy = (px[:, 1] + dy).clamp(0, self.h - 1).long()
                    j = qy * self.w + qx
                    dn = self._plane_depth_at(d[j], n[j], self.rays[j], self.rays[idx])
                    ok = (dn >= dmin) & (dn <= dmax)
                    c_new = self._cost(dn.clamp(dmin, dmax), n[j], idx)
                    better = ok & (c_new < cost[idx])
                    sel = idx[better]
                    d[sel], n[sel], cost[sel] = dn[better], n[j][better], c_new[better]
                # refinement: perturb depth and normal with a shrinking radius
                scale = 0.5 ** it
                cands = []
                inv_p = 1 / d[idx] + (self._rand(len(idx)) - 0.5) * scale * (1 / dmin - 1 / dmax) * 0.25
                d_pert = (1 / inv_p.clamp(1 / dmax, 1 / dmin))
                n_pert = n[idx] + torch.randn(n[idx].shape, generator=self.gen, device=self.dev) * 0.3 * scale
                n_pert = n_pert / n_pert.norm(dim=-1, keepdim=True)
                n_pert = torch.where((n_pert * self.rays[idx]).sum(-1, keepdim=True) > 0, -n_pert, n_pert)
                d_rand = 1 / (1 / dmax + self._rand(len(idx)) * (1 / dmin - 1 / dmax))
                n_rand = self._random_normals(self.rays[idx])
                cands = [(d_pert, n[idx]), (d[idx], n_pert), (d_pert, n_pert), (d_rand, n[idx]), (d[idx], n_rand)]
                for dc, nc in cands:
                    c_new = self._cost(dc, nc, idx)
                    better = c_new < cost[idx]
                    sel = idx[better]
                    d[sel], n[sel], cost[sel] = dc[better], nc[better], c_new[better]
        depth = d.view(self.h, self.w)
        depth = torch.where(cost.view(self.h, self.w) < 1.5, depth, torch.zeros_like(depth))
        out = depth.cpu().numpy().astype(np.float32)
        if return_normals:
            return out, n.view(self.h, self.w, 3).cpu().numpy(), cost.view(self.h, self.w).cpu().numpy()
        return out
