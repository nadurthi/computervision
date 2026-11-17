"""CPU stand-in for the Isaac Sim RGB-D camera mounted beside the lidar.

Pinhole model, optical axis along world +x (toward the stack), image u to the right (world -y), v down (world -z):
    pixel (u, v) <- K [R_wc^T (X - C)],   K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
The renderer ray-casts the same scene as the lidar and shades it with a Lambertian model (grey totes, wooden
pallet, concrete floor, one head lamp at the camera plus a ceiling light) and adds sensor noise. Outputs:
rgb (H,W,3 uint8), depth (H,W float32, metres along the optical axis), instance labels (tote id / -1 / -2 / -3).
"""
import numpy as np
import open3d as o3d

# camera axes in the world: x_c (image right) = -y_w, y_c (image down) = -z_w, z_c (optical axis) = +x_w
R_WC = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)

ALBEDO = {"tote": np.array([0.46, 0.47, 0.48]), -1: np.array([0.55, 0.42, 0.28]), -2: np.array([0.30, 0.30, 0.30]),
          -3: np.array([0.05, 0.05, 0.06])}


def intrinsics(width=1600, height=1200, vfov_deg=80.0):
    f = (height / 2) / np.tan(np.radians(vfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def render(scene, labels, C, K, width=1600, height=1200, rng=None, noise=2.0, R_wc=R_WC):
    rng = rng or np.random.default_rng(0)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pix = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3)
    dc = pix @ np.linalg.inv(K).T                                 # camera-frame rays, z = 1
    dw = dc @ R_wc.T
    dwn = dw / np.linalg.norm(dw, axis=1, keepdims=True)
    o = np.broadcast_to(np.asarray(C, np.float32), dwn.shape)
    ans = scene.cast_rays(o3d.core.Tensor(np.concatenate([o, dwn], 1).astype(np.float32)))
    t = ans["t_hit"].numpy(); gid = ans["geometry_ids"].numpy(); n = ans["primitive_normals"].numpy()
    hit = np.isfinite(t)
    lab = np.array([labels.get(int(g), -3) for g in gid]); lab[~hit] = -3
    alb = np.where((lab >= 0)[:, None], ALBEDO["tote"], 0)
    for k in (-1, -2, -3):
        alb = np.where((lab == k)[:, None], ALBEDO[k], alb)
    n = np.where((np.sum(n * dwn, 1) > 0)[:, None], -n, n)           # face the camera
    lamp = np.clip(-(n * dwn).sum(1), 0, 1)                           # head lamp at the camera
    ceil = np.clip(n[:, 2], 0, 1) * 0.35 + 0.25                        # ceiling light + ambient
    X = np.asarray(C) + dwn * np.where(hit, t, 0)[:, None]
    fall = 1.0 / (1.0 + 0.15 * np.where(hit, t, 10) ** 2)
    shade = (0.75 * lamp * fall * 2.2 + ceil)[:, None]
    img = np.clip(alb * shade, 0, 1) * 255 + rng.normal(0, noise, (len(t), 3))
    img[~hit] = 12
    depth = np.where(hit, t * (dwn @ R_wc[:, 2]), np.inf)              # z-depth along the optical axis
    return (np.clip(img, 0, 255).astype(np.uint8).reshape(height, width, 3),
            depth.astype(np.float32).reshape(height, width), lab.astype(np.int16).reshape(height, width))


def project(Xw, C, K, R_wc=R_WC):
    Xc = (np.asarray(Xw) - C) @ R_wc
    uv = Xc @ K.T
    return uv[..., :2] / uv[..., 2:3], Xc[..., 2]
