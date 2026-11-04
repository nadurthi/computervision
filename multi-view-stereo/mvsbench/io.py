"""File formats used across stereo and MVS datasets and tools.

PFM (Middlebury / MVSNet), MVSNet camera files and pair.txt, ETH3D raw depth, COLMAP text models (ETH3D),
KITTI 16-bit disparity PNGs, ACMMP .dmb.
"""
import os
import re
import struct

import cv2
import numpy as np


# ------------------------------------------------------------------------------------------------ PFM
def read_pfm(path):
    """Read a PFM file -> float32 array (H,W) or (H,W,3), top row first."""
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"not a PFM file: {path}")
        color = header == "PF"
        dims = re.match(r"^(\d+)\s+(\d+)\s*$", f.readline().decode("latin-1"))
        w, h = int(dims.group(1)), int(dims.group(2))
        scale = float(f.readline().decode("latin-1").rstrip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
    shape = (h, w, 3) if color else (h, w)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def write_pfm(path, image):
    image = np.asarray(image, dtype=np.float32)
    color = image.ndim == 3 and image.shape[2] == 3
    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())
        f.write(b"-1\n")                      # little endian
        np.flipud(image).astype("<f").tofile(f)


# ------------------------------------------------------------------------------------------------ MVSNet
def read_mvsnet_cam(path):
    """MVSNet cam.txt -> (K 3x3, E 4x4 world->cam, depth_min, depth_max or None, depth_interval or None)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f.readlines()]
    E = np.array(" ".join(lines[1:5]).split(), dtype=np.float64).reshape(4, 4)
    K = np.array(" ".join(lines[7:10]).split(), dtype=np.float64).reshape(3, 3)
    dmin = dmax = interval = None
    if len(lines) > 11 and lines[11]:
        vals = [float(v) for v in lines[11].split()]
        dmin = vals[0]
        if len(vals) >= 2:
            interval = vals[1]
        if len(vals) >= 4:
            dmax = vals[3]
        elif len(vals) == 3:                     # min interval num
            dmax = vals[0] + vals[1] * (vals[2] - 1)
        elif len(vals) == 2:                     # MVSNet default: 192 planes
            dmax = vals[0] + vals[1] * 191
    return K, E, dmin, dmax, interval


def write_mvsnet_cam(path, K, E, dmin, dmax, num_planes=192):
    interval = (dmax - dmin) / (num_planes - 1)
    with open(path, "w") as f:
        f.write("extrinsic\n")
        for r in np.asarray(E).reshape(4, 4):
            f.write(" ".join(f"{v:.10f}" for v in r) + "\n")
        f.write("\nintrinsic\n")
        for r in np.asarray(K).reshape(3, 3):
            f.write(" ".join(f"{v:.10f}" for v in r) + "\n")
        f.write(f"\n{dmin:.6f} {interval:.6f} {num_planes} {dmax:.6f}\n")


def read_pair(path):
    """pair.txt -> list of (ref_id, [src_ids by score])."""
    out = []
    with open(path) as f:
        n = int(f.readline())
        for _ in range(n):
            ref = int(f.readline())
            toks = f.readline().split()
            srcs = [int(v) for v in toks[1::2]]
            out.append((ref, srcs))
    return out


def write_pair(path, pairs):
    with open(path, "w") as f:
        f.write(f"{len(pairs)}\n")
        for ref, srcs in pairs:
            f.write(f"{ref}\n{len(srcs)} " + " ".join(f"{s} {100.0 - i:.1f}" for i, s in enumerate(srcs)) + "\n")


# ------------------------------------------------------------------------------------------------ ETH3D
def read_eth3d_depth(path, h, w):
    """ETH3D GT depth: raw little-endian float32, row-major, inf = invalid."""
    d = np.fromfile(path, dtype="<f4")
    if d.size != h * w:
        raise ValueError(f"{path}: {d.size} values, expected {h}x{w}")
    d = d.reshape(h, w)
    d[~np.isfinite(d)] = 0
    return d


def read_colmap_cameras_txt(path):
    cams = {}
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        cams[int(t[0])] = dict(model=t[1], w=int(t[2]), h=int(t[3]), params=np.array(t[4:], dtype=np.float64))
    return cams


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap_images_txt(path):
    """-> list of dict(name, camera_id, E 4x4 world->cam), in file order."""
    out, lines = [], [ln for ln in open(path) if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        t = lines[i].split()
        E = np.eye(4)
        E[:3, :3] = qvec2rotmat(np.array(t[1:5], dtype=np.float64))
        E[:3, 3] = np.array(t[5:8], dtype=np.float64)
        out.append(dict(name=t[9], camera_id=int(t[8]), E=E))
        i += 2                                   # skip POINTS2D line
    return out


def colmap_pinhole_K(cam):
    p = cam["params"]
    if cam["model"] == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif cam["model"] in ("SIMPLE_PINHOLE",):
        fx = fy = p[0]
        cx, cy = p[1:3]
    else:
        raise ValueError(f"camera model {cam['model']} is not undistorted")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ------------------------------------------------------------------------------------------------ KITTI
def read_kitti_disp(path):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    return d                                      # 0 = invalid


# ------------------------------------------------------------------------------------------------ ACMMP
def read_dmb(path):
    with open(path, "rb") as f:
        _type, h, w, nb = struct.unpack("<4i", f.read(16))
        data = np.fromfile(f, dtype="<f4", count=h * w * nb)
    return data.reshape(h, w, nb).squeeze(-1) if nb == 1 else data.reshape(h, w, nb)


# ------------------------------------------------------------------------------------------------ misc
def read_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_depth_npz(path, depth, **extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, depth=depth.astype(np.float32), **extra)
