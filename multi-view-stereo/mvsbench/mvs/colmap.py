"""COLMAP PatchMatch stereo (Schoenberger, Zheng, Pollefeys, Frahm; ECCV 2016): the de-facto classical MVS
reference. Joint per-pixel depth/normal/view-selection inference with a probabilistic (GEM) occlusion model,
photometric pass followed by a geometric-consistency pass. Uses the CUDA `colmap` binary (installed in the
`mvsa` micromamba env by setup/install_models.sh) on a temporary workspace with the known cameras.
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def rotmat2qvec(R):
    m = R
    K = np.array([
        [m[0, 0] - m[1, 1] - m[2, 2], 0, 0, 0],
        [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], 0, 0],
        [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], 0],
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def read_colmap_array(path):
    with open(path, "rb") as f:
        header = b""
        while header.count(b"&") < 3:
            header += f.read(1)
        w, h, c = [int(x) for x in header.split(b"&")[:3]]
        data = np.fromfile(f, np.float32)
    return np.transpose(data.reshape((w, h, c), order="F"), (1, 0, 2)).squeeze()


class COLMAP:
    name = "colmap"

    def __init__(self, binary="colmap", geom_consistency=True, timeout=3600):
        self.bin, self.geom, self.timeout = binary, geom_consistency, timeout

    def run_scene(self, images, Ks, Es, ranges, pairs):
        ws = tempfile.mkdtemp(prefix="colmap_")
        try:
            for sub in ("images", "sparse", "stereo/depth_maps", "stereo/normal_maps", "stereo/consistency_graphs"):
                os.makedirs(os.path.join(ws, sub))
            h, w = images[0].shape[:2]
            with open(os.path.join(ws, "sparse", "cameras.txt"), "w") as fc, \
                    open(os.path.join(ws, "sparse", "images.txt"), "w") as fi:
                for i, (img, K, E) in enumerate(zip(images, Ks, Es)):
                    name = f"{i:08d}.jpg"
                    cv2.imwrite(os.path.join(ws, "images", name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 98])
                    fc.write(f"{i + 1} PINHOLE {img.shape[1]} {img.shape[0]} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
                    q = rotmat2qvec(E[:3, :3])
                    t = E[:3, 3]
                    fi.write(f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n\n")
            open(os.path.join(ws, "sparse", "points3D.txt"), "w").close()
            todo = {r: s for r, s in pairs}
            if self.geom:                                  # geometric consistency needs the sources' depth maps
                for r, s in pairs:
                    for j in s:
                        todo.setdefault(j, [r] + [k for k in s if k != j])
            with open(os.path.join(ws, "stereo", "patch-match.cfg"), "w") as f:
                for r, s in todo.items():
                    f.write(f"{r:08d}.jpg\n" + ", ".join(f"{j:08d}.jpg" for j in s) + "\n")
            dmin = min(r[0] for r in ranges)
            dmax = max(r[1] for r in ranges)
            subprocess.run([self.bin, "patch_match_stereo", "--workspace_path", ws, "--workspace_format", "COLMAP",
                            "--PatchMatchStereo.geom_consistency", "true" if self.geom else "false",
                            "--PatchMatchStereo.depth_min", str(dmin), "--PatchMatchStereo.depth_max", str(dmax),
                            "--PatchMatchStereo.max_image_size", str(max(h, w))],
                           check=True, timeout=self.timeout, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            kind = "geometric" if self.geom else "photometric"
            out = {}
            for r, _ in pairs:
                d = read_colmap_array(os.path.join(ws, "stereo", "depth_maps", f"{r:08d}.jpg.{kind}.bin"))
                if d.shape != (h, w):
                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
                d[~np.isfinite(d) | (d <= 0)] = 0
                out[r] = d.astype(np.float32)
            return out
        finally:
            shutil.rmtree(ws, ignore_errors=True)
