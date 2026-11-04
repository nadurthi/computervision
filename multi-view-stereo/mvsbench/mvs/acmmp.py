"""ACMMP (Xu, Kong, Zuo, Tao, Zhang; TPAMI 2022) - multi-scale geometric consistency guided and planar prior
assisted PatchMatch MVS, the strongest open-source classical CUDA MVS. Runs the official binary built by
setup/install_models.sh on a temporary MVSNet-style folder that contains only the views we ask for.
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from ..io import read_dmb, write_mvsnet_cam, write_pair


class ACMMP:
    name = "acmmp"

    def __init__(self, binary, keep_tmp=False, timeout=3600):
        self.binary, self.keep, self.timeout = binary, keep_tmp, timeout

    def run_scene(self, images, Ks, Es, ranges, pairs):
        """images/Ks/Es/ranges: lists indexed 0..N-1; pairs: [(ref, [srcs])] in that index space.
        Returns {ref: depth (H,W) at the input resolution}."""
        tmp = tempfile.mkdtemp(prefix="acmmp_")
        try:
            os.makedirs(os.path.join(tmp, "images"))
            os.makedirs(os.path.join(tmp, "cams"))
            for i, (img, K, E, (dmin, dmax)) in enumerate(zip(images, Ks, Es, ranges)):
                cv2.imwrite(os.path.join(tmp, "images", f"{i:08d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 98])
                write_mvsnet_cam(os.path.join(tmp, "cams", f"{i:08d}_cam.txt"), K, E, dmin, dmax)
            # ACMMP indexes problems by position in pair.txt and expects every listed id to have its own entry
            full = {r: s for r, s in pairs}
            for i in range(len(images)):
                full.setdefault(i, [j for j in range(len(images)) if j != i][:4])
            write_pair(os.path.join(tmp, "pair.txt"), [(i, full[i]) for i in range(len(images))])
            subprocess.run([self.binary, tmp], check=True, timeout=self.timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            out = {}
            h, w = images[0].shape[:2]
            for r, _ in pairs:
                f = os.path.join(tmp, "ACMMP", f"2333_{r:08d}", "depths_geom.dmb")
                if not os.path.exists(f):
                    f = os.path.join(tmp, "ACMMP", f"2333_{r:08d}", "depths.dmb")
                d = read_dmb(f).astype(np.float32)
                if d.shape != (h, w):
                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
                d[~np.isfinite(d) | (d <= 0)] = 0
                out[r] = d
            return out
        finally:
            if not self.keep:
                shutil.rmtree(tmp, ignore_errors=True)
