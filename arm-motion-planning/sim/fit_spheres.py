"""Fit collision spheres to the Panda URDF collision meshes -> panda_spheres.json.

Run inside the container (the meshes come with moveit_resources_panda_description):
    ./run_local.sh python /work/sim/fit_spheres.py
k-means (k=8 per link, 4 for link7) on 6000 area-weighted surface samples per mesh; each sphere's
radius is the farthest sample assigned to it plus 4 mm. The hand group includes both fingers at
finger joint = 0, as MoveIt's default state has them.
"""
import json
import os
import struct
from pathlib import Path

import numpy as np

MESH = Path(os.environ.get("CONDA_PREFIX", "/opt/conda/envs/ros")) / "share/moveit_resources_panda_description/meshes/collision"


def load(p):
    b = open(p, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3, 3).astype(float)


def surface(T, k=6000, rng=np.random.default_rng(0)):
    A = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    i = rng.choice(len(T), k, p=A / A.sum())
    u, v = rng.random(k), rng.random(k)
    m = u + v > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    return T[i, 0] + u[:, None] * (T[i, 1] - T[i, 0]) + v[:, None] * (T[i, 2] - T[i, 0])


def kmeans(P, k, it=50):
    c = P.mean(0)
    d = np.linalg.svd(P - c)[2][0]
    t = (P - c) @ d
    C = np.array([P[np.argmin(abs(t - q))] for q in np.quantile(t, np.linspace(0.05, 0.95, k))])
    for _ in range(it):
        L = np.argmin(((P[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([P[L == j].mean(0) if (L == j).any() else C[j] for j in range(k)])
    L = np.argmin(((P[:, None] - C[None]) ** 2).sum(-1), 1)
    return C, np.array([np.linalg.norm(P[L == j] - C[j], axis=1).max() for j in range(k)])


def main():
    hand, finger = load(MESH / "hand.stl"), load(MESH / "finger.stl")
    left = finger + [0, 0, 0.0584]
    right = finger * [-1, -1, 1] + [0, 0, 0.0584]  # rotated pi about z
    parts = {f"link{i}": load(MESH / f"link{i}.stl") for i in range(1, 8)}
    parts["hand"] = np.concatenate([hand, left, right])
    out = {}
    for name, T in parts.items():
        C, R = kmeans(surface(T), 4 if name == "link7" else 8)
        out[name] = [[*map(float, np.round(c, 4)), round(float(r) + 0.004, 4)] for c, r in zip(C, R)]
    p = Path(__file__).with_name("panda_spheres.json")
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p, sum(len(v) for v in out.values()), "spheres")


if __name__ == "__main__":
    main()
