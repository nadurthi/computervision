"""Pallet with stacked totes: layout, ground-truth poses, pick order.

World frame: z up, the lidar looks along +x at the long (1200 mm) side of the pallet. Each layer holds 2 x 2
totes: two along world y (600 mm sides facing the lidar) and two deep along x. Tote front (-y in the tote
frame) is rotated to world -x, i.e. R = Rz(-90 deg).
"""
import json
import numpy as np
from .cad import ToteSpec, front_face, PALLET_TOP

R_TOTE = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float)     # Rz(-90): tote x -> world -y, tote y -> world +x


def rotz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_stack(layers=7, pallet_x=2.0, gap=0.003, jitter_xy=0.002, jitter_yaw=0.8, seed=0, spec=ToteSpec()):
    """Returns a list of dicts: id, layer, row (0 front, 1 back), col (0 = +y, 1 = -y), T (4x4 tote->world)."""
    rng = np.random.default_rng(seed)
    px, py = spec.rim_w + gap, spec.rim_l + gap               # pitch along world x (depth) and y
    totes = []
    k = 0
    for L in range(layers):
        for row in (0, 1):
            for col in (0, 1):
                T = np.eye(4)
                T[:3, :3] = rotz(rng.uniform(-jitter_yaw, jitter_yaw)) @ R_TOTE
                T[:3, 3] = (pallet_x + (row - 0.5) * px + rng.uniform(-jitter_xy, jitter_xy),
                            (0.5 - col) * py + rng.uniform(-jitter_xy, jitter_xy),
                            PALLET_TOP + L * spec.height)
                totes.append(dict(id=k, layer=L, row=row, col=col, T=T))
                k += 1
    return totes


def pick_order(totes):
    """Depalletise top layer first, front row before back row."""
    return [t["id"] for t in sorted(totes, key=lambda t: (-t["layer"], t["row"], t["col"]))]


def front_center_world(T, spec=ToteSpec(), cad_front=None):
    c, n = cad_front if cad_front is not None else front_face(spec)
    return T[:3, :3] @ c + T[:3, 3], T[:3, :3] @ n


def save_gt(path, totes, present_ids, sensor, spec=ToteSpec(), cad_front=None, extra=None):
    out = dict(sensor_position=list(map(float, sensor)), totes=[])
    for t in totes:
        c, n = front_center_world(t["T"], spec, cad_front)
        out["totes"].append(dict(id=t["id"], layer=t["layer"], row=t["row"], col=t["col"],
                                 present=t["id"] in present_ids, T=t["T"].tolist(),
                                 front_center=c.tolist(), front_normal=n.tolist()))
    if extra:
        out.update(extra)
    json.dump(out, open(path, "w"), indent=1)
