"""Tote CAD model and pallet geometry.

The stand-in tote approximates a 600 x 400 x 300 mm Euro stacking container: tapered walls, a rim flange,
two vertical ribs on each long side, hand-holes on the short sides and a 4 mm wall. Replace it with the real CAD
by passing --cad <file> (STL / OBJ / PLY) to any script; only `front_face()` needs to know the CAD frame.

Tote frame: origin at the centre of the base, x along the 600 mm side, y along the 400 mm side, z up.
The face that points at the lidar is the long side at -y.
"""
from dataclasses import dataclass
import numpy as np
import trimesh


@dataclass
class ToteSpec:
    rim_l: float = 0.600      # rim flange outer size
    rim_w: float = 0.400
    height: float = 0.300     # base to rim top (= stacking pitch)
    rim_h: float = 0.022      # rim flange height
    base_l: float = 0.560     # outer wall at the base
    base_w: float = 0.360
    top_l: float = 0.592      # outer wall just under the rim
    top_w: float = 0.392
    wall: float = 0.004
    floor: float = 0.006
    rib_w: float = 0.030      # vertical ribs on the long sides
    rib_d: float = 0.008
    rib_x: tuple = (-0.15, 0.15)
    hole_w: float = 0.120     # hand-holes in the short sides
    hole_h: float = 0.040
    hole_z: float = 0.235

    @property
    def wall_top(self):
        return self.height - self.rim_h


def _frustum(l0, w0, z0, l1, w1, z1):
    p = []
    for (l, w, z) in ((l0, w0, z0), (l1, w1, z1)):
        p += [(sx * l / 2, sy * w / 2, z) for sx in (-1, 1) for sy in (-1, 1)]
    return trimesh.convex.convex_hull(np.array(p))


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    return b


def wall_half_width(s: ToteSpec, z):
    """Half width (y) of the outer long wall at height z (linear taper)."""
    return s.base_w / 2 + (s.top_w - s.base_w) / 2 * np.clip(z, 0, s.wall_top) / s.wall_top


def make_tote(s: ToteSpec = ToteSpec()) -> trimesh.Trimesh:
    zt = s.wall_top
    outer = _frustum(s.base_l, s.base_w, 0.0, s.top_l, s.top_w, zt + 1e-4)
    rim = _box((-s.rim_l / 2, -s.rim_w / 2, zt), (s.rim_l / 2, s.rim_w / 2, s.height))
    parts = [outer, rim]
    z0, z1 = 0.03, zt
    for x in s.rib_x:                      # ribs follow the wall slope, rib_d proud of it
        for side in (-1, 1):
            pts = []
            for z in (z0, z1):
                yw = wall_half_width(s, z)
                for dx in (-s.rib_w / 2, s.rib_w / 2):
                    pts += [(x + dx, side * (yw - 0.002), z), (x + dx, side * (yw + s.rib_d), z)]
            parts.append(trimesh.convex.convex_hull(np.array(pts)))
    solid = trimesh.boolean.union(parts, engine="manifold")
    il0, iw0 = s.base_l - 2 * s.wall, s.base_w - 2 * s.wall
    il1, iw1 = s.top_l - 2 * s.wall, s.top_w - 2 * s.wall
    cav = _frustum(il0, iw0, s.floor, il1 + (il1 - il0) * 0.1, iw1 + (iw1 - iw0) * 0.1, s.height + 0.02)
    holes = [_box((sx * s.base_l / 2 - 0.03, -s.hole_w / 2, s.hole_z - s.hole_h / 2),
                  (sx * s.rim_l / 2 + 0.03, s.hole_w / 2, s.hole_z + s.hole_h / 2)) for sx in (-1, 1)]
    tote = trimesh.boolean.difference([solid, cav] + holes, engine="manifold")
    tote.metadata["name"] = "euro_tote_600x400x300_standin"
    return tote


def front_face(s: ToteSpec = ToteSpec()):
    """Centre and outward unit normal of the front (-y) outer wall, in the tote frame.

    'Theoretical front face centre' = centre of the planar wall quad between the base and the rim,
    i.e. x = 0, z = mid wall height, on the tapered wall surface (ribs ignored)."""
    zm = s.wall_top / 2
    c = np.array([0.0, -wall_half_width(s, zm), zm])
    taper = (s.top_w - s.base_w) / 2 / s.wall_top            # dy/dz of the wall
    n = np.array([0.0, -1.0, taper])                          # outward normal of the plane y = -(w0 + taper z)
    return c, n / np.linalg.norm(n)


def load_cad(path, front_center=None, front_normal=None):
    """Load a user CAD mesh (metres, tote frame as above). Returns (mesh, centre, normal)."""
    m = trimesh.load(path, force="mesh")
    if m.bounds[1].max() > 5:                                 # millimetres -> metres
        m.apply_scale(0.001)
    if front_center is None:                                  # default: centre of the -y bounding face
        lo, hi = m.bounds
        front_center = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
        front_normal = np.array([0.0, -1.0, 0.0])
    return m, np.asarray(front_center, float), np.asarray(front_normal, float)


# ---- EUR pallet (1200 x 800 x 144 mm): boards and blocks as boxes, pallet frame origin at floor centre
def make_pallet():
    L, W = 1.2, 0.8          # L along world y (faces the lidar), W along world x
    parts = []
    for y in (-0.5275, 0.0, 0.5275):                          # 3 bottom boards (along x)
        parts.append(_box((-W / 2, y - 0.0725, 0), (W / 2, y + 0.0725, 0.022)))
    for x in (-0.3275, 0.0, 0.3275):                          # 9 blocks
        for y in (-0.5275, 0.0, 0.5275):
            parts.append(_box((x - 0.0725, y - 0.0725, 0.022), (x + 0.0725, y + 0.0725, 0.100)))
    for x in (-0.3275, 0.0, 0.3275):                          # 3 stringer boards (along y)
        parts.append(_box((x - 0.0725, -L / 2, 0.100), (x + 0.0725, L / 2, 0.122)))
    for x in np.linspace(-W / 2 + 0.05, W / 2 - 0.05, 5):    # 5 top deck boards (along y)
        parts.append(_box((x - 0.05, -L / 2, 0.122), (x + 0.05, L / 2, 0.144)))
    return trimesh.util.concatenate(parts)


PALLET_TOP = 0.144
