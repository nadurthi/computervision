"""CPU stand-in for the Isaac Sim RTX lidar: ray-cast an Ouster-OS0-like pattern against the scene meshes.

Used to develop and test the pipeline without a GPU, and for the noise / initialisation studies. The Isaac Sim
script writes the same files (points in the world frame, sensor position, ground truth), so every downstream
script runs unchanged on either source.
"""
import numpy as np
import open3d as o3d
import trimesh

OS0_128 = dict(channels=128, vfov=(-45.0, 45.0), az_res=2048, az_fov=(-60.0, 60.0), max_range=50.0)


def _t(mesh):
    m = o3d.t.geometry.TriangleMesh()
    m.vertex.positions = o3d.core.Tensor(np.asarray(mesh.vertices, np.float32))
    m.triangle.indices = o3d.core.Tensor(np.asarray(mesh.faces, np.int32))
    return m


def build_scene(tote_mesh, totes, present_ids, pallet_mesh, pallet_x=2.0):
    """Returns (RaycastingScene, geometry-id -> label). Labels: tote id >= 0, -1 pallet, -2 floor."""
    sc = o3d.t.geometry.RaycastingScene()
    labels = {}
    for t in totes:
        if t["id"] in present_ids:
            m = tote_mesh.copy(); m.apply_transform(t["T"])
            labels[sc.add_triangles(_t(m))] = t["id"]
    p = pallet_mesh.copy(); p.apply_translation((pallet_x, 0, 0))
    labels[sc.add_triangles(_t(p))] = -1
    floor = trimesh.creation.box(extents=(20, 20, 0.02)); floor.apply_translation((0, 0, -0.01))
    labels[sc.add_triangles(_t(floor))] = -2
    return sc, labels


def scan(scene, labels, sensor, pattern=OS0_128, sigma=0.005, rng=None, dropout_grazing=80.0):
    """One sweep. Returns points (N,3) world, labels (N,), ranges, incidence angle (deg)."""
    rng = rng or np.random.default_rng(0)
    el = np.radians(np.linspace(*pattern["vfov"], pattern["channels"]))
    n_az = int(pattern["az_res"] * (pattern["az_fov"][1] - pattern["az_fov"][0]) / 360.0)
    az = np.radians(np.linspace(*pattern["az_fov"], n_az))
    E, A = np.meshgrid(el, az, indexing="ij")
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    o = np.broadcast_to(np.asarray(sensor, np.float32), d.shape)
    rays = o3d.core.Tensor(np.concatenate([o, d], 1).astype(np.float32))
    ans = scene.cast_rays(rays)
    r = ans["t_hit"].numpy()
    gid = ans["geometry_ids"].numpy()
    nrm = ans["primitive_normals"].numpy()
    ok = np.isfinite(r) & (r < pattern["max_range"])
    cosi = np.abs(np.sum(nrm * d, 1))
    inc = np.degrees(np.arccos(np.clip(cosi, 0, 1)))
    ok &= inc < dropout_grazing                                  # no return at grazing incidence
    r = r + rng.normal(0, sigma, r.shape)
    pts = np.asarray(sensor) + d * r[:, None]
    lab = np.array([labels.get(int(g), -3) for g in gid])
    return pts[ok], lab[ok], r[ok], inc[ok]
