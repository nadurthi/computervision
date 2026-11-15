"""OctoMap construction and FCL collision checking of the sphere-model arm."""
import time
import numpy as np
import fcl
import octomap

from robot import spheres, self_collision, N_BASE_SPHERES, LOWER, UPPER

BT_HEADER = b"data\n"


def build_octomap(points, origin, res, max_range=-1.0):
    """Insert one lidar scan (ray casting: free along rays, occupied at hits)."""
    t0 = time.perf_counter()
    tree = octomap.OcTree(res)
    tree.insertPointCloud(np.ascontiguousarray(points, dtype=np.float64),
                          np.asarray(origin, dtype=np.float64), maxrange=max_range)
    tree.updateInnerOccupancy()
    dt = time.perf_counter() - t0
    occ, free = tree.extractPointCloud()
    return tree, np.asarray(occ), np.asarray(free), dt


def octomap_bytes(tree, path):
    """Write a .bt file and return the raw binary payload (the octomap_msgs format)."""
    tree.writeBinary(str(path).encode())
    data = open(path, "rb").read()
    return data[data.index(BT_HEADER) + len(BT_HEADER):]


class ArmChecker:
    """State validity for the Panda: joint limits, floor, self-collision (sphere pairs), and the
    arm spheres against the OctoMap with FCL."""

    def __init__(self, raw_octomap, res, floor_z=0.0):
        self.tree = fcl.OcTree(res, data=np.frombuffer(raw_octomap, np.int8).tolist())
        self.obj = fcl.CollisionObject(self.tree, fcl.Transform())
        c, r = spheres(np.zeros(7))
        self.radii = r
        self.objs = [fcl.CollisionObject(fcl.Sphere(float(ri)), fcl.Transform()) for ri in r]
        self.req = fcl.CollisionRequest()
        self.floor_z = floor_z
        self.n_checks = 0
        self.t_checks = 0.0

    def valid(self, q):
        t0 = time.perf_counter()
        self.n_checks += 1
        ok = self._valid(q)
        self.t_checks += time.perf_counter() - t0
        return ok

    def _valid(self, q):
        q = np.asarray(q, float)
        if np.any(q < LOWER) or np.any(q > UPPER):
            return False
        c, r = spheres(q)
        if np.any(c[N_BASE_SPHERES:, 2] - r[N_BASE_SPHERES:] < self.floor_z):
            return False
        if self_collision(c):
            return False
        for o, p in zip(self.objs[::-1], c[::-1]):  # hand first: it hits most often, exit early
            o.setTranslation(p)
            res = fcl.CollisionResult()
            if fcl.collide(o, self.obj, self.req, res):
                return False
        return True

    def segment_valid(self, qa, qb, step=0.01):
        n = int(np.ceil(np.max(np.abs(qb - qa)) / step)) + 1
        return all(self.valid(qa + t * (qb - qa)) for t in np.linspace(0, 1, n))
