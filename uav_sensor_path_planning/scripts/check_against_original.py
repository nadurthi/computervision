"""Unit-level checks of the port against the original ResearchCodes functions.

    python scripts/check_against_original.py /path/to/ResearchCodes

Needs the original repo (with turtlebot/uq copied to uq/) on disk, plus
shapely. Compares, on random inputs:
  * Gaussian mutual information      (uq/information/distance.mutualInformation_covs)
  * trajectory enumeration + filter  (utils/math/gen_seq + traj_filter)
  * FOV / 1-sigma ellipse overlap    (PlanarSensorModel.intersectCovWithFOV, shapely)
  * Markov coverage update           (uq/filters/markovfilter.TargetMarkovF)
"""
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HERE)
from sensorplan.grid import NodeGrid, enumerate_trajectories  # noqa: E402
from sensorplan.models import CircularFOVSensor, CoverageMap, cov_fov_overlap  # noqa: E402
from sensorplan.planners import gaussian_mi  # noqa: E402


def main(orig):
    sys.path.insert(0, orig)
    import numpy
    if not hasattr(numpy, "int"):
        numpy.int = int  # 2021 code uses np.int
    from uq.information.distance import mutualInformation_covs
    from utils.math.gen_seq import dependent_seq_generator
    from sensortasking.exhaustive_tasking_seq_robot import traj_filter
    from physmodels.sensormodels_fov import XYcircularFOVsensor
    from maps import gridmap

    rng = np.random.default_rng(1)
    ok = True

    # 1. mutual information
    err = 0
    for _ in range(200):
        A = rng.standard_normal((4, 4)); P1 = A @ A.T + np.eye(4)
        
        H = np.hstack([np.eye(2), np.zeros((2, 2))])
        S = H @ P1 @ H.T + 0.5 * np.eye(2); K = P1 @ H.T @ np.linalg.inv(S); P2 = P1 - K @ S @ K.T
        err = max(err, abs(gaussian_mi(P1, P2) - mutualInformation_covs(P1, P2)))
    print(f"mutual information      max |diff| = {err:.2e}")
    ok &= err < 1e-9

    # 2. enumeration
    g = NodeGrid()
    rg = gridmap.Regular2DNodeGrid(xy0=(0, 0), xyf=(101, 101), d=(10, 10))
    same = True
    for s in [0, 5, 27, 60, 98, 120]:
        x0 = rg.XYgvec[s]
        Xset = rg.nodesInRadius(x0, 5 * (8 + 0.2), returnNodeIds=True)
        Xconst = {}
        for idx in Xset:
            Xconst[idx] = rg.nodesInRadius(rg.XYgvec[idx], 5 * 1.1 * 2, returnNodeIds=True)
            Xconst[idx].remove(idx)
        o = traj_filter(list(dependent_seq_generator(5, s, Xset, Xconst)))
        p = enumerate_trajectories(g, s, 5, 5, 2.0)
        same &= [list(map(int, a)) for a in o] == [list(map(int, b)) for b in p]
    print(f"trajectory enumeration  identical sequences and order: {same}")
    ok &= same

    # 3. overlap
    sen = XYcircularFOVsensor(R=np.eye(2), FOVradius=20, posstates=[0, 1], enforceConstraint=True,
                              TP=0.9, TN=0.9, FP=0.1, FN=0.1)  # values used in the paper sim
    errs = []
    for _ in range(300):
        A = rng.standard_normal((2, 2)) * rng.uniform(1, 12); P = A @ A.T + np.eye(2)
        m = rng.uniform(0, 100, 2); s = m + rng.uniform(-30, 30, 2)
        sen.updateparam(0, xc=s, dirn=0)
        errs.append(abs(sen.intersectCovWithFOV(m, P) - cov_fov_overlap(m, P, s, 20)))
    errs = np.array(errs)
    print(f"FOV/ellipse overlap     max |diff| = {errs.max():.3f}  (mean {errs.mean():.4f})")
    ok &= errs.max() < 0.03

    # 4. Markov coverage update (planning form, z = predicted detection)
    from uq.filters.markovfilter import TargetMarkovF
    from physmodels import transitionmodels as phytm
    from physmodels import targets as phytarg

    class T:  # minimal stand-in for phytarg.Target
        pass
    tm = phytm.TransitionModel(np.array([[0.99, 0.01], [0.01, 0.99]]), g.nodes.astype(float))
    t = T(); t.dynModel = tm; p = rng.uniform(0.05, 0.95, g.N); t.xfk = p
    sen.updateparam(0, xc=np.array([40.0, 60.0]), dirn=0)
    xo, _ = TargetMarkovF().measUpdate(0, 2, t, sen, None, updttarget=False)
    cm = CoverageMap(cells=g.nodes.astype(float))
    inside = np.linalg.norm(g.nodes - [40, 60], axis=1) <= 20
    xp = cm.update(p, inside, CircularFOVSensor(fov_radius=20, sigma=1))
    e4 = np.abs(xo - xp).max()
    xo_pred, _ = TargetMarkovF().propagate(0, 2, t, updttarget=False)
    e5 = np.abs(xo_pred - cm.predict(p)).max()
    print(f"coverage update/predict max |diff| = {e4:.2e} / {e5:.2e}")
    ok &= e4 < 1e-12 and e5 < 1e-12
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main(sys.argv[1])
