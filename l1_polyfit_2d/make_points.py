"""
Build points.pkl: the 2D sample point sets used by the comparison.

    uniform_5x5   : 5x5 tensor grid
    gh_5x5        : 5x5 tensor Gauss-Hermite (probabilists') nodes
    cut8          : CUT-8 points for a 2D standard Gaussian (21 pts), from nadurthi/CUTpoints

Each entry stores the raw nodes/weights and the nodes scaled to max|x| = 0.85
(the scaling used in get_interp_pdf_0I_2D_gobackget_L1.m).

points.pkl already ships with this folder, so you only need this script to regenerate it.
CUT-8 can only be regenerated from the .mat file in the CUTpoints repo:
    python make_points.py --cut8-mat "path/to/CUTpoints/CUT points- mat files/gaussian_mat/CUT/cut8_2D_gaussian.mat"
Without --cut8-mat, the CUT-8 entry is copied over from the existing points.pkl.
"""
import argparse
import itertools
import os
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "points.pkl")


def scale(X):
    return 0.85 * X / np.max(np.abs(X))


def tensor(x, w):
    X = np.array(list(itertools.product(x, x)))
    W = np.array([a * b for a, b in itertools.product(w, w)])
    return X, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut8-mat", default=None, help="path to cut8_2D_gaussian.mat")
    args = ap.parse_args()

    sets = {}
    g = np.linspace(-1, 1, 5)
    X, W = tensor(g, np.full(5, 1 / 5))
    sets["uniform_5x5"] = dict(label="Uniform grid 5×5", raw=X, w=W, X=scale(X))

    x, w = np.polynomial.hermite_e.hermegauss(5)
    X, W = tensor(x, w / w.sum())
    sets["gh_5x5"] = dict(label="Gauss–Hermite 5×5", raw=X, w=W, X=scale(X))

    if args.cut8_mat:
        import scipy.io as sio
        d = sio.loadmat(args.cut8_mat)
        X, W = d["X"], d["w"].ravel()
        sets["cut8"] = dict(label="CUT-8", raw=X, w=W, X=scale(X))
    else:
        with open(PKL, "rb") as f:
            sets["cut8"] = pickle.load(f)["cut8"]

    with open(PKL, "wb") as f:
        pickle.dump(sets, f)
    for k, v in sets.items():
        print(f"{k:12s} {len(v['X']):3d} points   sum(w) = {v['w'].sum():.6f}")


if __name__ == "__main__":
    main()
