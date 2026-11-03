import ast, sys, time, heapq, numpy as np
sys.path[:0] = [sys.argv[1], "../../python", "../",
                "."]
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from numba.typed import Dict
from numba.core import types
import lidarloc
import lidarprocessing.numba_codes.binmatchers as bm
src = open("../drivers/main_kitti_filter_localize.py").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "binMatcherAdaptive3")
ns = dict(np=np, nplinalg=np.linalg, heapq=heapq, nbpt2Dproc=nbpt2Dproc, Dict=Dict, types=types, float_2Darray=None)
exec(compile(ast.Module([fn], []), "driver", "exec"), ns)
orig = ns["binMatcherAdaptive3"]

rng = np.random.default_rng(1)
# synthetic city block: building walls along two streets + poles
walls = []
for _ in range(45):
    c = rng.uniform(-60, 60, 2); w, h = rng.uniform(3, 14, 2)
    per = np.r_[np.c_[np.linspace(0, w, int(w*8)), np.zeros(int(w*8))], np.c_[np.zeros(int(h*8)), np.linspace(0, h, int(h*8))],
                np.c_[np.linspace(0, w, int(w*8)), np.full(int(w*8), h)], np.c_[np.full(int(h*8), w), np.linspace(0, h, int(h*8))]]
    if np.linalg.norm(c) > 12: walls.append(per + c)
for x0 in []:
    L = rng.uniform(6, 12)
    for side in (-8, 8):
        walls.append(np.c_[np.linspace(x0, x0 + L, int(L * 10)), np.full(int(L * 10), side + rng.normal(0, .05, int(L * 10)))])
for y0 in range(-60, 60, 18):
    walls.append(np.c_[np.full(80, 30 + rng.normal(0, .05, 80)), np.linspace(y0, y0 + 8, 80)])
walls.append(rng.uniform(-60, 60, (60, 2)))
Xmap = np.vstack(walls)
th_true, t_true = np.deg2rad(20), np.array([5.0, 1.0])
Rt = np.array([[np.cos(th_true), -np.sin(th_true)], [np.sin(th_true), np.cos(th_true)]])
near = Xmap[np.linalg.norm(Xmap - t_true, axis=1) < 40]
Xs = (near - t_true) @ Rt + rng.normal(0, 0.05, near.shape)         # sensor-frame scan
H0 = np.eye(3); th0 = th_true + np.deg2rad(6); H0[:2, :2] = [[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]]
H0[:2, 2] = t_true + [2.5, -1.8]
args = (Xmap, Xs, H0, np.array([8.0, 8.0]), np.deg2rad(15), np.deg2rad(1.0), np.array([0.5, 0.5]))
for name, f in [("original _super (no JIT)", lambda *a: bm.binMatcherAdaptive_super(*a, np.array([30.0,30.0]))), ("numpy port", lidarloc.bin_matcher)]:
    st = time.time(); out = f(*args); et = time.time() - st
    H = np.linalg.inv(out[0]) if name.startswith("orig") else out[0]      # original returns H21
    th = np.arctan2(H[1, 0], H[0, 0])
    print(f"{name:32s} t=({H[0,2]:.2f},{H[1,2]:.2f}) err={np.linalg.norm(H[:2,2]-t_true):.2f} m  "
          f"yaw err={np.rad2deg(th-th_true):+.2f} deg  costs={tuple(float(v) for v in out[1:3])}  {et:.2f}s")
print(f"initial guess error: {np.linalg.norm(H0[:2,2]-t_true):.2f} m, 6.00 deg")
