"""Example 3: 6-D attitude regulation (CRP + Euler), CUT8 vs Gauss–Hermite-3 collocation."""
import numpy as np, sys, time, pickle, os
from hjbcoll import Dictionary, policy_iteration
from problems import attitude, attitude_points
HERE = os.path.dirname(os.path.abspath(__file__))
kind = sys.argv[1] if len(sys.argv) > 1 else "cut8"
batches = int(sys.argv[2]) if len(sys.argv) > 2 else 8
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 20
P = attitude(); X = attitude_points(kind); dic = Dictionary(6, 8)
print(f"{kind}: N = {len(X)} collocation points, m = {dic.m} monomials, batches = {batches}", flush=True)
t = time.time()
out = os.path.join(HERE, "outputs", f"attitude_{kind}.pkl")
ck = lambda h, t: pickle.dump(dict(c=h[-1], history=h, times=t, X=X, kind=kind), open(out, "wb"))
res = policy_iteration(P, dic, X, iters=iters, batches=batches, verbose=True, lp_method="highs-ipm", tol=1e-5, checkpoint=ck)
print(f"total {time.time()-t:.0f}s", flush=True)
pickle.dump(dict(c=res.c, history=res.history, times=res.times, X=X, kind=kind, batches=batches),
            open(os.path.join(HERE, "outputs", f"attitude_{kind}.pkl"), "wb"))
