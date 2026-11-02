"""Section 'Minimal degree interpolation vs sparse interpolation' of the paper (Examples 1 and 2)."""
import numpy as np, pickle, os
from hjbcoll import Dictionary, sparse_l1, least_degree_columns
HERE = os.path.dirname(os.path.abspath(__file__))

def gl_grid(q, a, b):
    x, w = np.polynomial.legendre.leggauss(q)
    x = 0.5 * (b - a) * x + 0.5 * (b + a)
    return np.array([[u, v] for u in x for v in x])

def p1(X):
    x, y = X[:, 0], X[:, 1]
    return (3*x**3 + 2*x**2*y + 2*x*y**2 + 5*y**3 + 8*x**6 + 16*x**3*y**3 + 8*y**6) / 1e3

def p2(X):
    x, y = X[:, 0], X[:, 1]
    return (x**4*np.sin(x) + 10*y**4*np.cos(y)**2 + x**8*y + 100*y**6) / 100

def fit(X, V, dic, how, weights=None):
    A = dic.phi(X)
    if how == "sparse":
        w = np.ones(dic.m) if weights is None else weights
        c, sup = sparse_l1(A, V, None, None, w, n_keep=len(X))
    else:
        sel = least_degree_columns(A, dic.deg)
        c = np.zeros(dic.m); c[sel] = np.linalg.lstsq(A[:, sel], V, rcond=None)[0]
    return c

def run(verbose=True):
    dic = Dictionary(2, 6)
    out = {}
    # Example 1: 4x4 Gauss-Legendre on [-5,5]^2, degree-6 polynomial
    X = gl_grid(4, -5, 5); V = p1(X)
    g = np.linspace(-5, 5, 101); GX, GY = np.meshgrid(g, g); G = np.c_[GX.ravel(), GY.ravel()]
    T = p1(G)
    for how in ["sparse", "leastdeg"]:
        c = fit(X, V, dic, how)
        E = np.abs(dic.phi(G) @ c - T)
        out[("ex1", how)] = dict(c=c, maxerr=E.max(), X=X, grid=(GX, GY), true=T.reshape(GX.shape), fit=(dic.phi(G) @ c).reshape(GX.shape))
        if verbose:
            print(f"Ex1 {how:9s}: max |error| = {E.max():.3e}, nonzero = {np.sum(np.abs(c)>1e-8)}")
    if verbose:
        c = out[("ex1","sparse")]["c"]; nm = dic.names()
        print("   recovered:", {nm[i]: round(c[i]*1e3, 6) for i in np.where(np.abs(c) > 1e-9)[0]}, "(x1e-3)")
    # Example 2: non-polynomial on [-4,4]^2, GL 3x3 and 4x4
    g = np.arange(-4, 4.0001, 0.25); GX, GY = np.meshgrid(g, g); G = np.c_[GX.ravel(), GY.ravel()]
    T = p2(G)
    for q in [3, 4]:
        X = gl_grid(q, -4, 4); V = p2(X)
        for how in ["sparse", "leastdeg"]:
            c = fit(X, V, dic, how)
            F = dic.phi(G) @ c
            rmse = np.sqrt(np.mean((F - T) ** 2))
            out[("ex2", q, how)] = dict(c=c, rmse=rmse, X=X, grid=(GX, GY), true=T.reshape(GX.shape), fit=F.reshape(GX.shape))
            if verbose:
                print(f"Ex2 GL{q}x{q} {how:9s}: RMSE = {rmse:8.2f}")
    return out

if __name__ == "__main__":
    out = run()
    pickle.dump(out, open(os.path.join(HERE, "outputs", "interp_examples.pkl"), "wb"))
