"""
L1-regularized polynomial fit of a 2D (log-)surface at sparse points.
Python port of nadurthi/moc2 : contourbasedfiltering/get_interp_pdf_0I_2D_gobackget_L1.m

Model: f(x) ≈ exp(p(x)),  p(x) = sum_i lam_i * x^a_i  (all monomials of total degree <= Nm,
same ordering as Basis_polyND). The coefficients solve the linear program

    min   C/lamdim * ||lam||_1  +  10 * norm_largest(teq, k)
    s.t.  Aeq   lam = beq + teq      points with log f >= CUTOFF  (fit, with slack)
          Aineq lam <= CUTOFF        points with log f <  CUTOFF  (just stay low)
          Aout  lam <= log N(x;0,I)  guard points outside [-1.5,1.5]^2

norm_largest(t,k) = sum of the k largest |t_i| (as in CVX), k = floor(0.75*lamdim).
The MATLAB used CVX; here the same problem is written out as an LP and solved with
scipy's HiGHS (like the linprog branch of solveforcoeff_reg.m).

C is swept upward; the sweep stops once the peak point's relative error reaches 0.01%
and the previous solution is kept -- the MATLAB selection rule.
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye as speye

CUTOFF = -9.0
CS = [0.01, 0.1, 25, 100, 250, 300, 500, 800, 1200, 1900, 2500, 3500, 5000, 7500, 10000]


# ---------------- test surfaces (log f) on [-1,1]^2 ----------------------------------------
def logf_tbanana(X):
    """Heavy-tailed (Student-t, nu=4) banana -- the default.
    log f = -(nu+1)/2 * log(1 + q/nu),  q = (x/0.35)^2 + ((y - 1.6x^2 + 0.35)/0.22)^2.
    Its log is not a polynomial, so no basis fits it exactly."""
    x, y = X[:, 0], X[:, 1]
    q = (x / 0.35) ** 2 + ((y - 1.6 * x ** 2 + 0.35) / 0.22) ** 2
    return -2.5 * np.log1p(q / 4) + 1.2


def logf_banana(X):
    """Gaussian banana. Its log IS a degree-4 polynomial (too easy: degree>=4 can be exact)."""
    x, y = X[:, 0], X[:, 1]
    return -0.5 * ((x / 0.35) ** 2 + ((y - 1.6 * x ** 2 + 0.35) / 0.22) ** 2) + 1.2


def logf_mixture(X, smooth=1.0):
    """3-component Gaussian mixture; smooth > 1 widens every covariance."""
    comps = [(np.array([-0.30, -0.15]), np.array([[0.060, 0.035], [0.035, 0.060]]), 0.45),
             (np.array([0.25, 0.20]), np.array([[0.050, -0.020], [-0.020, 0.040]]), 0.35),
             (np.array([0.05, -0.45]), np.array([[0.030, 0.0], [0.0, 0.015]]), 0.20)]
    p = 0.0
    for mu, P, w in comps:
        P = smooth * P
        d = X - mu
        q = np.einsum("ij,jk,ik->i", d, np.linalg.inv(P), d)
        p = p + w * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(P)))
    return np.log(p + 1e-300)


SURFACES = {"tbanana": logf_tbanana, "banana": logf_banana, "mixture": logf_mixture}


# ---------------- polynomial basis -----------------------------------------------------------
def basis_exponents(dim, Nm):
    """All monomial exponents of total degree <= Nm (Basis_polyND ordering)."""
    exps = []
    for N in range(Nm + 1):
        exps += [e for e in itertools.product(range(N + 1), repeat=dim) if sum(e) == N][::-1]
    return np.array(exps)


def design(X, exps):
    return np.prod(X[:, None, :] ** exps[None, :, :], axis=2)


def guard_points():
    """Points outside the [-1.5,1.5]^2 box (stand-in for the smolyak Xout set)."""
    t = np.linspace(-2.2, 2.2, 23)
    G = np.array(list(itertools.product(t, t)))
    return G[np.any(np.abs(G) > 1.5, axis=1)]


# ---------------- the LP ---------------------------------------------------------------------
def solve_lp(Aeq, beq, Aineq, bineq, Aout, bout, C, k):
    """z = [lam (n), tl (n), teq (m), u (m), s (1)];  |lam| <= tl;
    norm_largest(teq,k) = k*s + sum(u),  u >= |teq| - s,  u >= 0."""
    m, n = Aeq.shape
    il, it, ie, iu, isv = 0, n, 2 * n, 2 * n + m, 2 * n + 2 * m
    c = np.zeros(2 * n + 2 * m + 1)
    c[it:it + n] = C / n
    c[iu:iu + m] = 10.0
    c[isv] = 10.0 * k

    I_n, I_m = speye(n), speye(m)
    Z = lambda r, cc: csr_matrix((r, cc))
    one_m = csr_matrix(np.ones((m, 1)))
    rows = [hstack([I_n, -I_n, Z(n, m), Z(n, m), Z(n, 1)]),
            hstack([-I_n, -I_n, Z(n, m), Z(n, m), Z(n, 1)]),
            hstack([Z(m, n), Z(m, n), I_m, -I_m, -one_m]),
            hstack([Z(m, n), Z(m, n), -I_m, -I_m, -one_m])]
    b_ub = [np.zeros(n), np.zeros(n), np.zeros(m), np.zeros(m)]
    for A, b in ((Aineq, bineq), (Aout, bout)):
        if A.shape[0]:
            rows.append(hstack([csr_matrix(A), Z(A.shape[0], n + 2 * m + 1)]))
            b_ub.append(b)
    A_eq = hstack([csr_matrix(Aeq), Z(m, n), -I_m, Z(m, m), Z(m, 1)])
    bounds = [(None, None)] * n + [(0, None)] * n + [(None, None)] * m + [(0, None)] * (m + 1)
    res = linprog(c, A_ub=vstack(rows).tocsr(), b_ub=np.concatenate(b_ub),
                  A_eq=A_eq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        return None, None
    return res.x[il:il + n], res.x[ie:ie + m]


def fit_l1(X, logf, Nm=6):
    """Fit exp(poly) to exp(logf) sampled at points X (N x 2). Returns a dict."""
    exps = basis_exponents(2, Nm)
    lamdim = len(exps)
    logF = logf(X)
    order = np.argsort(-logF)                    # peak first, like dsX.SortByProb('descend')
    X, logF = X[order], logF[order]
    eq = logF >= CUTOFF
    Aeq, beq = design(X[eq], exps), logF[eq]
    Aineq = design(X[~eq], exps)
    bineq = CUTOFF * np.ones((~eq).sum())
    Xo = guard_points()
    Aout = design(Xo, exps)
    bout = -0.5 * np.sum(Xo ** 2, 1) - np.log(2 * np.pi)
    k = max(1, min(int(np.floor(0.75 * lamdim)), len(beq)))

    Asan = design(np.random.default_rng(0).uniform(-1.5, 1.5, (20000, 2)), exps)
    prev, first, usedC = None, None, None
    for C in CS:
        lam, teq = solve_lp(Aeq, beq, Aineq, bineq, Aout, bout, C, k)
        with np.errstate(over="ignore"):
            if lam is None or not np.all(np.isfinite(np.exp(Asan @ lam))):
                continue
        if first is None:
            first = (lam, C)
        if 100 * abs(teq[0]) / abs(beq[0]) >= 0.01:
            break
        prev, usedC = lam, C
    if prev is None:
        prev, usedC = first
    return dict(lam=prev, exps=exps, C=usedC, X=X, eq=eq, Nm=Nm)


def evaluate(fit, G):
    """Fitted surface exp(p(x)) at points G (M x 2)."""
    return np.exp(design(G, fit["exps"]) @ fit["lam"])
