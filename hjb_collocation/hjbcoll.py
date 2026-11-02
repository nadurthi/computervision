"""
Sparse-approximation collocation for the stationary Hamilton–Jacobi–Bellman equation.

Python port of
  N. Adurthi, P. Singla, M. Majji, "Sparse Approximation based Collocation Scheme for
  Nonlinear Optimal Feedback Control Design", J. Guidance, Control, and Dynamics (2017),
and of the MATLAB code in github.com/nadurthi/HJB.

Problem:   min_u  ∫ l(x) + uᵀRu dt,   ẋ = f(x) + g(x) u
HJB:       l + V_xᵀ f − ¼ V_xᵀ g R⁻¹ gᵀ V_x = 0,   V(0) = 0
Policy iteration (GHJB, linear in V):
           V_xᵀ (f + g u_k) + l + u_kᵀ R u_k = 0,     u_{k+1} = −½ R⁻¹ gᵀ V_x
Value function:  V(x) = cᵀ φ(x), φ = monomials of total degree ≤ d.
Collocation at N points gives D c = F (N equations, m ≫ N unknowns); the coefficients
come from the iterative weighted-ℓ1 problem (Algorithm 1 of the paper):

           min ‖W c‖₁   s.t.  D c = F,  A c ≥ 0,  cᵀφ(0) = 0,  c_C = 0
and the smallest surviving coefficients are moved into C until only N remain.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye as speye
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------------------------
# polynomial dictionary
# ----------------------------------------------------------------------------------------
def monomial_exponents(n: int, d: int) -> np.ndarray:
    """All exponents a ∈ ℕⁿ with |a| ≤ d, ordered by total degree (as Basis_polyND)."""
    out = []
    for k in range(d + 1):
        for e in itertools.product(range(k + 1), repeat=n):
            if sum(e) == k:
                out.append(e)
    return np.array(out[::1], dtype=int)


class Dictionary:
    def __init__(self, n: int, d: int):
        self.n, self.d = n, d
        self.E = monomial_exponents(n, d)          # m × n
        self.m = len(self.E)
        self.deg = self.E.sum(1)

    def phi(self, X: np.ndarray) -> np.ndarray:
        """N × m matrix of φ_j(x_i)."""
        X = np.atleast_2d(X)
        P = np.ones((X.shape[0], self.m))
        for k in range(self.n):
            P *= X[:, k:k + 1] ** self.E[None, :, k]
        return P

    def jac(self, X: np.ndarray) -> np.ndarray:
        """N × m × n array of ∂φ_j/∂x_k at each x_i."""
        X = np.atleast_2d(X)
        N = X.shape[0]
        J = np.empty((N, self.m, self.n))
        pw = [X[:, k:k + 1] ** np.maximum(self.E[None, :, k] - 1, 0) for k in range(self.n)]
        full = [X[:, k:k + 1] ** self.E[None, :, k] for k in range(self.n)]
        for k in range(self.n):
            t = self.E[None, :, k] * pw[k]
            for j in range(self.n):
                if j != k:
                    t = t * full[j]
            J[:, :, k] = t
        return J

    def names(self, var="x"):
        out = []
        for e in self.E:
            s = "".join(f"{var}{k+1}" + (f"^{p}" if p > 1 else "") for k, p in enumerate(e) if p)
            out.append(s or "1")
        return out


# ----------------------------------------------------------------------------------------
# control-affine problem
# ----------------------------------------------------------------------------------------
@dataclass
class Problem:
    n: int
    f: callable                 # f(X) -> N×n          (vectorised)
    g: callable                 # g(X) -> N×n×r
    l: callable                 # l(X) -> N
    R: np.ndarray               # r×r
    u0: callable                # u0(X) -> N×r         (initial admissible policy)
    name: str = ""

    @property
    def Rinv(self):
        return np.linalg.inv(self.R)


def policy_from_coeffs(prob: Problem, dic: Dictionary, c: np.ndarray):
    """u(x) = −½ R⁻¹ gᵀ J(x)ᵀ c   (vectorised)."""
    Rinv = prob.Rinv

    def u(X):
        X = np.atleast_2d(X)
        Vx = np.einsum("imk,m->ik", dic.jac(X), c)            # N×n
        return -0.5 * np.einsum("rs,iks,ik->ir", Rinv, prob.g(X), Vx)
    return u


# ----------------------------------------------------------------------------------------
# Algorithm 1: iterative weighted ℓ1
# ----------------------------------------------------------------------------------------
def _lp_weighted_l1(D, F, A, phi0, w, free, method="highs"):
    """min Σ w_j |c_j| over the free set,  D c = F,  A c ≥ 0,  φ(0)ᵀc = 0.  Returns c or None.
    Written with split variables c = p − q, p, q ≥ 0 (cost wᵀ(p+q)); at an optimum p_j q_j = 0
    whenever w_j > 0, so this is the same problem as min ‖W c‖₁ with half the rows of the
    |c| ≤ t form."""
    Df = D[:, free]
    k = len(free)
    wf = w[free].astype(float)
    cost = np.r_[wf, wf]
    A_eq = [np.c_[Df, -Df]]
    b_eq = [F]
    if phi0 is not None:                                # V(0) = 0
        p0 = phi0[free][None, :]
        A_eq.append(np.c_[p0, -p0]); b_eq.append([0.0])
    A_eq = csr_matrix(np.vstack(A_eq)); b_eq = np.concatenate(b_eq)
    if A is not None:                                   # positivity of V at the points
        Af = A[:, free]
        A_ub = csr_matrix(np.c_[-Af, Af]); b_ub = np.zeros(Af.shape[0])
    else:
        A_ub = b_ub = None
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (2 * k), method=method)
    if res.status != 0:
        return None
    c = np.zeros(D.shape[1])
    c[free] = res.x[:k] - res.x[k:]
    return c


def sparse_l1(D, F, A, phi0, weights, n_keep, batches=None, verbose=False, method="highs"):
    """Algorithm 1. Sequentially zero the smallest coefficients until n_keep remain.
    batches=None → one coefficient per LP (as in the paper); an int → that many LP solves,
    zeroing in equal chunks (as the archived MATLAB code did with Np = 5)."""
    m = D.shape[1]
    free = np.arange(m)
    c = _lp_weighted_l1(D, F, A, phi0, weights, free, method=method)
    if c is None:
        raise RuntimeError("weighted-l1 LP infeasible on the full dictionary")
    n_drop = m - n_keep
    if n_drop <= 0 or np.sum(np.abs(c) > 1e-9 * max(1.0, np.abs(c).max())) <= n_keep:
        # already at most n_keep nonzeros: forcing the remaining (zero) coefficients to zero
        # leaves the LP optimum unchanged, so Algorithm 1 would return this same c
        return c, np.where(np.abs(c) > 0)[0]
    steps = [1] * n_drop if batches is None else np.diff(np.round(np.linspace(0, n_drop, batches + 1)).astype(int))
    for s in steps:
        if s <= 0:
            continue
        order = free[np.argsort(np.abs(c[free]))]
        trial = np.setdiff1d(free, order[:s])
        c_new = _lp_weighted_l1(D, F, A, phi0, weights, trial, method=method)
        if c_new is None:                         # cannot drop further: keep last feasible
            if verbose:
                print(f"    l1: infeasible after keeping {len(free)} terms; stopping")
            break
        free, c = trial, c_new
    return c, free


# ----------------------------------------------------------------------------------------
# least-degree baseline (degree-graded pivoting on the Vandermonde matrix)
# ----------------------------------------------------------------------------------------
def least_degree_columns(A, deg, tol=1e-9):
    """Pick N columns of the Vandermonde matrix A (N×m) in order of increasing total degree,
    keeping a column only if it raises the rank (Gram–Schmidt with pivoting inside each
    degree block). This gives a minimal-degree monomial basis for the point set, in the
    spirit of de Boor & Ron's least interpolation (which the paper used as the baseline)."""
    N = A.shape[0]
    Q = np.zeros((N, 0))
    sel = []
    scale = np.linalg.norm(A, axis=0) + 1e-300
    for k in range(deg.max() + 1):
        block = list(np.where(deg == k)[0])
        while block and len(sel) < N:
            R = A[:, block] - Q @ (Q.T @ A[:, block])
            norms = np.linalg.norm(R, axis=0) / scale[block]
            j = int(np.argmax(norms))
            if norms[j] < tol:
                break
            q = R[:, j] / np.linalg.norm(R[:, j])
            Q = np.c_[Q, q]
            sel.append(block.pop(j))
        if len(sel) >= N:
            break
    return np.array(sorted(sel))


# ----------------------------------------------------------------------------------------
# Algorithm 2: collocation policy iteration
# ----------------------------------------------------------------------------------------
@dataclass
class Result:
    c: np.ndarray
    support: np.ndarray
    history: list = field(default_factory=list)       # coefficient vector per iteration
    times: list = field(default_factory=list)


def collocation_rows(prob: Problem, dic: Dictionary, X, u):
    """D (N×m) and F (N) of the GHJB collocation system for policy u."""
    U = u(X)                                                  # N×r
    xdot = prob.f(X) + np.einsum("ikr,ir->ik", prob.g(X), U)  # N×n
    D = np.einsum("imk,ik->im", dic.jac(X), xdot)
    F = -(prob.l(X) + np.einsum("ir,rs,is->i", U, prob.R, U))
    return D, F


def policy_iteration(prob: Problem, dic: Dictionary, X, iters=20, method="sparse",
                     weights=None, batches=None, n_keep=None, verbose=True, tol=1e-7, lp_method="highs",
                     checkpoint=None):
    """method: 'sparse' (Algorithm 1 inside Algorithm 2) or 'leastdeg' (fixed minimal-degree basis)."""
    A = dic.phi(X)
    phi0 = dic.phi(np.zeros((1, prob.n)))[0]
    N = X.shape[0]
    n_keep = N if n_keep is None else n_keep
    if weights is None:
        weights = dic.deg.astype(float)                       # penalty ∝ total degree (paper)
    u = prob.u0
    hist, times = [], []
    sel = least_degree_columns(A, dic.deg) if method == "leastdeg" else None
    c_prev = None
    for k in range(iters):
        t0 = time.time()
        D, F = collocation_rows(prob, dic, X, u)
        if method == "sparse":
            c, support = sparse_l1(D, F, A, phi0, weights, n_keep, batches=batches, method=lp_method)
        else:
            cs, *_ = np.linalg.lstsq(D[:, sel], F, rcond=None)
            c = np.zeros(dic.m); c[sel] = cs
            c[dic.deg == 0] = 0.0
            support = sel
        hist.append(c.copy()); times.append(time.time() - t0)
        u = policy_from_coeffs(prob, dic, c)
        dc = np.inf if c_prev is None else np.max(np.abs(c - c_prev)) / max(1e-12, np.max(np.abs(c)))
        if verbose:
            print(f"  iter {k+1:2d}: {np.sum(np.abs(c) > 1e-8):4d} nonzero terms, "
                  f"rel. change {dc:9.2e}, {times[-1]:.2f}s")
        if checkpoint is not None:
            checkpoint(hist, times)
        if dc < tol:
            break
        c_prev = c
    return Result(c=c, support=support, history=hist, times=times)


# ----------------------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------------------
def hjb_residual(prob: Problem, dic: Dictionary, c, X):
    """l + V_xᵀ f − ¼ V_xᵀ g R⁻¹ gᵀ V_x at X."""
    Vx = np.einsum("imk,m->ik", dic.jac(X), c)
    gTV = np.einsum("ikr,ik->ir", prob.g(X), Vx)
    return prob.l(X) + np.einsum("ik,ik->i", Vx, prob.f(X)) - 0.25 * np.einsum("ir,rs,is->i", gTV, prob.Rinv, gTV)


def simulate(prob: Problem, u, x0, T=30.0, n_eval=600, max_norm=1e3):
    """Closed-loop trajectory and accumulated cost ∫ l + uᵀRu."""
    n = prob.n

    def rhs(t, z):
        x = z[:n][None, :]
        uu = u(x)
        xd = prob.f(x)[0] + prob.g(x)[0] @ uu[0]
        return np.r_[xd, prob.l(x)[0] + uu[0] @ prob.R @ uu[0]]

    def blow(t, z):
        return max_norm - np.linalg.norm(z[:n])
    blow.terminal = True
    sol = solve_ivp(rhs, (0, T), np.r_[x0, 0.0], method="LSODA", rtol=1e-7, atol=1e-9,
                    t_eval=np.linspace(0, T, n_eval), events=blow)
    X = sol.y[:n].T
    U = u(X)
    diverged = (sol.status != 0) or (not np.all(np.isfinite(X))) or len(sol.t) < 2
    cost = np.inf if diverged else sol.y[n, -1]
    return dict(t=sol.t, x=X, u=U, cost=cost, diverged=diverged)


# ----------------------------------------------------------------------------------------
# independent check: open-loop optimum from Pontryagin's principle (state–costate BVP)
# ----------------------------------------------------------------------------------------
def _fx(prob, X, h=1e-6):
    """∂f/∂x by central differences, N×n×n."""
    N, n = X.shape
    J = np.empty((N, n, n))
    for k in range(n):
        e = np.zeros(n); e[k] = h
        J[:, :, k] = (prob.f(X + e) - prob.f(X - e)) / (2 * h)
    return J


def pmp_optimum(prob: Problem, Q, x0, P_term, guess=None, T=15.0, nodes=300, tol=1e-6):
    """Solve ẋ = f + g u, λ̇ = −(2Qx + f_xᵀλ), u = −½R⁻¹gᵀλ, x(0)=x0, λ(T) = 2 P_term x(T).
    (g is state-independent in all three examples.) Cost includes the terminal LQR cost
    x(T)ᵀP x(T), which approximates the infinite tail. Returns dict or None if no convergence."""
    from scipy.integrate import solve_bvp
    n = prob.n
    Rinv = prob.Rinv

    def rhs(t, Z):
        X, L = Z[:n].T, Z[n:].T
        G = prob.g(X)
        U = -0.5 * np.einsum("rs,iks,ik->ir", Rinv, G, L)
        xd = prob.f(X) + np.einsum("ikr,ir->ik", G, U)
        ld = -(2 * X @ Q + np.einsum("ijk,ij->ik", _fx(prob, X), L))
        return np.r_[xd.T, ld.T]

    def bc(Za, Zb):
        return np.r_[Za[:n] - x0, Zb[n:] - 2 * P_term @ Zb[:n]]

    t = np.linspace(0, T, nodes)
    if guess is None:
        Z0 = np.zeros((2 * n, nodes)); Z0[:n] = x0[:, None] * np.exp(-t)
    else:
        Z0 = guess(t)
    sol = solve_bvp(rhs, bc, t, Z0, tol=tol, max_nodes=200000)
    if not sol.success:
        return None
    tt = np.linspace(0, T, 4000)
    Z = sol.sol(tt)
    X, L = Z[:n].T, Z[n:].T
    U = -0.5 * np.einsum("rs,iks,ik->ir", Rinv, prob.g(X), L)
    run = prob.l(X) + np.einsum("ir,rs,is->i", U, prob.R, U)
    J = np.trapezoid(run, tt) + X[-1] @ P_term @ X[-1]
    return dict(t=tt, x=X, u=U, cost=J, lam0=L[0])
