"""
Rigid-body spin dynamics and detumbling controllers.

Body frame = principal axes, I = diag(I1, I2, I3).
Euler:        I ω̇ = (I ω) × ω + τ
Attitude:     q̇ = ½ Ω(ω) q          (scalar-last quaternion, body-to-inertial)
"""
import numpy as np
from scipy.integrate import solve_ivp

I_DEFAULT = np.array([14.0, 10.0, 8.0])


# ------------------------------------------------------------------ dynamics
def quat_mult(p, q):
    """Hamilton product, scalar-last [x, y, z, w]."""
    px, py, pz, pw = p; qx, qy, qz, qw = q
    return np.array([pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw,
                     pw * qw - px * qx - py * qy - pz * qz])


def quat_to_dcm(q):
    """Body-to-inertial rotation matrix from scalar-last quaternion."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def simulate(I, w0, torque=None, T=60.0, n=3000, q0=(0, 0, 0, 1.0), rtol=1e-10, atol=1e-12, extra_state=0, extra_rhs=None):
    """Integrate Euler + quaternion kinematics. torque(t, w, q) -> τ in body axes."""
    I = np.asarray(I, float)
    torque = torque or (lambda t, w, q: np.zeros(3))

    def rhs(t, z):
        w, q = z[:3], z[3:7]
        tau = torque(t, w, q)
        wd = (np.cross(I * w, w) + tau) / I
        qd = 0.5 * quat_mult(q, np.r_[w, 0.0])           # q̇ = ½ q ⊗ [ω; 0]
        out = np.r_[wd, qd]
        if extra_rhs is not None:
            out = np.r_[out, extra_rhs(t, w, q, tau)]
        return out

    z0 = np.r_[w0, q0, np.zeros(extra_state)]
    sol = solve_ivp(rhs, (0, T), z0, method="DOP853", rtol=rtol, atol=atol, t_eval=np.linspace(0, T, n))
    w, q = sol.y[:3].T, sol.y[3:7].T
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tau = np.array([torque(t, wi, qi) for t, wi, qi in zip(sol.t, w, q)])
    out = dict(t=sol.t, w=w, q=q, tau=tau)
    if extra_state:
        out["extra"] = sol.y[7:].T
    return out


def energy(I, w):
    return 0.5 * np.sum(I * w * w, axis=-1)


def hmag(I, w):
    return np.linalg.norm(I * w, axis=-1)


# ------------------------------------------------------------------ controllers
def rate_damping(k):
    return lambda t, w, q: -k * w


def debs_athans(q, r):
    """Optimal for J = ∫ q‖ω‖² + r‖u‖² dt:  V = √(qr) ωᵀIω,  u* = −√(q/r) ω."""
    k = np.sqrt(q / r)
    return lambda t, w, qq: -k * w


def saturated(k, umax):
    return lambda t, w, q: -umax * np.tanh(k * w / umax)


def bang_bang(umax, eps=1e-3):
    return lambda t, w, q: -umax * np.tanh(w / eps)


def internal_damping(I, k):
    """Energy sink that conserves |H| (e.g. sloshing fuel, flexible antennas):
       τ = k H × (H × ω)  ⇒  d|H|/dt = 0,  dT/dt = −k(|H|²|ω|² − (H·ω)²) ≤ 0."""
    I = np.asarray(I, float)
    return lambda t, w, q: k * np.cross(I * w, np.cross(I * w, w))


# ------------------------------------------------------------------ magnetic field and B-dot
MU_E = 3.986004418e14            # m³/s²
B0 = 3.12e-5                     # T, equatorial surface dipole field
RE = 6.371e6


def orbit_field_inertial(t, alt=500e3, inc=np.deg2rad(97.4)):
    """Tilt-free dipole field (inertial frame, m̂ = −z) along a circular orbit."""
    r = RE + alt
    n = np.sqrt(MU_E / r ** 3)
    u = n * t
    rhat = np.array([np.cos(u), np.sin(u) * np.cos(inc), np.sin(u) * np.sin(inc)])
    m = np.array([0, 0, -1.0])
    return B0 * (RE / r) ** 3 * (3 * np.dot(m, rhat) * rhat - m), n


def bdot_controller(I, mmax, alt=500e3, inc=np.deg2rad(97.4), k=None, log=None):
    """Normalized B-dot (Avanzini & Giulietti 2012):  m = −(k/‖B‖) · d(B/‖B‖)/dt,  saturated per axis,
    τ = m × B.  Gain k = 2n(1 + sin ξ) J_min, with ξ ≈ inclination to the geomagnetic equator."""
    I = np.asarray(I, float)
    r = RE + alt; n = np.sqrt(MU_E / r ** 3)
    if k is None:
        k = 2 * n * (1 + np.sin(inc)) * I.min()
    def bhat_body(t, q):
        BI, _ = orbit_field_inertial(t, alt, inc)
        return quat_to_dcm(q).T @ BI
    def dipole(t, w, q):
        Bb = bhat_body(t, q)
        Bn = np.linalg.norm(Bb); bhat = Bb / Bn
        # d(b̂)/dt in body axes = −ω × b̂ + (rotation of the field direction itself, slow)
        BI0, _ = orbit_field_inertial(t, alt, inc); BI1, _ = orbit_field_inertial(t + 1.0, alt, inc)
        C = quat_to_dcm(q)
        dbI = BI1 / np.linalg.norm(BI1) - BI0 / np.linalg.norm(BI0)
        bdot = -np.cross(w, bhat) + C.T @ dbI
        return np.clip(-(k / Bn) * bdot, -mmax, mmax), Bb

    def tau(t, w, q):
        m, Bb = dipole(t, w, q)
        return np.cross(m, Bb)
    tau.k = k; tau.n = n; tau.dipole = dipole
    return tau


I_CUBESAT = np.array([0.10, 0.08, 0.05])   # 6U-class, kg m^2
