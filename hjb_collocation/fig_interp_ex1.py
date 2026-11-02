"""Rerun figure for interpolation Example 1 (the polynomial case, which reproduces the paper)."""
import numpy as np
from figs_common import plt, save
from interp_examples import run as run_interp
out = run_interp(verbose=False)
s, l = out[("ex1", "sparse")], out[("ex1", "leastdeg")]
GX, GY = s["grid"]; T = s["true"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.9), constrained_layout=True)
ax[0].contourf(GX, GY, T, np.linspace(T.min(), T.max(), 16), cmap="viridis"); ax[0].plot(*s["X"].T, "wo", mec="k")
ax[0].set_title("true p(x): degree-6 polynomial, 16 Gauss–Legendre points")
E1 = np.abs(s["fit"] - T); E2 = np.abs(l["fit"] - T); vmax = E2.max()
for a, E, lab in [(ax[1], E1, "sparse ℓ1"), (ax[2], E2, "least degree")]:
    im = a.contourf(GX, GY, E, np.linspace(0, vmax, 16), cmap="magma"); a.plot(*s["X"].T, "wo", mec="k", ms=4)
    a.set_title(f"|error|, {lab}: max {E.max():.2e}")
fig.colorbar(im, ax=ax[1:], shrink=.85, label="|fit − true| (shared scale)")
for a in ax: a.set_aspect("equal"); a.set_xlabel("x₁"); a.set_ylabel("x₂")
save(fig, "fig_interp_ex1.png")
