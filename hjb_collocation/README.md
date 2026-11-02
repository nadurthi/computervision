# Sparse collocation for the HJB equation — Python rerun

Python port and rerun of

> N. Adurthi, P. Singla, M. Majji, *Sparse Approximation-Based Collocation Scheme for Nonlinear
> Optimal Feedback Control Design*, J. Guidance, Control, and Dynamics 40(2):248–264, 2017.
> doi:10.2514/1.G001755

The original MATLAB code is in [nadurthi/HJB](https://github.com/nadurthi/HJB).
Write-up: perceptionstack.dev → Tutorials → Algorithms → *Sparse collocation for the HJB equation*.

## Method in one paragraph

The HJB equation for ẋ = f(x) + g(x)u, cost ∫ l(x) + uᵀRu, is solved by policy iteration.
Each step solves the *linear* generalized HJB for the current policy by collocation at
CUT points, with V(x) = cᵀφ(x) over an over-complete monomial dictionary. The coefficients come
from a weighted-ℓ1 linear program: min ‖Wc‖₁ s.t. Dc = F, Ac ≥ 0, V(0) = 0. Then
u ← −½R⁻¹gᵀ∇V, and the next GHJB is solved.

## Files

| File | What it does |
|---|---|
| `hjbcoll.py` | The method: dictionary and Jacobian, weighted-ℓ1 LP (HiGHS), Algorithm 1 (iterative ℓ1), Algorithm 2 (collocation policy iteration), least-degree baseline, closed-loop simulation, HJB residual, Pontryagin BVP for the true optimum |
| `problems.py` | Van der Pol, spin stabilization, 6-D attitude (CRP + Euler), and their collocation sets |
| `points/points.pkl` | CUT8 (2-D/3-D uniform, 2–6-D Gaussian) and Gauss–Hermite 6-D tables from nadurthi/CUTpoints |
| `interp_examples.py` | §III-C interpolation benchmark (sparse vs least degree) |
| `fig_interp_points.py`, `fig_spin.py`, `fig_vanderpol.py`, `fig_attitude.py` | the figures in `outputs/figs/` |
| `run_attitude.py` | 6-D policy iteration (`cut8` or `gh3`); checkpoints to `outputs/attitude_*.pkl` |
| `paper_figs/` | Figs. 1 and 5 cropped from the paper |
| `blog/` | the page source |

## Results summary

| Example | Outcome |
|---|---|
| Interpolation, polynomial | reproduces: sparse ℓ1 recovers all 7 terms (max error 1.8e-12); least-degree max error 106.1224 = paper |
| Interpolation, non-polynomial | ranking reproduces (sparse 2.4–3.2× better); RMSE values larger than published |
| Spin stabilization (both cases) | reproduces exactly: 3 of 84 terms, V = √(qr)·Σ Iᵢωᵢ², error 1e-14 |
| Van der Pol | does **not** reproduce: published V is 1.13× optimal (median), rerun 2.33×, u₀ 2.42×; the published V does not satisfy the described collocation equations; symmetric points leave 14 independent equations |
| 6-D attitude | see `outputs/attitude_summary.pkl` and the write-up |

The port was checked against the saved MATLAB workspace `HJBcollSol_2D_vanderpoll_c1_lowcontrol.mat`:
D and F match to 1e-12, the LP matches CVX to 3e-7, and all 40 policy iterations match to ≤ 4e-6.

## Run

```bash
pip install numpy scipy matplotlib
python fig_interp_points.py
python fig_spin.py
python fig_vanderpol.py          # ~20 min (48 Pontryagin solves + sensitivity study)
python run_attitude.py cut8      # ~5 min per policy iteration (LP with 6006 variables)
python run_attitude.py gh3
python fig_attitude.py
```
