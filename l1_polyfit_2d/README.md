# L1 polynomial surface fit at sparse points: uniform grid vs Gauss–Hermite vs CUT-8

Python port of the MATLAB routine
`nadurthi/moc2 : contourbasedfiltering/get_interp_pdf_0I_2D_gobackget_L1.m`
(no MATLAB, CVX or toolboxes needed), plus a visual comparison of three sample-point sets.

The model fits a 2D density as `f(x) ≈ exp(p(x))`, where `p` is a polynomial of total degree `Nm`
(default 6, 28 coefficients). The coefficients come from one linear program:

```
min   C/n · ‖λ‖₁ + 10 · norm_largest(teq, k)
s.t.  A_eq  λ = log f + teq       sample points with log f ≥ −9
      A_ineq λ ≤ −9                sample points with log f < −9
      A_out λ ≤ log N(x; 0, I)     guard points outside [−1.5, 1.5]²
```

`C` is swept from 0.01 to 10000 using the same stopping rule as the MATLAB code.

## Contents

| File | What it is |
|---|---|
| `l1fit.py` | The method: test surfaces, monomial basis, LP (scipy HiGHS), C sweep, evaluation |
| `points.pkl` | The point sets: `uniform_5x5`, `gh_5x5`, `cut8`. Each has `raw` nodes, `w` weights, `X` nodes scaled to max\|x\| = 0.85, and `label` |
| `make_points.py` | Regenerates `points.pkl` (CUT-8 comes from the `.mat` file in `nadurthi/CUTpoints`) |
| `compare.py` | Fits every point set and writes the comparison plots and animations to `outputs/` |
| `blog_figures.py` | Extra analysis figures (moments, null space, residuals, C path, degree sweep, …) in `outputs/blog/` |
| `requirements.txt` | numpy, scipy, matplotlib, pillow |

## Run

```bash
pip install -r requirements.txt
python compare.py                       # heavy-tailed banana, degree 6  (~2 min, mostly the animations)
python compare.py --frames 36           # quicker animations
python compare.py --surface mixture --degree 5
python blog_figures.py                  # analysis figures for the write-up
```

MP4 animations need `ffmpeg` on the PATH. Without it, only the GIFs are written.

Surfaces: `tbanana` (default, heavy-tailed banana), `banana` (Gaussian banana; its log is
exactly degree 4, so it is too easy), `mixture` (3-component Gaussian mixture; `logf_mixture(X, smooth)`
widens it).

## Outputs (`outputs/`)

- `contours_filled.png`: truth and each fit on shared color levels, error maps on a shared scale, and an error bar chart
- `contours_overlay.png`: fitted contour lines drawn over the true contour lines
- `surfaces_3d.png`, `rotating_surfaces.gif/.mp4`: truth and fits in 3D, rotating together. Red dots are the samples at their true height
- `rotating_errors.gif/.mp4`: error surfaces, all on the same scale
- `fits.pkl`: coefficients, exponents, chosen C, and error for each method

## Results (heavy-tailed banana, degree 6, error inside the sampled region)

| Point set | Points | Relative RMSE |
|---|---|---|
| Uniform 5×5 | 25 | 14.9 % |
| Gauss–Hermite 5×5 | 25 | 13.9 % |
| CUT-8 | 21 | 4.1 % |

The main reason is rank. On a 5×5 tensor grid, six degree-≤6 polynomials vanish at every node
(the node polynomial ω(x) = x(x²−a²)(x²−b²) and its multiples), so the 25 equations have rank 22.
No degree-6 polynomial can pass through all 25 points, and the L1-type residual term puts all of
the misfit on the peak point. CUT-8 is not a tensor grid: its 21×28 design matrix has full row rank,
so it interpolates every point exactly.

## Notes on the port

- CVX → explicit LP solved with `scipy.optimize.linprog(method="highs")`. `norm_largest(t, k)` is
  written as `k·s + Σu` with `u ≥ |t| − s` and `u ≥ 0`.
- The Smolyak `Xout` guard set is replaced by a ring of grid points outside [−1.5, 1.5]².
- Point sets are scaled to max|x| = 0.85 with no mean/covariance transform. The MATLAB code first
  transforms them to the density's mean and covariance (`AddMeanCov_to_OI_Trasform`).
