# Spin stabilization and detumbling

Code behind the perceptionstack.dev deep dive "Spin stabilization and detumbling".

| File | What it does |
|---|---|
| `spin.py` | Euler's equations + quaternion kinematics (DOP853), energy/momentum invariants, controllers (rate damping, Debs–Athans optimal, saturated, bang-bang, internal dissipation), tilt-free dipole field and normalized B-dot |
| `fig_freebody.py` | polhodes on the momentum sphere, spin about each principal axis, dissipation → major-axis spin |
| `fig_control.py` | detumble comparison, cost vs gain (closed form J(k) = (q + r k²)/(2k) ω₀ᵀIω₀), value-function level sets, B-dot on a 6U CubeSat; writes `outputs/control_stats.json` |
| `animations.py` | Dzhanibekov flip and optimal detumble, MP4 + GIF |
| `blog/` | page source (`body.html`) and builder |

```
pip install numpy scipy matplotlib   # plus ffmpeg for MP4
python fig_freebody.py && python fig_control.py && python animations.py
```

Key numbers (I = diag(14, 10, 8), ω₀ = (0.6, −0.7, 0.8) rad/s, q = 50, r = 3):
optimal cost J* = V(ω₀) = 184.45, simulated minimum 184.50 at the grid gain closest to k* = 4.08.
B-dot (I = diag(0.10, 0.08, 0.05), 0.2 A·m², 500 km SSO): 15.7 °/s → < 1 °/s in 0.57 orbit, settles near 2n = 0.127 °/s.
