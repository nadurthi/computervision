"""Synthetic city + ray-cast 32-beam LIDAR for an end-to-end check of the localizer without real data."""
import numpy as np


def make_city(rng, n=60, extent=150):
    boxes = []
    for _ in range(n):
        c = rng.uniform(-extent, extent, 2)
        if min(abs(c[0]) % 50, abs(c[1]) % 50) < 9:       # keep a street grid every 50 m clear
            continue
        w, d, h = rng.uniform(4, 16), rng.uniform(4, 16), rng.uniform(3, 20)
        boxes.append([c[0] - w / 2, c[1] - d / 2, 0, c[0] + w / 2, c[1] + d / 2, h])
    for _ in range(80):                                     # poles / trees near the streets
        a = rng.integers(-3, 4) * 50 + rng.choice([-7, 7]); b = rng.uniform(-extent, extent)
        x, y = (a, b) if rng.random() < .5 else (b, a)
        boxes.append([x - .2, y - .2, 0, x + .2, y + .2, rng.uniform(3, 8)])
    return np.array(boxes)


def raycast(origin, R, boxes, rng, n_az=1000, elev=np.deg2rad(np.linspace(-30, 10, 32)), max_r=80):
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    A, E = np.meshgrid(az, elev)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3) @ R.T
    tmin = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = -origin[2] / d[:, 2]
        tmin = np.where((tg > 0), tg, tmin)                 # ground z = 0
        inv = 1.0 / d
        for b in boxes:
            t1 = (b[:3] - origin) * inv; t2 = (b[3:] - origin) * inv
            tn = np.nanmax(np.minimum(t1, t2), 1); tf = np.nanmin(np.maximum(t1, t2), 1)
            hit = (tf >= tn) & (tf > 0) & (tn > 0)
            tmin = np.where(hit & (tn < tmin), tn, tmin)
    ok = tmin < max_r
    P = origin + d[ok] * tmin[ok, None] + rng.normal(0, 0.02, (ok.sum(), 3))
    return (P - origin) @ R                                 # sensor frame


def drive(rng, boxes, n=150, dt=0.1, v=8.0):
    """Drive the street grid: east along y=0, then turn north at x=50."""
    poses, x, y, yaw = [], -60.0, 0.0, 0.0
    for k in range(n):
        if x >= 50 and yaw < np.pi / 2:
            yaw = min(np.pi / 2, yaw + np.deg2rad(25) * dt)
        x += v * dt * np.cos(yaw); y += v * dt * np.sin(yaw)
        H = np.eye(4); c, s = np.cos(yaw), np.sin(yaw)
        H[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; H[:3, 3] = [x, y, 1.8]
        poses.append((k * dt, H))
    return poses


def make_sequence(seed=0, n=150):
    rng = np.random.default_rng(seed)
    boxes = make_city(rng)
    frames, maps = [], []
    for t, H in drive(rng, boxes, n=n):
        X = raycast(H[:3, 3], H[:3, :3], boxes, rng)
        frames.append(dict(t=t, X=X, H=H))
        maps.append(X @ H[:3, :3].T + H[:3, 3])
    # map from a separate pass (different noise draw) so the filter does not see its own scans
    rng2 = np.random.default_rng(seed + 100)
    mp = [raycast(H[:3, 3], H[:3, :3], boxes, rng2) @ H[:3, :3].T + H[:3, 3] for _, H in drive(rng2, boxes, n=n)[::3]]
    return frames, np.vstack(mp), boxes
