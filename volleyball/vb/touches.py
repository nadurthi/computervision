"""Touch detection from the ball track, and attribution to a player from pose keypoints.

A contact shows up as a kink in the image track: cos(theta_t) = v_t . v_{t+k} / (|v_t| |v_{t+k}|).
The touch goes to the player whose wrist or elbow is nearest the ball at the kink, normalised by
that player's box height so near and far players are comparable. In 2D a player several metres
behind the ball can have the nearest elbow, so an optional "raised hand" preference is applied
(wrist above shoulder at contact), which the blog measured as fixing 8 cases and breaking 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .court import NET_Y

# COCO-17 indices
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16
HAND_POINTS = (L_WR, R_WR, L_EL, R_EL)


@dataclass
class PlayerObs:
    tid: int
    box: np.ndarray                  # xyxy pixels
    kp: np.ndarray | None = None     # (17,2) pixels
    kpc: np.ndarray | None = None    # (17,) scores
    court_xy: np.ndarray | None = None
    side: int = -1                   # 0 near, 1 far


@dataclass
class Touch:
    frame: int
    t: float
    ball_xy: tuple[float, float]
    angle_deg: float
    tid: int | None = None
    side: int = -1
    court_xy: tuple[float, float] | None = None
    at_net: bool = False
    raised: bool = False
    dist: float = float("inf")
    skill: str = ""
    team: str = ""                   # "*" home / "a" away, filled by the pipeline
    number: int | None = None
    extra: dict = field(default_factory=dict)


def kink_candidates(frames: np.ndarray, xy: np.ndarray, lag: int, angle_deg: float,
                    min_speed: float, refractory: int) -> list[tuple[int, float]]:
    """Indices into the ball track where the direction changes by more than angle_deg.

    Gravity also turns the ball (most visibly at the top of a lob), but smoothly: the image
    acceleration of a ball in flight is roughly constant over a few frames, while a contact is an
    impulse. So the local acceleration is estimated as the median second difference around the
    candidate (the median ignores the impulse itself), and both velocities are extrapolated to the
    candidate frame with it before the angle is measured. A pure parabola then has no kink at all.
    """
    if len(xy) < 2 * lag + 2:
        return []
    v = np.diff(xy, axis=0)                    # v[i] = xy[i+1] - xy[i]
    gaps = np.diff(frames)
    v = v / np.maximum(gaps, 1)[:, None]       # per-frame velocity across bridged gaps
    acc = np.diff(v, axis=0)                   # acc[i] = v[i+1] - v[i]
    out: list[tuple[int, float]] = []
    for i in range(lag, len(v) - lag):
        win = acc[max(0, i - 3 * lag): min(len(acc), i + 3 * lag)]
        a_hat = np.median(win, axis=0) if len(win) else np.zeros(2)
        jb = np.arange(i - lag, i)
        ja = np.arange(i, i + lag)
        a = (v[jb] + a_hat * (i - jb - 0.5)[:, None]).mean(0)     # velocity just before i
        b = (v[ja] - a_hat * (ja - i + 0.5)[:, None]).mean(0)     # velocity just after i
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < min_speed and nb < min_speed:
            continue
        cos = float(np.dot(a, b) / (na * nb + 1e-9))
        ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        if ang >= angle_deg:
            out.append((i, ang))
    # non-maximum suppression in time
    out.sort(key=lambda p: -p[1])
    kept: list[tuple[int, float]] = []
    for i, ang in out:
        if all(abs(frames[i] - frames[j]) >= refractory for j, _ in kept):
            kept.append((i, ang))
    kept.sort()
    return kept


def hand_distance(ball: np.ndarray, p: PlayerObs, min_conf: float = 0.3) -> tuple[float, bool]:
    """Normalised distance from ball to the nearest wrist/elbow, and whether a hand is raised."""
    h = max(float(p.box[3] - p.box[1]), 1.0)
    if p.kp is None:
        cx = 0.5 * (p.box[0] + p.box[2])
        return float(np.hypot(ball[0] - cx, ball[1] - p.box[1])) / h + 0.5, False   # weak fallback
    conf = p.kpc if p.kpc is not None else np.ones(17)
    ds = [np.linalg.norm(ball - p.kp[k]) for k in HAND_POINTS if conf[k] >= min_conf]
    if not ds:
        return float("inf"), False
    sh_y = min(p.kp[L_SH][1], p.kp[R_SH][1]) if min(conf[L_SH], conf[R_SH]) >= min_conf else p.box[1] + 0.25 * h
    raised = any(conf[k] >= min_conf and p.kp[k][1] < sh_y for k in (L_WR, R_WR))
    return float(min(ds)) / h, bool(raised)


def attribute(ball: np.ndarray, players: list[PlayerObs], max_dist: float,
              raised_rule: bool) -> tuple[PlayerObs | None, float, bool]:
    best, best_score, best_d, best_r = None, float("inf"), float("inf"), False
    for p in players:
        d, raised = hand_distance(ball, p)
        if d > max_dist:
            continue
        score = d * (0.6 if (raised_rule and raised) else 1.0)
        if score < best_score:
            best, best_score, best_d, best_r = p, score, d, raised
    return best, best_d, best_r


def detect_touches(ball_frames: np.ndarray, ball_xy: np.ndarray, players_by_frame: dict[int, list[PlayerObs]],
                   fps: float, cfg) -> list[Touch]:
    """cfg: TouchConfig.

    Candidates are the kinks plus the first point of every ball-track segment: a serve often has
    no kink in view (the toss is out of frame or the ball is first detected at contact), so the
    start of a segment is a touch whenever a hand is close enough to it.
    """
    touches: list[Touch] = []
    cands = kink_candidates(ball_frames, ball_xy, cfg.lag, cfg.angle_deg, cfg.min_speed_px, cfg.refractory)
    if len(ball_frames):
        starts = [0] + [i + 1 for i in np.where(np.diff(ball_frames) > cfg.refractory)[0]]
        for s in starts:
            if all(abs(int(ball_frames[s]) - int(ball_frames[i])) >= cfg.refractory for i, _ in cands):
                cands.append((s, 180.0))
        cands.sort()
    for i, ang in cands:
        f = int(ball_frames[i])
        ball = ball_xy[i]
        # search the contact frame +-1 for the best hand, since the kink is quantised
        best = (None, float("inf"), False, f)
        for df in (0, -1, 1):
            ps = players_by_frame.get(f + df, [])
            p, d, r = attribute(ball, ps, cfg.hand_dist, cfg.raised_hand_rule)
            if p is not None and d < best[1]:
                best = (p, d, r, f + df)
        p, d, r, _ = best
        t = Touch(frame=f, t=f / fps, ball_xy=(float(ball[0]), float(ball[1])), angle_deg=ang)
        if p is not None:
            t.tid, t.dist, t.raised, t.side = p.tid, d, r, int(p.side)
            if p.court_xy is not None:
                t.court_xy = (float(p.court_xy[0]), float(p.court_xy[1]))
                t.at_net = abs(t.court_xy[1] - NET_Y) <= cfg.net_band_m
        touches.append(t)
    return touches
