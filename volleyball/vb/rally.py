"""Rally segmentation: IDLE -> SERVE -> IN_PLAY -> RALLY_END.

A rally starts at a touch by a player standing behind an end line (the serve) and ends when the
ball has been unseen for end_gap_s, or when a long pause separates two touches. Touches before the
first serve-like touch of a segment are warm-up / dead-ball handling and are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .court import COURT_L
from .skills import TouchIn, label_rally
from .touches import Touch


@dataclass
class Rally:
    idx: int
    touches: list[Touch] = field(default_factory=list)
    start_t: float = 0.0
    end_t: float = 0.0
    serving_side: int = -1
    winner_side: int = -1            # side that won the point, if it could be inferred

    @property
    def labels(self) -> str:
        return " ".join(f"{'NF'[t.side] if t.side in (0, 1) else '?'}{t.skill}" for t in self.touches)


def is_serve_position(t: Touch, behind_m: float) -> bool:
    if t.court_xy is None:
        return False
    y = t.court_xy[1]
    return y < -behind_m or y > COURT_L + behind_m


def ball_gaps(ball_frames: np.ndarray, fps: float, end_gap_s: float) -> list[tuple[float, float]]:
    """Time intervals where the ball is unseen for longer than end_gap_s."""
    if len(ball_frames) < 2:
        return []
    f = np.asarray(ball_frames)
    d = np.diff(f)
    idx = np.where(d / fps > end_gap_s)[0]
    return [(f[i] / fps, f[i + 1] / fps) for i in idx]


def segment_rallies(touches: list[Touch], ball_frames: np.ndarray, fps: float,
                    end_gap_s: float, behind_m: float, max_touch_gap_s: float = 4.0) -> list[Rally]:
    touches = [t for t in touches if t.side in (0, 1)]
    gaps = ball_gaps(ball_frames, fps, end_gap_s)

    def split_between(a: Touch, b: Touch) -> bool:
        if b.t - a.t > max_touch_gap_s:
            return True
        return any(a.t <= g0 and g1 <= b.t for g0, g1 in gaps)

    segments: list[list[Touch]] = []
    cur: list[Touch] = []
    for t in touches:
        if cur and split_between(cur[-1], t):
            segments.append(cur)
            cur = []
        cur.append(t)
    if cur:
        segments.append(cur)

    rallies: list[Rally] = []
    for seg in segments:
        start = next((i for i, t in enumerate(seg) if is_serve_position(t, behind_m)), None)
        if start is None:
            continue                       # no serve seen: not a rally (or the serve was missed)
        seg = seg[start:]
        r = Rally(idx=len(rallies), touches=seg, start_t=seg[0].t, end_t=seg[-1].t,
                  serving_side=seg[0].side)
        rallies.append(r)
    return rallies


def label_skills(rallies: list[Rally], block_window_s: float) -> None:
    for r in rallies:
        ins = [TouchIn(team=str(t.side), t=t.t, at_net=t.at_net) for t in r.touches]
        for t, lab in zip(r.touches, label_rally(ins, block_window_s)):
            t.skill = lab
