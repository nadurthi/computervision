"""Player and ball tracking.

Players: IoU + Hungarian matching with a constant-velocity box predictor (SORT-style, no Kalman
covariance; at 25-50 fps on a broadcast camera that is enough, and it is deterministic).

Ball: one track at a time. Each frame, take the ball detection closest to the motion-model
prediction inside a gate; bridge short gaps; end the track when the ball is gone too long; then
recover onto the next confident detection. After a recovery, if the next few steps stay within one
ball diameter of the recovery point, the recovery was onto a static ball (scorer's table, ball
logo on an ad board) and those steps are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class _PTrack:
    tid: int
    box: np.ndarray
    vel: np.ndarray
    last: int
    hits: int = 1


class PlayerTracker:
    def __init__(self, iou_thresh: float = 0.25, max_age: int = 15, min_hits: int = 3):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[_PTrack] = []
        self.next_id = 1

    def update(self, frame_idx: int, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """boxes (N,4) xyxy -> track ids (N,) ; -1 for detections not yet confirmed."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        pred = np.array([t.box + t.vel * (frame_idx - t.last) for t in self.tracks], np.float32).reshape(-1, 4)
        ious = iou_matrix(pred, boxes)
        ids = np.full(len(boxes), -1, int)
        matched_t, matched_d = set(), set()
        if ious.size:
            r, c = linear_sum_assignment(-ious)
            for ti, di in zip(r, c):
                if ious[ti, di] >= self.iou_thresh:
                    t = self.tracks[ti]
                    dt = max(frame_idx - t.last, 1)
                    t.vel = 0.7 * t.vel + 0.3 * (boxes[di] - t.box) / dt
                    t.box, t.last, t.hits = boxes[di], frame_idx, t.hits + 1
                    matched_t.add(ti)
                    matched_d.add(di)
                    ids[di] = t.tid if t.hits >= self.min_hits else -1
        # new tracks, in a deterministic order (left to right)
        for di in sorted(set(range(len(boxes))) - matched_d, key=lambda i: (boxes[i, 0], boxes[i, 1])):
            self.tracks.append(_PTrack(self.next_id, boxes[di], np.zeros(4, np.float32), frame_idx))
            if self.min_hits <= 1:
                ids[di] = self.next_id
            self.next_id += 1
        self.tracks = [t for t in self.tracks if frame_idx - t.last <= self.max_age]
        return ids


@dataclass
class BallTrack:
    frames: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    observed: list[bool] = field(default_factory=list)  # False = bridged by the motion model
    recoveries: list[int] = field(default_factory=list)  # indices into frames where a recovery happened

    def as_arrays(self):
        return np.asarray(self.frames, int), np.asarray(self.xy, np.float32).reshape(-1, 2), np.asarray(self.observed, bool)


def drop_static_recovery(track: list[tuple[int, float, float]], recovered_at: int,
                         ball_diam_px: float, n_check: int = 4) -> list[tuple[int, float, float]]:
    """The blog's fix: a recovered ball that does not move for n_check steps is not the match ball."""
    x0, y0 = track[recovered_at][1:]
    nxt = track[recovered_at + 1: recovered_at + 1 + n_check]
    if len(nxt) == n_check and all((x - x0) ** 2 + (y - y0) ** 2 <= ball_diam_px ** 2 for _, x, y in nxt):
        return track[:recovered_at] + track[recovered_at + 1 + n_check:]
    return track


class BallTracker:
    def __init__(self, gate_px: float, max_gap: int, lost_after: int,
                 ball_diameter_px: float, static_check_steps: int = 4):
        self.gate = gate_px
        self.max_gap = max_gap
        self.lost_after = lost_after
        self.diam = ball_diameter_px
        self.n_check = static_check_steps
        self.track = BallTrack()
        self._pos: np.ndarray | None = None
        self._vel = np.zeros(2, np.float32)
        self._last_seen = -10**9
        self._pending_recovery: int | None = None
        self.static_rejections = 0
        self.static_points: list[tuple[float, float]] = []   # rejected static balls
        self._static_court = False                           # True: stored in court-plane coords
        self._H: np.ndarray | None = None

    def _static_in_image(self) -> list[tuple[float, float]]:
        if not self._static_court or self._H is None:
            return self.static_points
        p = np.float32(self.static_points).reshape(-1, 1, 2)
        return [tuple(q) for q in cv2.perspectiveTransform(p, self._H).reshape(-1, 2)]

    def _predict(self, f: int) -> np.ndarray | None:
        if self._pos is None:
            return None
        return self._pos + self._vel * (f - self._last_seen)

    def update(self, f: int, balls: np.ndarray, scores: np.ndarray,
               H: np.ndarray | None = None) -> tuple[float, float] | None:
        """balls (M,2) centres; returns the ball position used for frame f, or None.

        H (court -> pixels) lets rejected static balls be remembered through camera pans: a fixed
        3D point keeps the same ray-floor intersection while a broadcast camera rotates and zooms.
        """
        balls = np.asarray(balls, np.float32).reshape(-1, 2)
        scores = np.asarray(scores, np.float32).reshape(-1)
        self._H = H
        if self.static_points and len(balls):
            sp = np.asarray(self._static_in_image(), np.float32)
            near = (np.linalg.norm(balls[:, None] - sp[None], axis=2) <= 1.5 * self.diam).any(1)
            balls, scores = balls[~near], scores[~near]
        lost = self._pos is None or (f - self._last_seen) > self.lost_after
        pred = None if lost else self._predict(f)
        choice = None
        if len(balls):
            if pred is not None:
                d = np.linalg.norm(balls - pred, axis=1)
                ok = np.where(d <= self.gate)[0]
                if len(ok):
                    choice = int(ok[np.argmin(d[ok] - 20.0 * scores[ok])])
            else:
                choice = int(np.argmax(scores))      # recovery: most confident ball in view
        if choice is not None:
            p = balls[choice]
            if self._pos is not None and not lost:
                dt = max(f - self._last_seen, 1)
                self._vel = 0.5 * self._vel + 0.5 * (p - self._pos) / dt
            else:
                self._vel = np.zeros(2, np.float32)
                self.track.recoveries.append(len(self.track.frames))
                self._pending_recovery = len(self.track.frames)
            self._pos, self._last_seen = p, f
            self._append(f, p, True)
            self._check_static()
            return float(p[0]), float(p[1])
        if pred is not None and (f - self._last_seen) <= self.max_gap:
            self._append(f, pred, False)
            return float(pred[0]), float(pred[1])
        return None

    def _append(self, f, p, obs):
        self.track.frames.append(f)
        self.track.xy.append((float(p[0]), float(p[1])))
        self.track.observed.append(obs)

    def _check_static(self):
        r = self._pending_recovery
        if r is None:
            return
        n_after = len(self.track.frames) - 1 - r
        if n_after < self.n_check:
            return
        self._pending_recovery = None
        trip = [(f, x, y) for f, (x, y) in zip(self.track.frames, self.track.xy)]
        kept = drop_static_recovery(trip, r, self.diam, self.n_check)
        if len(kept) != len(trip):
            self.static_rejections += 1
            p = np.float32([trip[r][1:]])
            if self._H is not None:
                p = cv2.perspectiveTransform(p.reshape(-1, 1, 2), np.linalg.inv(self._H)).reshape(-1, 2)
            self.static_points.append((float(p[0, 0]), float(p[0, 1])))
            self._static_court = self._H is not None
            keep_idx = list(range(r)) + list(range(r + 1 + self.n_check, len(trip)))
            self.track.frames = [self.track.frames[i] for i in keep_idx]
            self.track.xy = [self.track.xy[i] for i in keep_idx]
            self.track.observed = [self.track.observed[i] for i in keep_idx]
            self.track.recoveries = [i for i in self.track.recoveries if i < r]
            self._pos = None                      # force a fresh recovery next frame
