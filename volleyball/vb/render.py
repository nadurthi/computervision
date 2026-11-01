"""Overlay video: court lines, player boxes with slot numbers, ball trail, touch labels."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .court import draw_court

SKILL_COL = {"S": (0, 200, 255), "R": (255, 200, 0), "E": (200, 255, 0), "A": (0, 0, 255),
             "B": (255, 0, 255), "D": (0, 255, 0), "F": (255, 255, 255)}


class Renderer:
    def __init__(self, path: str, fps: float, size: tuple[int, int], trail: int = 20):
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self.trail = deque(maxlen=trail)
        self.banner: tuple[str, int] | None = None

    def frame(self, img: np.ndarray, H: np.ndarray | None, players, ball, touch=None, numbers=None, f: int = 0):
        out = img.copy()
        if H is not None:
            draw_court(out, H, (0, 220, 255), 1)
        for p in players:
            x1, y1, x2, y2 = map(int, p.box)
            col = (80, 160, 255) if p.side == 0 else (255, 160, 80)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            n = numbers.get(p.tid) if numbers else None
            cv2.putText(out, f"#{n}" if n is not None else f"t{p.tid}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if ball is not None:
            self.trail.append(tuple(int(v) for v in ball))
        for a, b in zip(self.trail, list(self.trail)[1:]):
            cv2.line(out, a, b, (0, 255, 255), 2, cv2.LINE_AA)
        if touch is not None:
            lab = f"{touch.skill} {('#' + str(touch.number)) if touch.number is not None else ''}".strip()
            self.banner = (lab, f + 25)
            cv2.circle(out, tuple(int(v) for v in touch.ball_xy), 14, SKILL_COL.get(touch.skill, (255, 255, 255)), 3)
        if self.banner and f <= self.banner[1]:
            cv2.putText(out, self.banner[0], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
        self.w.write(out)

    def close(self):
        self.w.release()
