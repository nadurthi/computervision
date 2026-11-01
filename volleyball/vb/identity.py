"""Player identity: shirt numbers on set-start slots, carried across rallies.

1. Read numbers only where they are readable: frames where pose says the back is to the camera
   (left shoulder on the image left, face not visible). Back digits are ~19-25 px from an end-line
   camera; front digits are ~10 px and are not attempted.
2. PARSeq reads the torso crop. Its output is restricted to digits and to at most two characters,
   and every read keeps its confidence.
3. Reads vote on slots, not on tracks: tracks break at the net, so track fragments are assigned to
   set-start slots (6 + 6 + liberos) using tracker continuity, a DINOv2-small appearance embedding
   and a shirt-colour class, and every read on any fragment of a slot votes for that slot.
4. The libero is the slot whose majority colour class differs from its team's; majority over the
   whole track, never the mean colour (lit red averaged with black gives a maroon in between).
5. Referees and staff are decided by voting where a track stood over its whole life.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from .touches import L_HIP, L_SH, NOSE, R_HIP, R_SH

# ------------------------------------------------------------------ geometry ----------------


def back_to_camera(kp: np.ndarray, conf: np.ndarray, min_conf: float = 0.5, min_width_px: float = 8.0) -> bool:
    if conf[L_SH] < min_conf or conf[R_SH] < min_conf:
        return False
    width = kp[R_SH][0] - kp[L_SH][0]          # > 0: left shoulder on the image left -> back view
    return bool(width > min_width_px and conf[NOSE] < 0.3)


def torso_quad(kp: np.ndarray, conf: np.ndarray, min_conf: float = 0.4) -> np.ndarray | None:
    """Quadrilateral L-shoulder, R-shoulder, R-hip, L-hip, slightly enlarged. None if hips are missing."""
    idx = [L_SH, R_SH, R_HIP, L_HIP]
    if min(conf[i] for i in idx) < min_conf:
        return None
    q = kp[idx].astype(np.float32)
    c = q.mean(0)
    return c + 1.15 * (q - c)


def warp_torso(img_bgr: np.ndarray, quad: np.ndarray, size: int = 64) -> np.ndarray:
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def letterbox_text(crop_bgr: np.ndarray, out_hw=(32, 128)) -> np.ndarray:
    """Square torso crop -> PARSeq's 32x128 text-line input, digits kept at their aspect ratio."""
    h, w = out_hw
    sq = cv2.resize(crop_bgr, (h, h), interpolation=cv2.INTER_AREA)
    pad = np.empty((h, w, 3), np.uint8)
    pad[:] = sq.reshape(-1, 3).mean(0).astype(np.uint8)
    x0 = (w - h) // 2
    pad[:, x0:x0 + h] = sq
    return pad


# ------------------------------------------------------------------ colour class ------------

COLOUR_NAMES = ("black", "white", "grey", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink")


def colour_class(crop_bgr: np.ndarray) -> str:
    """Dominant colour class of a shirt crop: every pixel votes, the majority wins."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0] * 2, hsv[:, 1], hsv[:, 2]        # hue in degrees
    lab = np.full(len(h), "grey", object)
    lab[v < 60] = "black"
    lab[(v >= 60) & (s < 40) & (v > 190)] = "white"
    chrom = (v >= 60) & (s >= 60)
    bins = [(0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 160, "green"), (160, 200, "cyan"),
            (200, 260, "blue"), (260, 290, "purple"), (290, 340, "pink"), (340, 361, "red")]
    for lo, hi, name in bins:
        lab[chrom & (h >= lo) & (h < hi)] = name
    return Counter(lab.tolist()).most_common(1)[0][0]


# ------------------------------------------------------------------ models ------------------


class JerseyReader:
    """PARSeq from torch.hub, output restricted to digits, max two characters."""

    def __init__(self, hub_repo: str = "baudm/parseq", model: str = "parseq", device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = device
        self.model = torch.hub.load(hub_repo, model, pretrained=True, trust_repo=True).eval().to(device)
        tok = self.model.tokenizer
        itos = list(tok._itos)                    # (EOS,) + charset + (BOS, PAD)
        n_out = len(itos) - 2
        allowed = [0] + [i for i in range(1, n_out) if itos[i].isdigit()]
        mask = torch.full((n_out,), float("-inf"))
        mask[allowed] = 0
        self.mask = mask.to(device)

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def __call__(self, crops_bgr: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops_bgr:
            return []
        torch = self.torch
        x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        x = (x - 0.5) / 0.5
        with torch.inference_mode():
            logits = self.model(x)
            probs = (logits + self.mask).softmax(-1)
            labels, confs = self.tokenizer.decode(probs)
        out = []
        for lab, c in zip(labels, confs):
            conf = float(c.prod()) if len(c) else 0.0
            out.append((lab, conf) if 1 <= len(lab) <= 2 else ("", 0.0))
        return out


class AppearanceEmbedder:
    """DINOv2-small CLS embedding of a player crop (224 x 112, both multiples of the 14 px patch)."""

    def __init__(self, model: str = "dinov2_vits14", device: str = "cuda"):
        import torch

        self.torch = torch
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model, trust_repo=True).eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        if not crops_bgr:
            return np.zeros((0, 384), np.float32)
        torch = self.torch
        x = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (112, 224)) for c in crops_bgr])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(self.device)
        with torch.inference_mode():
            f = self.model((x - self.mean) / self.std)
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()


# ------------------------------------------------------------------ voting ------------------


def vote_number(reads: list[tuple[str, float]], accept_ratio: float = 0.6, accept_mass: float = 1.5,
                roster: list[int] | None = None) -> tuple[int | None, float]:
    score: dict[str, float] = defaultdict(float)
    for num, c in reads:
        if not num:
            continue
        if roster and int(num) not in roster:
            continue
        score[num] += c
    if not score:
        return None, 0.0
    best, s = max(sorted(score.items()), key=lambda kv: kv[1])     # sorted: deterministic ties
    total = sum(score.values())
    if s / total > accept_ratio and s > accept_mass:
        return int(best), s / total
    return None, s / total


@dataclass
class TrackSummary:
    tid: int
    frames: list[int] = field(default_factory=list)
    court: list[tuple[float, float]] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    reads: list[tuple[str, float]] = field(default_factory=list)
    emb: list[np.ndarray] = field(default_factory=list)

    @property
    def span(self) -> tuple[int, int]:
        return (min(self.frames), max(self.frames)) if self.frames else (0, -1)

    def majority_side(self) -> int:
        ys = [y for _, y in self.court]
        return int(np.median(ys) >= 9.0) if ys else -1

    def majority_colour(self) -> str | None:
        return Counter(self.colours).most_common(1)[0][0] if self.colours else None

    def mean_emb(self) -> np.ndarray | None:
        if not self.emb:
            return None
        m = np.mean(self.emb, 0)
        return m / (np.linalg.norm(m) + 1e-9)


def is_staff(ts: TrackSummary, margin_m: float = 0.75) -> bool:
    """Referees, line judges, bench: most observations outside the court's playing width."""
    if not ts.court:
        return True
    xs = np.array([x for x, _ in ts.court])
    outside = (xs < -margin_m) | (xs > 9.0 + margin_m)
    return bool(outside.mean() > 0.6)


@dataclass
class Slot:
    key: str                  # e.g. "near-3"
    side: int
    tids: list[int] = field(default_factory=list)
    proto: np.ndarray | None = None
    colour: str | None = None
    number: int | None = None
    number_share: float = 0.0
    is_libero: bool = False
    is_sub: bool = False
    busy: list[tuple[int, int]] = field(default_factory=list)


class SlotManager:
    """Assign track fragments to set-start slots and vote numbers on slots."""

    def __init__(self, cfg, max_cost: float = 0.55):
        self.cfg = cfg
        self.max_cost = max_cost
        self.slots: list[Slot] = []
        self.tid_to_slot: dict[int, str] = {}

    def build(self, tracks: dict[int, TrackSummary], set_start_frame: int) -> None:
        players = {t: ts for t, ts in tracks.items() if not is_staff(ts)}
        team_colour: dict[int, str] = {}
        for side in (0, 1):
            cols = [ts.majority_colour() for ts in players.values() if ts.majority_side() == side and ts.majority_colour()]
            team_colour[side] = Counter(cols).most_common(1)[0][0] if cols else ""
        # seed slots with tracks alive at the start of the set, left to right, per side
        alive = [ts for ts in players.values() if ts.span[0] <= set_start_frame + 25 <= ts.span[1] + 25]
        for side in (0, 1):
            seeds = sorted([ts for ts in alive if ts.majority_side() == side],
                           key=lambda ts: ts.court[0][0] if ts.court else 0)[:7]
            for k, ts in enumerate(seeds):
                self._add_slot(f"{'near' if side == 0 else 'far'}-{k + 1}", side, ts, team_colour[side])
        # the rest in time order
        for ts in sorted(players.values(), key=lambda t: (t.span[0], t.tid)):
            if ts.tid in self.tid_to_slot:
                continue
            self._assign(ts, team_colour)
        self._vote(tracks)

    def _add_slot(self, key, side, ts, team_col, sub=False):
        s = Slot(key=key, side=side, is_sub=sub)
        s.colour = ts.majority_colour()
        s.is_libero = bool(s.colour and team_col and s.colour != team_col)
        self.slots.append(s)
        self._attach(s, ts)

    def _attach(self, s: Slot, ts: TrackSummary):
        s.tids.append(ts.tid)
        s.busy.append(ts.span)
        e = ts.mean_emb()
        if e is not None:
            s.proto = e if s.proto is None else (s.proto * (len(s.tids) - 1) + e) / len(s.tids)
            s.proto = s.proto / (np.linalg.norm(s.proto) + 1e-9)
        self.tid_to_slot[ts.tid] = s.key

    def _assign(self, ts: TrackSummary, team_colour):
        side = ts.majority_side()
        e = ts.mean_emb()
        best, best_cost = None, float("inf")
        a, b = ts.span
        for s in self.slots:
            if s.side != side or any(not (b < x or a > y) for x, y in s.busy):
                continue       # wrong side, or the slot already has a player on court at that time
            cost = 0.5
            if e is not None and s.proto is not None:
                cost = 1.0 - float(e @ s.proto)
            if ts.majority_colour() != s.colour:
                cost += 0.3
            if cost < best_cost:
                best, best_cost = s, cost
        if best is not None and best_cost <= self.max_cost:
            self._attach(best, ts)
        else:
            n = sum(1 for s in self.slots if s.side == side)
            self._add_slot(f"{'near' if side == 0 else 'far'}-{n + 1}", side, ts, team_colour.get(side, ""), sub=True)

    def _vote(self, tracks):
        for s in self.slots:
            reads = [r for t in s.tids for r in tracks[t].reads]
            roster = self.cfg.home_roster if s.side == 0 else self.cfg.away_roster
            s.number, s.number_share = vote_number(reads, self.cfg.accept_ratio, self.cfg.accept_mass, roster or None)

    def number_for(self, tid: int | None) -> int | None:
        if tid is None or tid not in self.tid_to_slot:
            return None
        key = self.tid_to_slot[tid]
        return next(s.number for s in self.slots if s.key == key)

    def table(self) -> list[dict]:
        return [{"slot": s.key, "side": "near" if s.side == 0 else "far", "number": s.number,
                 "share": round(s.number_share, 2), "libero": s.is_libero, "sub": s.is_sub,
                 "colour": s.colour, "tracks": len(s.tids)} for s in self.slots]
