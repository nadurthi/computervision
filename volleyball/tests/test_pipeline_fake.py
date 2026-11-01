"""Run the whole pipeline on a generated clip with stand-in models (no GPU, no downloads).

The clip: a court drawn through a known homography, seven players, and a ball that is served from
behind the near end line, received, set and attacked on the far side, dug, set and attacked on the
near side. The fake detector and pose model return ground truth, so this checks the plumbing:
decoding, tracking, homography, touches, rallies, skills, identity voting, .dvw and the render.
"""
import json

import cv2
import numpy as np
import pytest

from conftest import HAS_DV
from vb.court import draw_court, to_image
from vb.detect import FrameDets
from vb.touches import L_ANK, L_SH, L_WR, R_ANK, R_SH, R_WR, NOSE

H = np.array([[60.0, 8.0, 400.0], [0.0, -25.0, 900.0], [0.0, 0.012, 1.0]])
W, HGT, FPS = 1280, 960, 25.0
# player id -> (court X, Y, shirt number)
PLAYERS = {1: (4.5, -1.0, 7), 2: (3.0, 13.0, 3), 3: (4.0, 10.5, 9), 4: (5.0, 10.0, 11),
           5: (4.0, 4.0, 2), 6: (5.0, 7.5, 5), 7: (4.5, 8.0, 8)}
CONTACTS = [(10, 1), (40, 2), (55, 3), (70, 4), (95, 5), (110, 6), (125, 7)]


def _hand(pid):
    X, Y, _ = PLAYERS[pid]
    return to_image(H, np.float32([[X, Y]]))[0] + np.float32([0, -120])


def _ball(f):
    for (f0, p0), (f1, p1) in zip(CONTACTS, CONTACTS[1:]):
        if f0 <= f < f1:
            a = (f - f0) / (f1 - f0)
            return _hand(p0) + a * (_hand(p1) - _hand(p0)) + np.float32([0, -200 * a * (1 - a)])
    fl, pl = CONTACTS[-1]
    if fl <= f < fl + 12:
        return _hand(pl) + np.float32([3 * (f - fl), 8 * (f - fl)])
    return None


def _box(pid):
    X, Y, _ = PLAYERS[pid]
    feet = to_image(H, np.float32([[X, Y]]))[0]
    return np.float32([feet[0] - 25, feet[1] - 160, feet[0] + 25, feet[1]])


def make_clip(path, n=170):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, HGT))
    for f in range(n):
        img = np.full((HGT, W, 3), (60, 110, 160), np.uint8)
        draw_court(img, H, (255, 255, 255), 3)
        for pid in PLAYERS:
            x1, y1, x2, y2 = _box(pid).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 220) if PLAYERS[pid][1] < 9 else (220, 120, 40), -1)
        b = _ball(f)
        if b is not None:
            cv2.circle(img, (int(b[0]), int(b[1])), 7, (0, 230, 255), -1)
        vw.write(img)
    vw.release()


class FakeDetector:
    def __init__(self, *a, **k):
        self.f = 0

    def __call__(self, frames):
        out = []
        for fr in frames:
            boxes = np.stack([_box(p) for p in PLAYERS])
            b = _ball(fr.idx)
            bxy = np.zeros((0, 2), np.float32) if b is None else b[None].astype(np.float32)
            out.append(FrameDets(boxes, np.full(len(boxes), 0.9, np.float32), bxy,
                                 np.full(len(bxy), 0.8, np.float32), np.zeros((len(bxy), 4), np.float32)))
        return out


class FakePose:
    def __init__(self, *a, **k):
        pass

    def __call__(self, gpu, bgr, boxes):
        kps, scs = [], []
        for bx in boxes:
            kp = np.zeros((17, 2), np.float32)
            cx, top, bot = 0.5 * (bx[0] + bx[2]), bx[1], bx[3]
            kp[L_WR] = kp[R_WR] = (cx, top + 40)                 # hands 120 px above the feet
            kp[L_SH], kp[R_SH] = (cx - 12, top + 60), (cx + 12, top + 60)   # back view
            kp[11], kp[12] = (cx - 10, top + 110), (cx + 10, top + 110)
            kp[L_ANK] = kp[R_ANK] = (cx, bot)
            kp[7], kp[8] = (cx - 15, top + 50), (cx + 15, top + 50)
            sc = np.full(17, 0.9, np.float32)
            sc[NOSE] = 0.1
            kps.append(kp)
            scs.append(sc)
        return np.float32(kps), np.float32(scs)


class FakeReader:
    """Returns the true number for whichever player the torso crop came from (looked up by colour)."""
    def __init__(self, *a, **k):
        self.calls = 0

    def __call__(self, crops):
        self.calls += len(crops)
        return [("", 0.0) for _ in crops]      # replaced below by track-aware reads


class FakeEmbedder:
    def __init__(self, *a, **k):
        pass

    def __call__(self, crops):
        return np.tile(np.eye(8, dtype=np.float32)[0], (len(crops), 1))


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    import vb.pipeline as P
    from vb.config import load_config

    clip = tmp_path / "clip.mp4"
    make_clip(clip)
    monkeypatch.setattr(P, "Detector", FakeDetector)
    monkeypatch.setattr(P, "PoseEstimator", FakePose)
    monkeypatch.setattr(P, "JerseyReader", FakeReader)
    monkeypatch.setattr(P, "AppearanceEmbedder", FakeEmbedder)
    monkeypatch.setattr(P, "to_gpu_rgb", lambda frames, dev: (frames, [None] * len(frames)))

    # track-aware reads: every queued crop of track t reads the number of the player t was matched to
    orig_flush = P.Pipeline._flush_identity

    def flush(self, force):
        for i, (tid, crop) in enumerate(self._read_queue):
            pass
        reads = []
        for tid, _ in self._read_queue:
            # the fake tracker assigns ids in left-to-right order of first appearance; map via court position
            X, Y = self.tracks[tid].court[-1]
            pid = min(PLAYERS, key=lambda p: (PLAYERS[p][0] - X) ** 2 + (PLAYERS[p][1] - Y) ** 2)
            reads.append((str(PLAYERS[pid][2]), 0.95))
        for (tid, _), (num, conf) in zip(self._read_queue, reads):
            self.tracks[tid].reads.append((num, conf))
        self._read_queue.clear()
        if self._emb_queue:
            for tid, _ in self._emb_queue:
                self.tracks[tid].emb.append(np.eye(8, dtype=np.float32)[tid % 8])
            self._emb_queue.clear()
    monkeypatch.setattr(P.Pipeline, "_flush_identity", flush)

    seed_court = [[0, 0], [9, 0], [9, 18], [0, 18], [0, 9], [9, 9]]
    seed_px = to_image(H, np.float32(seed_court)).tolist()
    cfg = load_config(None, video={"source": str(clip), "decoder": "cpu", "batch": 8},
                      court={"seed_pixels": seed_px, "seed_court": seed_court},
                      output={"out_dir": str(tmp_path / "out")}, device="cpu")
    cfg.identity.accept_mass = 1.0
    res = P.run_offline(cfg, run_name="fake")

    assert len(res.rallies) == 1, res.summary()
    r = res.rallies[0]
    assert " ".join(t.skill for t in r.touches) == "S R E A D E A", r.labels
    assert [t.number for t in r.touches] == [7, 3, 9, 11, 2, 5, 8]
    assert [t.team for t in r.touches] == ["*", "a", "a", "a", "*", "*", "*"]
    nas = res.summary()["numbers_at_serve"]
    assert nas["total"] == 7 and nas["known"] == 7, nas
    out = res.out_dir
    for name in ("events.json", "slots.json", "summary.json", "match_auto.dvw", "render.mp4"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    ev = json.loads((out / "events.json").read_text())
    assert len(ev) == 7
    cap = cv2.VideoCapture(str(out / "render.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == res.frames
    if HAS_DV:
        from vb.dvw import check_with_datavolley
        plays, problems = check_with_datavolley(out / "match_auto.dvw")
        assert not problems
        codes = [p["code"] for p in plays]
        assert codes[:4] == ["*07SH+", "a03RH+", "a09EH+", "a11AH+"], codes
        assert plays[-1]["code"] == "*p01:00"
