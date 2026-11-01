"""The pipeline: per-frame perception, then per-rally logic, then identity, then the .dvw file.

    run_offline(cfg)  - whole clip; returns a Result and writes outputs/<run>/...
    run_stream(cfg)   - one pass over a file or an SRT stream; emits each rally as soon as it closes
                        (provisional numbers), and rewrites identities and the .dvw at the end.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .court import HomographyTracker, court_homography, side_of, to_court
from .detect import Detector
from .dvw import DvwMatch, write_dvw
from .identity import (AppearanceEmbedder, JerseyReader, SlotManager, TrackSummary, back_to_camera,
                       colour_class, letterbox_text, torso_quad, warp_torso)
from .io import VideoSource, fetch, to_gpu_rgb
from .pose import PoseEstimator
from .rally import Rally, label_skills, segment_rallies
from .touches import L_ANK, R_ANK, PlayerObs, Touch, detect_touches
from .track import BallTracker, PlayerTracker


class Timer:
    def __init__(self):
        self.t: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)

    def __call__(self, name: str):
        timer = self

        class _C:
            def __enter__(self_):
                self_.t0 = time.perf_counter()

            def __exit__(self_, *a):
                timer.t[name] += time.perf_counter() - self_.t0
                timer.n[name] += 1
        return _C()

    def report(self, frames: int) -> dict:
        return {k: {"total_s": round(v, 3), "ms_per_frame": round(1000 * v / max(frames, 1), 2)} for k, v in self.t.items()}


@dataclass
class Result:
    rallies: list[Rally]
    touches: list[Touch]
    slots: list[dict]
    timings: dict
    fps_video: float
    frames: int
    wall_s: float
    out_dir: Path
    dvw_path: Path | None = None
    latencies_s: list[float] = field(default_factory=list)
    numbers_at_serve: dict = field(default_factory=dict)

    def summary(self) -> dict:
        tt = [t for r in self.rallies for t in r.touches]
        return {
            "frames": self.frames, "video_fps": round(self.fps_video, 2), "wall_s": round(self.wall_s, 1),
            "processing_fps": round(self.frames / max(self.wall_s, 1e-9), 2),
            "rallies": len(self.rallies), "touches_in_rallies": len(tt),
            "touches_with_number": sum(t.number is not None for t in tt),
            "slots_with_number": sum(s["number"] is not None for s in self.slots), "slots": len(self.slots),
            "median_rally_latency_s": float(np.median(self.latencies_s)) if self.latencies_s else None,
            "numbers_at_serve": self.numbers_at_serve,
        }


def numbers_at_serve(pipe: "Pipeline", rallies: list[Rally], sm: SlotManager | None) -> dict:
    """The blog's metric: of the players on court at each serve, how many have a known number.

    On court = feet inside the court plus a 1 m margin (the server is allowed 3 m behind the line).
    Reported overall and per side, because the team facing the camera is the weak spot.
    """
    known = {0: 0, 1: 0}
    total = {0: 0, 1: 0}
    for r in rallies:
        f = r.touches[0].frame
        for p in pipe.players_by_frame.get(f, []):
            x, y = p.court_xy
            if not (-1.0 <= x <= 10.0 and -3.5 <= y <= 21.5) or p.side not in (0, 1):
                continue
            total[p.side] += 1
            known[p.side] += int(sm is not None and sm.number_for(p.tid) is not None)
    k, t = known[0] + known[1], total[0] + total[1]
    return {"known": k, "total": t, "near": f"{known[0]}/{total[0]}", "far": f"{known[1]}/{total[1]}"}


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        dev = cfg.device
        self.timer = Timer()
        with self.timer("load_models"):
            self.det = Detector(cfg.detector, dev)
            self.pose = PoseEstimator(cfg.pose, dev)
            self.reader = JerseyReader(cfg.identity.parseq_hub, cfg.identity.parseq_model, dev) if cfg.identity.enabled else None
            self.embedder = AppearanceEmbedder(cfg.identity.dino_model, dev) if cfg.identity.enabled else None
        self.ptrack = PlayerTracker()
        b = cfg.ball
        self.btrack = BallTracker(b.gate_px, b.max_gap, b.lost_after, b.ball_diameter_px, b.static_check_steps)
        self.htrack: HomographyTracker | None = None
        self.players_by_frame: dict[int, list[PlayerObs]] = {}
        self.ball_by_frame: dict[int, tuple[float, float]] = {}
        self.H_by_frame: dict[int, np.ndarray] = {}
        self.tracks: dict[int, TrackSummary] = {}
        self._read_queue: list[tuple[int, np.ndarray]] = []
        self._emb_queue: list[tuple[int, np.ndarray]] = []
        self._read_counter: dict[int, int] = defaultdict(int)
        self.frames = 0

    # ------------------------------------------------------------------ per frame
    def _init_h(self):
        c = self.cfg.court
        if len(c.seed_pixels) < 4 or len(c.seed_pixels) != len(c.seed_court):
            raise ValueError("court.seed_pixels / seed_court need >= 4 matching points (see notebook 01)")
        H0, _ = court_homography(np.float32(c.seed_court), np.float32(c.seed_pixels), c.ransac_px)
        self.htrack = HomographyTracker(H0, c.ransac_px)

    def process_batch(self, frames):
        import cv2

        cfg = self.cfg
        rgb01, bgr255 = to_gpu_rgb(frames, cfg.device)
        with self.timer("detect"):
            dets = self.det(rgb01)
        for k, (fr, d) in enumerate(zip(frames, dets)):
            f = fr.idx
            with self.timer("track_players"):
                ids = self.ptrack.update(f, d.player_boxes, d.player_scores)
                keep = ids >= 0
                boxes, tids = d.player_boxes[keep], ids[keep]
            with self.timer("homography"):
                if self.htrack is None:
                    self._init_h()
                    H = self.htrack.H
                    self.htrack.update(cv2.cvtColor(fr.bgr, cv2.COLOR_BGR2GRAY), d.player_boxes)
                elif cfg.court.propagate:
                    H = self.htrack.update(cv2.cvtColor(fr.bgr, cv2.COLOR_BGR2GRAY), d.player_boxes)
                else:
                    H = self.htrack.H
                self.H_by_frame[f] = H.copy()
            with self.timer("pose"):
                kp, kpc = self.pose(bgr255[k], fr.bgr, boxes)
            with self.timer("court_coords"):
                feet = np.zeros((len(boxes), 2), np.float32)
                for j, bx in enumerate(boxes):
                    if len(kp) and min(kpc[j][L_ANK], kpc[j][R_ANK]) > 0.4:
                        feet[j] = 0.5 * (kp[j][L_ANK] + kp[j][R_ANK])
                    else:
                        feet[j] = (0.5 * (bx[0] + bx[2]), bx[3])
                cxy = to_court(H, feet)
                sides = side_of(cxy) if len(cxy) else np.zeros(0, np.int8)
                obs = [PlayerObs(int(t), boxes[j], kp[j] if len(kp) else None, kpc[j] if len(kp) else None,
                                 cxy[j], int(sides[j])) for j, t in enumerate(tids)]
                self.players_by_frame[f] = obs
            with self.timer("track_ball"):
                pos = self.btrack.update(f, d.ball_xy, d.ball_scores, H)
                if pos is not None:
                    self.ball_by_frame[f] = pos
            if self.reader is not None:
                with self.timer("identity_sampling"):
                    self._sample_identity(fr, obs)
            self.frames += 1
        if self.reader is not None:
            with self.timer("identity_models"):
                self._flush_identity(force=False)

    def _sample_identity(self, fr, obs: list[PlayerObs]):
        ic = self.cfg.identity
        for p in obs:
            ts = self.tracks.setdefault(p.tid, TrackSummary(p.tid))
            ts.frames.append(fr.idx)
            ts.court.append((float(p.court_xy[0]), float(p.court_xy[1])))
            if p.kp is None:
                continue
            q = torso_quad(p.kp, p.kpc)
            if q is None:
                continue
            if fr.idx % 5 == 0:
                ts.colours.append(colour_class(warp_torso(fr.bgr, q, 32)))
            if fr.idx % 10 == 0:
                x1, y1, x2, y2 = np.clip(p.box, 0, None).astype(int)
                crop = fr.bgr[y1:y2, x1:x2]
                if crop.size:
                    self._emb_queue.append((p.tid, crop))
            if back_to_camera(p.kp, p.kpc, ic.min_shoulder_conf, ic.min_shoulder_px):
                self._read_counter[p.tid] += 1
                if self._read_counter[p.tid] % ic.read_every == 0:
                    self._read_queue.append((p.tid, letterbox_text(warp_torso(fr.bgr, q, 64))))

    def _flush_identity(self, force: bool):
        if self._read_queue and (force or len(self._read_queue) >= 64):
            reads = self.reader([c for _, c in self._read_queue])
            for (tid, _), (num, conf) in zip(self._read_queue, reads):
                if num and conf >= self.cfg.identity.min_read_conf:
                    self.tracks[tid].reads.append((num, conf))
            self._read_queue.clear()
        if self._emb_queue and (force or len(self._emb_queue) >= 64):
            embs = self.embedder([c for _, c in self._emb_queue])
            for (tid, _), e in zip(self._emb_queue, embs):
                self.tracks[tid].emb.append(e)
            self._emb_queue.clear()

    # ------------------------------------------------------------------ per rally / set
    def events(self, fps: float, frame_lo: int = 0, frame_hi: int | None = None) -> list[Rally]:
        frames = np.array(sorted(f for f in self.ball_by_frame if f >= frame_lo and (frame_hi is None or f <= frame_hi)))
        if len(frames) == 0:
            return []
        xy = np.array([self.ball_by_frame[f] for f in frames], np.float32)
        touches = detect_touches(frames, xy, self.players_by_frame, fps, self.cfg.touch)
        rallies = segment_rallies(touches, frames, fps, self.cfg.rally.end_gap_s, self.cfg.rally.behind_line_m)
        label_skills(rallies, self.cfg.touch.block_window_s)
        return rallies

    def identities(self, set_start_frame: int = 0) -> SlotManager | None:
        if self.reader is None:
            return None
        self._flush_identity(force=True)
        sm = SlotManager(self.cfg.identity)
        sm.build(self.tracks, set_start_frame)
        return sm

    def finish(self, rallies: list[Rally], sm: SlotManager | None, out_dir: Path) -> Path:
        near_home = self.cfg.court.near_side_is_home
        for r in rallies:
            for t in r.touches:
                t.number = sm.number_for(t.tid) if sm else None
                t.team = "*" if (t.side == 0) == near_home else "a"
        home_nums = [s["number"] for s in (sm.table() if sm else []) if s["number"] is not None and (s["side"] == "near") == near_home]
        away_nums = [s["number"] for s in (sm.table() if sm else []) if s["number"] is not None and (s["side"] == "near") != near_home]
        o = self.cfg.output
        return write_dvw(out_dir / "match_auto.dvw", rallies,
                         DvwMatch(o.home_team, o.away_team, o.match_date, home_nums, away_nums),
                         set_no=1, near_is_home=near_home)


def _write_outputs(res: Result, cfg: Config, pipe: Pipeline):
    out = res.out_dir
    rows = [{"rally": r.idx, "frame": t.frame, "t": round(t.t, 3), "side": t.side, "team": t.team,
             "skill": t.skill, "number": t.number, "tid": t.tid, "at_net": t.at_net, "raised": t.raised,
             "court_x": t.court_xy[0] if t.court_xy else None, "court_y": t.court_xy[1] if t.court_xy else None,
             "ball_x": t.ball_xy[0], "ball_y": t.ball_xy[1], "angle": round(t.angle_deg, 1)}
            for r in res.rallies for t in r.touches]
    (out / "events.json").write_text(json.dumps(rows, indent=1))
    (out / "slots.json").write_text(json.dumps(res.slots, indent=1))
    (out / "summary.json").write_text(json.dumps({"summary": res.summary(), "timings": res.timings,
                                                  "static_ball_rejections": pipe.btrack.static_rejections,
                                                  "homography_failures": pipe.htrack.failures if pipe.htrack else None,
                                                  "config": cfg.to_dict()}, indent=1, default=str))


def run_offline(cfg: Config, run_name: str | None = None, max_frames: int | None = None) -> Result:
    t0 = time.perf_counter()
    path = fetch(cfg.video.source, cfg.video.local_cache)
    src = VideoSource(path, cfg.video.decoder, cfg.device, cfg.video.start_s, cfg.video.end_s, cfg.video.stride)
    out_dir = Path(cfg.output.out_dir) / (run_name or time.strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(cfg)
    fps = src.fps / cfg.video.stride
    batch = []
    for fr in src:
        batch.append(fr)
        if len(batch) == cfg.video.batch:
            pipe.process_batch(batch)
            batch = []
        if max_frames and pipe.frames + len(batch) >= max_frames:
            break
    if batch:
        pipe.process_batch(batch)
    with pipe.timer("events"):
        rallies = pipe.events(src.fps)
    with pipe.timer("identity_slots"):
        sm = pipe.identities(rallies[0].touches[0].frame if rallies else 0)
    dvw = pipe.finish(rallies, sm, out_dir)
    res = Result(rallies, [t for r in rallies for t in r.touches], sm.table() if sm else [],
                 pipe.timer.report(pipe.frames), src.fps, pipe.frames, time.perf_counter() - t0, out_dir, dvw)
    res.numbers_at_serve = numbers_at_serve(pipe, rallies, sm)
    _write_outputs(res, cfg, pipe)
    if cfg.output.render:
        with pipe.timer("render"):
            render(path, pipe, res, out_dir / "render.mp4", sm)
        res.wall_s = time.perf_counter() - t0
        res.timings = pipe.timer.report(pipe.frames)
        _write_outputs(res, cfg, pipe)
    return res


def run_stream(cfg: Config, run_name: str | None = None, max_frames: int | None = None, on_rally=None) -> Result:
    """One decode pass; a rally is emitted when the ball has been gone for end_gap_s after touches.

    Latency per rally = wall time at emit - wall time when the rally's last frame was decoded.
    Identities are provisional per rally (votes so far) and re-voted at the end of the run.
    """
    t0 = time.perf_counter()
    path = fetch(cfg.video.source, cfg.video.local_cache)
    src = VideoSource(path, cfg.video.decoder, cfg.device, cfg.video.start_s, cfg.video.end_s, cfg.video.stride)
    out_dir = Path(cfg.output.out_dir) / (run_name or ("live-" + time.strftime("%Y%m%d-%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(cfg)
    gap_frames = int(cfg.rally.end_gap_s * src.fps)
    emitted_upto = -1
    wall_at: dict[int, float] = {}
    latencies: list[float] = []
    emitted: list[Rally] = []
    batch = []
    for fr in src:
        batch.append(fr)
        wall_at[fr.idx] = fr.wall
        if len(batch) < cfg.video.batch:
            continue
        pipe.process_batch(batch)
        batch = []
        last_ball = max(pipe.ball_by_frame) if pipe.ball_by_frame else -1
        now_f = fr.idx
        if last_ball > emitted_upto and now_f - last_ball >= gap_frames:
            new = [r for r in pipe.events(src.fps, emitted_upto + 1, last_ball) if r.touches]
            sm = pipe.identities(new[0].touches[0].frame) if (new and pipe.reader) else None
            for r in new:
                for t in r.touches:
                    t.number = sm.number_for(t.tid) if sm else None
                end_f = r.touches[-1].frame
                lat = time.perf_counter() - wall_at.get(end_f, time.perf_counter())
                latencies.append(lat)
                r.idx = len(emitted)
                emitted.append(r)
                if on_rally:
                    on_rally(r, lat)
            emitted_upto = last_ball
        if max_frames and pipe.frames >= max_frames:
            break
    if batch:
        pipe.process_batch(batch)
    # set-end correction: re-run events and identity on everything
    rallies = pipe.events(src.fps)
    sm = pipe.identities(rallies[0].touches[0].frame if rallies else 0)
    dvw = pipe.finish(rallies, sm, out_dir)
    res = Result(rallies, [t for r in rallies for t in r.touches], sm.table() if sm else [],
                 pipe.timer.report(pipe.frames), src.fps, pipe.frames, time.perf_counter() - t0, out_dir, dvw, latencies)
    res.numbers_at_serve = numbers_at_serve(pipe, rallies, sm)
    _write_outputs(res, cfg, pipe)
    return res


def render(path: str, pipe: Pipeline, res: Result, out_path: Path, sm: SlotManager | None):
    """Second, CPU-decoded pass for drawing only (does not influence any event)."""
    from .render import Renderer

    src = VideoSource(path, "cpu", "cpu", pipe.cfg.video.start_s, pipe.cfg.video.end_s, pipe.cfg.video.stride)
    numbers = {}
    if sm:
        for tid in sm.tid_to_slot:
            numbers[tid] = sm.number_for(tid)
    touch_at = {t.frame: t for t in res.touches}
    rr = Renderer(str(out_path), src.fps / pipe.cfg.video.stride, src.size)
    n = 0
    for fr in src:
        if n >= pipe.frames:
            break
        rr.frame(fr.bgr, pipe.H_by_frame.get(fr.idx), pipe.players_by_frame.get(fr.idx, []),
                 pipe.ball_by_frame.get(fr.idx), touch_at.get(fr.idx), numbers, fr.idx)
        n += 1
    rr.close()
