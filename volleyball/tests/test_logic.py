"""Tests for everything that does not need a GPU model."""
import numpy as np
import pytest

from conftest import HAS_DV, HAS_TORCH
from vb.config import load_config
from vb.court import HomographyTracker, court_homography, dlt_homography, to_court, to_image
from vb.dvw import DvwMatch, court_to_dv_coord, write_dvw
from vb.identity import SlotManager, TrackSummary, back_to_camera, colour_class, vote_number
from vb.pose import (affine_theta, batched_crops_cpu, boxes_to_center_scale, crop_affines, simcc_decode)
from vb.rally import Rally, label_skills, segment_rallies
from vb.skills import TouchIn, label_rally
from vb.touches import PlayerObs, Touch, detect_touches, kink_candidates
from vb.track import BallTracker, PlayerTracker, drop_static_recovery

H_TRUE = np.array([[60.0, 8.0, 400.0], [0.0, -25.0, 900.0], [0.0, 0.012, 1.0]])


# ------------------------------------------------------------------ skills
def seq(s):
    """'n n f f' style: team letters, '!' marks a touch at the net."""
    out = []
    for i, tok in enumerate(s.split()):
        out.append(TouchIn(team=tok[0], t=0.4 * i, at_net=tok.endswith("!")))
    return out


@pytest.mark.parametrize("rally,expected", [
    ("n f f f", "S R E A"),                      # serve, reception, set, attack (kill)
    ("n f f f n! n n n", "S R E A B D E A"),     # blocked attack covered on the blocking side
    ("n f f f n n n", "S R E A D E A"),          # dig, set, attack
    ("n f n", "S R A"),                          # overpass attacked straight to the floor
    ("n f n f", "S R A D"),                      # overpass attacked, dug
    ("n f f n", "S R A D"),                      # setter dump on two
    ("n f f f n! f f f", "S R E A B D E A"),     # block rebound, attacking side covers
])
def test_label_rally(rally, expected):
    assert " ".join(label_rally(seq(rally))) == expected


def test_free_ball_after_overpass_is_seen():
    # reception overpass, opponent plays it with a set, so the first touch did not cross: F
    assert label_rally(seq("n f n n n"))[2] == "F"


# ------------------------------------------------------------------ court
def test_dlt_recovers_h():
    court = np.float32([[0, 0], [9, 0], [9, 18], [0, 18], [0, 6], [9, 12]])
    img = to_image(H_TRUE, court)
    H = dlt_homography(court, img)
    np.testing.assert_allclose(H, H_TRUE / H_TRUE[2, 2], rtol=1e-5, atol=1e-4)
    H2, inl = court_homography(court, img)
    assert inl.all()
    pts = np.float32([[4.5, 9], [1, 2], [8, 17]])
    np.testing.assert_allclose(to_court(H2, to_image(H2, pts)), pts, atol=1e-3)


def test_homography_tracker_follows_pan():
    rng = np.random.default_rng(0)
    big = (rng.random((900, 1600)) * 255).astype(np.uint8)
    import cv2
    big = cv2.GaussianBlur(big, (5, 5), 0)
    frame0 = big[100:820, 100:1380].copy()
    frame1 = big[100:820, 112:1392].copy()          # camera panned 12 px right -> content moves left
    tr = HomographyTracker(np.eye(3))
    tr.update(frame0)
    H = tr.update(frame1)
    np.testing.assert_allclose(H[:2, 2], [-12, 0], atol=0.5)


# ------------------------------------------------------------------ tracking
def test_player_tracker_keeps_ids():
    tr = PlayerTracker(min_hits=1)
    a = np.float32([[100, 100, 150, 220], [400, 100, 450, 220]])
    ids0 = tr.update(0, a, np.ones(2))
    ids1 = tr.update(1, a + 3, np.ones(2))
    assert list(ids0) == list(ids1) and len(set(ids1)) == 2


def test_static_recovery_dropped_and_blacklisted():
    bt = BallTracker(gate_px=100, max_gap=3, lost_after=3, ball_diameter_px=10, static_check_steps=4)
    # spare ball sits at (50, 20); no match ball in view
    for f in range(8):
        bt.update(f, np.float32([[50, 20]]), np.float32([0.9]))
    assert bt.static_rejections >= 1
    assert all(abs(x - 50) > 15 or abs(y - 20) > 15 for x, y in bt.track.xy)
    # a moving ball afterwards is tracked
    for k, f in enumerate(range(8, 16)):
        bt.update(f, np.float32([[50, 20], [300 + 10 * k, 400 - 5 * k]]), np.float32([0.9, 0.6]))
    assert bt.track.xy[-1][0] > 300


def test_drop_static_recovery_blog_function():
    tr = [(0, 0, 0), (1, 100, 100), (2, 101, 100), (3, 100, 101), (4, 100, 100), (5, 101, 101), (6, 150, 60)]
    assert drop_static_recovery(tr, 1, 14.0) == [(0, 0, 0), (6, 150, 60)]
    moving = [(0, 0, 0), (1, 100, 100), (2, 120, 90), (3, 140, 80), (4, 160, 70), (5, 180, 60)]
    assert drop_static_recovery(moving, 1, 14.0) == moving


# ------------------------------------------------------------------ pose crops
def _grid_sample_np(img, theta, out_hw):
    """numpy replica of torch affine_grid + grid_sample (bilinear, zeros, align_corners=False)."""
    Hs, Ws = img.shape[:2]
    Ho, Wo = out_hw
    j, i = np.meshgrid(np.arange(Wo), np.arange(Ho))
    xn, yn = (2 * j + 1) / Wo - 1, (2 * i + 1) / Ho - 1
    xs = theta[0, 0] * xn + theta[0, 1] * yn + theta[0, 2]
    ys = theta[1, 0] * xn + theta[1, 1] * yn + theta[1, 2]
    x = ((xs + 1) * Ws - 1) / 2
    y = ((ys + 1) * Hs - 1) / 2
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    out = np.zeros((Ho, Wo, img.shape[2]))
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = x0 + dx, y0 + dy
            w = (1 - np.abs(x - xx)) * (1 - np.abs(y - yy))
            ok = (xx >= 0) & (xx < Ws) & (yy >= 0) & (yy < Hs)
            out += np.where(ok[..., None], img[np.clip(yy, 0, Hs - 1), np.clip(xx, 0, Ws - 1)] * w[..., None], 0)
    return out


def test_gpu_crop_math_matches_cv2():
    rng = np.random.default_rng(1)
    img = (rng.random((360, 640, 3)) * 255).astype(np.float32)
    boxes = np.float32([[200, 80, 260, 240], [400, 120, 470, 300]])
    c, s = boxes_to_center_scale(boxes)
    A = crop_affines(c, s)
    theta = affine_theta(A, (640, 360), (192, 256))
    ref = batched_crops_cpu(img, c, s)                         # normalised, NCHW
    from vb.pose import MEAN, STD
    for n in range(len(boxes)):
        mine = (_grid_sample_np(img, theta[n], (256, 192)) - np.array(MEAN)) / np.array(STD)
        diff = np.abs(mine.transpose(2, 0, 1) - ref[n])[:, 2:-2, 2:-2]
        assert diff.max() < 0.05, diff.max()                   # < 0.05 std units ~ 3 grey levels at the very edge


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_crops_match_cv2():
    import torch
    from vb.pose import batched_crops_torch
    rng = np.random.default_rng(2)
    img = (rng.random((360, 640, 3)) * 255).astype(np.float32)
    boxes = np.float32([[200, 80, 260, 240]])
    c, s = boxes_to_center_scale(boxes)
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    got = batched_crops_torch(t, c, s).numpy()
    ref = batched_crops_cpu(img, c, s)
    assert np.abs(got - ref)[:, :, 2:-2, 2:-2].max() < 0.05


def test_simcc_decode_inverts_affine():
    c, s = boxes_to_center_scale(np.float32([[100, 50, 160, 210]]))
    K, Wx, Wy = 17, 384, 512
    sx = np.zeros((1, K, Wx), np.float32)
    sy = np.zeros((1, K, Wy), np.float32)
    sx[0, :, 192] = 1.0      # output x = 96 = centre column
    sy[0, :, 256] = 1.0      # output y = 128 = centre row
    kp, sc = simcc_decode(sx, sy, c, s)
    np.testing.assert_allclose(kp[0, 0], c[0], atol=1e-4)
    assert (sc == 1).all()


# ------------------------------------------------------------------ identity
def _kp(back=True):
    kp = np.zeros((17, 2), np.float32)
    conf = np.full(17, 0.9, np.float32)
    kp[5], kp[6] = ([100, 100], [130, 100]) if back else ([130, 100], [100, 100])
    kp[11], kp[12] = [102, 160], [128, 160]
    conf[0] = 0.1 if back else 0.9
    return kp, conf


def test_back_to_camera():
    assert back_to_camera(*_kp(True))
    assert not back_to_camera(*_kp(False))


def test_vote_number_outvotes_bad_read():
    assert vote_number([("12", 0.9), ("17", 0.8), ("12", 0.85)], accept_ratio=0.6, accept_mass=1.5)[0] == 12
    assert vote_number([("12", 0.9)])[0] is None                    # one read is not enough
    assert vote_number([("12", 0.9), ("12", 0.9), ("7", 0.9)], roster=[7, 9])[0] is None


def test_majority_colour_beats_mean():
    crop = np.zeros((20, 20, 3), np.uint8)
    crop[:, :12] = (30, 30, 230)        # lit red (BGR), 60 %
    crop[:, 12:] = (10, 10, 10)         # black trim, 40 %
    assert colour_class(crop) == "red"            # per-pixel vote: the shirt colour, not the trim
    # track level: shadowed frames vote "black", the majority over the track is still red
    from collections import Counter
    frames = [crop] * 7 + [np.full((20, 20, 3), 15, np.uint8)] * 3
    assert Counter(colour_class(c) for c in frames).most_common(1)[0][0] == "red"


def test_slot_manager_links_fragments_and_votes():
    from vb.config import IdentityConfig
    rng = np.random.default_rng(0)
    protos = {k: rng.normal(size=8) for k in range(3)}
    tracks = {}
    tid = 1
    for k in range(3):                       # three near-side players, each split into two fragments
        for part, (a, b) in enumerate([(0, 100), (150, 300)]):
            ts = TrackSummary(tid)
            ts.frames = list(range(a, b))
            ts.court = [(1.0 + 3 * k, 4.0)] * (b - a)
            ts.colours = ["blue"] * 10
            e = protos[k] + 0.05 * rng.normal(size=8)
            ts.emb = [e / np.linalg.norm(e)]
            ts.reads = [(str(10 + k), 0.9)] * (1 if part == 0 else 2)
            tracks[tid] = ts
            tid += 1
    sm = SlotManager(IdentityConfig())
    sm.build(tracks, set_start_frame=0)
    assert len(sm.slots) == 3
    assert sorted(s.number for s in sm.slots) == [10, 11, 12]
    assert sm.number_for(4) == sm.number_for(3)               # second fragment joined the first


# ------------------------------------------------------------------ touches -> rallies -> dvw (synthetic)
def _synthetic_rally(H, fps=25.0):
    """Ball path with kinks at hand positions of players on alternating sides."""
    # contacts: (frame, court X, court Y of player feet, hand pixel offset above feet)
    contacts = [(10, 4.5, -1.0), (40, 3.0, 13.0), (55, 4.0, 10.5), (70, 5.0, 10.0), (95, 4.0, 4.0),
                (110, 5.0, 7.5), (125, 4.5, 8.0)]
    pts_img = []
    players = {}
    for k, (f, X, Y) in enumerate(contacts):
        feet = to_image(H, np.float32([[X, Y]]))[0]
        hand = feet + np.float32([0, -120])
        pts_img.append((f, hand))
        box = np.float32([feet[0] - 25, feet[1] - 160, feet[0] + 25, feet[1]])
        kp = np.zeros((17, 2), np.float32)
        kp[9] = hand
        kp[10] = hand + 2
        kp[5], kp[6] = hand + [-10, 40], hand + [10, 40]
        kp[15], kp[16] = feet, feet
        players[f] = PlayerObs(100 + k, box, kp, np.full(17, 0.9, np.float32), np.float32([X, Y]), int(Y >= 9))
    frames, xy = [], []
    for (f0, p0), (f1, p1) in zip(pts_img, pts_img[1:]):
        for f in range(f0, f1):
            a = (f - f0) / (f1 - f0)
            arc = np.float32([0, -200 * a * (1 - a)])      # lob between contacts
            frames.append(f)
            xy.append(p0 + a * (p1 - p0) + arc)
    frames.append(pts_img[-1][0])
    xy.append(pts_img[-1][1])
    for f in range(pts_img[-1][0] + 1, pts_img[-1][0] + 12):   # ball dies after the last contact
        frames.append(f)
        xy.append(pts_img[-1][1] + np.float32([3 * (f - pts_img[-1][0]), 8 * (f - pts_img[-1][0])]))
    pbf = {f: [p] for f, p in players.items()}
    return np.array(frames), np.float32(xy), pbf


def test_synthetic_rally_end_to_end(tmp_path):
    cfg = load_config()
    frames, xy, pbf = _synthetic_rally(H_TRUE)
    touches = detect_touches(frames, xy, pbf, 25.0, cfg.touch)
    got = [t.frame for t in touches]
    # the serve is the first point of the ball track (no kink); the other six are kinks, none at lob apexes
    assert len(got) == 7 and all(abs(a - b) <= 1 for a, b in zip(got, [10, 40, 55, 70, 95, 110, 125])), got
    assert all(t.tid is not None for t in touches)          # every contact attributed to a player
    rallies = segment_rallies(touches, frames, 25.0, cfg.rally.end_gap_s, cfg.rally.behind_line_m)
    assert len(rallies) == 1
    label_skills(rallies, cfg.touch.block_window_s)
    labs = " ".join(t.skill for t in rallies[0].touches)
    assert labs.startswith("S R E A"), labs
    for t, n in zip(rallies[0].touches, [7, 3, 9, 11, 2, 5, 8]):
        t.number = n
    path = write_dvw(tmp_path / "x.dvw", rallies, DvwMatch(home_numbers=[7, 2, 5, 8], away_numbers=[3, 9, 11]))
    text = path.read_text()
    assert "*07SH+" in text and "a03RH+" in text
    if HAS_DV:
        from vb.dvw import check_with_datavolley
        plays, problems = check_with_datavolley(path)
        assert not problems
        skills = [p["skill"] for p in plays if p.get("skill")]
        assert skills[:4] == ["Serve", "Reception", "Set", "Attack"]
        assert plays[0]["player_number"] == 7


def test_dv_coord_roundtrip():
    if not HAS_DV:
        pytest.skip("datavolley not installed")
    from datavolley import dv_index2xy
    for X, Y in [(0, 0), (9, 18), (4.5, 9), (2, 3)]:
        x, y = dv_index2xy(court_to_dv_coord(X, Y))
        assert abs((x - 0.5) * 3 - X) < 0.15 and abs((y - 0.5) * 3 - Y) < 0.15


@pytest.mark.skipif(not HAS_DV, reason="datavolley not installed")
def test_rule_on_bundled_expert_files():
    import datavolley as dv
    from vb.evaluate import score_rule_on_dvw_files
    r = score_rule_on_dvw_files(dv.example_file_list())
    assert r["acc_excl_free_ball"] > 0.96, r


def test_kink_detector_finds_bounce():
    f = np.arange(30)
    xy = np.float32([[10 * i, 100 + 5 * i] for i in range(15)] + [[140 + 10 * i, 170 - 8 * i] for i in range(15)])
    k = kink_candidates(f, xy, lag=3, angle_deg=25, min_speed=1, refractory=6)
    assert len(k) == 1 and abs(f[k[0][0]] - 14) <= 1


def test_no_kink_at_top_of_a_lob():
    f = np.arange(40)
    t = f / 25.0
    xy = np.float32(np.c_[300 + 200 * t, 700 - 600 * t + 0.5 * 900 * t ** 2])   # parabola, apex at t = 0.67 s
    assert kink_candidates(f, xy, lag=3, angle_deg=25, min_speed=2, refractory=6) == []


def test_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("video:\n  source: s3://b/k.mp4\npose:\n  input_wh: [192, 256]\n")
    c = load_config(p)
    assert c.video.source == "s3://b/k.mp4" and c.pose.input_wh == (192, 256)
    with pytest.raises(KeyError):
        load_config(None, nope=1)
