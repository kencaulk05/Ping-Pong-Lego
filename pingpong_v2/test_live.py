"""
test_live.py -- headless tests for the live game's pieces.

    python test_live.py        (or: pytest test_live.py)

What these can and can't show: the cards, the wrist mapping, the swing handoff,
haptics' non-blocking rule, MQTT's hold-and-retry, and a full game run end to
end with fakes. They can't show MediaPipe finding your wrist (the model file
isn't reachable from where these were written) or the real hub's buzz.
"""

import csv
import glob
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

import config as C
import pingpong_live as live
from bots import BotPlayer
from haptics import Haptics, cues_for_events
from hub_input import SwingFeed
from pingpong_core import Game, Inputs, Phase, Swing
from pose_input import WristFilter, WristMapper
from score_publisher import DEFAULT_BROKER, ScorePublisher
from swing_detect import SwingModel, SwingRecognizer
from tag_input import CardDebouncer, TagReader
import make_cards


class Skip(Exception):
    """Raised by a test whose data files aren't here: reported as skipped, not failed."""

DT = 1 / 30


# ---------------------------------------------------------------- AprilTag cards

def _scene_with_card(tag_id, tag_px, angle=20, seed=0):
    """A 640x480 'webcam frame': the printed card, shrunk so the tag is tag_px wide,
    tilted, on a cluttered background, with sensor noise and blur."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(60, 180, (480, 640), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (9, 9), 3)
    page = make_cards.card(tag_id)
    scale = tag_px / (15 / 2.54 * make_cards.DPI)
    small = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += [320 - w / 2, 240 - h / 2]
    mask = cv2.warpAffine(np.full_like(small, 255), M, (640, 480))
    warped = cv2.warpAffine(small, M, (640, 480))
    frame[mask > 0] = warped[mask > 0]
    frame = np.clip(frame + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0.8), cv2.COLOR_GRAY2BGR)


def test_every_printed_card_is_read_with_the_right_id():
    reader = TagReader()
    for tag_id in (0, 1, 2, 3):
        for px in (45, 80):            # ~45 px is a 15 cm tag a few meters from a 640x480 webcam
            assert reader.detect(_scene_with_card(tag_id, px, angle=15 + tag_id * 7)) == {tag_id}, (tag_id, px)


def test_no_card_means_no_tag():
    frame = cv2.cvtColor(np.random.default_rng(3).integers(0, 255, (480, 640), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    assert TagReader().detect(frame) == set()


def test_a_card_must_be_steady_for_three_frames():
    d = CardDebouncer()
    assert d.update({2}) == (None, False)
    assert d.update({2}) == (None, False)
    assert d.update({2}) == (2, False)
    assert d.update(set()) == (None, False)
    assert d.update({2}) == (None, False), "a gap restarts the count"


def test_start_fires_once_per_showing():
    d = CardDebouncer()
    starts = [d.update({0})[1] for _ in range(10)]
    assert starts.count(True) == 1 and starts[2]
    d.update(set())
    assert [d.update({0})[1] for _ in range(3)] == [False, False, True], "lowered and shown again: starts again"


def test_the_pdf_has_four_cards():
    from PIL import Image
    import os, tempfile
    path = os.path.join(tempfile.mkdtemp(), "cards.pdf")
    imgs = [Image.fromarray(make_cards.card(i)).convert("RGB") for i in range(4)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], resolution=make_cards.DPI)
    assert os.path.getsize(path) > 10000


# ------------------------------------------------------------------- the wrist

def test_calibrated_spot_is_the_middle_of_the_hitting_line():
    m = WristMapper(reach=0.6)
    m.calibrate((0.62, 0.48))
    x, y = m.to_court((0.62, 0.48))
    assert abs(x - 0.5) < 1e-9 and abs(y - C.RING_Y) < 1e-9


def test_reach_spans_the_court():
    m = WristMapper(cx=0.5, cy=0.5, reach=0.6)
    assert abs(m.to_court((0.5 + 0.3, 0.5))[0] - 1.0) < 1e-9
    assert abs(m.to_court((0.5 - 0.3, 0.5))[0] - 0.0) < 1e-9
    assert m.to_court((0.95, 0.5))[0] == 1.0, "clamped at the edge"


def test_moving_your_hand_right_moves_the_paddle_right():
    m = WristMapper()
    assert m.to_court((0.6, 0.55))[0] > m.to_court((0.4, 0.55))[0]
    assert m.to_court((0.5, 0.65))[1] > m.to_court((0.5, 0.45))[1], "lower hand = lower on screen"


def test_the_ring_is_round_in_real_space():
    """The same real distance up or sideways should count the same toward 'in the ring'."""
    m = WristMapper(cx=0.5, cy=0.5, reach=0.6, cam_aspect=4 / 3)
    d = 0.02                                        # 0.02 of the camera WIDTH, in each direction
    side = m.to_court((0.5 + d, 0.5))
    up = m.to_court((0.5, 0.5 + d * 4 / 3))         # the same distance, in camera-height units
    g = Game()
    g.ring_x = 0.5
    assert abs(g.ring_distance(side) - g.ring_distance(up)) < 1e-9


def test_camera_and_court_mappings_are_inverses():
    m = WristMapper(cx=0.4, cy=0.6, reach=0.5, cam_aspect=16 / 9)
    for p in [(0.3, 0.7), (0.5, C.RING_Y), (0.8, 0.6)]:
        back = m.to_court(m.to_camera(p))
        assert abs(back[0] - p[0]) < 1e-9 and abs(back[1] - p[1]) < 1e-9


def test_a_tracking_blink_is_ridden_out_but_a_real_loss_is_reported():
    f = WristFilter(hold_s=0.25)
    assert f.update((0.5, 0.5), 0.0) == (0.5, 0.5)
    assert f.update(None, 0.1) == (0.5, 0.5)
    assert f.update(None, 0.4) is None


def test_smoothing_follows_the_hand():
    f = WristFilter(smooth=0.6)
    f.update((0.0, 0.0), 0.0)
    x = [f.update((1.0, 0.0), t)[0] for t in (0.03, 0.06, 0.09, 0.12)]
    assert x == sorted(x) and x[-1] > 0.95


# -------------------------------------------------------------- swings from the hub

def _stream(path):
    with open(path) as f:
        r = csv.DictReader(f)
        rows = [(float(d["t"]), tuple(float(d[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll")))
                for d in r]
    return rows


def test_your_recorded_swings_come_through_the_live_feed():
    """Plays a real run from your hub (waiting style, 30 cued swings) through the live
    feed. Plumbing check only: the shipped model trained on this run, so this isn't
    an accuracy number (swing_check.py on a fresh run is)."""
    path = sorted(glob.glob("guided-*-034425-stream.csv"))[0]
    model = SwingModel.load("swing_model.json")
    feed = SwingFeed(SwingRecognizer(model))
    got = []
    for t, v in _stream(path):
        feed.process(t, v)
        for rec, sw in feed.drain(t + 0.05):        # the game picks it up one frame later
            got.append((rec, sw))
    kinds = [r.kind for r, _ in got]
    assert 27 <= len(got) <= 33, len(got)
    assert min(kinds.count(k) for k in ("forehand", "backhand", "flat")) >= 8, kinds
    for rec, sw in got:
        assert abs(sw.age - (0.05 + rec.age)) < 1e-6, "age is measured at handoff, not at recognition"
        assert 0.15 < sw.age < 0.45


def test_a_late_report_still_hits_if_the_swing_was_on_time():
    """The swing's peak is right at the ball's arrival, but the report reaches the
    game 0.31 s later, after the +/-0.20 s window has closed. Measuring the age at
    handoff, and waiting LIVE_SWING_REPORT_DELAY, still makes it a hit."""
    fake = SimpleNamespace(pending=[])

    class FakeFeed:
        def drain(self, now):
            out = [(r, Swing(r.kind, r.power, now - r.t_peak)) for r in fake.pending if r.ready <= now]
            fake.pending = [r for r in fake.pending if r.ready > now]
            return out

    g = Game(level=1, rng=random.Random(1))
    s = live.LiveSession(g, ScorePublisher("T", enabled=False), feed=FakeFeed())
    t = 0.0
    s.step(t, DT, (0.5, C.RING_Y), start_card=True)
    while g.phase != Phase.INCOMING:
        t += DT
        s.step(t, DT, (0.5, C.RING_Y))
    wrist = (g.ring_x, C.RING_Y)
    arrival = t + g.ball.T
    fake.pending.append(SimpleNamespace(kind="forehand", power=0.6, confidence=0.9, peak=700.0,
                                        t_peak=arrival, ready=arrival + 0.31))
    hit = False
    while g.phase == Phase.INCOMING:
        t += DT
        hit |= any(e["type"] == "hit" for e in s.step(t, DT, wrist))
    assert hit
    assert abs(s.timing[-1]) < DT + 1e-6, "timing readout should say on time"


def test_an_early_swing_is_reported_as_early():
    g = Game(level=1, rng=random.Random(2))
    s = live.LiveSession(g, ScorePublisher("T", enabled=False))
    t = 0.0
    s.step(t, DT, (0.5, C.RING_Y), start_card=True)
    while g.phase != Phase.INCOMING:
        t += DT
        s.step(t, DT, (0.5, C.RING_Y))
    T = g.ball.T
    while g.phase_t < T - 0.5:
        t += DT
        s.step(t, DT, (g.ring_x, C.RING_Y))
    s.step(t + DT, DT, (g.ring_x, C.RING_Y), key_swings=[Swing("flat", 0.5)])
    assert s.timing[-1] < -0.4 and "early" in s.timing_text()


# ------------------------------------------------------------------- haptics

class SlowHub:
    """Records every call; takes 0.3 s per call, like a hub that waits."""

    def __init__(self, delay=0.3):
        self.calls, self.delay = [], delay

    def __getattr__(self, name):
        def call(*a, **k):
            time.sleep(self.delay)
            self.calls.append((name, a, k))
        return call


def test_every_hub_command_is_non_blocking():
    hub = SlowHub(delay=0)
    h = Haptics(hub, le=live.fake_le())
    for name in ("incoming", "window", "hit", "winner", "record", "miss", "count", "serve", "idle"):
        h.play_now(name, 0.7)
    assert hub.calls and all(k.get("blocking") is False for _, _, k in hub.calls)


def test_queueing_a_cue_never_waits_on_the_hub():
    h = Haptics(SlowHub(delay=0.3), le=live.fake_le()).start()
    t0 = time.perf_counter()
    for _ in range(20):
        h.cue("hit", 1.0)
    took = time.perf_counter() - t0
    h.stop()
    assert took < 0.05, f"cue() took {took:.3f} s"


def test_a_harder_hit_buzzes_longer():
    h = Haptics(SlowHub(0), le=live.fake_le())
    ms = lambda p: [a[0] for f, a, k in h.commands("hit", p) if f == "motor_run_for_time"][0]
    assert ms(1.0) > ms(0.0) and ms(1.0) <= 250, "kept short so it can't look like a swing"


def test_game_events_map_to_cues():
    ev = [{"type": "hit", "power": 0.9}, {"type": "record", "best": 3}, {"type": "miss"}, {"type": "serve"}]
    assert [c for c, _ in cues_for_events(ev)] == ["hit", "record", "miss", "serve", "incoming"]


def test_the_paddle_turns_green_once_per_ball_just_before_the_window():
    class Spy(Haptics):
        def __init__(self):
            super().__init__(enabled=False)
            self.log = []

        def cue(self, name, power=0.5):
            self.log.append((name, game.phase_t, game.ball_serial))

    game = Game(level=1, rng=random.Random(3))
    spy = Spy()
    s = live.LiveSession(game, ScorePublisher("T", enabled=False), haptics=spy)
    t = 0.0
    s.step(t, DT, (0.5, C.RING_Y), start_card=True)
    for _ in range(int(20 / DT)):
        t += DT
        s.step(t, DT, (0.5, C.RING_Y))
    windows = [(pt, serial) for name, pt, serial in spy.log if name == "window"]
    serials = [serial for _, serial in windows]
    assert len(serials) == len(set(serials)) >= 3
    T = C.LEVELS[1].incoming_time
    for pt, _ in windows:
        w = C.LEVELS[1].hit_window
        assert T - w - C.HAPTIC_LEAD <= pt < T - w - C.HAPTIC_LEAD + DT + 1e-9


def test_check_haptics_runs_against_the_fake_hub():
    args = live.parse_args(["--fake", "--check-haptics"])
    rows = live.check_haptics(args, repeats=1, gap=0.3, listen=0.5)
    assert [r[0] for r in rows] == ["hit", "winner", "miss"]


# ----------------------------------------------------------------------- MQTT

class FlakyClient:
    def __init__(self):
        self.up, self.sent = False, []

    def is_connected(self):
        return self.up

    def publish(self, topic, payload, retain=False):
        self.sent.append((topic, payload, retain))


def test_a_score_made_while_the_broker_is_down_is_sent_when_it_returns():
    c = FlakyClient()
    pub = ScorePublisher("Alex", client=c)
    pub.publish_best(4)
    pub.tick()
    assert c.sent == [] and pub.status == "waiting for broker"
    pub.publish_best(5)
    c.up = True
    pub.tick()
    assert c.sent == [("ME193/Rogers/Alex", "5.0", False)], "only the latest record, once"
    pub.tick()
    assert len(c.sent) == 1 and pub.status == "ok"


def test_the_class_broker_is_the_default():
    assert DEFAULT_BROKER == "test.mosquitto.org"
    a = live.parse_args([])
    assert a.broker == "test.mosquitto.org" and a.hand == "right" and a.opponent == "q"


# ------------------------------------------------------------------ handedness

def test_right_handed_forehand_goes_left():
    saved = dict(C.SHOT_OFFSET)
    try:
        live.apply_handedness("right")
        assert C.SHOT_OFFSET["forehand"] < 0 < C.SHOT_OFFSET["backhand"]
        live.apply_handedness("left")
        assert C.SHOT_OFFSET["forehand"] > 0 > C.SHOT_OFFSET["backhand"]
        live.apply_handedness("right", flip=True)
        assert C.SHOT_OFFSET["forehand"] > 0
    finally:
        C.SHOT_OFFSET = saved


# ------------------------------------------------------------ the whole thing

def test_a_full_session_runs_end_to_end():
    c = FlakyClient()
    c.up = True
    pub = ScorePublisher("Alex", client=c)
    rng = random.Random(7)
    game = Game(level=2, rng=rng)
    hub = SlowHub(0)
    hap = Haptics(hub, le=live.fake_le())
    hap.cue = lambda name, power=0.5: hap.play_now(name, power)      # synchronous, to count
    s = live.LiveSession(game, pub, haptics=hap)
    bot = BotPlayer(skill=0.8, rng=rng)
    t = 0.0
    for _ in range(int(90 / DT)):
        inp = bot.act(game, DT)
        s.step(t, DT, inp.wrist, inp.level_card, inp.start_card, inp.swings)
        t += DT
    assert game.hits > 10 and game.misses > 0
    assert [float(p) for _, p, _ in c.sent] == list(range(1, game.best + 1)), "every new record, in order"
    names = [n for n, _ in hap.sent]
    for cue in ("hit", "miss", "window", "record", "serve"):
        assert cue in names, cue


def test_the_live_screen_draws():
    view = live.LiveView()
    g = Game(level=1, rng=random.Random(0))
    s = live.LiveSession(g, ScorePublisher("T", enabled=False))
    t = 0.0
    s.step(t, DT, (0.5, C.RING_Y), start_card=True)
    while g.phase != Phase.INCOMING:
        t += DT
        s.step(t, DT, (0.5, C.RING_Y))
    img = view.render(g, (0.5, C.RING_Y), t)
    frame = np.full((480, 640, 3), 90, np.uint8)
    pose = SimpleNamespace(idx=16, landmarks=[(0.5, 0.5, 0.9)] * 33)
    tags = SimpleNamespace(last=[(2, np.array([[100, 100], [160, 100], [160, 160], [100, 160]], float))])
    view.pip(img, frame, pose, tags, WristMapper(), g, ["30 fps", "MQTT ok", "hub 32/s"])
    assert img.shape == (live.H, live.W, 3)
    assert img[live.PIP_Y + 5, live.PIP_X + 5].tolist() != [0, 0, 0]


def test_every_new_tool_starts():
    for tool in ("pingpong_live.py", "make_cards.py", "mqtt_check.py", "analyze_live.py", "eval_steering.py"):
        r = subprocess.run([sys.executable, tool, "--help"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (tool, r.stderr[-500:])


def test_pose_tracker_is_built_the_way_hw1_built_it():
    """MediaPipe's Tasks API pieces the tracker uses all exist in this version.
    (Running it needs pose_landmarker_lite.task, which it downloads on first run.)"""
    try:
        import mediapipe as mp
    except ImportError:
        print("  (skipped: mediapipe not installed)")
        return
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    assert hasattr(PoseLandmarker, "create_from_options") and hasattr(RunningMode, "VIDEO")
    assert hasattr(mp, "Image") and hasattr(mp.ImageFormat, "SRGB")


def test_the_main_loop_runs_with_stand_in_camera_keys_and_hub():
    """Runs pingpong_live.run() itself: a fake webcam showing the START card then the
    LEVEL 2 card, a fake pose tracker, the fake hub, keyboard presses for C, ], a
    keyboard swing and Q. Catches crashes in the code only the real laptop runs."""
    import os, tempfile
    import pose_input
    frames = {"n": 0}
    level_card = _scene_with_card(2, 80)
    start_card = _scene_with_card(0, 80)
    blank = _scene_with_card(1, 1)

    class Cam:
        def __init__(self, *a): pass
        def isOpened(self): return True
        def set(self, *a): return True
        def release(self): pass
        def read(self):
            frames["n"] += 1
            n = frames["n"]
            time.sleep(0.02)                     # about real time, so the countdown finishes
            return True, (level_card if 5 <= n < 12 else start_card if 15 <= n < 22 else blank)

    class Pose:
        def __init__(self, **k):
            self.idx, self.landmarks = 16, [(0.6, 0.5, 0.9)] * 33
        def process(self, frame, now): return (0.6, 0.5)
        def close(self): pass

    keys = {3: ord("c"), 4: ord("]"), 40: ord("f")}
    seen = []

    def waitKey(ms):
        return keys.get(frames["n"], ord("q") if frames["n"] >= 260 else 255)

    saved = (cv2.VideoCapture, pose_input.PoseTracker, cv2.waitKey, cv2.imshow, cv2.namedWindow,
             cv2.destroyAllWindows, live.LiveSession.step)
    orig_step = live.LiveSession.step

    def step(self, now, dt, wrist, level_card=None, start_card=False, key_swings=()):
        ev = orig_step(self, now, dt, wrist, level_card, start_card, key_swings)
        seen.extend(e["type"] for e in ev)
        return ev

    cwd = os.getcwd()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    tmp = tempfile.mkdtemp()
    import shutil
    shutil.copy("swing_model.json", tmp)
    os.chdir(tmp)
    try:
        cv2.VideoCapture, pose_input.PoseTracker = Cam, Pose
        cv2.waitKey, cv2.imshow = waitKey, lambda *a: None
        cv2.namedWindow, cv2.destroyAllWindows = (lambda *a, **k: None), (lambda: None)
        live.LiveSession.step = step
        live.run(live.parse_args(["--name", "Tester", "--fake", "--keys", "--no-mqtt", "--no-sound", "--opponent", "rule"]))
        assert "level" in seen and "start" in seen, seen[:20]
        assert "serve" in seen, "the countdown finished and the first ball was served"
        assert os.path.exists(live.SETTINGS), "C saved the calibration"
        assert glob.glob("live-*-events.csv")
    finally:
        (cv2.VideoCapture, pose_input.PoseTracker, cv2.waitKey, cv2.imshow, cv2.namedWindow,
         cv2.destroyAllWindows, live.LiveSession.step) = saved
        os.chdir(cwd)


# ------------------------------------------------------------- the accept rule

def test_the_swing_rule_beats_the_kind_rule_on_the_beam_run():
    """Your 06:14 run (beam on) was never trained on by swing_model_v1.json. The old
    rule reproduces your live 25/30; the swing rule gets 29/30, still 0 false alarms."""
    from rescore import rescore
    model = SwingModel.load("swing_model_v1.json")
    old, _ = rescore("guided-20261012-061454", model, rule="kind")
    new, _ = rescore("guided-20261012-061454", model, rule="swing")
    assert old["right"] == 25 and old["false_alarms"] == 0
    assert new["right"] >= 29 and new["false_alarms"] == 0
    lost = sum(new["table"][k]["ignored"] + new["table"][k]["nothing"] for k in new["table"])
    assert lost == 0


def test_the_swing_rule_accepts_a_swing_it_cannot_name_confidently():
    """A swing split between forehand and flat is still clearly a swing."""
    rec = SwingRecognizer(SwingModel.load("swing_model_v1.json"))
    rec.model = SimpleNamespace(classes=["backhand", "flat", "forehand", "other"],
                                predict=lambda f: ("forehand", 0.48, np.array([0.02, 0.45, 0.48, 0.05])),
                                power=lambda peak: 0.5)
    ev = SimpleNamespace(peak=700.0, features=None, age=0.26, t_peak=1.0)
    rec.detector = SimpleNamespace(n_fresh=0, push=lambda t, v: [ev])
    got = rec.push(1.0, (0,) * 6)
    assert len(got) == 1 and got[0].kind == "forehand" and abs(got[0].confidence - 0.95) < 1e-9
    rec.rule, rec.min_confidence = "kind", 0.6
    assert rec.push(1.0, (0,) * 6) == [], "the old rule threw this swing away"


def test_the_swing_rule_still_rejects_what_looks_like_other():
    rec = SwingRecognizer(SwingModel.load("swing_model_v1.json"))
    rec.model = SimpleNamespace(classes=["backhand", "flat", "forehand", "other"],
                                predict=lambda f: ("other", 0.85, np.array([0.05, 0.05, 0.05, 0.85])),
                                power=lambda peak: 0.5)
    ev = SimpleNamespace(peak=373.0, features=None, age=0.26, t_peak=1.0)
    rec.detector = SimpleNamespace(n_fresh=0, push=lambda t, v: [ev])
    assert rec.push(1.0, (0,) * 6) == [] and rec.rejected[-1][1] == "other"


def test_the_backup_model_is_the_untouched_original():
    import hashlib
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert h("swing_model_v1.json") == "816e3cc0921dfb2c55705e24281a651b23a23f60c5c4c3aa21ffbf78ca8ff99c"


# ------------------------------------------------- tuning from the first live game

def _to_incoming(level, seed=5):
    g = Game(level=level, rng=random.Random(seed))
    g.update(DT, Inputs(wrist=(0.5, C.RING_Y), start_card=True))
    while g.phase != Phase.INCOMING:
        g.update(DT, Inputs(wrist=(0.5, C.RING_Y)))
    return g


def _swing_at(g, offset, wrist_fn):
    """Swing `offset` seconds from the ball's arrival; returns the events that decided it."""
    T, out = g.ball.T, []
    swung = False
    while g.phase == Phase.INCOMING:
        sw = []
        if not swung and g.phase_t + DT >= T + offset:
            sw, swung = [Swing("forehand", 0.5)], True
        out += g.update(DT, Inputs(wrist=wrist_fn(g), swings=sw))
    return [e["type"] for e in out if e["type"] in ("hit", "miss")]


def test_the_level_card_sets_the_timing_window():
    """Your early misses were 0.21-0.35 s early. Easy forgives 0.27 s early; medium doesn't."""
    on = lambda g: (g.ring_x, C.RING_Y)
    assert _swing_at(_to_incoming(1), -0.27, on) == ["hit"]
    assert _swing_at(_to_incoming(2), -0.27, on) == ["miss"]
    assert _swing_at(_to_incoming(3), -0.18, on) == ["miss"]
    assert _swing_at(_to_incoming(3), +0.12, on) == ["hit"], "your hits ranged -0.12..+0.14 s"


def test_lining_up_then_swinging_away_still_counts():
    """Wrist in the ring until 0.1 s before the swing's peak, then carried out by the swing."""
    def wrist(g):
        out = g.phase_t > g.ball.T - 0.1
        return (g.ring_x + (0.3 if out else 0.0), C.RING_Y)
    assert _swing_at(_to_incoming(2), 0.0, wrist) == ["hit"]


def test_never_in_the_ring_is_still_a_miss_and_says_how_far():
    g = _to_incoming(2)
    far = lambda g: (g.ring_x + 0.3 if g.ring_x < 0.5 else g.ring_x - 0.3, C.RING_Y)
    T, ev, swung = g.ball.T, [], False
    while g.phase == Phase.INCOMING:
        sw = [] if swung or g.phase_t + DT < T else [Swing("flat", 0.5)]
        swung = swung or bool(sw)
        ev += g.update(DT, Inputs(wrist=far(g), swings=sw))
    miss = [e for e in ev if e["type"] == "miss"][0]
    assert miss["reason"] == "out of position" and miss["dist"] > 1.0


def test_power_follows_how_hard_you_actually_swing():
    s = live.LiveSession(Game(), ScorePublisher("T", enabled=False))
    assert s.session_power(1500) == 1.0, "before 5 swings: the recorded scale"
    for p in (1300, 1500, 1600, 1700, 1800, 1750):
        s.peaks.append(p)
    assert 0.6 < s.session_power(1300) < 0.8 and s.session_power(1800) > 0.95
    s2 = live.LiveSession(Game(), ScorePublisher("T", enabled=False))
    for p in (400, 450, 500, 420, 480):
        s2.peaks.append(p)
    assert s2.session_power(500) < 0.25, "never below the recorded scale: gentle swings stay gentle"


# -------------------------------- fixes from reviewing the first live games (steering, cards, record, analysis)

def test_a_start_card_left_in_view_does_not_restart_the_game_after_an_abort():
    # The scenario: show START (game begins), keep the card in view through the whole game, the game aborts back
    # to idle (wrist lost too long). A debouncer that only watched while idle would see a "new" card and restart.
    d = CardDebouncer()
    assert any(s for _, s in (d.update({0}, idle=True) for _ in range(4))), "shown while idle: starts"
    assert not any(s for _, s in (d.update({0}, idle=False) for _ in range(40))), "the game is running: ignored"
    assert not any(s for _, s in (d.update({0}, idle=True) for _ in range(12))), "back to idle, card never left: no restart"
    for _ in range(2):
        d.update(set(), idle=True)                                     # lower the card
    assert any(s for _, s in (d.update({0}, idle=True) for _ in range(4))), "shown again: starts"


def test_flashing_a_card_mid_rally_does_nothing_and_does_not_start_the_next_game_by_surprise():
    d = CardDebouncer()
    out = [d.update({0, 2}, idle=False) for _ in range(10)]
    assert all(o == (None, False) for o in out), "cards are ignored while a game is on"
    assert not any(s for _, s in (d.update({0}, idle=True) for _ in range(10))), \
        "a start card that appeared mid-rally and never left is already 'shown'"


def test_saving_a_calibration_does_not_erase_the_saved_best_streak(tmp_path=None):
    import tempfile
    old, cwd = live.SETTINGS, os.getcwd()
    d = tempfile.mkdtemp()
    os.chdir(d)
    try:
        live.save_settings("Kendrick", {"best": 18})
        live.save_settings("Kendrick", {"cx": 0.4, "cy": 0.5, "reach": 0.6})      # a calibration, saved later
        live.save_settings("Other", {"best": 3})
        s = live.load_settings("Kendrick")
        assert s == {"best": 18, "cx": 0.4, "cy": 0.5, "reach": 0.6}, s
        assert live.load_settings("Other") == {"best": 3}, "another name is untouched"
    finally:
        os.chdir(cwd)
        live.SETTINGS = old


def test_restore_best_carries_the_record_over_unless_a_fresh_one_is_asked_for():
    import tempfile
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        live.save_settings("Kendrick", {"best": 18})
        g = Game(level=1, rng=random.Random(0))
        assert live.restore_best(g, "Kendrick") == 18 and g.best == 18
        g2 = Game(level=1, rng=random.Random(0))
        assert live.restore_best(g2, "Nobody") == 0
        g3 = Game(level=1, rng=random.Random(0))
        assert live.restore_best(g3, "Kendrick", fresh=True) == 0 and g3.best == 0
        assert live.load_settings("Kendrick")["best"] == 0, "a fresh record is remembered as 0"
    finally:
        os.chdir(cwd)


def _run_main_loop(extra_args, keys, quit_at=40):
    """live.run() itself, with a stand-in camera, pose and window, and a recording publisher."""
    import pose_input

    class Cam:
        def __init__(self, *a): self.n = 0
        def isOpened(self): return True
        def set(self, *a): return True
        def release(self): pass
        def read(self):
            frames["n"] += 1
            time.sleep(0.005)
            return True, np.full((480, 640, 3), 90, np.uint8)

    class Pose:
        def __init__(self, **k): self.idx, self.landmarks = 16, [(0.6, 0.5, 0.9)] * 33
        def process(self, frame, now): return (0.6, 0.5)
        def close(self): pass

    class FakePub:
        published = []
        status = "ok"
        def __init__(self, *a, **k): pass
        def connect(self): pass
        def publish_best(self, b): FakePub.published.append(float(b))
        def tick(self): pass
        def close(self): pass

    frames = {"n": 0}
    waitKey = lambda ms: keys.get(frames["n"], ord("q") if frames["n"] >= quit_at else 255)
    saved = (cv2.VideoCapture, pose_input.PoseTracker, cv2.waitKey, cv2.imshow, cv2.namedWindow, cv2.destroyAllWindows,
             live.ScorePublisher)
    try:
        cv2.VideoCapture, pose_input.PoseTracker, live.ScorePublisher = Cam, Pose, FakePub
        cv2.waitKey, cv2.imshow = waitKey, lambda *a: None
        cv2.namedWindow, cv2.destroyAllWindows = (lambda *a, **k: None), (lambda: None)
        live.run(live.parse_args(["--name", "Tester", "--no-hub", "--no-mqtt", "--no-sound", "--no-log",
                                  "--opponent", "rule"] + extra_args))
    finally:
        (cv2.VideoCapture, pose_input.PoseTracker, cv2.waitKey, cv2.imshow, cv2.namedWindow, cv2.destroyAllWindows,
         live.ScorePublisher) = saved
    return FakePub.published


def test_launching_the_game_again_does_not_publish_zero_over_a_saved_best():
    import tempfile
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        live.save_settings("Tester", {"best": 9})
        published = _run_main_loop([], {3: ord("c")})               # also calibrates (saves cx, cy, reach)
        assert published[0] == 9.0, f"the first publish must be the saved best, not 0.0: {published}"
        s = live.load_settings("Tester")
        assert s.get("best") == 9 and "cx" in s, f"calibration must not erase the best: {s}"
        published = _run_main_loop(["--fresh-record"], {})
        assert published[0] == 0.0 and live.load_settings("Tester")["best"] == 0
    finally:
        os.chdir(cwd)


def test_the_main_loop_does_not_restart_the_game_after_an_abort_while_the_start_card_never_leaves():
    # End to end through live.run(): the START card stays in the picture the whole time, the wrist is lost long
    # enough to abort the game back to idle, then comes back. Exactly one game may start. (The original bug was in
    # the loop, which fed the card reader nothing while a game ran, so it saw a "new" card at the abort.)
    import tempfile
    import pose_input
    frames = {"n": 0}
    card = _scene_with_card(0, 80)

    class Cam:
        def __init__(self, *a): pass
        def isOpened(self): return True
        def set(self, *a): return True
        def release(self): pass
        def read(self):
            frames["n"] += 1
            time.sleep(0.02)
            return True, card

    class Pose:
        def __init__(self, **k): self.idx, self.landmarks = 16, [(0.6, 0.5, 0.9)] * 33
        def process(self, frame, now): return None if 20 <= frames["n"] < 60 else (0.6, 0.5)
        def close(self): pass

    starts, aborts = [], []
    orig_step = live.LiveSession.step

    def step(self, now, dt, wrist, level_card=None, start_card=False, key_swings=()):
        ev = orig_step(self, now, dt, wrist, level_card, start_card, key_swings)
        starts.extend(e for e in ev if e["type"] == "start")
        aborts.extend(e for e in ev if e["type"] == "aborted")
        return ev

    saved = (cv2.VideoCapture, pose_input.PoseTracker, cv2.waitKey, cv2.imshow, cv2.namedWindow, cv2.destroyAllWindows,
             live.LiveSession.step, C.POSE_LOST_TIMEOUT)
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        cv2.VideoCapture, pose_input.PoseTracker, live.LiveSession.step = Cam, Pose, step
        cv2.waitKey = lambda ms: ord("q") if frames["n"] >= 130 else 255
        cv2.imshow, cv2.namedWindow, cv2.destroyAllWindows = (lambda *a: None), (lambda *a, **k: None), (lambda: None)
        C.POSE_LOST_TIMEOUT = 0.3
        live.run(live.parse_args(["--name", "Tester", "--no-hub", "--no-mqtt", "--no-sound", "--no-log", "--opponent", "rule"]))
    finally:
        (cv2.VideoCapture, pose_input.PoseTracker, cv2.waitKey, cv2.imshow, cv2.namedWindow, cv2.destroyAllWindows,
         live.LiveSession.step, C.POSE_LOST_TIMEOUT) = saved
        os.chdir(cwd)
    assert len(aborts) >= 1, "the wrist was lost long enough to abort the game, or this test proves nothing"
    assert len(starts) == 1, f"the card never left the picture, so the game must not restart: {len(starts)} starts"


def test_the_live_game_steers_by_rotation_by_default_and_can_be_switched_back():
    assert live.parse_args([]).steer == "rotation" and live.parse_args([]).fresh_record is False
    assert live.parse_args(["--steer", "model"]).steer == "model"


def test_the_screen_and_the_log_say_when_steering_overrode_the_classifier():
    from swing_detect import Recognized
    rec = Recognized("forehand", 0.6, 0.25, 1.0, 1500.0, 0.0, None, "flat")
    sw = rec.to_swing()

    class Feed:
        def drain(self, now): return [(rec, sw)]
        def latest_gyro(self): return 0.0

    g = Game(level=1, rng=random.Random(0))
    s = live.LiveSession(g, ScorePublisher("T", enabled=False), feed=Feed())
    s.step(1.0, 1 / 30, (0.5, 0.6))
    assert "forehand (model: flat)" in s.last_swing, s.last_swing
    seen = [e for _, e in s.log if e["type"] == "swing_seen"][0]
    assert seen["kind"] == "forehand" and seen["model_kind"] == "flat"


def _ev(t, type_, **kw):
    return dict(kw, type=type_, t=t)


def test_the_live_analysis_counts_hits_by_level_and_ring_misses_in_the_levels_own_units():
    import analyze_live as al
    events = [_ev(0, "start", level=1), _ev(1, "hit"), _ev(2, "hit"), _ev(3, "hit"),
              _ev(4, "miss", reason="out of position", dist=1.3),                 # old log: base radii
              _ev(5, "miss", reason="out of position", dist=1.6),
              _ev(6, "miss", reason="no swing"),
              _ev(7, "level", level=3),
              _ev(8, "miss", reason="out of position", dist=1.1, ring_scale=1.2)]  # new log: 1.1 of a x1.2 ring = 1.32 base
    t = al.by_level([events])
    assert (t[1]["hit"], t[1]["miss"], t[1]["oop"], t[3]["miss"]) == (3, 3, 2, 1)
    assert [round(d, 2) for _, d in al.miss_distances([events])] == [1.3, 1.6, 1.32]
    extra, before, after = al.ring_counterfactual([events], {1: 1.5, 3: 1.2})
    assert extra == 1, "only the 1.3 miss on level 1 fits inside x1.5; 1.32 is outside x1.2"
    assert abs(before - 3 / 7) < 1e-9 and abs(after - 4 / 7) < 1e-9


def test_the_live_analysis_reproduces_the_numbers_from_the_real_sessions():
    if not glob.glob("live-*-events.csv"):
        raise Skip("no live sessions here")
    out = subprocess.run([sys.executable, "analyze_live.py", "--model", "swing_model.json"], capture_output=True,
                         text=True, timeout=200)
    assert out.returncode == 0, out.stderr[-400:]
    assert "L1: 55 hits / 97 (57%)" in out.stdout and "L3: 160 hits / 275 (58%)" in out.stdout, out.stdout[:900]
    assert "477 swings: median +0.01 s" in out.stdout
    assert "would add  39 hits" in out.stdout


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as why:
            skipped += 1
            print(f"SKIP  {name}: {why}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
