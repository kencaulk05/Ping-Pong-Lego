"""
pingpong_live.py -- the real game: webcam pose + AprilTag cards + the hub in your hand.

    python pingpong_live.py --name YourName

One program owns everything, because the hub allows one Bluetooth connection at a
time (HW3's lesson):

    webcam  -> MediaPipe Pose       -> your right wrist  -> is the paddle in the ring?
            -> AprilTag (36h11)     -> START / LEVEL 1-3 cards
    hub     -> IMU, read on its own thread -> swing detector + classifier
                                    -> are you swinging, and forehand/backhand/flat?
    game    -> pingpong_core.Game (the same rules the simulator and the 100+ tests use)
    out     -> score to ME193/Rogers/<name> on test.mosquitto.org
            -> haptics on the hub (buzz, light, beep), tunes on the laptop

How to play
  1. Hold the hub in your RIGHT hand, blue side against your palm (the grip the
     swing model learned).
  2. Stand where the camera sees you from the waist up. Hold the hub where a
     comfortable hit is and press C. That spot becomes the middle of the hitting line.
  3. Show a LEVEL card (1, 2 or 3), then the START card. 3-2-1.
  4. Move your wrist into the ring, and swing as the ball lands (the hub turns
     green when it's time).

Keys
    C          calibrate: the wrist's current spot = middle of the hitting line
    [ / ]      less / more side-to-side reach needed to cover the court
    1 2 3 / S  level and start, in case a card won't read
    R          reset the record          Q / Esc  quit
    F B Space  keyboard swings (only with --no-hub or --keys)

Fallbacks, so every piece can be tried on its own:
    --no-camera   the mouse is your wrist and the keys are the cards
    --no-hub      the keyboard swings (F / B / Space, Shift = harder)
    --fake        a synthetic hub (plumbing only; its swings don't look like yours)
    --check-haptics   does the buzz ever register as a swing? (hub only, no camera)
"""

import argparse
import csv
import json
import random
import time
from collections import deque
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np

import config as C
from haptics import Haptics, cues_for_events
from pingpong_core import Game, Inputs, Phase, Swing
from pingpong_sim import (AMBER, GREEN, H, LINE, MUTED, PURPLE, RED, TEAL, W, View,
                          log_events, make_policy, put)
from score_publisher import DEFAULT_BROKER, ScorePublisher
from sounds import Sounds, tune_for_events
from stage import SAME, blend_rect

WIN = "Virtual ping-pong (live)"
SETTINGS = "live_settings.json"
PIP_W, PIP_H = 208, 156
PIP_X, PIP_Y = W - PIP_W - 8, 52
POWER_ADAPT_AFTER = 5
DEFAULT_BODY_REACH = 0.8        # how much of the camera width your body steps across to cover the court (]/[ change it)


# ------------------------------------------------------------------ the session

CAL_OK = {True: "Calibrated: the middle of the court is where you stand now",
          False: "Calibrated: the middle of the court is where your hand is now"}
CAL_FAIL = {True: "Can't see you: step back so your shoulders are in the picture, then press C",
            False: "Can't see your wrist: hold the hub up where the camera sees it, then press C"}


def pose_status(body_mode, body_cam, wrist_cam):
    """The seen-or-not line shown with the other status lines, so you can tell at a glance what the camera has."""
    if body_mode:
        return f"you: {'ok' if body_cam is not None else 'NOT SEEN'}   hand: {'ok' if wrist_cam is not None else 'not seen'}"
    return f"wrist: {'ok' if wrist_cam is not None else 'NOT SEEN'}"


class HandWatch:
    """Body tracking decides WHERE you are, so a missing hand no longer stopped the ball by itself, but it should: if the hand
    holding the hub has left the picture (you put it down, or stepped out with your arm), you are not playing. A swing can hide
    the wrist for a moment, though, so the hand is only called lost after it has been missing for `grace` seconds
    (on top of the 0.25 s the wrist filter already rides out). grace=0 stops the ball the moment the wrist is lost, as
    wrist tracking always did."""

    def __init__(self, grace=0.75):
        self.grace = grace
        self._since = None

    def lost(self, hand_visible, now):
        if hand_visible:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self.grace


class LiveSession:
    """Everything one frame does, minus the camera and the window, so it can be
    tested headless: inputs in, game step, then haptics / sounds / MQTT out."""

    def __init__(self, game, pub, haptics=None, sounds=None, feed=None, report_delay=C.LIVE_SWING_REPORT_DELAY):
        self.game, self.pub, self.feed = game, pub, feed
        self.haptics = haptics or Haptics(enabled=False)
        self.sounds = sounds or Sounds(enabled=False)
        self.report_delay = report_delay
        self.log = []                         # (t, event dict) for the saved events file
        self.timing = deque(maxlen=10)        # recent swing timing: seconds late (+) or early (-)
        self.last_swing = ""                  # what the recognizer last said, for the HUD
        self._window_cued = 0
        # Power scale. The recordings put "full power" at a peak of ~1234, but in a real game you
        # swing harder (1300-1800 in your first game), so every shot came out at power 1.0. The top
        # of the scale follows your own swings: your 90th percentile this session, once you've made
        # POWER_ADAPT_AFTER swings, and never below the recorded scale.
        model = getattr(getattr(feed, "rec", None), "model", None)
        self.power_lo, self.power_hi = getattr(model, "power_range", (330.0, 1234.0))
        self.peaks = deque(maxlen=40)

    def step(self, now, dt, wrist, level_card=None, start_card=False, key_swings=()):
        C.SWING_REPORT_DELAY = self.report_delay if self.feed is not None else 0.0
        g = self.game
        swings = list(key_swings)
        if self.feed is not None:
            for rec, sw in self.feed.drain(now):
                self.peaks.append(rec.peak)
                sw.power = self.session_power(rec.peak)
                swings.append(sw)
                mk = getattr(rec, "model_kind", None)
                said = f" (model: {mk})" if mk not in (None, rec.kind) else ""
                self.last_swing = f"{rec.kind}{said} {rec.confidence:.2f}  power {sw.power:.1f}  peak {rec.peak:.0f}"
                entry = {"type": "swing_seen", "kind": rec.kind, "conf": round(rec.confidence, 3),
                         "power": round(sw.power, 3), "peak": round(rec.peak, 1), "age": round(sw.age, 3)}
                if mk not in (None, rec.kind):
                    entry["model_kind"] = mk
                self.log.append((now, entry))
        # How early/late each swing was against the ball's arrival: the number to tune HIT_WINDOW by.
        if g.phase == Phase.INCOMING and g.ball is not None and wrist is not None:
            for sw in swings:
                off = (g.phase_t + dt - sw.age) - g.ball.T
                self.timing.append(off)
                self.log.append((now, {"type": "swing_timing", "offset": round(off, 3)}))

        events = g.update(dt, Inputs(wrist=wrist, swings=swings, level_card=level_card, start_card=start_card))

        # The paddle turns green as the swing window opens.
        if (g.phase == Phase.INCOMING and g.ball is not None and g.ball_serial != self._window_cued
                and g.phase_t >= g.ball.T - g.hit_window - C.HAPTIC_LEAD):
            self._window_cued = g.ball_serial
            self.haptics.cue("window")

        for name, power in cues_for_events(events):
            self.haptics.cue(name, power)
        tune = tune_for_events(events)
        if tune:
            self.sounds.play(tune)
        if any(e["type"] == "record" for e in events):
            self.pub.publish_best(g.best)
        self.pub.tick()
        for e in events:
            self.log.append((now, e))
        return events

    def session_power(self, peak):
        hi = self.power_hi
        if len(self.peaks) >= POWER_ADAPT_AFTER:
            hi = max(hi, float(np.percentile(self.peaks, 90)))
        return float(min(1.0, max(0.0, (peak - self.power_lo) / max(1.0, hi - self.power_lo))))

    def timing_text(self):
        if not self.timing:
            return ""
        last = self.timing[-1]
        med = float(np.median(self.timing))
        word = lambda v: f"{abs(v):.2f} s {'late' if v > 0 else 'early'}"
        return f"last swing {word(last)}, typical {word(med)}"


# --------------------------------------------------------------------- drawing

class LiveView(View):
    def __init__(self, feed=None, min_peak=330.0, stage=True, right_handed=True, track="wrist"):
        super().__init__(stage=stage, right_handed=right_handed)
        self.track = track
        self.incoming_text = ("Step into the ring, swing as the ball lands" if track == "body"
                              else "Get your wrist in the ring, swing as the ball lands")
        self.feed, self.min_peak = feed, min_peak
        self.banner_cx = PIP_X // 2 + 60               # hint text sits left of the camera preview, not under it
        self.banner_right = PIP_X - 6
        self.idle_text = "Show a LEVEL card (1-3), then the START card"
        self.pose_lost_text = "Can't see you - ball frozen" if track == "body" else "Can't see your wrist - ball frozen"
        self.legend = "C calibrate | [ ] reach | 1-3 / S cards | R reset record | Q quit"

    def _signal_value(self, now):
        if self.feed is None:
            return super()._signal_value(now)
        return min(1.2, self.feed.latest_gyro() / 700.0) - 0.75     # rest near the bottom, swings rise

    def _legend(self, img):
        put(img, self.legend, (16, H - 9), 0.45, MUTED, 1)

    def pip(self, img, frame, pose, tags, mapper, game, status):
        """Camera preview: what the camera sees, your skeleton, the cards it reads,
        and where the hitting line and ring are in YOUR space."""
        if frame is None:
            return
        small = cv2.resize(cv2.flip(frame, 1), (PIP_W, PIP_H))
        to_px = lambda x, y: (int(x * PIP_W), int(y * PIP_H))
        if pose is not None and pose.landmarks:
            from pose_input import SKELETON
            for a, b in SKELETON:
                pa, pb = pose.landmarks[a], pose.landmarks[b]
                if pa[2] > 0.5 and pb[2] > 0.5:
                    cv2.line(small, to_px(*pa[:2]), to_px(*pb[:2]), TEAL, 2, cv2.LINE_AA)
            wx, wy, wv = pose.landmarks[pose.idx]
            if wv > 0.5:
                cv2.circle(small, to_px(wx, wy), 6, GREEN, -1, cv2.LINE_AA)
            body = getattr(pose, "body", None)
            if self.track == "body":
                sure = getattr(pose, "body_visibility", 0.5)
                for idx in (0, 11, 12):                                       # nose and shoulders: white = trusted, red = the camera doubts it
                    px_, py_, pv = pose.landmarks[idx]
                    cv2.circle(small, to_px(px_, py_), 3, (255, 255, 255) if pv >= sure else RED, -1, cv2.LINE_AA)
            if body is not None and self.track == "body":                    # the point that decides where you stand
                cv2.circle(small, to_px(*body), 7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(small, to_px(*body), 2, (255, 255, 255), -1, cv2.LINE_AA)
        if tags is not None:
            fh, fw = frame.shape[:2]
            for tag_id, quad in tags.last:
                pts = np.array([((fw - x) / fw * PIP_W, y / fh * PIP_H) for x, y in quad], np.int32)
                cv2.polylines(small, [pts], True, AMBER, 2, cv2.LINE_AA)
                c = pts.mean(axis=0).astype(int)
                put(small, str(tag_id), (c[0] - 5, c[1] + 5), 0.6, AMBER, 2)
        # the hitting line, and the current ring, mapped back into the camera picture
        a, b = mapper.to_camera((0.0, C.RING_Y)), mapper.to_camera((1.0, C.RING_Y))
        cv2.line(small, to_px(*a), to_px(*b), LINE, 1, cv2.LINE_AA)
        if game.phase == Phase.INCOMING:
            cx, cy = mapper.to_camera((game.ring_x, C.RING_Y))
            if mapper.lateral_only:
                # Body tracking only looks side to side, so the target is a LANE, not a circle: stand with the white
                # dot (your body) inside the amber band.
                half = game.ring_radius / C.ASPECT * mapper.reach
                cv2.rectangle(small, to_px(cx - half, 0.0), to_px(cx + half, 1.0), AMBER, 1, cv2.LINE_AA)
            else:
                r = int(game.ring_radius * mapper.reach * mapper.cam_aspect / C.ASPECT * PIP_H)
                cv2.circle(small, to_px(cx, cy), max(4, r), AMBER, 1, cv2.LINE_AA)
        img[PIP_Y:PIP_Y + PIP_H, PIP_X:PIP_X + PIP_W] = small
        cv2.rectangle(img, (PIP_X, PIP_Y), (PIP_X + PIP_W, PIP_Y + PIP_H), LINE, 1)
        self.text_lines(img, status, PIP_X + PIP_W, PIP_Y + PIP_H + 16)

    def text_lines(self, img, lines, right, top):
        """Small status lines, right-aligned, with a translucent backing so they stay readable over the crowd."""
        for i, line in enumerate(lines):
            if self.stage is not None:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                y = top + 16 * i
                blend_rect(img, (right - tw - 6, y - th - 4), (right + 4, y + 5), (20, 18, 18), 0.7)
            put(img, line, (right, top + 16 * i), 0.42, MUTED if self.stage is None else (205, 205, 205), 1, right=True)


# ---------------------------------------------------------------------- set-up

def load_settings(name):
    try:
        with open(SETTINGS) as f:
            return json.load(f).get(name, {})
    except (OSError, ValueError):
        return {}


def save_settings(name, values):
    data = {}
    try:
        with open(SETTINGS) as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    data.setdefault(name, {}).update(values)          # merge: saving a calibration must not erase a saved best
    with open(SETTINGS, "w") as f:
        json.dump(data, f, indent=2)


def restore_best(game, name, fresh=False):
    """Carry your best streak over from earlier sessions, so launching the game again can never lower the score
    that is already published (the game used to publish 0.0 at every start, overwriting a retained best).
    fresh=True starts again from 0. Returns the best the game starts with."""
    if fresh:
        save_settings(name, {"best": 0})
        game.best = 0
    else:
        game.best = int(load_settings(name).get("best", 0))
        if game.best:
            print(f"[record] your best streak so far is {game.best}; it is published at the start so a restart can't "
                  f"lower it (--fresh-record starts again from 0).")
    return game.best


def apply_handedness(hand, flip=False):
    off = dict(C.SHOT_OFFSET_RIGHT_HANDED)
    if (hand == "left") != flip:
        off = {k: -v for k, v in off.items()}
    C.SHOT_OFFSET = off


def connect_hub(args):
    """Returns (dm, hub_imu). Same connection code the recorder and swing_check use."""
    from imu_hub import HubIMU
    from imu_recorder import _FakeAdapter, connect_hub as _connect
    dm = _connect(args)
    hub = _FakeAdapter(dm) if args.fake else HubIMU(dm)
    hub.check()
    if hub.set_notification_delay(args.notify_ms):
        print(f"[hub] IMU readings requested every {args.notify_ms} ms")
    return dm, hub


def fake_le():
    """Stand-in constants so haptics can run against the fake hub."""
    names = ["LEGO_COLOR_" + c for c in ("GREEN", "YELLOW", "RED", "AZURE", "MAGENTA", "PURPLE")]
    names += ["SOUND_PATTERN_BEEP_SINGLE", "SOUND_PATTERN_BEEP_DOUBLE", "LIGHT_PATTERN_BREATHE", "MOTOR_LEFT", "MOTOR_RIGHT"]
    return SimpleNamespace(**{n: i for i, n in enumerate(names)})


def start_swings(args):
    """Hub + recognizer + reader thread. Returns (dm, feed, reader, model)."""
    from hub_input import HubReader, SwingFeed
    from swing_detect import SwingModel, SwingRecognizer
    model = SwingModel.load(args.model)
    dm, hub = connect_hub(args)
    rec = SwingRecognizer(model, min_confidence=args.min_confidence, min_peak=args.min_peak, rule=args.rule,
                          steer=args.steer)
    feed = SwingFeed(rec, keep_readings=not args.no_log)
    reader = HubReader(hub, feed).start()
    return dm, feed, reader, model


def save_logs(session, feed, stamp):
    base = f"live-{stamp}"
    with open(base + "-events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "type", "details"])
        for t, e in session.log:
            w.writerow([f"{t:.4f}", e["type"], json.dumps({k: v for k, v in e.items() if k != "type"})])
    if feed is not None and feed.readings:
        with open(base + "-stream.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"])
            for t, v in feed.readings:
                w.writerow([f"{t:.4f}"] + [f"{x:.3f}" for x in v])
    print(f"[log] saved {base}-events.csv" + (f" and {base}-stream.csv" if feed is not None and feed.readings else ""))


# ------------------------------------------------------------------------ main

KEY_SWINGS = {ord("f"): ("forehand", 0.5), ord("F"): ("forehand", 1.0), ord("b"): ("backhand", 0.5),
              ord("B"): ("backhand", 1.0), ord(" "): ("flat", 0.5), ord("v"): ("flat", 1.0)}


def run(args):
    from pose_input import PoseTracker, WristMapper
    from tag_input import CardDebouncer, TagReader

    apply_handedness(args.hand, args.flip_shots)
    body_mode = args.track == "body" and not args.no_camera       # without a camera the mouse is the position either way
    if args.ring_y is not None:
        C.RING_Y = min(0.8, max(0.35, args.ring_y))
    rng = random.Random(args.seed)
    policy, q_path = make_policy(args, rng)
    game = Game(level=args.level, rng=rng, policy=policy)
    restore_best(game, args.name, args.fresh_record)
    if body_mode:
        game.position_scale = C.BODY_RING_SCALE

    # --- camera, pose and tags
    cap = pose = tags = None
    mouse = {"xy": (0.5, C.RING_Y)}
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            raise SystemExit(f"Couldn't read camera {args.camera}. Try --camera 1, or --no-camera to use the mouse.")
        fh, fw = frame.shape[:2]
        pose = PoseTracker(hand=args.hand, swap_hands=args.swap_hands, body_part=args.body_part,
                           body_visibility=args.body_visibility)
        tags = TagReader()
        saved = load_settings(args.name)
        mapper = WristMapper(cx=saved.get("cx", 0.5), cy=saved.get("cy", 0.55),
                             reach=(None if body_mode else args.reach) or saved.get("reach", 0.6), cam_aspect=fw / fh)
        body_mapper = WristMapper(cx=saved.get("bx", 0.5), cy=saved.get("by", 0.45), cam_aspect=fw / fh, lateral_only=True,
                                  reach=(args.reach if body_mode else None) or saved.get("breach", DEFAULT_BODY_REACH))
        calibrated = ("bx" if body_mode else "cx") in saved
        if body_mode:
            print(f"[pose] tracking your body ({args.body_part}) for where you stand, so your arm is free to swing; "
                  f"your {args.hand} hand holds the racket. " +
                  ("Using your saved calibration." if calibrated else "Stand where you want the middle of the court and press C."))
        else:
            print(f"[pose] tracking your {args.hand} wrist. " +
                  ("Using your saved calibration." if calibrated else "Press C while holding the hub where you like to hit."))
    else:
        mapper = WristMapper()
        body_mapper = WristMapper(lateral_only=True)
        calibrated = True
    active = body_mapper if body_mode else mapper
    debouncer = CardDebouncer()

    # --- hub
    dm = feed = reader = None
    haptics = Haptics(enabled=False)
    if not args.no_hub:
        dm, feed, reader, model = start_swings(args)
        haptics = Haptics(dm, le=fake_le() if args.fake else None, enabled=not args.no_haptics).start()
        haptics.cue("idle")
    keys_swing = args.no_hub or args.keys

    # --- outputs
    pub = ScorePublisher(args.name, broker=args.broker, enabled=not args.no_mqtt, retain=not args.no_retain)
    pub.connect()
    pub.publish_best(game.best)
    if args.name == "YourName":
        print("[mqtt] NOTE: pass --name <your name> so the topic is ME193/Rogers/<your name>")
    session = LiveSession(game, pub, haptics, Sounds(enabled=not args.no_sound), feed)
    view = LiveView(feed, min_peak=getattr(feed.rec, "min_peak", 330.0) if feed else 330.0,
                    stage=not args.plain, right_handed=args.hand == "right", track="body" if body_mode else "wrist")

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    if args.no_camera:
        cv2.setMouseCallback(WIN, lambda ev, x, y, fl, p: mouse.update(xy=(x / W, y / H)))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    last_pose_log = -1.0
    hand_watch = HandWatch(args.hand_grace)
    prev = time.perf_counter()
    frame = None
    fps = deque(maxlen=30)
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame grab failed.")
                    break
            now = time.perf_counter()
            dt = min(now - prev, 0.1)
            fps.append(now - prev)
            prev = now

            # inputs
            if pose is not None:
                wrist_cam = pose.process(frame, now)
                body_cam = getattr(pose, "body", None)
                hand = mapper.to_court(wrist_cam)                                       # where your racket hand is
                wrist = body_mapper.to_court(body_cam) if body_mode else hand           # the position the game checks
                if body_mode and hand_watch.lost(hand is not None, now):
                    wrist = None                                                        # the hand has really gone: freeze the ball
                idle = game.phase == Phase.IDLE
                level, start = debouncer.update(tags.detect(frame), idle=idle)    # always watching; only acts while idle
                if not idle:
                    tags.last = []                                                 # no card outlines mid-rally
            else:
                wrist_cam, body_cam, wrist, hand = None, None, mouse["xy"], SAME
                level, start = None, False

            key = cv2.waitKey(1) & 0xFF
            key_swings = []
            if key in (ord("q"), 27):
                break
            elif key in (ord("c"), ord("C")) and pose is None:
                view.notice("No camera: the mouse is your position, so there is nothing to calibrate", AMBER, now)
            elif key in (ord("c"), ord("C")):
                need = body_cam if body_mode else wrist_cam
                if need is None:
                    print(f"[pose] can't see your {'body' if body_mode else 'wrist'}, so there's nothing to calibrate to")
                    view.notice(CAL_FAIL[body_mode], RED, now, 4.0)
                else:
                    if body_cam is not None:                                           # calibrate both while you stand there
                        body_mapper.calibrate(body_cam)
                        save_settings(args.name, {"bx": body_mapper.cx, "by": body_mapper.cy})
                    if wrist_cam is not None:
                        mapper.calibrate(wrist_cam)
                        save_settings(args.name, {"cx": mapper.cx, "cy": mapper.cy})
                    calibrated = True
                    where = body_cam if body_mode else wrist_cam
                    print(f"[pose] calibrated: the middle of the court is at camera ({where[0]:.2f}, {where[1]:.2f})")
                    view.notice(CAL_OK[body_mode], GREEN, now, 2.5)
            elif key in (ord("["), ord("]")) and pose is not None:
                active.reach = min(1.5, max(0.25, active.reach + (0.05 if key == ord("]") else -0.05)))
                save_settings(args.name, {("breach" if body_mode else "reach"): active.reach})
                what = "stepping" if body_mode else "your hand moving"
                print(f"[pose] {what} across {active.reach:.2f} of the camera width covers the court")
                view.notice(f"{what} across {active.reach:.2f} of the camera width now covers the court", AMBER, now, 2.0)
            elif key in (ord("1"), ord("2"), ord("3")):
                level = int(chr(key))
            elif key in (ord("s"), ord("S")):
                start = True
            elif key in (ord("r"), ord("R")):
                game.reset_record()
                pub.publish_best(0)
                save_settings(args.name, {"best": 0})
            elif key in KEY_SWINGS and keys_swing:
                key_swings.append(Swing(*KEY_SWINGS[key]))

            if pose is not None and now - last_pose_log >= 0.2:                        # 5 a second, for tuning later
                last_pose_log = now
                session.log.append((now, {"type": "pose", "track": args.track,
                                          "body": None if body_cam is None else [round(body_cam[0], 3), round(body_cam[1], 3)],
                                          "wrist": None if wrist_cam is None else [round(wrist_cam[0], 3), round(wrist_cam[1], 3)]}))
            events = session.step(now, dt, wrist, level, start, key_swings)
            if any(e["type"] == "record" for e in events):
                save_settings(args.name, {"best": game.best})
            log_events(events)
            view.handle(events, now)

            # draw
            view.calibrate_text = None if (calibrated or game.phase != Phase.IDLE) else (
                "Stand where you want the middle of the court and press C" if body_mode
                else "Hold the hub where you like to hit and press C")
            if body_mode:
                view.pose_lost_text = "Can't see you - ball frozen" if body_cam is None else "Can't see your hand - ball frozen"
            img = view.render(game, wrist, now, hand=hand if body_mode else SAME)
            status = [f"{1 / max(1e-3, np.mean(fps)):.0f} fps   MQTT {pub.status}"]
            if pose is not None:
                status.append(pose_status(body_mode, body_cam, wrist_cam))
            if feed is not None:
                hz = feed.fresh_rate()
                status.append((f"hub {hz:.0f}/s" if hz else "hub: waiting for data") +
                              ("   WRONG IMU RATE" if hz and not feed.rate_ok() else ""))
            info = []
            if session.last_swing:
                info.append(("hub saw: " + session.last_swing, PURPLE))
            if session.timing_text():
                info.append((session.timing_text(), MUTED))
            if (C.RING_Y + game.ring_radius) * H < H - 130:       # room under the ring: show it there
                for i, (text, color) in enumerate(info):
                    put(img, text, (16, H - 108 + 16 * i), 0.5, color, 1)
            else:
                status += [text for text, _ in info]
            if frame is not None:
                view.pip(img, frame, pose, tags, active, game, status)
            else:
                view.text_lines(img, status, W - 16, 64)
            cv2.imshow(WIN, img)
    finally:
        if q_path:
            policy.save(q_path)
            print(f"\n[q] saved {q_path} after {policy.shots} shots. What it has learned about you:")
            print(policy.describe())
        print(f"Best streak this session: {game.best}  (hits {game.hits}, misses {game.misses}, winners {game.winners})")
        if not args.no_log:
            save_logs(session, feed, stamp)
        haptics.stop()
        if reader is not None:
            reader.stop()
        if dm is not None:
            try:
                dm.disconnect()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        if pose is not None:
            pose.close()
        pub.close()
        cv2.destroyAllWindows()


def check_haptics(args, repeats=5, gap=1.6, listen=1.2):
    """Play each buzz with the hub held still and count whether the swing recognizer
    fires. A buzz that registers as a swing could steal a hit in the game."""
    dm, feed, reader, model = start_swings(args)
    hap = Haptics(dm, le=fake_le() if args.fake else None)
    total = 3 * repeats
    secs = total * (gap + listen)
    print(f"""
HAPTICS CHECK: does the buzz ever look like a swing?

  * Hold the hub in your playing grip (right hand, blue side on your palm).
  * Keep your hand STILL the whole time. Do NOT swing.
  * The hub will buzz {total} times over about {secs:.0f} seconds. Just let it.

Any 'swing' the recognizer reports during this test is a false alarm caused by
the buzz, which is exactly what we're checking for.
""")
    for n in (3, 2, 1):
        print(f"  starting in {n}...", flush=True)
        time.sleep(1.0)
    rows = []
    k = 0
    try:
        for name in ("hit", "winner", "miss"):
            fired, peaks = 0, []
            for _ in range(repeats):
                time.sleep(gap)
                feed.drain(time.perf_counter())
                t0 = time.perf_counter()
                k += 1
                hap.play_now(name, power=1.0)
                time.sleep(listen)
                swings = feed.drain(time.perf_counter())
                fired += len(swings)
                with feed.lock:
                    peak = max((g for t, g in feed.gyro if t >= t0), default=0.0)
                peaks.append(peak)
                note = "  <-- counted as a swing!" if swings else ""
                print(f"  buzz {k:>2}/{total} ({name:<6})  biggest gyro {peak:5.0f}{note}", flush=True)
            rows.append((name, fired, max(peaks)))
    finally:
        reader.stop()
        try:
            dm.disconnect()
        except Exception:
            pass
    floor = model.min_peak
    print("\nDone, you can relax.\n")
    print(f"  {'buzz':<8}{'counted as a swing':>20}{'biggest gyro':>15}")
    for name, fired, peak in rows:
        print(f"  {name:<8}{f'{fired} of {repeats}':>20}{peak:>15.0f}")
    print(f"\n(A reading has to reach {floor:.0f} before it can count as a swing.)")
    if any(f for _, f, _ in rows):
        print("RESULT: some buzzes registered as swings. Use a shorter or lighter beam, "
              "or shorten the buzz times in haptics.py, then run this again.")
    else:
        print("RESULT: no buzz registered as a swing. Haptics are safe to leave on.")
    if all(p < 0.2 * floor for _, _, p in rows):
        print("NOTE: the buzz barely moved the hub. Is the beam on the LEFT motor, held by one end, "
              "and turning with the shaft (not slipping)?")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Virtual ping-pong, live: webcam pose, AprilTag cards, hub IMU")
    p.add_argument("--name", default="YourName", help="your name: the score goes to ME193/Rogers/<name>")
    p.add_argument("--broker", default=DEFAULT_BROKER)
    p.add_argument("--no-mqtt", action="store_true")
    p.add_argument("--no-retain", action="store_true", help="don't publish the score as a retained message")
    p.add_argument("--level", type=int, default=1, choices=[1, 2, 3], help="starting level (the cards change it)")
    p.add_argument("--opponent", choices=["q", "rule", "random"], default="q",
                   help="q = learns where you are weak (default), rule = fixed tactic")
    p.add_argument("--q-file", default=None)
    p.add_argument("--q-fresh", action="store_true")
    p.add_argument("--hand", choices=["right", "left"], default="right", help="the hand holding the hub")
    p.add_argument("--flip-shots", action="store_true", help="swap where forehands and backhands go")
    p.add_argument("--swap-hands", action="store_true", help="if the camera already mirrors, track the other wrist label")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--ring-y", type=float, default=None,
                   help=f"how high the ring sits on screen, 0 = top, 1 = bottom (default {C.RING_Y}); your hand position doesn't change")
    p.add_argument("--track", choices=["body", "wrist"], default="body",
                   help="what decides where you are for the ring. body (default): the middle of your shoulders, side to side only, "
                        "so your arm is free to swing from wherever it is; wrist: your racket hand, as before")
    p.add_argument("--body-part", choices=["shoulders", "nose", "hips"], default="shoulders",
                   help="with --track body: which point is 'you'. shoulders is steadiest; nose bobs and turns; hips are usually out of frame")
    p.add_argument("--body-visibility", type=float, default=0.5, metavar="0..1",
                   help="with --track body: how sure the camera must be that your shoulders (or nose) are in view. If the preview "
                        "shows red dots, or 'you: NOT SEEN' while you are clearly in the picture, try 0.3")
    p.add_argument("--hand-grace", type=float, default=2.75, metavar="SECONDS",
                   help="with --track body: how long your racket hand may be out of the picture before the ball freezes "
                        "(a swing can hide the wrist for a moment). 0 = freeze as soon as the wrist is lost, as --track wrist does")
    p.add_argument("--reach", type=float, default=None,
                   help="fraction of the camera width that spans the court (for the thing --track follows; ]/[ change it in the game)")
    p.add_argument("--no-camera", action="store_true", help="mouse = wrist, keys = cards")
    p.add_argument("--no-hub", action="store_true", help="keyboard swings instead of the hub")
    p.add_argument("--keys", action="store_true", help="allow keyboard swings even with the hub")
    p.add_argument("--fake", action="store_true", help="synthetic hub (plumbing test)")
    p.add_argument("--serial", default=None)
    p.add_argument("--color", default=None)
    p.add_argument("--notify-ms", type=int, default=30)
    p.add_argument("--model", default="swing_model.json")
    p.add_argument("--min-confidence", type=float, default=None, help="accept cutoff (default: 0.9 for the swing rule, 0.6 for the kind rule)")
    p.add_argument("--rule", choices=["swing", "kind"], default="swing",
                   help="swing = accept if it is confidently a swing of any kind (default); kind = the old rule")
    p.add_argument("--steer", choices=["rotation", "model"], default="rotation",
                   help="what steers the shot. rotation (default): a swing that rotates mainly about the hub's y-axis "
                        "goes by the DIRECTION of that rotation, so how hard you swing can't change where the ball goes; "
                        "model: the classifier's kind (what the guided tools report)")
    p.add_argument("--min-peak", type=float, default=None)
    p.add_argument("--fresh-record", action="store_true",
                   help="start the record at 0 and publish 0.0 (default: carry your best streak over from earlier "
                        "sessions, so a restart never lowers the published score)")
    p.add_argument("--plain", action="store_true", help="the original look: no crowd, characters or rackets")
    p.add_argument("--no-haptics", action="store_true")
    p.add_argument("--no-sound", action="store_true")
    p.add_argument("--no-log", action="store_true", help="don't save live-<time>-events.csv / -stream.csv")
    p.add_argument("--check-haptics", action="store_true", help="test whether the buzz registers as a swing, then exit")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.check_haptics:
        check_haptics(args)
    else:
        try:
            run(args)
        except KeyboardInterrupt:
            print("\nStopped with Ctrl+C (everything above was saved). Tip: press Q in the game window to quit.")


if __name__ == "__main__":
    main()
