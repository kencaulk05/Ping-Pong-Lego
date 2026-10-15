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
import os
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
from pingpong_sim import (AMBER, GREEN, H, LINE, MUTED, PURPLE, RED, TEAL, TEXT, W, View,
                          log_events, make_policy, put)
from score_publisher import DEFAULT_BROKER, ScorePublisher
from sounds import Sounds, tune_for_events

WIN = "Virtual ping-pong (live)"
SETTINGS = "live_settings.json"
PIP_W, PIP_H = 208, 156
PIP_X, PIP_Y = W - PIP_W - 8, 52
POWER_ADAPT_AFTER = 5


# ------------------------------------------------------------------ the session

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
                self.last_swing = f"{rec.kind} {rec.confidence:.2f}  power {sw.power:.1f}  peak {rec.peak:.0f}"
                self.log.append((now, {"type": "swing_seen", "kind": rec.kind, "conf": round(rec.confidence, 3),
                                       "power": round(sw.power, 3), "peak": round(rec.peak, 1),
                                       "age": round(sw.age, 3)}))
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
    def __init__(self, feed=None, min_peak=330.0):
        super().__init__()
        self.feed, self.min_peak = feed, min_peak
        self.idle_text = "Show a LEVEL card (1-3), then the START card"
        self.pose_lost_text = "Can't see your wrist - ball frozen"
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
            r = int(C.RING_RADIUS * mapper.reach * mapper.cam_aspect / C.ASPECT * PIP_H)
            cv2.circle(small, to_px(cx, cy), max(4, r), AMBER, 1, cv2.LINE_AA)
        img[PIP_Y:PIP_Y + PIP_H, PIP_X:PIP_X + PIP_W] = small
        cv2.rectangle(img, (PIP_X, PIP_Y), (PIP_X + PIP_W, PIP_Y + PIP_H), LINE, 1)
        for i, line in enumerate(status):
            put(img, line, (PIP_X + PIP_W, PIP_Y + PIP_H + 16 + 16 * i), 0.42, MUTED, 1, right=True)


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
    data[name] = values
    with open(SETTINGS, "w") as f:
        json.dump(data, f, indent=2)


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
    rec = SwingRecognizer(model, min_confidence=args.min_confidence, min_peak=args.min_peak, rule=args.rule)
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
    if args.ring_y is not None:
        C.RING_Y = min(0.8, max(0.35, args.ring_y))
    rng = random.Random(args.seed)
    policy, q_path = make_policy(args, rng)
    game = Game(level=args.level, rng=rng, policy=policy)

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
        pose = PoseTracker(hand=args.hand, swap_hands=args.swap_hands)
        tags = TagReader()
        saved = load_settings(args.name)
        mapper = WristMapper(cx=saved.get("cx", 0.5), cy=saved.get("cy", 0.55),
                             reach=args.reach or saved.get("reach", 0.6), cam_aspect=fw / fh)
        calibrated = "cx" in saved
        print(f"[pose] tracking your {args.hand} wrist. " +
              ("Using your saved calibration." if calibrated else "Press C while holding the hub where you like to hit."))
    else:
        mapper = WristMapper()
        calibrated = True
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
    pub.publish_best(0)
    if args.name == "YourName":
        print("[mqtt] NOTE: pass --name <your name> so the topic is ME193/Rogers/<your name>")
    session = LiveSession(game, pub, haptics, Sounds(enabled=not args.no_sound), feed)
    view = LiveView(feed, min_peak=getattr(feed.rec, "min_peak", 330.0) if feed else 330.0)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    if args.no_camera:
        cv2.setMouseCallback(WIN, lambda ev, x, y, fl, p: mouse.update(xy=(x / W, y / H)))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
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
                wrist = mapper.to_court(wrist_cam)
                level, start = debouncer.update(tags.detect(frame) if game.phase == Phase.IDLE else set())
                if game.phase != Phase.IDLE:
                    tags.last = []
            else:
                wrist_cam, wrist = None, mouse["xy"]
                level, start = None, False

            key = cv2.waitKey(1) & 0xFF
            key_swings = []
            if key in (ord("q"), 27):
                break
            elif key in (ord("c"), ord("C")) and pose is not None:
                if wrist_cam is None:
                    print("[pose] can't see your wrist, so there's nothing to calibrate to")
                else:
                    mapper.calibrate(wrist_cam)
                    calibrated = True
                    save_settings(args.name, mapper.as_dict())
                    print(f"[pose] calibrated: hitting line at camera ({wrist_cam[0]:.2f}, {wrist_cam[1]:.2f})")
            elif key in (ord("["), ord("]")) and pose is not None:
                mapper.reach = min(1.0, max(0.25, mapper.reach + (0.05 if key == ord("]") else -0.05)))
                save_settings(args.name, mapper.as_dict())
                print(f"[pose] reach {mapper.reach:.2f} of the camera width covers the court")
            elif key in (ord("1"), ord("2"), ord("3")):
                level = int(chr(key))
            elif key in (ord("s"), ord("S")):
                start = True
            elif key in (ord("r"), ord("R")):
                game.reset_record()
                pub.publish_best(0)
            elif key in KEY_SWINGS and keys_swing:
                key_swings.append(Swing(*KEY_SWINGS[key]))

            events = session.step(now, dt, wrist, level, start, key_swings)
            log_events(events)
            view.handle(events, now)

            # draw
            img = view.render(game, wrist, now)
            if not calibrated and game.phase == Phase.IDLE:
                put(img, "Hold the hub where you like to hit and press C", (W // 2, 140), 0.7, AMBER, 2, center=True)
            status = [f"{1 / max(1e-3, np.mean(fps)):.0f} fps   MQTT {pub.status}"]
            if feed is not None:
                hz = feed.fresh_rate()
                status.append((f"hub {hz:.0f}/s" if hz else "hub: waiting for data") +
                              ("   WRONG IMU RATE" if hz and not feed.rate_ok() else ""))
            info = []
            if session.last_swing:
                info.append(("hub saw: " + session.last_swing, PURPLE))
            if session.timing_text():
                info.append((session.timing_text(), MUTED))
            if (C.RING_Y + C.RING_RADIUS) * H < H - 130:       # room under the ring: show it there
                for i, (text, color) in enumerate(info):
                    put(img, text, (16, H - 108 + 16 * i), 0.5, color, 1)
            else:
                status += [text for text, _ in info]
            if frame is not None:
                view.pip(img, frame, pose, tags, mapper, game, status)
            else:
                for i, line in enumerate(status):
                    put(img, line, (W - 16, 64 + 16 * i), 0.42, MUTED, 1, right=True)
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
    p.add_argument("--reach", type=float, default=None, help="fraction of the camera width that spans the court")
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
    p.add_argument("--min-peak", type=float, default=None)
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
