"""
pingpong_sim.py -- play the game with a mouse and keyboard standing in for the
real sensors. Same game core as the real thing; only the input source differs.

    mouse                     = your wrist (what MediaPipe pose will supply)
    f / b / space             = forehand / backhand / flat swing (what the hub IMU + classifier will supply)
    Shift+f / Shift+b / v     = the same swings, hit hard
    1 / 2 / 3                 = show a level card (AprilTag 1/2/3), only while idle
    s                         = show the start card (AprilTag 0)
    x                         = toggle "pose lost" (tests what happens when the wrist drops out)
    r                         = reset the record
    q or Esc                  = quit

Usage:
    python pingpong_sim.py --name YourName
    python pingpong_sim.py --name YourName --bot            # watch a scripted player
    python pingpong_sim.py --no-mqtt                         # don't touch the network
    python pingpong_sim.py --headless 60 --bot --no-mqtt     # no window, just run 60 simulated seconds
"""

import argparse
import math
import os
import random
import time
from collections import deque

import cv2
import numpy as np

import config as C
from bots import BotPlayer
from opponent import RandomOpponent, RuleOpponent
from pingpong_core import Game, Inputs, Phase, Swing
from qopponent import QOpponent
from score_publisher import DEFAULT_BROKER, ScorePublisher
from stage import BALL_ARC, SAME, Stage, blend_rect

W, H = 960, 600
WIN = "Virtual ping-pong (simulator)"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors
BG = (34, 30, 30)
COURT = (62, 74, 52)
LINE = (120, 140, 110)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
BALL = (39, 159, 239)
BALL_EDGE = (11, 79, 133)
TEAL = (117, 158, 29)
GREEN = (90, 210, 110)
RED = (74, 75, 226)
PURPLE = (221, 119, 127)
AMBER = (60, 190, 250)

Y_FAR = int(0.16 * H)
Y_NEAR = int(C.RING_Y * H)   # (read live in project(), so --ring-y can move it)


def lerp(a, b, t):
    return a + (b - a) * t


def project(x_norm, depth):
    """Court coordinates -> pixels. depth 0 = far wall, 1 = the ring line."""
    scale = lerp(0.55, 1.0, depth)
    return (int(W * (0.5 + (x_norm - 0.5) * scale)), int(lerp(Y_FAR, int(C.RING_Y * H), depth)))


def put(img, text, org, scale=0.6, color=TEXT, thick=1, center=False, right=False):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    if center:
        x -= tw // 2
    elif right:
        x -= tw
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def dashed_circle(img, center, r, color, thick=2, dash=14, gap=10):
    for a in range(0, 360, dash + gap):
        cv2.ellipse(img, center, (r, r), 0, a, a + dash, color, thick, cv2.LINE_AA)


class View:
    """Everything that is only about drawing. The game never knows it exists."""

    NOTICE_WIDTH = 640                  # the widest a line of a notice may be, in pixels; longer messages wrap

    def __init__(self, stage=True, right_handed=True):
        # stage=False (--plain) gives the original look: no crowd, no characters, a bar and a dot.
        self.stage = Stage(W, H, project, right_handed) if stage else None
        self.incoming_text = "Get your wrist in the ring, swing as the ball lands"
        self.calibrate_text = None                # the live game sets this until you have calibrated (shown under the idle hint)
        self._notice = None                       # (text, color, until): a message that must be seen, e.g. the result of pressing C
        self.banner_cx = W // 2                   # where hint text is centered (the live game moves it off the camera preview)
        self.banner_right = W - 8                 # ...and the right edge it must not run past (the preview's left edge)
        self.trail = deque(maxlen=7)
        self.flash_color = None
        self.flash_until = 0.0
        self.last_decision = "waiting for a swing"
        self.pulses = []                          # (time, power) of recent swings
        self.signal = deque(maxlen=180)           # ~3 s of the swing-signal strip
        # Wording the live game replaces (it uses cards and a real wrist, not keys and a mouse)
        self.idle_text = "Press 1 / 2 / 3 for a level, then S to start"
        self.pose_lost_text = "POSE LOST - ball frozen (x to toggle)"

    # ------------------------------------------------------------- bookkeeping

    def handle(self, events, now):
        for e in events:
            t = e["type"]
            if t == "hit":
                self.pulses.append((now, e["power"]))
                self.flash(GREEN, now, 0.35)
                self.last_decision = (f"HIT  {e['kind']}  power {e['power']:.1f}  "
                                      f"accuracy {e['accuracy']:.0%}  -> streak {e['streak']}")
            elif t == "miss":
                self.flash(RED, now, 0.5)
                self.last_decision = f"MISS  {e['reason']}"
            elif t == "whiff":
                self.pulses.append((now, 0.5))
                self.last_decision = f"swing ignored: {e['reason']}"
            elif t == "swing_ignored":
                self.last_decision = "swing ignored: cooldown"
            elif t == "winner":
                self.flash(AMBER, now, 0.5)
                self.last_decision = "WINNER - opponent couldn't reach it"
            elif t == "pose_lost":
                self.last_decision = "wrist lost: ball frozen"
            elif t == "pose_back":
                self.last_decision = "wrist found again"
        if self.stage is not None:
            self.stage.handle(events, now)

    def notice(self, text, color, now, seconds=3.0):
        """Show a message in the middle of the court for a few seconds. For things the player has to be told: the result
        of pressing a key must never be visible only in the terminal."""
        self._notice = (text, color, now + seconds)

    def flash(self, color, now, seconds):
        self.flash_color = color
        self.flash_until = now + seconds

    def _signal_value(self, now):
        v = 0.04 * math.sin(now * 17.0) + 0.03 * math.sin(now * 5.3)   # idle sensor noise
        for t0, power in self.pulses:
            age = now - t0
            if 0 <= age < 0.35:
                v += (0.35 + 0.65 * power) * (1 - age / 0.35) * math.sin(age * 38.0)
        return v

    # ----------------------------------------------------------------- drawing

    def render(self, game, wrist, now, hand=SAME):
        """wrist: your position (where the ring check looks). hand: where your racket hand is drawn, when that is not
        the same place (body tracking); SAME draws it at `wrist`, and None means the hand can't be seen."""
        img = np.full((H, W, 3), BG, np.uint8)
        if self.stage is not None:
            self.stage.draw_stands(img, game, now)               # the crowd, behind and beside the court
        self._court(img)
        self._opponent(img, game, now)
        if self.stage is not None:
            self.stage.draw_net(img)                              # between the two players, in front of the opponent
        if game.phase == Phase.INCOMING:
            self._ring(img, game, wrist)
        self._ball(img, game)
        self._player(img, game, wrist, now, hand)
        if self.stage is not None:
            self.stage.draw_pops(img, game, now)
            self.stage.draw_confetti(img, now)
        self._hud(img, game)
        if self._notice is not None:
            if now < self._notice[2]:
                for i, line in enumerate(self._wrap(self._notice[0], 0.8, 2, self.NOTICE_WIDTH)):   # a long message wraps; it must never run off the edge
                    self._banner(img, line, 300 + 36 * i, 0.8, self._notice[1], 2, cx=W // 2)
            else:
                self._notice = None
        self._signal_strip(img, now)
        self._legend(img)
        if self.flash_color is not None and now < self.flash_until:
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), self.flash_color, 12)
        return img

    def _court(self, img):
        poly = np.array([project(0, 0), project(1, 0), project(1, 1.25), project(0, 1.25)], np.int32)
        cv2.fillPoly(img, [poly], COURT)
        for edge in (0.2, 0.4, 0.6, 0.8):
            cv2.line(img, project(edge, 0), project(edge, 1.25), LINE, 1, cv2.LINE_AA)
        cv2.line(img, project(0, 0.5), project(1, 0.5), LINE, 1, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, LINE, 2, cv2.LINE_AA)
        cv2.line(img, project(0, 0), project(1, 0), TEXT, 3, cv2.LINE_AA)

    def _opponent(self, img, game, now=None):
        if self.stage is not None and now is not None:
            self.stage.draw_opponent(img, game, now)
            return
        cx, cy = project(game.opp_x, 0)
        half = int(game.level.opp_half_width * W * 0.55)
        cv2.rectangle(img, (cx - half, cy - 9), (cx + half, cy + 9), RED, -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - half, cy - 9), (cx + half, cy + 9), TEXT, 1, cv2.LINE_AA)

    def _ring(self, img, game, wrist):
        center = (int(game.ring_x * W), int(C.RING_Y * H))
        r = int(game.ring_radius * H)
        inside = game.in_ring(wrist)
        in_window = abs(game.phase_t - game.ball.T) <= game.hit_window
        if inside:
            cv2.circle(img, center, r, GREEN, 3, cv2.LINE_AA)
        else:
            dashed_circle(img, center, r, TEAL, 2)
        if in_window:                                   # timing cue: swing now
            cv2.circle(img, center, r + 10, AMBER, 2, cv2.LINE_AA)
            put(img, "SWING", (center[0], center[1] - r - 18), 0.7, AMBER, 2, center=True)

    def _ball(self, img, game):
        pos = game.ball_position()
        if pos is None:
            self.trail.clear()
            return
        x_norm, depth = pos
        px, py = project(x_norm, depth)
        rad = int(lerp(7, 24, depth))
        if self.stage is not None:
            # With a net in the middle the ball has to fly over it: rise on a gentle arc, with a shadow on the court
            # below. The arc is 0 at the opponent's end and 0 at the ring, so the ball still lands exactly on the ring.
            lift = int(BALL_ARC * math.sin(math.pi * min(1.0, max(0.0, depth))))
            cv2.ellipse(img, (px, py + 2), (int(rad * 0.9), max(2, int(rad * 0.32))), 0, 0, 360, (18, 22, 18), -1, cv2.LINE_AA)
            py -= lift
        self.trail.append((px, py, rad))
        n = len(self.trail)
        for i, (tx, ty, tr) in enumerate(list(self.trail)[:-1]):
            f = (i + 1) / n
            col = tuple(int(lerp(BG[k], BALL[k], f * 0.6)) for k in range(3))
            cv2.circle(img, (tx, ty), max(2, int(tr * (0.35 + 0.5 * f))), col, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), rad, BALL, -1, cv2.LINE_AA)
        cv2.circle(img, (px, py), rad, BALL_EDGE, 2, cv2.LINE_AA)

    def _player(self, img, game, wrist, now=None, hand=SAME):
        if self.stage is not None and now is not None:
            self.stage.draw_player(img, game, wrist, now, game.phase == Phase.INCOMING, hand)
            return
        if wrist is None:
            return
        cx, cy = int(wrist[0] * W), int(wrist[1] * H)
        inside = game.phase == Phase.INCOMING and game.in_ring(wrist)
        cv2.rectangle(img, (cx - 24, cy - 8), (cx + 4, cy + 8), PURPLE, -1, cv2.LINE_AA)   # the hub
        cv2.circle(img, (cx, cy), 9, GREEN if inside else TEAL, -1, cv2.LINE_AA)           # the wrist

    def _hud(self, img, game):
        cv2.rectangle(img, (0, 0), (W, 44), (24, 22, 22), -1)
        put(img, f"Level {game.level_id}  {game.level.name}", (16, 29), 0.7, TEXT, 2)
        put(img, f"Streak {game.streak}", (W // 2, 29), 0.8, TEXT, 2, center=True)
        put(img, f"Best {float(game.best):.1f}", (W - 16, 29), 0.7, TEXT, 2, right=True)
        put(img, game.policy.status(), (W - 130, 28), 0.45, MUTED, 1, right=True)

        word = self.stage.current_word() if self.stage is not None else None
        hy = 148 if self.stage is not None else 80          # with the crowd and characters up top, hints sit on the court
        if game.paused:
            self._banner(img, self.pose_lost_text, hy, 0.8, RED, 2)
        elif word is not None and game.phase != Phase.COUNTDOWN:
            self._banner(img, word[0], hy, 0.95, word[1], 2)                   # a word from the crowd
        elif game.phase == Phase.IDLE:
            # Every idle message gets its own line, top to bottom, so none can print over another: the level/start
            # hint, then the calibration prompt (until you have calibrated), then the last game's result.
            step = 34 if self.stage is not None else 28
            y = hy
            self._banner(img, self.idle_text, y, 0.8, TEXT, 2)
            if self.calibrate_text:
                y += step
                self._banner(img, self.calibrate_text, y, 0.7, AMBER, 2)
            if game.last_outcome:
                y += step
                self._banner(img, game.last_outcome, y, 0.6, MUTED, 1)
        elif game.phase == Phase.COUNTDOWN:
            n = max(1, math.ceil(game.countdown_remaining))
            put(img, str(n), (W // 2, int(H * 0.45)), 4.0, TEXT, 6, center=True)
        elif game.phase == Phase.INCOMING:
            self._banner(img, self.incoming_text, hy, 0.7, MUTED, 1)
        elif game.phase == Phase.RETURNING:
            self._banner(img, "Opponent is chasing it...", hy, 0.7, MUTED, 1)
        elif game.phase == Phase.RESERVE:
            self._banner(img, game.last_outcome, hy, 0.8, TEXT, 2)

    @staticmethod
    def _wrap(text, scale, thick, max_width):
        """Break text into lines that each fit in max_width pixels (a single over-long word still gets its own line)."""
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if cur and cv2.getTextSize(trial, FONT, scale, thick)[0][0] > max_width:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        return lines + [cur] if cur else lines

    def _banner(self, img, text, y, scale, color, thick, cx=None):
        """Centered text; over the crowd it gets a translucent backing so it stays readable. cx=None keeps it clear of
        the camera preview; give cx to center it exactly (a notice in the middle of the court, below the preview)."""
        if self.stage is not None:
            (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
            cx = self._banner_x(tw) if cx is None else cx
            blend_rect(img, (cx - tw // 2 - 14, y - th - 9), (cx + tw // 2 + 14, y + 9), (20, 18, 18), 0.6)
            if color == MUTED:
                color = (205, 205, 205)                     # the grey is too dim over a picture
        else:
            cx = W // 2 if cx is None else cx
        put(img, text, (cx, y), scale, color, thick, center=True)

    def _banner_x(self, text_width):
        """Where to center a banner: at banner_cx, slid left if it would run past banner_right."""
        cx = min(self.banner_cx, self.banner_right - (text_width // 2 + 14))
        return max(cx, text_width // 2 + 22)

    def _signal_strip(self, img, now):
        self.signal.append(self._signal_value(now))
        x0, y0, x1, y1 = 16, H - 78, W - 16, H - 30
        cv2.rectangle(img, (x0, y0), (x1, y1), (24, 22, 22), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), LINE, 1)
        put(img, "hub swing signal", (x0 + 8, y0 + 16), 0.45, MUTED, 1)
        put(img, self.last_decision, (x1 - 8, y0 + 16), 0.5, TEXT, 1, right=True)
        mid = (y0 + y1) // 2 + 8
        amp = (y1 - y0) * 0.42
        pts = []
        n = len(self.signal)
        for i, v in enumerate(self.signal):
            px = int(lerp(x0 + 8, x1 - 8, i / max(1, self.signal.maxlen - 1)))
            pts.append((px, int(mid - max(-1.2, min(1.2, v)) * amp)))
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, PURPLE, 2, cv2.LINE_AA)
        self.pulses = [(t, p) for t, p in self.pulses if now - t < 1.0]

    def _legend(self, img):
        put(img, "mouse = wrist | f / b / space = swing (Shift or v = hard) | 1-3 level | s start | x lose pose | r reset record | q quit",
            (16, H - 9), 0.45, MUTED, 1)


# ---------------------------------------------------------------------- inputs

class SimInput:
    """Mouse + keyboard standing in for pose, IMU and AprilTag."""

    def __init__(self):
        self.mouse = (0.5, 0.85)
        self.pose_lost = False

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (min(1.0, max(0.0, x / W)), min(1.0, max(0.0, y / H)))

    def read(self, key):
        cmd = None
        swings, level, start = [], None, False
        if key == ord("x"):
            self.pose_lost = not self.pose_lost
        elif key in (ord("f"), ord("F")):
            swings.append(Swing("forehand", 1.0 if key == ord("F") else 0.5))
        elif key in (ord("b"), ord("B")):
            swings.append(Swing("backhand", 1.0 if key == ord("B") else 0.5))
        elif key == ord(" "):
            swings.append(Swing("flat", 0.5))
        elif key == ord("v"):
            swings.append(Swing("flat", 1.0))
        elif key in (ord("1"), ord("2"), ord("3")):
            level = int(chr(key))
        elif key == ord("s"):
            start = True
        elif key == ord("r"):
            cmd = "reset"
        elif key in (ord("q"), 27):
            cmd = "quit"
        wrist = None if self.pose_lost else self.mouse
        return Inputs(wrist=wrist, swings=swings, level_card=level, start_card=start), cmd


# ------------------------------------------------------------------------ main

def log_events(events):
    for e in events:
        t = e["type"]
        if t in ("start", "level", "serve", "incoming"):
            continue
        extra = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items() if k != "type"}
        print(f"[game] {t} {extra}")


def make_policy(args, rng):
    if args.opponent == "random":
        return RandomOpponent(rng), None
    if args.opponent == "rule":
        return RuleOpponent(rng), None
    q = QOpponent(rng)
    path = args.q_file or f"qtable_{args.name}.json"
    if os.path.exists(path) and not args.q_fresh:
        try:
            q.load(path)
            print(f"[q] loaded {path}: {q.shots} shots of experience, exploring {q.epsilon:.0%}")
        except Exception as exc:                   # unreadable or wrong shape
            print(f"[q] WARNING: could not load {path} ({exc}); starting fresh")
    else:
        print("[q] starting with an empty Q-table")
    return q, path


def make_bot(args, rng):
    weak = dict(weak_zones={0: 0.70, 1: 0.25, 4: 0.15}, fast_penalty=0.30) if args.bot_weak else {}
    return BotPlayer(skill=args.bot_skill, rng=rng, level=args.level, **weak)


def parse_args():
    p = argparse.ArgumentParser(description="Virtual ping-pong simulator")
    p.add_argument("--name", default="YourName", help="your name, used in the MQTT topic ME193/Rogers/<name>")
    p.add_argument("--broker", default=DEFAULT_BROKER)
    p.add_argument("--no-mqtt", action="store_true", help="don't connect to any broker")
    p.add_argument("--retain", action="store_true", help="publish the score as a retained message")
    p.add_argument("--level", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--plain", action="store_true", help="the original look: no crowd or characters")
    p.add_argument("--bot", action="store_true", help="let a scripted player play")
    p.add_argument("--bot-skill", type=float, default=0.85)
    p.add_argument("--bot-weak", action="store_true",
                   help="give the bot a built-in weakness (bad at zone 0 and at fast shots) so you can watch the Q opponent find it")
    p.add_argument("--opponent", choices=["rule", "q", "random"], default="rule",
                   help="rule = fixed tactic, q = learns where you are weak, random = baseline")
    p.add_argument("--q-file", default=None, help="where the Q-table is saved (default: qtable_<name>.json)")
    p.add_argument("--q-fresh", action="store_true", help="ignore any saved Q-table and start learning from scratch")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--headless", type=float, default=0.0, metavar="SECONDS",
                   help="no window: simulate this many seconds with the bot and print a summary")
    p.add_argument("--snapshot", default=None, help="with --headless: save the last frame to this PNG")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    policy, q_path = make_policy(args, rng)
    game = Game(level=args.level, rng=rng, policy=policy)
    view = View(stage=not args.plain)

    pub = ScorePublisher(args.name, broker=args.broker, enabled=not args.no_mqtt, retain=args.retain)
    pub.connect()
    pub.publish_best(0)

    if args.name == "YourName":
        print("[mqtt] NOTE: pass --name <your name> so the topic is ME193/Rogers/<your name>")

    try:
        if args.headless > 0:
            run_headless(args, game, view, pub, rng)
        else:
            run_window(args, game, view, pub, rng)
    finally:
        if q_path:
            policy.save(q_path)
            print(f"\n[q] saved {q_path} after {policy.shots} shots. What it has learned about you:")
            print(policy.describe())
        pub.close()


def run_window(args, game, view, pub, rng):
    sim = SimInput()
    bot = make_bot(args, rng) if args.bot else None
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, sim.on_mouse)
    prev = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            dt = min(now - prev, 0.1)
            prev = now

            key = cv2.waitKey(1) & 0xFF
            inp, cmd = sim.read(key)
            if cmd == "quit":
                break
            if cmd == "reset":
                game.reset_record()
                pub.publish_best(0)
            if bot:
                inp = bot.act(game, dt)

            events = game.update(dt, inp)
            log_events(events)
            view.handle(events, now)
            if any(e["type"] == "record" for e in events):
                pub.publish_best(game.best)

            cv2.imshow(WIN, view.render(game, inp.wrist, now))
    finally:
        cv2.destroyAllWindows()


def run_headless(args, game, view, pub, rng):
    bot = make_bot(args, rng)
    dt, now = 1 / 30, 0.0
    img = None
    while now < args.headless:
        inp = bot.act(game, dt)
        events = game.update(dt, inp)
        view.handle(events, now)
        if any(e["type"] == "record" for e in events):
            pub.publish_best(game.best)
        img = view.render(game, inp.wrist, now)
        now += dt
    print(f"\nSimulated {args.headless:.0f}s at level {game.level_id}: "
          f"hits {game.hits}, misses {game.misses}, winners {game.winners}, best streak {game.best}")
    if args.snapshot and img is not None:
        cv2.imwrite(args.snapshot, img)
        print(f"Saved {args.snapshot}")


if __name__ == "__main__":
    main()
