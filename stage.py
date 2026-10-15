"""
stage.py -- the look of the game: a crowd, a character at each end holding a racket, and the
celebrations. Drawing and animation only. It reads the game's events and state, and the game never
reads it, so it cannot change a score, a hit or a timing.

    the crowd        stands behind the far wall and down both sides, in perspective (nearer = bigger).
                     It sways at rest, cheers harder as your streak grows, groans when you miss, does a wave
                     every 5 hits in a row, and showers confetti on a new record.
    your character   seen from behind, standing on your tracked body (or behind your wrist with --track wrist); the
                     racket is in your real hand, wherever your arm is, and swings when you swing (a forehand sweeps across the body one way, a backhand
                     the other, a flat shot pushes forward). A green ring on the hand = you are in the ring.
    the net          across the middle of the court, with posts, a white tape and mesh. The ball arcs over it (with
                     a shadow on the court), and lands exactly on the ring: the arc is zero at both ends.
    the opponent     at the far end with a black racket; it slides to the ball and swings when it returns it.
                     A thin bar under its feet still shows how far it can reach (that is gameplay information).
    words            a short word from the crowd (WINNER! / NEW RECORD! / OH NO...) in place of the hint text.

Turn it all off with --plain (pingpong_sim.py and pingpong_live.py) for the original look.
"""

import math
import random

import cv2
import numpy as np

AA = cv2.LINE_AA

SHIRTS = [(90, 120, 230), (70, 180, 230), (200, 140, 60), (120, 200, 120), (180, 90, 180), (60, 200, 200),
          (220, 200, 90), (200, 110, 110), (150, 150, 240), (110, 160, 90)]
SKINS = [(170, 200, 235), (140, 180, 225), (110, 150, 200), (90, 120, 170), (70, 95, 135)]
HAIRS = [(30, 30, 35), (40, 60, 90), (60, 110, 170), (150, 190, 230), (25, 25, 25), (90, 90, 110)]
WOOD = (115, 170, 220)
RED_RUBBER = (45, 45, 205)
BLACK_RUBBER = (45, 45, 48)
CONFETTI = [(60, 190, 250), (90, 210, 110), (221, 119, 127), (74, 75, 226), (235, 235, 235), (220, 200, 80)]
GREEN, TEAL = (90, 210, 110), (117, 158, 29)
WAVE_EVERY = 5                                 # a wave at 5, 10, 15 ... hits in a row
NET_HEIGHT = 34                                # the net's height, in pixels at the NEAR side of the court (it shrinks with distance)
BALL_ARC = 50                                  # how high the ball rises at mid-flight, in pixels (0 at both ends)
SWING_SECONDS = 0.30
ARM_MAX = 170                                   # the longest the drawn arm may be, in pixels
SAME = object()                                 # draw_player: 'the hand is where the wrist/position is' (the classic look)


def _lerp(a, b, t):
    return a + (b - a) * t


def _ease(p):
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def blend_rect(img, p1, p2, color, alpha):
    """A translucent rectangle (clipped to the picture)."""
    x0, y0 = max(0, min(p1[0], p2[0])), max(0, min(p1[1], p2[1]))
    x1, y1 = min(img.shape[1], max(p1[0], p2[0])), min(img.shape[0], max(p1[1], p2[1]))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


def draw_racket(img, hand, angle_deg, length=62.0, rubber=RED_RUBBER, scale=1.0):
    """A ping-pong racket held at `hand`. angle 0 = pointing straight up, positive = leaning right.
    Returns the (x, y) of the blade's centre."""
    a = math.radians(angle_deg)
    ux, uy = math.sin(a), -math.cos(a)
    length *= scale
    handle = length * 0.42
    ry, rx = length * 0.34, length * 0.31
    base = (int(hand[0] + ux * handle), int(hand[1] + uy * handle))
    centre = (int(hand[0] + ux * (handle + ry * 0.92)), int(hand[1] + uy * (handle + ry * 0.92)))
    cv2.line(img, (int(hand[0]), int(hand[1])), base, WOOD, max(3, int(length * 0.075)), AA)      # handle
    cv2.ellipse(img, centre, (int(rx + 3), int(ry + 3)), angle_deg, 0, 360, WOOD, -1, AA)          # wooden rim
    cv2.ellipse(img, centre, (int(rx), int(ry)), angle_deg, 0, 360, rubber, -1, AA)                # rubber
    cv2.ellipse(img, centre, (int(rx * 0.55), int(ry * 0.55)), angle_deg, 200, 290,                # a gloss highlight
                tuple(min(255, c + 70) for c in rubber), 2, AA)
    return centre


class Stage:
    def __init__(self, width, height, project, right_handed=True, seed=11):
        self.w, self.h, self.project = width, height, project
        self.right_handed = right_handed
        self.rng = random.Random(seed)
        self._crowd = None                      # laid out on first use (the ring height can be set after import)
        self.energy = 0.0                       # how worked up the crowd is, 0..1
        self.groan_until = 0.0
        self.wave_t0 = None
        self.word = None                        # (text, color, until)
        self.confetti = []                      # [x, y, vx, vy, color, size, spin, born]
        self.pops = []                          # impact flashes: start times
        self.swing = None                       # (start time, kind, strength) of YOUR racket
        self.opp_swing = None                   # start time of the opponent's racket
        self.now = 0.0
        self._last = None
        self._body_x = None

    # ------------------------------------------------------------------- events

    def handle(self, events, now):
        for e in events:
            t = e["type"]
            if t == "hit":
                self.swing = (now, e.get("kind", "flat"), 0.6 + 0.4 * e.get("power", 0.5))
                self.pops.append(now)
                self.energy = min(1.0, self.energy + 0.12 + 0.18 * e.get("power", 0.5))
                n = e.get("streak", 0)
                if n and n % WAVE_EVERY == 0:
                    self.wave_t0 = now
                    self._say(f"{n} IN A ROW!", (60, 190, 250), now, 1.4)
            elif t == "whiff":
                self.swing = (now, e.get("kind", "flat"), 0.5)
            elif t == "miss":
                if e.get("reason") == "out of position":
                    self.swing = (now, "flat", 0.5)           # you did swing, from the wrong place
                self.groan_until = now + 1.0
                self.energy *= 0.25
                self._say("OH NO...", (110, 110, 235), now, 1.0)
            elif t == "winner":
                self.energy = 1.0
                self.wave_t0 = now
                self._burst(24, now)
                self._say("WINNER!", (60, 190, 250), now, 1.4)
            elif t == "record":
                self.energy = 1.0
                self._burst(90, now)
                self._say("NEW RECORD!", (90, 210, 110), now, 2.0)
            elif t in ("opp_return", "serve", "incoming"):
                self.opp_swing = now
            elif t == "start":
                self.energy = max(self.energy, 0.45)
                self._say("LET'S GO!", (235, 235, 235), now, 1.2)
            elif t == "aborted":
                self.energy = 0.0

    def _say(self, text, color, now, seconds):
        self.word = (text, color, now + seconds)

    def current_word(self):
        if self.word and self.now < self.word[2]:
            return self.word[0], self.word[1]
        return None

    def _burst(self, n, now):
        for _ in range(n):
            self.confetti.append([self.rng.uniform(0, self.w), self.rng.uniform(44, 130), self.rng.uniform(-70, 70),
                                  self.rng.uniform(20, 140), self.rng.choice(CONFETTI), self.rng.uniform(3.0, 6.5),
                                  self.rng.uniform(-9, 9), now])

    # ---------------------------------------------------------------- the crowd

    def _layout(self):
        """Two rows behind the far wall, and rows down both sides that shrink with distance."""
        rng = random.Random(5)
        people = []

        def person(x, y, size):
            return (int(x), int(y), size, rng.uniform(0, 6.28), rng.uniform(0.55, 1.25), rng.choice(SHIRTS),
                    rng.choice(SKINS), rng.choice(HAIRS))

        y_far = self.project(0.5, 0)[1]
        for row, y in enumerate((y_far - 38, y_far - 19)):                       # behind the far wall
            x = 12 + (row * 13)
            while x < self.w - 10:
                people.append(person(x + rng.uniform(-3, 3), y + rng.uniform(-2, 2), 7.0 + row * 0.8))
                x += 27
        for d in (0.06, 0.18, 0.30, 0.42, 0.54, 0.66, 0.78):                     # down the sides, in perspective
            y = _lerp(y_far, self.project(0.5, 1)[1], d)
            size = _lerp(6.8, 12.5, d)
            for side, edge in ((-1, self.project(0, d)[0]), (1, self.project(1, d)[0])):
                x = edge + side * size * 2.1
                while size < x < self.w - size:
                    people.append(person(x + rng.uniform(-2, 2), y + rng.uniform(-3, 3), size))
                    x += side * size * 3.0
        people.sort(key=lambda p: p[1])                                          # far to near
        return people

    def draw_stands(self, img, game, now):
        dt = 0.0 if self._last is None else min(0.1, max(0.0, now - self._last))
        self._last, self.now = now, now
        base = min(0.55, 0.05 * game.streak)                                      # a streak warms the crowd up
        self.energy = max(base, self.energy - dt * 0.55)
        if self._crowd is None:
            self._crowd = self._layout()
        groaning = now < self.groan_until
        wave_front = None
        if self.wave_t0 is not None:
            wave_front = -80 + (now - self.wave_t0) * 620.0
            if wave_front > self.w + 120:
                self.wave_t0 = None
        for x, y, size, phase, eager, shirt, skin, hair in self._crowd:
            lift = 0.0
            if wave_front is not None:
                lift = size * 1.0 * max(0.0, 1.0 - abs(x - wave_front) / 120.0)
            e = self.energy * eager
            bob = max(0.0, math.sin(phase + now * (2.5 + 6.0 * self.energy))) * size * 0.6 * e
            sway = math.sin(phase + now * 0.9) * 0.6 * (1 - self.energy)
            droop = size * 0.4 if groaning else 0.0
            self._spectator(img, x + sway, y - lift - bob + droop, size, shirt, skin, hair,
                            arms=(e > 0.5 or lift > size * 0.45) and not groaning, cheer=e > 0.35 or lift > 0,
                            groan=groaning)

    @staticmethod
    def _spectator(img, x, y, size, shirt, skin, hair, arms, cheer, groan):
        x, cy, r = int(x), int(y), max(3, int(size))
        body_y = cy + int(size * 1.4)
        if arms:
            for sgn in (-1, 1):
                cv2.line(img, (x + sgn * int(size * 0.95), body_y), (x + sgn * int(size * 1.5), cy - int(size * 0.8)),
                         skin, max(2, int(size * 0.34)), AA)
        cv2.ellipse(img, (x, body_y + int(size * 0.25)), (int(size * 1.3), int(size * 1.0)), 0, 0, 360, shirt, -1, AA)
        cv2.circle(img, (x, cy), r, skin, -1, AA)
        cv2.ellipse(img, (x, cy), (r, r), 0, 190, 350, hair, -1, AA)
        if size >= 8.5:                                                           # a little face on the nearer ones
            ey = cy + int(size * 0.12)
            for sgn in (-1, 1):
                cv2.circle(img, (x + sgn * int(size * 0.36), ey), 1, (30, 30, 35), -1, AA)
            my = cy + int(size * 0.5)
            if groan:
                cv2.ellipse(img, (x, my + 2), (int(size * 0.3), 2), 0, 180, 360, (40, 40, 60), 1, AA)
            elif cheer:
                cv2.ellipse(img, (x, my), (int(size * 0.3), int(size * 0.24)), 0, 0, 360, (40, 40, 90), -1, AA)
            else:
                cv2.line(img, (x - int(size * 0.25), my), (x + int(size * 0.25), my), (60, 60, 90), 1, AA)

    def draw_confetti(self, img, now):
        if not self.confetti:
            return
        dt = 0.033
        keep = []
        for p in self.confetti:
            age = now - p[7]
            p[3] += 280 * dt
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            if age < 3.2 and p[1] < self.h:
                a = age * p[6]
                c, s = math.cos(a), math.sin(a)
                hw, hh = p[5], p[5] * 0.55
                pts = np.array([(p[0] + c * hw - s * hh, p[1] + s * hw + c * hh), (p[0] - c * hw - s * hh, p[1] - s * hw + c * hh),
                                (p[0] - c * hw + s * hh, p[1] - s * hw - c * hh), (p[0] + c * hw + s * hh, p[1] + s * hw - c * hh)], np.int32)
                cv2.fillConvexPoly(img, pts, p[4], AA)
                keep.append(p)
        self.confetti = keep[-160:]

    # --------------------------------------------------------------- the players

    def _swing_pose(self, now, swing, side):
        """(racket angle, size scale, forward push in pixels, progress 0..1 or None) for your racket."""
        rest = -14.0 * side
        if swing is None or now - swing[0] > SWING_SECONDS:
            return rest, 1.0, 0.0, None
        p = (now - swing[0]) / SWING_SECONDS
        kind, strength = swing[1], swing[2]
        e = _ease(p)
        if kind == "forehand":
            ang = _lerp(55.0 * side, -75.0 * side, e)               # sweeps from the outside, across the body
        elif kind == "backhand":
            ang = _lerp(-60.0 * side, 75.0 * side, e)
        else:                                                         # flat: a push forward
            ang = _lerp(rest, rest * 0.2, e)
        thrust = math.sin(p * math.pi) * (16 if kind == "flat" else 6) * strength
        return ang, 1.0 + (0.28 * math.sin(p * math.pi) if kind == "flat" else 0.1 * math.sin(p * math.pi)), thrust, p

    def draw_opponent(self, img, game, now):
        cx, y0 = self.project(game.opp_x, 0)
        feet = y0 + 16
        s = 0.92
        half = int(game.level.opp_half_width * self.w * 0.55)
        blend_rect(img, (cx - half, feet + 2), (cx + half, feet + 7), (74, 75, 226), 0.45)         # its reach
        cv2.rectangle(img, (cx - half, feet + 2), (cx + half, feet + 7), (235, 235, 235), 1)
        cv2.ellipse(img, (cx, feet + 1), (int(20 * s), 4), 0, 0, 360, (15, 15, 15), -1, AA)           # shadow
        for sgn in (-1, 1):                                                                         # legs
            cv2.line(img, (cx + sgn * int(5 * s), feet - int(15 * s)), (cx + sgn * int(7 * s), feet - 1), (45, 45, 60), max(2, int(5 * s)), AA)
        cv2.ellipse(img, (cx, feet - int(28 * s)), (int(13 * s), int(15 * s)), 0, 0, 360, (74, 75, 226), -1, AA)   # shirt
        cv2.circle(img, (cx, feet - int(49 * s)), int(9 * s), SKINS[1], -1, AA)
        cv2.ellipse(img, (cx, feet - int(49 * s)), (int(9 * s), int(9 * s)), 0, 190, 350, HAIRS[0], -1, AA)
        for sgn in (-1, 1):
            cv2.circle(img, (cx + sgn * int(3.4 * s), feet - int(48 * s)), 1, (30, 30, 35), -1, AA)
        # the racket arm, on the screen-left (its right hand, facing us); it swings when the opponent returns the ball
        shoulder = (cx - int(11 * s), feet - int(36 * s))
        towards = 1.0 if game.ring_x >= game.opp_x else -1.0
        p = None if self.opp_swing is None else (now - self.opp_swing) / 0.32
        ang = -28.0                                              # held out to its side, blade clear of the head
        if p is not None and 0 <= p <= 1:
            ang = _lerp(-60.0 * towards, 62.0 * towards, _ease(p))
        hand = (cx - int(33 * s), feet - int(24 * s))
        cv2.line(img, shoulder, hand, (74, 75, 226), max(3, int(6 * s)), AA)
        cv2.circle(img, hand, max(3, int(4 * s)), SKINS[1], -1, AA)
        draw_racket(img, hand, ang, length=52 * s, rubber=BLACK_RUBBER)

    def draw_player(self, img, game, wrist, now, phase_incoming, hand=SAME):
        """Your character, from behind. `phase_incoming`: a ball is coming.

        Classic (hand left as SAME): `wrist` is where your hand is, and the character stands behind it.
        Body tracking: `wrist` is where your BODY is (the game's position) and `hand` is where your hand is (or None if
        it can't be seen): the character stands on your body and the racket is in your real hand, free to swing."""
        side = 1.0 if self.right_handed else -1.0
        hand_pos = wrist if hand is SAME else hand
        if wrist is None:
            target_x, ref_y = (self._body_x if self._body_x is not None else self.w * 0.5), self.h * 0.62
        else:
            target_x = wrist[0] * self.w - (side * 30 if hand is SAME else 0)
            ref_y = wrist[1] * self.h
        hand = None if hand_pos is None else (hand_pos[0] * self.w, hand_pos[1] * self.h)
        dt = 0.033
        self._body_x = target_x if self._body_x is None else self._body_x + (target_x - self._body_x) * min(1.0, dt * 7)
        by = int(min(510, max(330, ref_y + 84)))
        bx = int(self._body_x)
        cv2.ellipse(img, (bx, by + 40), (46, 11), 0, 0, 360, (15, 15, 15), -1, AA)                  # shadow
        cv2.line(img, (bx - int(side * 36), by + 2), (bx - int(side * 44), by + 44), (200, 150, 90), 11, AA)   # the other arm, hanging
        cv2.circle(img, (bx - int(side * 44), by + 47), 6, SKINS[1], -1, AA)
        cv2.ellipse(img, (bx, by + 8), (42, 30), 0, 0, 360, (200, 150, 90), -1, AA)                 # shoulders (a teal-blue shirt)
        cv2.ellipse(img, (bx, by + 2), (42, 30), 0, 200, 340, (230, 190, 130), 2, AA)
        cv2.line(img, (bx, by + 6), (bx, by + 36), (230, 190, 130), 3, AA)                          # a stripe down the back
        cv2.circle(img, (bx, by - 30), 17, HAIRS[2], -1, AA)                                       # head, seen from behind
        cv2.ellipse(img, (bx, by - 24), (13, 9), 0, 0, 360, SKINS[1], -1, AA)                      # neck
        cv2.circle(img, (bx, by - 32), 17, HAIRS[2], -1, AA)
        if hand is None:
            draw_racket(img, (bx + side * 36, by + 14), -14.0 * side, length=80, rubber=RED_RUBBER)
            return
        shoulder = (bx + int(side * 34), by + 2)
        hx, hy = int(hand[0]), int(hand[1])
        reach = math.hypot(hx - shoulder[0], hy - shoulder[1])
        if reach > ARM_MAX:                                                       # an arm only reaches so far
            hx = int(shoulder[0] + (hx - shoulder[0]) * ARM_MAX / reach)
            hy = int(shoulder[1] + (hy - shoulder[1]) * ARM_MAX / reach)
        ang, scale, thrust, p = self._swing_pose(now, self.swing, side)
        cv2.line(img, shoulder, (hx, hy), (200, 150, 90), 11, AA)                                  # the arm, to where your wrist is
        if p is not None:                                                                          # a swoosh behind the racket
            tip_pts = []
            for k in range(0, 8):
                q = max(0.0, p - 0.05 * (8 - k))
                a2, _, _, _ = self._swing_pose(now - (p - q) * SWING_SECONDS, self.swing, side)
                a2r = math.radians(a2)
                tip_pts.append((int(hx + math.sin(a2r) * 88 * scale), int(hy - math.cos(a2r) * 88 * scale)))
            for k in range(len(tip_pts) - 1):
                v = int(110 + 120 * k / 7)
                cv2.line(img, tip_pts[k], tip_pts[k + 1], (v, v, min(255, v + 25)), 3, AA)
        draw_racket(img, (hx, hy - int(thrust)), ang, length=84, rubber=RED_RUBBER, scale=scale)
        inside = phase_incoming and game.in_ring(wrist)
        cv2.circle(img, (hx, hy), 11, SKINS[1], -1, AA)                                            # the hand
        cv2.circle(img, (hx, hy), 13, GREEN if inside else TEAL, 3, AA)                            # green = in the ring

    def draw_net(self, img):
        """The net across the middle of the court: two posts, a white top tape, and see-through mesh. Drawn after the
        opponent (who is behind it) and before the ball (which flies over it)."""
        (xl, ym), (xr, _) = self.project(0, 0.5), self.project(1, 0.5)
        h = int(NET_HEIGHT * _lerp(0.55, 1.0, 0.5))                      # shrunk by the same perspective as the court
        x0, x1, top = xl - 2, xr + 2, ym - h
        blend_rect(img, (x0, ym), (x1, ym + 6), (10, 10, 10), 0.35)      # its shadow on the court
        mesh_roi = img[top:ym, x0:x1]
        mesh = mesh_roi.copy()
        for x in range(0, x1 - x0, 9):                                   # the mesh: a grid of thin white lines
            cv2.line(mesh, (x, 0), (x, h), (235, 235, 235), 1, AA)
        for y in range(3, h, 7):
            cv2.line(mesh, (0, y), (x1 - x0, y), (235, 235, 235), 1, AA)
        mesh_roi[:] = cv2.addWeighted(mesh, 0.5, mesh_roi, 0.5, 0)
        cv2.line(img, (x0, top), (x1, top), (245, 245, 245), 3, AA)      # the white tape along the top
        cv2.line(img, (x0, top + 3), (x1, top + 3), (170, 170, 170), 1, AA)
        for px in (x0, x1):                                              # the posts, with a clamp at the base
            cv2.rectangle(img, (px - 3, top - 5), (px + 3, ym + 3), (190, 190, 195), -1, AA)
            cv2.rectangle(img, (px - 3, top - 5), (px + 3, ym + 3), (60, 60, 65), 1, AA)
            cv2.rectangle(img, (px - 5, ym - 4), (px + 5, ym + 3), (50, 50, 55), -1, AA)

    def draw_pops(self, img, game, now):
        """A starburst where the ball met your racket."""
        keep = []
        for t0 in self.pops:
            p = (now - t0) / 0.30
            if 0 <= p <= 1:
                cx, cy = int(game.ring_x * self.w), self.project(0.5, 1)[1]
                e = _ease(p)
                col = tuple(int(_lerp(c, 60, p * 0.8)) for c in (150, 235, 255))
                for k in range(12):
                    a = k * math.pi / 6 + 0.25
                    r0 = _lerp(16, 46, e)
                    r1 = r0 + (22 if k % 2 == 0 else 12) * (1 - p)
                    cv2.line(img, (cx + int(math.cos(a) * r0), cy + int(math.sin(a) * r0)),
                             (cx + int(math.cos(a) * r1), cy + int(math.sin(a) * r1)), col, max(1, int(4 * (1 - p))), AA)
                keep.append(t0)
        self.pops = keep
