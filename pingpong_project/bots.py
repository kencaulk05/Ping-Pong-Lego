"""
bots.py -- a scripted stand-in for a human player.

It peeks at the game's internals (where the ring is, when the ball lands), so
it is NOT a fair opponent; it exists so we can (a) test the rules headless,
(b) tune difficulty numbers by running thousands of simulated balls, and
(c) show a full exchange in the window with --bot.

skill is the chance each ball is played correctly. On a failure it picks one
of: no swing, swing way too early, or swing from the wrong place.
"""

import math
import random

import config as C
from pingpong_core import Inputs, Phase, Swing

REST = (0.5, 0.88)   # where the "hand" idles between balls


class BotPlayer:
    def __init__(self, skill=1.0, rng=None, wrist_speed=3.0, level=None, kinds=None,
                 weak_zones=None, fast_penalty=0.0):
        self.skill = skill
        # A built-in weakness, so a learning opponent has something to find:
        # weak_zones = {zone: extra chance of missing a ball that lands there},
        # fast_penalty = extra chance of missing any fast shot.
        self.weak_zones = weak_zones or {}
        self.fast_penalty = fast_penalty
        self.rng = rng or random.Random()
        self.wrist_speed = wrist_speed      # screen-heights per second, roughly
        self.level = level                  # if set, flash this level card once while idle
        self.kinds = kinds or ["forehand", "backhand", "flat"]
        self.wx, self.wy = REST
        self._seen_serial = 0
        self._plan = None
        self._swung = True
        self._sent_level = False

    def act(self, game, dt) -> Inputs:
        inp = Inputs(wrist=(self.wx, self.wy))

        if game.phase == Phase.IDLE:
            if self.level and not self._sent_level:
                inp.level_card = self.level
                self._sent_level = True
            else:
                inp.start_card = True
            self._move(REST, dt)
            inp.wrist = (self.wx, self.wy)
            return inp

        # A new incoming ball: decide up front how this one will go.
        if game.phase == Phase.INCOMING and game.ball_serial != self._seen_serial:
            self._seen_serial = game.ball_serial
            self._plan = self._make_plan(game)
            self._swung = False

        if game.phase == Phase.INCOMING and self._plan:
            target = (self._plan["wrist_x"], self._plan["wrist_y"])
        else:
            target = REST
        self._move(target, dt)
        inp.wrist = (self.wx, self.wy)

        if (game.phase == Phase.INCOMING and self._plan and not self._swung
                and self._plan["swings"] and game.phase_t >= self._plan["t_swing"]):
            self._swung = True
            inp.swings.append(Swing(kind=self.rng.choice(self.kinds),
                                    power=self.rng.uniform(0.3, 1.0)))
        return inp

    # ----------------------------------------------------------------- internals

    def _make_plan(self, game):
        T = game.ball.T
        ring_x, ring_y = game.ring_x, C.RING_Y
        plan = {"wrist_x": ring_x + self.rng.uniform(-0.02, 0.02),
                "wrist_y": ring_y + self.rng.uniform(-0.03, 0.03),
                "swings": True,
                "t_swing": T + max(-0.12, min(0.12, self.rng.gauss(0, 0.05)))}
        p_ok = self.skill
        shot = game.current_shot
        if shot is not None:
            p_ok *= 1.0 - self.weak_zones.get(shot.zone, 0.0)
            if shot.fast:
                p_ok *= 1.0 - self.fast_penalty
        if self.rng.random() >= p_ok:
            fail = self.rng.choice(["no_swing", "early", "wrong_place"])
            if fail == "no_swing":
                plan["swings"] = False
            elif fail == "early":
                plan["t_swing"] = T - 0.6
            else:
                plan["wrist_x"] = ring_x + (0.3 if ring_x < 0.5 else -0.3)
        return plan

    def _move(self, target, dt):
        dx, dy = target[0] - self.wx, target[1] - self.wy
        dist = math.hypot(dx * C.ASPECT, dy)
        step = self.wrist_speed * dt
        if dist <= step or dist == 0:
            self.wx, self.wy = target
        else:
            f = step / dist
            self.wx += dx * f
            self.wy += dy * f
