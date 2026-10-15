"""
pingpong_core.py -- the rules of the game, with NO display and NO hardware.

Feed it Inputs every frame, it hands back a list of event dicts. That's the
whole interface, which is what makes it possible to test the game headless and
to swap the simulated inputs for the real ones later:

    simulator today                      real thing later
    ---------------------------------    ------------------------------------
    mouse position        -> wrist       MediaPipe wrist landmark
    f / b / space keys    -> Swing       hub IMU + swing classifier
    1 / 2 / 3 keys        -> level_card  AprilTag id 1 / 2 / 3
    s key                 -> start_card  AprilTag id 0

One rally:
    IDLE --start--> COUNTDOWN --> INCOMING --hit--> RETURNING --opp returns--> INCOMING ...
                                      |                 |
                                    miss            winner (you beat the opponent)
                                      v                 v
                                   RESERVE  --------> RESERVE --> serve --> INCOMING

Streak = consecutive hits by YOU. Only a miss resets it; beating the opponent
(a "winner") keeps it going. The published score is the best streak.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import config as C
from opponent import OpponentPolicy, RuleOpponent, Shot


class Phase(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    INCOMING = "incoming"      # opponent's ball flying toward you
    RETURNING = "returning"    # your ball flying toward the opponent
    RESERVE = "reserve"        # short pause before the next serve


@dataclass
class Swing:
    kind: str = "flat"         # "forehand" | "backhand" | "flat"
    power: float = 0.5         # 0..1
    age: float = 0.0           # seconds since the swing actually happened, to
                               # compensate for sensor/BLE latency (0 = just now)


@dataclass
class Inputs:
    wrist: Optional[Tuple[float, float]] = None   # normalized (x, y); None = pose lost
    swings: List[Swing] = field(default_factory=list)
    level_card: Optional[int] = None              # AprilTag 1/2/3 seen this frame
    start_card: bool = False                      # AprilTag 0 seen this frame


@dataclass
class Ball:
    x0: float            # normalized x at the start of the flight
    x1: float            # normalized x where it lands
    T: float             # flight time in seconds
    outgoing: bool       # False = toward you, True = toward the opponent


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _lerp(a, b, t):
    return a + (b - a) * t


class Game:
    def __init__(self, level: int = 2, rng: Optional[random.Random] = None,
                 policy: Optional[OpponentPolicy] = None):
        self.rng = rng or random.Random()
        self.policy = policy or RuleOpponent(self.rng)

        self.level_id = level
        self.level = C.LEVELS[level]
        self.position_scale = 1.0         # the live game sets C.BODY_RING_SCALE when position comes from your body

        self.phase = Phase.IDLE
        self.phase_t = 0.0

        self.streak = 0
        self.best = 0
        self.hits = self.misses = self.winners = self.returns = 0

        self.opp_x = 0.5                 # opponent paddle position
        self.ball: Optional[Ball] = None
        self.ball_serial = 0             # increments for each new incoming ball
        self.ring_x = 0.5                # where the current incoming ball lands
        self.player_x: Optional[float] = None   # where you last hit from

        self.paused = False              # True while the wrist is lost mid-point
        self.pose_lost_time = 0.0
        self.last_outcome = ""           # human-readable line for the UI

        self._reserve_len = 0.0
        self._wrist_hist = deque()       # (time in this ball's flight, wrist), for latency compensation
        self._last_count = None
        self._last_swing_t = -1e9
        self._whiffed_early = False
        self._shot: Optional[Shot] = None
        self._ctx: Optional[dict] = None

    # ------------------------------------------------------------------ helpers

    @property
    def current_shot(self) -> Optional[Shot]:
        """The opponent's most recent shot (zone, fast)."""
        return self._shot

    @property
    def countdown_remaining(self) -> float:
        return max(0.0, C.COUNTDOWN_SECONDS - self.phase_t)

    def ball_position(self):
        """(x_norm, depth) for drawing. depth: 0 = far wall, 1 = your side.
        None when no ball is in flight."""
        if self.ball is None or self.phase not in (Phase.INCOMING, Phase.RETURNING):
            return None
        p = _clamp(self.phase_t / self.ball.T, 0.0, 1.0)
        x = _lerp(self.ball.x0, self.ball.x1, p)
        depth = (1.0 - p) if self.ball.outgoing else p
        return x, depth

    @property
    def hit_window(self) -> float:
        """+/- seconds a swing may be off the ball's arrival; set by the level card."""
        return getattr(self.level, "hit_window", C.HIT_WINDOW)

    @property
    def ring_radius(self) -> float:
        """How close your wrist must be to the ring center, in height units; the level scales it."""
        return C.RING_RADIUS * getattr(self.level, "ring_scale", 1.0) * self.position_scale

    def ring_distance(self, wrist) -> float:
        dx = (wrist[0] - self.ring_x) * C.ASPECT      # width units -> height units
        dy = wrist[1] - C.RING_Y
        return math.hypot(dx, dy)

    def in_ring(self, wrist) -> bool:
        return wrist is not None and self.ring_distance(wrist) <= self.ring_radius

    def set_level(self, level_id: int):
        self.level_id = level_id
        self.level = C.LEVELS[level_id]

    def reset_record(self):
        self.best = 0

    # ------------------------------------------------------------------- update

    def update(self, dt: float, inp: Inputs) -> List[dict]:
        ev: List[dict] = []

        # Level cards and the start card only mean something when idle, so
        # flashing a tag mid-rally can't change the difficulty by accident.
        if self.phase == Phase.IDLE:
            if inp.level_card in C.LEVELS and inp.level_card != self.level_id:
                self.set_level(inp.level_card)
                ev.append({"type": "level", "level": self.level_id, "name": self.level.name})
            if inp.start_card:
                self._enter(Phase.COUNTDOWN)
                self._last_count = None
                self.streak = 0
                ev.append({"type": "start", "level": self.level_id})
            return ev

        # No wrist = pose lost. Freeze the point instead of punishing you for
        # a tracking dropout; give up after POSE_LOST_TIMEOUT.
        if inp.wrist is None:
            if not self.paused:
                self.paused = True
                ev.append({"type": "pose_lost"})
            self.pose_lost_time += dt
            if self.pose_lost_time > C.POSE_LOST_TIMEOUT:
                self._abort(ev, "pose lost too long")
            return ev
        if self.paused:
            self.paused = False
            ev.append({"type": "pose_back"})
        self.pose_lost_time = 0.0

        self.phase_t += dt

        if self.phase == Phase.COUNTDOWN:
            n = math.ceil(self.countdown_remaining)
            if n != self._last_count and n > 0:
                self._last_count = n
                ev.append({"type": "countdown", "n": n})
            if self.phase_t >= C.COUNTDOWN_SECONDS:
                self._opponent_shot(ev, serving=True)

        elif self.phase == Phase.INCOMING:
            self._update_incoming(inp, ev)

        elif self.phase == Phase.RETURNING:
            self._update_returning(dt, ev)

        elif self.phase == Phase.RESERVE:
            if self.phase_t >= self._reserve_len:
                self._opponent_shot(ev, serving=True)

        return ev

    # ------------------------------------------------------------ phase logic

    def _enter(self, phase: Phase):
        self.phase = phase
        self.phase_t = 0.0

    def _opponent_shot(self, ev, serving: bool):
        ctx = {"level": self.level, "serving": serving,
               "player_x": self.player_x, "streak": self.streak}
        shot = self.policy.choose(ctx)
        self._shot, self._ctx = shot, ctx

        self.ring_x = C.ZONE_X[shot.zone]
        T = self.level.incoming_time * (C.FAST_SHOT_TIME_MULT if shot.fast else 1.0)
        self.ball = Ball(x0=self.opp_x, x1=self.ring_x, T=T, outgoing=False)
        self.ball_serial += 1
        self._last_swing_t = -1e9
        self._whiffed_early = False
        self._wrist_hist.clear()
        self._enter(Phase.INCOMING)
        ev.append({"type": "serve" if serving else "incoming",
                   "zone": shot.zone, "fast": shot.fast, "T": T})

    def _wrist_at(self, t, default):
        """Where the wrist was at time t of this ball's flight. A swing reported
        late (sensor latency) must be judged by where you were when you SWUNG,
        not where your hand is by the time the report arrives."""
        best = None
        for when, pos in self._wrist_hist:
            if when <= t:
                best = pos
            else:
                break
        if best is None:
            best = self._wrist_hist[0][1] if self._wrist_hist else default
        return best

    def _closest_before(self, t, default):
        """Closest the wrist came to the ring center during the POSITION_LOOKBACK seconds
        before time t (the swing's peak). Lining up and then swinging counts as in position,
        even though the swing itself carries the hand away."""
        ds = [self.ring_distance(pos) for when, pos in self._wrist_hist
              if t - C.POSITION_LOOKBACK <= when <= t and pos is not None]
        if not ds:
            return self.ring_distance(self._wrist_at(t, default))
        return min(ds)

    def _update_incoming(self, inp: Inputs, ev):
        T = self.ball.T
        w = self.hit_window
        self._wrist_hist.append((self.phase_t, inp.wrist))
        while self._wrist_hist and self._wrist_hist[0][0] < self.phase_t - 1.5:
            self._wrist_hist.popleft()

        for sw in inp.swings:
            t_s = self.phase_t - sw.age     # when the swing really happened
            if t_s - self._last_swing_t < C.SWING_COOLDOWN:
                ev.append({"type": "swing_ignored", "reason": "cooldown"})
                continue
            self._last_swing_t = t_s

            if t_s < T - w:
                self._whiffed_early = True
                ev.append({"type": "whiff", "reason": "too early", "kind": sw.kind})
                continue
            if t_s > T + w:
                ev.append({"type": "whiff", "reason": "too late", "kind": sw.kind})
                continue

            # In the timing window: now does your wrist agree you were in position?
            d = self._closest_before(t_s, inp.wrist)
            if d > self.ring_radius:
                self._miss(ev, "out of position", dist=round(d / self.ring_radius, 2),
                           ring_scale=getattr(self.level, "ring_scale", 1.0))
                return
            self._hit(ev, sw, d)
            return

        if self.phase_t > T + w + C.SWING_REPORT_DELAY:
            self._miss(ev, "too early" if self._whiffed_early else "no swing")

    def _hit(self, ev, sw: Swing, dist: float):
        accuracy = max(0.0, 1.0 - dist / self.ring_radius)      # 1 = dead center
        offset = C.SHOT_OFFSET.get(sw.kind, 0.0)
        jitter = self.rng.gauss(0.0, C.AIM_JITTER * (1.0 - accuracy))
        target = _clamp(self.ring_x + offset + jitter, 0.05, 0.95)
        power = _clamp(sw.power, 0.0, 1.0)
        ret_T = _lerp(C.RET_TIME_SLOW, C.RET_TIME_FAST, power)

        self.streak += 1
        self.hits += 1
        self.player_x = self.ring_x
        if self._shot is not None:
            self.policy.observe(self._ctx, self._shot, C.REWARD_PLAYER_RETURNED)

        ev.append({"type": "hit", "kind": sw.kind, "power": power, "accuracy": accuracy,
                   "target": target, "T": ret_T, "streak": self.streak})
        if self.streak > self.best:
            self.best = self.streak
            ev.append({"type": "record", "best": self.best})

        self.ball = Ball(x0=self.ring_x, x1=target, T=ret_T, outgoing=True)
        self._enter(Phase.RETURNING)

    def _update_returning(self, dt, ev):
        lvl = self.level
        # The opponent reads your shot, waits out its reaction time, then slides.
        move_t = max(0.0, min(dt, self.phase_t - lvl.opp_reaction))
        step = lvl.opp_speed * move_t
        self.opp_x += _clamp(self.ball.x1 - self.opp_x, -step, step)

        if self.phase_t >= self.ball.T:
            if abs(self.opp_x - self.ball.x1) <= lvl.opp_half_width:
                self.returns += 1
                ev.append({"type": "opp_return", "streak": self.streak})
                self._opponent_shot(ev, serving=False)
            else:
                self.winners += 1
                self.last_outcome = f"WINNER! The opponent couldn't reach it (streak {self.streak})"
                ev.append({"type": "winner", "streak": self.streak})
                self._reserve_len = C.RESERVE_AFTER_WINNER
                self._enter(Phase.RESERVE)

    def _miss(self, ev, reason: str, **extra):
        lost = self.streak
        self.streak = 0
        self.misses += 1
        if self._shot is not None:
            self.policy.observe(self._ctx, self._shot, C.REWARD_PLAYER_MISSED)
        self.last_outcome = f"MISS: {reason}" + (f" (streak of {lost} lost)" if lost else "")
        ev.append({"type": "miss", "reason": reason, "lost": lost, **extra})
        self._reserve_len = C.RESERVE_AFTER_MISS
        self._enter(Phase.RESERVE)

    def _abort(self, ev, reason: str):
        self.streak = 0
        self.paused = False
        self.pose_lost_time = 0.0
        self.ball = None
        self.last_outcome = f"Stopped: {reason}"
        ev.append({"type": "aborted", "reason": reason})
        self._enter(Phase.IDLE)
