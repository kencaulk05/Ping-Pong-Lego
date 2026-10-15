"""
qopponent.py -- an opponent that learns where YOU are weak (stage 3).

Same two-method interface as the rule-based opponent, so the game doesn't
change. It's a Q-table, exactly like the Silly Walks one:

    state   where you are standing: 5 zones, plus one "serving" state   (6 rows)
    action  where to aim (5 zones) x slow/fast                          (10 columns)
    reward  +1 when you miss its shot, -0.2 when you return it

Update after each shot, once we know what state it led to:

    Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a'))

Choice is epsilon-greedy: with probability epsilon pick a random action
(explore), otherwise the best-known one (exploit). Epsilon shrinks by a factor
of 0.98 after every shot, but never below eps_min.

Because rewards only arrive after you've reacted, the update for shot N is
applied at the start of shot N+1, when its next state (s') is known.

Guardrails, so a good learner doesn't make the game unplayable:
  * epsilon never reaches 0, so it keeps testing whether you've improved
  * fast shots only if the level allows them (config.Level.q_fast)
  * after `ease_off_after` misses in a row it serves you an easy, slow shot
    right at you, and doesn't learn from that forced shot
"""

import json
import random

import config as C
from opponent import OpponentPolicy, Shot

N_ZONES = len(C.ZONE_X)
SERVE_STATE = N_ZONES
N_STATES = N_ZONES + 1
N_ACTIONS = N_ZONES * 2          # action = zone * 2 + (1 if fast else 0)


def zone_of_x(x):
    """Nearest landing zone to a normalized screen x."""
    return min(range(N_ZONES), key=lambda z: abs(C.ZONE_X[z] - x))


def action_shot(a):
    return Shot(zone=a // 2, fast=bool(a % 2))


class QOpponent(OpponentPolicy):
    def __init__(self, rng=None, alpha=0.25, gamma=0.6, epsilon=0.6,
                 eps_decay=0.98, eps_min=0.10, ease_off_after=3, learning=True):
        self.rng = rng or random.Random()
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.eps_decay, self.eps_min = epsilon, eps_decay, eps_min
        self.ease_off_after = ease_off_after     # None disables the guardrail
        self.learning = learning

        self.q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]
        self.visits = [[0] * N_ACTIONS for _ in range(N_STATES)]
        self.shots = 0

        self._last = None        # (state, action) of the last shot the policy chose
        self._pending = None     # that shot's reward, waiting for the next state
        self._miss_run = 0       # how many shots in a row you've missed

    # --------------------------------------------------------- policy interface

    def choose(self, ctx):
        s = self._state(ctx)
        allowed = self._allowed(ctx)
        self._flush(s, allowed)          # finish learning from the previous shot

        if self.ease_off_after and self._miss_run >= self.ease_off_after:
            self._miss_run = 0
            self._last = None            # a forced shot teaches us nothing
            return self._easy_shot(ctx)

        if self.rng.random() < self.epsilon:
            a = self.rng.choice(allowed)
        else:
            best = max(self.q[s][a] for a in allowed)
            a = self.rng.choice([a for a in allowed if self.q[s][a] == best])

        if self.learning:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.shots += 1
        self._last = (s, a)
        return action_shot(a)

    def observe(self, ctx, shot, reward):
        self._miss_run = self._miss_run + 1 if reward >= C.REWARD_PLAYER_MISSED else 0
        if self._last is not None and action_shot(self._last[1]) == shot:
            self._pending = reward

    def status(self):
        return f"Q-learning: exploring {self.epsilon:.0%}, {self.shots} shots"

    # ----------------------------------------------------------------- internals

    def _state(self, ctx):
        if ctx["serving"] or ctx["player_x"] is None:
            return SERVE_STATE
        return zone_of_x(ctx["player_x"])

    def _allowed(self, ctx):
        if ctx["level"].q_fast:
            return list(range(N_ACTIONS))
        return [z * 2 for z in range(N_ZONES)]          # slow shots only

    def _flush(self, s_next, allowed_next):
        if self._last is not None and self._pending is not None and self.learning:
            s, a = self._last
            best_next = max(self.q[s_next][a2] for a2 in allowed_next)
            self.q[s][a] = ((1 - self.alpha) * self.q[s][a]
                            + self.alpha * (self._pending + self.gamma * best_next))
            self.visits[s][a] += 1
        self._pending = None

    def _easy_shot(self, ctx):
        zone = 2 if ctx["player_x"] is None else zone_of_x(ctx["player_x"])
        return Shot(zone, fast=False)

    # --------------------------------------------------------------- save / look

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"version": 1, "q": self.q, "visits": self.visits,
                       "epsilon": self.epsilon, "shots": self.shots}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        q = data["q"]
        if len(q) != N_STATES or any(len(row) != N_ACTIONS for row in q):
            raise ValueError("saved Q-table has a different shape; delete it and start fresh")
        self.q = q
        self.visits = data.get("visits", [[0] * N_ACTIONS for _ in range(N_STATES)])
        self.epsilon = data.get("epsilon", self.epsilon)
        self.shots = data.get("shots", 0)

    def describe(self, min_visits=3):
        """What it has learned, in words: the best action per state."""
        names = [f"zone {z}" for z in range(N_ZONES)]
        lines = []
        for s in range(N_STATES):
            seen = [a for a in range(N_ACTIONS) if self.visits[s][a] >= min_visits]
            who = "serving" if s == SERVE_STATE else f"you in zone {s}"
            if not seen:
                lines.append(f"  {who:<14} not enough data yet")
                continue
            a = max(seen, key=lambda a: self.q[s][a])
            shot = action_shot(a)
            lines.append(f"  {who:<14} -> aim {names[shot.zone]}, {'fast' if shot.fast else 'slow'}"
                         f"   (Q {self.q[s][a]:+.2f}, tried {self.visits[s][a]}x)")
        return "\n".join(lines)
