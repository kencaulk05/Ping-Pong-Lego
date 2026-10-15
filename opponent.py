"""
opponent.py -- who decides where the opponent's shots go.

The game talks to the opponent through two calls only:

    choose(ctx)               -> Shot     "where do I hit it, and how hard?"
    observe(ctx, shot, reward)            "here's how that shot worked out"

RuleOpponent (stage 2) uses a fixed tactic and ignores observe().
Stage 3 replaces it with a Q-learning opponent that implements the same two
methods, so nothing in pingpong_core.py has to change.

ctx is a plain dict the game fills in:
    level      the Level (opponent speed, aggression, ...)
    serving    True if this is a fresh serve rather than a return
    player_x   where you last hit from (normalized screen x), or None
    streak     your current streak
"""

import random
from dataclasses import dataclass

import config as C


@dataclass(frozen=True)
class Shot:
    zone: int      # index into config.ZONE_X: where the ball will land on your side
    fast: bool     # fast shots arrive in less time (config.FAST_SHOT_TIME_MULT)


class OpponentPolicy:
    def choose(self, ctx):
        raise NotImplementedError

    def observe(self, ctx, shot, reward):
        pass

    def status(self):
        return type(self).__name__


class RandomOpponent(OpponentPolicy):
    """Baseline for comparisons: any zone, always a slow shot."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, ctx):
        return Shot(self.rng.randrange(len(C.ZONE_X)), fast=False)


class RuleOpponent(OpponentPolicy):
    """Fixed tactic: usually aim at the zone farthest from where you just hit
    from (so you have to run), sometimes pick a zone at random, and hit a fast
    shot some of the time. How often is set by the level."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, ctx):
        n = len(C.ZONE_X)
        if ctx["serving"] or ctx["player_x"] is None:
            return Shot(self.rng.randrange(n), fast=False)

        level = ctx["level"]
        if self.rng.random() < level.aggression:
            # Tiny random tie-break so it doesn't always pick the same corner.
            zone = max(range(n), key=lambda z: abs(C.ZONE_X[z] - ctx["player_x"])
                                              + self.rng.random() * 0.05)
        else:
            zone = self.rng.randrange(n)
        return Shot(zone, fast=self.rng.random() < level.fast_prob)

    def status(self):
        return "rule-based"
