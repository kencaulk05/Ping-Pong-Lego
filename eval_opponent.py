"""
eval_opponent.py -- does the Q-learning opponent actually learn anything?

Plays a scripted player with a KNOWN weakness against each opponent and
reports how often the player misses, in blocks of 100 shots, averaged over
several random seeds.

The player: skill 0.95, but zone 0 is hard for it (70% extra miss chance),
zone 1 a little (25%), zone 4 slightly (15%), and fast shots cost another 30%.
So the best possible opponent aims fast at zone 0. If the Q-learner works, its
miss-rate should climb well above the random and rule-based opponents as it
discovers that.

    python eval_opponent.py                 # 5 seeds x 800 shots
    python eval_opponent.py --seeds 10 --shots 1500
"""

import argparse
import random

from bots import BotPlayer
from opponent import RandomOpponent, RuleOpponent
from pingpong_core import Game
from qopponent import QOpponent

DT = 1 / 30
WEAK = dict(skill=0.95, weak_zones={0: 0.70, 1: 0.25, 4: 0.15}, fast_penalty=0.30)


def make(kind, rng, ease_off):
    if kind == "random":
        return RandomOpponent(rng)
    if kind == "rule":
        return RuleOpponent(rng)
    return QOpponent(rng, ease_off_after=ease_off)


def play(kind, seed, shots, level=2, ease_off=3):
    rng = random.Random(seed)
    policy = make(kind, rng, ease_off)
    game = Game(level=level, rng=rng, policy=policy)
    bot = BotPlayer(rng=random.Random(seed + 1000), level=level, **WEAK)
    outcomes = []                      # 1 = the player missed this shot, 0 = returned it
    while len(outcomes) < shots:
        for e in game.update(DT, bot.act(game, DT)):
            if e["type"] == "miss":
                outcomes.append(1)
            elif e["type"] == "hit":
                outcomes.append(0)
    return outcomes[:shots], policy


def block_means(outcomes, size=100):
    return [100.0 * sum(outcomes[i:i + size]) / size for i in range(0, len(outcomes), size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shots", type=int, default=800)
    ap.add_argument("--level", type=int, default=2)
    args = ap.parse_args()

    runs = {
        "random": ("random", None),
        "rule-based": ("rule", None),
        "Q (guardrail off)": ("q", None),
        "Q (guardrail on)": ("q", 3),
    }
    blocks = args.shots // 100
    print(f"Player misses per block of 100 shots (%), mean of {args.seeds} seeds, level {args.level}")
    print(f"{'opponent':<20}" + "".join(f"{f'{i*100+1}-{i*100+100}':>10}" for i in range(blocks)))
    last_q = None
    for label, (kind, ease) in runs.items():
        per_seed = []
        for seed in range(args.seeds):
            out, policy = play(kind, seed, args.shots, args.level, ease)
            per_seed.append(block_means(out))
            if kind == "q" and ease is None and seed == 0:
                last_q = policy
        means = [sum(col) / len(col) for col in zip(*per_seed)]
        print(f"{label:<20}" + "".join(f"{m:>10.0f}" for m in means))

    print("\nWhat the Q-learner (guardrail off, seed 0) settled on:")
    print(last_q.describe())


if __name__ == "__main__":
    main()
