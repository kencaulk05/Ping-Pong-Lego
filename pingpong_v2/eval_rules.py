"""
eval_rules.py -- which ACCEPT RULE should the game use for a swing?

    python eval_rules.py

The model gives four probabilities per event: forehand, backhand, flat, other.
Two ways to decide "that was a swing":

  kind rule    (what swing_check.py used)  the single most likely class is a swing
               kind AND its probability >= the cutoff (0.6).
  swing rule   P(forehand) + P(backhand) + P(flat) = 1 - P(other) >= the cutoff,
               and the kind is the most likely of the three swing classes.

The swing rule asks "was that a swing at all?", which is the question that decides
a hit or a miss in the game; the kind only steers the shot. A swing the kind rule
throws away because it can't decide between forehand and flat is still a swing.

Every run is scored by a model trained on swings.csv plus the OTHER runs, never on
itself (leave-one-run-out), so these are numbers on unseen data.

Counts per run:
  right   the swing was accepted and named correctly
  wrong   accepted, but the wrong kind (in the game: still a hit, aimed differently)
  lost    rejected or not detected (in the game: a miss)
  FA      false alarms: swings accepted during the 45 s of random movement
  breakFA swings accepted during the breaks between attempts, in 'waiting' runs
          (you were holding still, so these are false alarms too)
"""

import warnings

import numpy as np

import swing_detect as sd
from eval_runs import load_run_events, training_data
from train_swing_classifier import make_pipeline, parse_run_spec

warnings.filterwarnings("ignore")

RUNS = ["guided-20261012-023715", "guided-20261012-024425", "guided-20261012-031646:waiting",
        "guided-20261012-034425:waiting", "guided-20261012-061026:waiting", "guided-20261012-061454:waiting"]
MIN_PEAK = 330.0
PRE, POST = 0.4, 2.2


def decide(classes, p, rule, cutoff):
    p = dict(zip(classes, p))
    swing = {k: p[k] for k in sd.SWING_KINDS}
    if rule == "kind":
        top = max(p, key=p.get)
        return (top, True) if top in sd.SWING_KINDS and p[top] >= cutoff else (top, False)
    kind = max(swing, key=swing.get)
    return kind, (1.0 - p["other"]) >= cutoff


def score(evs, probs, classes, attempts, window, rule, cutoff):
    dec = []
    for e, p in zip(evs, probs):
        kind, ok = decide(classes, p, rule, cutoff)
        dec.append((e, kind, ok and e.peak >= MIN_PEAK))
    right = wrong = lost = 0
    used = set()
    for phase, _, go in attempts:
        near = [(e, k, ok) for e, k, ok in dec if go - PRE <= e.t_peak <= go + POST]
        used.update(id(e) for e, _, _ in near)
        if not near:
            lost += 1
            continue
        e, k, ok = max(near, key=lambda d: d[0].peak)       # the biggest event is the verdict
        if not ok:
            lost += 1
        elif k == phase:
            right += 1
        else:
            wrong += 1
    lo, hi = window
    fa = sum(1 for e, k, ok in dec if ok and lo <= e.t_peak <= hi)
    brk = sum(1 for e, k, ok in dec if ok and id(e) not in used and not (lo <= e.t_peak <= hi))
    return right, wrong, lost, fa, brk


def main():
    cols = [i for i, n in enumerate(sd.FEATURE_NAMES) if n not in sd.CONTEXT_FEATURES]
    rules = [("kind", 0.6), ("kind", 0.5), ("swing", 0.5), ("swing", 0.9), ("swing", 0.95)]
    totals = {r: np.zeros(5, int) for r in rules}
    print(f"{'run (never seen by its model)':<34}" + "".join(f"{r[0]:>6} {r[1]:.2f}".rjust(22) for r in rules))
    print(" " * 34 + "".join("right/wrong/lost FA brk".rjust(22) for _ in rules))
    for spec in RUNS:
        base, style = parse_run_spec(spec)
        evs, attempts, window = load_run_events(base)
        X, y = training_data("swings.csv", RUNS, skip=base)
        pipe = make_pipeline().fit(np.asarray(X)[:, cols], y)
        probs = pipe.predict_proba(np.asarray([e.features for e in evs])[:, cols]) if evs else []
        row = f"{base[-6:]} ({style})".ljust(34)
        for r in rules:
            s = score(evs, probs, list(pipe.classes_), attempts, window, *r)
            if style != "waiting":          # continuous runs: you were swinging in the breaks on purpose
                s = s[:4] + (0,)
            totals[r] += s
            brk = s[4] if style == "waiting" else "-"
            row += f"{s[0]:>2}/{s[1]}/{s[2]:<2} {s[3]:>2} {brk:>3}".rjust(22)
        print(row)
    print("-" * (34 + 22 * len(rules)))
    row = f"TOTAL ({30 * len(RUNS)} swings; breaks: waiting runs only)".ljust(34)
    for r in rules:
        t = totals[r]
        row += f"{t[0]:>3}/{t[1]}/{t[2]:<2} {t[3]:>2} {t[4]:>3}".rjust(22)
    print(row)
    print()
    for r in rules:
        t = totals[r]
        n = t[0] + t[1] + t[2]
        print(f"  {r[0]:>5} rule, cutoff {r[1]:.2f}: accepted as a swing {100 * (t[0] + t[1]) / n:5.1f}%   "
              f"named right {100 * t[0] / n:5.1f}%   lost {t[2]:>2}   false alarms {t[3]} random + {t[4]} breaks")


if __name__ == "__main__":
    main()
