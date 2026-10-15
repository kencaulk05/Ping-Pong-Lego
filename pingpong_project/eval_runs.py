"""
eval_runs.py -- how well does the recognizer do on a run it has NEVER seen?

    python eval_runs.py swings.csv \\
        --run guided-20261012-023715 \\
        --run guided-20261012-024425 \\
        --run guided-20261012-031646:waiting

For each saved guided run, this trains a model on everything EXCEPT that run (the
original recording plus the other runs) and scores the run exactly the way
swing_check.py does. That is the number to believe, because a model that has seen a
run will do well on it. Training accuracy and ordinary cross-validation can't show
what actually went wrong earlier (a model that read a still hand as a different kind
of swing), since every example in them has the same kind of context.

For a "waiting" run it adds a second line: the model also gets 4/5 of that run's own
attempts and tests on the remaining 1/5, five times. That says how much a short
calibration in your own style would help. It is optimistic compared with a brand-new
session, because neighbouring attempts in one run share your grip and pace.

The minimum swing size applies to everything (default 330, or --min-peak).
"""

import argparse
import collections
import warnings

import numpy as np

import swing_detect as sd
from swing_check import SWING_KINDS, resolve_attempt, score
from train_swing_classifier import (build_events, events_from_run, load_takes, make_pipeline, parse_run_spec)

warnings.filterwarnings("ignore")


def load_run_events(base):
    """(detector events, attempts, random window) for a saved run."""
    from rescore import load_run
    readings, attempts, window = load_run(base)
    det = sd.SwingDetector()
    evs = []
    for t, v in readings:
        evs += det.push(t, v)
    return evs, attempts, window


def classify(pipe, cols, features):
    classes = list(pipe.classes_)
    p = pipe.predict_proba(np.asarray(features)[cols].reshape(1, -1))[0]
    k = int(p.argmax())
    return classes[k], float(p[k])


def score_run(evs, attempts, window, pipe, cols, min_peak, min_conf=0.6, only_attempts=None, only_chunks=None):
    """Score a run with a model. only_* restrict which attempts / random-phase chunks count (for folds)."""
    events = []
    for e in evs:
        kind, conf = classify(pipe, cols, e.features)
        recognized = kind in SWING_KINDS and conf >= min_conf and e.peak >= min_peak
        events.append(dict(t_peak=e.t_peak, peak=e.peak, recognized=recognized, kind=kind, conf=conf))
    results = [(p, resolve_attempt(events, g)) for i, (p, n, g) in enumerate(attempts)
               if only_attempts is None or i in only_attempts]
    lo, hi = window
    false_alarms = sum(1 for d, e in zip(events, evs)
                       if lo <= e.t_peak <= hi and d["recognized"]
                       and (only_chunks is None or int(e.t_peak // 9) in only_chunks))
    return results, false_alarms


def training_data(takes_csv, run_specs, skip=None):
    main, _, _, _ = build_events(load_takes(takes_csv))
    X = [e["features"] for e in main]
    y = [e["label"] for e in main]
    for spec in run_specs:
        base, style = parse_run_spec(spec)
        if base == skip:
            continue
        rm, _, _, _ = events_from_run(base, style)
        X += [e["features"] for e in rm]
        y += [e["label"] for e in rm]
    return X, y


def fit(X, y, cols):
    return make_pipeline().fit(np.asarray(X)[:, cols], y)


def line(label, results, false_alarms):
    sc = score(results, [], (0.0, 0.0))
    t = sc["table"]
    n = sc["total"]
    return (f"  {label:<46} forehand {t['forehand']['forehand']:>2}/{t['forehand']['n']}  backhand {t['backhand']['backhand']:>2}/{t['backhand']['n']}"
            f"  flat {t['flat']['flat']:>2}/{t['flat']['n']}   {sc['right']:>2}/{n} right   false alarms {false_alarms}")


def main():
    ap = argparse.ArgumentParser(description="Score each saved run with a model that never saw it")
    ap.add_argument("csv", help="the original recording, e.g. swings.csv")
    ap.add_argument("--run", action="append", default=[], required=True, metavar="NAME[:STYLE]")
    ap.add_argument("--min-peak", type=float, default=330.0)
    args = ap.parse_args()

    cols = [i for i, n in enumerate(sd.FEATURE_NAMES) if n not in sd.CONTEXT_FEATURES]
    print(f"Each run is scored by a model trained on {args.csv} plus the OTHER runs (minimum swing size {args.min_peak:.0f}).\n")
    totals = []
    for spec in args.run:
        base, style = parse_run_spec(spec)
        evs, attempts, window = load_run_events(base)
        X, y = training_data(args.csv, args.run, skip=base)
        pipe = fit(X, y, cols)
        results, fa = score_run(evs, attempts, window, pipe, cols, args.min_peak)
        print(line(f"{base} ({style}), never seen", results, fa))
        totals.append(score(results, [], (0.0, 0.0))["right"] / max(1, len(results)))

        if style == "waiting":
            labeled = events_from_run(base, style)[0]
            all_results, fa_total = {}, 0
            for f in range(5):
                Xf, yf = list(X), list(y)
                for e in labeled:
                    fold = e["take"] % 5
                    if fold != f:
                        Xf.append(e["features"])
                        yf.append(e["label"])
                pf = fit(Xf, yf, cols)
                chunks = {c for c in range(0, 100) if c % 5 == f}
                res, fa_f = score_run(evs, attempts, window, pf, cols, args.min_peak,
                                      only_attempts={i for i in range(len(attempts)) if i % 5 == f}, only_chunks=chunks)
                for (p, r), i in zip(res, [i for i in range(len(attempts)) if i % 5 == f]):
                    all_results[i] = (p, r)
                fa_total += fa_f
            print(line(f"{base} + 4/5 of its own attempts, 5 folds", [all_results[i] for i in sorted(all_results)], fa_total))
    print(f"\nAverage on runs the model never saw: {100 * np.mean(totals):.0f}% of swings the right kind.")


if __name__ == "__main__":
    main()
