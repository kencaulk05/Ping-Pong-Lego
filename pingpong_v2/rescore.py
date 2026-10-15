"""
rescore.py -- score a saved guided run again, with any model, without the hub.

    python rescore.py guided-20261012-023715
    python rescore.py guided-20261012-023715 --model swing_model.json
    python rescore.py guided-20261012-023715 --compare old_model.json     # side by side

swing_check.py saves three files per run. This reads the raw stream
(<name>-stream.csv) and the cue log (<name>-cues.csv), feeds the readings through
the recognizer exactly as the live run did, and scores each attempt the same way.

Why it matters: a run that wasn't used for training is a fair test. When you change
the model (or the detector), this tells you how it would have done on every run you
have saved, so a change can be judged on real data and not just hoped about.
"""

import argparse
import collections
import csv

from swing_check import (SWING_KINDS, direction_report, event_from_recognized, event_from_rejected,
                         format_report, resolve_attempt, score)
from swing_detect import SwingModel, SwingRecognizer


def load_run(base):
    """readings [(t, values)], attempts [(phase, number, go_t)], random window (lo, hi)."""
    with open(base + "-stream.csv", newline="") as f:
        readings = [(float(r["t"]), tuple(float(r[c]) for c in ("ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll")))
                    for r in csv.DictReader(f)]
    with open(base + "-cues.csv", newline="") as f:
        cues = [(float(r["t"]), r["what"], r["phase"]) for r in csv.DictReader(f)]
    attempts, count = [], collections.Counter()
    for t, what, phase in cues:
        if what == "go" and phase in SWING_KINDS:
            count[phase] += 1
            attempts.append((phase, count[phase], t))
    rand = [t for t, what, phase in cues if what == "go" and phase == "random"]
    window = (rand[0], rand[1]) if len(rand) >= 2 else ((rand[0], readings[-1][0]) if rand else (1e9, 1e9))
    return readings, attempts, window


def rescore(base, model, min_confidence=None, min_peak=None, rule="swing"):
    readings, attempts, window = load_run(base)
    rec = SwingRecognizer(model, min_confidence=min_confidence, min_peak=min_peak, rule=rule)
    events = []
    for t, v in readings:
        for s in rec.push(t, v):
            events.append(event_from_recognized(s))
        for ev, kind, conf in rec.rejected:
            events.append(event_from_rejected(ev, kind, conf))
        rec.rejected.clear()
    results = [(phase, resolve_attempt(events, go_t)) for phase, _, go_t in attempts]
    return score(results, events, window), results


def main():
    ap = argparse.ArgumentParser(description="Score a saved guided run again with any model")
    ap.add_argument("run", help="the run's name without the suffix, e.g. guided-20261012-023715")
    ap.add_argument("--model", default="swing_model.json")
    ap.add_argument("--compare", metavar="MODEL", help="a second model to score the same run with")
    ap.add_argument("--min-confidence", type=float, default=None, help="accept cutoff (default: 0.9 for the swing rule, 0.6 for the kind rule)")
    ap.add_argument("--rule", choices=["swing", "kind"], default="swing",
                   help="swing = accept if it is confidently a swing of any kind (default); kind = the old rule")
    ap.add_argument("--min-peak", type=float, default=None, help="smallest peak that can count as a swing (default: the model's)")
    args = ap.parse_args()

    for name in [args.model] + ([args.compare] if args.compare else []):
        model = SwingModel.load(name)
        sc, results = rescore(args.run, model, args.min_confidence, args.min_peak, args.rule)
        print(f"\n##### {name}  ({len(model.feature_names)} features)")
        print("\n".join(format_report(sc)))
        print("\n".join(direction_report(results, model.meta.get("class_direction"))))


if __name__ == "__main__":
    main()
