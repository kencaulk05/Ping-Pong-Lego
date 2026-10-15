"""
eval_steering.py -- what should decide which way a swing sends the shot?

    python eval_steering.py
    python eval_steering.py --extra-run guided-<time>:waiting     # add a run, e.g. one swung at game speed

Three ways to name a swing's kind (which sets the shot direction: forehand goes left and backhand right for a
right-hander, flat goes straight):

  model     the classifier's kind (what swing_check.py and the guided tools report)
  rotation  a swing that rotates mainly about the hub's y-axis is a forehand (+) or backhand (-) by the direction
            of that rotation; anything else is flat
  hybrid    rotation when it is clear (|uy| >= 0.6), otherwise the classifier. The game's default (--steer rotation)

Scored on the labeled guided swings, each run held out (a model trained on swings.csv and the OTHER runs), and
reported two ways: as recorded, and with the swings EMULATED as live-style (performed ~2-3.5x faster and ~1.5-2x
harder, with a quiet hand around them), because live play looks like that: 90% of live swings were 3 readings wide
or narrower, against 5% in the guided runs. The emulation is a rough proxy, not the real thing: training the
classifier on emulated swings (augmentation) scored better here and then made the labels on REAL live swings less
consistent with the rotation, so it was not adopted. That is why the last section compares against real live play
instead of trusting this table alone.

"wrong side" = a forehand sent right, or a backhand sent left: the error that matters most.
"""

import argparse
import glob
import warnings

import numpy as np

import swing_detect as sd
from swing_detect import H, SwingModel, steer_kind
from train_swing_classifier import build_events, events_from_run, load_takes, make_pipeline, parse_run_spec

warnings.filterwarnings("ignore")

RUNS = ["guided-20261012-023715", "guided-20261012-024425", "guided-20261012-031646:waiting",
        "guided-20261012-034425:waiting", "guided-20261012-061026:waiting", "guided-20261012-061454:waiting"]
CONDITIONS = {"as recorded": None, "live-style A (3x faster, 1.8x harder)": (3.0, 1.8, 0.3),
              "live-style B (2x, 1.5x)": (2.0, 1.5, 0.6), "live-style C (2.5x, 2x, quieter)": (2.5, 2.0, 0.15)}
SWING = sd.SWING_KINDS


def liveify(window, k, s, surround, base=8.0):
    """The same swing performed k times faster and s times harder, with the hand around it turned down."""
    G, A = window
    n = len(G)
    src = np.clip(H + (np.arange(n) - H) * k, 0, n - 1)
    ii = np.arange(n)
    G2 = np.stack([np.interp(src, ii, G[:, a]) for a in range(3)], 1)
    mid = np.median(A, axis=0)
    A2 = np.stack([np.interp(src, ii, A[:, a]) for a in range(3)], 1)
    G2, A2 = sd.isolate_swing(G2 * s, mid + (A2 - mid) * s, surround)
    return sd.event_features(G2, A2, base)


def model_kind(pipe, cols, features):
    p = dict(zip(pipe.classes_, pipe.predict_proba(np.asarray(features)[cols].reshape(1, -1))[0]))
    return max(SWING, key=lambda k: p.get(k, 0.0))


def all_events(extra):
    ev = {"swings.csv": build_events(load_takes("swings.csv"))[0]}
    for spec in RUNS + extra:
        base, style = parse_run_spec(spec)
        ev[base] = events_from_run(base, style)[0]
    return ev


def evaluate(ev):
    cols = [i for i, n in enumerate(sd.FEATURE_NAMES) if n not in sd.CONTEXT_FEATURES]
    runs = [k for k in ev if k != "swings.csv"]
    rules = ["model", "rotation", "hybrid"]
    acc = {r: {c: [0, 0] for c in CONDITIONS} for r in rules}
    wrong = {r: {c: [0, 0] for c in CONDITIONS} for r in rules}
    for tgt in runs:
        train = list(ev["swings.csv"]) + [e for r in runs if r != tgt for e in ev[r]]
        pipe = make_pipeline().fit(np.asarray([e["features"] for e in train])[:, cols], [e["label"] for e in train])
        for e in ev[tgt]:
            if e["label"] not in SWING:
                continue
            for cname, par in CONDITIONS.items():
                f = e["features"] if par is None else liveify(e["window"], *par)
                mk = model_kind(pipe, cols, f)
                chosen = {"model": mk, "rotation": steer_kind("flat", f, "rotation"), "hybrid": steer_kind(mk, f, "rotation")}
                for r in rules:
                    acc[r][cname][0] += chosen[r] == e["label"]
                    acc[r][cname][1] += 1
                    if e["label"] != "flat":
                        wrong[r][cname][1] += 1
                        wrong[r][cname][0] += chosen[r] not in ("flat", e["label"])
    return rules, acc, wrong


def live_consistency():
    """On REAL live swings (no labels): how often two differently trained models name a swing differently, with
    and without steering. A label that changes when the training data barely does isn't telling you much."""
    import csv
    from swing_detect import SwingRecognizer
    streams = sorted(glob.glob("live-*-stream.csv"))
    models = [m for m in ("swing_model.json", "swing_model_v1.json") if glob.glob(m)]
    if not streams or len(models) < 2:
        return None
    out = {}
    for steer in ("model", "rotation"):
        labels = []
        for m in models:
            model = SwingModel.load(m)
            d = {}
            for f in streams:
                rec = SwingRecognizer(model, steer=steer)
                for r in csv.DictReader(open(f)):
                    v = tuple(float(r[c]) for c in ("ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"))
                    for s in rec.push(float(r["t"]), v):
                        d[(f, round(s.t_peak, 3))] = s.kind
                    rec.rejected.clear()
            labels.append(d)
        common = set(labels[0]) & set(labels[1])
        out[steer] = (sum(labels[0][k] != labels[1][k] for k in common), len(common))
    return models, out


def main():
    ap = argparse.ArgumentParser(description="Compare ways of steering a shot")
    ap.add_argument("--extra-run", action="append", default=[], metavar="NAME[:STYLE]")
    args = ap.parse_args()
    ev = all_events(args.extra_run)
    runs = [k for k in ev if k != "swings.csv"]
    rules, acc, wrong = evaluate(ev)
    n = acc["model"]["as recorded"][1]
    print(f"Labeled guided swings: {n} in {len(runs)} runs, each run held out.\n")
    print("Kind named right, and (in brackets) forehands/backhands sent the WRONG SIDE:\n")
    print(f"{'':<12}" + "".join(f"{c:>38}" for c in CONDITIONS))
    for r in rules:
        print(f"{r:<12}" + "".join(f"{100 * acc[r][c][0] / acc[r][c][1]:>30.0f}% ({100 * wrong[r][c][0] / max(1, wrong[r][c][1]):>2.0f}%)" for c in CONDITIONS))
    res = live_consistency()
    if res:
        models, out = res
        print(f"\nReal live swings (no labels), labelled by two differently trained models ({models[0]} and {models[1]}):")
        for steer, (diff, total) in out.items():
            print(f"   {'classifier alone' if steer == 'model' else 'steered by rotation':<22} the two models disagree on {diff} of {total} swings ({100 * diff / total:.0f}%)")


if __name__ == "__main__":
    main()
