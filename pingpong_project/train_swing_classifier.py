"""
train_swing_classifier.py -- train the swing classifier from your recordings.

    python train_swing_classifier.py swings.csv
    python train_swing_classifier.py swings.csv guided-20261012-101500.csv    # learn from several sessions
    python train_swing_classifier.py swings.csv --out swing_model.json --reps 10

What it does:
  1. Replays every recorded take through the SAME detector used live
     (swing_detect.py), so training sees what play will see.
  2. In a swing take, the largest event is the swing and gets the take's label.
     Smaller events in the same take (a back-swing, a recovery) are NOT used for
     training or for the accuracy numbers; they're counted separately below.
     In a shake / carry / idle take, every event is labeled "other".
  3. Reports cross-validated accuracy where a take is never split between
     training and testing (otherwise neighbouring readings of one swing would
     leak the answer), plus two checks: the same with the movement-level
     features removed, and a train-on-early / test-on-late split.
  4. Fits the final model on everything and saves it.

Needs scikit-learn (training only; running the model needs just numpy).
"""

import argparse
import collections
import csv
import json
import math
import warnings

import numpy as np

import swing_detect as sd

SWING_LABELS = sd.SWING_KINDS
CLASSES = ["forehand", "backhand", "flat", "other"]


def load_takes(paths):
    """(session, label, take) -> (times, rows of ax..gz), in recorded order. Takes one
    CSV path or a list of them; the session id keeps takes from different files apart."""
    if isinstance(paths, str):
        paths = [paths]
    takes = collections.OrderedDict()
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["session"], r["label"], int(r["take"]))
                t = takes.setdefault(key, ([], []))
                t[0].append(float(r["t"]))
                t[1].append([float(r[c]) for c in ("ax", "ay", "az", "gx", "gy", "gz")])
    return takes


def build_events(takes):
    """Returns (train_events, secondary_events, missed, rates).
    Each event is a dict: label, group, take, features, peak."""
    main, secondary = [], []
    missed = collections.Counter()
    rates = []
    for (session, label, take), (times, rows) in takes.items():
        det = sd.SwingDetector()
        events = []
        for t, v in zip(times, rows):
            events += det.push(t, v)
        if times[-1] > times[0]:
            rates.append(det.n_fresh / (times[-1] - times[0]))
        group = (session, label, take)
        if label in SWING_LABELS:
            if not events:
                missed[label] += 1
                continue
            best = max(events, key=lambda e: e.peak)
            for e in events:
                (main if e is best else secondary).append(
                    dict(label=label, group=group, take=take, features=e.features, peak=e.peak, window=e.window))
        else:
            for e in events:
                main.append(dict(label="other", group=group, take=take, features=e.features, peak=e.peak))
    return main, secondary, missed, rates


def parse_run_spec(spec):
    """'guided-20261012-031646:waiting' -> ('guided-20261012-031646', 'waiting'). Style defaults to continuous."""
    base, _, style = spec.partition(":")
    style = style or "continuous"
    if style not in ("waiting", "continuous"):
        raise SystemExit(f"--run style must be 'waiting' or 'continuous', not '{style}'")
    return base, style


def events_from_run(base, style="continuous"):
    """Labeled events from a saved guided run (its -stream.csv and -cues.csv), by these rules:

    * In each attempt, the biggest event near the cue is the swing and gets the attempt's label;
      smaller events in that window (a back-swing, a recovery) are secondary and not trained on.
    * During the random-movement phase, every event is 'other'.
    * Everywhere else (the breaks, the ready periods), events are 'other' ONLY for style 'waiting',
      where you held still between swings. For style 'continuous' you were swinging all the time,
      so those events are swings with no label, and are left out.

    Returns (main, secondary, missed, n_attempts)."""
    from rescore import load_run
    readings, attempts, (lo, hi) = load_run(base)
    det = sd.SwingDetector()
    evs = []
    for t_, v in readings:
        evs += det.push(t_, v)
    main, secondary, missed, used = [], [], collections.Counter(), set()
    for idx, (phase, number, go) in enumerate(attempts):
        near = [e for e in evs if go - 0.4 <= e.t_peak <= go + 2.2]
        used.update(id(e) for e in near)
        if not near:
            missed[phase] += 1
            continue
        best = max(near, key=lambda e: e.peak)
        for e in near:
            d = dict(label=phase, group=(base, phase, idx), take=idx, features=e.features, peak=e.peak, window=e.window)
            (main if e is best else secondary).append(d)
    for e in evs:
        if id(e) in used:
            continue
        if lo <= e.t_peak <= hi or style == "waiting":
            main.append(dict(label="other", group=(base, "other", int(e.t_peak // 9)), take=int(e.t_peak // 9),
                             features=e.features, peak=e.peak, window=e.window))
    return main, secondary, missed, len(attempts)


def make_pipeline(C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline as mp
    from sklearn.preprocessing import StandardScaler
    return mp(StandardScaler(), LogisticRegression(max_iter=5000, C=C))


def export_model(pipe, meta, feature_names=None):
    """The sklearn pipeline's numbers, in the format swing_detect.SwingModel reads.
    feature_names says which features the pipeline was trained on (default: all)."""
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]
    data = dict(version=1, classes=[str(c) for c in lr.classes_],
                feature_names=list(feature_names or sd.FEATURE_NAMES),
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(),
                detector=sd.DETECTOR_PARAMS)
    data.update(meta)
    return data


def cross_validate(X, y, groups, secondary, reps=10, columns=None, seed0=0):
    """Grouped 5-fold CV, repeated. Returns the pooled confusion matrix, mean/sd of
    accuracy, and how many secondary events were called a swing."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    cols = list(range(X.shape[1])) if columns is None else columns
    # Integer ids by order of appearance. (Python's hash() of a string changes every run,
    # which would change the folds, and so the reported numbers, from run to run.)
    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    gid = np.array([ids[g] for g in groups])
    sec_gid = np.array([ids.get(s["group"], -1) for s in secondary]) if secondary else np.array([])
    sec_X = np.array([s["features"] for s in secondary])[:, cols] if secondary else None
    cm = np.zeros((4, 4), int)
    accs, f1s, sec_swings = [], [], []
    for rep in range(reps):
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed0 + rep)
        pred = np.empty(len(y), dtype=object)
        n_sec = 0
        for tr, te in cv.split(X, y, gid):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = make_pipeline().fit(X[tr][:, cols], y[tr])
            pred[te] = m.predict(X[te][:, cols])
            if secondary:
                mask = np.isin(sec_gid, gid[te])
                if mask.any():
                    n_sec += int(np.isin(m.predict(sec_X[mask]), list(SWING_LABELS)).sum())
        accs.append(accuracy_score(y, pred))
        f1s.append(f1_score(y, pred, average="macro"))
        sec_swings.append(n_sec)
        for t_, p_ in zip(y, pred):
            cm[CLASSES.index(t_), CLASSES.index(p_)] += 1
    return dict(cm=cm, acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), f1=float(np.mean(f1s)),
                secondary_called_swing=float(np.mean(sec_swings)))


def still_hand_check(X, y, cols, baseline=20.0):
    """Train a model on ALL features, then re-score the real forehands and backhands as if the
    hand had been nearly still before the swing (only the two movement-level features change)."""
    li = sd.FEATURE_NAMES.index("log_baseline")
    ri = sd.FEATURE_NAMES.index("peak_over_baseline")
    pk = sd.FEATURE_NAMES.index("log_peak")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_pipeline().fit(X[:, cols], y)
    sel = np.isin(y, ["forehand", "backhand"])
    Xs = X[sel].copy()
    Xs[:, li] = np.log1p(baseline)
    Xs[:, ri] = np.expm1(Xs[:, pk]) / (baseline + 1.0)
    pred = m.predict(Xs[:, cols])
    return dict(flat_pct=100.0 * float(np.mean(pred == "flat")), right_pct=100.0 * float(np.mean(pred == y[sel])))


def isolated_check(pipe, cols, events, surround=0.25, still_base=8.0):
    """Re-score the real swings as if made from a nearly still hand (see swing_detect.isolate_swing)."""
    right = flat_wrong = n = fb = 0
    for e in events:
        if e["label"] not in SWING_LABELS or e.get("window") is None:
            continue
        G, A = sd.isolate_swing(e["window"][0], e["window"][1], surround)
        f = sd.event_features(G, A, still_base)
        pred = pipe.predict(f[cols].reshape(1, -1))[0]
        n += 1
        right += pred == e["label"]
        if e["label"] in ("forehand", "backhand"):
            fb += 1
            flat_wrong += pred == "flat"
    return dict(n=n, right_pct=100.0 * right / max(1, n), flat_pct=100.0 * flat_wrong / max(1, fb))


def recalls(cm):
    return cm.diagonal() / np.maximum(1, cm.sum(1))


def precisions(cm):
    return cm.diagonal() / np.maximum(1, cm.sum(0))


def chronological(X, y, groups, columns=None):
    """Train on the earlier ~2/3 of each recorded kind, test on the later ~1/3,
    where 'earlier' means recorded earlier (across sessions, in file order)."""
    from sklearn.metrics import accuracy_score
    cols = list(range(X.shape[1])) if columns is None else columns
    order = collections.defaultdict(list)             # recorded label -> its takes, in order
    for g in groups:
        if g not in order[g[1]]:
            order[g[1]].append(g)
    train_groups = set()
    for lab, gs in order.items():
        train_groups.update(gs[:round(2 * len(gs) / 3)])
    train = np.array([g in train_groups for g in groups])
    test = ~train
    if train.sum() < 10 or test.sum() < 5 or len(set(y[train])) < 4:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make_pipeline().fit(X[train][:, cols], y[train])
    pred = m.predict(X[test][:, cols])
    return dict(n_train=int(train.sum()), n_test=int(test.sum()), acc=float(accuracy_score(y[test], pred)))


def main():
    ap = argparse.ArgumentParser(description="Train the swing classifier from swings.csv")
    ap.add_argument("csv", nargs="+", help="one or more recordings (swings.csv, guided-....csv)")
    ap.add_argument("--out", default="swing_model.json")
    ap.add_argument("--reps", type=int, default=10, help="cross-validation repeats")
    ap.add_argument("--run", action="append", default=[], metavar="NAME[:STYLE]",
                    help="a saved guided run to learn from, e.g. guided-20261012-031646:waiting "
                         "(STYLE is 'waiting' if you held still between swings, else 'continuous'). Repeatable.")
    ap.add_argument("--min-peak", type=float, default=330.0,
                    help="smallest peak that can count as a swing, saved in the model (default 330)")
    ap.add_argument("--no-cv", action="store_true", help="skip the evaluation, just fit and save")
    ap.add_argument("--with-context", action="store_true",
                    help="also use the movement-level features (not recommended: it misreads swings made from a still hand)")
    args = ap.parse_args()

    takes = load_takes(args.csv)
    print(f"Training from: {', '.join(args.csv)}")
    main_ev, secondary, missed, rates = build_events(takes)
    run_styles = {}
    extra_attempts = 0
    for spec in args.run:
        base, style = parse_run_spec(spec)
        rm, rs, rmiss, n_att = events_from_run(base, style)
        extra_attempts += n_att
        main_ev += rm
        secondary += rs
        missed.update(rmiss)
        run_styles[base] = style
        n_other = sum(1 for e in rm if e["label"] == "other")
        print(f"  + run {base} ({style}): {len(rm) - n_other} swing events, {n_other} 'not a swing' events")
    X = np.array([e["features"] for e in main_ev])
    y = np.array([e["label"] for e in main_ev])
    groups = [e["group"] for e in main_ev]
    n_swing_takes = sum(1 for k in takes if k[1] in SWING_LABELS) + extra_attempts

    print(f"{len(takes)} takes, about {np.mean(rates):.0f} fresh readings per second.")
    print(f"Events used: {dict(collections.Counter(y))}")
    print(f"Swing takes where no complete swing was found: {dict(missed)} "
          f"({sum(missed.values())} of {n_swing_takes}; includes swings cut off by the recording window)")
    print(f"Smaller extra events inside swing takes (not trained on): {len(secondary)}")
    if len(set(y)) < 4:
        raise SystemExit("Need events from all four classes (forehand, backhand, flat, other) to train.")

    names = sd.FEATURE_NAMES
    cols_all = list(range(len(names)))
    cols_used = cols_all if args.with_context else [i for i, n in enumerate(names) if n not in sd.CONTEXT_FEATURES]
    used_names = [names[i] for i in cols_used]
    print(f"Features used: {len(used_names)} of {len(names)}"
          + ("" if args.with_context else f"  (left out: {', '.join(sd.CONTEXT_FEATURES)}, the movement-level features)"))

    if not args.no_cv:
        res = cross_validate(X, y, groups, secondary, args.reps, cols_used)
        other = cross_validate(X, y, groups, None, args.reps, cols_all if not args.with_context else
                               [i for i, n in enumerate(names) if n not in sd.CONTEXT_FEATURES])
        print(f"\nCross-validation: 5 folds, {args.reps} repeats, a take is never split across train and test")
        print(f"  always guessing the biggest class : {max(collections.Counter(y).values()) / len(y):.2f}")
        print(f"  logistic regression (this model)  : {res['acc']:.2f} accuracy (+/- {res['acc_sd']:.2f}), "
              f"macro-F1 {res['f1']:.2f}")
        print(f"  ...{'without' if args.with_context else 'with'} the movement-level features: {other['acc']:.2f}  "
              f"(similar accuracy here, but see the robustness check below: accuracy on this recording "
              f"can't show whether a model leans on them)")
        rec, prec = recalls(res["cm"]), precisions(res["cm"])
        print(f"\n  {'':<10}{'recall':>8}{'precision':>11}")
        for i, c in enumerate(CLASSES):
            print(f"  {c:<10}{rec[i]:>8.2f}{prec[i]:>11.2f}")
        print(f"\n  confusion matrix, rows = truth, columns = predicted (pooled over repeats)")
        print("  " + " " * 10 + "".join(f"{c:>10}" for c in CLASSES))
        for c, row in zip(CLASSES, res["cm"]):
            print(f"  {c:<10}" + "".join(f"{v:>10d}" for v in row))
        per_take = res["secondary_called_swing"] / max(1, n_swing_takes)
        print(f"\n  Extra events inside swing takes called a swing: {per_take:.2f} per swing take "
              f"(these are mostly back-swings; the game's timing window and cooldown filter most of them)")

        chrono = chronological(X, y, groups, cols_used)
        if chrono:
            print(f"\n  Train on earlier recordings, test on later ones: {chrono['acc']:.2f} accuracy "
                  f"({chrono['n_train']} train / {chrono['n_test']} test events). With several runs this mostly holds out the\n"
                  f"  newest one, so it is a rough cross-session number; eval_runs.py scores each run properly, one at a time.")

        # The check that matters for play: every swing in a recording was made with the hand already
        # moving. If the model uses "how much was I moving before", a swing from a STILL hand looks
        # like a different kind of swing. Replace only those two features with a still hand's values.
        still = still_hand_check(X, y, cols_all)
        print(f"\n  Robustness: if your hand were still before the swing (everything else unchanged),")
        print(f"    a model that USES the movement-level features would call {still['flat_pct']:.0f}% of your forehands and "
              f"backhands 'flat' (it gets {still['right_pct']:.0f}% right).")
        if not args.with_context:
            print("    This model leaves those features out, so it can't be fooled that way.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = make_pipeline().fit(X[:, cols_used], y)
        iso = isolated_check(probe, cols_used, main_ev)
        print(f"    And with the hand around each swing turned down to 25% (an emulated still hand): "
              f"{iso['right_pct']:.0f}% of {iso['n']} swings are still called the right kind, "
              f"{iso['flat_pct']:.0f}% of forehands and backhands wrongly 'flat'.")
        print("    (An emulation, not real still-hand swings. Real ones are the true test: see swing_check.py.)")

    peaks = [e["peak"] for e in main_ev if e["label"] in SWING_LABELS]
    waiting = [b for b, st in run_styles.items() if st == "waiting"]
    if waiting:
        w_swing = [e["peak"] for e in main_ev if e["label"] in SWING_LABELS and e["group"][0] in waiting]
        w_other = [e["peak"] for e in main_ev if e["label"] == "other" and e["group"][0] in waiting]
        if w_swing and w_other:
            print(f"\nStill-hand runs: ordinary movement peaked at most {max(w_other):.0f} (median {np.median(w_other):.0f}); "
                  f"your weakest swing peaked at {min(w_swing):.0f}. Minimum swing size: {args.min_peak:.0f}"
                  + ("" if max(w_other) < args.min_peak < min(w_swing) else "  <-- NOT between those two; check it"))
    iu = [sd.FEATURE_NAMES.index(n) for n in ("ux", "uy", "uz")]
    class_direction = {}
    for lab in SWING_LABELS:
        d = np.mean([e["features"][iu] for e in main_ev if e["label"] == lab], axis=0)
        class_direction[lab] = (d / (np.linalg.norm(d) + 1e-9)).round(3).tolist()
    meta = dict(expected_hz=float(np.mean(rates)), class_direction=class_direction,
                power_range=[float(args.min_peak), float(np.percentile(peaks, 90))], min_swing_peak=float(args.min_peak),
                trained_on=dict(collections.Counter(y)), source=args.csv + args.run)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = make_pipeline().fit(X[:, cols_used], y)
    with open(args.out, "w") as f:
        json.dump(export_model(pipe, meta, used_names), f, indent=1)
    print(f"\nSaved {args.out}. Minimum swing size {args.min_peak:.0f}. "
          f"Power 0..1 maps a swing peak of {meta['power_range'][0]:.0f} (the minimum) to {meta['power_range'][1]:.0f} (your 90th percentile).")


if __name__ == "__main__":
    main()
