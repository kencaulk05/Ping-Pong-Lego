"""
analyze_live.py -- what actually happened in your live games.

    python analyze_live.py                      every live-*-events.csv in this folder
    python analyze_live.py live-20261013-081141 just those sessions (name without the suffix)

Every live session saves live-<time>-events.csv (game events and swings, with timing) and
live-<time>-stream.csv (every IMU reading). This reads them and answers the questions that matter for tuning:

  1. How did the games go?            hits, misses, why you missed, hit rate by level
  2. Is the timing right?             how early or late your swings were against the ball (bias = latency off)
  3. Are live swings like the guided  swing size and width in play vs in the guided runs the model learned from
     swings the model learned from?   (a big gap means the classifier is guessing outside what it has seen)
  4. Does steering agree with the     how the classifier's kind compares with the direction the hub rotated, and
     rotation?                        how stable each is
  5. Is the position check too tight? how far outside the ring "out of position" misses were, and what other
                                      ring sizes would have done to your hit rate
  6. Does the buzz look like a swing? the biggest gyro reading after hits and after misses where you didn't swing

The questions are answered from the logs alone; nothing here can say what you INTENDED on a swing.
"""

import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

import config as C

FLOOR = 330.0


# ------------------------------------------------------------------- loading

def load_events(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = json.loads(r["details"])
            d["type"], d["t"] = r["type"], float(r["t"])
            out.append(d)
    return out


def load_stream(path):
    """(t, values 9) arrays from a -stream.csv."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), tuple(float(r[c]) for c in ("ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"))))
    return rows


def session_files(names):
    if names:
        return sorted(f"{n.replace('-events.csv', '')}-events.csv" for n in names)
    return sorted(glob.glob("live-*-events.csv"))


# ------------------------------------------------------------------ analysis

def with_levels(events):
    """Each event with the level in play when it happened (from the start / level events)."""
    level, out = None, []
    for e in events:
        if e["type"] in ("start", "level") and "level" in e:
            level = e["level"]
        out.append((level, e))
    return out


def summarize(events):
    c = collections.Counter(e["type"] for e in events)
    best = max([e.get("streak", 0) for e in events if e["type"] == "hit"] + [0])
    reasons = collections.Counter(e.get("reason") for e in events if e["type"] == "miss")
    minutes = (events[-1]["t"] - events[0]["t"]) / 60 if events else 0.0
    return dict(hits=c["hit"], misses=c["miss"], winners=c["winner"], best=best, minutes=minutes,
                swings=c["swing_seen"], reasons=dict(reasons))


def by_level(all_events):
    tot = collections.defaultdict(collections.Counter)
    for events in all_events:
        for level, e in with_levels(events):
            if e["type"] == "hit":
                tot[level]["hit"] += 1
            elif e["type"] == "miss":
                tot[level]["miss"] += 1
                if e.get("reason") == "out of position":
                    tot[level]["oop"] += 1
    return tot


def timing_offsets(all_events):
    return np.array([e["offset"] for events in all_events for e in events if e["type"] == "swing_timing"])


def swing_ages(all_events):
    return np.array([e["age"] for events in all_events for e in events if e["type"] == "swing_seen"])


def miss_distances(all_events):
    """(level, distance outside the ring in units of the BASE ring radius) for each position miss. Older logs
    give the distance in base radii; newer ones in the level's scaled radius and say which scale."""
    out = []
    for events in all_events:
        for level, e in with_levels(events):
            if e["type"] == "miss" and e.get("reason") == "out of position" and "dist" in e:
                out.append((level, e["dist"] * e.get("ring_scale", 1.0)))
    return out


def ring_counterfactual(all_events, ladder):
    """Extra hits if the ring had been `ladder[level]` times the base radius, among position misses."""
    tot = by_level(all_events)
    hits = sum(t["hit"] for t in tot.values())
    total = hits + sum(t["miss"] for t in tot.values())
    extra = sum(1 for lv, d in miss_distances(all_events) if d <= ladder.get(lv, 1.0))
    return extra, hits / max(1, total), (hits + extra) / max(1, total)


def recognized_swings(stream_rows, model, steer="model"):
    from swing_detect import SwingRecognizer
    rec = SwingRecognizer(model, steer=steer)
    out = []
    for t, v in stream_rows:
        out += rec.push(t, v)
        rec.rejected.clear()
    return out


def rotation_label(features, threshold=0.6):
    import swing_detect as sd
    uy = float(features[sd._UY])
    return "flat" if abs(uy) < threshold else ("forehand" if uy > 0 else "backhand")


def buzz_check(events, stream_rows):
    """Largest gyro reading in the 0.7 s after each hit, and after each miss where you did not swing."""
    t = np.array([r[0] for r in stream_rows])
    g = np.array([np.linalg.norm(r[1][3:6]) for r in stream_rows])
    after = {"hit": [], "miss (no swing)": []}
    for e in events:
        key = "hit" if e["type"] == "hit" else ("miss (no swing)" if e["type"] == "miss" and e.get("reason") == "no swing" else None)
        if key is None:
            continue
        lo = e["t"] + (0.05 if key == "hit" else 0.0)
        w = (t >= lo) & (t <= e["t"] + 0.7)
        if w.sum() > 5:
            after[key].append(g[w].max())
    return {k: np.array(v) for k, v in after.items()}


# --------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser(description="What happened in your live games")
    ap.add_argument("sessions", nargs="*", help="session names, e.g. live-20261013-081141 (default: all)")
    ap.add_argument("--model", default="swing_model.json")
    args = ap.parse_args()

    files = session_files(args.sessions)
    if not files:
        raise SystemExit("No live-*-events.csv files here. Play a game with pingpong_live.py first.")
    sessions = {f: load_events(f) for f in files}
    all_events = list(sessions.values())

    print("1. THE GAMES")
    print(f"   {'session':<22}{'min':>5}{'hits':>6}{'miss':>6}{'best':>5}{'swings':>8}   why you missed")
    for f, ev in sessions.items():
        s = summarize(ev)
        print(f"   {f[5:20]:<22}{s['minutes']:>5.1f}{s['hits']:>6}{s['misses']:>6}{s['best']:>5}{s['swings']:>8}   {s['reasons']}")
    tot = by_level(all_events)
    print("   by level:  " + "   ".join(f"L{lv}: {t['hit']} hits / {t['hit'] + t['miss']} ({100 * t['hit'] / max(1, t['hit'] + t['miss']):.0f}%)"
                                      for lv, t in sorted(tot.items(), key=lambda kv: (kv[0] is None, kv[0])) if lv))

    off = timing_offsets(all_events)
    if len(off) >= 5:
        print("\n2. TIMING against the ball's arrival (negative = early)")
        print(f"   {len(off)} swings: median {np.median(off):+.2f} s, 10-90% {np.percentile(off, 10):+.2f} to {np.percentile(off, 90):+.2f}")
        print("   within the windows: " + ", ".join(f"+-{w:.2f} s {100 * np.mean(np.abs(off) <= w):.0f}%" for w in (0.30, 0.20, 0.15)))
        print("   (a median near 0 means the latency compensation is right; a steady lean means raise or lower the swing age)")
        ages = swing_ages(all_events)
        if len(ages):
            print(f"   age when the game received a swing: median {np.median(ages):.2f} s, 90% {np.percentile(ages, 90):.2f} s; "
                  f"the game waits {C.LIVE_SWING_REPORT_DELAY:.2f} s past the window")

    streams = [f.replace("-events", "-stream") for f in files if os.path.exists(f.replace("-events", "-stream"))]
    if streams and os.path.exists(args.model):
        from swing_detect import SwingModel, FEATURE_NAMES
        model = SwingModel.load(args.model)
        swings = []
        for s in streams:
            swings += recognized_swings(load_stream(s), model)
        if swings:
            w = np.array([x.features[FEATURE_NAMES.index("width")] for x in swings])
            pk = np.array([x.peak for x in swings])
            print(f"\n3. ARE LIVE SWINGS LIKE THE ONES THE MODEL LEARNED?  ({len(swings)} swings replayed through {args.model})")
            print(f"   swing peak: median {np.median(pk):.0f} (10-90%: {np.percentile(pk, 10):.0f}-{np.percentile(pk, 90):.0f})")
            print(f"   swing width in readings: median {np.median(w):.0f}; {100 * np.mean(w <= 3):.0f}% are 3 or narrower")
            print("   (guided runs: peak median ~880, width median ~7, 5% are 3 or narrower. A big gap means the classifier is guessing.)")

            tab = collections.Counter((x.kind, rotation_label(x.features)) for x in swings)
            print("\n4. STEERING: the classifier's kind (rows) against the direction the hub rotated (columns)")
            print(f"   {'':<10}{'forehand':>10}{'backhand':>10}{'flat':>8}")
            for k in ("forehand", "backhand", "flat"):
                print(f"   {k:<10}" + "".join(f"{tab[(k, c)]:>10}" for c in ("forehand", "backhand", "flat")))
            agree = sum(tab[(k, k)] for k in ("forehand", "backhand", "flat")) / len(swings)
            flat_n = sum(tab[("flat", c)] for c in ("forehand", "backhand", "flat"))
            flat_y = (tab[("flat", "forehand")] + tab[("flat", "backhand")]) / max(1, flat_n)
            print(f"   they agree on {100 * agree:.0f}% of swings; of the {flat_n} the classifier called flat, {100 * flat_y:.0f}% rotated mainly about y")
            print("   (the game's default --steer rotation uses the rotation direction, so size can't change where the ball goes)")

    oops = miss_distances(all_events)
    if oops:
        d = np.array([x for _, x in oops])
        print(f"\n5. POSITION MISSES ({len(d)} with a logged distance; 1.0 = the edge of the BASE ring)")
        print(f"   median {np.median(d):.2f}; within 1.2: {100 * np.mean(d <= 1.2):.0f}%, 1.35: {100 * np.mean(d <= 1.35):.0f}%, 1.5: {100 * np.mean(d <= 1.5):.0f}%")
        ladders = [("level ring scales in config.py", {lv: L.ring_scale for lv, L in C.LEVELS.items()}),
                   ("one size x1.5", {1: 1.5, 2: 1.5, 3: 1.5})]
        for name, lad in ladders:
            extra, h0, h1 = ring_counterfactual(all_events, lad)
            print(f"   {name:<34} would add {extra:>3} hits: hit rate {100 * h0:.0f}% -> {100 * h1:.0f}%")
        print("   (counted on the misses that were rejected; misses already forgiven by a scaled ring don't appear)")

    checks = collections.defaultdict(list)
    for f in files:
        sf = f.replace("-events", "-stream")
        if os.path.exists(sf):
            for k, v in buzz_check(sessions[f], load_stream(sf)).items():
                checks[k] += list(v)
    if checks:
        print(f"\n6. DOES THE BUZZ LOOK LIKE A SWING?  (largest gyro reading in the 0.7 s after; a swing needs {FLOOR:.0f})")
        for k, v in checks.items():
            v = np.array(v)
            if len(v):
                print(f"   after {k:<16} n={len(v):>3}  median {np.median(v):>4.0f}  90% {np.percentile(v, 90):>4.0f}  above the floor: {int((v > FLOOR).sum())}")
        print("   (after a miss you may also simply have moved; python pingpong_live.py --check-haptics tests it with a still hand)")


if __name__ == "__main__":
    main()
