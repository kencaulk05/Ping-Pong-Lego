"""
swing_check.py -- a guided, scored test of the swing recognizer on your hub.

    python swing_check.py

Steps, with a break between each so you can reset:
  1. FOREHAND  x10   yellow light = get ready, green light + beep = swing once
  2. BACKHAND  x10   same
  3. FLAT      x10   same
  4. RANDOM    45 s  no cues: wave, walk around, tap the hub, shake it, move it
                     the way you will between balls. Any swing it reports is a false alarm.

Every attempt is scored on its own: right kind, wrong kind, ignored, or nothing.
At the end you get a table. The raw readings are saved too, in the same format
as swings.csv, so this run doubles as a second recording session:

    python train_swing_classifier.py swings.csv guided-....csv

The numbers here are the honest test of the model: it never saw these swings.
"""

import argparse
import collections
import csv
import os
import random
import threading
import time
from datetime import datetime

from imu_hub import HubIMU, read_stable
from swing_detect import FEATURE_NAMES, SWING_KINDS, SwingModel, SwingRecognizer

PRE = 0.4        # an event counts for an attempt if its peak is this long before the cue...
POST = 2.2       # ...or up to this long after it
SETTLE = 0.6     # wait this long past POST before scoring (events report ~0.26 s after their peak)
CSV_HEADER = ["session", "label", "take", "t", "ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"]


# ------------------------------------------------------------ pure logic (tested)

def build_schedule(attempts=10, spacing=3.5, ready_s=1.5, break_s=5.0, random_s=45.0, start_delay=2.0):
    """The whole script as data: timed steps, the attempt windows, and the random window."""
    steps, attempt_list = [], []
    t = start_delay
    for phase in SWING_KINDS:
        steps.append((t, "announce", phase))
        t += break_s
        for n in range(1, attempts + 1):
            steps.append((t, "ready", phase))
            go = t + ready_s
            steps.append((go, "go", phase))
            attempt_list.append(dict(phase=phase, number=n, go_t=go))
            t += spacing
        # Leave time for the last attempt to be scored (its window must close and the
        # events report ~0.26 s late) so its result prints BEFORE the next step is announced.
        t += max(0.0, POST + SETTLE - (spacing - ready_s))
    steps.append((t, "announce", "random"))
    t += break_s
    steps.append((t, "random_start", "random"))
    random_window = (t, t + random_s)
    steps.append((t + random_s, "random_end", "random"))
    return dict(steps=steps, attempts=attempt_list, random=random_window, end=t + random_s)


def resolve_attempt(events, go_t):
    """Decide what happened in one attempt. The biggest event near the cue is the
    verdict (the same rule training used); anything else is counted as extra.
    Also reports WHEN it happened relative to the cue, and its direction and width,
    so a bad result can be diagnosed instead of just counted."""
    near = [e for e in events if go_t - PRE <= e["t_peak"] <= go_t + POST]
    if not near:
        return dict(outcome="nothing", kind=None, conf=None, extras=0, peak=None, offset=None,
                    dir=None, width=None, extra_offsets=[])
    main = max(near, key=lambda e: e["peak"])
    others = [e for e in near if e is not main]
    return dict(outcome="recognized" if main["recognized"] else "ignored",
                kind=main["kind"], conf=main["conf"], extras=len(others), peak=main["peak"],
                offset=main["t_peak"] - go_t, dir=main.get("dir"), width=main.get("width"),
                extra_offsets=[round(e["t_peak"] - go_t, 2) for e in others])


def direction_report(results, class_direction):
    """Compare which way your swings rotated with how they rotated in the recording."""
    import numpy as np
    lines = []
    for phase in SWING_KINDS:
        rs = [r for p, r in results if p == phase and r.get("dir") is not None]
        if not rs:
            continue
        d = np.mean([r["dir"] for r in rs], axis=0)
        d = d / (np.linalg.norm(d) + 1e-9)
        off = float(np.median([r["offset"] for r in rs]))
        wid = float(np.median([r["width"] for r in rs if r.get("width") is not None] or [0]))
        text = f"  {phase:<9} now ({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})"
        rec = (class_direction or {}).get(phase)
        if rec:
            match = float(np.dot(d, np.array(rec)))
            text += f"   recorded ({rec[0]:+.2f},{rec[1]:+.2f},{rec[2]:+.2f})   match {match:+.2f}"
            text += "  <-- differs" if match < 0.5 else ""
        text += f"   | median event {off:+.2f} s from the cue, width {wid:.0f} readings"
        lines.append(text)
    if lines:
        lines = ["", "Which way your swings rotated (x,y,z), against the recording (match 1.0 = identical):"] + lines
    return lines


def score(results, events, random_window, upto=None):
    """results: list of (phase, resolve_attempt dict). Returns the table and totals.
    `upto` is the time you stopped, if you stopped early."""
    table = {p: dict(forehand=0, backhand=0, flat=0, ignored=0, nothing=0, n=0) for p in SWING_KINDS}
    for phase, r in results:
        row = table[phase]
        row["n"] += 1
        if r["outcome"] == "recognized":
            row[r["kind"]] += 1
        else:
            row[r["outcome"]] += 1
    right = sum(table[p][p] for p in SWING_KINDS)
    total = sum(table[p]["n"] for p in SWING_KINDS)
    lo, hi = random_window
    if upto is not None:
        hi = max(lo, min(hi, upto))          # only the part of the random phase you actually did
    in_random = [e for e in events if lo <= e["t_peak"] <= hi]
    alarms = [e for e in in_random if e["recognized"]]
    return dict(table=table, right=right, total=total,
                false_alarms=len(alarms), false_by_kind={k: sum(1 for e in alarms if e["kind"] == k) for k in SWING_KINDS},
                ignored_in_random=len(in_random) - len(alarms), random_seconds=hi - lo, stopped=upto is not None)


def format_report(sc, csv_path=None):
    lines = ["", "=" * 62, "RESULTS" + ("  (you stopped early: only finished attempts are scored)" if sc.get("stopped") else ""), "=" * 62,
             f"{'you did':<14}{'forehand':>9}{'backhand':>9}{'flat':>6}{'ignored':>9}{'nothing':>9}   right"]
    for p in SWING_KINDS:
        r = sc["table"][p]
        if r["n"]:
            lines.append(f"{p + ' x' + str(r['n']):<14}{r['forehand']:>9}{r['backhand']:>9}{r['flat']:>6}"
                         f"{r['ignored']:>9}{r['nothing']:>9}   {r[p]}/{r['n']}")
    if sc["total"]:
        lines.append(f"\nSwings recognized as the right kind: {sc['right']} of {sc['total']} "
                     f"({100 * sc['right'] / sc['total']:.0f}%)")
    mins = sc["random_seconds"] / 60.0
    if mins > 0:
        fb = ", ".join(f"{k} {v}" for k, v in sc["false_by_kind"].items() if v) or "none"
        lines.append(f"Random movement ({sc['random_seconds']:.0f} s): {sc['false_alarms']} false alarms ({fb}), "
                     f"about {sc['false_alarms'] / mins:.1f} per minute; {sc['ignored_in_random']} more events ignored")
    lines.append("  'ignored' = it saw a movement but wasn't confident or thought it wasn't a swing;")
    lines.append("  'nothing' = no movement big enough to count. Both mean the swing was lost.")
    if csv_path:
        lines.append(f"\nRaw readings saved to {csv_path}.")
        lines.append("To learn from this session too:  python train_swing_classifier.py swings.csv " + csv_path)
    return lines


def save_session(path, readings, attempts, random_window, session=None):
    """Write each attempt's window of readings, plus the whole random phase, in swings.csv's format."""
    session = session or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    new = not os.path.exists(path)
    rows = 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEADER)
        windows = [(a["phase"], a["number"], a["go_t"] - 0.5, a["go_t"] + 2.5) for a in attempts]
        windows.append(("random", 1, random_window[0], random_window[1]))
        for label, take, lo, hi in windows:
            for t, v in readings:
                if lo <= t <= hi:
                    w.writerow([session, label, take, f"{t - lo:.4f}"] + [f"{x:.3f}" for x in v])
                    rows += 1
    return rows


def save_stream(base, readings, cues):
    """Everything the recognizer saw, continuously, plus every cue and how long its call took.
    If a run goes wrong, these two files show exactly what happened."""
    with open(base + "-stream.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"])
        for t, v in readings:
            w.writerow([f"{t:.4f}"] + [f"{x:.3f}" for x in v])
    with open(base + "-cues.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "what", "phase", "seconds_the_call_took"])
        for row in cues:
            w.writerow([f"{row[0]:.4f}", row[1], row[2], f"{row[3]:.4f}"])


# ----------------------------------------------------------- replay (for testing)

class ReplayHub:
    """A stand-in hub that plays back YOUR recorded takes, so the whole guided
    check can be run, and its scoring tested, without the hardware."""

    def __init__(self, csv_path, seed=0):
        from train_swing_classifier import load_takes
        self.rng = random.Random(seed)
        self.by_label = {}
        for (_, label, _), (times, rows) in load_takes(csv_path).items():
            self.by_label.setdefault(label, []).append(rows)
        self.rest = self.by_label["idle"][0][0]
        self._play = None
        self._random = False

    def _start(self, rows):
        self._play = (time.perf_counter(), rows)

    def begin_attempt(self, label):
        self._random = False
        self._start(self.rng.choice(self.by_label[label]))

    def begin_random(self):
        self._random = True
        self._start(self.rng.choice(self.by_label["shake"] + self.by_label["carry"] + self.by_label["idle"]))

    def end_random(self):
        self._random = False
        self._play = None

    def read(self):
        if self._play is None:
            row = self.rest
        else:
            start, rows = self._play
            i = int((time.perf_counter() - start) / 0.01)
            if i >= len(rows):
                if self._random:
                    self.begin_random()
                    return self.read()
                row = rows[-1]
            else:
                row = rows[i]
        return tuple(row) + (0.0, 0.0, 0.0)

    def check(self):
        pass

    def set_notification_delay(self, ms):
        return True

    def cue_ready(self):
        pass

    def cue_go(self, beep=True):
        pass


# --------------------------------------------------------------------- the run

class TerminalCues:
    """Cues on this screen only: nothing is sent to the hub. For telling apart
    'the hub's light/beep commands disturb the readings' from everything else."""

    def cue_ready(self):
        print("    ... get ready ...")

    def cue_go(self, beep=True):
        print("    >>> SWING NOW <<<\a")


def shape_of(features):
    """(rotation direction, width) of an event, for diagnostics."""
    if features is None:
        return None, None
    i = FEATURE_NAMES.index
    return (float(features[i("ux")]), float(features[i("uy")]), float(features[i("uz")])), float(features[i("width")])


def event_from_recognized(s, t0=0.0):
    d, w = shape_of(s.features)
    return dict(t_peak=s.t_peak - t0, peak=s.peak, recognized=True, kind=s.kind, conf=s.confidence, dir=d, width=w)


def event_from_rejected(ev, kind, conf, t0=0.0):
    d, w = shape_of(ev.features)
    return dict(t_peak=ev.t_peak - t0, peak=ev.peak, recognized=False, kind=kind, conf=conf, dir=d, width=w)


class Collector:
    """Reads the hub on its own thread and feeds the recognizer.

    This has to be separate from the main loop. LEGO's beep() and light_color()
    wait for the command to finish, so if one loop did both, the reading would
    stop during every cue, and your swing (which comes right after the beep)
    would fall into the gap. The gap would also look like a sudden jump in the
    data and be read as a swing. (The first version did exactly that.)"""

    def __init__(self, hub, rec, t0):
        self.hub, self.rec, self.t0 = hub, rec, t0
        self.lock = threading.Lock()
        self.events, self.readings, self._alarms = [], [], []
        self.errors = 0
        self._last = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                v = read_stable(self.hub.read)
            except Exception:
                self.errors += 1
                time.sleep(0.01)
                continue
            if v is not None:
                t = time.perf_counter()
                with self.lock:
                    if v != self._last:
                        self.readings.append((t - self.t0, v))
                        self._last = v
                    for s in self.rec.push(t, v):
                        e = event_from_recognized(s, self.t0)
                        self.events.append(e)
                        self._alarms.append(e)
                    for ev, kind, conf in self.rec.rejected:
                        self.events.append(event_from_rejected(ev, kind, conf, self.t0))
                    self.rec.rejected.clear()
            time.sleep(0.0005)

    def snapshot(self):
        with self.lock:
            return list(self.events)

    def fresh_rate(self):
        with self.lock:
            return self.rec.fresh_rate()

    def rate_ok(self):
        with self.lock:
            return self.rec.rate_ok()

    def new_recognized(self):
        with self.lock:
            out, self._alarms = self._alarms, []
        return out


PHASE_TEXT = {
    "forehand": "FOREHAND",
    "backhand": "BACKHAND",
    "flat": "FLAT (a short straight push)",
    "random": "RANDOM MOVEMENT: wave, walk around, tap the hub, shake it, move it like between balls",
}


def run(args, hub, model):
    sched = build_schedule(args.attempts, args.spacing, args.ready, args.break_s, args.random_seconds)
    rec = SwingRecognizer(model, min_confidence=args.min_confidence, min_peak=getattr(args, "min_peak", None),
                          rule=getattr(args, "rule", "swing"))
    results, cues = [], []
    pending = list(sched["attempts"])
    steps = list(sched["steps"])
    rs, re_ = sched["random"]
    beep = not getattr(args, "no_beep", False)
    use_cues = not getattr(args, "no_cues", False)
    t0 = time.perf_counter()
    col = Collector(hub, rec, t0).start()

    cue_source = hub if use_cues else TerminalCues()      # --no-cues: prompts on screen, nothing sent to the hub

    def cue(what, phase):
        """Send a cue (to the hub, or just the screen), logging when and how long the call took."""
        started = time.perf_counter() - t0
        if what == "ready":
            cue_source.cue_ready()
        else:
            cue_source.cue_go(beep=beep)
        cues.append((started, what, phase, time.perf_counter() - t0 - started))

    def rate_note():
        hz = col.fresh_rate()
        if hz is None:
            return
        msg = f"    (hub is sending {hz:.0f} readings per second; the model expects about {model.expected_hz:.0f})"
        if not col.rate_ok():
            msg += "  <-- WRONG RATE: results can't be trusted"
        print(msg)

    interrupted = False
    try:
        while True:
            now = time.perf_counter() - t0
            for e in col.new_recognized():
                if rs <= e["t_peak"] <= re_:
                    print(f"    FALSE ALARM: it reported a {e['kind']} (conf {e['conf']:.2f})")

            # Score finished attempts first, so a result is never printed after the next step's announcement.
            while pending and pending[0]["go_t"] + POST + SETTLE <= now:
                a = pending.pop(0)
                r = resolve_attempt(col.snapshot(), a["go_t"])
                results.append((a["phase"], r))
                print(f"  {a['phase']:<9}{a['number']:>2}/{args.attempts}  " + describe(a["phase"], r))

            while steps and steps[0][0] <= now:
                _, what, phase = steps.pop(0)
                if what == "announce":
                    n = args.attempts
                    head = f"{PHASE_TEXT[phase]}" + (f" x{n}" if phase in SWING_KINDS else f" for {args.random_seconds:.0f} s")
                    print(f"\n=== NEXT: {head}. Starting in {args.break_s:.0f} s ===")
                    if phase in SWING_KINDS:
                        how = ("Yellow light = get ready, green light" + (" + beep" if beep else "") + " = swing once"
                               if use_cues else "Watch the screen: 'get ready', then '>>> SWING NOW <<<' = swing once")
                        print(f"    {how}, then relax.")
                    rate_note()
                    cue("ready", phase)
                elif what == "ready":
                    cue("ready", phase)
                elif what == "go":
                    if hasattr(hub, "begin_attempt"):
                        hub.begin_attempt(phase)
                    cue("go", phase)
                elif what == "random_start":
                    if hasattr(hub, "begin_random"):
                        hub.begin_random()
                    cue("go", phase)
                    print("    GO: move around however you like.")
                    if isinstance(hub, ReplayHub):
                        print("    (replay mode stitches recorded takes together; false alarms right at the joins are an artifact of that, not the model.)")
                elif what == "random_end":
                    if hasattr(hub, "end_random"):
                        hub.end_random()
                    cue("go", phase)
                    print("    Done with random movement.")

            if not steps and not pending and now > sched["end"] + POST + SETTLE:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped early. Scoring what was finished, and saving everything.")
    finally:
        col.stop()

    stopped_at = time.perf_counter() - t0 if interrupted else None
    events, readings = col.snapshot(), list(col.readings)
    if col.errors:
        print(f"\n(note: {col.errors} hub reads failed during the run)")
    sc = score(results, events, sched["random"], upto=stopped_at)
    path = None
    if not args.no_save:
        path = args.save or datetime.now().strftime("guided-%Y%m%d-%H%M%S.csv")
        finished = set(_numbered(results))
        done = [a for a in sched["attempts"] if (a["phase"], a["number"]) in finished] if interrupted else sched["attempts"]
        window = sched["random"]
        if interrupted:
            window = (window[0], max(window[0], min(window[1], stopped_at)))     # only the part you actually did
        save_session(path, readings, done, window)
        base = path[:-4] if path.endswith(".csv") else path
        save_stream(base, readings, cues)
    print("\n".join(format_report(sc, path)))
    print("\n".join(direction_report(results, getattr(model, "meta", {}).get("class_direction"))))
    if path:
        base = path[:-4] if path.endswith(".csv") else path
        print(f"\nAlso saved the full raw stream ({base}-stream.csv) and the cue log ({base}-cues.csv).")
        print("If the results look wrong, send me those two files; they show exactly what the hub reported.")
    return sc


def _numbered(results):
    """results as (phase, r) -> the same list, numbering attempts per phase in order."""
    seen = collections.Counter()
    out = []
    for phase, r in results:
        seen[phase] += 1
        out.append((phase, seen[phase]))
    return out


def describe(phase, r):
    extra = f"  (+{r['extras']} extra event{'s' if r['extras'] != 1 else ''}" + \
            (f" at {', '.join(f'{o:+.1f}' for o in r['extra_offsets'])} s" if r.get("extra_offsets") else "") + ")" if r["extras"] else ""
    if r["outcome"] == "nothing":
        return "nothing detected" + extra
    d = r.get("dir")
    detail = f"   [peak {r['peak']:.0f}, {r['offset']:+.2f} s from the cue" + \
             (f", width {r['width']:.0f}" if r.get("width") is not None else "") + \
             (f", rotating {d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f}" if d else "") + "]"
    if r["outcome"] == "ignored":
        return f"ignored (looked like '{r['kind']}', conf {r['conf']:.2f})" + extra + detail
    verdict = "OK" if r["kind"] == phase else f"WRONG, expected {phase}"
    return f"{r['kind']} (conf {r['conf']:.2f})  {verdict}" + extra + detail


def parse_args():
    p = argparse.ArgumentParser(description="Guided, scored test of the swing recognizer")
    p.add_argument("--model", default="swing_model.json")
    p.add_argument("--notify-ms", type=int, default=30)
    p.add_argument("--attempts", type=int, default=10, help="swings per kind")
    p.add_argument("--random-seconds", type=float, default=45.0)
    p.add_argument("--break", dest="break_s", type=float, default=5.0, help="seconds between steps")
    p.add_argument("--spacing", type=float, default=3.5, help="seconds per attempt")
    p.add_argument("--ready", type=float, default=1.5, help="yellow-light seconds before each GO")
    p.add_argument("--min-confidence", type=float, default=None, help="accept cutoff (default: 0.9 for the swing rule, 0.6 for the kind rule)")
    p.add_argument("--rule", choices=["swing", "kind"], default="swing",
                   help="swing = accept if it is confidently a swing of any kind (default); kind = the old rule")
    p.add_argument("--min-peak", type=float, default=None, help="smallest peak that can count as a swing (default: the model's)")
    p.add_argument("--save", default=None, help="where to save the raw readings (default: guided-<time>.csv)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-cues", action="store_true",
                   help="don't touch the hub's light or beeper at all; just print 'get ready' / 'GO' on screen (for A/B testing)")
    p.add_argument("--no-beep", action="store_true", help="light cues only (in case the beeper's vibration shows up in the sensor)")
    p.add_argument("--yes", action="store_true", help="don't wait for Enter before starting")
    p.add_argument("--replay", metavar="CSV", help="play back your recorded takes instead of using the hub (tests the harness)")
    p.add_argument("--serial", default=None)
    p.add_argument("--color", default=None)
    p.add_argument("--fake", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = SwingModel.load(args.model)
    if args.replay:
        hub, dm = ReplayHub(args.replay), None
        print(f"[replay] Playing back recorded takes from {args.replay}. Results are optimistic: the model trained on them.")
    else:
        from imu_recorder import _FakeAdapter, connect_hub
        dm = connect_hub(args)
        hub = _FakeAdapter(dm) if args.fake else HubIMU(dm)
        hub.check()
    total = 3 * args.attempts * args.spacing + args.random_seconds + 4 * args.break_s
    print(f"\nSwing check: {args.attempts} each of forehand, backhand and flat, then {args.random_seconds:.0f} s of "
          f"random movement. About {total / 60:.1f} minutes, with a {args.break_s:.0f} s break between steps.")
    print("Hold the hub the way you will in the game. Ctrl+C stops early; what finished is still scored and saved.")
    if not args.yes and not args.replay:
        input("Press Enter to begin... ")
    if not args.replay and hub.set_notification_delay(args.notify_ms):
        # Set just before starting, not before the wait for Enter, so it can't go stale in between.
        print(f"Asked the hub for an IMU notification every {args.notify_ms} ms.")
        time.sleep(0.5)
    try:
        run(args, hub, model)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if dm is not None:
            try:
                dm.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
