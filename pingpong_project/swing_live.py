"""
swing_live.py -- try the swing recognizer on your hub, live.

    python swing_live.py --notify-ms 30            # hold the hub, swing, watch what it says
    python swing_live.py --replay swings.csv       # feed recorded takes through the same code

This is the real test. The accuracy numbers from training come from the same
session the model learned from; what matters is whether it recognizes swings
you make NOW, on a different day, in a different grip. Try each kind a dozen
times, plus some things that are NOT swings (waving, walking, tapping the hub),
and see what it says. Ctrl+C prints a tally.

Each line shows the kind, how confident it was, the power (0..1), and the age:
how long after the swing's peak the report arrived (about 0.26 s, which is why
the game has SWING_REPORT_DELAY).
"""

import argparse
import collections
import time

from imu_hub import HubIMU, read_stable
from swing_detect import SWING_KINDS, SwingModel, SwingRecognizer


class Tally:
    """Counts what the recognizer said, and scores it against what you said you were doing."""

    def __init__(self):
        self.recognized = collections.Counter()
        self.ignored = collections.Counter()

    def add_recognized(self, kind):
        self.recognized[kind] += 1

    def add_ignored(self, looked_like):
        self.ignored[looked_like] += 1

    def summary(self, expect=None, count=None):
        total, ign = sum(self.recognized.values()), sum(self.ignored.values())
        lines = [f"Recognized as swings: {dict(self.recognized)}",
                 f"Ignored (not confident enough, or looked like a non-swing): {dict(self.ignored)}"]
        if expect in SWING_KINDS:
            right = self.recognized[expect]
            wrong = total - right
            if count:
                none = max(0, count - total - ign)
                lines.append(f"You made {count} {expect} swings: {right} recognized correctly ({100 * right / count:.0f}%), "
                             f"{wrong} recognized as the wrong kind, {ign} ignored, about {none} produced nothing.")
            else:
                lines.append(f"{right} of {total} recognized swings were {expect} "
                             f"({100 * right / max(1, total):.0f}%); {ign} more were ignored.")
        elif expect == "none":
            lines.append(f"You weren't swinging, and it recognized {total} swings anyway "
                         f"(false alarms), and ignored {ign} more events.")
        return lines


def print_swing(s, label=""):
    print(f"  {label}{s.kind:<9} conf {s.confidence:.2f}  power {s.power:.2f}  peak {s.peak:6.0f}  age {s.age * 1000:3.0f} ms")


def run_live(args):
    from imu_recorder import connect_hub, _FakeAdapter
    dm = connect_hub(args)
    hub = _FakeAdapter(dm) if args.fake else HubIMU(dm)
    hub.check()
    if hub.set_notification_delay(args.notify_ms):
        print(f"Asked the hub for an IMU notification every {args.notify_ms} ms.")
    rec = SwingRecognizer(SwingModel.load(args.model), min_confidence=args.min_confidence, min_peak=args.min_peak,
                          rule=args.rule)
    print(f"Model expects about {rec.model.expected_hz:.0f} readings per second. Swing away; Ctrl+C to stop.\n")

    tally = Tally()
    expect = args.expect
    if expect:
        what = "NOT swinging (wave, walk, tap the hub)" if expect == "none" else f"making {expect} swings"
        print(f"Scoring against: you are {what}.\n")
    last_warn = time.perf_counter()
    try:
        while True:
            v = read_stable(hub.read)
            t = time.perf_counter()
            if v is not None:
                for s in rec.push(t, v):
                    tally.add_recognized(s.kind)
                    mark = ""
                    if expect in SWING_KINDS:
                        mark = "OK     " if s.kind == expect else "WRONG  "
                    elif expect == "none":
                        mark = "FALSE  "
                    print_swing(s, mark)
                for ev, kind, conf in rec.rejected:
                    tally.add_ignored(kind)
                    if args.show_rejected:
                        print(f"  (ignored: looked like '{kind}', conf {conf:.2f}, peak {ev.peak:.0f})")
                rec.rejected.clear()
            if t - last_warn > 5.0:
                last_warn = t
                if not rec.rate_ok():
                    print(f"  WARNING: readings are arriving at {rec.fresh_rate():.0f} per second, but the model expects "
                          f"about {rec.model.expected_hz:.0f}. Is --notify-ms right? Results can't be trusted at the wrong rate.")
            time.sleep(0.0005)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + "\n".join(tally.summary(expect, args.count)))
        try:
            dm.disconnect()
        except Exception:
            pass


def run_replay(args):
    """Push recorded takes through the recognizer exactly as live readings would be.
    NOTE: if the model was trained on this file, this is not a fair accuracy test."""
    from train_swing_classifier import load_takes
    model = SwingModel.load(args.model)
    takes = load_takes(args.replay)
    table = collections.defaultdict(collections.Counter)
    for (session, label, take), (times, rows) in takes.items():
        rec = SwingRecognizer(model, min_confidence=args.min_confidence, min_peak=args.min_peak, rule=args.rule)
        found = []
        for t, v in zip(times, rows):
            found += rec.push(t, v)
        if args.verbose:
            print(f"{label} {take}:")
            for s in found:
                print_swing(s, "  ")
        # the biggest recognized swing in the take is the one that counts here
        best = max(found, key=lambda s: s.peak).kind if found else "(nothing)"
        table[label][best] += 1
    print(f"\n{'recorded as':<12}" + "what was recognized")
    for label, c in table.items():
        print(f"  {label:<10}" + "  ".join(f"{k}: {n}" for k, n in c.most_common()))
    print("\n(This replays training data, so it is optimistic. Use live mode for a fair test.)")


def parse_args():
    p = argparse.ArgumentParser(description="Try the swing recognizer live or on a recording")
    p.add_argument("--model", default="swing_model.json")
    p.add_argument("--replay", metavar="CSV", help="run recorded takes through the recognizer instead of a live hub")
    p.add_argument("--verbose", action="store_true", help="with --replay: list every recognized swing")
    p.add_argument("--notify-ms", type=int, default=30)
    p.add_argument("--min-confidence", type=float, default=None, help="accept cutoff (default: 0.9 for the swing rule, 0.6 for the kind rule)")
    p.add_argument("--rule", choices=["swing", "kind"], default="swing",
                   help="swing = accept if it is confidently a swing of any kind (default); kind = the old rule")
    p.add_argument("--min-peak", type=float, default=None, help="smallest peak that can count as a swing (default: the model's)")
    p.add_argument("--show-rejected", action="store_true", help="also print events it decided were not swings")
    p.add_argument("--expect", choices=list(SWING_KINDS) + ["none"],
                   help="tell it what you are about to do (a kind of swing, or 'none' for non-swings) and it scores each result")
    p.add_argument("--count", type=int, default=None,
                   help="with --expect: how many swings you actually made, so it can report how many it missed entirely")
    p.add_argument("--fake", action="store_true", help="synthetic hub (plumbing test only; its signals don't resemble yours)")
    p.add_argument("--serial", default=None)
    p.add_argument("--color", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_replay(a) if a.replay else run_live(a)
