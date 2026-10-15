"""
imu_recorder.py -- measure the hub's IMU, then record swings to train on.

    python imu_recorder.py --probe            # how fast and how late is the IMU data?
    python imu_recorder.py --record           # guided capture -> swings.csv
    python imu_recorder.py --summary swings.csv
    python imu_recorder.py --fake --probe     # no hub: preview the output with a synthetic one

Hold the hub the way you will in the game. Everything here is about YOUR hub in
YOUR hand, which is the whole point: the numbers it prints replace the guesses
in config.py (HIT_WINDOW) and give the swing classifier its training data.

Connecting: with no options it connects to the first Double Motor advertising,
like the rest of your projects. Use --serial / --color to pick a specific one.
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime

from imu_hub import (FIELDS, FakeDoubleMotor, HubIMU, ImuSampler, gyro_mag, idle_noise,
                     measure_kick_latency, measure_update_rate, percentile, recommend_delay,
                     swing_peak, sweep_notification_rates)

# What to record, and how many of each. The three swings are what the game
# needs to tell apart; the other three are things the classifier must learn NOT
# to call a swing (so a shake or a walk doesn't score a hit).
DEFAULT_COUNTS = {"forehand": 30, "backhand": 30, "flat": 30,
                  "idle": 10, "shake": 15, "carry": 10}
SWING_LABELS = ("forehand", "backhand", "flat")
INSTRUCTIONS = {
    "forehand": "FOREHAND: swing the hub across your body to your dominant side, like hitting a ping-pong ball",
    "backhand": "BACKHAND: swing back the other way, across the other side of your body",
    "flat":     "FLAT: a short straight push forward, like blocking the ball",
    "idle":     "IDLE: just hold the hub still in your hand",
    "shake":    "SHAKE: shake the hub around randomly (not a swing!)",
    "carry":    "CARRY: walk around holding the hub naturally",
}
CSV_HEADER = ["session", "label", "take", "t", "ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"]


# --------------------------------------------------------------------- connect

def connect_hub(args):
    if args.fake:
        print("[fake] Using a synthetic hub. The numbers show what the output looks like, nothing more.")
        return FakeDoubleMotor()
    from lelib import doubleMotor                   # your course helper
    kwargs = {}
    if args.serial:
        kwargs["card_serial"] = args.serial
    if args.color:
        import legoeducation as le
        kwargs["card_color"] = getattr(le, f"LEGO_COLOR_{args.color.upper()}")
    dm = doubleMotor()
    print("Connecting to the Double Motor...")
    dm.connect(**kwargs)
    print("Connected.")
    return dm


# ----------------------------------------------------------------------- probe

def run_probe(hub, sampler, read, kick_trials=8, rate_seconds=3.0, noise_seconds=2.0,
              swing_seconds=2.0, settle=1.2, sweep_delays=(100, 50, 30, 20, 15), do_kick=None):
    print("\n=== 1. How often does the hub send fresh IMU data, and can we speed it up? ===")
    print("The hub sends its IMU readings on a timer, and LEGO's API lets us ask for a faster one")
    print("(15 to 1000 ms between notifications). I'll request several speeds and measure what")
    print("actually arrives. Keep wiggling the hub gently the whole time (about "
          f"{len(sweep_delays) * (rate_seconds + 0.4):.0f} seconds).")
    input("Press Enter to start... ")
    results = sweep_notification_rates(hub, read, delays=sweep_delays, seconds=rate_seconds)
    print(f"  {'requested':>10}{'= Hz':>8}{'arrived':>10}{'typical gap':>14}{'longest gap':>14}")
    for r in results:
        st = r["stats"]
        if st is None:
            print(f"  {str(r['delay_ms']) + ' ms':>10}{r['requested_hz']:>8.0f}{'nothing seen':>10}")
        else:
            print(f"  {str(r['delay_ms']) + ' ms':>10}{r['requested_hz']:>8.0f}{st['hz']:>9.1f}Hz"
                  f"{st['median_ms']:>11.0f} ms{st['max_ms']:>11.0f} ms")
    best = recommend_delay(results)
    best_hz = None
    if best is None:
        print("  The hub didn't speed up (or the readings never changed). Keep the default and lean on")
        print("  pose for swing timing.")
        hub.set_notification_delay(100)
    else:
        hub.set_notification_delay(best)
        best_hz = next(r["stats"]["hz"] for r in results if r["delay_ms"] == best)
        print(f"  Fastest setting the hub kept up with: {best} ms, about {best_hz:.0f} readings per second.")
        print(f"  Use it with:  python imu_recorder.py --record --notify-ms {best}")
        if best_hz >= 30:
            print("  That is fast enough to time a swing from the IMU alone.")
        else:
            print("  Better, but a swing is still only a few readings; pose will do the timing.")

    rtt_med = None
    if do_kick is None:
        print("\n=== 2. Round-trip delay (optional) ===")
        print("This twitches the left motor and times how long until the IMU feels it. A bare motor")
        print("shaft is too light to move the hub, so it only works with something lopsided on the shaft")
        print("(a wheel with a brick on one side). Skip it otherwise; it isn't needed for the game.")
        do_kick = input("Run it? (y/N) ").strip().lower().startswith("y")
    if do_kick:
        lat = measure_kick_latency(hub, sampler, trials=kick_trials, settle=settle)
        ok = sorted(x for x in lat if x is not None)
        for i, x in enumerate(lat, 1):
            print(f"  trial {i}: " + (f"{x * 1000:.0f} ms" if x is not None else "no motion detected"))
        if ok:
            rtt_med, rtt_p95 = ok[len(ok) // 2], percentile(ok, 0.95)
            print(f"  median {rtt_med * 1000:.0f} ms, 95% under {rtt_p95 * 1000:.0f} ms ({len(ok)}/{len(lat)} detected)")
            print("  This includes the motor spinning up, so it is an upper bound on how late a real swing arrives.")
        else:
            print("  Nothing detected: the twitch didn't move the hub enough.")

    print("\n=== 3. How big is a swing, compared with noise? ===")
    print("First, hold the hub completely still for 2 seconds.")
    input("Press Enter to start... ")
    noise = idle_noise(sampler, noise_seconds)
    print(f"  resting gyro: mean {noise['gyro_mean']:.1f}, spread {noise['gyro_std']:.1f}")
    if noise["n"] < 20:
        # Too few readings to know what "still" looks like (a busy computer, or a hub that stopped
        # streaming). Judging swings against that would produce a meaningless threshold.
        print(f"  Only {noise['n']} readings arrived in that time, which is too few to judge. Check the hub is")
        print("  connected and streaming (and the computer isn't busy), then run --probe again.")
        return {"hz": best_hz, "notify_ms": best, "rtt_median": rtt_med, "swing_threshold": None}
    peaks = []
    for i in range(3):
        input(f"Swing #{i + 1}: press Enter, then make a normal swing within 2 seconds... ")
        hub_cue(hub)
        t0 = time.perf_counter()
        time.sleep(swing_seconds)
        peaks.append(swing_peak(sampler.since(t0)))
        print(f"  peak gyro {peaks[-1]:.0f}")
    floor = noise["gyro_mean"] + 5 * noise["gyro_std"]
    if min(peaks) <= floor:
        thr = None
        print(f"  At least one swing barely rose above the resting noise (weakest peak {min(peaks):.0f}, "
              f"noise ceiling about {floor:.0f}).")
        print("  Not suggesting a threshold from that. Run --probe again and make clearer, bigger swings.")
    else:
        thr = noise["gyro_mean"] + 0.4 * (min(peaks) - noise["gyro_mean"])
        print(f"  Your weakest swing was {min(peaks) / max(noise['gyro_mean'] + noise['gyro_std'], 1e-9):.0f}x the resting level.")
        print("  Careful: that is against a hub held perfectly still. A hand holding it is never that still")
        print("  (often 100-300 in real recordings), so a fixed threshold would fire constantly. The swing")
        print("  recognizer (swing_detect.py) uses a threshold that adapts to how much you are moving.")
        print("\nSave the rate table: the recognizer expects the setting that gave about 30 readings a second.")
    return {"hz": best_hz, "notify_ms": best, "rtt_median": rtt_med, "swing_threshold": thr}


def hub_cue(hub):
    if hasattr(hub, "cue_go"):
        hub.cue_go()


# ---------------------------------------------------------------------- record

def record_session(hub_imu, sampler, labels, counts, out_csv, prep_s=1.5, window_s=1.5,
                   min_peak=None, fake=None, log=print, label_pause=2.0, noise_s=1.5):
    """Guided capture. Appends to out_csv so sessions accumulate. A swing label
    that comes out without a clear swing is redone and doesn't count."""
    new_file = not os.path.exists(out_csv)
    session = datetime.now().strftime("%Y%m%d-%H%M%S-%f")     # microseconds: two sessions never collide
    log("\nHold the hub still for a moment so I can measure the resting noise...")
    noise = idle_noise(sampler, noise_s)
    needed = min_peak if min_peak is not None else noise["gyro_mean"] + max(15 * noise["gyro_std"], 30.0)
    log(f"Resting gyro {noise['gyro_mean']:.1f} +/- {noise['gyro_std']:.1f}; "
        f"a swing must peak above {needed:.0f} to count.")

    taken, redone, rows_written = {}, 0, 0
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_HEADER)
        try:
            for label in labels:
                want = counts[label]
                log(f"\n--- {INSTRUCTIONS[label]}")
                log(f"    {want} takes. Yellow light = get ready, green + beep = GO. Ctrl+C stops and keeps what's saved.")
                time.sleep(label_pause)
                n = 0
                while n < want:
                    if fake is not None:
                        fake.scenario = label
                    hub_imu.cue_ready()
                    time.sleep(prep_s)
                    t0 = time.perf_counter()
                    hub_imu.cue_go()
                    time.sleep(window_s)
                    rows = sampler.since(t0, t0 + window_s)
                    peak = swing_peak(rows)
                    if label in SWING_LABELS and peak < needed:
                        redone += 1
                        log(f"    no clear swing (peak {peak:.0f}); redoing")
                        continue
                    n += 1
                    for t, v in rows:
                        writer.writerow([session, label, n, f"{t - t0:.4f}"] + [f"{x:.3f}" for x in v])
                    f.flush()
                    rows_written += len(rows)
                    log(f"    {label} {n}/{want}  peak {peak:.0f}")
                taken[label] = n
        except KeyboardInterrupt:
            log("\nStopped early. Everything recorded so far is saved.")
    log(f"\nWrote {rows_written} samples to {out_csv}.")
    return {"takes": taken, "redone": redone, "rows": rows_written, "needed": needed}


# --------------------------------------------------------------------- summary

def summarize_csv(path):
    """takes per label, plus the mean and spread of each take's peak gyro."""
    peaks = {}
    cur_key, cur_peak = None, 0.0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["session"], row["label"], row["take"])
            g = math.sqrt(float(row["gx"]) ** 2 + float(row["gy"]) ** 2 + float(row["gz"]) ** 2)
            if key != cur_key:
                if cur_key is not None:
                    peaks.setdefault(cur_key[1], []).append(cur_peak)
                cur_key, cur_peak = key, 0.0
            cur_peak = max(cur_peak, g)
    if cur_key is not None:
        peaks.setdefault(cur_key[1], []).append(cur_peak)
    out = {}
    for label, vals in peaks.items():
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1))
        out[label] = {"takes": len(vals), "mean_peak": m, "std_peak": sd}
    return out


def fresh_rate_from_csv(path):
    """Roughly how many FRESH IMU readings per second the recordings contain:
    the number of times the values changed, over the recorded time. Tells you
    whether the faster notification setting really took effect."""
    changes, duration = 0, 0.0
    key = prev = t_first = t_last = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k = (row["session"], row["label"], row["take"])
            vals = tuple(row[c] for c in ("ax", "ay", "az", "gx", "gy", "gz"))
            t = float(row["t"])
            if k != key:
                if key is not None:
                    duration += t_last - t_first
                key, prev, t_first = k, vals, t
            elif vals != prev:
                changes += 1
                prev = vals
            t_last = t
    if key is not None:
        duration += t_last - t_first
    return changes / duration if duration > 0 else None


def print_summary(path):
    s = summarize_csv(path)
    print(f"\n{path}")
    print(f"  {'label':<10}{'takes':>7}{'mean peak':>12}{'spread':>10}")
    for label in DEFAULT_COUNTS:
        if label in s:
            d = s[label]
            flag = "   <- record more" if d["takes"] < 10 else ""
            print(f"  {label:<10}{d['takes']:>7}{d['mean_peak']:>12.0f}{d['std_peak']:>10.0f}{flag}")
    hz = fresh_rate_from_csv(path)
    if hz:
        print(f"  Fresh IMU readings in these recordings: about {hz:.0f} per second.")


# ------------------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser(description="Probe the hub's IMU and record swings")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="measure update rate, round-trip delay, and swing size")
    mode.add_argument("--record", action="store_true", help="guided capture of labeled swings")
    mode.add_argument("--summary", metavar="CSV", help="summarize an existing recording")
    p.add_argument("--fake", action="store_true", help="use a synthetic hub (no hardware)")
    p.add_argument("--serial", default=None)
    p.add_argument("--color", default=None, help="connection card color, e.g. orange")
    p.add_argument("--out", default="swings.csv")
    p.add_argument("--labels", nargs="+", choices=list(DEFAULT_COUNTS), default=list(DEFAULT_COUNTS))
    p.add_argument("--counts", type=int, default=None, help="takes per label (default: 30/30/30/10/15/10)")
    p.add_argument("--notify-ms", type=int, default=20,
                   help="ask the hub for a notification every this many ms when recording (15-1000; probe finds the best)")
    p.add_argument("--rate", type=int, default=100, help="sampling rate in Hz")
    p.add_argument("--prep", type=float, default=1.5, help="seconds of get-ready before each take")
    p.add_argument("--window", type=float, default=2.5,
                   help="seconds recorded per take (1.5 cut off about a third of swings in the first recording)")
    p.add_argument("--min-peak", type=float, default=None, help="override the 'is this a real swing' threshold")
    return p.parse_args()


def main():
    args = parse_args()
    if args.summary:
        print_summary(args.summary)
        return

    dm = connect_hub(args)
    if args.fake and args.probe:
        dm.scenario = "forehand"            # so the fake produces swings for step 3
    hub = HubIMU(dm) if not args.fake else _FakeAdapter(dm)
    hub.check()
    sampler = ImuSampler(hub.read, rate_hz=args.rate).start()
    try:
        if args.probe:
            run_probe(hub, sampler, hub.read)
        else:
            if hub.set_notification_delay(args.notify_ms):
                print(f"Asked the hub for an IMU notification every {args.notify_ms} ms.")
            counts = {l: (args.counts or DEFAULT_COUNTS[l]) for l in args.labels}
            record_session(hub, sampler, args.labels, counts, args.out, prep_s=args.prep,
                           window_s=args.window, min_peak=args.min_peak,
                           fake=dm if args.fake else None)
            print_summary(args.out)
    finally:
        sampler.stop()
        try:
            dm.disconnect()
        except Exception:
            pass


class _FakeAdapter(HubIMU):
    """HubIMU's cue methods call real legoeducation constants; the fake hub
    doesn't need them, so route cues straight to the fake."""

    def cue_ready(self):
        self.dm.light_color()

    def cue_go(self, beep=True):
        self.dm.beep()

    def kick(self):
        self.dm.motor_run_for_degrees()


if __name__ == "__main__":
    main()
