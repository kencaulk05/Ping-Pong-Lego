"""
imu_hub.py -- everything that touches the hub's IMU, kept in one small place.

Only HubIMU talks to the real hardware (a connected lelib.doubleMotor). The
sampler, statistics and FakeDoubleMotor have no hardware dependency, which is
what lets the recorder be tested without a hub.

IMU fields come straight from LEGO's docs (doublemotor.imu_device):
    accelerometerX/Y/Z, gyroscopeX/Y/Z, yaw, pitch, roll
The units are whatever the hub reports (raw counts), so nothing downstream
assumes degrees or g; thresholds are derived from your own recordings.
"""

import math
import random
import threading
import time
from collections import deque

FIELDS = ("accelerometerX", "accelerometerY", "accelerometerZ",
          "gyroscopeX", "gyroscopeY", "gyroscopeZ",
          "yaw", "pitch", "roll")


def gyro_mag(v):
    return math.sqrt(v[3] ** 2 + v[4] ** 2 + v[5] ** 2)


def accel_mag(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


# --------------------------------------------------------------- real hardware

class HubIMU:
    """Thin adapter over a connected lelib.doubleMotor."""

    def __init__(self, dm):
        self.dm = dm
        self._warned = set()

    def read(self):
        d = self.dm.imu_device
        return tuple(float(getattr(d, f)) for f in FIELDS)

    def check(self):
        """Fail early, with a clear message, if the IMU isn't readable."""
        try:
            self.read()
        except Exception as exc:
            raise RuntimeError(
                f"Couldn't read the hub's IMU ({type(exc).__name__}: {exc}). "
                "Is the Double Motor connected, and does help(le.DoubleMotor) "
                "list imu_device with gyroscopeX etc.?") from exc

    def set_notification_delay(self, delay_ms):
        """Ask the hub to send its automatic notifications (IMU, gesture, motors)
        every delay_ms milliseconds. LEGO's docs: 0 turns them off, otherwise
        15 to 1000. Returns True if the hub accepted the request."""
        delay_ms = max(15, min(1000, int(delay_ms)))
        try:
            self.dm.device_notification_request(delay_ms)
            return True
        except Exception as exc:
            if "notify" not in self._warned:
                self._warned.add("notify")
                print(f"[hub] WARNING: device_notification_request({delay_ms}) failed "
                      f"({type(exc).__name__}: {exc}); the rate will stay at the hub's default")
            return False

    def cue_ready(self):
        self._try("cue", lambda le: self.dm.light_color(le.LEGO_COLOR_YELLOW, intensity=100))

    def cue_go(self, beep=True):
        def go(le):
            self.dm.light_color(le.LEGO_COLOR_GREEN, intensity=100)
            if beep:
                self.dm.beep(pattern=le.SOUND_PATTERN_BEEP_SINGLE, frequency=880)
        self._try("cue", go)

    def kick(self):
        """A short, sharp motor twitch, used only to measure round-trip delay."""
        def run(le):
            try:
                self.dm.motor_run_for_degrees(60, motor=le.MOTOR_LEFT, speed=100, blocking=False)
            except TypeError:
                self.dm.motor_run_for_degrees(60, motor=le.MOTOR_LEFT, speed=100)
        self._try("kick", run)

    def _try(self, name, fn):
        try:
            import legoeducation as le
            fn(le)
        except Exception as exc:          # cues are nice-to-have, never fatal
            if name not in self._warned:
                self._warned.add(name)
                print(f"[hub] WARNING: {name} failed ({type(exc).__name__}: {exc}); continuing without it")


# -------------------------------------------------------------------- sampling

class ImuSampler:
    """Reads the IMU at a fixed rate on a background thread and keeps a short
    history, as (time, values) pairs. Fixed-rate sampling keeps the recorded
    data uniform; whether the hub actually reports FRESH data that often is
    what the probe measures separately. A reading is only kept if two reads in
    a row agree, so a half-updated tuple (some fields old, some new) never
    ends up in a recording."""

    def __init__(self, read_fn, rate_hz=100, history_s=30):
        self.read_fn = read_fn
        self.dt = 1.0 / rate_hz
        self._buf = deque(maxlen=int(history_s * rate_hz))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.errors = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        next_t = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                v = read_stable(self.read_fn)        # skip a half-updated (torn) reading
                if v is not None:
                    with self._lock:
                        self._buf.append((t, v))
            except Exception:
                self.errors += 1
            next_t += self.dt
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def since(self, t0, t1=None):
        with self._lock:
            items = list(self._buf)
        return [(t, v) for t, v in items if t >= t0 and (t1 is None or t <= t1)]


# ------------------------------------------------------------------ statistics

def interval_stats(times):
    """Update rate and gap statistics from the timestamps of fresh readings."""
    if len(times) < 3:
        return None
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    n = len(gaps)
    duration = times[-1] - times[0]
    return {"updates": len(times),
            "hz": (len(times) - 1) / duration if duration > 0 else 0.0,
            "median_ms": gaps[n // 2] * 1000,
            "p95_ms": gaps[min(n - 1, int(0.95 * n))] * 1000,
            "max_ms": gaps[-1] * 1000}


def measure_update_rate(read_fn, seconds=5.0, min_gap=0.003):
    """How often does the hub deliver FRESH values? Polls as fast as it can and
    notes each time the reading changes. Move the hub a little while this runs
    (a perfectly still sensor can repeat identical values).

    Reading nine fields one after another can straddle an update, giving a
    half-old, half-new tuple that looks like a second change a millisecond
    after the first. A change within `min_gap` of the previous one is counted as
    the same update, so torn reads don't inflate the rate."""
    times, last = [], None
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        v = read_fn()
        if v != last:
            now = time.perf_counter()
            if not times or now - times[-1] >= min_gap:
                times.append(now)
            last = v
        time.sleep(0.0005)
    return interval_stats(times)


def sweep_notification_rates(hub, read_fn, delays=(100, 50, 30, 20, 15), seconds=3.0, settle=0.4):
    """Request each notification delay in turn and measure what the hub really
    delivers. Asking for 15 ms doesn't mean you get 15 ms; Bluetooth may not keep up."""
    results = []
    for d in delays:
        ok = hub.set_notification_delay(d)
        time.sleep(settle)
        results.append({"delay_ms": d, "requested_hz": 1000.0 / d, "accepted": ok,
                        "stats": measure_update_rate(read_fn, seconds)})
    return results


def recommend_delay(results, keep_up=0.85):
    """The fastest request the hub actually kept up with (delivering at least
    keep_up x the requested rate), or None if nothing worked."""
    good = [r for r in results
            if r["accepted"] and r["stats"] and r["stats"]["hz"] >= keep_up * r["requested_hz"]]
    return min((r["delay_ms"] for r in good), default=None)


def read_stable(read_fn):
    """Two reads in a row must agree before a value is trusted. Reading several
    fields one after another can straddle an update and return a half-old,
    half-new tuple; such a tuple never repeats, so it is skipped. Returns None
    when the two reads disagree (try again in a moment)."""
    a = read_fn()
    b = read_fn()
    return a if a == b else None


def percentile(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def idle_noise(sampler, seconds=1.5):
    """Mean and spread of the gyro and accel magnitudes while the hub is still."""
    t0 = time.perf_counter()
    time.sleep(seconds)
    rows = sampler.since(t0)
    g = [gyro_mag(v) for _, v in rows]
    a = [accel_mag(v) for _, v in rows]
    return {"gyro_mean": _mean(g), "gyro_std": _std(g), "accel_mean": _mean(a), "accel_std": _std(a), "n": len(rows)}


def measure_kick_latency(hub, sampler, trials=8, timeout=1.0, settle=1.2, baseline_s=0.5):
    """Round trip: command a motor twitch, then watch for the first sample where
    the IMU deviates from its resting level. Includes the command's trip to the
    hub, the motor spinning up, and the IMU data's trip back, so it OVERestimates
    the one-way delay of a swing reaching the game."""
    results = []
    for _ in range(trials):
        time.sleep(settle)                                # let it settle
        base = idle_noise(sampler, baseline_s)
        thr_g = max(8 * base["gyro_std"], 2.0)
        thr_a = max(8 * base["accel_std"], 2.0)
        t_cmd = time.perf_counter()
        hub.kick()
        found = None
        deadline = t_cmd + timeout
        while time.perf_counter() < deadline and found is None:
            for t, v in sampler.since(t_cmd):
                if (abs(gyro_mag(v) - base["gyro_mean"]) > thr_g
                        or abs(accel_mag(v) - base["accel_mean"]) > thr_a):
                    found = t - t_cmd
                    break
            time.sleep(0.005)
        results.append(found)
    return results


def swing_peak(rows):
    """Largest gyro magnitude seen in a recorded window."""
    return max((gyro_mag(v) for _, v in rows), default=0.0)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ------------------------------------------------------------------- fake hub

class _FakeImuDevice:
    def __init__(self, owner):
        self._o = owner

    def __getattr__(self, name):
        if name in FIELDS:
            return self._o._value(name)
        raise AttributeError(name)


class FakeDoubleMotor:
    """Stands in for lelib.doubleMotor so the recorder can be tested (and its
    output previewed) with no hardware. Produces noisy readings that refresh at
    `rate_hz`, plus plausible-looking pulses for each kind of motion. Set
    `.scenario` to a label before each take. The shapes are invented, so a
    classifier trained on this proves nothing about the real hub."""

    def __init__(self, rate_hz=50, rtt_s=0.08, weak_swing_rate=0.0, seed=0, max_rate_hz=None):
        self.connected = True
        self.rate_hz = rate_hz
        self.max_rate_hz = max_rate_hz       # simulate a link that can't go faster than this
        self.rtt_s = rtt_s
        self.weak_swing_rate = weak_swing_rate
        self.scenario = "idle"
        self.imu_device = _FakeImuDevice(self)
        self._rng = random.Random(seed)
        self._seed = seed
        self._pulses = []                 # (t_start, duration, gyro(3), accel(3))
        self._motion = None               # (kind, t_start)

    # -- what the recorder calls on a real hub
    def light_color(self, *a, **k):
        pass

    def device_notification_request(self, delay_ms, **k):
        if delay_ms != 0 and not 15 <= delay_ms <= 1000:
            raise ValueError("delay_ms must be 0 or between 15 and 1000")
        if delay_ms:
            rate = 1000.0 / delay_ms
            self.rate_hz = min(rate, self.max_rate_hz) if self.max_rate_hz else rate

    def beep(self, *a, **k):
        t0 = time.perf_counter()
        self._pulses = []
        self._motion = None
        s, r = self.scenario, self._rng
        amp = r.uniform(2500, 4500) * (0.005 if r.random() < self.weak_swing_rate else 1.0)
        acc = r.uniform(800, 1600) * (0.05 if amp < 300 else 1.0)
        when = t0 + r.uniform(0.1, 0.3)
        if s == "forehand":
            self._pulses.append((when, 0.22, (0, 0, +amp), (+acc, 0, 0)))
        elif s == "backhand":
            self._pulses.append((when, 0.22, (0, 0, -amp), (-acc, 0, 0)))
        elif s == "flat":
            self._pulses.append((when, 0.22, (+amp, 0, 0), (0, +acc, 0)))
        elif s in ("shake", "carry"):
            self._motion = (s, t0)

    def motor_run_for_degrees(self, *a, **k):
        self._pulses.append((time.perf_counter() + self.rtt_s, 0.15, (900, 600, 0), (400, 0, 0)))

    def motor_run_for_time(self, time_ms, *a, **k):
        """A haptic buzz: the off-center weight shakes the whole hub while it spins."""
        t0 = time.perf_counter() + self.rtt_s
        dur = time_ms / 1000.0
        self._pulses.append((t0, dur, (180, 140, 90), (120, 90, 60)))

    def disconnect(self):
        self.connected = False

    # -- the model
    def _value(self, name):
        tick = math.floor(time.perf_counter() * self.rate_hz)
        t = tick / self.rate_hz
        i = FIELDS.index(name)
        noise = random.Random(tick * 7919 + self._seed + i * 104729)    # independent noise per axis
        base = 1000.0 if name == "accelerometerZ" else 0.0
        v = base + noise.gauss(0, 3.0)
        g, a = [0.0] * 3, [0.0] * 3
        for t_start, dur, pg, pa in self._pulses:
            tau = t - t_start
            if 0 <= tau < dur:
                s = math.sin(math.pi * tau / dur)
                g = [g[k] + pg[k] * s for k in range(3)]
                a = [a[k] + pa[k] * s for k in range(3)]
        if self._motion:
            kind, t0 = self._motion
            tau = t - t0
            if kind == "shake" and 0.2 <= tau < 1.2:
                g[0] += 900 * math.sin(2 * math.pi * 6 * tau)
                a[1] += 700 * math.sin(2 * math.pi * 6 * tau + 1)
            elif kind == "carry":
                g[1] += 250 * math.sin(2 * math.pi * 1.5 * tau)
                a[2] += 300 * math.sin(2 * math.pi * 1.5 * tau)
        if name.startswith("gyro"):
            v += g["XYZ".index(name[-1])]
        elif name.startswith("accel"):
            v += a["XYZ".index(name[-1])]
        return v
