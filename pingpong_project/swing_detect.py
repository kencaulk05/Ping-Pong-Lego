"""
swing_detect.py -- find swings in the hub's IMU stream and say which kind.

Training (train_swing_classifier.py) and live use both call THIS file, so the
classifier is always trained on exactly what it will see in play. Needs only
numpy (the training script is the only thing that needs scikit-learn).

How it works, in three steps:

  1. Candidate: a reading is a swing candidate if its gyro magnitude is the
     largest within +/-NMS readings, above an absolute floor (T_ABS), and above
     K_BASE x the player's recent movement level. That last part is why it's an
     ADAPTIVE threshold: a hand holding a hub is never still (typically 100-300
     in your recordings, versus ~5 on a table), so a fixed threshold either
     fires constantly or misses gentle swings.
  2. Features: a small window around the candidate (H readings each side) is
     summarized into ~17 numbers: how big, which way it rotated, how long it
     lasted, how many peaks it had, how much the accelerometer moved.
  3. Classify: logistic regression picks forehand / backhand / flat / other.
     "other" is shaking, walking, and anything else that isn't a swing.

Latency: an event is only complete H readings after its peak (about 0.26 s at
~31 readings per second), so that is how late it is reported. `age` on the
Swing says how long ago the peak was, so the game can place it correctly.

CHANGING H, NMS, BASE_N, K_BASE OR T_ABS MEANS RETRAINING: the saved model
records the values it was trained with and refuses to load if they differ.
"""

import json
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# --- detector settings (the model is trained with these exact values) --------
H = 8            # readings kept on each side of a peak (~0.25 s each side at ~31 Hz)
NMS = 6          # an event must be the largest reading within +/-NMS readings
BASE_N = 32      # baseline = median gyro magnitude over the previous BASE_N readings
K_BASE = 1.5     # the peak must exceed K_BASE x the recent baseline ...
T_ABS = 100.0    # ... and this floor, in the hub's raw gyro units

DETECTOR_PARAMS = {"H": H, "NMS": NMS, "BASE_N": BASE_N, "K_BASE": K_BASE, "T_ABS": T_ABS}

FEATURE_NAMES = ["log_peak", "ux", "uy", "uz", "net_x", "net_y", "net_z", "width", "n_peaks",
                 "outside_energy", "flips", "log_rng_ax", "log_rng_ay", "log_rng_az", "log_rng_a",
                 "log_baseline", "peak_over_baseline"]
# The two context features describe how much the player was moving BEFORE the swing,
# not the swing itself. They are left out of the model: the recording only ever had
# swings made while the hand was already moving (typically 100-300), so the model
# learned "a quiet hand before the swing means a flat shot", and called almost every
# swing made from a still hand "flat" (checked on real swings: 22 of 24 forehands and
# 24 of 24 backhands). Cross-validation could not show this, because every training
# swing had the same kind of context.
CONTEXT_FEATURES = ("log_baseline", "peak_over_baseline")

SWING_KINDS = ("forehand", "backhand", "flat")


def baseline_at(g, i):
    """Recent movement level just before reading i."""
    if i > 3:
        return float(np.median(g[max(0, i - BASE_N):i]))
    return float(np.mean(g[:i + 1]))


def event_features(G, A, base):
    """Summarize a window of 2H+1 readings. G and A are (2H+1, 3) gyro and
    accelerometer arrays, the peak reading is at index H."""
    gm = np.linalg.norm(G, axis=1)
    pk = float(gm.max())
    unit = G[H] / (np.linalg.norm(G[H]) + 1e-9)                    # which way it was rotating at the peak
    net = G.sum(0) / (np.abs(G).sum() + 1e-9)                      # net signed rotation, scale-free
    width = float((gm > 0.5 * pk).sum())                           # readings above half the peak
    n_peaks = float(sum(1 for k in range(1, len(gm) - 1)
                        if gm[k] > gm[k - 1] and gm[k] >= gm[k + 1] and gm[k] > 0.4 * pk))
    outside = float(((gm[:H - 2] ** 2).sum() + (gm[H + 3:] ** 2).sum()) / ((gm ** 2).sum() + 1e-9))
    dom = int(np.abs(G[H]).argmax())
    flips = float((np.diff(np.sign(G[:, dom])) != 0).sum())
    rng = A.max(0) - A.min(0)
    return np.array([np.log1p(pk), *unit, *net, width, n_peaks, outside, flips,
                     *np.log1p(rng), np.log1p(np.linalg.norm(rng)),
                     np.log1p(base), pk / (base + 1.0)])


def isolate_swing(G, A, surround=0.25, keep=2):
    """Emulate the same swing made from a nearly still hand: everything farther than `keep`
    readings from the peak is turned down to `surround` of its size. Returns (G, A).
    Used to check that a model judges the swing's shape, not how much the hand was moving
    around it."""
    k = np.abs(np.arange(len(G)) - H)
    w = np.where(k <= keep, 1.0, surround)[:, None]
    mid = np.median(A, axis=0)
    return G * w, mid + (A - mid) * w


@dataclass
class Event:
    t_peak: float            # time of the peak reading
    t_ready: float           # time of the reading that completed the window
    index: int               # position of the peak in the stream of fresh readings
    peak: float              # peak gyro magnitude
    base: float              # movement level before it
    features: np.ndarray = field(repr=False)
    window: object = field(default=None, repr=False)      # (gyro, accel) readings around the peak, (2H+1, 3) each

    @property
    def age(self):
        return self.t_ready - self.t_peak


class SwingDetector:
    """Feed it every reading; it returns the swing candidates it has just
    completed. Only fresh readings count: a value identical to the previous one
    is the same reading seen twice (we poll faster than the hub updates), and is
    ignored."""

    def __init__(self):
        self._buf = deque(maxlen=BASE_N + 2 * H + 6)     # (t, ax, ay, az, gx, gy, gz, |g|)
        self._last = None
        self.n_fresh = 0                                 # fresh readings seen so far

    def push(self, t, values):
        v = tuple(float(x) for x in values[:6])
        if v == self._last:
            return []
        self._last = v
        g = float(np.linalg.norm(v[3:6]))
        self._buf.append((t,) + v + (g,))
        self.n_fresh += 1

        n = len(self._buf)
        c = n - 1 - H                       # the reading whose window just became complete
        if c < 0 or self.n_fresh - 1 - H < H:     # need H real readings before the peak too
            return []
        arr = np.array(self._buf)
        gm = arr[:, 7]
        lo, hi = max(0, c - NMS), min(n, c + NMS + 1)
        is_peak = (gm[c] > gm[lo:c].max() if c > lo else True) and gm[c] >= gm[c + 1:hi].max()
        if not (is_peak and gm[c] > T_ABS):
            return []
        base = baseline_at(gm, c)
        if gm[c] <= K_BASE * base:
            return []
        idx = np.arange(c - H, c + H + 1)
        W = arr[idx]
        return [Event(t_peak=float(arr[c, 0]), t_ready=float(arr[-1, 0]),
                      index=self.n_fresh - 1 - H, peak=float(gm[c]), base=base,
                      features=event_features(W[:, 4:7], W[:, 1:4], base),
                      window=(W[:, 4:7].copy(), W[:, 1:4].copy()))]


def events_from_rows(times, rows):
    """Replay a recording through a fresh detector (used by training and tests)."""
    det = SwingDetector()
    out = []
    for t, v in zip(times, rows):
        out += det.push(t, v)
    return out


# --------------------------------------------------------------------- the model

class SwingModel:
    """A standardized multinomial logistic regression, evaluated with numpy."""

    def __init__(self, data):
        self.meta = data
        self.classes = list(data["classes"])
        self.feature_names = list(data["feature_names"])
        # A model may use only some of the features (the saved list says which), so it can be
        # trained to ignore ones that proved misleading. Old full-feature models still work.
        self._idx = [FEATURE_NAMES.index(n) for n in self.feature_names]
        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        self.coef = np.array(data["coef"])
        self.intercept = np.array(data["intercept"])
        self.expected_hz = float(data.get("expected_hz", 31.0))
        self.power_range = tuple(data.get("power_range", (200.0, 1500.0)))
        # Events smaller than this are never swings. In still-hand runs, ordinary hand movement
        # (repositioning, ready position, waving) peaked around 100-290 while real swings peaked
        # at 370 and up, so a floor between them removes false alarms without losing swings.
        self.min_peak = float(data.get("min_swing_peak", 0.0))
        saved = data.get("detector", {})
        if saved != DETECTOR_PARAMS:
            raise ValueError(f"model was trained with detector settings {saved}, but swing_detect.py "
                             f"now uses {DETECTOR_PARAMS}. Retrain with train_swing_classifier.py.")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def proba(self, features):
        x = np.asarray(features)[self._idx]
        z = ((x - self.mean) / self.scale) @ self.coef.T + self.intercept
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, features):
        p = self.proba(features)
        k = int(p.argmax())
        return self.classes[k], float(p[k]), p

    def power(self, peak):
        lo, hi = self.power_range
        return float(min(1.0, max(0.0, (peak - lo) / (hi - lo))))


@dataclass
class Recognized:
    kind: str                # forehand | backhand | flat
    power: float             # 0..1, from the peak gyro relative to your recorded swings
    age: float               # seconds since the peak (the detection latency)
    confidence: float        # "swing" rule: P(it's a swing of any kind); "kind" rule: P(this kind)
    peak: float
    t_peak: float
    features: object = field(default=None, repr=False)     # what the model saw (see FEATURE_NAMES)

    def to_swing(self):
        from pingpong_core import Swing
        return Swing(kind=self.kind, power=self.power, age=self.age)


ACCEPT_RULES = ("swing", "kind")
DEFAULT_CUTOFF = {"swing": 0.9, "kind": 0.6}


class SwingRecognizer:
    """Detector + classifier. push() takes every IMU reading and returns the
    swings recognized so far; rejected events (classed "other", or low
    confidence) are kept in .rejected for debugging.

    Two ways to accept an event as a swing (eval_rules.py compares them on every
    saved run, each scored by a model that never saw it):

      "swing" (default)  P(forehand) + P(backhand) + P(flat) >= cutoff (0.9), and
                         the kind is the likeliest of the three. Asks "was that a
                         swing at all?", which is what decides a hit in the game.
                         98% of 150 unseen swings accepted, 95% named right,
                         0 false alarms in 3.75 minutes of random movement.
      "kind"             the single likeliest class is a swing kind with
                         probability >= cutoff (0.6). The original rule: it threw
                         away swings it couldn't name confidently (92% accepted).

    The real swing/non-swing filter is the minimum swing size (min_peak): in your
    runs no random movement reached it. The cutoff catches what does."""

    def __init__(self, model, min_confidence=None, min_peak=None, rule="swing"):
        if rule not in ACCEPT_RULES:
            raise ValueError(f"rule must be one of {ACCEPT_RULES}")
        self.model = model
        self.rule = rule
        self.min_confidence = DEFAULT_CUTOFF[rule] if min_confidence is None else float(min_confidence)
        self.min_peak = model.min_peak if min_peak is None else float(min_peak)
        self.detector = SwingDetector()
        self.rejected = []
        self._times = deque(maxlen=200)         # arrival times of fresh readings, for a rate check

    def push(self, t, values):
        before = self.detector.n_fresh
        events = self.detector.push(t, values)
        if self.detector.n_fresh != before:
            self._times.append(t)
        swings = []
        for ev in events:
            if ev.peak < self.min_peak:                     # too small to be a swing, whatever it looks like
                self.rejected.append((ev, "too small", 1.0))
                continue
            kind, conf, probs = self.model.predict(ev.features)
            if self.rule == "swing":
                p = dict(zip(self.model.classes, probs))
                p_swing = 1.0 - p.get("other", 0.0)
                best = max(SWING_KINDS, key=lambda k: p.get(k, 0.0))
                if p_swing >= self.min_confidence:
                    swings.append(Recognized(best, self.model.power(ev.peak), ev.age, p_swing, ev.peak,
                                             ev.t_peak, ev.features))
                else:
                    self.rejected.append((ev, kind, p_swing if kind in SWING_KINDS else conf))
            elif kind in SWING_KINDS and conf >= self.min_confidence:
                swings.append(Recognized(kind, self.model.power(ev.peak), ev.age, conf, ev.peak, ev.t_peak, ev.features))
            else:
                self.rejected.append((ev, kind, conf))
        return swings

    def fresh_rate(self, window_s=3.0):
        """Fresh readings per second over the last few seconds, or None."""
        if len(self._times) < 5:
            return None
        recent = [x for x in self._times if x >= self._times[-1] - window_s]
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else None

    def rate_ok(self, tolerance=0.3):
        """The model's window assumes about expected_hz readings a second. If the
        hub is sending much faster or slower, the windows cover the wrong amount
        of time and the classification can't be trusted."""
        hz = self.fresh_rate()
        if hz is None:
            return True
        return abs(hz - self.model.expected_hz) <= tolerance * self.model.expected_hz
