"""
hub_input.py -- swings from the hub's IMU, for the live game.

The hub is read on its own thread (the lesson from swing_check.py: anything that
waits on the hub, like a beep or a light command, must never pause the reading).
Every fresh reading goes through the same SwingRecognizer that swing_check.py
scored at 92% on runs it had never seen.

Timing. A swing is only recognized about 0.26 s after its peak, and the game loop
picks it up a little later still (a camera frame, up to ~60 ms with MediaPipe
running). So the age handed to the game is measured at the moment of handoff,
`now - t_peak`, not when the recognizer finished. That way the game places the
swing at the instant you actually swung.
"""

import threading
import time
from collections import deque

from imu_hub import gyro_mag, read_stable


class SwingFeed:
    """Thread-safe: the reader thread calls process(), the game loop calls drain().
    No threads of its own, so it can be tested by feeding it recorded readings."""

    def __init__(self, recognizer, keep_readings=True):
        self.rec = recognizer
        self.lock = threading.Lock()
        self._ready = []                         # Recognized swings waiting for the game
        self.rejected = deque(maxlen=20)         # (t, kind, conf, peak) of events that weren't swings
        self.gyro = deque(maxlen=200)            # (t, |gyro|) for the on-screen signal strip
        self.readings = [] if keep_readings else None   # (t, values) for the saved stream
        self._last = None
        self.n_recognized = 0

    def process(self, t, values):
        with self.lock:
            if values != self._last:
                self._last = values
                self.gyro.append((t, gyro_mag(values)))
                if self.readings is not None:
                    self.readings.append((t, values))
            for s in self.rec.push(t, values):
                self._ready.append(s)
                self.n_recognized += 1
            for ev, kind, conf in self.rec.rejected:
                self.rejected.append((t, kind, conf, ev.peak))
            self.rec.rejected.clear()

    def drain(self, now):
        """Swings recognized since the last call, as (Recognized, Swing) pairs, with
        Swing.age measured from the swing's peak to `now` (same clock as process())."""
        with self.lock:
            out, self._ready = self._ready, []
        pairs = []
        for s in out:
            sw = s.to_swing()
            sw.age = max(0.0, now - s.t_peak)
            pairs.append((s, sw))
        return pairs

    def latest_gyro(self):
        with self.lock:
            return self.gyro[-1][1] if self.gyro else 0.0

    def fresh_rate(self):
        with self.lock:
            return self.rec.fresh_rate()

    def rate_ok(self):
        with self.lock:
            return self.rec.rate_ok()


class HubReader:
    """Polls the hub on a background thread and feeds a SwingFeed."""

    def __init__(self, hub, feed, clock=time.perf_counter):
        self.hub, self.feed, self.clock = hub, feed, clock
        self.errors = 0
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
                self.feed.process(self.clock(), v)
            time.sleep(0.0005)
