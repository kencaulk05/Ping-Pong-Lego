"""
haptics.py -- the paddle talks back: buzz, light and beep on the hub in your hand.

Buzz: an off-center weight on the hub's LEFT motor shaft (a LEGO beam or a few
bricks pushed on lopsided). Spinning it briefly shakes the hub, like a phone's
vibration motor. Without the weight the motor still runs but you'll barely feel it;
the light and beep still work.

What you feel and see:
    ball incoming        light yellow
    swing window open    light green           (the "now!" cue, on the paddle itself)
    hit                  short buzz (harder hit = longer) + high beep
    winner               medium buzz + double beep, light azure
    new record           light magenta, breathing
    miss                 long rumble + low beep, light red
    countdown 3-2-1      a beep each second, a higher one on the serve

Two rules learned the hard way in swing_check.py:
  * Never wait on the hub. Every command is sent with blocking=False (LEGO's API
    supports it on beep, light_color and motor_run_for_time), from a worker thread,
    through a short queue. If the hub falls behind, old cues are dropped rather than
    piling up late.
  * The buzz shakes the IMU. Buzzes are kept short and only happen right after a
    hit or a miss, when the game isn't listening for swings, and
    `python pingpong_live.py --check-haptics` measures whether one ever registers
    as a swing.
"""

import queue
import threading


class Haptics:
    def __init__(self, dm=None, le=None, enabled=True, motor="left", buzz_speed=100):
        self.dm = dm
        self.enabled = enabled and dm is not None
        self.le = le
        if self.enabled and self.le is None:
            import legoeducation
            self.le = legoeducation
        self.motor = motor
        self.buzz_speed = buzz_speed
        self.q = queue.Queue(maxsize=3)
        self.sent = []                  # (name, [calls]) for tests and debugging
        self.errors = 0
        self._warned = False
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ public

    def start(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            try:
                self.q.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=1.0)

    def cue(self, name, power=0.5):
        """Queue a cue by name. Never blocks; drops the oldest cue if the queue is full."""
        if not self.enabled:
            return
        item = (name, power)
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(item)
            except queue.Full:
                pass

    def play_now(self, name, power=0.5):
        """Send a cue's commands on the calling thread (still non-blocking on the hub)."""
        calls = self.commands(name, power)
        for fn, args, kwargs in calls:
            try:
                getattr(self.dm, fn)(*args, blocking=False, **kwargs)
            except TypeError:
                getattr(self.dm, fn)(*args, **kwargs)        # an older library without `blocking`
            except Exception as exc:
                self.errors += 1
                if not self._warned:
                    self._warned = True
                    print(f"[haptics] WARNING: {fn} failed ({type(exc).__name__}: {exc}); continuing")
        self.sent.append((name, calls))

    # --------------------------------------------------------------- the cues

    def _c(self, name, fallback):
        return getattr(self.le, name, getattr(self.le, fallback, 0))

    def commands(self, name, power=0.5):
        """The hub commands for one cue, as (method, args, kwargs). Pure, so it can be tested."""
        light = lambda color, **kw: ("light_color", (self._c(f"LEGO_COLOR_{color}", "LEGO_COLOR_GREEN"),), kw)
        beep = lambda freq, pattern="SINGLE": ("beep", (), {
            "pattern": self._c(f"SOUND_PATTERN_BEEP_{pattern}", "SOUND_PATTERN_BEEP_SINGLE"), "frequency": freq})
        side = self._c("MOTOR_LEFT" if self.motor == "left" else "MOTOR_RIGHT", "MOTOR_LEFT")
        buzz = lambda ms: ("motor_run_for_time", (int(ms),), {"motor": side, "speed": self.buzz_speed})

        power = min(1.0, max(0.0, power))
        if name == "incoming":
            return [light("YELLOW")]
        if name == "window":
            return [light("GREEN")]
        if name == "hit":
            return [buzz(90 + 110 * power), beep(880)]
        if name == "winner":
            return [light("AZURE"), buzz(220), beep(1047, "DOUBLE")]
        if name == "record":
            return [("light_color", (self._c("LEGO_COLOR_MAGENTA", "LEGO_COLOR_PURPLE"),),
                     {"pattern": self._c("LIGHT_PATTERN_BREATHE", "LIGHT_PATTERN_SOLID")})]
        if name == "miss":
            return [light("RED"), buzz(450), beep(220)]
        if name == "count":
            return [beep(660)]
        if name == "serve":
            return [beep(990)]
        if name == "idle":
            return [light("PURPLE")]
        raise ValueError(f"unknown haptic cue {name!r}")

    # ------------------------------------------------------------------ worker

    def _run(self):
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                break
            self.play_now(*item)


def cues_for_events(events):
    """Which cues the game's events call for, in order (pure, tested)."""
    out = []
    for e in events:
        t = e["type"]
        if t == "countdown":
            out.append(("count", 0.5))
        elif t == "serve":
            out += [("serve", 0.5), ("incoming", 0.5)]
        elif t == "incoming":
            out.append(("incoming", 0.5))
        elif t == "hit":
            out.append(("hit", e.get("power", 0.5)))
        elif t == "winner":
            out.append(("winner", 0.5))
        elif t == "record":
            out.append(("record", 0.5))
        elif t == "miss":
            out.append(("miss", 0.5))
        elif t == "aborted":
            out.append(("idle", 0.5))
    return out
