"""
sounds.py -- short laptop tunes for hits, misses and records (the HW3 songs idea).

Uses Windows' built-in winsound when it's there (nothing to install), otherwise
pyaudio if it's installed (HW3's songs.py approach), otherwise stays silent.
Plays on its own thread and never queues: if a tune is still playing, the new one
is skipped, so the game never waits on audio.
"""

import threading

TUNES = {
    "hit":    [(880, 60)],
    "winner": [(784, 90), (1047, 160)],
    "record": [(523, 90), (659, 90), (784, 90), (1047, 220)],
    "miss":   [(392, 120), (262, 260)],
    "count":  [(660, 120)],
    "serve":  [(990, 160)],
}


def _backend():
    try:
        import winsound
        return lambda f, ms: winsound.Beep(int(f), int(ms))
    except ImportError:
        pass
    try:
        import numpy as np
        import pyaudio
        pa = pyaudio.PyAudio()

        def play(f, ms, rate=44100):
            n = int(rate * ms / 1000)
            t = np.arange(n) / rate
            wave = (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)
            fade = min(n // 2, int(rate * 0.008))
            if fade:
                wave[:fade] *= np.linspace(0, 1, fade)
                wave[-fade:] *= np.linspace(1, 0, fade)
            s = pa.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)
            try:
                s.write(wave.tobytes())
            finally:
                s.close()
        return play
    except Exception:
        return None


class Sounds:
    def __init__(self, enabled=True):
        self.play_fn = _backend() if enabled else None
        self.enabled = self.play_fn is not None
        self._busy = threading.Lock()

    def play(self, name):
        if not self.enabled or name not in TUNES:
            return
        if not self._busy.acquire(blocking=False):
            return

        def run():
            try:
                for f, ms in TUNES[name]:
                    self.play_fn(f, ms)
            except Exception:
                pass
            finally:
                self._busy.release()
        threading.Thread(target=run, daemon=True).start()


def tune_for_events(events):
    """The one tune worth playing for this frame's events (the most important wins)."""
    kinds = {e["type"] for e in events}
    for name in ("record", "miss", "winner", "hit", "serve"):
        if name in kinds:
            return name
    if "countdown" in kinds:
        return "count"
    return None
