"""
test_imu.py -- tests for the IMU recorder, run against a FAKE hub.

These check the tooling (timing statistics, guided capture, redo logic, CSV
format, summary), not how your real hub behaves. That's what running the
recorder on the real hub is for.

    python test_imu.py        (or: pytest test_imu.py)
"""

import builtins
import csv
import os
import tempfile

import imu_recorder as rec
from imu_hub import (FakeDoubleMotor, HubIMU, ImuSampler, interval_stats, measure_kick_latency,
                     measure_update_rate, recommend_delay, sweep_notification_rates)


class FakeAdapter(rec._FakeAdapter):
    pass


def fake_setup(**kw):
    dm = FakeDoubleMotor(**kw)
    hub = FakeAdapter(dm)
    hub.check()
    sampler = ImuSampler(hub.read, rate_hz=100).start()
    return dm, hub, sampler


def test_interval_stats():
    s = interval_stats([i * 0.020 for i in range(51)])
    assert abs(s["hz"] - 50.0) < 1e-6
    assert abs(s["median_ms"] - 20.0) < 1e-6 and abs(s["max_ms"] - 20.0) < 1e-6
    assert interval_stats([0.0, 0.1]) is None, "too few points to say anything"


def test_update_rate_is_measured_from_fresh_readings_not_poll_rate():
    dm = FakeDoubleMotor(rate_hz=50)
    s = measure_update_rate(FakeAdapter(dm).read, seconds=1.5)
    assert 45 <= s["hz"] <= 55, s          # polled at ~2 kHz, but data refreshes at 50 Hz
    dm = FakeDoubleMotor(rate_hz=20)
    s = measure_update_rate(FakeAdapter(dm).read, seconds=2.5)
    # The fake produces torn reads (fields from two different ticks in one
    # reading); the rate must still come out at 20, not inflated by them.
    assert 18 <= s["hz"] <= 22, s


def test_round_trip_measurement_tracks_the_real_delay():
    dm, hub, sampler = fake_setup(rtt_s=0.08)
    try:
        lat = measure_kick_latency(hub, sampler, trials=3, settle=0.3, baseline_s=0.3)
    finally:
        sampler.stop()
    assert all(x is not None for x in lat), lat
    med = sorted(lat)[1]
    assert 0.06 < med < 0.25, med


def test_guided_recording_counts_and_format():
    dm, hub, sampler = fake_setup()
    path = os.path.join(tempfile.mkdtemp(), "swings.csv")
    try:
        out = rec.record_session(hub, sampler, ["forehand", "backhand", "idle", "shake"],
                                 {"forehand": 3, "backhand": 2, "idle": 2, "shake": 2}, path,
                                 prep_s=0.05, window_s=0.6, fake=dm, log=lambda *a: None,
                                 label_pause=0.05, noise_s=0.4)
    finally:
        sampler.stop()
    assert out["takes"] == {"forehand": 3, "backhand": 2, "idle": 2, "shake": 2}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        assert next(reader) == rec.CSV_HEADER
        rows = list(reader)
    assert len(rows) == out["rows"] and len(rows) > 100
    assert all(len(r) == len(rec.CSV_HEADER) for r in rows)
    assert {r[1] for r in rows} == {"forehand", "backhand", "idle", "shake"}
    assert float(rows[0][3]) >= 0.0, "t is relative to the GO cue"


def test_takes_without_a_clear_swing_are_redone_and_not_counted():
    dm, hub, sampler = fake_setup(weak_swing_rate=0.6, seed=3)
    path = os.path.join(tempfile.mkdtemp(), "swings.csv")
    try:
        out = rec.record_session(hub, sampler, ["forehand"], {"forehand": 4}, path,
                                 prep_s=0.05, window_s=0.6, fake=dm, log=lambda *a: None,
                                 label_pause=0.05, noise_s=0.4)
    finally:
        sampler.stop()
    assert out["takes"]["forehand"] == 4
    assert out["redone"] >= 1, "some weak swings should have been rejected"
    s = rec.summarize_csv(path)
    assert s["forehand"]["takes"] == 4
    assert s["forehand"]["mean_peak"] > out["needed"], "every saved swing cleared the bar"


def test_summary_separates_swings_from_non_swings():
    dm, hub, sampler = fake_setup()
    path = os.path.join(tempfile.mkdtemp(), "swings.csv")
    try:
        rec.record_session(hub, sampler, ["flat", "idle"], {"flat": 2, "idle": 2}, path,
                           prep_s=0.05, window_s=0.6, fake=dm, log=lambda *a: None,
                           label_pause=0.05, noise_s=0.4)
    finally:
        sampler.stop()
    s = rec.summarize_csv(path)
    assert s["flat"]["takes"] == 2 and s["idle"]["takes"] == 2
    assert s["flat"]["mean_peak"] > 20 * s["idle"]["mean_peak"], s


def test_second_session_appends_without_a_second_header():
    path = os.path.join(tempfile.mkdtemp(), "swings.csv")
    for _ in range(2):
        dm, hub, sampler = fake_setup()
        try:
            rec.record_session(hub, sampler, ["idle"], {"idle": 1}, path, prep_s=0.05,
                               window_s=0.3, fake=dm, log=lambda *a: None,
                               label_pause=0.05, noise_s=0.3)
        finally:
            sampler.stop()
    with open(path) as f:
        text = f.read()
    assert text.count("session,label") == 1
    assert rec.summarize_csv(path)["idle"]["takes"] == 2


def test_hub_check_gives_a_helpful_error():
    class Broken:
        pass
    try:
        HubIMU(Broken()).check()
    except RuntimeError as exc:
        assert "imu_device" in str(exc)
        return
    raise AssertionError("expected RuntimeError")


def test_probe_runs_end_to_end_on_the_fake():
    dm, hub, sampler = fake_setup(rate_hz=50, rtt_s=0.08)
    dm.scenario = "forehand"
    lines = []
    real_input, real_print = builtins.input, builtins.print
    builtins.input = lambda *a, **k: ""
    builtins.print = lambda *a, **k: lines.append(" ".join(str(x) for x in a))
    try:
        result = rec.run_probe(hub, sampler, hub.read, kick_trials=2, rate_seconds=1.0,
                               noise_seconds=0.5, swing_seconds=0.7, settle=0.3,
                               sweep_delays=(100, 20), do_kick=True)
    finally:
        builtins.input, builtins.print = real_input, real_print
        sampler.stop()
    assert result["notify_ms"] == 20, "the fast setting kept up, so it should be recommended"
    assert 45 <= result["hz"] <= 55
    assert result["rtt_median"] is not None and 0.05 < result["rtt_median"] < 0.3
    assert result["swing_threshold"] > 0
    assert any("--notify-ms 20" in l for l in lines)


def test_probe_refuses_to_suggest_a_threshold_when_no_swing_was_seen():
    dm, hub, sampler = fake_setup(rate_hz=50, rtt_s=0.08)
    dm.scenario = "idle"                       # the "player" never actually swings
    lines = []
    real_input, real_print = builtins.input, builtins.print
    builtins.input = lambda *a, **k: ""
    builtins.print = lambda *a, **k: lines.append(" ".join(str(x) for x in a))
    try:
        result = rec.run_probe(hub, sampler, hub.read, kick_trials=1, rate_seconds=1.0,
                               noise_seconds=0.5, swing_seconds=0.5, settle=0.3,
                               sweep_delays=(100,), do_kick=False)
    finally:
        builtins.input, builtins.print = real_input, real_print
        sampler.stop()
    assert result["swing_threshold"] is None
    assert any("Not suggesting a threshold" in l for l in lines)


def test_requesting_a_faster_notification_rate_changes_what_arrives():
    dm = FakeDoubleMotor(rate_hz=10)
    hub = FakeAdapter(dm)
    assert hub.set_notification_delay(100)
    slow = measure_update_rate(hub.read, seconds=1.5)["hz"]
    assert hub.set_notification_delay(20)
    fast = measure_update_rate(hub.read, seconds=1.5)["hz"]
    assert 8 <= slow <= 12 and 45 <= fast <= 55, (slow, fast)


def test_sweep_recommends_the_fastest_setting_a_capped_link_keeps_up_with():
    dm = FakeDoubleMotor(rate_hz=10, max_rate_hz=40)        # like a link that tops out at 40 Hz
    hub = FakeAdapter(dm)
    results = sweep_notification_rates(hub, hub.read, delays=(100, 50, 30, 15), seconds=1.3, settle=0.2)
    got = {r["delay_ms"]: r["stats"]["hz"] for r in results}
    assert 33 <= got[30] <= 45, got       # 33 Hz requested, delivered
    assert got[15] < 50, got              # 66 Hz requested, capped near 40
    assert recommend_delay(results) == 30


def test_recommend_delay_logic():
    def r(d, hz, ok=True):
        return {"delay_ms": d, "requested_hz": 1000 / d, "accepted": ok, "stats": {"hz": hz}}
    assert recommend_delay([r(100, 9.6), r(20, 20.0)]) == 100, "20 ms asked for 50 Hz but only 20 arrived"
    assert recommend_delay([r(100, 9.6), r(50, 19.5), r(20, 49.0)]) == 20
    assert recommend_delay([r(20, 49.0, ok=False)]) is None, "a rejected request doesn't count"
    assert recommend_delay([{"delay_ms": 20, "requested_hz": 50, "accepted": True, "stats": None}]) is None


def test_notification_request_is_clamped_to_the_documented_range_and_survives_failure():
    class Rec:
        def __init__(self):
            self.calls = []
        def device_notification_request(self, d, **k):
            self.calls.append(d)
    rec_dm = Rec()
    h = HubIMU(rec_dm)
    h.set_notification_delay(5)
    h.set_notification_delay(5000)
    assert rec_dm.calls == [15, 1000]

    class Boom:
        def device_notification_request(self, d, **k):
            raise RuntimeError("nope")
    assert HubIMU(Boom()).set_notification_delay(20) is False


def test_recordings_report_the_fresh_reading_rate():
    dm, hub, sampler = fake_setup(rate_hz=10)
    hub.set_notification_delay(20)
    path = os.path.join(tempfile.mkdtemp(), "swings.csv")
    try:
        rec.record_session(hub, sampler, ["idle"], {"idle": 2}, path, prep_s=0.05, window_s=0.8,
                           fake=dm, log=lambda *a: None, label_pause=0.05, noise_s=0.3)
    finally:
        sampler.stop()
    hz = rec.fresh_rate_from_csv(path)
    assert 40 <= hz <= 60, hz


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)