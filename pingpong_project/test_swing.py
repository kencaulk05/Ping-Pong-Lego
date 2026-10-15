"""
test_swing.py -- tests for swing detection and recognition.

    python test_swing.py        (or: pytest test_swing.py)

Most tests use synthetic signals and need nothing else. A few use your real
recording (swings.csv) and the trained model (swing_model.json) and are skipped
if those files aren't next to this one; the cross-validation test also needs
scikit-learn.
"""

import collections
import os
import random

import numpy as np

import swing_detect as sd
from swing_detect import H, SwingDetector, SwingModel, SwingRecognizer

HERE = os.path.dirname(os.path.abspath(__file__))
HZ = 31.0


class Skip(Exception):
    pass


# ------------------------------------------------------------------ helpers

def make_stream(mags, direction=(0, 1, 0), seed=0, noise=0.5):
    """Readings (ax..gz) at ~31 Hz whose gyro magnitudes are `mags`, rotating about
    `direction`. Tiny jitter on the accelerometer keeps each reading 'fresh'."""
    rng = random.Random(seed)
    d = np.array(direction, float)
    d = d / np.linalg.norm(d)
    out = []
    for i, m in enumerate(mags):
        g = d * m
        out.append((i / HZ, (rng.gauss(0, noise), rng.gauss(0, noise), 1000 + rng.gauss(0, noise), *g)))
    return out


def baseline_with_pulses(n, base, pulses, seed=1):
    """Gyro magnitude: a noisy baseline plus gaussian bumps at (index, height)."""
    rng = random.Random(seed)
    g = np.array([base + rng.uniform(-0.15, 0.15) * base for _ in range(n)])
    idx = np.arange(n)
    for c, h in pulses:
        g += (h - base) * np.exp(-(((idx - c) / 1.5) ** 2))
    return g


def run(det, stream):
    ev = []
    for t, v in stream:
        ev += det.push(t, v)
    return ev


def toy_model(classes=("backhand", "flat", "forehand", "other"), uy_class=None, **extra):
    """A hand-built model: forehand if uy is high, backhand if uy is low, else 'other'."""
    n = len(sd.FEATURE_NAMES)
    coef = np.zeros((4, n))
    uy = sd.FEATURE_NAMES.index("uy")
    coef[2, uy] = +8.0      # forehand
    coef[0, uy] = -8.0      # backhand
    data = dict(version=1, classes=list(classes), feature_names=sd.FEATURE_NAMES,
                mean=[0.0] * n, scale=[1.0] * n, coef=coef.tolist(),
                intercept=[0.0, 0.0, 0.0, 2.0], detector=sd.DETECTOR_PARAMS,
                expected_hz=HZ, power_range=[200.0, 1000.0])
    data.update(extra)
    return SwingModel(data)


# -------------------------------------------------------------------- tests

def test_a_repeated_reading_is_not_a_new_reading():
    det = SwingDetector()
    for _ in range(10):
        det.push(0.0, (1, 2, 3, 4, 5, 6))
    assert det.n_fresh == 1
    det.push(0.1, (1, 2, 3, 4, 5, 7))
    assert det.n_fresh == 2


def test_a_quiet_hand_produces_no_events():
    g = baseline_with_pulses(300, 5, [])
    assert run(SwingDetector(), make_stream(g)) == []


def test_a_pulse_is_found_exactly_h_readings_after_its_peak():
    g = baseline_with_pulses(120, 60, [(60, 800)])
    stream = make_stream(g)
    ev = run(SwingDetector(), stream)
    assert len(ev) == 1
    e = ev[0]
    assert e.index == 60 and abs(e.t_peak - stream[60][0]) < 1e-9
    assert abs(e.t_ready - stream[60 + H][0]) < 1e-9, "complete only H readings after the peak"
    assert abs(e.age - H / HZ) < 1e-6
    assert 700 < e.peak < 900


def test_a_peak_below_the_absolute_floor_is_ignored():
    g = baseline_with_pulses(120, 10, [(60, sd.T_ABS - 25)])
    assert run(SwingDetector(), make_stream(g)) == []


def test_a_peak_that_is_not_clearly_above_the_hands_own_movement_is_ignored():
    # The hand is already moving at ~400; a bump to 500 is under 1.5x that.
    g = baseline_with_pulses(120, 400, [(60, 500)])
    assert run(SwingDetector(), make_stream(g)) == []
    # ...but a real swing on top of the same movement is found.
    g = baseline_with_pulses(120, 400, [(60, 1500)])
    assert len(run(SwingDetector(), make_stream(g))) == 1


def test_two_peaks_close_together_give_one_event_the_larger():
    g = baseline_with_pulses(120, 60, [(60, 800), (63, 950)])
    ev = run(SwingDetector(), make_stream(g))
    assert len(ev) == 1 and ev[0].index == 63


def test_two_separate_swings_give_two_events():
    g = baseline_with_pulses(160, 60, [(50, 800), (100, 900)])
    assert [e.index for e in run(SwingDetector(), make_stream(g))] == [50, 100]


def test_nothing_is_reported_before_there_are_h_real_readings_before_the_peak():
    g = baseline_with_pulses(80, 60, [(H - 3, 800)])
    assert run(SwingDetector(), make_stream(g)) == []


def test_features_know_which_way_the_hub_rotated():
    g = baseline_with_pulses(120, 60, [(60, 800)])
    up = run(SwingDetector(), make_stream(g, direction=(0, 1, 0)))[0].features
    down = run(SwingDetector(), make_stream(g, direction=(0, -1, 0)))[0].features
    uy = sd.FEATURE_NAMES.index("uy")
    assert up[uy] > 0.9 and down[uy] < -0.9


def test_the_numpy_model_picks_the_class_the_weights_say():
    m = toy_model()
    g = baseline_with_pulses(120, 60, [(60, 800)])
    fh = run(SwingDetector(), make_stream(g, direction=(0, 1, 0)))[0].features
    bh = run(SwingDetector(), make_stream(g, direction=(0, -1, 0)))[0].features
    flat = run(SwingDetector(), make_stream(g, direction=(1, 0, 0)))[0].features
    assert m.predict(fh)[0] == "forehand" and m.predict(bh)[0] == "backhand"
    assert m.predict(flat)[0] == "other"
    assert abs(m.proba(fh).sum() - 1.0) < 1e-9


def test_a_model_trained_with_other_detector_settings_is_refused():
    bad = toy_model().meta.copy()
    bad["detector"] = dict(sd.DETECTOR_PARAMS, H=sd.H + 1)
    try:
        SwingModel(bad)
    except ValueError as exc:
        assert "Retrain" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_the_recognizer_turns_events_into_game_swings():
    rec = SwingRecognizer(toy_model())
    g = baseline_with_pulses(120, 60, [(60, 800)])
    found = []
    for t, v in make_stream(g, direction=(0, 1, 0)):
        found += rec.push(t, v)
    assert len(found) == 1
    s = found[0]
    assert s.kind == "forehand" and s.confidence > 0.9
    assert abs(s.age - H / HZ) < 1e-6
    assert abs(s.power - (s.peak - 200) / 800) < 1e-9 and 0 <= s.power <= 1
    swing = s.to_swing()
    assert (swing.kind, swing.power, swing.age) == (s.kind, s.power, s.age)


def test_events_called_other_or_low_confidence_are_rejected_not_reported():
    rec = SwingRecognizer(toy_model())
    g = baseline_with_pulses(120, 60, [(60, 800)])
    found = []
    for t, v in make_stream(g, direction=(1, 0, 0)):          # sideways: the toy model says "other"
        found += rec.push(t, v)
    assert found == [] and len(rec.rejected) == 1 and rec.rejected[0][1] == "other"

    strict = SwingRecognizer(toy_model(), min_confidence=0.999999)
    found = []
    for t, v in make_stream(g, direction=(0, 1, 0)):
        found += strict.push(t, v)
    assert found == [], "a forehand-looking event below the confidence bar must not be reported"


def test_the_rate_check_notices_a_wrong_notification_setting():
    ok, fast = SwingRecognizer(toy_model()), SwingRecognizer(toy_model())
    for i in range(120):
        ok.push(i / 31.0, (i, 0, 0, 0, 0, 0))
        fast.push(i / 100.0, (i, 0, 0, 0, 0, 0))
    assert ok.rate_ok() and abs(ok.fresh_rate() - 31) < 1
    assert not fast.rate_ok()


def test_sklearn_export_matches_the_numpy_model():
    try:
        from train_swing_classifier import export_model, make_pipeline
    except ImportError:
        raise Skip("scikit-learn not installed")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, len(sd.FEATURE_NAMES)))
    y = np.array(["forehand", "backhand", "flat", "other"] * 30)
    X[y == "forehand", 2] += 2.0
    X[y == "backhand", 2] -= 2.0
    pipe = make_pipeline().fit(X, y)
    model = SwingModel(export_model(pipe, dict(expected_hz=31.0, power_range=[100, 900])))
    for x in X[:25]:
        expected = dict(zip(pipe.classes_, pipe.predict_proba([x])[0]))
        got = dict(zip(model.classes, model.proba(x)))
        assert all(abs(expected[c] - got[c]) < 1e-9 for c in expected), (expected, got)


def test_the_tally_scores_against_what_you_said_you_were_doing():
    from swing_live import Tally
    t = Tally()
    for k in ["forehand"] * 8 + ["backhand"] * 2:
        t.add_recognized(k)
    t.add_ignored("other")
    lines = "\n".join(t.summary(expect="forehand", count=12))
    assert "8 recognized correctly (67%)" in lines and "2 recognized as the wrong kind" in lines
    assert "1 ignored" in lines and "about 1 produced nothing" in lines
    quiet = Tally()
    quiet.add_ignored("other")
    quiet.add_ignored("other")
    assert "recognized 0 swings anyway" in "\n".join(quiet.summary(expect="none"))
    noisy = Tally()
    noisy.add_recognized("flat")
    assert "recognized 1 swings anyway" in "\n".join(noisy.summary(expect="none"))
    assert "5 of 5" in "\n".join(sum_lines(5)), "all recognized correctly when none were wrong"


def sum_lines(n):
    from swing_live import Tally
    t = Tally()
    for _ in range(n):
        t.add_recognized("flat")
    return t.summary(expect="flat")


# ------------------------------------- not fooled by a still hand (the bug that broke swing_check)

def test_isolating_a_swing_keeps_the_peak_and_quiets_everything_else():
    G = np.ones((2 * H + 1, 3)) * 100.0
    G[H] = (0, 900.0, 0)
    A = np.zeros((2 * H + 1, 3))
    A[H + 5] = (500, 0, 0)
    G2, A2 = sd.isolate_swing(G, A, surround=0.1)
    assert np.allclose(G2[H], G[H]) and np.allclose(G2[H - 1], G[H - 1]), "the pulse itself is untouched"
    assert np.allclose(G2[0], 10.0) and np.allclose(G2[-1], 10.0), "far from the peak it is turned down to 10%"
    assert abs(A2[H + 5, 0]) < 60, "an accelerometer spike away from the peak shrinks too"


def test_a_model_that_uses_the_movement_features_is_fooled_by_a_still_hand():
    # This documents the failure and proves the check can see it: a model trained WITH the
    # movement-level features calls nearly every swing made from a still hand 'flat'.
    try:
        from train_swing_classifier import build_events, still_hand_check
    except ImportError:
        raise Skip("scikit-learn not installed")
    main, _, _, _ = build_events(_takes())
    X = np.array([e["features"] for e in main])
    y = np.array([e["label"] for e in main])
    try:
        r = still_hand_check(X, y, list(range(X.shape[1])))
    except ImportError:
        raise Skip("scikit-learn not installed")
    assert r["flat_pct"] >= 80, r


def test_the_shipped_model_does_not_use_the_movement_features():
    _need("swing_model.json")
    m = SwingModel.load(os.path.join(HERE, "swing_model.json"))
    assert not set(m.feature_names) & set(sd.CONTEXT_FEATURES), m.feature_names
    # ...so changing them can't change its answer
    f = np.linspace(0.1, 2.0, len(sd.FEATURE_NAMES))
    g = f.copy()
    for n in sd.CONTEXT_FEATURES:
        g[sd.FEATURE_NAMES.index(n)] += 50.0
    assert np.allclose(m.proba(f), m.proba(g))


def test_a_model_may_use_a_subset_of_the_features_and_old_full_models_still_load():
    full = toy_model()
    keep = [n for n in sd.FEATURE_NAMES if n not in sd.CONTEXT_FEATURES]
    idx = [sd.FEATURE_NAMES.index(n) for n in keep]
    data = dict(full.meta)
    data["feature_names"] = keep
    data["mean"] = [0.0] * len(keep)
    data["scale"] = [1.0] * len(keep)
    data["coef"] = [[row[i] for i in idx] for row in np.array(full.meta["coef"])]
    sub = SwingModel(data)
    f = np.random.default_rng(0).normal(size=len(sd.FEATURE_NAMES))
    assert np.allclose(sub.proba(f), full.proba(f)), "dropping features the weights ignored changes nothing"


def test_the_shipped_model_still_classifies_real_swings_made_from_a_quiet_hand():
    # An EMULATION of still-hand swings, made from the original recording. The failure this guards
    # against is the model reading a still hand as "flat" (the first model called ~100% of
    # forehands and backhands flat). It is only a proxy: real still-hand runs now exist and the model
    # trains on them, so eval_runs.py (scored on held-out runs) is the number to believe. The overall
    # floor here is therefore loose; the 'flat' collapse is what must never come back.
    _need("swing_model.json")
    model = SwingModel.load(os.path.join(HERE, "swing_model.json"))
    right = flat_wrong = n = fb = 0
    for (sess, label, take), (times, rows) in _takes().items():
        if label not in sd.SWING_KINDS:
            continue
        evs = run_rows(times, rows)
        if not evs:
            continue
        e = max(evs, key=lambda x: x.peak)
        if e.peak < model.min_peak:                       # too small to be called a swing in use anyway
            continue
        G, A = sd.isolate_swing(e.window[0], e.window[1], surround=0.25)
        pred = model.predict(sd.event_features(G, A, 8.0))[0]
        n += 1
        right += pred == label
        if label in ("forehand", "backhand"):
            fb += 1
            flat_wrong += pred == "flat"
    assert n >= 40 and flat_wrong / fb <= 0.15, (right, n, flat_wrong, fb)
    assert right / n >= 0.70, (right, n)


# ------------------------------------- the minimum swing size, and learning from saved runs

def _stream_with_pulses(pulses, n=500, base=40):
    g = baseline_with_pulses(n, base, pulses)
    return make_stream(g, direction=(0, 1, 0))


def test_events_below_the_minimum_swing_size_are_rejected_whatever_they_look_like():
    big, small = 900, 250
    for peak, expect in ((big, True), (small, False)):
        rec = SwingRecognizer(toy_model(), min_peak=330)
        found = []
        for t, v in _stream_with_pulses([(60, peak)], n=130):
            found += rec.push(t, v)
        assert bool(found) == expect, (peak, found)
        if not expect:
            assert rec.rejected and rec.rejected[0][1] == "too small"


def test_the_minimum_swing_size_comes_from_the_model_unless_overridden():
    m = toy_model(min_swing_peak=500)
    assert m.min_peak == 500 and SwingRecognizer(m).min_peak == 500
    assert SwingRecognizer(m, min_peak=100).min_peak == 100
    assert toy_model().min_peak == 0.0, "models saved before this existed still load, with no floor"


def test_the_shipped_model_has_a_minimum_swing_size_between_movement_and_swings():
    _need("swing_model.json")
    m = SwingModel.load(os.path.join(HERE, "swing_model.json"))
    assert 250 <= m.min_peak <= 400, m.min_peak
    assert m.power_range[0] == m.min_peak, "power 0 should start at the minimum swing size"


def test_a_run_is_labeled_by_its_attempts_and_by_style():
    import tempfile
    import train_swing_classifier as tr
    # Cues at 3 s ('forehand' go), 8 s ('forehand' go); random window 12-13 s. Pulses: a swing
    # 0.6 s after each cue, plus an event at 6 s (between attempts) and one at 12.5 s (random phase).
    # (a smaller back-swing 1.0 s after the first attempt's swing, inside that attempt's window)
    pulses = [(int(3.6 * HZ), 900), (int(4.7 * HZ), 450), (int(8.6 * HZ), 900), (int(6.4 * HZ), 700), (int(12.5 * HZ), 600)]
    stream = _stream_with_pulses(pulses, n=int(16 * HZ))
    cues = [(3.0, "go", "forehand"), (8.0, "go", "forehand"), (12.0, "go", "random"), (13.0, "go", "random")]
    base = _write_run(tempfile.mkdtemp(), [(t, v) for t, v in stream], cues)

    main, secondary, missed, n_att = tr.events_from_run(base, "waiting")
    labels = collections.Counter(e["label"] for e in main)
    assert n_att == 2 and labels["forehand"] == 2 and labels["other"] == 2, labels   # between-attempt + random events
    assert all(e["peak"] > 800 for e in main if e["label"] == "forehand"), "the attempt's label goes to its BIGGEST event"
    assert len(secondary) == 1 and secondary[0]["peak"] < 600, "the smaller back-swing is secondary, not trained on"
    main_c, _, _, _ = tr.events_from_run(base, "continuous")
    labels_c = collections.Counter(e["label"] for e in main_c)
    assert labels_c["forehand"] == 2 and labels_c["other"] == 1, "when you were swinging continuously, an event between attempts is a swing, not 'other'"


def test_run_specs_default_to_continuous_and_reject_unknown_styles():
    import train_swing_classifier as tr
    assert tr.parse_run_spec("guided-x") == ("guided-x", "continuous")
    assert tr.parse_run_spec("guided-x:waiting") == ("guided-x", "waiting")
    try:
        tr.parse_run_spec("guided-x:sometimes")
    except SystemExit:
        return
    raise AssertionError("expected an error for an unknown style")


def test_saved_runs_score_without_false_alarms_with_the_shipped_model():
    import rescore
    base = os.path.join(HERE, "guided-20261012-031646")
    _need("swing_model.json", "guided-20261012-031646-stream.csv", "guided-20261012-031646-cues.csv")
    model = SwingModel.load(os.path.join(HERE, "swing_model.json"))
    sc, _ = rescore.rescore(base, model)
    assert sc["false_alarms"] <= 1, sc["false_alarms"]
    assert sc["right"] >= 24, sc["right"]          # in-sample for this run (it trained on it): a plumbing check


def test_cross_run_scores_on_runs_the_model_never_saw_stay_above_a_floor():
    import subprocess
    import sys
    runs = ["guided-20261012-023715", "guided-20261012-024425", "guided-20261012-031646", "guided-20261012-034425"]
    _need("swings.csv", *[r + "-stream.csv" for r in runs], *[r + "-cues.csv" for r in runs])
    cmd = [sys.executable, os.path.join(HERE, "eval_runs.py"), "swings.csv"]
    for r in runs:
        cmd += ["--run", r + (":waiting" if r.endswith(("031646", "034425")) else "")]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=250, cwd=HERE)
    assert out.returncode == 0, out.stderr[-300:]
    last = [l for l in out.stdout.splitlines() if l.startswith("Average on runs")][0]
    pct = int(last.split(":")[1].split("%")[0])
    assert pct >= 85, last
    assert "false alarms 0" in out.stdout and not any(
        f"false alarms {n}" in out.stdout for n in range(1, 10)), "no unseen run may raise a false alarm"


# ------------------------------------------------------- re-scoring a saved run (rescore.py)

def _write_run(tmpdir, readings, cues):
    import csv as _csv
    base = os.path.join(tmpdir, "run")
    with open(base + "-stream.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz", "yaw", "pitch", "roll"])
        for t, v in readings:
            w.writerow([f"{t:.4f}", *v, 0, 0, 0] if len(v) == 6 else [f"{t:.4f}", *v])
    with open(base + "-cues.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["t", "what", "phase", "seconds_the_call_took"])
        for c in cues:
            w.writerow([*c, 0.0])
    return base


def test_rescoring_a_saved_run_numbers_attempts_and_scores_them():
    import tempfile
    import rescore
    # Two 'forehand' attempts (cues at 3 s and 8 s), each with a forehand-shaped pulse 0.6 s later.
    g = baseline_with_pulses(400, 60, [(int((3.6) * HZ), 900), (int((8.6) * HZ), 900)])
    stream = make_stream(g, direction=(0, 1, 0))
    cues = [(3.0, "go", "forehand"), (8.0, "go", "forehand"), (12.0, "go", "random"), (13.0, "go", "random")]
    base = _write_run(tempfile.mkdtemp(), [(t, v) for t, v in stream], cues)
    readings, attempts, window = rescore.load_run(base)
    assert [(p, n) for p, n, _ in attempts] == [("forehand", 1), ("forehand", 2)]
    assert window == (12.0, 13.0)
    sc, results = rescore.rescore(base, toy_model())
    assert sc["table"]["forehand"]["forehand"] == 2 and sc["right"] == 2 and sc["total"] == 2


# ------------------------------------------------------------------------------------------

def test_every_command_line_tool_starts():
    # Regression: swing_check crashed on startup (a duplicated option) and no test noticed,
    # because the tests call run() directly and never go through argument parsing.
    import subprocess
    import sys
    for script in ("swing_check.py", "swing_live.py", "imu_recorder.py", "train_swing_classifier.py",
                   "pingpong_sim.py", "eval_opponent.py", "rescore.py", "eval_runs.py"):
        r = subprocess.run([sys.executable, os.path.join(HERE, script), "--help"], capture_output=True,
                           text=True, timeout=60, cwd=HERE)
        assert r.returncode == 0, f"{script} --help failed:\n{r.stderr[-400:]}"
        assert "usage" in r.stdout.lower(), script


def test_swing_check_parses_its_flags():
    import sys
    import swing_check as sc
    old = sys.argv
    try:
        sys.argv = ["swing_check.py", "--no-cues", "--no-beep", "--attempts", "4", "--break", "3"]
        a = sc.parse_args()
    finally:
        sys.argv = old
    assert a.no_cues and a.no_beep and a.attempts == 4 and a.break_s == 3.0


# ------------------------------------------------ the guided check (swing_check.py)

def _ev(t, peak=500, recognized=True, kind="forehand", conf=0.9):
    return dict(t_peak=t, peak=peak, recognized=recognized, kind=kind, conf=conf)


def test_the_schedule_has_every_step_in_order_with_the_breaks():
    import swing_check as sc
    s = sc.build_schedule(attempts=4, spacing=3.5, ready_s=1.5, break_s=5.0, random_s=30.0, start_delay=2.0)
    times = [t for t, _, _ in s["steps"]]
    assert times == sorted(times)
    assert [a["phase"] for a in s["attempts"]] == ["forehand"] * 4 + ["backhand"] * 4 + ["flat"] * 4
    assert [a["number"] for a in s["attempts"][:4]] == [1, 2, 3, 4]
    announces = [(t, p) for t, w, p in s["steps"] if w == "announce"]
    assert [p for _, p in announces] == ["forehand", "backhand", "flat", "random"]
    firsts = {p: min(a["go_t"] for a in s["attempts"] if a["phase"] == p) for p in sc.SWING_KINDS}
    for (t, p) in announces[:3]:
        assert abs(firsts[p] - (t + 5.0 + 1.5)) < 1e-9, "a 5 s break, then the yellow light, then GO"
    assert abs(s["attempts"][1]["go_t"] - s["attempts"][0]["go_t"] - 3.5) < 1e-9
    lo, hi = s["random"]
    assert abs((hi - lo) - 30.0) < 1e-9 and announces[3][0] + 5.0 == lo
    # the next step is announced only after the previous step's last attempt has been scored,
    # so results always print in order
    for i, last in ((1, 3), (2, 7), (3, 11)):
        assert announces[i][0] >= s["attempts"][last]["go_t"] + sc.POST + sc.SETTLE - 1e-9


def test_an_attempt_is_judged_by_its_biggest_event_near_the_cue():
    import swing_check as sc
    go = 10.0
    assert sc.resolve_attempt([], go)["outcome"] == "nothing"
    assert sc.resolve_attempt([_ev(5.0), _ev(20.0)], go)["outcome"] == "nothing", "events far from the cue don't count"
    r = sc.resolve_attempt([_ev(10.9, peak=900, kind="backhand")], go)
    assert (r["outcome"], r["kind"], r["extras"]) == ("recognized", "backhand", 0)
    # a big back-swing that was ignored, plus a smaller recognized event: the biggest decides
    r = sc.resolve_attempt([_ev(10.6, peak=800, recognized=False, kind="other", conf=0.4),
                            _ev(11.2, peak=300, kind="flat")], go)
    assert r["outcome"] == "ignored" and r["extras"] == 1
    # the window edges: a bit before the cue counts, long before does not
    assert sc.resolve_attempt([_ev(go - sc.PRE + 0.01)], go)["outcome"] == "recognized"
    assert sc.resolve_attempt([_ev(go - sc.PRE - 0.01)], go)["outcome"] == "nothing"
    assert sc.resolve_attempt([_ev(go + sc.POST + 0.01)], go)["outcome"] == "nothing"


def test_the_score_counts_attempts_and_false_alarms_separately():
    import swing_check as sc
    results = ([("forehand", dict(outcome="recognized", kind="forehand")) for _ in range(8)]
               + [("forehand", dict(outcome="recognized", kind="backhand")),
                  ("forehand", dict(outcome="nothing", kind=None))]
               + [("backhand", dict(outcome="ignored", kind="other"))]
               + [("flat", dict(outcome="recognized", kind="flat")) for _ in range(3)])
    events = [_ev(50.0, kind="flat"), _ev(55.0, kind="flat"), _ev(60.0, recognized=False, kind="other"),
              _ev(120.0, kind="forehand")]                  # the last is outside the random window
    r = sc.score(results, events, random_window=(40.0, 100.0))
    assert r["table"]["forehand"]["forehand"] == 8 and r["table"]["forehand"]["backhand"] == 1
    assert r["table"]["forehand"]["nothing"] == 1 and r["table"]["backhand"]["ignored"] == 1
    assert (r["right"], r["total"]) == (11, 14)
    assert r["false_alarms"] == 2 and r["false_by_kind"]["flat"] == 2 and r["ignored_in_random"] == 1
    text = "\n".join(sc.format_report(r, "guided-x.csv"))
    assert "11 of 14 (79%)" in text and "2 false alarms" in text and "guided-x.csv" in text


def test_stopping_early_scores_only_the_part_of_random_movement_you_did():
    import swing_check as sc
    events = [_ev(10.0, kind="flat"), _ev(25.0, kind="flat"), _ev(45.0, kind="flat")]
    full = sc.score([], events, (0.0, 60.0))
    early = sc.score([], events, (0.0, 60.0), upto=30.0)
    assert full["false_alarms"] == 3 and full["random_seconds"] == 60.0 and not full["stopped"]
    assert early["false_alarms"] == 2 and early["random_seconds"] == 30.0 and early["stopped"]
    assert "you stopped early" in "\n".join(sc.format_report(early))


def test_the_saved_session_uses_the_recorders_format_and_appends():
    import csv as _csv
    import tempfile
    import swing_check as sc
    readings = [(t / 10.0, tuple(float(t + k) for k in range(9))) for t in range(0, 400)]   # 0..39.9 s
    attempts = [dict(phase="forehand", number=1, go_t=5.0), dict(phase="backhand", number=1, go_t=10.0)]
    path = os.path.join(tempfile.mkdtemp(), "g.csv")
    n = sc.save_session(path, readings, attempts, random_window=(20.0, 22.0), session="S1")
    sc.save_session(path, readings, attempts[:1], random_window=(20.0, 22.0), session="S2")
    with open(path, newline="") as f:
        rows = list(_csv.reader(f))
    assert rows[0] == sc.CSV_HEADER and sum(1 for r in rows if r[0] == "session") == 1
    body = [r for r in rows[1:] if r[0] == "S1"]
    assert len(body) == n
    fh = [r for r in body if r[1] == "forehand"]
    assert abs(float(fh[0][3])) < 1e-9, "t is relative to the start of the saved window"
    assert float(fh[-1][3]) <= 3.0 + 1e-9 and len(fh) == 31          # go-0.5 .. go+2.5 at 10 readings/s
    assert {r[1] for r in body} == {"forehand", "backhand", "random"}
    # the same file loads in the trainer, with the random phase counted as 'not a swing'
    from train_swing_classifier import load_takes
    assert {k[1] for k in load_takes(path)} == {"forehand", "backhand", "random"}


def test_reading_the_hub_does_not_stop_while_a_cue_blocks():
    # Regression: the first swing_check read the hub in the same loop that sent the
    # beep, so every beep paused the reading and swallowed the swing that followed.
    import threading
    import time as _time
    import swing_check as sc

    class Hub:
        def __init__(self):
            self.n = 0
            self.lock = threading.Lock()

        def read(self):
            with self.lock:
                self.n = int(_time.perf_counter() * 100)      # a fresh value every 10 ms
            return (self.n, 0, 0, 0, 0, 0, 0, 0, 0)

    col = sc.Collector(Hub(), SwingRecognizer(toy_model()), _time.perf_counter()).start()
    try:
        _time.sleep(0.2)
        before = len(col.readings)
        _time.sleep(0.8)                 # stands in for beep(): the MAIN thread is stuck here
        during = len(col.readings) - before
    finally:
        col.stop()
    assert during >= 50, f"only {during} readings arrived while the main thread was blocked for 0.8 s"


def test_the_whole_guided_check_survives_cues_that_block():
    import argparse
    import contextlib
    import io
    import tempfile
    import time as _time
    import swing_check as sc
    (path,) = _need("swings.csv", "swing_model.json")[:1]
    model = SwingModel.load(os.path.join(HERE, "swing_model.json"))

    class BlockingHub(sc.ReplayHub):                    # cue calls that wait, like a real beep / light command
        def cue_ready(self):
            _time.sleep(0.15)

        def cue_go(self, beep=True):
            _time.sleep(0.80)

    out = os.path.join(tempfile.mkdtemp(), "guided.csv")
    args = argparse.Namespace(attempts=1, spacing=3.5, ready=1.5, break_s=0.3, random_seconds=1.0,
                              min_confidence=0.6, no_save=False, save=out, no_beep=False, no_cues=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = sc.run(args, BlockingHub(path, seed=0), model)
    assert res["right"] == 3 and res["total"] == 3, res["table"]
    assert "Which way your swings rotated" in buf.getvalue(), "the direction report must still be printed"

    # The diagnostic files: the full stream, and every cue with how long its call took.
    base = out[:-4]
    import csv as _csv
    with open(base + "-stream.csv", newline="") as f:
        stream = list(_csv.DictReader(f))
    assert len(stream) > 50 and list(stream[0].keys())[0] == "t"
    with open(base + "-cues.csv", newline="") as f:
        cues = list(_csv.DictReader(f))
    go = [float(c["seconds_the_call_took"]) for c in cues if c["what"] == "go"]
    ready = [float(c["seconds_the_call_took"]) for c in cues if c["what"] == "ready"]
    assert go and min(go) >= 0.7, "the log should show the beep blocking for ~0.8 s"
    assert ready and min(ready) >= 0.1


def test_no_cues_mode_never_touches_the_hub_cues():
    import argparse
    import contextlib
    import io
    import swing_check as sc
    (path,) = _need("swings.csv", "swing_model.json")[:1]
    model = SwingModel.load(os.path.join(HERE, "swing_model.json"))
    calls = []

    class Hub(sc.ReplayHub):
        def cue_ready(self):
            calls.append("ready")

        def cue_go(self, beep=True):
            calls.append("go")

    args = argparse.Namespace(attempts=1, spacing=3.5, ready=1.5, break_s=0.3, random_seconds=1.0,
                              min_confidence=0.6, no_save=True, save=None, no_beep=False, no_cues=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = sc.run(args, Hub(path, seed=0), model)
    assert calls == [], f"--no-cues must not send hub cues, but sent {calls}"
    assert ">>> SWING NOW <<<" in buf.getvalue() and "get ready" in buf.getvalue()
    assert res["right"] == 3 and res["total"] == 3


def test_training_accepts_several_recordings():
    from train_swing_classifier import load_takes
    (path,) = _need("swings.csv")
    one = load_takes(path)
    two = load_takes([path, path])
    assert len(two) == len(one), "same session ids, so the same takes, not duplicates under new keys"


# --------------------------------------------- tests on the real recording

def _need(*names):
    for n in names:
        if not os.path.exists(os.path.join(HERE, n)):
            raise Skip(f"{n} not found")
    return [os.path.join(HERE, n) for n in names]


def _takes():
    from train_swing_classifier import load_takes
    (path,) = _need("swings.csv")
    return load_takes(path)


def test_real_recording_swings_in_the_window_are_found_and_stillness_is_quiet():
    found_in_window = total = 0
    for (session, label, take), (times, rows) in _takes().items():
        events = run_rows(times, rows)
        if label == "idle":
            assert events == [], f"idle take {take} produced events"
        if label in sd.SWING_KINDS:
            g = [np.linalg.norm(r[3:6]) for r in rows]
            t_peak = times[int(np.argmax(g))]
            if 0.3 < t_peak < times[-1] - 0.3:          # the main swing is inside the window
                total += 1
                found_in_window += bool(events)
    assert total >= 40 and found_in_window / total >= 0.90, (found_in_window, total)


def run_rows(times, rows):
    det = SwingDetector()
    ev = []
    for t, v in zip(times, rows):
        ev += det.push(t, v)
    return ev


def test_the_shipped_model_agrees_with_the_detector_on_the_recording():
    # A plumbing check, not an accuracy claim: the model was trained on features from THIS detector, so
    # it should agree with the original recording far more than chance. A model/detector mismatch (wrong
    # window, shifted features) collapses this to near chance, nowhere near the 80% bar. It only counts
    # events at or above the model's minimum swing size, as in real use. The bar is below the ~91%
    # an earlier model reached because training on more varied runs fits this oldest recording
    # a little less tightly (88%), which is the price of the better scores on runs it never saw.
    _need("swing_model.json")
    from train_swing_classifier import build_events
    model = SwingModel.load(os.path.join(HERE, "swing_model.json"))
    main, _, _, _ = build_events(_takes())
    main = [e for e in main if e["peak"] >= model.min_peak]
    hits = sum(model.predict(e["features"])[0] == e["label"] for e in main)
    assert len(main) >= 40 and hits / len(main) >= 0.80, (hits, len(main))


def test_cross_validated_accuracy_stays_high():
    try:
        from train_swing_classifier import build_events, cross_validate
    except ImportError:
        raise Skip("scikit-learn not installed")
    main, secondary, _, _ = build_events(_takes())
    X = np.array([e["features"] for e in main])
    y = np.array([e["label"] for e in main])
    try:
        res = cross_validate(X, y, [e["group"] for e in main], secondary, reps=2)
    except ImportError:
        raise Skip("scikit-learn not installed")
    assert res["acc"] >= 0.85, res["acc"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as why:
            skipped += 1
            print(f"SKIP  {name}: {why}")
        except Exception as exc:                  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    raise SystemExit(1 if failed else 0)
