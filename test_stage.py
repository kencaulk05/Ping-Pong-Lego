"""
test_stage.py -- tests for the look of the game (crowd, characters, rackets, hint text).

    python test_stage.py        (or: pytest test_stage.py)

The stage is drawing only, so the most important test is that it cannot change the game. The rest check that
the pieces react to the right events and that the screen stays readable and affordable.
"""

import random
import subprocess
import sys
import time

import cv2
import numpy as np

import config as C
import pingpong_live as live
import pingpong_sim as sim
import stage as st
from bots import BotPlayer
from pingpong_core import Game, Inputs, Phase
from pingpong_sim import View

DT = 1 / 30


def new_stage(right_handed=True):
    return st.Stage(sim.W, sim.H, sim.project, right_handed)


def game_at(streak=0, level=2, seed=3):
    g = Game(level=level, rng=random.Random(seed))
    g.update(DT, Inputs(wrist=(0.5, 0.6), start_card=True))
    while g.phase != Phase.INCOMING:
        g.update(DT, Inputs(wrist=(0.5, 0.6)))
    g.streak = streak
    return g


def hit(kind="forehand", streak=1, power=0.6):
    return {"type": "hit", "kind": kind, "power": power, "accuracy": 0.9, "target": 0.4, "T": 0.8, "streak": streak}


# ------------------------------------------------------------------------ the crowd

def test_the_crowd_warms_up_with_the_streak_and_settles_when_it_ends():
    s, img = new_stage(), np.zeros((sim.H, sim.W, 3), np.uint8)
    calm = game_at(streak=0)
    s.draw_stands(img, calm, 1.0)
    assert s.energy == 0.0
    streaky = game_at(streak=8)
    s.draw_stands(img, streaky, 1.1)
    assert abs(s.energy - 0.40) < 0.02, "a streak of 8 keeps the crowd at 0.05 per hit"
    streaky.streak = 20
    s.draw_stands(img, streaky, 1.2)
    assert s.energy <= 0.55 + 1e-9, "the baseline is capped; only big moments reach full volume"


def test_a_winner_sends_the_crowd_wild_and_it_calms_down_afterwards():
    s, img, g = new_stage(), np.zeros((sim.H, sim.W, 3), np.uint8), game_at()
    s.draw_stands(img, g, 1.0)
    s.handle([{"type": "winner", "streak": 3}], 1.0)
    assert s.energy == 1.0 and s.current_word()[0] == "WINNER!" and s.wave_t0 == 1.0 and len(s.confetti) > 10
    for i in range(1, 90):                       # three seconds of frames
        s.draw_stands(img, g, 1.0 + i * DT)
    assert s.energy < 0.1, s.energy


def test_a_miss_makes_the_crowd_groan_and_quiets_it():
    s, img, g = new_stage(), np.zeros((sim.H, sim.W, 3), np.uint8), game_at(streak=8)
    s.draw_stands(img, g, 1.0)
    before = s.energy
    s.handle([{"type": "miss", "reason": "no swing", "lost": 8}], 1.0)
    assert s.energy < before * 0.5 and s.groan_until > 1.5 and s.current_word()[0] == "OH NO..."


def test_every_fifth_hit_in_a_row_starts_a_wave_and_others_do_not():
    for streak, wave in ((4, False), (5, True), (6, False), (10, True)):
        s = new_stage()
        s.handle([hit(streak=streak)], 2.0)
        assert (s.wave_t0 is not None) == wave, streak
        if wave:
            assert s.current_word()[0] == f"{streak} IN A ROW!"


def test_a_new_record_throws_confetti_that_falls_away_and_is_cleaned_up():
    s, img = new_stage(), np.zeros((sim.H, sim.W, 3), np.uint8)
    s.handle([{"type": "record", "best": 9}], 1.0)
    assert len(s.confetti) >= 60 and s.current_word()[0] == "NEW RECORD!"
    for i in range(1, 160):                      # ~5 s
        s.draw_confetti(img, 1.0 + i * DT)
    assert len(s.confetti) == 0, "confetti must not pile up forever"


def test_a_burst_of_events_cannot_grow_the_confetti_without_limit():
    s, img = new_stage(), np.zeros((sim.H, sim.W, 3), np.uint8)
    for i in range(40):
        s.handle([{"type": "record", "best": i}], 1.0 + i * 0.01)
        s.draw_confetti(img, 1.0 + i * 0.01)
    assert len(s.confetti) <= 160


def test_the_crowd_is_drawn_in_perspective_and_stays_off_the_hud_bar():
    s, img = new_stage(), np.zeros((sim.H, sim.W, 3), np.uint8)
    s.draw_stands(img, game_at(), 1.0)
    people = s._crowd
    assert 100 <= len(people) <= 220
    sizes_by_y = sorted((p[1], p[2]) for p in people)
    assert sizes_by_y[-1][1] > sizes_by_y[0][1] * 1.4, "people nearer the front are bigger"
    assert min(p[1] - p[2] * 1.2 for p in people) >= 44, "no head may poke into the HUD bar (top 44 px)"
    assert all(0 < p[0] < sim.W for p in people)


# ---------------------------------------------------------------------------- rackets

def test_your_racket_swings_across_the_body_one_way_for_a_forehand_and_the_other_for_a_backhand():
    s = new_stage()
    start = lambda kind: (10.0, kind, 1.0)
    fh0 = s._swing_pose(10.0, start("forehand"), +1.0)[0]
    fh1 = s._swing_pose(10.0 + st.SWING_SECONDS * 0.99, start("forehand"), +1.0)[0]
    bh0 = s._swing_pose(10.0, start("backhand"), +1.0)[0]
    bh1 = s._swing_pose(10.0 + st.SWING_SECONDS * 0.99, start("backhand"), +1.0)[0]
    assert fh0 > 40 and fh1 < -60, "a right-hander's forehand sweeps from the outside to across the body"
    assert bh0 < -40 and bh1 > 60, "and the backhand sweeps back the other way"
    # left-handed is the mirror image
    assert s._swing_pose(10.0, start("forehand"), -1.0)[0] < -40


def test_a_flat_shot_pushes_the_racket_forward_and_makes_it_loom_larger():
    s = new_stage()
    ang, scale, thrust, p = s._swing_pose(10.0 + st.SWING_SECONDS * 0.5, (10.0, "flat", 1.0), 1.0)
    assert thrust > 8 and scale > 1.2 and p is not None


def test_the_racket_returns_to_rest_when_the_swing_is_over():
    s = new_stage()
    ang, scale, thrust, p = s._swing_pose(10.0 + st.SWING_SECONDS + 0.05, (10.0, "forehand", 1.0), 1.0)
    assert p is None and scale == 1.0 and thrust == 0.0 and ang == -14.0


def test_the_opponent_swings_when_it_returns_the_ball():
    g = game_at()
    img0, img1 = (np.full((sim.H, sim.W, 3), 40, np.uint8) for _ in range(2))
    s = new_stage()
    s.draw_opponent(img0, g, 5.0)                                     # at rest
    s.handle([{"type": "opp_return", "streak": 1}], 5.0)
    s.draw_opponent(img1, g, 5.0 + 0.16)                              # half way through its swing
    assert np.abs(img0.astype(int) - img1.astype(int)).sum() > 2000, "the racket must be somewhere else mid-swing"


def test_hits_and_misses_trigger_your_swing_and_a_hit_pops():
    s = new_stage()
    s.handle([hit("backhand")], 3.0)
    assert s.swing[1] == "backhand" and s.pops == [3.0]
    s.handle([{"type": "miss", "reason": "out of position", "lost": 2, "dist": 1.3}], 4.0)
    assert s.swing[0] == 4.0, "you swung from the wrong place: the racket still swings"
    s2 = new_stage()
    s2.handle([{"type": "miss", "reason": "no swing", "lost": 2}], 4.0)
    assert s2.swing is None, "you didn't swing, so the racket doesn't"


# ----------------------------------------------------------- it must not change the game

def test_drawing_the_stage_cannot_change_the_game():
    """The same seeded game, played by the same bot, with and without a view rendering every frame: the events
    must be identical. The stage reads the game; nothing it does is allowed to feed back."""
    def play(with_view, seconds=60):
        rng = random.Random(9)
        g = Game(level=2, rng=rng)
        bot = BotPlayer(skill=0.85, rng=random.Random(4))
        view = View(stage=True) if with_view else None
        out, now = [], 0.0
        for _ in range(int(seconds / DT)):
            now += DT
            inp = bot.act(g, DT)
            ev = g.update(DT, inp)
            out.extend((e["type"], e.get("streak"), e.get("reason")) for e in ev)
            if view:
                view.handle(ev, now)
                view.render(g, inp.wrist, now)
        return out, g.best
    a, best_a = play(False)
    b, best_b = play(True)
    assert len(a) > 40 and best_a >= 3, "the test game must actually be played (a 60 s game gives ~65 events)"
    assert a == b and best_a == best_b


def test_the_view_passes_game_events_to_the_stage_which_draws_the_results():
    v, g = View(stage=True), game_at(streak=5)
    v.handle([hit("backhand", streak=5), {"type": "record", "best": 5}], 2.0)
    assert v.stage.swing is not None and v.stage.swing[1] == "backhand" and len(v.stage.confetti) > 50
    assert v.stage.wave_t0 == 2.0
    img = v.render(g, (g.ring_x, C.RING_Y), 2.1)
    assert v.stage.current_word()[0] in ("NEW RECORD!", "5 IN A ROW!")
    top = img[46:150]
    confetti_like = sum(int((np.abs(top.astype(int) - np.array(c)).sum(axis=2) < 25).sum()) for c in st.CONFETTI[:4])
    assert confetti_like > 150, "confetti should be visible over the crowd and court"
    plain = View(stage=False)
    plain.handle([hit(), {"type": "record", "best": 5}], 2.0)         # the plain view must simply ignore the stage
    assert plain.render(g, (g.ring_x, C.RING_Y), 2.1).shape == img.shape


# ------------------------------------------------------------------------- the net

def _ball_centroid(img, tol=25):
    """Where the orange ball is, found from its pixels (None if absent)."""
    mask = (np.abs(img.astype(int) - np.array(sim.BALL)).sum(axis=2) < tol)
    ys, xs = np.nonzero(mask)
    return None if len(xs) < 20 else (xs.mean(), ys.mean())


def _net_geometry():
    (xl, ym), (xr, _) = sim.project(0, 0.5), sim.project(1, 0.5)
    h = int(st.NET_HEIGHT * sim.lerp(0.55, 1.0, 0.5))
    return xl, xr, ym, h, ym - h


def test_the_net_is_drawn_across_the_middle_in_the_full_look_and_not_in_plain():
    g = game_at()
    full = View(stage=True).render(g, (g.ring_x, C.RING_Y), 1.0)
    plain = View(stage=False).render(g, (g.ring_x, C.RING_Y), 1.0)
    xl, xr, ym, h, top = _net_geometry()
    white = lambda im, y: int((im[y, xl:xr].astype(int).min(axis=1) > 225).sum())
    assert white(full, top) > 0.9 * (xr - xl), "the white tape runs the full width of the court"
    assert white(plain, top) < 20, "--plain keeps the original look: no net"
    mesh = full[top + 5:ym - 2, xl + 20:xr - 20].astype(int)
    court = plain[top + 5:ym - 2, xl + 20:xr - 20].astype(int)
    unchanged = (np.abs(mesh - court).sum(axis=2) < 25).mean()      # pixels where the court shows straight through
    assert 0.25 < unchanged < 0.90, f"the net must be a visible mesh with the court showing through ({unchanged:.0%} see-through)"
    for px in (xl - 2, xr + 2):
        post = full[top - 3:ym, px].astype(int)
        assert (np.abs(post - plain[top - 3:ym, px].astype(int)).sum(axis=1) > 60).mean() > 0.6, "a post at each end"


def test_the_ball_clears_the_net_with_room_to_spare():
    xl, xr, ym, h, top = _net_geometry()
    rad = int(sim.lerp(7, 24, 0.5))
    assert st.BALL_ARC - rad > h + 2, "at mid-flight the ball's lowest point must be above the net's top"
    g = game_at()
    g.phase_t = g.ball.T * 0.5                                  # the ball is over the middle of the court
    full = View(stage=True).render(g, None, 1.0)
    cx, cy = _ball_centroid(full)
    assert cy + rad < top + 3, f"the ball (centre y {cy:.0f}) is not clearing the net top at y {top}"
    plain = View(stage=False).render(g, None, 1.0)
    py = _ball_centroid(plain)[1]
    assert abs(py - ym) < 4 and cy < py - 40, "in plain mode the ball stays on the court surface; in the full look it rises"


def test_the_ball_arc_is_zero_at_both_ends_so_it_still_lands_exactly_on_the_ring():
    # The arc must not move where the ball arrives: that is where you judge the swing.
    g = game_at()
    g.phase_t = g.ball.T * 0.999
    img = View(stage=True).render(g, None, 1.0)
    cx, cy = _ball_centroid(img)
    ring_x, ring_y = g.ring_x * sim.W, C.RING_Y * sim.H
    assert abs(cx - ring_x) < 6 and abs(cy - ring_y) < 6, f"ball at ({cx:.0f}, {cy:.0f}), ring at ({ring_x:.0f}, {ring_y:.0f})"
    plain_img = View(stage=False).render(g, None, 1.0)
    pcx, pcy = _ball_centroid(plain_img)
    assert abs(cx - pcx) < 3 and abs(cy - pcy) < 3, "the full look and the plain look put the ball in the same place on arrival"


def _centroid(img, color, tol=40, region=None):
    sub = img if region is None else img[region[1]:region[3], region[0]:region[2]]
    mask = np.abs(sub.astype(int) - np.array(color)).sum(axis=2) < tol
    ys, xs = np.nonzero(mask)
    if len(xs) < 15:
        return None
    ox, oy = (0, 0) if region is None else (region[0], region[1])
    return xs.mean() + ox, ys.mean() + oy


def test_the_character_stands_on_your_body_and_the_racket_is_in_your_real_hand():
    g = game_at()
    body, hand = (0.20, C.RING_Y), (0.34, 0.50)                          # body at the left, a hand held out to the right
    img = np.full((sim.H, sim.W, 3), 40, np.uint8)
    new_stage().draw_player(img, g, body, 1.0, True, hand=hand)
    shirt = _centroid(img, (200, 150, 90))
    racket = _centroid(img, st.RED_RUBBER)
    assert shirt is not None and racket is not None
    assert abs(shirt[0] - 0.20 * sim.W) < 45, f"the character stands on the body (shirt centre x={shirt[0]:.0f})"
    assert racket[0] > shirt[0] + 60, "and the racket is out where the hand is, not on the body"
    # the classic look is unchanged: with no separate hand the character stands behind the wrist
    a, b = (np.full((sim.H, sim.W, 3), 40, np.uint8) for _ in range(2))
    new_stage().draw_player(a, g, (0.5, 0.6), 1.0, True)
    new_stage().draw_player(b, g, (0.5, 0.6), 1.0, True, hand=st.SAME)
    assert (a == b).all()


def test_the_drawn_arm_never_stretches_past_a_real_arms_length():
    g = game_at()
    img = np.full((sim.H, sim.W, 3), 40, np.uint8)
    new_stage(True).draw_player(img, g, (0.15, C.RING_Y), 1.0, True, hand=(0.95, 0.30))   # a hand absurdly far from the body
    bx, by = 0.15 * sim.W, min(510, max(330, C.RING_Y * sim.H + 84))
    shoulder = (bx + 34, by + 2)
    ring = _centroid(img, sim.GREEN, tol=60) or _centroid(img, sim.TEAL, tol=60)
    assert ring is not None
    assert np.hypot(ring[0] - shoulder[0], ring[1] - shoulder[1]) <= st.ARM_MAX + 16


def test_when_the_hand_cannot_be_seen_the_racket_rests_by_the_body():
    g = game_at()
    img = np.full((sim.H, sim.W, 3), 40, np.uint8)
    new_stage().draw_player(img, g, (0.30, C.RING_Y), 1.0, True, hand=None)
    racket = _centroid(img, st.RED_RUBBER)
    assert racket is not None and abs(racket[0] - 0.30 * sim.W) < 110


# --------------------------------------------------------------------- the screen

def test_plain_mode_is_the_original_look_and_the_flag_exists_in_both_programs():
    g = game_at()
    plain, fancy = View(stage=False), View(stage=True)
    assert plain.stage is None
    a, b = plain.render(g, (g.ring_x, C.RING_Y), 1.0), fancy.render(g, (g.ring_x, C.RING_Y), 1.0)
    assert a.shape == b.shape == (sim.H, sim.W, 3) and np.abs(a.astype(int) - b.astype(int)).mean() > 5
    assert plain.render(g, None, 2.0).shape == a.shape, "no wrist: still draws"
    old_argv = sys.argv
    try:
        sys.argv = ["pingpong_sim.py", "--plain"]
        assert sim.parse_args().plain
    finally:
        sys.argv = old_argv
    assert live.parse_args(["--plain"]).plain and not live.parse_args([]).plain


def test_the_hint_text_stays_clear_of_the_opponent_and_off_the_live_cameras_preview():
    g = game_at()
    v = View(stage=True)
    img = v.render(g, (g.ring_x, C.RING_Y), 1.0)
    # the opponent's red shirt must still be visible: count its pixels near the far wall
    cx, y0 = sim.project(g.opp_x, 0)
    patch = img[y0 - 40:y0 + 20, max(0, cx - 40):cx + 40].astype(int)
    shirt = (np.abs(patch - np.array(sim.RED)).sum(axis=2) < 40).sum()
    assert shirt > 150, f"the opponent is hidden behind something ({shirt} shirt pixels)"
    # in the live game no banner may run under the camera preview, however long its text
    lv = live.LiveView(None)
    for text, scale, thick in (("Get your wrist in the ring, swing as the ball lands", 0.7, 1), (lv.idle_text, 0.8, 2),
                               (live.LiveView(None, track="body").incoming_text, 0.7, 1),
                               (lv.pose_lost_text, 0.8, 2), ("NEW RECORD!", 0.95, 2), ("10 IN A ROW!", 0.95, 2),
                               ("Opponent is chasing it...", 0.7, 1), ("Best streak this game: 12", 0.6, 1)):
        (tw, th), _ = cv2.getTextSize(text, sim.FONT, scale, thick)
        cx = lv._banner_x(tw)
        assert cx + tw // 2 + 14 <= live.PIP_X, f"'{text}' runs under the camera preview"
        assert cx - tw // 2 - 14 >= 0, f"'{text}' runs off the left edge"
    assert View(stage=True)._banner_x(300) == sim.W // 2, "the simulator keeps hint text centered"


def test_the_stage_costs_only_a_few_milliseconds_a_frame():
    def avg(plain):
        g = game_at(streak=6)
        v = View(stage=not plain)
        w, now = (g.ring_x, C.RING_Y), 5.0
        v.render(g, w, now)
        t0 = time.perf_counter()
        for i in range(60):
            now += DT
            if i % 20 == 0:
                v.handle([{"type": "record", "best": 5}, hit(streak=5)], now)
            v.render(g, w, now)
        return (time.perf_counter() - t0) / 60 * 1000
    extra = avg(False) - avg(True)
    assert extra < 15, f"the stage adds {extra:.1f} ms a frame (a 30 fps frame is 33 ms)"


def test_every_game_state_draws_without_error_with_and_without_a_wrist():
    v = View(stage=True)
    for level in (1, 2, 3):
        g = game_at(level=level)
        now = 1.0
        for wrist in ((g.ring_x, C.RING_Y), None, (0.0, 0.0), (1.0, 1.0)):
            for events in ([], [hit("flat")], [{"type": "winner", "streak": 2}], [{"type": "record", "best": 3}],
                           [{"type": "miss", "reason": "too early", "lost": 1}], [{"type": "aborted", "reason": "x"}]):
                v.handle(events, now)
                now += DT
                assert v.render(g, wrist, now).shape == (sim.H, sim.W, 3)
    idle = Game(level=1, rng=random.Random(1))
    assert v.render(idle, (0.5, 0.8), 9.0).shape == (sim.H, sim.W, 3)


def test_lefthanded_play_puts_the_characters_racket_arm_on_the_other_side():
    g = game_at()
    a, b = (np.full((sim.H, sim.W, 3), 40, np.uint8) for _ in range(2))
    w = (0.5, 0.65)
    new_stage(True).draw_player(a, g, w, 1.0, True)
    new_stage(False).draw_player(b, g, w, 1.0, True)
    left_half = lambda im: int((im[:, :sim.W // 2].astype(int) != 40).sum())
    assert left_half(a) != left_half(b)
    mirrored = np.abs(a.astype(int) - b[:, ::-1].astype(int)).mean()
    assert mirrored < np.abs(a.astype(int) - b.astype(int)).mean(), "the left-handed picture is closer to a mirror image"


def test_the_commands_still_start():
    for tool in ("pingpong_sim.py", "pingpong_live.py"):
        r = subprocess.run([sys.executable, tool, "--help"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "--plain" in r.stdout, tool


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
