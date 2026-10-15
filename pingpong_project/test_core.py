"""
test_core.py -- headless tests of the game rules.

    python test_core.py          # plain run, prints a summary
    pytest test_core.py          # also works if you have pytest

Nothing here needs a window, a webcam, the hub, or a network.
"""

import random

import config as C
from bots import BotPlayer
from opponent import OpponentPolicy, Shot
from pingpong_core import Game, Inputs, Phase, Swing
from score_publisher import ScorePublisher

DT = 1 / 60


def new_game(level=2, seed=1, policy=None):
    return Game(level=level, rng=random.Random(seed), policy=policy)


def run(game, seconds, wrist=(0.5, 0.88), swings_at=None, collect=None):
    """Step the game for `seconds`. swings_at = {phase_time: Swing} fired once
    the INCOMING phase_t reaches that time."""
    swings_at = dict(swings_at or {})
    t = 0.0
    while t < seconds:
        inp = Inputs(wrist=wrist)
        if game.phase == Phase.INCOMING:
            for when in sorted(swings_at):
                if game.phase_t + DT >= when:
                    inp.swings.append(swings_at.pop(when))
        events = game.update(DT, inp)
        if collect is not None:
            collect.extend(events)
        t += DT
    return game


def to_first_incoming(game, ring_zone=None):
    """Start the game and step until the first ball is in flight. Optionally
    force the first landing zone."""
    game.update(DT, Inputs(wrist=(0.5, 0.88), start_card=True))
    assert game.phase == Phase.COUNTDOWN
    events = []
    while game.phase != Phase.INCOMING:
        events += game.update(DT, Inputs(wrist=(0.5, 0.88)))
    return events


def ring_wrist(game):
    return (game.ring_x, C.RING_Y)


def step_to(game, phase_time, wrist):
    """Advance an INCOMING ball to just before `phase_time` with no swings."""
    while game.phase == Phase.INCOMING and game.phase_t + DT < phase_time:
        game.update(DT, Inputs(wrist=wrist))


# ------------------------------------------------------------------ the tests

def test_start_requires_card_and_counts_down():
    g = new_game()
    assert g.phase == Phase.IDLE
    for _ in range(120):
        g.update(DT, Inputs(wrist=(0.5, 0.5)))
    assert g.phase == Phase.IDLE, "nothing should happen without the start card"
    events = to_first_incoming(g)
    counts = [e["n"] for e in events if e["type"] == "countdown"]
    assert counts == [3, 2, 1], counts
    assert g.phase == Phase.INCOMING and g.ball is not None


def test_level_card_only_changes_level_while_idle():
    g = new_game(level=2)
    g.update(DT, Inputs(wrist=(0.5, 0.5), level_card=3))
    assert g.level_id == 3
    to_first_incoming(g)
    g.update(DT, Inputs(wrist=(0.5, 0.5), level_card=1))
    assert g.level_id == 3, "a tag flashed mid-rally must not change the difficulty"


def test_good_swing_is_a_hit_and_starts_a_return():
    g = new_game()
    to_first_incoming(g)
    T = g.ball.T
    events = []
    run(g, T + 0.1, wrist=ring_wrist(g), swings_at={T: Swing("forehand", 0.8)}, collect=events)
    hits = [e for e in events if e["type"] == "hit"]
    assert len(hits) == 1
    assert g.streak == 1 and g.best == 1
    assert any(e["type"] == "record" and e["best"] == 1 for e in events)
    assert hits[0]["accuracy"] > 0.95, "wrist was dead center on the ring"


def test_no_swing_is_a_miss():
    g = new_game()
    to_first_incoming(g)
    events = []
    run(g, g.ball.T + C.HIT_WINDOW + 0.2, wrist=ring_wrist(g), collect=events)
    miss = [e for e in events if e["type"] == "miss"]
    assert miss and miss[0]["reason"] == "no swing"
    assert g.streak == 0 and g.phase == Phase.RESERVE


def test_swing_too_early_is_a_miss():
    g = new_game()
    to_first_incoming(g)
    T = g.ball.T
    events = []
    run(g, T + C.HIT_WINDOW + 0.3, wrist=ring_wrist(g),
        swings_at={T - 0.7: Swing("flat", 0.5)}, collect=events)
    assert any(e["type"] == "whiff" and e["reason"] == "too early" for e in events)
    miss = [e for e in events if e["type"] == "miss"]
    assert miss and miss[0]["reason"] == "too early"


def test_swing_from_the_wrong_place_is_a_miss():
    g = new_game()
    to_first_incoming(g)
    T = g.ball.T
    far = (g.ring_x + 0.35 if g.ring_x < 0.5 else g.ring_x - 0.35, C.RING_Y)
    events = []
    run(g, T + 0.2, wrist=far, swings_at={T: Swing("flat", 0.5)}, collect=events)
    miss = [e for e in events if e["type"] == "miss"]
    assert miss and miss[0]["reason"] == "out of position"


def test_swing_cooldown_ignores_spam():
    g = new_game()
    to_first_incoming(g)
    T = g.ball.T
    events = []
    run(g, T + 0.15, wrist=ring_wrist(g),
        swings_at={T - 0.7: Swing("flat"), T - 0.6: Swing("flat"), T - 0.5: Swing("flat")},
        collect=events)
    assert sum(e["type"] == "swing_ignored" for e in events) == 2
    # The first early swing used up the cooldown across the window start...
    assert not any(e["type"] == "hit" for e in events)


def test_latency_compensation_moves_a_late_arriving_swing_into_the_window():
    g = new_game()
    to_first_incoming(g)
    T = g.ball.T
    # The swing is *reported* 0.3 s after it happened (slow BLE), but it
    # carries age=0.3 so the game places it where it really belongs.
    step_to(g, T + 0.15, ring_wrist(g))   # now late relative to the ball...
    # ...but a swing that actually happened at ~T (age = phase_t - T) is still a hit.
    age = g.phase_t - T
    events = g.update(DT, Inputs(wrist=ring_wrist(g), swings=[Swing("flat", 0.5, age=age)]))
    assert any(e["type"] == "hit" for e in events), events


def test_a_swing_reported_after_the_window_still_counts_when_the_game_waits_for_it():
    # An IMU recognizer reports ~0.26 s after the swing's peak. Without a report
    # delay the game has already called a miss; with one it accepts the swing.
    def play(delay):
        old = C.SWING_REPORT_DELAY
        C.SWING_REPORT_DELAY = delay
        try:
            g = new_game()
            to_first_incoming(g)
            T = g.ball.T
            wrist = ring_wrist(g)
            events = []
            while g.phase == Phase.INCOMING and g.phase_t < T + C.HIT_WINDOW + 0.12:
                events += g.update(DT, Inputs(wrist=wrist))
            if g.phase == Phase.INCOMING:
                age = g.phase_t - T                   # the swing happened right at T
                events += g.update(DT, Inputs(wrist=wrist, swings=[Swing("flat", 0.5, age=age)]))
            return g, events
        finally:
            C.SWING_REPORT_DELAY = old
    g, ev = play(0.0)
    assert any(e["type"] == "miss" for e in ev) and g.streak == 0
    g, ev = play(0.3)
    assert any(e["type"] == "hit" for e in ev) and g.streak == 1


def test_position_is_judged_where_the_wrist_was_when_you_swung():
    old = C.SWING_REPORT_DELAY
    C.SWING_REPORT_DELAY = 0.3
    try:
        # In the ring when the swing happened, wandered off by the time it was reported -> hit.
        g = new_game()
        to_first_incoming(g)
        T = g.ball.T
        away = (g.ring_x + 0.4 if g.ring_x < 0.5 else g.ring_x - 0.4, C.RING_Y)
        while g.phase_t + DT < T:
            g.update(DT, Inputs(wrist=ring_wrist(g)))
        for _ in range(int(0.15 / DT)):
            g.update(DT, Inputs(wrist=away))
        events = g.update(DT, Inputs(wrist=away, swings=[Swing("flat", 0.5, age=0.18)]))
        assert any(e["type"] == "hit" for e in events), events

        # The reverse: out of position when swinging, back in the ring when reported -> miss.
        g = new_game()
        to_first_incoming(g)
        T = g.ball.T
        while g.phase_t + DT < T - 0.05:
            g.update(DT, Inputs(wrist=away))
        for _ in range(int(0.15 / DT)):
            g.update(DT, Inputs(wrist=ring_wrist(g)))
        events = g.update(DT, Inputs(wrist=ring_wrist(g), swings=[Swing("flat", 0.5, age=0.18)]))
        assert any(e["type"] == "miss" and e["reason"] == "out of position" for e in events), events
    finally:
        C.SWING_REPORT_DELAY = old


def test_miss_resets_streak_but_keeps_best():
    g = new_game()
    to_first_incoming(g)
    g.streak, g.best = 4, 4
    events = []
    run(g, g.ball.T + C.HIT_WINDOW + 0.2, wrist=ring_wrist(g), collect=events)
    assert g.streak == 0 and g.best == 4
    assert not any(e["type"] == "record" for e in events)


def test_pose_loss_freezes_the_point_then_aborts():
    g = new_game()
    to_first_incoming(g)
    before = g.phase_t
    for _ in range(60):
        g.update(DT, Inputs(wrist=None))
    assert g.paused and g.phase == Phase.INCOMING
    assert abs(g.phase_t - before) < 1e-9, "ball must not advance while the wrist is lost"
    g.update(DT, Inputs(wrist=(0.5, 0.8)))
    assert not g.paused
    # Stay lost past the timeout:
    for _ in range(int((C.POSE_LOST_TIMEOUT + 1) / DT)):
        g.update(DT, Inputs(wrist=None))
    assert g.phase == Phase.IDLE


def test_opponent_can_be_beaten_by_a_hard_shot_across_the_court():
    g = new_game(level=3)
    to_first_incoming(g)
    g.opp_x = 0.10                       # opponent parked on the left
    g.ring_x = 0.90                      # ball lands on the far right for us
    g.ball.x1 = 0.90
    T = g.ball.T
    events = []
    # forehand + max power from the right edge sends it further right still.
    run(g, T + 1.2, wrist=ring_wrist(g), swings_at={T: Swing("forehand", 1.0)}, collect=events)
    assert any(e["type"] == "winner" for e in events), [e["type"] for e in events]
    assert g.streak == 1, "a winner keeps the streak alive"
    assert g.phase in (Phase.RESERVE, Phase.INCOMING)


def test_opponent_returns_when_it_can_reach():
    g = new_game(level=1)
    to_first_incoming(g)
    g.opp_x = 0.5
    T = g.ball.T
    events = []
    # Flat, soft shot straight back: easy to reach at level 1.
    g.ring_x = 0.5
    g.ball.x1 = 0.5
    run(g, T + 2.0, wrist=ring_wrist(g), swings_at={T: Swing("flat", 0.0)}, collect=events)
    assert any(e["type"] == "opp_return" for e in events)
    assert g.phase == Phase.INCOMING and g.ball_serial >= 2


def test_opponent_policy_gets_rewards():
    class Spy(OpponentPolicy):
        def __init__(self):
            self.rewards = []
        def choose(self, ctx):
            return Shot(2, False)
        def observe(self, ctx, shot, reward):
            self.rewards.append(reward)

    spy = Spy()
    g = new_game(policy=spy)
    to_first_incoming(g)
    T = g.ball.T
    run(g, T + 0.1, wrist=ring_wrist(g), swings_at={T: Swing("flat", 0.5)})
    assert spy.rewards == [C.REWARD_PLAYER_RETURNED]
    # Now let the next ball go by untouched:
    for _ in range(int(8 / DT)):
        g.update(DT, Inputs(wrist=(0.5, 0.88)))
        if spy.rewards[-1] == C.REWARD_PLAYER_MISSED:
            break
    assert spy.rewards[-1] == C.REWARD_PLAYER_MISSED


class FakeClient:
    def __init__(self):
        self.sent = []
    def publish(self, topic, payload, retain=False):
        self.sent.append((topic, payload, retain))


def test_publisher_topic_and_float_format():
    fake = FakeClient()
    pub = ScorePublisher("Alex", client=fake)
    pub.publish_best(7)
    pub.publish_best(12.0)
    assert fake.sent[0][:2] == ("ME193/Rogers/Alex", "7.0")
    assert fake.sent[1][:2] == ("ME193/Rogers/Alex", "12.0")


def test_publisher_survives_a_dead_broker():
    pub = ScorePublisher("Alex", broker="invalid.invalid")
    pub.connect()                 # must not raise (paho may not even be installed)
    pub.publish_best(3)           # must not raise either
    assert pub.last_payload == "3.0"


def test_only_new_records_are_published():
    fake = FakeClient()
    pub = ScorePublisher("Alex", client=fake)
    g = new_game()
    bot = BotPlayer(skill=0.7, rng=random.Random(5))
    published = []
    for _ in range(int(120 / DT)):
        for e in g.update(DT, bot.act(g, DT)):
            if e["type"] == "record":
                pub.publish_best(g.best)
                published.append(g.best)
    assert published == sorted(set(published)), "records must strictly increase"
    assert [float(p[1]) for p in fake.sent] == [float(x) for x in published]


def test_perfect_bot_builds_a_streak_and_a_hopeless_one_never_does():
    good = new_game(level=1, seed=3)
    bot = BotPlayer(skill=1.0, rng=random.Random(3))
    for _ in range(int(60 / DT)):
        good.update(DT, bot.act(good, DT))
    assert good.misses == 0 and good.best >= 10, (good.best, good.misses)

    bad = new_game(level=1, seed=3)
    bot = BotPlayer(skill=0.0, rng=random.Random(3))
    for _ in range(int(60 / DT)):
        bad.update(DT, bot.act(bad, DT))
    assert bad.hits == 0 and bad.best == 0


def difficulty_report(seconds=600):
    """Not a pass/fail test: prints how each level plays against a perfect
    bot, so we can see whether rallies and winners both actually happen."""
    print("\nDifficulty report (perfect bot, %d simulated seconds per level):" % seconds)
    print(f"  {'level':<8}{'hits':>6}{'returns':>9}{'winners':>9}{'winner %':>10}")
    for lvl in (1, 2, 3):
        g = new_game(level=lvl, seed=11)
        bot = BotPlayer(skill=1.0, rng=random.Random(11))
        for _ in range(int(seconds / DT)):
            g.update(DT, bot.act(g, DT))
        pct = 100.0 * g.winners / max(1, g.hits)
        print(f"  {C.LEVELS[lvl].name:<8}{g.hits:>6}{g.returns:>9}{g.winners:>9}{pct:>9.0f}%")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:          # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    difficulty_report()
    raise SystemExit(1 if failed else 0)
