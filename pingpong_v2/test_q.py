"""
test_q.py -- tests for the Q-learning opponent.

    python test_q.py        (or: pytest test_q.py)
"""

import os
import random
import tempfile

import config as C
from bots import BotPlayer
from eval_opponent import play
from opponent import Shot
from pingpong_core import Game
from qopponent import N_ACTIONS, N_STATES, SERVE_STATE, QOpponent

DT = 1 / 30
L2, L1 = C.LEVELS[2], C.LEVELS[1]


def ctx(level=L2, serving=False, player_x=0.5):
    return {"level": level, "serving": serving, "player_x": player_x, "streak": 0}


def test_update_matches_the_formula():
    q = QOpponent(random.Random(0), epsilon=0.0, eps_min=0.0, ease_off_after=None)
    shot = q.choose(ctx(serving=True))                 # state = SERVE_STATE, all Q = 0
    a = shot.zone * 2 + int(shot.fast)
    q.observe(ctx(serving=True), shot, C.REWARD_PLAYER_MISSED)   # +1
    q.choose(ctx(serving=True))                        # next state known -> update applies
    expected = q.alpha * (C.REWARD_PLAYER_MISSED + q.gamma * 0.0)
    assert abs(q.q[SERVE_STATE][a] - expected) < 1e-9, q.q[SERVE_STATE][a]
    assert q.visits[SERVE_STATE][a] == 1


def test_bootstrap_uses_the_best_value_of_the_next_state():
    q = QOpponent(random.Random(0), epsilon=0.0, eps_min=0.0, ease_off_after=None)
    q.q[2] = [0.0] * N_ACTIONS
    q.q[2][4] = 2.0                                    # best known value in state 2
    shot = q.choose(ctx(serving=True))
    a = shot.zone * 2 + int(shot.fast)
    q.observe(ctx(serving=True), shot, C.REWARD_PLAYER_RETURNED)   # -0.2
    q.choose(ctx(player_x=C.ZONE_X[2]))                # next state = zone 2
    expected = q.alpha * (C.REWARD_PLAYER_RETURNED + q.gamma * 2.0)
    assert abs(q.q[SERVE_STATE][a] - expected) < 1e-9


def test_it_finds_a_planted_weakness():
    def miss_rate(kind, seeds=(0, 1, 2), shots=500, tail=250):
        rates = []
        for s in seeds:
            out, _ = play(kind, s, shots, level=2, ease_off=None)
            rates.append(sum(out[-tail:]) / tail)
        return sum(rates) / len(rates)

    q, rand, rule = miss_rate("q"), miss_rate("random"), miss_rate("rule")
    assert q > rand + 0.20, (q, rand)
    assert q > rule + 0.15, (q, rule)


def test_ease_off_after_a_run_of_misses():
    q = QOpponent(random.Random(1), ease_off_after=3)
    for _ in range(3):
        shot = q.choose(ctx(player_x=C.ZONE_X[4]))
        q.observe(ctx(player_x=C.ZONE_X[4]), shot, C.REWARD_PLAYER_MISSED)
    easy = q.choose(ctx(player_x=C.ZONE_X[4]))
    assert easy == Shot(4, False), easy              # right at the player, slow
    assert q._last is None, "must not learn from a forced shot"
    # ...and the run counter restarted, so the following shot is a normal one.
    q.observe(ctx(player_x=C.ZONE_X[4]), easy, C.REWARD_PLAYER_RETURNED)
    q.choose(ctx(player_x=C.ZONE_X[4]))
    assert q._last is not None


def test_no_fast_shots_where_the_level_forbids_them():
    assert L1.q_fast is False
    q = QOpponent(random.Random(2), epsilon=1.0, eps_min=1.0, ease_off_after=None)
    shots = [q.choose(ctx(level=L1, player_x=random.Random(i).random())) for i in range(500)]
    assert not any(s.fast for s in shots)
    q2 = QOpponent(random.Random(2), epsilon=1.0, eps_min=1.0, ease_off_after=None)
    assert any(q2.choose(ctx(level=L2)).fast for _ in range(200))


def test_epsilon_decays_but_never_below_the_floor():
    q = QOpponent(random.Random(3), epsilon=0.6, eps_decay=0.9, eps_min=0.10)
    for _ in range(200):
        q.choose(ctx())
    assert abs(q.epsilon - 0.10) < 1e-12


def test_learning_off_freezes_the_table():
    q = QOpponent(random.Random(4), learning=False, epsilon=0.3)
    before = [row[:] for row in q.q]
    for _ in range(50):
        s = q.choose(ctx())
        q.observe(ctx(), s, C.REWARD_PLAYER_MISSED)
    assert q.q == before and q.epsilon == 0.3


def test_save_and_load_round_trip():
    q = QOpponent(random.Random(5))
    for _ in range(40):
        s = q.choose(ctx())
        q.observe(ctx(), s, C.REWARD_PLAYER_MISSED)
    q.choose(ctx())
    path = os.path.join(tempfile.mkdtemp(), "q.json")
    q.save(path)
    r = QOpponent(random.Random(6))
    r.load(path)
    assert r.q == q.q and r.visits == q.visits and r.shots == q.shots
    assert abs(r.epsilon - q.epsilon) < 1e-12


def test_load_rejects_a_table_of_the_wrong_shape():
    path = os.path.join(tempfile.mkdtemp(), "bad.json")
    with open(path, "w") as f:
        f.write('{"q": [[0, 0]]}')
    try:
        QOpponent().load(path)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_describe_only_reports_what_it_has_seen():
    q = QOpponent(random.Random(7))
    assert "not enough data" in q.describe()
    q.q[0][0], q.visits[0][0] = 1.0, 10
    assert "you in zone 0" in q.describe() and "aim zone 0" in q.describe()


def test_runs_inside_the_real_game():
    rng = random.Random(9)
    q = QOpponent(rng)
    game = Game(level=2, rng=rng, policy=q)
    bot = BotPlayer(skill=0.8, rng=random.Random(9), level=2)
    for _ in range(int(120 / DT)):
        game.update(DT, bot.act(game, DT))
    assert q.shots > 10 and game.hits > 0
    assert sum(sum(v) for v in q.visits) > 0, "it should have learned from some shots"


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
