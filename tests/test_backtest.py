from __future__ import annotations

import numpy as np

from absengine.config import abs_rules
from absengine.eval.backtest import bootstrap_ci, simulate


def test_simulate_respects_rules() -> None:
    tg = np.array([0, 0, 0, 0, 0, 0, 1])
    inn = np.array([1, 2, 3, 4, 10, 10, 1])
    ok = np.array([True, False, False, False, False, False, False])
    ch = simulate(tg, inn, ok, lambda j, c: True, abs_rules())
    # win retained, two losses -> 0 left, extra-inning grant -> 1 more, new team-game resets
    assert ch.tolist() == [True, True, True, False, True, False, True]


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    games = np.repeat(np.arange(200), 2)
    vals = rng.normal(1.0, 1.0, 400)
    ci = bootstrap_ci(games, vals, 500, rng, scale=1.0)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert abs(ci["mean"] - vals.mean()) < 1e-12


def test_challenges_left_before() -> None:
    from absengine.eval.players import challenges_left_before

    gp = np.array([1, 1, 1, 1, 1, 2])
    home = np.array([True, True, False, True, True, True])
    inn = np.array([1, 2, 3, 4, 10, 1])
    ok = np.array([False, True, False, False, False, False])
    left = challenges_left_before(gp, home, inn, ok, abs_rules())
    # home: 2 -> fail -> 1 -> win (kept) -> 1 -> fail -> 0 -> extras grant -> 1; away and new
    # game start fresh
    assert left.tolist() == [2, 1, 2, 1, 1, 2]
