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
