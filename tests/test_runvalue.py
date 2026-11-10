from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from absengine.features.gamestate import N_BASE_OUT, N_COUNT
from absengine.runvalue import re24 as R
from absengine.runvalue import win_expectancy as W
from tests.conftest import frame, row


def _half_inning(
    game: int, inning: int, top: bool, runs_after_pa: list[int]
) -> list[dict[str, object]]:
    """PAs at bases-empty with outs 0,1,2; batting score climbs per runs_after_pa."""
    rows, score = [], 0
    for ab, add in enumerate(runs_after_pa):
        rows.append(
            row(
                game_pk=game,
                inning=inning,
                is_top=top,
                at_bat_number=inning * 10 + ab + int(not top) * 5,
                outs_when_up=min(ab, 2),
                bat_score=score,
                post_bat_score=score + add,
            )
        )
        score += add
    return rows


def test_runs_to_end_and_re24() -> None:
    rows = _half_inning(1, 1, True, [1, 0, 0]) + _half_inning(2, 1, True, [0, 0, 0])
    rte = R.runs_to_end(frame(rows).lazy())
    t = R.re24(rte)
    empty0 = t.filter(pl.col("base_out") == 0)
    assert empty0["re"].item() == 0.5  # (1 + 0) / 2
    assert empty0["n"].item() == 2


def test_bottom_ninth_excluded() -> None:
    rows = _half_inning(1, 9, False, [1, 0, 0]) + _half_inning(1, 9, True, [0, 0, 0])
    out = R.runs_to_end(frame(rows).lazy()).collect()
    assert out["is_top"].all()


@settings(max_examples=25, deadline=None)
@given(arrays(np.float64, (N_COUNT, N_BASE_OUT), elements=st.floats(0, 3)))
def test_monotone_counts_projection(re: np.ndarray) -> None:
    out = R.monotone_counts(re, np.ones_like(re), iters=30).reshape(4, 3, N_BASE_OUT)
    assert np.all(np.diff(out, axis=0) >= -1e-6)  # more balls never hurts the batter
    assert np.all(np.diff(out, axis=1) <= 1e-6)  # more strikes never helps


def test_flip_values_nonterminal_nonnegative() -> None:
    rng = np.random.default_rng(0)
    re = R.monotone_counts(
        rng.uniform(0, 2, (N_COUNT, N_BASE_OUT)), np.ones((N_COUNT, N_BASE_OUT)), 40
    )
    tbl = pl.DataFrame(
        {
            "count_idx": np.repeat(np.arange(N_COUNT), N_BASE_OUT),
            "base_out": np.tile(np.arange(N_BASE_OUT), N_COUNT),
            "re": re.ravel(),
            "n": 1,
        }
    )
    f = R.flip_values(tbl)
    assert f.height == 12 * 24
    nonterm = f.filter((pl.col("balls") < 3) & (pl.col("strikes") < 2))
    assert (nonterm["flip_runs"] >= -1e-6).all()
    # 3-2, bases loaded: ball forces in a run, strike ends the PA with an out
    r = f.filter(
        (pl.col("balls") == 3)
        & (pl.col("strikes") == 2)
        & (pl.col("base") == 7)
        & (pl.col("outs") == 2)
    )
    assert r["re_if_strike"].item() == 0.0
    assert r["re_if_ball"].item() >= 1.0


# ------------------------------------------------------------------ win expectancy


def _we(p: float = 0.5) -> W.WETable:
    shape = (12, 2, 3, 8, 21)
    diff = np.arange(-10, 11)
    grid = np.broadcast_to(1 / (1 + np.exp(-0.6 * diff)), shape).copy()
    return W.WETable(
        p=grid, n=np.ones(shape), max_inning=12, max_diff=10, regulation=9, ghost_runner=True
    )


@settings(max_examples=25, deadline=None)
@given(arrays(np.float64, (5, 21), elements=st.floats(0, 1)))
def test_monotone_in_diff(p: np.ndarray) -> None:
    out = W.monotone_in_diff(p, np.ones_like(p))
    assert np.all(np.diff(out, axis=-1) >= -1e-9)
    assert out.min() >= 0 and out.max() <= 1


def test_we_terminals() -> None:
    we = _we()
    assert W.we_state(we, 9, False, 1, 0, 1) == 1.0  # walk-off
    assert W.we_state(we, 9, True, 3, 0, 2) == 1.0  # home leads after top 9: game over
    assert W.we_state(we, 9, False, 3, 0, -1) == 0.0  # home trails, 3 outs bottom 9
    # tie after bottom 9 -> top 10 with ghost runner on 2nd (base state 2)
    assert W.we_state(we, 9, False, 3, 0, 0) == we.lookup(10, True, 0, 2, 0)


def test_we_after_call_walkoff_walk() -> None:
    we = _we()
    re = np.zeros((N_COUNT, N_BASE_OUT))
    p = W.we_after_call(we, re, 9, False, 3, 2, 7, 2, 0, strike=False)
    assert p == 1.0
    p_k = W.we_after_call(we, re, 9, False, 3, 2, 7, 2, 0, strike=True)
    assert p_k == we.lookup(10, True, 0, 2, 0)


def test_count_adjustment_direction() -> None:
    we = _we()
    re = np.zeros((N_COUNT, N_BASE_OUT))
    re[9, 0] = 0.3  # 3-0, bases empty, 0 out is better for the batter
    home_bat = W.we_after_call(we, re, 5, False, 2, 0, 0, 0, 0, strike=False)
    away_bat = W.we_after_call(we, re, 5, True, 2, 0, 0, 0, 0, strike=False)
    base = we.lookup(5, False, 0, 0, 0)
    assert home_bat > base
    assert away_bat < we.lookup(5, True, 0, 0, 0)
