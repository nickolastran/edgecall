from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from absengine.features import gamestate as G
from tests.conftest import frame, row

bases = st.integers(0, 7)
outs = st.integers(0, 2)
balls = st.integers(0, 3)
strikes = st.integers(0, 2)


@given(bases)
def test_base_roundtrip(b: int) -> None:
    assert G.base_state(*G.runners(b)) == b


@given(bases, outs)
def test_base_out_bijective(b: int, o: int) -> None:
    bo = G.base_out(b, o)
    assert 0 <= bo < G.N_BASE_OUT
    assert G.split_base_out(bo) == (b, o)


@given(bases)
def test_walk_conserves_runners(b: int) -> None:
    nb, runs = G.walk(b)
    assert sum(G.runners(nb)) + runs == sum(G.runners(b)) + 1
    assert G.runners(nb)[0]  # batter always on first
    assert runs == (1 if b == 7 else 0)


@given(balls, strikes, bases, outs, st.booleans())
def test_after_call(b: int, s: int, base: int, o: int, strike: bool) -> None:
    t = G.after_call(b, s, base, o, strike)
    if strike:
        assert t.outs == o + (s == 2)
        assert t.pa_over == (s == 2)
        assert t.runs == 0 and t.base == base
    else:
        assert t.outs == o
        assert t.pa_over == (b == 3)
        if b < 3:
            assert (t.balls, t.strikes) == (b + 1, s)
    if t.pa_over:
        assert (t.balls, t.strikes) == (0, 0)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        G.count_index(4, 0)
    with pytest.raises(ValueError):
        G.base_out(0, 3)
    with pytest.raises(ValueError):
        G.runners(8)


def test_polars_matches_pure() -> None:
    rows = [
        row(
            on_1b=a,
            on_2b=b,
            on_3b=c,
            outs_when_up=o,
            balls=bl,
            strikes=s,
            is_top=t,
            bat_score=2,
            fld_score=5,
            pitch_number=i + 1,
        )
        for i, (a, b, c, o, bl, s, t) in enumerate(
            [(True, False, True, 2, 3, 1, True), (False, True, False, 0, 0, 2, False)]
        )
    ]
    out = G.with_state(frame(rows).lazy()).collect()
    for r, src in zip(out.iter_rows(named=True), rows, strict=True):
        assert r["base"] == G.base_state(src["on_1b"], src["on_2b"], src["on_3b"])
        assert r["base_out"] == G.base_out(r["base"], src["outs_when_up"])
        assert r["count_idx"] == G.count_index(src["balls"], src["strikes"])
        assert r["score_diff_home"] == (-3 if src["is_top"] is False else 3)
    assert out.schema["base"] == pl.Int64
