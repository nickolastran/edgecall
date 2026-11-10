"""Spec'd properties of EV and the DP policy, on synthetic stationary inputs."""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from absengine.config import abs_rules
from absengine.policy import dp
from absengine.value.ev import ev_challenge, threshold

EDGES = [-3, -1, 0, 1, 3]
T, NB, MAXC = 18, len(EDGES) + 1, 3


def _solve(retain: bool = True, scale: float = 1.0):  # type: ignore[no-untyped-def]
    rules = abs_rules().model_copy(update={"retain_on_success": retain})
    rng = np.random.default_rng(0)
    opps = {}
    for side in ("home", "away"):
        for t in range(T):
            for b in range(NB):
                p = np.where(rng.uniform(size=400) < 0.1, rng.uniform(0, 1, 400), 0.0)
                g = rng.uniform(0, 0.05 * scale, 400)
                opps[(side, t, b)] = (p, g)
    steps = np.full(T, 8.0)
    stay = np.zeros((T, NB, NB + 1))
    for b in range(NB):
        stay[:, b, b] = 1.0
    stay[-1] = 0.0
    stay[-1, :, NB] = 1.0  # game ends after the last half
    trans = {"home": stay, "away": stay}
    return dp.solve(opps, steps, trans, T, NB, rules, MAXC, EDGES)


R = _solve()


@given(st.floats(0, 1), st.floats(0, 1), st.floats(0.001, 0.2), st.floats(0, 0.2))
def test_ev_monotone_in_p(p1: float, p2: float, g: float, dv: float) -> None:
    lo, hi = sorted((p1, p2))
    assert ev_challenge(np.array(hi), np.array(g), np.array(dv)) >= ev_challenge(
        np.array(lo), np.array(g), np.array(dv)
    )


@given(st.floats(0.001, 0.2), st.floats(0, 0.2))
def test_threshold_is_breakeven(g: float, dv: float) -> None:
    t = threshold(np.array(g), np.array(dv))
    assert abs(float(ev_challenge(t, np.array(g), np.array(dv)))) < 1e-9


def test_never_challenge_with_zero_left() -> None:
    assert R.option_value("home", 5, True, 0, 0, 0) == float("inf")
    assert float(threshold(np.array(0.05), np.array(float("inf")))) == 1.0 or np.isnan(
        threshold(np.array(0.05), np.array(float("inf")))
    )


def test_values_nonnegative_and_increasing_in_c() -> None:
    for side in ("home", "away"):
        v = R.V[side]
        assert np.all(v >= -1e-12)
        assert np.all(np.diff(v, axis=-1) >= -1e-12)  # more challenges never hurt
        assert np.all(
            np.diff(v, axis=-1)[..., 1:] <= np.diff(v, axis=-1)[..., :-1] + 1e-12
        )  # diminishing


def test_option_value_falls_late() -> None:
    early = [R.option_value("home", 1, True, 0, 0, c) for c in (1, 2)]
    late = [R.option_value("home", 9, False, 2, 0, c) for c in (1, 2)]
    assert all(lt < e for lt, e in zip(late, early, strict=True))
    seq = [R.option_value("away", inn, True, 0, 0, 1) for inn in range(1, 10)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(seq))


def test_retention_raises_willingness() -> None:
    no_ret = _solve(retain=False)
    g = 0.03
    for inn in (1, 5, 8):
        # with retention a failed challenge is the only cost -> option value is still lower-or-equal
        # in threshold terms: p* = Δ/(g+Δ) (retain) vs p* = Δ/g (no retain)
        dv_r = R.option_value("home", inn, True, 0, 0, 1)
        dv_n = no_ret.option_value("home", inn, True, 0, 0, 1)
        p_ret = float(threshold(np.array(g), np.array(dv_r)))
        p_no = min(1.0, dv_n / g)
        assert p_ret <= p_no


def test_bucket_and_half_index() -> None:
    assert dp.bucket(np.array([-5, -3, -2, -1, 0, 1, 2, 3, 4]), EDGES).tolist() == [
        0,
        0,
        1,
        1,
        2,
        3,
        4,
        4,
        5,
    ]
    assert dp.half_index(np.array([1, 1, 9]), np.array([True, False, False])).tolist() == [0, 1, 17]
