"""Immediate value of a successful overturn: ΔRE and ΔWE from flipping the call.

Gains are from the CHALLENGING team's perspective and are >= 0: the offense challenges strike
calls (wants a ball), the defense challenges ball calls (wants a strike); either way the gain
is |value(ball) - value(strike)| for that team.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from absengine.runvalue.re24 import re_array, value_after
from absengine.runvalue.win_expectancy import WETable, we_after_call


@dataclass(frozen=True)
class ValueTables:
    re: np.ndarray  # [count_idx, base_out]
    we: WETable


def load_tables(count_re: pl.DataFrame, we: WETable) -> ValueTables:
    return ValueTables(re_array(count_re), we)


def flip_gain(
    vt: ValueTables,
    inning: int,
    is_top: bool,
    outs: int,
    base: int,
    diff_home: int,
    balls: int,
    strikes: int,
) -> tuple[float, float]:
    """(gain_runs, gain_wins) for the team that wants the call flipped. Symmetric in who
    challenges: the offense gains (ball - strike), the defense gains (strike - ball)."""
    runs = value_after(vt.re, balls, strikes, base, outs, strike=False) - value_after(
        vt.re, balls, strikes, base, outs, strike=True
    )
    args = (vt.we, vt.re, inning, is_top, balls, strikes, base, outs, diff_home)
    we_ball = we_after_call(*args, strike=False)
    we_strike = we_after_call(*args, strike=True)
    off_wins = (we_ball - we_strike) * (-1.0 if is_top else 1.0)
    return abs(runs), abs(off_wins)


def flip_gain_frame(vt: ValueTables, df: pl.DataFrame) -> pl.DataFrame:
    """Vectorised-ish over a called-pitch frame (unique states are cached)."""
    cols = ["inning", "is_top", "outs_when_up", "base", "score_diff_home", "balls", "strikes"]
    uniq = df.select(cols).unique()
    vals = [flip_gain(vt, *r) for r in uniq.iter_rows()]
    uniq = uniq.with_columns(
        gain_runs=pl.Series([v[0] for v in vals]), gain_wins=pl.Series([v[1] for v in vals])
    )
    return df.join(uniq, on=cols, how="left")


def ev_challenge(p: np.ndarray, gain: np.ndarray, option_value: np.ndarray) -> np.ndarray:
    """Expected value of challenging vs holding: p*g - (1-p)*Δ (retention on success)."""
    return np.asarray(p * gain - (1 - p) * option_value, dtype=float)


def threshold(gain: np.ndarray, option_value: np.ndarray) -> np.ndarray:
    """Minimum P(overturn) at which challenging is worth it: Δ / (g + Δ)."""
    g, dv = np.asarray(gain, float), np.asarray(option_value, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(g + dv > 0, dv / (g + dv), 1.0)
    return np.asarray(t, dtype=float)
