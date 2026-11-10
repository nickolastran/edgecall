"""Game-state encoding. Pure functions + equivalent polars expressions.

base state: bit-packed 1B=1, 2B=2, 3B=4  -> 0..7
base-out:   base * 3 + outs              -> 0..23
count:      balls * 3 + strikes          -> 0..11
"""

from __future__ import annotations

from typing import NamedTuple

import polars as pl

N_BASE, N_OUTS, N_BASE_OUT, N_COUNT = 8, 3, 24, 12


def base_state(on1: bool, on2: bool, on3: bool) -> int:
    return int(on1) | int(on2) << 1 | int(on3) << 2


def runners(base: int) -> tuple[bool, bool, bool]:
    if not 0 <= base < N_BASE:
        raise ValueError(f"base state {base} out of range")
    return bool(base & 1), bool(base & 2), bool(base & 4)


def base_out(base: int, outs: int) -> int:
    if not 0 <= outs < N_OUTS:
        raise ValueError(f"outs {outs} out of range")
    return base * N_OUTS + outs


def split_base_out(bo: int) -> tuple[int, int]:
    return divmod(bo, N_OUTS)


def count_index(balls: int, strikes: int) -> int:
    if not (0 <= balls <= 3 and 0 <= strikes <= 2):
        raise ValueError(f"invalid count {balls}-{strikes}")
    return balls * 3 + strikes


class Transition(NamedTuple):
    """Result of a single called pitch."""

    pa_over: bool
    balls: int
    strikes: int
    base: int
    outs: int  # may be 3 => half-inning over
    runs: int


def walk(base: int) -> tuple[int, int]:
    """Forced advance on a walk -> (new base state, runs scored)."""
    on1, on2, on3 = runners(base)
    runs = int(on1 and on2 and on3)
    return base_state(True, on2 or on1, on3 or (on1 and on2)), runs


def after_call(balls: int, strikes: int, base: int, outs: int, is_strike: bool) -> Transition:
    """State after a called ball/strike. Ignores dropped-third-strike / WP on ball four:
    the challenge question is only about the call itself."""
    count_index(balls, strikes)  # validate
    if is_strike:
        if strikes == 2:
            return Transition(True, 0, 0, base, outs + 1, 0)
        return Transition(False, balls, strikes + 1, base, outs, 0)
    if balls == 3:
        nb, runs = walk(base)
        return Transition(True, 0, 0, nb, outs, runs)
    return Transition(False, balls + 1, strikes, base, outs, 0)


# --------------------------------------------------------------------- polars expressions


def base_expr() -> pl.Expr:
    return (
        pl.col("on_1b").cast(pl.Int64)
        + pl.col("on_2b").cast(pl.Int64) * 2
        + pl.col("on_3b").cast(pl.Int64) * 4
    ).alias("base")


def with_state(df: pl.LazyFrame) -> pl.LazyFrame:
    """Adds base, base_out, count_idx, half (game_pk/inning/top key), score_diff_home."""
    return (
        df.with_columns(base_expr())
        .with_columns(
            base_out=pl.col("base") * N_OUTS + pl.col("outs_when_up"),
            count_idx=pl.col("balls") * 3 + pl.col("strikes"),
            home_score=pl.when(pl.col("is_top")).then("fld_score").otherwise("bat_score"),
            away_score=pl.when(pl.col("is_top")).then("bat_score").otherwise("fld_score"),
        )
        .with_columns(score_diff_home=pl.col("home_score") - pl.col("away_score"))
    )
