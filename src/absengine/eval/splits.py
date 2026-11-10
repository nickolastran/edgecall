"""Time-ordered splits + the leakage guard every model fit goes through."""

from __future__ import annotations

from datetime import date

import polars as pl


class LeakageError(AssertionError):
    pass


def assert_no_leakage(
    train: pl.DataFrame | pl.LazyFrame, test: pl.DataFrame | pl.LazyFrame
) -> None:
    """Train must strictly precede test in time and share no game."""
    tr = train.lazy().select("game_pk", "game_date")
    te = test.lazy().select("game_pk", "game_date")
    tr_max = tr.select(pl.col("game_date").max()).collect().item()
    te_min = te.select(pl.col("game_date").min()).collect().item()
    if tr_max is not None and te_min is not None and tr_max >= te_min:
        raise LeakageError(f"train ends {tr_max} but test starts {te_min}")
    shared = (
        tr.select("game_pk").unique().join(te.select("game_pk").unique(), on="game_pk").collect()
    )
    if shared.height:
        raise LeakageError(f"{shared.height} games in both train and test")


def by_season(
    df: pl.LazyFrame, train: list[int], test: list[int]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    if max(train) >= min(test):
        raise LeakageError(f"train seasons {train} not strictly before test {test}")
    return df.filter(pl.col("season").is_in(train)), df.filter(pl.col("season").is_in(test))


def by_date(df: pl.LazyFrame, cutoff: date) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """train: game_date <= cutoff; test: after. Whole games stay together by construction."""
    return df.filter(pl.col("game_date") <= cutoff), df.filter(pl.col("game_date") > cutoff)
