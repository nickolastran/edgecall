"""pandera contracts for every boundary. Validation failures raise (fail loudly).

Row-level problems that are expected in raw Statcast (missing tracking, bad counts) are
routed to quarantine *before* validation by `quarantine_split`, so the contract itself
describes clean data only.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

CALLED = ("ball", "called_strike", "blocked_ball")


class Pitches(pa.DataFrameModel):
    game_pk: int = pa.Field(gt=0)
    game_date: pl.Date
    season: int = pa.Field(ge=2015, le=2100)
    at_bat_number: int = pa.Field(ge=1)
    pitch_number: int = pa.Field(ge=1)
    batter: int
    pitcher: int
    fielder_2: int = pa.Field(nullable=True)
    stand: str = pa.Field(isin=["L", "R"])
    p_throws: str = pa.Field(isin=["L", "R"])
    balls: int = pa.Field(ge=0, le=3)
    strikes: int = pa.Field(ge=0, le=2)
    outs_when_up: int = pa.Field(ge=0, le=2)
    on_1b: bool
    on_2b: bool
    on_3b: bool
    inning: int = pa.Field(ge=1, le=30)
    is_top: bool
    bat_score: int = pa.Field(ge=0)
    fld_score: int = pa.Field(ge=0)
    post_bat_score: int = pa.Field(ge=0)
    description: str
    events: str = pa.Field(nullable=True)
    plate_x: float = pa.Field(nullable=True)
    plate_z: float = pa.Field(nullable=True)
    sz_top: float = pa.Field(nullable=True)
    sz_bot: float = pa.Field(nullable=True)

    class Config:
        strict = False  # extra physics columns allowed
        unique = ["game_pk", "at_bat_number", "pitch_number"]


class Challenges(pa.DataFrameModel):
    game_pk: int
    at_bat_number: int = pa.Field(ge=1)
    pitch_number: int = pa.Field(ge=1)
    challenge_team_id: int
    challenger_id: int = pa.Field(nullable=True)
    is_overturned: bool
    final_call: str = pa.Field(isin=["B", "C"])
    original_call: str = pa.Field(isin=["B", "C"])  # umpire call before the challenge

    class Config:
        strict = False
        unique = ["game_pk", "at_bat_number", "pitch_number"]


class CalledPitchFeatures(pa.DataFrameModel):
    """Output of the zone/feature stage, input to the overturn model."""

    is_strike_call: bool
    edge_dist_in: float  # signed: > 0 means outside the ABS zone (+ball radius)
    edge: str = pa.Field(isin=["top", "bottom", "inside", "outside"])
    abs_top_ft: float = pa.Field(gt=2.0, lt=4.5)
    abs_bot_ft: float = pa.Field(gt=0.8, lt=2.5)

    class Config:
        strict = False


def quarantine_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(clean, quarantined-with-reason). Pure."""
    reason = (
        pl.when(pl.col("batter").is_null() | pl.col("pitcher").is_null())
        .then(pl.lit("missing_participant"))
        .when(~pl.col("balls").is_between(0, 3) | ~pl.col("strikes").is_between(0, 2))
        .then(pl.lit("invalid_count"))
        .when(~pl.col("outs_when_up").is_between(0, 2))
        .then(pl.lit("invalid_outs"))
        .when(pl.col("stand").is_null() | pl.col("p_throws").is_null())
        .then(pl.lit("missing_handedness"))
        .when(pl.col("bat_score").is_null() | pl.col("post_bat_score").is_null())
        .then(pl.lit("missing_score"))
        .when(
            pl.col("description").is_in(CALLED)
            & (pl.col("plate_x").is_null() | pl.col("plate_z").is_null())
        )
        .then(pl.lit("called_pitch_missing_location"))
        .otherwise(None)
    )
    tagged = df.with_columns(_quarantine_reason=reason)
    bad = tagged.filter(pl.col("_quarantine_reason").is_not_null())
    good = tagged.filter(pl.col("_quarantine_reason").is_null()).drop("_quarantine_reason")
    return good, bad
