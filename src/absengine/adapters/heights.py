"""Batter height for the ABS zone.

2026 Statcast sz_top/sz_bot ARE the ABS zone (constant per batter, ratio exactly
53.5/27 -- verified in recon), so certified height = sz_top / top_frac. For batters with
no 2026 appearance we fall back to MLB listed height (flagged).
"""

from __future__ import annotations

import polars as pl

from absengine.config import ZoneRules


def certified_heights(pitches_2026: pl.LazyFrame, z: ZoneRules) -> pl.DataFrame:
    return (
        pitches_2026.filter(pl.col("sz_top").is_not_null())
        .group_by("batter")
        .agg(pl.col("sz_top").median().alias("sz_top"))
        .select(
            pl.col("batter").alias("player_id"),
            (pl.col("sz_top") * 12 / z.top_frac_height).alias("height_in"),
        )
        .collect()
    )


def combine_heights(certified: pl.DataFrame, listed: pl.DataFrame) -> pl.DataFrame:
    """player_id, height_in, height_source in {certified_2026, listed}."""
    c = certified.with_columns(height_source=pl.lit("certified_2026"))
    lst = (
        listed.filter(pl.col("listed_height_in").is_not_null())
        .join(certified.select("player_id"), on="player_id", how="anti")
        .select(
            "player_id",
            pl.col("listed_height_in").alias("height_in"),
            height_source=pl.lit("listed"),
        )
    )
    return pl.concat([c, lst]).sort("player_id")
