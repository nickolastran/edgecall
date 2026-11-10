"""Run expectancy: RE24, count x base-out RE, and the run value of flipping a call.

Convention (documented in DECISIONS.md): half-innings in the bottom of the 9th or later
are excluded because walk-offs truncate them (runs-to-end is censored).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from absengine.features.gamestate import N_BASE, N_BASE_OUT, N_COUNT, N_OUTS, after_call
from absengine.features.gamestate import with_state as _with_state

HALF = ["game_pk", "inning", "is_top"]


def runs_to_end(pitches: pl.LazyFrame, regulation_innings: int = 9) -> pl.LazyFrame:
    """Each pitch + runs its batting team scores from this pitch to the end of the half."""
    return (
        _with_state(pitches)
        .filter(~((~pl.col("is_top")) & (pl.col("inning") >= regulation_innings)))
        .with_columns(half_final=pl.col("post_bat_score").max().over(HALF))
        .with_columns(runs_to_end=pl.col("half_final") - pl.col("bat_score"))
    )


def re24(rte: pl.LazyFrame) -> pl.DataFrame:
    """RE at the start of each PA, by base-out state (0..23)."""
    return (
        rte.filter(pl.col("pitch_number") == 1)
        .group_by("base_out")
        .agg(re=pl.col("runs_to_end").mean(), n=pl.len())
        .sort("base_out")
        .collect()
    )


def count_re(rte: pl.LazyFrame, shrink_k: float) -> pl.DataFrame:
    """RE by (count_idx, base_out), shrunk toward an additive prior
    RE(bo) + [RE(count) - RE(all)] for thin cells (count effects are ~state-independent)."""
    cells = rte.group_by("count_idx", "base_out").agg(s=pl.col("runs_to_end").sum(), n=pl.len())
    overall = rte.select(pl.col("runs_to_end").mean().alias("all"))
    cnt = rte.group_by("count_idx").agg(c=pl.col("runs_to_end").mean())
    marg = (
        rte.group_by("base_out")
        .agg(bo=pl.col("runs_to_end").mean())
        .join(cnt, how="cross")
        .join(overall, how="cross")
        .select("count_idx", "base_out", prior=pl.col("bo") + pl.col("c") - pl.col("all"))
    )
    grid = pl.LazyFrame(
        {
            "count_idx": np.repeat(np.arange(N_COUNT), N_BASE_OUT),
            "base_out": np.tile(np.arange(N_BASE_OUT), N_COUNT),
        }
    )
    tbl = (
        grid.join(cells, on=["count_idx", "base_out"], how="left")
        .join(marg, on=["count_idx", "base_out"], how="left")
        .with_columns(pl.col("s").fill_null(0.0), pl.col("n").fill_null(0))
        .with_columns(re=(pl.col("s") + shrink_k * pl.col("prior")) / (pl.col("n") + shrink_k))
        .select("count_idx", "base_out", "re", "n")
        .sort("count_idx", "base_out")
        .collect()
    )
    re = monotone_counts(re_array(tbl), re_array(tbl.with_columns(re=pl.col("n") + shrink_k)))
    return tbl.with_columns(re=pl.Series(re[tbl["count_idx"], tbl["base_out"]]))


def monotone_counts(re: np.ndarray, w: np.ndarray, iters: int = 20) -> np.ndarray:
    """Project RE[count, base_out] so that, within each base-out state, RE is nondecreasing
    in balls and nonincreasing in strikes (alternating weighted isotonic passes)."""
    out = re.reshape(4, 3, N_BASE_OUT).copy()
    ww = w.reshape(4, 3, N_BASE_OUT)
    inc, dec = IsotonicRegression(increasing=True), IsotonicRegression(increasing=False)
    for _ in range(iters):
        for bo in range(N_BASE_OUT):
            for s in range(3):
                out[:, s, bo] = inc.fit_transform(
                    np.arange(4), out[:, s, bo], sample_weight=ww[:, s, bo]
                )
            for b in range(4):
                out[b, :, bo] = dec.fit_transform(
                    np.arange(3), out[b, :, bo], sample_weight=ww[b, :, bo]
                )
    return out.reshape(N_COUNT, N_BASE_OUT)


def re_array(count_tbl: pl.DataFrame) -> np.ndarray:
    """Dense [count_idx, base_out] array."""
    a = np.zeros((N_COUNT, N_BASE_OUT))
    a[count_tbl["count_idx"].to_numpy(), count_tbl["base_out"].to_numpy()] = count_tbl["re"]
    return a


def value_after(
    re: np.ndarray, balls: int, strikes: int, base: int, outs: int, strike: bool
) -> float:
    """Runs (to end of half) for the batting team after the call, incl. runs forced in."""
    t = after_call(balls, strikes, base, outs, strike)
    if t.outs >= N_OUTS:
        return float(t.runs)
    return t.runs + float(re[t.balls * 3 + t.strikes, t.base * N_OUTS + t.outs])


def flip_values(count_tbl: pl.DataFrame) -> pl.DataFrame:
    """For every count x base x outs: RE if ball, RE if strike, and flip_runs = ball - strike
    (the batting team's run value of the pitch being called a ball rather than a strike)."""
    re = re_array(count_tbl)
    rows = []
    for b in range(4):
        for s in range(3):
            for base in range(N_BASE):
                for outs in range(N_OUTS):
                    vb = value_after(re, b, s, base, outs, strike=False)
                    vs = value_after(re, b, s, base, outs, strike=True)
                    rows.append((b, s, base, outs, re[b * 3 + s, base * N_OUTS + outs], vb, vs))
    return pl.DataFrame(
        rows,
        schema=["balls", "strikes", "base", "outs", "re_now", "re_if_ball", "re_if_strike"],
        orient="row",
    ).with_columns(flip_runs=pl.col("re_if_ball") - pl.col("re_if_strike"))
