"""Called-pitch frame, ABS labels, and the two noise models.

Two different uncertainties, deliberately separated (DECISIONS.md, O1-O3):

* sigma_track -- how well Statcast coordinates reproduce the ABS ruling. Fitted by MLE on
  REAL 2026 overturns. Used to turn geometry into (soft) "would be overturned" labels for
  every called pitch: P(ABS strike | d) = Phi(-(d - mu) / sigma_track).
* sigma_perceive -- how noisy the PLAYER's read of the location is at decision time.
  Fitted from the 2026 challenge-rate curve: P(challenge | d, call) =
  alpha * Phi((s*d - tau) / sigma_perceive), s = +1 for strike calls, -1 for ball calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm

from absengine.config import AbsRules
from absengine.features.gamestate import with_state
from absengine.features.zone import add_zone_features, zone_bounds

CALLED = ("ball", "called_strike", "blocked_ball")


def called_pitches(
    pitches: pl.LazyFrame,
    challenges: pl.DataFrame | None,
    heights: pl.DataFrame,
    umpires: pl.DataFrame,
    rules: AbsRules,
    midplate_from: int,
    plane: str = "auto",
) -> pl.DataFrame:
    """All umpire-called balls/strikes with ABS zone, signed distance, umpire's ORIGINAL
    call, and (2026) the challenge outcome. Pitches without a usable zone are dropped."""
    df = (
        with_state(pitches.filter(pl.col("description").is_in(CALLED)))
        .with_columns(final_is_strike=pl.col("description") == "called_strike")
        .collect()
    )
    if challenges is not None and challenges.height:
        ch = challenges.select(
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "is_overturned",
            "challenger_id",
            "challenge_team_id",
            "home_team_id",
        )
        df = df.join(ch, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
    else:
        df = df.with_columns(
            is_overturned=pl.lit(None, pl.Boolean),
            challenger_id=pl.lit(None, pl.Int64),
            challenge_team_id=pl.lit(None, pl.Int64),
            home_team_id=pl.lit(None, pl.Int64),
        )
    df = df.with_columns(
        challenged=pl.col("is_overturned").is_not_null(),
        # Statcast records the post-challenge call; undo overturns to get the umpire's call.
        is_strike_call=pl.when(pl.col("is_overturned").fill_null(False))
        .then(~pl.col("final_is_strike"))
        .otherwise(pl.col("final_is_strike")),
        challenger_role=pl.when(pl.col("challenger_id").is_null())
        .then(None)
        .when(pl.col("challenger_id") == pl.col("batter"))
        .then(pl.lit("batter"))
        .when(pl.col("challenger_id") == pl.col("pitcher"))
        .then(pl.lit("pitcher"))
        .when(pl.col("challenger_id") == pl.col("fielder_2"))
        .then(pl.lit("catcher"))
        .otherwise(pl.lit("other")),
    )
    # ABS zone: 2026+ Statcast sz_top/sz_bot are the certified zone; earlier seasons use height.
    h = heights.select(pl.col("player_id").alias("batter"), "height_in", "height_source")
    df = df.join(h, on="batter", how="left")
    top_h, bot_h = zone_bounds(df["height_in"].fill_null(np.nan).to_numpy(), rules.zone)
    abs_season = df["season"] >= rules.first_season
    df = df.with_columns(
        abs_top_ft=pl.when(abs_season).then(pl.col("sz_top")).otherwise(pl.Series(top_h)),
        abs_bot_ft=pl.when(abs_season).then(pl.col("sz_bot")).otherwise(pl.Series(bot_h)),
        zone_source=pl.when(abs_season).then(pl.lit("statcast_abs")).otherwise("height_source"),
    ).filter(
        pl.col("abs_top_ft").is_not_null()
        & pl.col("abs_top_ft").is_not_nan()
        & pl.col("plate_x").is_not_null()
    )
    df = df.join(umpires.select("game_pk", "umpire_id"), on="game_pk", how="left")
    if plane == "auto":  # front-edge seasons are moved to the mid-plate ABS plane
        project: np.ndarray | bool = (df["season"] < midplate_from).to_numpy()
    elif plane in ("as_is", "mid_from_front"):
        project = plane == "mid_from_front"
    else:
        raise ValueError(f"unknown plane {plane!r}")
    return add_zone_features(df, rules.zone, project=project)


def wrong_call_dist(d_in: np.ndarray, is_strike_call: np.ndarray) -> np.ndarray:
    """Distance in the direction that makes the call wrong: > 0 means ABS disagrees.
    Strike call: wrong when outside (d > 0). Ball call: wrong when inside (d < 0)."""
    return np.where(is_strike_call, d_in, -d_in)


# ------------------------------------------------------------------ sigma_track


@dataclass(frozen=True)
class TrackFit:
    mu_in: float
    sigma_in: float
    nll: float
    n: int


def p_overturn_track(s_dist: np.ndarray, fit: TrackFit) -> np.ndarray:
    return np.asarray(norm.cdf((s_dist - fit.mu_in) / fit.sigma_in), dtype=float)


def fit_track(s_dist: np.ndarray, overturned: np.ndarray) -> TrackFit:
    """MLE of (mu, sigma) for P(overturn | wrong-call distance) on real challenges."""
    y = overturned.astype(float)

    def nll(theta: np.ndarray) -> float:
        mu, log_s = theta
        p = np.clip(norm.cdf((s_dist - mu) / np.exp(log_s)), 1e-9, 1 - 1e-9)
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    r = minimize(
        nll,
        x0=np.array([0.0, np.log(0.5)]),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000},
    )
    return TrackFit(float(r.x[0]), float(np.exp(r.x[1])), float(r.fun), len(y))


# ------------------------------------------------------------------ sigma_perceive


@dataclass(frozen=True)
class PerceiveFit:
    alpha: float  # max challenge propensity
    tau_in: float  # perceived wrong-call distance at which propensity is half of alpha
    sigma_in: float  # perception noise
    n: int
    n_challenged: int


def fit_perceive(s_dist: np.ndarray, challenged: np.ndarray) -> PerceiveFit:
    """MLE of P(challenge | s) = alpha * Phi((s - tau) / sigma)."""
    y = challenged.astype(float)

    def nll(theta: np.ndarray) -> float:
        a = 1 / (1 + np.exp(-theta[0]))
        p = np.clip(a * norm.cdf((s_dist - theta[1]) / np.exp(theta[2])), 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    r = minimize(
        nll,
        x0=np.array([-1.0, 1.0, np.log(1.0)]),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 4000},
    )
    return PerceiveFit(
        float(1 / (1 + np.exp(-r.x[0]))),
        float(r.x[1]),
        float(np.exp(r.x[2])),
        len(y),
        int(y.sum()),
    )


def read_given_challenged(
    s_true: np.ndarray,
    is_strike_call: np.ndarray,
    perceive: dict[str, PerceiveFit],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the player's wrong-call read given that they challenged:
    N(s_true, sigma) truncated to (tau, inf) -- the challenge rule the propensity fit implies."""
    from scipy.stats import truncnorm

    sig = np.where(is_strike_call, perceive["batter"].sigma_in, perceive["catcher"].sigma_in)
    tau = np.where(is_strike_call, perceive["batter"].tau_in, perceive["catcher"].tau_in)
    a = (tau - s_true) / sig
    return np.asarray(
        truncnorm.rvs(a, np.inf, loc=s_true, scale=sig, random_state=rng), dtype=float
    )
