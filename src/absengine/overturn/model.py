"""P(overturn | what the player knows) -- LightGBM on soft ABS labels + isotonic calibration.

Inputs are DECISION-TIME information only: the call, the PERCEIVED distance (true distance +
perception noise, never the true location), count, handedness, zone size, pitch type and
movement, umpire x edge and catcher posteriors. The label is the soft geometric probability
that ABS disagrees with the call: Phi((s_true - mu) / sigma_track).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from absengine.overturn.labels import PerceiveFit, TrackFit, wrong_call_dist

PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS", "FO", "SC", "KN", "EP"]
EDGE_CODES = ["top", "bottom", "inside", "outside"]
FEATURES = [
    "s_perc",
    "d_perc",
    "is_strike_call",
    "edge_code",
    "balls",
    "strikes",
    "stand_r",
    "p_throws_r",
    "zone_height_in",
    "pitch_code",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "ump_mean",
    "ump_sd",
    "catch_mean",
    "catch_sd",
]
CATEGORICAL = ["edge_code", "pitch_code"]


def soft_label(df: pl.DataFrame, track: TrackFit) -> np.ndarray:
    s = wrong_call_dist(df["edge_dist_in"].to_numpy(), df["is_strike_call"].to_numpy())
    return np.asarray(norm.cdf((s - track.mu_in) / track.sigma_in), dtype=float)


def sigma_for(is_strike_call: np.ndarray, perceive: dict[str, PerceiveFit]) -> np.ndarray:
    """Strike calls are challenged by the batter, ball calls by the defense (catcher)."""
    return np.where(is_strike_call, perceive["batter"].sigma_in, perceive["catcher"].sigma_in)


def features(
    df: pl.DataFrame,
    perceive: dict[str, PerceiveFit] | None,
    rng: np.random.Generator | None,
    d_perceived: np.ndarray | None = None,
) -> pl.DataFrame:
    """Feature frame. Pass `d_perceived` to use a given location read (serving); otherwise
    perception noise is simulated with the fitted sigma (training/evaluation)."""
    is_k = df["is_strike_call"].to_numpy()
    if d_perceived is None:
        assert perceive is not None and rng is not None
        d_perceived = df["edge_dist_in"].to_numpy() + rng.normal(0, 1, len(df)) * sigma_for(
            is_k, perceive
        )
    return pl.DataFrame(
        {
            "s_perc": wrong_call_dist(d_perceived, is_k),
            "d_perc": d_perceived,
            "is_strike_call": is_k.astype(np.int8),
            "edge_code": [EDGE_CODES.index(e) for e in df["edge"].to_list()],
            "balls": df["balls"],
            "strikes": df["strikes"],
            "stand_r": (df["stand"] == "R").cast(pl.Int8),
            "p_throws_r": (df["p_throws"] == "R").cast(pl.Int8),
            "zone_height_in": (df["abs_top_ft"] - df["abs_bot_ft"]) * 12,
            "pitch_code": [
                PITCH_TYPES.index(p) if p in PITCH_TYPES else len(PITCH_TYPES)
                for p in df["pitch_type"].fill_null("UN").to_list()
            ],
            "release_speed": df["release_speed"],
            "pfx_x": df["pfx_x"],
            "pfx_z": df["pfx_z"],
            "ump_mean": df["ump_mean"],
            "ump_sd": df["ump_sd"],
            "catch_mean": df["catch_mean"],
            "catch_sd": df["catch_sd"],
        }
    ).select(FEATURES)


@dataclass
class OverturnModel:
    booster: lgb.Booster
    iso: IsotonicRegression
    track: TrackFit
    perceive: dict[str, PerceiveFit]
    meta: dict[str, Any] = field(default_factory=dict)

    def predict_raw(self, X: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(X.to_pandas()), dtype=float)

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return np.asarray(self.iso.predict(self.predict_raw(X)), dtype=float)


def physics_baseline(
    X: pl.DataFrame, track: TrackFit, perceive: dict[str, PerceiveFit]
) -> np.ndarray:
    """P(ABS disagrees | perceived distance) ignoring all context:
    s_true | s_perc ~ N(s_perc, sigma_p^2) (flat prior), then the tracking probit."""
    is_k = X["is_strike_call"].to_numpy().astype(bool)
    sp = sigma_for(is_k, perceive)
    s = X["s_perc"].to_numpy()
    return np.asarray(norm.cdf((s - track.mu_in) / np.sqrt(track.sigma_in**2 + sp**2)), dtype=float)


def fit(
    X_tr: pl.DataFrame,
    y_tr: np.ndarray,
    X_cal: pl.DataFrame,
    y_cal: np.ndarray,
    params: dict[str, Any],
    seed: int,
    track: TrackFit,
    perceive: dict[str, PerceiveFit],
) -> OverturnModel:
    p = dict(params)
    n_est = p.pop("n_estimators")
    booster = lgb.train(
        {
            "objective": "cross_entropy",
            "seed": seed,
            "verbose": -1,
            "deterministic": True,
            "force_row_wise": True,
            **p,
        },
        lgb.Dataset(X_tr.to_pandas(), label=y_tr, categorical_feature=CATEGORICAL),
        num_boost_round=n_est,
    )
    raw_cal = np.asarray(booster.predict(X_cal.to_pandas()), dtype=float)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(raw_cal, y_cal)
    return OverturnModel(booster, iso, track, perceive)
