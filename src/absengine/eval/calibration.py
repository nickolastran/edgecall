"""Calibration metrics and game-clustered bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Equal-width bins: mean predicted, observed rate, count."""
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return (
        pl.DataFrame({"bin": b, "p": p, "y": y})
        .group_by("bin")
        .agg(p_mean=pl.col("p").mean(), y_rate=pl.col("y").mean(), n=pl.len())
        .sort("bin")
    )


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    r = reliability(y, p, n_bins)
    w = r["n"].to_numpy() / len(y)
    return float(np.sum(w * np.abs(r["p_mean"].to_numpy() - r["y_rate"].to_numpy())))


def cluster_bootstrap(
    clusters: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point, lo, hi). Resamples whole clusters (games / team-seasons), not rows,
    because pitches within a game are dependent. `stat` receives row indices."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    point = stat(np.arange(len(clusters)))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[i] = stat(np.concatenate([rows_of[k] for k in pick]))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
