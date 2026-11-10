from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from absengine.eval import calibration as cal
from absengine.eval.splits import LeakageError, assert_no_leakage, by_date, by_season


def _games(pks: list[int], d: list[date]) -> pl.DataFrame:
    return pl.DataFrame({"game_pk": pks, "game_date": d, "season": [x.year for x in d]})


def test_leakage_guard_passes_on_clean_split() -> None:
    df = _games(
        [1, 2, 3, 4], [date(2026, 4, 1), date(2026, 5, 1), date(2026, 7, 1), date(2026, 8, 1)]
    )
    tr, te = by_date(df.lazy(), date(2026, 6, 30))
    assert_no_leakage(tr, te)


def test_leakage_guard_catches_time_overlap() -> None:
    tr = _games([1, 2], [date(2026, 4, 1), date(2026, 8, 1)])
    te = _games([3], [date(2026, 7, 1)])
    with pytest.raises(LeakageError, match="train ends"):
        assert_no_leakage(tr, te)


def test_leakage_guard_catches_shared_game() -> None:
    tr = _games([1], [date(2026, 4, 1)])
    te = _games([1], [date(2026, 4, 1)]).with_columns(game_date=pl.lit(date(2026, 4, 2)))
    with pytest.raises(LeakageError, match="games in both"):
        assert_no_leakage(tr, te)


def test_by_season_rejects_future_train() -> None:
    df = _games([1], [date(2026, 4, 1)]).lazy()
    with pytest.raises(LeakageError):
        by_season(df, [2026], [2025])


def test_calibration_metrics() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert cal.ece(y, p) < 0.01
    assert abs(cal.brier(y, p) - 1 / 6) < 0.005  # E[p(1-p)] for U(0,1)
    assert cal.ece(y, np.clip(p + 0.2, 0, 1)) > 0.1
    assert cal.log_loss(y, p) < cal.log_loss(y, np.full_like(p, 0.5))
    r = cal.reliability(y, p)
    assert r["n"].sum() == p.size


def test_cluster_bootstrap() -> None:
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(200), 10)
    x = rng.normal(size=clusters.size) + np.repeat(rng.normal(size=200), 10)
    pt, lo, hi = cal.cluster_bootstrap(clusters, lambda i: float(x[i].mean()), 300, seed=3)
    assert lo < pt < hi
    # clustered CI must be wider than a naive iid CI
    assert hi - lo > 2 * 1.96 * x.std() / np.sqrt(x.size)
