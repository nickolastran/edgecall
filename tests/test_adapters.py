from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from absengine.adapters import heights, mlb_api, statcast
from absengine.config import Settings, abs_rules, data_config
from absengine.contracts.schemas import Pitches, quarantine_split
from tests.conftest import frame, row


def test_parse_challenges(game_fixture: dict[str, Any]) -> None:
    rows = mlb_api.parse_challenges(game_fixture)
    assert len(rows) == 1
    r = rows[0]
    assert (r["game_pk"], r["at_bat_number"], r["pitch_number"]) == (823244, 28, 1)
    assert r["is_overturned"] is False and r["challenge_team_id"] == 147
    assert r["original_call"] == "C"
    g = mlb_api.parse_game_abs(game_fixture)
    assert g["has_abs"] and g["away_used_failed"] == 1


def test_parse_schedule_officials(schedule_fixture: dict[str, Any]) -> None:
    rows = mlb_api.parse_schedule_officials(schedule_fixture)
    assert len(rows) == 2
    assert rows[0]["umpire_id"] is not None
    assert rows[1]["umpire_id"] is None  # missing officials tolerated


@pytest.mark.parametrize(
    ("s", "v"), [("6' 2\"", 74.0), ("5' 11\"", 71.0), (None, None), ("bad", None)]
)
def test_parse_height(s: str | None, v: float | None) -> None:
    assert mlb_api.parse_height_in(s) == v


def test_quarantine_split() -> None:
    df = frame(
        [
            row(),
            row(pitch_number=2, balls=4),
            row(pitch_number=3, plate_x=None, description="ball"),
            row(pitch_number=4, plate_x=None, description="foul"),
        ]
    )
    good, bad = quarantine_split(df)
    assert good.height == 2
    assert sorted(bad["_quarantine_reason"].to_list()) == [
        "called_pitch_missing_location",
        "invalid_count",
    ]
    Pitches.validate(good)


def _raw(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Statcast-shaped raw rows (runner ids instead of booleans, inning_topbot, game_type)."""
    out = frame(rows).with_columns(
        on_1b=pl.when(pl.col("on_1b")).then(pl.lit(123)).otherwise(None),
        on_2b=pl.lit(None, dtype=pl.Int64),
        on_3b=pl.lit(None, dtype=pl.Int64),
        inning_topbot=pl.when(pl.col("is_top")).then(pl.lit("Top")).otherwise(pl.lit("Bot")),
        game_type=pl.lit("R"),
    )
    return out.drop("is_top", "season")


def test_build_lake_idempotent(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    real = Path(__file__).parents[1] / "configs"
    for f in ("abs_rules.yaml", "model.yaml"):
        (cfg_dir / f).write_text((real / f).read_text())
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    data = (real / "data.yaml").read_text()
    data = data.replace(
        "/home/ntran/stat-sightline/data/raw/statcast_*.parquet",
        str(raw_dir / "statcast_*.parquet"),
    )
    (cfg_dir / "data.yaml").write_text(data)
    s = Settings(root=tmp_path, config_dir=cfg_dir)
    d = data_config(s)

    rows = [row(pitch_number=i, on_1b=i % 2 == 0) for i in range(1, 6)] + [
        row(pitch_number=9, balls=7)
    ]
    _raw(rows).write_parquet(raw_dir / "statcast_2026-04-01_2026-04-03.parquet")
    # overlapping window duplicates rows: must be deduped
    _raw(rows[:2]).write_parquet(raw_dir / "statcast_2026-04-03_2026-04-05.parquet")

    assert statcast.build_lake(d, s, [2026]) == {2026: 5}
    part = tmp_path / "data/lake/pitches/season=2026/part.parquet"
    mtime = part.stat().st_mtime_ns
    assert statcast.build_lake(d, s, [2026]) == {2026: 5}
    assert part.stat().st_mtime_ns == mtime  # untouched on rerun
    q = json.loads((tmp_path / "reports/quarantine/pitches_2026.json").read_text())
    assert q == {"rows_in": 6, "quarantined": {"invalid_count": 1}}
    lake = statcast.scan_pitches(d, s, [2026]).collect()
    assert lake["on_1b"].to_list() == [False, True, False, True, False]
    with pytest.raises(FileNotFoundError):
        statcast.scan_pitches(d, s, [2021])


def test_heights() -> None:
    z = abs_rules().zone
    p = frame([row(batter=1, sz_top=0.535 * 74 / 12), row(batter=2, sz_top=None, pitch_number=2)])
    cert = heights.certified_heights(p.lazy(), z)
    assert cert.height == 1 and abs(cert["height_in"].item() - 74) < 1e-9
    listed = pl.DataFrame({"player_id": [1, 3, 4], "listed_height_in": [70.0, 72.0, None]})
    h = heights.combine_heights(cert, listed)
    assert h.to_dicts() == [
        {"player_id": 1, "height_in": pytest.approx(74.0), "height_source": "certified_2026"},
        {"player_id": 3, "height_in": 72.0, "height_source": "listed"},
    ]
