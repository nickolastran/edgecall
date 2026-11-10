from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def row(**kw: Any) -> dict[str, Any]:
    """One lake-shaped pitch row with sane defaults."""
    base: dict[str, Any] = {
        "game_pk": 1,
        "game_date": date(2026, 4, 1),
        "season": 2026,
        "at_bat_number": 1,
        "pitch_number": 1,
        "batter": 10,
        "pitcher": 20,
        "fielder_2": 30,
        "stand": "R",
        "p_throws": "R",
        "balls": 0,
        "strikes": 0,
        "outs_when_up": 0,
        "on_1b": False,
        "on_2b": False,
        "on_3b": False,
        "inning": 1,
        "is_top": True,
        "home_team": "AAA",
        "away_team": "BBB",
        "bat_score": 0,
        "fld_score": 0,
        "post_bat_score": 0,
        "description": "ball",
        "events": None,
        "pitch_type": "FF",
        "release_speed": 94.0,
        "pfx_x": -0.5,
        "pfx_z": 1.2,
        "plate_x": 0.0,
        "plate_z": 2.5,
        "sz_top": 3.4,
        "sz_bot": 1.6,
        "vx0": 5.0,
        "vy0": -135.0,
        "vz0": -5.0,
        "ax": -10.0,
        "ay": 28.0,
        "az": -15.0,
        "release_extension": 6.5,
    }
    base.update(kw)
    return base


def frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    df = pl.DataFrame(rows, infer_schema_length=None)
    floats = ["plate_x", "plate_z", "sz_top", "sz_bot"]
    return df.with_columns(pl.col("events").cast(pl.String), pl.col(floats).cast(pl.Float64))


@pytest.fixture
def game_fixture() -> dict[str, Any]:
    out: dict[str, Any] = json.loads((FIXTURES / "game_823244.json").read_text())
    return out


@pytest.fixture
def schedule_fixture() -> dict[str, Any]:
    out: dict[str, Any] = json.loads((FIXTURES / "schedule_small.json").read_text())
    return out
