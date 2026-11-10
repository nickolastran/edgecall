"""Network adapters (client stubbed) and table fits on simulated games."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from absengine.adapters import mlb_api
from absengine.config import data_config
from absengine.eval import plots
from absengine.eval.calibration import reliability
from absengine.runvalue import re24 as R
from absengine.runvalue import win_expectancy as W
from tests.conftest import frame, row


def _raw_feed(pk: int) -> dict[str, Any]:
    ev = {
        "isPitch": True,
        "pitchNumber": 2,
        "details": {"call": {"code": "B"}},
        "reviewDetails": {
            "reviewType": "MJ",
            "isOverturned": True,
            "inProgress": False,
            "challengeTeamId": 1,
            "player": {"id": 99},
        },
        "pitchData": {
            "coordinates": {"pX": 0.9, "pZ": 2.0},
            "strikeZoneTop": 3.3,
            "strikeZoneBottom": 1.6,
        },
    }
    other = {"isPitch": False, "details": {}, "reviewDetails": {"reviewType": "MA"}}
    play = {
        "atBatIndex": 4,
        "about": {"inning": 2, "isTopInning": False},
        "matchup": {"batter": {"id": 5}, "pitcher": {"id": 6}},
        "playEvents": [ev, other],
    }
    last = {"isPitch": True, "pitchNumber": 6, "details": {"call": {"code": "B"}}}
    walk = {
        "atBatIndex": 5,
        "about": {"inning": 2, "isTopInning": False},
        "matchup": {"batter": {"id": 7}, "pitcher": {"id": 6}},
        "playEvents": [last],
        "reviewDetails": {
            "reviewType": "MJ",
            "isOverturned": False,
            "challengeTeamId": 2,
            "player": {"id": 8},
        },
    }
    return {
        "gameData": {
            "game": {"pk": pk},
            "teams": {"home": {"id": 1}, "away": {"id": 2}},
            "absChallenges": {"home": {"usedSuccessful": 1, "usedFailed": 0}, "away": {}},
        },
        "liveData": {"plays": {"allPlays": [play, walk]}},
    }


def test_fetch_challenges_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(self: object, path: str, params: Any = None) -> Any:
        calls.append(path)
        return _raw_feed(int(path.split("/")[3]))

    monkeypatch.setattr(mlb_api._RateLimitedClient, "get_json", fake)
    cfg = data_config()
    ch, games = mlb_api.fetch_challenges(cfg, [11, 12], tmp_path)
    assert len(calls) == 2
    assert ch.height == 4
    first = ch.filter(pl.col("game_pk") == 11).sort("at_bat_number")
    # pitch-level review, overturned: feed shows the FINAL call (B), umpire originally said C
    r0, r1 = first.row(0, named=True), first.row(1, named=True)
    assert {k: r0[k] for k in ("at_bat_number", "pitch_number", "final_call", "original_call")} == {
        "at_bat_number": 5,
        "pitch_number": 2,
        "final_call": "B",
        "original_call": "C",
    }
    # play-level review (PA-ending pitch) attaches to the play's last pitch
    assert {
        k: r1[k] for k in ("at_bat_number", "pitch_number", "original_call", "challenger_id")
    } == {"at_bat_number": 6, "pitch_number": 6, "original_call": "B", "challenger_id": 8}
    assert games["home_used_success"].to_list() == [1, 1]
    mlb_api.fetch_challenges(cfg, [11, 12], tmp_path)
    assert len(calls) == 2  # fully cached


def test_fetch_umpires_and_heights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, schedule_fixture: dict[str, Any]
) -> None:
    def fake(self: object, path: str, params: Any = None) -> Any:
        if path == "/v1/schedule":
            return schedule_fixture
        ids = [int(x) for x in params["personIds"].split(",")]
        return {"people": [{"id": i, "height": "6' 1\""} for i in ids if i != 3]}

    monkeypatch.setattr(mlb_api._RateLimitedClient, "get_json", fake)
    cfg = data_config()
    u = mlb_api.fetch_umpires(cfg, [2022], tmp_path)
    assert u.height == 2 and u["umpire_id"].null_count() == 1
    h = mlb_api.fetch_listed_heights(cfg, [1, 2, 3], tmp_path / "people.parquet")
    assert h.sort("player_id")["listed_height_in"].to_list() == [73.0, 73.0, None]
    h2 = mlb_api.fetch_listed_heights(cfg, [1, 4], tmp_path / "people.parquet")
    assert h2.height == 4


def _simulate(n_games: int, seed: int = 0) -> pl.DataFrame:
    """Crude game simulator: 3-4 PAs per half, runs ~ Poisson, random base states."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(1, n_games + 1):
        score = {True: 0, False: 0}
        ab = 0
        for inning in range(1, 10):
            for top in (True, False):
                if inning == 9 and not top and score[False] > score[True]:
                    break
                for pa in range(3):
                    ab += 1
                    add = int(rng.poisson(0.18))
                    b = int(rng.integers(0, 8))
                    rows.append(
                        row(
                            game_pk=g,
                            inning=inning,
                            is_top=top,
                            at_bat_number=ab,
                            outs_when_up=pa,
                            on_1b=bool(b & 1),
                            on_2b=bool(b & 2),
                            on_3b=bool(b & 4),
                            balls=int(rng.integers(0, 4)),
                            strikes=int(rng.integers(0, 3)),
                            bat_score=score[top],
                            fld_score=score[not top],
                            post_bat_score=score[top] + add,
                        )
                    )
                    score[top] += add
        if score[True] == score[False]:  # force a winner (no extras in the toy sim)
            rows[-1]["post_bat_score"] += 1
    return frame(rows)


def test_fit_tables_on_simulation() -> None:
    sim = _simulate(150).lazy()
    rte = R.runs_to_end(sim)
    t24 = R.re24(rte)
    cre = R.count_re(rte, shrink_k=50)
    assert cre.height == 288 and cre["re"].is_finite().all()
    we = W.fit_we(sim, t24, max_inning=10, max_diff=6, prior_strength=20)
    assert we.p.shape == (10, 2, 3, 8, 13)
    assert np.all(np.diff(we.p, axis=-1) >= -1e-9)
    assert we.lookup(1, True, 0, 0, 6) > we.lookup(1, True, 0, 0, -6)
    assert 0.3 < we.lookup(1, True, 0, 0, 0) < 0.7
    games = W.game_outcomes(sim).collect()
    assert games.height == 150


def test_plots_write(tmp_path: Path) -> None:
    plots.heatmap(np.eye(3), ["a", "b", "c"], ["x", "y", "z"], "t", tmp_path / "h.png", center=0.0)
    y = np.array([0.0, 1.0, 1.0, 0.0])
    plots.reliability_plot(
        {"m": reliability(y, np.array([0.1, 0.9, 0.8, 0.3]))}, "r", tmp_path / "r.png"
    )
    plots.lines(np.arange(3), {"a": np.arange(3)}, "x", "y", "t", tmp_path / "l.png")
    assert {p.name for p in tmp_path.iterdir()} == {"h.png", "r.png", "l.png"}
