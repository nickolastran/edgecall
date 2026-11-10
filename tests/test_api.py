"""API smoke test against the --sample artifacts (skipped when they haven't been built)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from absengine import config as C
from absengine.api import main

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "artifacts/sample/policy/dp.pkl").exists(), reason="run `make all SAMPLE=1` first"
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ABS_SAMPLE", "1")
    monkeypatch.setenv("ABS_ROOT", str(ROOT))
    for f in (C.settings, main.engine, main.thresholds, main.players_table):
        f.cache_clear()
    yield TestClient(main.app)
    for f in (C.settings, main.engine, main.thresholds, main.players_table):
        f.cache_clear()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"ok": True, "sample": True}


def test_evaluate_obvious_miss_vs_clear_strike(client: TestClient) -> None:
    # low leverage (1st inning, 0-0, empty): only a likely overturn is worth a challenge
    base = {
        "inning": 1,
        "is_top": True,
        "balls": 0,
        "strikes": 0,
        "base": 0,
        "outs": 0,
        "call": "strike",
        "edge": "outside",
        "challenges_left": 1,
    }
    miss = client.post("/evaluate", json={**base, "d_perc": 6.0}).json()
    good = client.post("/evaluate", json={**base, "d_perc": -4.0}).json()
    assert miss["side"] == "offense"
    assert miss["p_overturn"] > good["p_overturn"]
    assert miss["challenge"] and not good["challenge"]
    none_left = client.post("/evaluate", json={**base, "d_perc": 6.0, "challenges_left": 0}).json()
    assert not none_left["challenge"]


def test_thresholds_and_umpire_404(client: TestClient) -> None:
    rows = client.get("/policy/thresholds", params={"side": "defense", "balls": 3, "limit": 5})
    assert len(rows.json()) == 5 and all(r["side"] == "defense" for r in rows.json())
    assert client.get("/umpires/1").status_code == 404


def test_players(client: TestClient) -> None:
    rows = client.get(
        "/players", params={"role": "catcher", "min_challenges": 20, "limit": 3}
    ).json()
    assert len(rows) == 3 and all(r["role"] == "catcher" and r["challenges"] >= 20 for r in rows)
    assert rows[0]["net_wins"] >= rows[1]["net_wins"]
