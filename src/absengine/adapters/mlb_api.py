"""MLB Stats API adapter: home-plate umpires, ABS challenge events, player heights.

Network I/O lives here only. Parsers (`parse_*`) are pure and unit-tested on fixtures.
Raw responses are cached on disk so reruns are incremental and offline-capable.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from absengine.config import DataConfig

log = logging.getLogger(__name__)

ABS_REVIEW_TYPE = "MJ"  # reviewType the feed uses for ABS ball/strike challenges (2026)


class _RateLimitedClient:
    def __init__(self, cfg: DataConfig) -> None:
        self._client = httpx.Client(
            base_url=cfg.mlb_api_base,
            headers={"User-Agent": cfg.http_user_agent},
            timeout=30.0,
        )
        self._min = cfg.http_min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(4):
            with self._lock:
                wait = self._last + self._min - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            try:
                r = self._client.get(path, params=params)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == 3:
                    raise
                log.warning("retry %s after %s", path, e)
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def _read_gz(p: Path) -> Any:
    with gzip.open(p, "rt") as f:
        return json.load(f)


def _write_gz(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    tmp.replace(p)  # atomic: a crash never leaves a truncated cache file


# --------------------------------------------------------------------------- umpires


def parse_schedule_officials(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per game: game_pk, home-plate umpire id/name."""
    rows: list[dict[str, Any]] = []
    for day in schedule.get("dates", []):
        for g in day.get("games", []):
            hp = next(
                (
                    o["official"]
                    for o in g.get("officials", [])
                    if o["officialType"] == "Home Plate"
                ),
                None,
            )
            rows.append(
                {
                    "game_pk": int(g["gamePk"]),
                    "umpire_id": int(hp["id"]) if hp else None,
                    "umpire_name": hp["fullName"] if hp else None,
                }
            )
    return rows


def fetch_umpires(cfg: DataConfig, seasons: Iterable[int], cache_dir: Path) -> pl.DataFrame:
    client = _RateLimitedClient(cfg)
    rows: list[dict[str, Any]] = []
    for season in seasons:
        p = cache_dir / f"schedule_officials_{season}.json.gz"
        if not p.exists():
            sched = client.get_json(
                "/v1/schedule",
                {"sportId": 1, "season": season, "gameType": "R", "hydrate": "officials"},
            )
            _write_gz(p, sched)
        rows.extend(parse_schedule_officials(_read_gz(p)))
    df = pl.DataFrame(
        rows,
        schema={"game_pk": pl.Int64, "umpire_id": pl.Int64, "umpire_name": pl.String},
        orient="row",
    )
    # Suspended/resumed games appear twice in schedules; keep the first non-null umpire.
    return df.sort("umpire_id", nulls_last=True).unique("game_pk", keep="first").sort("game_pk")


# --------------------------------------------------------------------------- ABS challenges

CHALLENGE_SCHEMA = {
    "game_pk": pl.Int64,
    "at_bat_number": pl.Int64,
    "pitch_number": pl.Int64,
    "inning": pl.Int64,
    "is_top": pl.Boolean,
    "challenge_team_id": pl.Int64,
    "challenger_id": pl.Int64,
    "batter_id": pl.Int64,
    "pitcher_id": pl.Int64,
    "is_overturned": pl.Boolean,
    "final_call": pl.String,
    "original_call": pl.String,
    "feed_px": pl.Float64,
    "feed_pz": pl.Float64,
    "feed_sz_top": pl.Float64,
    "feed_sz_bot": pl.Float64,
    "home_team_id": pl.Int64,
    "away_team_id": pl.Int64,
}

GAME_ABS_SCHEMA = {
    "game_pk": pl.Int64,
    "home_team_id": pl.Int64,
    "away_team_id": pl.Int64,
    "has_abs": pl.Boolean,
    "home_used_success": pl.Int64,
    "home_used_failed": pl.Int64,
    "away_used_success": pl.Int64,
    "away_used_failed": pl.Int64,
}


def _event_row(play: dict[str, Any], ev: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    pdata = ev.get("pitchData", {})
    return {
        "at_bat_index": play["atBatIndex"],
        "pitch_number": ev.get("pitchNumber"),
        "inning": play["about"]["inning"],
        "is_top": play["about"]["isTopInning"],
        "batter_id": play["matchup"]["batter"]["id"],
        "pitcher_id": play["matchup"]["pitcher"]["id"],
        "call": ev["details"].get("call", {}).get("code"),  # FINAL call (post-challenge)
        "review": review,
        "px": pdata.get("coordinates", {}).get("pX"),
        "pz": pdata.get("coordinates", {}).get("pZ"),
        "sz_top": pdata.get("strikeZoneTop"),
        "sz_bot": pdata.get("strikeZoneBottom"),
    }


def extract_game(feed: dict[str, Any]) -> dict[str, Any]:
    """Compact the ~1 MB live feed down to what this project needs.

    ABS reviews live in two places (verified on 2026 feeds): on the pitch event, or -- when
    the challenged pitch ended the PA (ball four / strike three) -- on the *play*, in which
    case it belongs to the play's last pitch.
    """
    gd, ld = feed["gameData"], feed["liveData"]
    events = []
    for play in ld["plays"]["allPlays"]:
        pitches = [e for e in play.get("playEvents", []) if e.get("isPitch", False)]
        for ev in pitches:
            if "reviewDetails" in ev:
                events.append(_event_row(play, ev, ev["reviewDetails"]))
        if "reviewDetails" in play and pitches:
            events.append(_event_row(play, pitches[-1], play["reviewDetails"]))
    return {
        "game_pk": gd["game"]["pk"],
        "home_team_id": gd["teams"]["home"]["id"],
        "away_team_id": gd["teams"]["away"]["id"],
        "abs": gd.get("absChallenges"),
        "events": events,
    }


def parse_challenges(game: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for e in game["events"]:
        rv = e["review"]
        if rv.get("reviewType") != ABS_REVIEW_TYPE or rv.get("inProgress"):
            continue
        over = rv.get("isOverturned")
        final = {"*B": "B", "V": "B", "A": "C"}.get(e["call"], e["call"])  # normalize variants
        rows.append(
            {
                "game_pk": int(game["game_pk"]),
                "at_bat_number": int(e["at_bat_index"]) + 1,  # Statcast is 1-based
                "pitch_number": e["pitch_number"],
                "inning": e["inning"],
                "is_top": e["is_top"],
                "challenge_team_id": rv.get("challengeTeamId"),
                "challenger_id": (rv.get("player") or {}).get("id"),
                "batter_id": e["batter_id"],
                "pitcher_id": e["pitcher_id"],
                "is_overturned": over,
                "final_call": final,
                # The feed (and Statcast `description`) record the call AFTER the challenge.
                "original_call": ({"B": "C", "C": "B"}.get(final) if over else final),
                "feed_px": e["px"],
                "feed_pz": e["pz"],
                "feed_sz_top": e["sz_top"],
                "feed_sz_bot": e["sz_bot"],
                "home_team_id": game["home_team_id"],
                "away_team_id": game["away_team_id"],
            }
        )
    return rows


def parse_game_abs(game: dict[str, Any]) -> dict[str, Any]:
    a = game.get("abs") or {}
    h, w = a.get("home") or {}, a.get("away") or {}
    return {
        "game_pk": int(game["game_pk"]),
        "home_team_id": game["home_team_id"],
        "away_team_id": game["away_team_id"],
        "has_abs": bool(a),
        "home_used_success": h.get("usedSuccessful"),
        "home_used_failed": h.get("usedFailed"),
        "away_used_success": w.get("usedSuccessful"),
        "away_used_failed": w.get("usedFailed"),
    }


def fetch_challenges(
    cfg: DataConfig, game_pks: Iterable[int], cache_dir: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(challenge events, per-game ABS summary). Fetches only uncached games.

    The RAW feed is cached (~140 KB/game gz) so parser fixes never require a re-scrape."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pks = sorted(set(game_pks))
    todo = [pk for pk in pks if not (cache_dir / f"{pk}.json.gz").exists()]
    log.info("feeds: %d games, %d cached, %d to fetch", len(pks), len(pks) - len(todo), len(todo))
    if todo:
        client = _RateLimitedClient(cfg)

        def one(pk: int) -> None:
            try:
                _write_gz(
                    cache_dir / f"{pk}.json.gz", client.get_json(f"/v1.1/game/{pk}/feed/live")
                )
            except Exception:  # one bad game must not kill a 2.4k-game scrape
                log.exception("feed %s failed", pk)

        with ThreadPoolExecutor(cfg.http_concurrency) as ex:
            for i, _ in enumerate(ex.map(one, todo), 1):
                if i % 200 == 0:
                    log.info("feeds fetched %d/%d", i, len(todo))
    events: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    for pk in pks:
        p = cache_dir / f"{pk}.json.gz"
        if p.exists():
            g = extract_game(_read_gz(p))
            events.extend(parse_challenges(g))
            games.append(parse_game_abs(g))
    return (
        pl.DataFrame(events, schema=CHALLENGE_SCHEMA, orient="row"),
        pl.DataFrame(games, schema=GAME_ABS_SCHEMA, orient="row"),
    )


# --------------------------------------------------------------------------- heights


def parse_height_in(h: str | None) -> float | None:
    """MLB people API height string, e.g. `6' 2"` -> 74.0."""
    if not h:
        return None
    try:
        ft, inch = h.replace('"', "").split("'")
        return float(ft) * 12 + float(inch.strip() or 0)
    except ValueError:
        return None


def fetch_listed_heights(cfg: DataConfig, player_ids: Iterable[int], cache: Path) -> pl.DataFrame:
    """Listed (not ABS-certified) heights, cached incrementally."""
    have = pl.read_parquet(cache) if cache.exists() else None
    known = set(have["player_id"].to_list()) if have is not None else set()
    todo = sorted(set(player_ids) - known)
    rows: list[dict[str, Any]] = []
    if todo:
        client = _RateLimitedClient(cfg)
        for i in range(0, len(todo), 150):
            chunk = todo[i : i + 150]
            js = client.get_json("/v1/people", {"personIds": ",".join(map(str, chunk))})
            got = {int(p["id"]): parse_height_in(p.get("height")) for p in js.get("people", [])}
            rows.extend({"player_id": pid, "listed_height_in": got.get(pid)} for pid in chunk)
    new = pl.DataFrame(
        rows, schema={"player_id": pl.Int64, "listed_height_in": pl.Float64}, orient="row"
    )
    out = pl.concat([have, new]) if have is not None else new
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache)
    return out
