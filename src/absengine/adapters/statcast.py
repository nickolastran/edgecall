"""Statcast adapter: platform parquet cache (read-only) + own cache -> partitioned lake.

Lake layout: <lake>/pitches/season=YYYY/part.parquet. A season is rebuilt only when the
set of source files feeding it changes (manifest), so `make data` is idempotent and
incremental. Regular-season games only.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import duckdb
import polars as pl

from absengine.config import DataConfig, Settings
from absengine.contracts.schemas import Pitches, quarantine_split
from absengine.obs import log, stage

_SELECT = """
SELECT
  CAST(game_pk AS BIGINT)                          AS game_pk,
  CAST(game_date AS DATE)                          AS game_date,
  CAST(year(CAST(game_date AS DATE)) AS BIGINT)    AS season,
  CAST(at_bat_number AS BIGINT)                    AS at_bat_number,
  CAST(pitch_number AS BIGINT)                     AS pitch_number,
  CAST(batter AS BIGINT)                           AS batter,
  CAST(pitcher AS BIGINT)                          AS pitcher,
  CAST(fielder_2 AS BIGINT)                        AS fielder_2,
  CAST(stand AS VARCHAR)                           AS stand,
  CAST(p_throws AS VARCHAR)                        AS p_throws,
  CAST(balls AS BIGINT)                            AS balls,
  CAST(strikes AS BIGINT)                          AS strikes,
  CAST(outs_when_up AS BIGINT)                     AS outs_when_up,
  on_1b IS NOT NULL                                AS on_1b,
  on_2b IS NOT NULL                                AS on_2b,
  on_3b IS NOT NULL                                AS on_3b,
  CAST(inning AS BIGINT)                           AS inning,
  inning_topbot = 'Top'                            AS is_top,
  CAST(home_team AS VARCHAR)                       AS home_team,
  CAST(away_team AS VARCHAR)                       AS away_team,
  CAST(bat_score AS BIGINT)                        AS bat_score,
  CAST(fld_score AS BIGINT)                        AS fld_score,
  CAST(post_bat_score AS BIGINT)                   AS post_bat_score,
  CAST(description AS VARCHAR)                     AS description,
  CAST(events AS VARCHAR)                          AS events,
  CAST(pitch_type AS VARCHAR)                      AS pitch_type,
  CAST(release_speed AS DOUBLE)                    AS release_speed,
  CAST(pfx_x AS DOUBLE)                            AS pfx_x,
  CAST(pfx_z AS DOUBLE)                            AS pfx_z,
  CAST(plate_x AS DOUBLE)                          AS plate_x,
  CAST(plate_z AS DOUBLE)                          AS plate_z,
  CAST(sz_top AS DOUBLE)                           AS sz_top,
  CAST(sz_bot AS DOUBLE)                           AS sz_bot,
  CAST(vx0 AS DOUBLE) AS vx0, CAST(vy0 AS DOUBLE) AS vy0, CAST(vz0 AS DOUBLE) AS vz0,
  CAST(ax AS DOUBLE)  AS ax,  CAST(ay AS DOUBLE)  AS ay,  CAST(az AS DOUBLE)  AS az,
  CAST(release_extension AS DOUBLE)                AS release_extension
FROM read_parquet({files}, union_by_name = true)
WHERE game_type = 'R' AND year(CAST(game_date AS DATE)) = {season}
"""


def source_files(cfg: DataConfig, s: Settings) -> list[str]:
    files = [f for g in cfg.platform_cache_globs for f in glob.glob(g)]
    files += glob.glob(str(s.path(cfg.own_cache_glob)))
    return sorted(set(files))


def files_for_season(files: list[str], season: int) -> list[str]:
    """Cache windows are named statcast_<start>_<end>.parquet; select by start year."""
    return [f for f in files if Path(f).name.startswith(f"statcast_{season}-")]


def read_season(files: list[str], season: int) -> pl.DataFrame:
    sql = _SELECT.format(files=json.dumps(files).replace('"', "'"), season=season)
    rel = duckdb.sql(sql)
    assert rel is not None
    df: pl.DataFrame = rel.pl()
    # Platform and own caches may overlap; the grain key is unique per pitch.
    return df.unique(["game_pk", "at_bat_number", "pitch_number"], keep="first").sort(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    )


def _digest(files: list[str]) -> str:
    h = hashlib.sha256()
    for f in files:
        st = Path(f).stat()
        h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def build_lake(cfg: DataConfig, s: Settings, seasons: list[int]) -> dict[int, int]:
    """Returns rows written (or already present) per season."""
    lake = s.path(cfg.lake_dir) / "pitches"
    qdir = s.path(cfg.quarantine_dir)
    files = source_files(cfg, s)
    out: dict[int, int] = {}
    for season in seasons:
        sf = files_for_season(files, season)
        part = lake / f"season={season}" / "part.parquet"
        manifest = part.with_name("manifest.json")
        if not sf:
            log.warning("season %s: no source files, skipping", season)
            continue
        digest = _digest(sf)
        if (
            part.exists()
            and manifest.exists()
            and json.loads(manifest.read_text())["digest"] == digest
        ):
            out[season] = pl.scan_parquet(part).select(pl.len()).collect().item()
            log.info("season %s: up to date (%d rows)", season, out[season])
            continue
        with stage(f"lake:{season}") as st:
            raw = read_season(sf, season)
            good, bad = quarantine_split(raw)
            Pitches.validate(good)
            part.parent.mkdir(parents=True, exist_ok=True)
            good.write_parquet(part)
            qdir.mkdir(parents=True, exist_ok=True)
            bad.write_parquet(qdir / f"pitches_{season}.parquet")
            summary = bad.group_by("_quarantine_reason").len().sort("len", descending=True)
            (qdir / f"pitches_{season}.json").write_text(
                json.dumps(
                    {"rows_in": raw.height, "quarantined": dict(summary.iter_rows())}, indent=1
                )
            )
            manifest.write_text(json.dumps({"digest": digest, "n_files": len(sf)}))
            st["out"] = out[season] = good.height
            log.info("season %s: %d in, %d quarantined", season, raw.height, bad.height)
    return out


_FETCH = """
import sys; sys.path.insert(0, '.')
from src.stat_sightline.etl.ingest_statcast import fetch_statcast
fetch_statcast(sys.argv[1], sys.argv[2], cache_dir=sys.argv[3])
"""


def fetch_missing(cfg: DataConfig, s: Settings, seasons: list[int], retries: int = 3) -> None:
    """Pull seasons the platform hasn't cached, using the PLATFORM's own fetcher in its own
    venv (subprocess), writing into edgecall's cache. Window caching makes retries resume."""
    import subprocess

    have = source_files(cfg, s)
    own_dir = s.path(cfg.own_cache_glob).parent
    py = cfg.platform_root / ".venv" / "bin" / "python"
    for season in seasons:
        if len(files_for_season(have, season)) >= cfg.min_windows_per_season:
            continue
        start, end = (f"{season}-{md}" for md in cfg.season_window)
        for attempt in range(1, retries + 1):
            log.info("fetching statcast %s (%s..%s) attempt %d", season, start, end, attempt)
            r = subprocess.run(
                [str(py), "-c", _FETCH, start, end, str(own_dir)],
                cwd=cfg.platform_root,
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode == 0:
                break
            log.warning("statcast %s failed: %s", season, r.stderr.strip().splitlines()[-1:])
        else:
            raise RuntimeError(f"statcast fetch for {season} failed after {retries} attempts")


def scan_pitches(cfg: DataConfig, s: Settings, seasons: list[int]) -> pl.LazyFrame:
    lake = s.path(cfg.lake_dir) / "pitches"
    paths = [lake / f"season={y}" / "part.parquet" for y in seasons]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"lake partitions missing, run `make data`: {missing}")
    return pl.scan_parquet(paths)
