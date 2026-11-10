"""Pipeline entry points. Every command takes --sample (dev mode)."""

from __future__ import annotations

from typing import Annotated

import polars as pl
import typer

from absengine import config as C
from absengine.obs import log, setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
Sample = Annotated[bool, typer.Option("--sample", help="dev mode: sample seasons only")]


@app.callback()
def _main() -> None:
    """ABS challenge decision engine pipelines."""


def _ctx(sample: bool) -> tuple[C.Settings, C.DataConfig, list[int]]:
    setup_logging()
    s = C.Settings(sample=sample)
    d = C.data_config(s)
    return s, d, (d.sample_seasons if sample else d.seasons)


@app.command()
def data(sample: Sample = False) -> None:
    """Statcast lake + umpires + 2026 ABS challenges + heights."""
    from absengine.adapters import heights, mlb_api, statcast
    from absengine.contracts.schemas import Challenges

    s, d, seasons = _ctx(sample)
    rules = C.abs_rules(s)
    if not sample:
        statcast.fetch_missing(d, s, seasons)
    statcast.build_lake(d, s, seasons)
    lake = s.path(d.lake_dir)
    feed = s.path(d.feed_cache_dir)

    umps = mlb_api.fetch_umpires(d, seasons, feed)
    umps.write_parquet(lake / "umpires.parquet")
    log.info(
        "umpires: %d games, %d with HP ump", umps.height, umps["umpire_id"].is_not_null().sum()
    )

    abs_seasons = [y for y in seasons if y >= rules.first_season]
    if abs_seasons:
        pks = (
            statcast.scan_pitches(d, s, abs_seasons).select("game_pk").unique().collect()["game_pk"]
        )
        ch, games = mlb_api.fetch_challenges(d, pks.to_list(), feed / "games_raw")
        ch = ch.filter(pl.col("is_overturned").is_not_null())
        Challenges.validate(ch)
        ch.write_parquet(lake / "challenges.parquet")
        games.write_parquet(lake / "game_abs.parquet")
        log.info("challenges: %d events over %d games", ch.height, games.height)

    all_pitches = statcast.scan_pitches(d, s, seasons)
    cert = (
        heights.certified_heights(statcast.scan_pitches(d, s, [rules.first_season]), rules.zone)
        if rules.first_season in seasons
        else pl.DataFrame(schema={"player_id": pl.Int64, "height_in": pl.Float64})
    )
    batters = all_pitches.select("batter").unique().collect()["batter"].to_list()
    need = sorted(set(batters) - set(cert["player_id"].to_list()))
    listed = mlb_api.fetch_listed_heights(d, need, s.path(d.people_cache))
    h = heights.combine_heights(cert, listed)
    h.write_parquet(lake / "heights.parquet")
    log.info("heights: %d batters (%d certified)", h.height, cert.height)


def _fit_seasons(s: C.Settings, seasons: list[int]) -> list[int]:
    """Seasons used to FIT tables applied to the ABS season: strictly before it (no leakage),
    except in --sample mode where only the sample season exists."""
    m = C.model_config(s)
    pre = [y for y in seasons if y in m.splits["train_seasons"] + m.splits["valid_seasons"]]
    return pre or seasons


@app.command()
def runvalue(sample: Sample = False) -> None:
    """RE24 per season, count RE + flip values, win expectancy; tables + plots."""
    import pickle

    import numpy as np

    from absengine.adapters.statcast import scan_pitches
    from absengine.eval import calibration as cal
    from absengine.eval import plots
    from absengine.features.gamestate import N_BASE, N_OUTS
    from absengine.obs import stage
    from absengine.runvalue import re24 as R
    from absengine.runvalue import win_expectancy as W

    s, d, seasons = _ctx(sample)
    m = C.model_config(s)
    rules = C.abs_rules(s)
    out, fig = s.artifacts / "runvalue", s.reports / "figures"
    out.mkdir(parents=True, exist_ok=True)

    per_season = []
    for y in seasons:
        rte = R.runs_to_end(scan_pitches(d, s, [y]), rules.regulation_innings)
        per_season.append(R.re24(rte).with_columns(season=pl.lit(y)))
    re24_all = pl.concat(per_season)
    re24_all.write_parquet(out / "re24_by_season.parquet")

    fit = _fit_seasons(s, seasons)
    with stage("count_re") as st:
        rte = R.runs_to_end(scan_pitches(d, s, fit), rules.regulation_innings)
        re24_fit = R.re24(rte)
        cre = R.count_re(rte, m.runvalue["min_state_n"])
        flips = R.flip_values(cre)
        st["out"] = flips.height
    re24_fit.write_parquet(out / "re24.parquet")
    cre.write_parquet(out / "count_re.parquet")
    flips.write_parquet(out / "flip_values.parquet")

    wcfg = m.win_expectancy
    with stage("win_expectancy") as st:
        we = W.fit_we(
            scan_pitches(d, s, fit),
            re24_fit,
            max_inning=wcfg["max_inning"],
            max_diff=wcfg["max_score_diff"],
            prior_strength=wcfg["prior_strength"],
            regulation=rules.regulation_innings,
            ghost_runner=wcfg.get("ghost_runner", True),
        )
        st["out"] = int(we.p.size)
    with (out / "we.pkl").open("wb") as f:
        pickle.dump(we, f)

    # WE out-of-sample check on held-out seasons (full mode) -- PA starts.
    test = [y for y in seasons if y not in fit]
    if test:
        tp = scan_pitches(d, s, test)
        from absengine.features.gamestate import with_state

        ev = (
            with_state(tp)
            .filter(pl.col("pitch_number") == 1)
            .join(W.game_outcomes(tp), on="game_pk")
            .select("inning", "is_top", "outs_when_up", "base", "score_diff_home", "home_win")
            .collect()
        )
        p = np.array([we.lookup(*r[:5]) for r in ev.iter_rows()])
        yv = ev["home_win"].to_numpy().astype(float)
        metrics = {"brier": cal.brier(yv, p), "ece": cal.ece(yv, p), "n": len(yv)}
        log.info("WE holdout %s: %s", test, metrics)
        (out / "we_holdout.json").write_text(__import__("json").dumps(metrics, indent=1))
        plots.reliability_plot(
            {"WE": cal.reliability(yv, p)}, f"WE calibration {test}", fig / "we_calibration.png"
        )

    # Figures
    bo_labels = [f"{b} {o}o" for b in plots.BASE_LABELS for o in range(N_OUTS)]
    a = np.zeros((N_BASE, N_OUTS))
    a[re24_fit["base_out"].to_numpy() // 3, re24_fit["base_out"].to_numpy() % 3] = re24_fit["re"]
    plots.heatmap(a, ["0 out", "1 out", "2 out"], plots.BASE_LABELS, "RE24", fig / "re24.png")
    fl = np.zeros((12, 24))
    fl[
        (flips["balls"] * 3 + flips["strikes"]).to_numpy(),
        (flips["base"] * 3 + flips["outs"]).to_numpy(),
    ] = flips["flip_runs"]
    plots.heatmap(
        fl,
        bo_labels,
        plots.COUNT_LABELS,
        "Run value of ball vs strike (batting team)",
        fig / "flip_runs.png",
        fmt="{:.2f}",
    )
    diffs = np.arange(-we.max_diff, we.max_diff + 1)
    plots.lines(
        diffs,
        {
            f"inn {i} {'top' if t else 'bot'}": np.array(
                [we.lookup(i, bool(t), 0, 0, int(x)) for x in diffs]
            )
            for i in (1, 5, 9)
            for t in (1, 0)
        },
        "home score diff",
        "P(home win)",
        "Win expectancy (start of half, bases empty)",
        fig / "we_curves.png",
    )
    log.info(
        "RE24 empty/0 out = %.3f; flip 3-2 bases-loaded 2-out = %.3f",
        a[0, 0],
        flips.filter(
            (pl.col("balls") == 3)
            & (pl.col("strikes") == 2)
            & (pl.col("base") == 7)
            & (pl.col("outs") == 2)
        )["flip_runs"].item(),
    )


@app.command()
def overturn(sample: Sample = False) -> None:
    """Plane choice, sigma fits, umpire/catcher effects, calibrated P(overturn) model."""
    import json

    from absengine.overturn.pipeline import run

    s, d, seasons = _ctx(sample)
    rep = run(s, d, seasons, sample)
    log.info(
        "overturn report:\n%s",
        json.dumps(
            {k: rep[k] for k in ("plane_best_2026", "track", "perceive", "umpire_catcher_holdout")},
            indent=1,
            default=str,
        ),
    )


@app.command()
def value(sample: Sample = False) -> None:
    """Immediate ΔRE / ΔWE of an overturn in every state."""
    from absengine.policy.build import value_stage

    s, _, _ = _ctx(sample)
    value_stage(s)


@app.command()
def policy(sample: Sample = False) -> None:
    """DP option values, threshold table, heatmaps."""
    from absengine.policy.build import policy_stage

    s, _, _ = _ctx(sample)
    policy_stage(s)


@app.command()
def backtest(sample: Sample = False) -> None:
    """Policies vs actual humans on held-out 2026: wins/runs per team-season with CIs."""
    from absengine.eval.backtest import backtest_stage

    s, _, _ = _ctx(sample)
    backtest_stage(s)


@app.command("export-web")
def export_web(sample: Sample = False) -> None:
    """Write web/public/data/engine.json for the static web app."""
    from absengine.serve import Engine
    from absengine.serve import export_web as _export

    s, _, _ = _ctx(sample)
    f = _export(Engine.load(s), s.root / "web" / "public" / "data")
    log.info("wrote %s (%.0f KB)", f, f.stat().st_size / 1024)


if __name__ == "__main__":
    app()
