"""Per-player ABS challenge value, 2026.

For every real challenge: the win-probability the challenger's team gained when the call was
overturned, and the value of the challenge it gave up when it wasn't (the DP option value of
the c-th challenge at that moment). Net wins = gained - spent.
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl

from absengine import config as C
from absengine.adapters import mlb_api
from absengine.adapters.statcast import scan_pitches
from absengine.config import AbsRules
from absengine.features.gamestate import with_state
from absengine.obs import log
from absengine.policy import dp
from absengine.policy.build import load_value_tables
from absengine.value import ev

KEY = ["game_pk", "at_bat_number", "pitch_number"]


def challenges_left_before(
    game_pk: np.ndarray, team_home: np.ndarray, inning: np.ndarray, ok: np.ndarray, r: AbsRules
) -> np.ndarray:
    """Challenges the team held right before each of its challenges (rows in game order).
    A challenge made in extras with none left must have used the extra-inning grant."""
    order = np.lexsort((np.arange(len(game_pk)), team_home, game_pk))
    out = np.empty(len(game_pk), dtype=np.int64)
    prev, left = None, 0
    for j in order:
        k = (game_pk[j], team_home[j])
        if k != prev:
            prev, left = k, r.challenges_per_team
        if left == 0 and inning[j] > r.regulation_innings:
            left = r.extra_inning_grant
        out[j] = max(left, 1)
        if not (ok[j] and r.retain_on_success):
            left = max(left - 1, 0)
    return out


def ledger(s: C.Settings) -> pl.DataFrame:
    d, rules = C.data_config(s), C.abs_rules(s)
    lake = s.path(d.lake_dir)
    ch = pl.read_parquet(lake / "challenges.parquet").filter(pl.col("is_overturned").is_not_null())
    state = (
        with_state(scan_pitches(d, s, [rules.first_season]))
        .select(
            *KEY,
            "outs_when_up",
            "base",
            "score_diff_home",
            "balls",
            "strikes",
            "fielder_2",
            "home_team",
            "away_team",
        )
        .join(ch.select(KEY).lazy(), on=KEY, how="semi")
        .collect()
    )
    df = (
        ch.join(state, on=KEY, how="inner")
        .with_columns(
            is_strike_call=pl.col("original_call") == "C",
            team_home=pl.col("challenge_team_id") == pl.col("home_team_id"),
            role=pl.when(pl.col("challenger_id") == pl.col("batter_id"))
            .then(pl.lit("batter"))
            .when(pl.col("challenger_id") == pl.col("pitcher_id"))
            .then(pl.lit("pitcher"))
            .when(pl.col("challenger_id") == pl.col("fielder_2"))
            .then(pl.lit("catcher"))
            .otherwise(pl.lit("other")),
        )
        .with_columns(
            team=pl.when("team_home").then("home_team").otherwise("away_team"),
            diff_team=pl.when("team_home")
            .then(pl.col("score_diff_home"))
            .otherwise(-pl.col("score_diff_home")),
        )
        .sort(KEY)
    )
    log.info("player ledger: %d of %d challenges matched to Statcast", df.height, ch.height)
    df = ev.flip_gain_frame(load_value_tables(s), df)
    with (s.artifacts / "policy" / "dp.pkl").open("rb") as f:
        res: dp.DPResult = pickle.load(f)
    ok = df["is_overturned"].to_numpy()
    left = challenges_left_before(
        df["game_pk"].to_numpy(), df["team_home"].to_numpy(), df["inning"].to_numpy(), ok, rules
    )
    ov = np.array(
        [
            res.option_value("home" if h else "away", i, t, o, dt, int(c))
            for h, i, t, o, dt, c in zip(
                df["team_home"].to_list(),
                df["inning"].to_list(),
                df["is_top"].to_list(),
                df["outs_when_up"].to_list(),
                df["diff_team"].to_list(),
                left,
                strict=True,
            )
        ]
    )
    return df.with_columns(
        left_before=pl.Series(left),
        wins_gained=pl.Series(np.where(ok, df["gain_wins"].to_numpy(), 0.0)),
        runs_gained=pl.Series(np.where(ok, df["gain_runs"].to_numpy(), 0.0)),
        wins_spent=pl.Series(np.where(ok, 0.0, ov)),
    ).with_columns(net_wins=pl.col("wins_gained") - pl.col("wins_spent"))


def summarize(led: pl.DataFrame) -> pl.DataFrame:
    return (
        led.group_by("challenger_id", "role")
        .agg(
            team=pl.col("team").mode().first(),
            challenges=pl.len(),
            overturned=pl.col("is_overturned").sum(),
            wins_gained=pl.col("wins_gained").sum(),
            wins_spent=pl.col("wins_spent").sum(),
            net_wins=pl.col("net_wins").sum(),
            runs_gained=pl.col("runs_gained").sum(),
            avg_stakes=pl.col("gain_wins").mean(),
        )
        .with_columns(success=pl.col("overturned") / pl.col("challenges"))
        .sort("net_wins", descending=True)
    )


def players_stage(s: C.Settings) -> pl.DataFrame:
    """`make players`: ledger + per-player table with names."""
    led = ledger(s)
    tbl = summarize(led)
    d = C.data_config(s)
    names = mlb_api.fetch_player_names(
        d, tbl["challenger_id"].to_list(), s.path(d.people_cache).with_name("player_names.parquet")
    )
    tbl = tbl.join(
        names.rename({"player_id": "challenger_id"}), on="challenger_id", how="left"
    ).with_columns(pl.col("name").fill_null(pl.col("challenger_id").cast(pl.String)))
    out = s.artifacts / "players"
    out.mkdir(parents=True, exist_ok=True)
    led.write_parquet(out / "ledger.parquet")
    tbl.write_parquet(out / "players.parquet")
    log.info(
        "players: %d challengers; top net wins %s",
        tbl.height,
        tbl.head(3).select("name", "role", "net_wins").rows(),
    )
    return tbl
