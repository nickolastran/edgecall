"""Backtest challenge policies on held-out 2026 called pitches.

Each team walks its ADVERSE calls (strike while batting, ball while fielding) in game order,
holding challenges under the real rules. A policy decides per pitch; a successful challenge
earns the flip gain (wins / runs). Simulated outcomes use one common Bernoulli(y_soft) draw
so every policy faces the same ABS rulings; the human policy uses the real outcomes.

Approximation: game states are the ones that actually happened (a counterfactual overturn
does not re-route the rest of the game).
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from absengine import config as C
from absengine.config import AbsRules
from absengine.obs import log
from absengine.overturn.labels import wrong_call_dist
from absengine.policy import dp
from absengine.policy.build import load_value_tables
from absengine.value import ev

Decide = Callable[[int, int], bool]  # (row index, challenges left) -> challenge?


def simulate(
    team_game: np.ndarray, inning: np.ndarray, overturned: np.ndarray, decide: Decide, r: AbsRules
) -> np.ndarray:
    """Rows sorted by team-game then pitch order. Returns which rows were challenged."""
    n = len(team_game)
    chal = np.zeros(n, dtype=bool)
    c, cur = 0, 0
    for j in range(n):
        if j == 0 or team_game[j] != team_game[j - 1]:
            c, cur = r.challenges_per_team, inning[j]
        elif inning[j] != cur:
            cur = inning[j]
            if cur > r.regulation_innings and (c == 0 or not r.extra_inning_grant_only_if_empty):
                c += r.extra_inning_grant
        if c > 0 and decide(j, c):
            chal[j] = True
            if not (overturned[j] and r.retain_on_success):
                c -= 1
    return chal


def adverse_rows(called: pl.DataFrame) -> pl.DataFrame:
    """One row per (team, adverse call), sorted for `simulate`."""
    batting_home = ~pl.col("is_top")
    is_k = pl.col("is_strike_call")
    team_is_home = pl.when(is_k).then(batting_home).otherwise(~batting_home)
    return (
        called.with_columns(
            team_is_home=team_is_home,
            side=pl.when(is_k).then(pl.lit("offense")).otherwise(pl.lit("defense")),
        )
        .with_columns(
            team=pl.when("team_is_home").then("home_team").otherwise("away_team"),
            diff_team=pl.when("team_is_home")
            .then(pl.col("score_diff_home"))
            .otherwise(-pl.col("score_diff_home")),
        )
        .sort("game_pk", "team_is_home", "at_bat_number", "pitch_number")
        .with_columns(team_game=pl.struct("game_pk", "team_is_home").rank("dense"))
    )


def bootstrap_ci(
    games: np.ndarray, values: np.ndarray, n_boot: int, rng: np.random.Generator, scale: float
) -> dict[str, float]:
    """Mean per team-game x scale, with a game-level bootstrap 95% CI."""
    ug, inv = np.unique(games, return_inverse=True)
    per_game = np.bincount(inv, weights=values, minlength=len(ug))
    team_games = np.bincount(inv, minlength=len(ug)).astype(float)  # rows here = team-games
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    boots = per_game[idx].sum(1) / team_games[idx].sum(1) * scale
    return {
        "mean": float(per_game.sum() / team_games.sum() * scale),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
    }


# --------------------------------------------------------------------------- stage

SEASON_GAMES = 162


def backtest_stage(s: C.Settings) -> dict[str, Any]:
    """`make backtest`: never / naive distance / DP (perceived, oracle) / actual humans."""
    m, rules = C.model_config(s), C.abs_rules(s)
    bc = m.backtest
    rng = np.random.default_rng(m.seed)
    art = s.artifacts
    te = pl.read_parquet(art / "overturn" / "called_test.parquet")
    a = adverse_rows(ev.flip_gain_frame(load_value_tables(s), te))
    with (art / "policy" / "dp.pkl").open("rb") as f:
        res: dp.DPResult = pickle.load(f)
    perceive = json.loads((art / "overturn" / "report.json").read_text())["perceive"]

    # Decision thresholds for every challenges-left level.
    key = ["team_is_home", "inning", "is_top", "outs_when_up", "diff_team"]
    states = a.select(key).unique()
    ov = {
        c: pl.Series(
            [
                res.option_value("home" if h else "away", i, t, o, d, c)
                for h, i, t, o, d in states.iter_rows()
            ]
        )
        for c in range(1, res.max_c + 1)
    }
    a = a.join(states.with_columns(**{f"ov{c}": v for c, v in ov.items()}), on=key, how="left")
    g_w = a["gain_wins"].to_numpy()
    thr = {c: ev.threshold(g_w, a[f"ov{c}"].to_numpy()) for c in ov}

    is_k = a["is_strike_call"].to_numpy()
    s_true = wrong_call_dist(a["edge_dist_in"].to_numpy(), is_k)
    sig = np.where(is_k, perceive["batter"]["sigma_in"], perceive["catcher"]["sigma_in"])
    s_read = s_true + rng.normal(0, 1, len(a)) * sig
    y_soft, p_model = a["y_soft"].to_numpy(), a["p_overturn"].to_numpy()
    sim_ok = rng.random(len(a)) < y_soft
    tg, inn = a["team_game"].to_numpy(), a["inning"].to_numpy()

    def naive_rule(x: float) -> Decide:
        return lambda j, c: bool(s_read[j] > x)

    def dp_rule(p: np.ndarray) -> Decide:
        return lambda j, c: bool(p[j] > thr[min(c, res.max_c)][j])

    policies: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "never": (np.zeros(len(a), bool), sim_ok),
    }
    for x in bc["naive_thresholds_in"]:
        policies[f"naive_{x}in"] = (simulate(tg, inn, sim_ok, naive_rule(x), rules), sim_ok)
    policies["dp_perceived"] = (simulate(tg, inn, sim_ok, dp_rule(p_model), rules), sim_ok)
    policies["dp_oracle"] = (simulate(tg, inn, sim_ok, dp_rule(y_soft), rules), sim_ok)
    human = a["challenged"].fill_null(False).to_numpy()
    policies["human"] = (human, a["is_overturned"].fill_null(False).to_numpy())

    g_r = a["gain_runs"].to_numpy()
    side = a["side"].to_numpy()
    tg_frame = a.select("game_pk", "team_game", "team").unique("team_game").sort("team_game")
    games = tg_frame["game_pk"].to_numpy()
    out: dict[str, Any] = {"n_team_games": len(games), "n_adverse_calls": a.height, "policies": {}}
    team_rows = []
    for name, (chal, ok) in policies.items():
        won = chal & ok
        entry: dict[str, Any] = {
            "challenges_per_team_game": float(chal.sum() / len(games)),
            "overturn_rate": float(won.sum() / max(chal.sum(), 1)),
        }
        for part, mask in (
            ("all", np.ones(len(a), bool)),
            ("offense", side == "offense"),
            ("defense", side == "defense"),
        ):
            for unit, g in (("wins", g_w), ("runs", g_r)):
                per_tg = np.bincount(tg - 1, weights=won * mask * g, minlength=len(games))
                entry[f"{unit}_{part}"] = bootstrap_ci(
                    games, per_tg, bc["n_boot"], rng, SEASON_GAMES
                )
        out["policies"][name] = entry
        per_tg = np.bincount(tg - 1, weights=won * g_w, minlength=len(games))
        team_rows.append(tg_frame.with_columns(policy=pl.lit(name), wins=pl.Series(per_tg)))
    teams = (
        pl.concat(team_rows)
        .group_by("team", "policy")
        .agg(wins_per_season=pl.col("wins").mean() * SEASON_GAMES, team_games=pl.len())
        .sort("team", "policy")
    )
    dpc = policies["dp_perceived"][0]
    out["human_vs_dp"] = {
        name: {
            "n": int(mk.sum()),
            # ABS-implied overturn probability (y_soft), comparable across cells
            "p_overturn_abs": float(y_soft[mk].mean()) if mk.any() else None,
            "mean_gain_wins": float(g_w[mk].mean()) if mk.any() else None,
        }
        for name, mk in (
            ("both", human & dpc),
            ("human_only", human & ~dpc),
            ("dp_only", ~human & dpc),
        )
    }
    d = art / "backtest"
    d.mkdir(parents=True, exist_ok=True)
    teams.write_parquet(d / "teams.parquet")
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    log.info(
        "backtest wins/team-season: %s",
        {k: round(v["wins_all"]["mean"], 3) for k, v in out["policies"].items()},
    )
    return out
