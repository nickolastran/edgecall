"""`make value` and `make policy` stages."""

from __future__ import annotations

import itertools
import json
import pickle

import numpy as np
import polars as pl

from absengine import config as C
from absengine.eval import plots
from absengine.obs import log, stage
from absengine.policy import dp
from absengine.runvalue.win_expectancy import WETable
from absengine.value import ev


def load_value_tables(s: C.Settings) -> ev.ValueTables:
    rv = s.artifacts / "runvalue"
    with (rv / "we.pkl").open("rb") as f:
        we: WETable = pickle.load(f)
    return ev.load_tables(pl.read_parquet(rv / "count_re.parquet"), we)


def state_grid(max_inning: int, max_diff: int) -> pl.DataFrame:
    rows = itertools.product(
        range(1, max_inning + 1),
        (True, False),
        range(3),
        range(8),
        range(-max_diff, max_diff + 1),
        range(4),
        range(3),
    )
    return pl.DataFrame(
        list(rows),
        schema=["inning", "is_top", "outs_when_up", "base", "score_diff_home", "balls", "strikes"],
        orient="row",
    )


def value_stage(s: C.Settings) -> pl.DataFrame:
    """EV table: gain (runs, wins) of a successful overturn in every state."""
    vt = load_value_tables(s)
    with stage("ev_table") as st:
        tbl = ev.flip_gain_frame(vt, state_grid(12, 6))
        st["out"] = tbl.height
    out = s.artifacts / "value"
    out.mkdir(parents=True, exist_ok=True)
    tbl.write_parquet(out / "ev_table.parquet")
    return tbl


def policy_stage(s: C.Settings) -> tuple[dp.DPResult, pl.DataFrame]:
    m, rules = C.model_config(s), C.abs_rules(s)
    pc = m.policy
    vt = load_value_tables(s)
    cal = pl.read_parquet(s.artifacts / "overturn" / "called_calibrate.parquet")
    cal = ev.flip_gain_frame(vt, cal)
    T = 2 * m.win_expectancy["max_inning"]
    with stage("dp_inputs", cal.height):
        opps, steps, trans = dp.build_inputs(
            cal, pc["score_buckets"], T, min_samples=300, max_samples=20_000, seed=m.seed
        )
    with stage("dp_solve"):
        res = dp.solve(
            opps,
            steps,
            trans,
            T,
            len(pc["score_buckets"]) + 1,
            rules,
            pc["max_challenges"],
            pc["score_buckets"],
        )
    res.meta = {"steps_per_half": steps.round(2).tolist(), "fit_rows": cal.height}
    out = s.artifacts / "policy"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dp.pkl").open("wb") as f:
        pickle.dump(res, f)

    # threshold table over decision states x challenges left x challenger side
    ev_tbl = pl.read_parquet(s.artifacts / "value" / "ev_table.parquet").filter(
        pl.col("inning") <= 12
    )
    rows = []
    for side_name in ("offense", "defense"):
        for c in range(1, rules.challenges_per_team + 2):
            team_is_home = (
                (~ev_tbl["is_top"]).to_numpy()
                if side_name == "offense"
                else ev_tbl["is_top"].to_numpy()
            )
            diff_team = np.where(team_is_home, 1, -1) * ev_tbl["score_diff_home"].to_numpy()
            dv = np.array(
                [
                    res.option_value("home" if h else "away", i, t, o, dt, c)
                    for h, i, t, o, dt in zip(
                        team_is_home,
                        ev_tbl["inning"].to_list(),
                        ev_tbl["is_top"].to_list(),
                        ev_tbl["outs_when_up"].to_list(),
                        diff_team.tolist(),
                        strict=True,
                    )
                ]
            )
            g = ev_tbl["gain_wins"].to_numpy()
            rows.append(
                ev_tbl.with_columns(
                    side=pl.lit(side_name),
                    challenges_left=pl.lit(c),
                    option_value=pl.Series(dv),
                    p_threshold=pl.Series(ev.threshold(g, dv)),
                )
            )
    # Bottom of the 9th+ with the home team ahead never happens (the game is over).
    impossible = (
        (~pl.col("is_top"))
        & (pl.col("inning") >= rules.regulation_innings)
        & (pl.col("score_diff_home") > 0)
    )
    thr = pl.concat(rows).filter(~impossible)
    thr.write_parquet(out / "thresholds.parquet")
    _figures(s, thr, res)
    summary = {
        "V_start_home": res.V["home"][0, 2].round(4).tolist(),
        "V_start_away": res.V["away"][0, 2].round(4).tolist(),
        "option_value_first_challenge_inning1": res.option_value("home", 1, True, 0, 0, 1),
        "option_value_first_challenge_inning9": res.option_value("home", 9, True, 0, 0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("policy summary %s", summary)
    return res, thr


def _figures(s: C.Settings, thr: pl.DataFrame, res: dp.DPResult) -> None:
    fig = s.reports / "figures"
    diffs = list(range(-4, 5))
    for side, count in (("offense", (0, 0)), ("offense", (3, 2)), ("defense", (3, 2))):
        for c in (1, 2):
            sub = thr.filter(
                (pl.col("side") == side)
                & (pl.col("challenges_left") == c)
                & (pl.col("base") == 0)
                & (pl.col("outs_when_up") == 0)
                & (pl.col("balls") == count[0])
                & (pl.col("strikes") == count[1])
                & (pl.col("inning") <= 9)
                & pl.col("score_diff_home").is_in(diffs)
            )
            a = np.full((18, len(diffs)), np.nan)
            for r in sub.iter_rows(named=True):
                i = 2 * (r["inning"] - 1) + (0 if r["is_top"] else 1)
                a[i, diffs.index(r["score_diff_home"])] = r["p_threshold"]
            ylab = [f"{i}{'T' if t else 'B'}" for i in range(1, 10) for t in (True, False)]
            plots.heatmap(
                a,
                [f"{x:+d}" for x in diffs],
                ylab,
                f"Min P(overturn) to challenge: {side}, {count[0]}-{count[1]}, "
                f"bases empty, 0 out, {c} left",
                fig / f"threshold_{side}_{count[0]}{count[1]}_c{c}.png",
                fmt="{:.2f}",
                cmap="magma",
            )
    t = np.arange(18)
    plots.lines(
        t,
        {
            f"{side} c={c}": np.array(
                [res.option_value(side, int(i // 2 + 1), bool(i % 2 == 0), 0, 0, c) for i in t]
            )
            for side in ("home", "away")
            for c in (1, 2)
        },
        "half-inning (0 = top 1st)",
        "option value (wins)",
        "Value of holding the c-th challenge (tied game)",
        fig / "option_value.png",
    )
