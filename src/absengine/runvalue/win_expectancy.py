"""Win expectancy (home perspective) by inning x half x outs x base x score diff.

Empirical cell rates are shrunk toward a parametric prior: remaining run margin ~ Normal
with mean/variance driven by halves left and the current base-out RE, mapped through a
2-parameter logistic fit. Count is folded in at lookup time via the local run->win slope
(`we_after_call`), see DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from absengine.features.gamestate import N_BASE, N_OUTS, after_call
from absengine.features.gamestate import with_state as _with_state


@dataclass(frozen=True)
class WETable:
    p: np.ndarray  # [inning-1, is_top, outs, base, diff+M] -> P(home win)
    n: np.ndarray
    max_inning: int
    max_diff: int
    regulation: int
    ghost_runner: bool

    def lookup(self, inning: int, is_top: bool, outs: int, base: int, diff_home: int) -> float:
        i = min(inning, self.max_inning) - 1
        d = int(np.clip(diff_home, -self.max_diff, self.max_diff)) + self.max_diff
        return float(self.p[i, int(is_top), outs, base, d])


def game_outcomes(pitches: pl.LazyFrame) -> pl.LazyFrame:
    """game_pk -> home_win (ties/suspended-unresolved dropped)."""
    s = _with_state(pitches).with_columns(
        post_home=pl.when(pl.col("is_top")).then("fld_score").otherwise("post_bat_score"),
        post_away=pl.when(pl.col("is_top")).then("post_bat_score").otherwise("fld_score"),
    )
    return (
        s.group_by("game_pk")
        .agg(h=pl.col("post_home").max(), a=pl.col("post_away").max())
        .filter(pl.col("h") != pl.col("a"))
        .select("game_pk", home_win=pl.col("h") > pl.col("a"))
    )


def _halves_left(inning: np.ndarray, is_top: np.ndarray, reg: int) -> tuple[np.ndarray, np.ndarray]:
    """Full halves remaining after the current one, (home, away). Extras treated as the 9th."""
    i = np.minimum(inning, reg)
    away = reg - i
    home = np.where(is_top, reg - i + 1, reg - i)
    return home.astype(float), away.astype(float)


def _prior_z(
    inning: np.ndarray,
    is_top: np.ndarray,
    re_now: np.ndarray,
    diff: np.ndarray,
    reg: int,
    runs_per_half: float,
    var_per_half: float,
) -> np.ndarray:
    home_h, away_h = _halves_left(inning, is_top, reg)
    sign = np.where(is_top, -1.0, 1.0)  # current half's RE helps the batting team
    mean = diff + (home_h - away_h) * runs_per_half + sign * re_now
    var = (home_h + away_h) * var_per_half + np.maximum(re_now, 0.2) * var_per_half / runs_per_half
    return np.asarray(mean / np.sqrt(var), dtype=float)


def fit_we(
    pitches: pl.LazyFrame,
    re24_tbl: pl.DataFrame,
    *,
    max_inning: int,
    max_diff: int,
    prior_strength: float,
    regulation: int = 9,
    ghost_runner: bool = True,
) -> WETable:
    pa = (
        _with_state(pitches)
        .filter(pl.col("pitch_number") == 1)
        .join(game_outcomes(pitches), on="game_pk")
        .select(
            pl.col("inning").clip(1, max_inning),
            pl.col("is_top").cast(pl.Int64),
            outs=pl.col("outs_when_up"),
            base="base",
            base_out="base_out",
            diff=pl.col("score_diff_home").clip(-max_diff, max_diff),
            y=pl.col("home_win").cast(pl.Int64),
        )
        .collect()
    )
    re = np.zeros(N_BASE * N_OUTS)
    re[re24_tbl["base_out"].to_numpy()] = re24_tbl["re"].to_numpy()
    rph = float(re[0])  # bases empty, 0 out = runs per half
    var_ph = 1.0  # runs-per-half variance; calibrated jointly by the logistic slope below

    def z_of(df: pl.DataFrame) -> np.ndarray:
        return _prior_z(
            df["inning"].to_numpy(),
            df["is_top"].to_numpy().astype(bool),
            re[df["base_out"].to_numpy()],
            df["diff"].to_numpy().astype(float),
            regulation,
            rph,
            var_ph,
        )

    lr = LogisticRegression(C=1e6).fit(z_of(pa).reshape(-1, 1), pa["y"].to_numpy())

    agg = pa.group_by("inning", "is_top", "outs", "base", "diff").agg(
        w=pl.col("y").sum(), n=pl.len()
    )
    shape = (max_inning, 2, N_OUTS, N_BASE, 2 * max_diff + 1)
    grid = np.indices(shape).reshape(len(shape), -1).T
    g = pl.DataFrame(
        {
            "inning": grid[:, 0] + 1,
            "is_top": grid[:, 1],
            "outs": grid[:, 2],
            "base": grid[:, 3],
            "diff": grid[:, 4] - max_diff,
        }
    ).with_columns(base_out=pl.col("base") * N_OUTS + pl.col("outs"))
    g = g.join(agg, on=["inning", "is_top", "outs", "base", "diff"], how="left").with_columns(
        pl.col("w").fill_null(0), pl.col("n").fill_null(0)
    )
    prior = lr.predict_proba(z_of(g).reshape(-1, 1))[:, 1]
    k = prior_strength
    n = g["n"].to_numpy()
    p = ((g["w"].to_numpy() + k * prior) / (n + k)).reshape(shape)
    p = monotone_in_diff(p, (n + k).reshape(shape))
    return WETable(
        p=p,
        n=n.reshape(shape),
        max_inning=max_inning,
        max_diff=max_diff,
        regulation=regulation,
        ghost_runner=ghost_runner,
    )


def monotone_in_diff(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic projection along the last axis (home score diff):
    P(home win) must be nondecreasing in the home lead."""
    flat_p = p.reshape(-1, p.shape[-1])
    flat_w = w.reshape(-1, p.shape[-1])
    x = np.arange(p.shape[-1])
    out = np.empty_like(flat_p)
    iso = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0)
    for i in range(flat_p.shape[0]):
        out[i] = iso.fit_transform(x, flat_p[i], sample_weight=flat_w[i])
    return out.reshape(p.shape)


def _start_of_half(we: WETable, inning: int, is_top: bool, diff: int) -> float:
    base = 2 if (we.ghost_runner and inning > we.regulation) else 0  # runner on 2nd in extras
    return we.lookup(inning, is_top, 0, base, diff)


def we_state(
    we: WETable,
    inning: int,
    is_top: bool,
    outs: int,
    base: int,
    diff: int,
    count_re_delta: float = 0.0,
) -> float:
    """WE with game-ending terminals; count adjustment via the local run->win slope.

    `count_re_delta` = RE(count, state) - RE(0-0, state) for the batting team.
    """
    reg = we.regulation
    if not is_top and inning >= reg and diff > 0:
        return 1.0  # walk-off
    if outs >= N_OUTS:
        if is_top:
            if inning >= reg and diff > 0:
                return 1.0
            return _start_of_half(we, inning, False, diff)
        if inning >= reg and diff != 0:
            return float(diff > 0)
        return _start_of_half(we, inning + 1, True, diff)
    p = we.lookup(inning, is_top, outs, base, diff)
    if count_re_delta:
        slope = (
            we.lookup(inning, is_top, outs, base, diff + 1)
            - we.lookup(inning, is_top, outs, base, diff - 1)
        ) / 2
        p += slope * count_re_delta * (-1.0 if is_top else 1.0)
    return float(np.clip(p, 0.0, 1.0))


def we_after_call(
    we: WETable,
    re_count: np.ndarray,
    inning: int,
    is_top: bool,
    balls: int,
    strikes: int,
    base: int,
    outs: int,
    diff_home: int,
    strike: bool,
) -> float:
    """P(home win) right after the called pitch resolves to ball/strike."""
    t = after_call(balls, strikes, base, outs, strike)
    diff = diff_home + (-t.runs if is_top else t.runs)
    delta = 0.0
    if not t.pa_over and t.outs < N_OUTS:
        bo = t.base * N_OUTS + t.outs
        delta = float(re_count[t.balls * 3 + t.strikes, bo] - re_count[0, bo])
    return we_state(we, inning, is_top, t.outs, t.base, diff, delta)
