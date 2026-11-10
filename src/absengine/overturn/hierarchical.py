"""Partial-pooling umpire x edge tendencies and catcher framing (PyMC).

On borderline called pitches (|d| <= max_abs_distance_in):

    logit P(strike call) = a[edge] + b[edge] * d + u[ump, edge] + c[catcher]
    u[., edge] ~ N(0, sigma_u[edge]),  c ~ N(0, sigma_c),  sigma ~ HalfNormal(0.5)

u > 0: the umpire calls more strikes than average on that edge (a larger zone there).
c > 0: the catcher gets more strikes (framing). Small-sample umpires/catchers shrink to 0.
Fit on AGGREGATED binomial cells (ump, edge, catcher, distance bin) so it stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

EDGES = ["top", "bottom", "inside", "outside"]


@dataclass(frozen=True)
class Effects:
    umpire: pl.DataFrame  # umpire_id, edge, ump_mean, ump_sd, n
    catcher: pl.DataFrame  # catcher_id, catch_mean, catch_sd, n
    sigma_u: dict[str, float]
    sigma_c: float
    a: dict[str, float]  # edge intercepts (posterior means)
    b: dict[str, float]  # edge slopes per inch


def aggregate(df: pl.DataFrame, bin_in: float, max_abs_in: float) -> pl.DataFrame:
    return (
        df.filter(
            (pl.col("edge_dist_in").abs() <= max_abs_in)
            & pl.col("umpire_id").is_not_null()
            & pl.col("fielder_2").is_not_null()
        )
        .with_columns(dbin=((pl.col("edge_dist_in") / bin_in).floor() + 0.5) * bin_in)
        .group_by("umpire_id", "edge", "fielder_2", "dbin")
        .agg(n=pl.len(), k=pl.col("is_strike_call").sum())
    )


def fit_effects(df: pl.DataFrame, cfg: dict[str, Any], seed: int, sample: bool) -> Effects:
    import pymc as pm

    agg = aggregate(df, cfg["distance_bin_in"], cfg["max_abs_distance_in"])
    umps = np.sort(agg["umpire_id"].unique().to_numpy())
    cats = np.sort(agg["fielder_2"].unique().to_numpy())
    ui = np.searchsorted(umps, agg["umpire_id"].to_numpy())
    ci = np.searchsorted(cats, agg["fielder_2"].to_numpy())
    ei = np.array([EDGES.index(e) for e in agg["edge"].to_list()])
    d = agg["dbin"].to_numpy()
    n, k = agg["n"].to_numpy(), agg["k"].to_numpy()

    with pm.Model():
        a = pm.Normal("a", 0.0, 2.0, shape=len(EDGES))
        b = pm.Normal("b", -1.0, 1.0, shape=len(EDGES))
        sigma_u = pm.HalfNormal("sigma_u", 0.5, shape=len(EDGES))
        sigma_c = pm.HalfNormal("sigma_c", 0.5)
        z_u = pm.Normal("z_u", 0.0, 1.0, shape=(len(umps), len(EDGES)))  # non-centered
        z_c = pm.Normal("z_c", 0.0, 1.0, shape=len(cats))
        u = pm.Deterministic("u", z_u * sigma_u)
        c = pm.Deterministic("c", z_c * sigma_c)
        eta = a[ei] + b[ei] * d + u[ui, ei] + c[ci]
        pm.Binomial("k", n=n, logit_p=eta, observed=k)
        if sample and cfg["sample_draws"] == 0:
            approx = pm.fit(cfg["advi_iters"], method="advi", random_seed=seed, progressbar=False)
            idata = approx.sample(1000, random_seed=seed)
        else:
            idata = pm.sample(
                draws=cfg["draws"],
                tune=cfg["tune"],
                chains=cfg["chains"],
                random_seed=seed,
                progressbar=False,
                target_accept=0.9,
            )
    post = idata.posterior
    u_s = post["u"].stack(s=("chain", "draw")).to_numpy()  # (umps, edges, S)
    c_s = post["c"].stack(s=("chain", "draw")).to_numpy()
    n_ue = agg.group_by("umpire_id", "edge").agg(n=pl.col("n").sum())
    ump_tbl = (
        pl.DataFrame(
            {
                "umpire_id": np.repeat(umps, len(EDGES)),
                "edge": np.tile(EDGES, len(umps)),
                "ump_mean": u_s.mean(-1).ravel(),
                "ump_sd": u_s.std(-1).ravel(),
            }
        )
        .join(n_ue, on=["umpire_id", "edge"], how="left")
        .with_columns(pl.col("n").fill_null(0))
    )
    n_c = agg.group_by("fielder_2").agg(n=pl.col("n").sum()).rename({"fielder_2": "catcher_id"})
    cat_tbl = pl.DataFrame(
        {"catcher_id": cats, "catch_mean": c_s.mean(-1), "catch_sd": c_s.std(-1)}
    ).join(n_c, on="catcher_id", how="left")
    su = post["sigma_u"].mean(("chain", "draw")).to_numpy()
    return Effects(
        umpire=ump_tbl,
        catcher=cat_tbl,
        sigma_u=dict(zip(EDGES, map(float, su), strict=True)),
        sigma_c=float(post["sigma_c"].mean()),
        a=dict(zip(EDGES, map(float, post["a"].mean(("chain", "draw")).to_numpy()), strict=True)),
        b=dict(zip(EDGES, map(float, post["b"].mean(("chain", "draw")).to_numpy()), strict=True)),
    )


def strike_prob(df: pl.DataFrame, eff: Effects, use_effects: bool) -> np.ndarray:
    """P(umpire calls strike) for an `attach`-ed frame; use_effects=False is the pooled model."""
    edge = df["edge"].to_list()
    eta = (
        np.array([eff.a[e] for e in edge])
        + np.array([eff.b[e] for e in edge]) * df["edge_dist_in"].to_numpy()
    )
    if use_effects:
        eta = eta + df["ump_mean"].to_numpy() + df["catch_mean"].to_numpy()
    return np.asarray(1 / (1 + np.exp(-eta)), dtype=float)


def attach(df: pl.DataFrame, eff: Effects) -> pl.DataFrame:
    """Join posterior means/SDs; unseen umpires/catchers get the prior (0, sigma)."""
    prior_sd = pl.col("edge").replace_strict(
        eff.sigma_u, default=float(np.mean(list(eff.sigma_u.values())))
    )
    return (
        df.join(
            eff.umpire.select("umpire_id", "edge", "ump_mean", "ump_sd"),
            on=["umpire_id", "edge"],
            how="left",
        )
        .join(
            eff.catcher.select(pl.col("catcher_id").alias("fielder_2"), "catch_mean", "catch_sd"),
            on="fielder_2",
            how="left",
        )
        .with_columns(
            pl.col("ump_mean").fill_null(0.0),
            pl.col("ump_sd").fill_null(prior_sd),
            pl.col("catch_mean").fill_null(0.0),
            pl.col("catch_sd").fill_null(eff.sigma_c),
        )
    )
