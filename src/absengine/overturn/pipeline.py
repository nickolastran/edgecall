"""`make overturn`: plane choice, sigma fits, umpire/catcher effects, LightGBM, evaluation."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import date

import numpy as np
import polars as pl

from absengine import config as C
from absengine.adapters.statcast import scan_pitches
from absengine.eval import calibration as cal
from absengine.eval import plots
from absengine.eval.splits import assert_no_leakage
from absengine.obs import log, stage
from absengine.overturn import hierarchical as H
from absengine.overturn import labels as L
from absengine.overturn import model as M


def challenges_remaining(called: pl.DataFrame, rules: C.AbsRules) -> pl.DataFrame:
    """Per called pitch in ABS seasons: challenges the OFFENSE and DEFENSE had available
    before this pitch, reconstructed from the order of failed challenges in the game.
    Extra-inning grant: a team at 0 entering an extra inning gets `extra_inning_grant`."""
    abs_df = called.filter(pl.col("season") >= rules.first_season).sort(
        "game_pk", "at_bat_number", "pitch_number"
    )
    if abs_df.is_empty():
        return called.with_columns(off_left=pl.lit(None, pl.Int64), def_left=pl.lit(None, pl.Int64))
    # Failures accrue per TEAM (challenge_team_id); a team bats one half, fields the other.
    fail = (pl.col("challenged") & ~pl.col("is_overturned").fill_null(False)).cast(pl.Int64)
    chal_team = (
        pl.when(pl.col("challenge_team_id") == pl.col("home_team_id"))
        .then(pl.lit("home"))
        .otherwise(pl.lit("away"))
    )
    rows = abs_df.with_columns(
        bat_team=pl.when(pl.col("is_top")).then(pl.lit("away")).otherwise(pl.lit("home")),
        fld_team=pl.when(pl.col("is_top")).then(pl.lit("home")).otherwise(pl.lit("away")),
        chal_team=pl.when(pl.col("challenged")).then(chal_team),
    )
    key4 = ["game_pk", "at_bat_number", "pitch_number", "inning"]
    long = pl.concat(
        [
            rows.select(
                *key4,
                team=pl.lit("home"),
                f=pl.when(pl.col("chal_team") == "home").then(fail).otherwise(0),
            ),
            rows.select(
                *key4,
                team=pl.lit("away"),
                f=pl.when(pl.col("chal_team") == "away").then(fail).otherwise(0),
            ),
        ]
    ).sort("game_pk", "team", "at_bat_number", "pitch_number")
    # walk each team's sequence once (python loop over ~5k team-games is fine)
    out_left = np.empty(long.height, dtype=np.int64)
    gp, tm, inn, f = (long[c].to_numpy() for c in ("game_pk", "team", "inning", "f"))
    start = rules.challenges_per_team
    i = 0
    while i < long.height:
        j = i
        left, cur_inn = start, inn[i]
        while j < long.height and gp[j] == gp[i] and tm[j] == tm[i]:
            if inn[j] != cur_inn:
                cur_inn = inn[j]
                if cur_inn > rules.regulation_innings and (
                    left == 0 or not rules.extra_inning_grant_only_if_empty
                ):
                    left += rules.extra_inning_grant
            out_left[j] = left
            left -= f[j] if rules.retain_on_success else 0
            j += 1
        i = j
    long = long.with_columns(left=pl.Series(out_left))
    key = ["game_pk", "at_bat_number", "pitch_number"]
    left_by_team = long.select(*key, "team", "left")
    joined = (
        rows.join(
            left_by_team.rename({"team": "bat_team", "left": "off_left"}), on=[*key, "bat_team"]
        )
        .join(left_by_team.rename({"team": "fld_team", "left": "def_left"}), on=[*key, "fld_team"])
        .select(*key, "off_left", "def_left")
    )
    return called.join(joined, on=key, how="left")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": cal.brier(y, p),
        "log_loss": cal.log_loss(y, p),
        "ece": cal.ece(y, p),
        "mean_p": float(p.mean()),
        "mean_y": float(y.mean()),
        "n": len(y),
    }


def run(s: C.Settings, d: C.DataConfig, seasons: list[int], sample: bool) -> dict[str, object]:
    rules, m = C.abs_rules(s), C.model_config(s)
    oc = m.overturn
    rng = np.random.default_rng(m.seed)
    lake = s.path(d.lake_dir)
    out, fig = s.artifacts / "overturn", s.reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ch = (
        pl.read_parquet(lake / "challenges.parquet")
        if (lake / "challenges.parquet").exists()
        else None
    )
    heights = pl.read_parquet(lake / "heights.parquet")
    umps = pl.read_parquet(lake / "umpires.parquet")
    report: dict[str, object] = {}
    cutoff = date.fromisoformat(m.splits["challenge_fit_end"])

    # ---- 1. evidence: which coordinate treatment reproduces real 2026 ABS rulings?
    # (2026 Statcast is already mid-plate -- D7 -- so "as_is" should win by a mile.)
    abs_lf = scan_pitches(d, s, [rules.first_season])
    fits = {}
    for plane in ("as_is", "mid_from_front"):
        cp = L.called_pitches(
            abs_lf.filter(pl.col("game_pk").is_in(ch["game_pk"].implode()))
            if ch is not None
            else abs_lf,
            ch,
            heights,
            umps,
            rules,
            d.statcast_midplate_from_season,
            plane,
        ).filter(pl.col("challenged"))
        sd = L.wrong_call_dist(cp["edge_dist_in"].to_numpy(), cp["is_strike_call"].to_numpy())
        early = (cp["game_date"] <= cutoff).to_numpy()
        f = L.fit_track(sd[early], cp["is_overturned"].to_numpy()[early])
        p = L.p_overturn_track(sd[~early], f)
        y = cp["is_overturned"].to_numpy()[~early].astype(float)
        fits[plane] = (f, cal.log_loss(y, p), cp, sd, early)
        log.info(
            "plane %s: mu=%.3f in sigma=%.3f in, held-out log-loss %.4f (n=%d)",
            plane,
            f.mu_in,
            f.sigma_in,
            fits[plane][1],
            len(y),
        )
    best = min(fits, key=lambda k: fits[k][1])
    if best != "as_is":
        log.warning(
            "2026 coordinates fit best as %s -- revisit statcast_midplate_from_season", best
        )
    track, _, cp, sd, early = fits[best]
    yv = cp["is_overturned"].to_numpy()
    p_hold = L.p_overturn_track(sd[~early], track)
    report["plane"] = {
        k: {"mu_in": v[0].mu_in, "sigma_in": v[0].sigma_in, "heldout_log_loss": v[1]}
        for k, v in fits.items()
    }
    report["plane_best_2026"] = best
    report["track_holdout"] = _metrics(yv[~early].astype(float), p_hold)
    plots.reliability_plot(
        {"tracked location": cal.reliability(yv[~early].astype(float), p_hold)},
        "Real 2026 challenges (held-out H2): P(overturn | Statcast location)",
        fig / "overturn_track_real.png",
    )

    # ---- 2. all called pitches, chosen plane, challenges remaining
    with stage("called_pitches") as st:
        called = L.called_pitches(
            scan_pitches(d, s, seasons), ch, heights, umps, rules, d.statcast_midplate_from_season
        )
        called = challenges_remaining(called, rules)
        st["out"] = called.height
    called = called.with_columns(y_soft=pl.Series(M.soft_label(called, track)))

    # ---- 3. perception noise per challenger side (2026 H1, team had a challenge left)
    a26 = called.filter((pl.col("season") >= rules.first_season) & (pl.col("game_date") <= cutoff))
    perceive = {}
    for side, want_k, left_col, roles in (
        ("batter", True, "off_left", ["batter"]),
        ("catcher", False, "def_left", ["catcher", "pitcher"]),
    ):
        sub = a26.filter(
            (pl.col("is_strike_call") == want_k)
            & (pl.col(left_col) > 0)
            & (pl.col("challenger_role").is_null() | pl.col("challenger_role").is_in(roles))
        )
        sdist = L.wrong_call_dist(sub["edge_dist_in"].to_numpy(), sub["is_strike_call"].to_numpy())
        perceive[side] = L.fit_perceive(sdist, sub["challenged"].to_numpy())
        log.info("perceive %s: %s", side, perceive[side])
    report["perceive"] = {k: asdict(v) for k, v in perceive.items()}
    report["track"] = asdict(track)

    # ---- 4. time-ordered windows
    if sample:
        tr = called.filter(pl.col("game_date") <= date(cutoff.year, 5, 31))
        ca = called.filter(
            (pl.col("game_date") > date(cutoff.year, 5, 31)) & (pl.col("game_date") <= cutoff)
        )
    else:
        tr = called.filter(pl.col("season").is_in(m.splits["train_seasons"]))
        ca = called.filter(pl.col("season").is_in(m.splits["valid_seasons"]))
    te = called.filter((pl.col("season") >= rules.first_season) & (pl.col("game_date") > cutoff))
    assert_no_leakage(tr, ca)
    assert_no_leakage(ca, te)
    report["windows"] = {
        k: {"n": v.height, "from": str(v["game_date"].min()), "to": str(v["game_date"].max())}
        for k, v in (("train", tr), ("calibrate", ca), ("test", te))
    }

    # ---- 5. hierarchical umpire/catcher effects on TRAIN only
    with stage("hierarchical", tr.height):
        eff = H.fit_effects(tr, oc["hierarchical"], m.seed, sample)
    eff.umpire.write_parquet(out / "umpire_effects.parquet")
    eff.catcher.write_parquet(out / "catcher_framing.parquet")
    report["sigma_u"], report["sigma_c"] = eff.sigma_u, eff.sigma_c
    tr, ca, te = (H.attach(x, eff) for x in (tr, ca, te))
    band = te.filter(pl.col("edge_dist_in").abs() <= oc["hierarchical"]["max_abs_distance_in"])
    yk = band["is_strike_call"].to_numpy().astype(float)
    pooled = _metrics(yk, H.strike_prob(band, eff, use_effects=False))
    with_eff = _metrics(yk, H.strike_prob(band, eff, use_effects=True))
    report["umpire_catcher_holdout"] = {"pooled": pooled, "with_effects": with_eff}
    log.info(
        "held-out called-strike log-loss pooled %.4f -> with ump/catcher %.4f",
        pooled["log_loss"],
        with_eff["log_loss"],
    )

    # ---- 6. LightGBM + isotonic
    with stage("lgbm", tr.height) as st:
        X_tr, X_ca, X_te = (M.features(x, perceive, rng) for x in (tr, ca, te))
        model = M.fit(
            X_tr,
            tr["y_soft"].to_numpy(),
            X_ca,
            ca["y_soft"].to_numpy(),
            oc["lgbm"],
            m.seed,
            track,
            perceive,
        )
        st["out"] = model.booster.num_trees()
    model.meta = {"features": M.FEATURES, "sample": sample}
    with (out / "model.pkl").open("wb") as fh:
        pickle.dump(model, fh)

    # ---- 7. evaluation on held-out test window
    y_te = te["y_soft"].to_numpy()
    p_model, p_raw, p_phys = (
        model.predict(X_te),
        model.predict_raw(X_te),
        M.physics_baseline(X_te, track, perceive),
    )
    near = np.abs(X_te["d_perc"].to_numpy()) <= oc["near_edge_in"]
    report["test"] = {
        name: {"all": _metrics(y_te, p), "near_edge": _metrics(y_te[near], p[near])}
        for name, p in (
            ("lgbm_isotonic", p_model),
            ("lgbm_raw", p_raw),
            ("physics_baseline", p_phys),
        )
    }
    plots.reliability_plot(
        {
            "LGBM+iso": cal.reliability(y_te[near], p_model[near]),
            "physics": cal.reliability(y_te[near], p_phys[near]),
        },
        f"Near-edge (|d|<={oc['near_edge_in']} in) calibration, test window",
        fig / "overturn_calibration_near_edge.png",
    )
    # Real challenged pitches (test window). Players challenged BECAUSE their read was
    # favourable, so draw the read from the fitted perception model conditional on having
    # challenged: s_read ~ N(s, sigma) truncated to s_read > tau. Posterior predictive check
    # of the whole perception + overturn model against observed overturn rates.
    rc = te["challenged"].to_numpy()
    if rc.any():
        tc = te.filter(pl.col("challenged"))
        yr = tc["is_overturned"].to_numpy().astype(float)
        s_true = L.wrong_call_dist(tc["edge_dist_in"].to_numpy(), tc["is_strike_call"].to_numpy())
        is_k = tc["is_strike_call"].to_numpy()
        s_read = L.read_given_challenged(s_true, is_k, perceive, rng)
        Xr = M.features(tc, None, None, d_perceived=np.where(is_k, s_read, -s_read))
        pr = model.predict(Xr)
        report["test_real_challenges"] = {
            "lgbm_isotonic": _metrics(yr, pr),
            "physics_baseline": _metrics(yr, M.physics_baseline(Xr, track, perceive)),
            "by_side": {
                side: _metrics(yr[m_], pr[m_])
                for side, m_ in (("batter", is_k), ("defense", ~is_k))
            },
        }
        plots.reliability_plot(
            {"model (read | challenged)": cal.reliability(yr, pr, 8)},
            "Real 2026 challenges, held-out: predicted vs actual overturn",
            fig / "overturn_real_challenges.png",
        )
    imp = dict(
        zip(M.FEATURES, model.booster.feature_importance("gain").round(1).tolist(), strict=True)
    )
    report["feature_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    # Decision-time P(overturn) for the DP (calibration window) and the backtest (test window).
    keep = [
        "game_pk",
        "game_date",
        "season",
        "at_bat_number",
        "pitch_number",
        "inning",
        "is_top",
        "outs_when_up",
        "base",
        "score_diff_home",
        "balls",
        "strikes",
        "is_strike_call",
        "edge_dist_in",
        "edge",
        "y_soft",
        "challenged",
        "is_overturned",
        "challenger_role",
        "off_left",
        "def_left",
        "umpire_id",
        "fielder_2",
        "home_team",
        "away_team",
    ]
    p_ca = model.predict(X_ca)
    ca.select(keep).with_columns(p_overturn=pl.Series(p_ca)).write_parquet(
        out / "called_calibrate.parquet"
    )
    te.select(keep).with_columns(p_overturn=pl.Series(p_model)).write_parquet(
        out / "called_test.parquet"
    )
    return report
