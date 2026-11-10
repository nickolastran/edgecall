"""Loaded engine for serving: one-situation evaluation (FastAPI) and the static web export."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from absengine import config as C
from absengine.overturn.model import EDGE_CODES, FEATURES, PITCH_TYPES, OverturnModel
from absengine.policy import dp
from absengine.policy.build import load_value_tables
from absengine.value import ev

Edge = Literal["top", "bottom", "inside", "outside"]


class Situation(BaseModel):
    inning: int = Field(1, ge=1, le=30)
    is_top: bool = True
    outs: int = Field(0, ge=0, le=2)
    base: int = Field(0, ge=0, le=7, description="bit-packed runners: 1B=1, 2B=2, 3B=4")
    balls: int = Field(0, ge=0, le=3)
    strikes: int = Field(0, ge=0, le=2)
    score_diff_home: int = 0
    call: Literal["strike", "ball"] = "strike"
    d_perc: float = Field(
        0.5,
        ge=-12,
        le=12,
        description="perceived signed distance to the zone edge, inches; > 0 = outside",
    )
    edge: Edge = "outside"
    challenges_left: int = Field(2, ge=0, le=4)
    umpire_id: int | None = None
    catcher_id: int | None = None
    stand: Literal["R", "L"] = "R"
    p_throws: Literal["R", "L"] = "R"
    pitch_type: str = "FF"
    release_speed: float = 93.0
    pfx_x: float = -0.5
    pfx_z: float = 1.2
    zone_height_in: float = 19.0


@dataclass
class Engine:
    s: C.Settings
    vt: ev.ValueTables
    res: dp.DPResult
    model: OverturnModel
    ump: pl.DataFrame
    catcher: pl.DataFrame
    report: dict[str, Any]

    @classmethod
    def load(cls, s: C.Settings) -> Engine:
        a = s.artifacts
        with (a / "policy" / "dp.pkl").open("rb") as f:
            res = pickle.load(f)
        with (a / "overturn" / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        return cls(
            s,
            load_value_tables(s),
            res,
            model,
            pl.read_parquet(a / "overturn" / "umpire_effects.parquet"),
            pl.read_parquet(a / "overturn" / "catcher_framing.parquet"),
            json.loads((a / "overturn" / "report.json").read_text()),
        )

    def _prior_ump_sd(self, edge: str) -> float:
        su: dict[str, float] = self.report["sigma_u"]
        return float(su.get(edge, np.mean(list(su.values()))))

    def features(self, x: Situation) -> pl.DataFrame:
        u = self.ump.filter(
            (pl.col("umpire_id") == (x.umpire_id or -1)) & (pl.col("edge") == x.edge)
        )
        c = self.catcher.filter(pl.col("catcher_id") == (x.catcher_id or -1))
        is_k = x.call == "strike"
        row = {
            "s_perc": x.d_perc if is_k else -x.d_perc,
            "d_perc": x.d_perc,
            "is_strike_call": int(is_k),
            "edge_code": EDGE_CODES.index(x.edge),
            "balls": x.balls,
            "strikes": x.strikes,
            "stand_r": int(x.stand == "R"),
            "p_throws_r": int(x.p_throws == "R"),
            "zone_height_in": x.zone_height_in,
            "pitch_code": PITCH_TYPES.index(x.pitch_type)
            if x.pitch_type in PITCH_TYPES
            else len(PITCH_TYPES),
            "release_speed": x.release_speed,
            "pfx_x": x.pfx_x,
            "pfx_z": x.pfx_z,
            "ump_mean": u["ump_mean"].item() if u.height else 0.0,
            "ump_sd": u["ump_sd"].item() if u.height else self._prior_ump_sd(x.edge),
            "catch_mean": c["catch_mean"].item() if c.height else 0.0,
            "catch_sd": c["catch_sd"].item() if c.height else float(self.report["sigma_c"]),
        }
        return pl.DataFrame([row]).select(FEATURES)

    def evaluate(self, x: Situation) -> dict[str, Any]:
        offense = x.call == "strike"  # batters challenge strikes, the defense challenges balls
        team_is_home = (not x.is_top) if offense else x.is_top
        diff_team = x.score_diff_home if team_is_home else -x.score_diff_home
        g_runs, g_wins = ev.flip_gain(
            self.vt, x.inning, x.is_top, x.outs, x.base, x.score_diff_home, x.balls, x.strikes
        )
        ov = self.res.option_value(
            "home" if team_is_home else "away",
            x.inning,
            x.is_top,
            x.outs,
            diff_team,
            x.challenges_left,
        )
        p = float(self.model.predict(self.features(x))[0])
        if x.challenges_left <= 0:
            thr, ev_ = 1.0, float("-inf")
        else:
            thr = float(ev.threshold(np.array([g_wins]), np.array([ov]))[0])
            ev_ = float(ev.ev_challenge(np.array([p]), np.array([g_wins]), np.array([ov]))[0])
        return {
            "side": "offense" if offense else "defense",
            "p_overturn": p,
            "gain_wins": g_wins,
            "gain_runs": g_runs,
            "option_value": ov if x.challenges_left > 0 else None,
            "p_threshold": thr,
            "ev_wins": ev_ if x.challenges_left > 0 else None,
            "challenge": x.challenges_left > 0 and p > thr,
        }


def _umpire_names(s: C.Settings) -> dict[int, str]:
    p = s.path(C.data_config(s).lake_dir) / "umpires.parquet"
    if not p.exists():
        return {}
    u = pl.read_parquet(p).drop_nulls("umpire_id").unique("umpire_id")
    return dict(zip(u["umpire_id"].to_list(), u["umpire_name"].to_list(), strict=True))


def umpire_table(e: Engine) -> list[dict[str, Any]]:
    names = _umpire_names(e.s)
    out = []
    for (uid,), g in e.ump.group_by("umpire_id"):
        out.append(
            {
                "id": uid,
                "name": names.get(int(uid), str(uid)),
                "n": int(g["n"].sum()),
                "edges": {
                    r["edge"]: {"mean": round(r["ump_mean"], 4), "sd": round(r["ump_sd"], 4)}
                    for r in g.iter_rows(named=True)
                },
            }
        )
    return sorted(out, key=lambda r: r["name"])


GRID_D = np.round(np.arange(-6, 6.001, 0.25), 2)
GRID_UMP = [-0.4, -0.2, 0.0, 0.2, 0.4]


def p_grid(e: Engine) -> dict[str, Any]:
    """Model P(overturn) on [call][edge][count][d_perc][ump_mean]; other features at
    typical values. The web app interpolates in d and ump_mean."""
    rows = []
    for k in (0, 1):
        for ei, edge in enumerate(EDGE_CODES):
            for b in range(4):
                for st in range(3):
                    for d in GRID_D:
                        for um in GRID_UMP:
                            rows.append(
                                {
                                    "s_perc": d if k else -d,
                                    "d_perc": d,
                                    "is_strike_call": k,
                                    "edge_code": ei,
                                    "balls": b,
                                    "strikes": st,
                                    "stand_r": 1,
                                    "p_throws_r": 1,
                                    "zone_height_in": 19.0,
                                    "pitch_code": 0,
                                    "release_speed": 93.0,
                                    "pfx_x": -0.5,
                                    "pfx_z": 1.2,
                                    "ump_mean": um,
                                    "ump_sd": e._prior_ump_sd(edge),
                                    "catch_mean": 0.0,
                                    "catch_sd": float(e.report["sigma_c"]),
                                }
                            )
    p = e.model.predict(pl.DataFrame(rows).select(FEATURES))
    return {
        "calls": ["ball", "strike"],
        "edges": EDGE_CODES,
        "d": GRID_D.tolist(),
        "ump": GRID_UMP,
        "p": np.round(p, 4).tolist(),
    }


def golden(e: Engine, n: int = 40, seed: int = 0) -> list[dict[str, Any]]:
    """Python answers for random situations on the grid (default pitch/umpire features),
    so the web port can be checked for parity (web/src/engine.test.ts)."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        x = Situation(
            inning=int(rng.integers(1, 13)),
            is_top=bool(rng.integers(0, 2)),
            outs=int(rng.integers(0, 3)),
            base=int(rng.integers(0, 8)),
            balls=int(rng.integers(0, 4)),
            strikes=int(rng.integers(0, 3)),
            score_diff_home=int(rng.integers(-5, 6)),
            call="strike" if rng.integers(0, 2) else "ball",
            d_perc=float(rng.choice(GRID_D)),
            edge=EDGE_CODES[int(rng.integers(0, 4))],
            challenges_left=int(rng.integers(0, 4)),
        )
        cases.append({"in": x.model_dump(), "out": e.evaluate(x)})
    return cases


def export_web(e: Engine, out: Path) -> Path:
    s = e.s
    rules = C.abs_rules(s)
    we = e.vt.we
    bt = s.artifacts / "backtest"
    payload = {
        "meta": {"sample": s.sample, "generated": datetime.now(UTC).isoformat(timespec="seconds")},
        "rules": rules.model_dump(exclude={"zone", "eligible_challengers", "first_season"}),
        "count_re": np.round(e.vt.re, 4).tolist(),
        "we": {
            "p": np.round(we.p, 4).ravel().tolist(),
            "shape": list(we.p.shape),
            "max_inning": we.max_inning,
            "max_diff": we.max_diff,
            "regulation": we.regulation,
            "ghost_runner": we.ghost_runner,
        },
        "dp": {
            "V": {k: np.round(v, 5).tolist() for k, v in e.res.V.items()},
            "cont": {k: np.round(v, 5).tolist() for k, v in e.res.cont.items()},
            "edges": e.res.edges,
            "max_c": e.res.max_c,
        },
        "grid": p_grid(e),
        "golden": golden(e),
        "umpires": umpire_table(e),
        "fits": {"track": e.report["track"], "perceive": e.report["perceive"]},
        "overturn_eval": e.report.get("test_real_challenges"),
        "windows": e.report.get("windows"),
        "backtest": json.loads((bt / "summary.json").read_text())
        if (bt / "summary.json").exists()
        else None,
        "players": _players(s),
        "teams": pl.read_parquet(bt / "teams.parquet").to_dicts()
        if (bt / "teams.parquet").exists()
        else [],
    }
    out.mkdir(parents=True, exist_ok=True)
    f = out / "engine.json"
    f.write_text(json.dumps(payload, separators=(",", ":"), default=_json_default))
    return f


def _players(s: C.Settings) -> list[dict[str, Any]]:
    p = s.artifacts / "players" / "players.parquet"
    if not p.exists():
        return []
    t = pl.read_parquet(p).select(
        pl.col("challenger_id").alias("id"),
        "name",
        "role",
        "team",
        "challenges",
        "overturned",
        *(pl.col(c).round(4) for c in ("wins_gained", "wins_spent", "net_wins", "avg_stakes")),
        pl.col("runs_gained").round(2),
    )
    return t.to_dicts()


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)  # type: ignore[call-overload]
    raise TypeError(type(o))
