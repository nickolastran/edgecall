"""FastAPI: /health, /evaluate, /policy/thresholds, /umpires/{id}.

Artifacts come from ABS_SAMPLE / ABS_ROOT (pydantic-settings), default full mode.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from absengine import config as C
from absengine.serve import Engine, Situation, umpire_table

app = FastAPI(title="absengine", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"]
)


@lru_cache
def engine() -> Engine:
    return Engine.load(C.settings())


@lru_cache
def thresholds() -> pl.DataFrame:
    return pl.read_parquet(C.settings().artifacts / "policy" / "thresholds.parquet")


@app.get("/health")
def health() -> dict[str, Any]:
    s = C.settings()
    return {"ok": (s.artifacts / "policy" / "dp.pkl").exists(), "sample": s.sample}


@app.post("/evaluate")
def evaluate(x: Situation) -> dict[str, Any]:
    return engine().evaluate(x)


@app.get("/policy/thresholds")
def policy_thresholds(
    side: Literal["offense", "defense"] | None = None,
    challenges_left: int | None = None,
    inning: int | None = None,
    is_top: bool | None = None,
    outs: int | None = None,
    base: int | None = None,
    balls: int | None = None,
    strikes: int | None = None,
    score_diff_home: int | None = None,
    limit: int = Query(500, le=5000),
) -> list[dict[str, Any]]:
    t = thresholds()
    for col, v in (
        ("side", side),
        ("challenges_left", challenges_left),
        ("inning", inning),
        ("is_top", is_top),
        ("outs_when_up", outs),
        ("base", base),
        ("balls", balls),
        ("strikes", strikes),
        ("score_diff_home", score_diff_home),
    ):
        if v is not None:
            t = t.filter(pl.col(col) == v)
    return t.head(limit).to_dicts()


@app.get("/umpires/{umpire_id}")
def umpire(umpire_id: int) -> dict[str, Any]:
    for u in umpire_table(engine()):
        if u["id"] == umpire_id:
            return u
    raise HTTPException(404, "umpire not found")
