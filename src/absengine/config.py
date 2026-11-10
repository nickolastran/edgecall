"""Typed configuration loaded from configs/*.yaml. No constants live in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ZoneRules(_Frozen):
    plate_width_in: float
    plane_depth_in: float
    top_frac_height: float
    bot_frac_height: float
    ball_radius_in: float


class AbsRules(_Frozen):
    first_season: int
    challenges_per_team: int
    retain_on_success: bool
    extra_inning_grant: int
    extra_inning_grant_only_if_empty: bool
    regulation_innings: int
    eligible_challengers: list[str]
    zone: ZoneRules


class DataConfig(_Frozen):
    seasons: list[int]
    sample_seasons: list[int]
    sample_game_frac: float
    platform_root: Path
    platform_cache_globs: list[str]
    own_cache_glob: str
    season_window: tuple[str, str]
    min_windows_per_season: int
    statcast_midplate_from_season: int
    lake_dir: Path
    feed_cache_dir: Path
    people_cache: Path
    quarantine_dir: Path
    mlb_api_base: str
    http_user_agent: str
    http_min_interval_s: float
    http_concurrency: int


class ModelConfig(_Frozen):
    model_config = ConfigDict(frozen=True, extra="allow")
    seed: int
    splits: dict[str, Any]
    runvalue: dict[str, Any]
    win_expectancy: dict[str, Any]
    overturn: dict[str, Any]
    policy: dict[str, Any]
    backtest: dict[str, Any]


class Settings(BaseSettings):
    """Process-level settings (env vars: ABS_CONFIG_DIR, ABS_ROOT, ABS_SAMPLE)."""

    model_config = SettingsConfigDict(env_prefix="ABS_")
    root: Path = Path.cwd()
    config_dir: Path | None = None
    sample: bool = False

    @property
    def configs(self) -> Path:
        return self.config_dir or self.root / "configs"

    def path(self, p: Path | str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / ("sample" if self.sample else "full")

    @property
    def reports(self) -> Path:
        return self.root / "reports" / ("sample" if self.sample else "full")


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        out: dict[str, Any] = yaml.safe_load(f)
    return out


@lru_cache
def settings() -> Settings:
    return Settings()


def abs_rules(s: Settings | None = None) -> AbsRules:
    return AbsRules(**_load((s or settings()).configs / "abs_rules.yaml"))


def data_config(s: Settings | None = None) -> DataConfig:
    return DataConfig(**_load((s or settings()).configs / "data.yaml"))


def model_config(s: Settings | None = None) -> ModelConfig:
    return ModelConfig(**_load((s or settings()).configs / "model.yaml"))
