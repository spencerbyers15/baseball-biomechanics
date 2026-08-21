"""Configuration loaded from config.toml. Every section has defaults so a
partial file (or none at all) still yields a runnable config."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelsConfig:
    foreground: str = "claude-opus-5"
    idle: str = "claude-sonnet-5"
    compressor: str = "claude-sonnet-5"
    consolidator: str = "claude-sonnet-5"
    summarizer: str = "claude-haiku-4-5"
    vision: str = "claude-opus-5"
    embedding: str = ""


@dataclass
class IdleConfig:
    tick_seconds: float = 45.0
    tick_jitter_seconds: float = 15.0
    backoff_after_minutes: float = 10.0
    max_tick_seconds: float = 600.0
    context_thoughts: int = 8
    transcript_summary_every_turns: int = 10
    transcript_verbatim_turns: int = 3
    max_sim_ticks: int = 30


@dataclass
class MemoryConfig:  # Phase 2
    consolidate_every_ticks: int = 20
    evoke_threshold: float = 0.78
    evoke_max: int = 3
    salience_floor: float = 0.15
    archive_after_days: int = 30


@dataclass
class PerceptionConfig:  # Phase 3
    enabled: bool = False
    webcam_interval_minutes: int = 5
    feeds: list[str] = field(default_factory=list)


@dataclass
class BudgetConfig:
    daily_tokens: int = 2_000_000


@dataclass
class ForegroundConfig:
    max_transcript_turns: int = 40


@dataclass
class StorageConfig:
    db_path: str = "data/idle_mind.db"


@dataclass
class LoggingConfig:
    dir: str = "logs"


@dataclass
class Config:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    foreground: ForegroundConfig = field(default_factory=ForegroundConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None or not Path(path).exists():
            return cls()
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        sections = {
            "models": ModelsConfig,
            "idle": IdleConfig,
            "memory": MemoryConfig,
            "perception": PerceptionConfig,
            "budget": BudgetConfig,
            "foreground": ForegroundConfig,
            "storage": StorageConfig,
            "logging": LoggingConfig,
        }
        kwargs = {}
        for name, klass in sections.items():
            data = raw.get(name, {})
            known = {k: v for k, v in data.items() if k in klass.__dataclass_fields__}
            kwargs[name] = klass(**known)
        return cls(**kwargs)
