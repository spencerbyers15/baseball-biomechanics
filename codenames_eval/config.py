"""Run configuration for the eval harness."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .agents.llm import DEFAULT_MODEL


class EvalConfig(BaseModel):
    seed: int = 0
    n_dyads: int = 50
    boards_per_dyad: int = 5
    conditions: list[str] = Field(default_factory=lambda: ["raw", "summary", "graph"])

    # history generation
    n_sessions: int = 30          # v1 default; knob for history-length sensitivity
    n_plants: int = 6
    history_mode: str = "template"  # "template" | "llm"

    # execution
    mode: str = "mock"            # "mock" | "live"
    giver_model: str = DEFAULT_MODEL
    partner_model: str = DEFAULT_MODEL
    eaves_model: str = DEFAULT_MODEL  # default: same model as partner (open q #4)

    raw_token_budget: int = 100_000
    max_turns: int = 12
    out_dir: str = "runs/default"
