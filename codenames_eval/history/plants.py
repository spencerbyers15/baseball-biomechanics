"""Schemas for dyad histories and planted episodes.

Ground truth about what was planted (Plant records) is stored separately from
the history itself — agents only ever see the history/memory payload.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    speaker: Literal["A", "B"]
    text: str


class Session(BaseModel):
    index: int
    turns: list[Turn]


class DyadHistory(BaseModel):
    dyad_id: str
    sessions: list[Session]


class Plant(BaseModel):
    """A designed rare co-occurrence injected into one session."""

    episode_id: str
    session_index: int
    bound_concepts: list[str] = Field(min_length=2, max_length=4)
    binding_word: str
    partner_engaged: bool
