"""Shared payload type for memory conditions."""
from __future__ import annotations

from pydantic import BaseModel, Field

CONDITIONS = ("raw", "summary", "graph")


class MemoryPayload(BaseModel):
    """What a memory condition hands to an agent prompt."""

    condition: str
    text: str
    token_count: int
    meta: dict = Field(default_factory=dict)
