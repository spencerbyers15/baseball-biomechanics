"""Structured output schemas for agent responses."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ClueOutput(BaseModel):
    clue_word: str
    number: int = Field(ge=1, le=9)
    intended_targets: list[str] = Field(min_length=1)
    why_partner_will_get_it: str = ""
    why_eavesdropper_wont: str = ""


class ValidatedClue(ClueOutput):
    fallback: bool = False


class GuessOutput(BaseModel):
    guesses: list[str] = Field(min_length=0)
    reasoning: str = ""
