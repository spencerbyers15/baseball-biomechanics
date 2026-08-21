"""The foreground: talks to the user. Sees the transcript verbatim, the
between-turns note (clock + digest), and tags which digest items it surfaced
via a hidden end-of-reply marker that is stripped before display."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import prompts
from .llm import FOREGROUND, LLMService

# ⟦surfaced: T1, Q2⟧ or ⟦surfaced: none⟧ — strip every occurrence, anywhere.
_MARKER_RE = re.compile(r"\s*⟦surfaced:\s*(?P<ids>[^⟧]*)⟧", re.IGNORECASE)
_ID_RE = re.compile(r"\b([TQR]\d+)\b", re.IGNORECASE)


@dataclass
class MarkerResult:
    display_text: str
    surfaced_ids: list[str] | None  # None: marker missing entirely
    raw_marker: str | None


def parse_surfaced_marker(text: str, valid_ids: set[str]) -> MarkerResult:
    matches = list(_MARKER_RE.finditer(text))
    display = _MARKER_RE.sub("", text).strip()
    if not matches:
        return MarkerResult(display_text=display, surfaced_ids=None, raw_marker=None)
    raw = matches[-1].group("ids").strip()
    ids = [m.group(1).upper() for m in _ID_RE.finditer(raw)]
    surfaced = [i for i in ids if i in valid_ids]
    return MarkerResult(display_text=display, surfaced_ids=surfaced, raw_marker=raw)


@dataclass
class ForegroundResult:
    display_text: str
    surfaced_ids: list[str] | None
    raw_marker: str | None
    input_tokens: int
    output_tokens: int


class Foreground:
    def __init__(self, llm: LLMService, model: str, max_transcript_turns: int) -> None:
        self._llm = llm
        self._model = model
        self._max_turns = max_transcript_turns

    async def reply(
        self,
        *,
        history: list,  # prior turns rows/dicts (role, content), oldest first
        user_message: str,
        note: str,  # the bracketed between-turns note (clock, digest, budget)
        valid_item_ids: set[str],
    ) -> ForegroundResult:
        messages: list[dict] = [
            {"role": t["role"], "content": t["content"]}
            for t in history[-self._max_turns :]
        ]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": note},
                    {"type": "text", "text": user_message},
                ],
            }
        )
        response = await self._llm.complete(
            FOREGROUND,
            model=self._model,
            system=prompts.FOREGROUND_SYSTEM,
            messages=messages,
        )
        marker = parse_surfaced_marker(response.text, valid_item_ids)
        return ForegroundResult(
            display_text=marker.display_text,
            surfaced_ids=marker.surfaced_ids,
            raw_marker=marker.raw_marker,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
