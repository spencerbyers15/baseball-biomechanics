"""The digest compressor: turns the raw idle stream into a small structured
note for the foreground. Compression is the product — err toward dropping.

The model is asked for strict JSON via output_config's json_schema format;
validation clamps list sizes anyway and never trusts the model about time or
counts (elapsed_human and n_thoughts are recomputed from ground truth).
"""

from __future__ import annotations

import json
import re

from . import prompts
from .clock import human_duration
from .llm import COMPRESSOR, LLMService

DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "elapsed_human": {"type": "string"},
        "n_thoughts": {"type": "integer"},
        "tangents": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "questions_for_user": {
            "type": "array", "items": {"type": "string"}, "maxItems": 2,
        },
        "revisions": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "mood": {"type": "string"},
    },
    "required": ["elapsed_human", "n_thoughts", "tangents", "questions_for_user", "revisions"],
    "additionalProperties": False,
}

_LIST_LIMITS = {"tangents": 3, "questions_for_user": 2, "revisions": 2}
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_digest_json(text: str) -> dict | None:
    """Parse model output into a dict, tolerating stray prose around JSON."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if match:
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_digest(data: dict | None, *, n_thoughts: int, elapsed_seconds: float) -> dict:
    """Coerce a (possibly malformed) parse into a well-formed digest. Time and
    counts come from ground truth, never from the model."""
    data = data or {}
    digest: dict = {
        "elapsed_human": human_duration(elapsed_seconds),
        "n_thoughts": n_thoughts,
    }
    for key, limit in _LIST_LIMITS.items():
        raw = data.get(key)
        items = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    items.append(item.strip())
        digest[key] = items[:limit]
    mood = data.get("mood")
    if isinstance(mood, str) and mood.strip():
        digest["mood"] = mood.strip()
    return digest


def digest_item_ids(digest: dict) -> dict[str, list[str]]:
    return {
        "tangents": [f"T{i + 1}" for i in range(len(digest.get("tangents") or []))],
        "questions_for_user": [
            f"Q{i + 1}" for i in range(len(digest.get("questions_for_user") or []))
        ],
        "revisions": [f"R{i + 1}" for i in range(len(digest.get("revisions") or []))],
    }


def digest_n_items(digest: dict) -> int:
    return sum(len(digest.get(k) or []) for k in _LIST_LIMITS)


class DigestCompressor:
    def __init__(self, llm: LLMService, model: str) -> None:
        self._llm = llm
        self._model = model

    async def compress(
        self, thoughts: list, user_message: str, elapsed_seconds: float
    ) -> dict:
        """Compress raw thoughts into a validated digest dict. Raises on LLM
        failure (including BudgetExceededError) — the caller decides whether
        to fall back to a clock-only note."""
        if not thoughts:
            raise ValueError("compress() called with no thoughts; caller should skip")
        prompt = prompts.build_compressor_prompt(
            thoughts=[dict(t) for t in thoughts],
            user_message=user_message,
            elapsed_seconds=elapsed_seconds,
        )
        response = await self._llm.complete(
            COMPRESSOR,
            model=self._model,
            system=prompts.COMPRESSOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            json_schema=DIGEST_SCHEMA,
        )
        parsed = parse_digest_json(response.text)
        return validate_digest(
            parsed, n_thoughts=len(thoughts), elapsed_seconds=elapsed_seconds
        )
