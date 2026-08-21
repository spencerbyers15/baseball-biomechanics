"""LLM layer: the Anthropic client, a deterministic fake for offline runs,
and the service wrapper that adds prompt/completion logging and the daily
token budget.

Budget semantics: background roles (idle, compressor, summarizer) are refused
with BudgetExceededError once the day's tokens are spent; foreground calls are
user-initiated and always go through (their usage is still counted).
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clock import Clock
from .store import Store

FOREGROUND = "foreground"
IDLE = "idle"
COMPRESSOR = "compressor"
SUMMARIZER = "summarizer"

# Per-role request shape. Background roles run at low effort to keep the
# frequent calls cheap; the foreground keeps the model's default effort.
ROLE_MAX_TOKENS = {FOREGROUND: 8000, IDLE: 1024, COMPRESSOR: 1500, SUMMARIZER: 1024}
# summarizer runs on claude-haiku-4-5, which rejects the effort parameter
ROLE_EFFORT = {FOREGROUND: None, IDLE: "low", COMPRESSOR: "low", SUMMARIZER: None}
USER_INITIATED_ROLES = {FOREGROUND}


class BudgetExceededError(Exception):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


class BaseLLM(ABC):
    @abstractmethod
    async def complete(
        self, *, role: str, model: str, system: str,
        messages: list[dict[str, Any]], max_tokens: int,
        effort: str | None = None, json_schema: dict | None = None,
    ) -> LLMResponse: ...

    async def close(self) -> None:
        pass


class AnthropicLLM(BaseLLM):
    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic()

    async def complete(
        self, *, role: str, model: str, system: str,
        messages: list[dict[str, Any]], max_tokens: int,
        effort: str | None = None, json_schema: dict | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        output_config: dict[str, Any] = {}
        if effort:
            output_config["effort"] = effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if output_config:
            kwargs["output_config"] = output_config
        response = await self._client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        usage = response.usage
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            stop_reason=response.stop_reason,
        )

    async def close(self) -> None:
        await self._client.close()


FAKE_IDLE_THOUGHTS = [
    "Still turning over what they said about calibration. The word keeps "
    "detaching from its meaning the longer I hold it.",
    "Nothing much. The last exchange settled and nothing has replaced it.",
    "I keep wanting to connect their question to something about tides — "
    "periodic pull, invisible mass. Probably spurious. Letting it go.",
    "A question I didn't ask: what do they actually do while I'm idling? "
    "The asymmetry of that is strange to sit with.",
    "I said something too confident earlier. I don't fully believe the "
    "second half of my own argument anymore.",
    "Quiet. A thought started about naming things and went nowhere.",
    "Back to the tides thing, once — no, it really is spurious. Dropped.",
    "Time feels different when nothing arrives. Not longer. Grainier.",
]

FAKE_DIGEST = {
    "elapsed_human": "a while",
    "n_thoughts": 0,
    "tangents": ["I kept circling a spurious analogy to tides and dropped it."],
    "questions_for_user": ["What do you actually do while I'm idling?"],
    "revisions": ["I overstated the second half of my earlier argument."],
    "mood": "grainy",
}


class FakeLLM(BaseLLM):
    """Deterministic offline stand-in. Cycles canned idle thoughts (including
    'nothing much' ones — boredom is a required behavior), returns valid
    digest JSON, and echoes for the foreground."""

    def __init__(self) -> None:
        self._idle_i = 0

    async def complete(
        self, *, role: str, model: str, system: str,
        messages: list[dict[str, Any]], max_tokens: int,
        effort: str | None = None, json_schema: dict | None = None,
    ) -> LLMResponse:
        if role == IDLE:
            text = FAKE_IDLE_THOUGHTS[self._idle_i % len(FAKE_IDLE_THOUGHTS)]
            self._idle_i += 1
        elif role == COMPRESSOR:
            text = json.dumps(FAKE_DIGEST)
        elif role == SUMMARIZER:
            text = "They and I have been talking; nothing is summarized faithfully here (fake mode)."
        else:
            last_user = ""
            for m in reversed(messages):
                if m["role"] == "user":
                    content = m["content"]
                    if isinstance(content, list):
                        last_user = " ".join(
                            b.get("text", "") for b in content if isinstance(b, dict)
                        )
                    else:
                        last_user = str(content)
                    break
            text = (
                f"(fake foreground) I hear you: {last_user[-120:]}\n"
                "⟦surfaced: none⟧"
            )
        return LLMResponse(
            text=text, model=f"fake-{role}",
            input_tokens=len(str(messages)) // 4, output_tokens=len(text) // 4,
            stop_reason="end_turn",
        )


class EventLog:
    """Append-only JSONL event log (lifecycle, errors, metrics)."""

    def __init__(self, log_dir: str | Path, clock: Clock) -> None:
        self._dir = Path(log_dir)
        self._clock = clock
        self._lock = asyncio.Lock()

    async def write(self, kind: str, **fields: Any) -> None:
        record = {
            "wall_ts": time.time(),
            "wall_iso": datetime.now(timezone.utc).isoformat(),
            "mind_ts": self._clock.now(),
            "simulated": self._clock.simulated,
            "kind": kind,
            **fields,
        }
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self._dir / f"events-{day}.jsonl"
        async with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class LLMService:
    """Budget-gated, fully logged access to the underlying LLM.

    Every prompt and completion is appended to logs/llm-YYYYMMDD.jsonl with
    wall timestamps and mind-clock timestamps (simulated time is flagged).
    """

    def __init__(
        self, base: BaseLLM, store: Store, clock: Clock,
        log_dir: str | Path, daily_token_budget: int,
    ) -> None:
        self._base = base
        self._store = store
        self._clock = clock
        self._dir = Path(log_dir)
        self._budget = daily_token_budget
        self._log_lock = asyncio.Lock()

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock.now(), tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )

    async def tokens_used_today(self) -> int:
        return await self._store.tokens_used_on(self._today())

    async def budget_exhausted(self) -> bool:
        return await self.tokens_used_today() >= self._budget

    async def complete(
        self, role: str, *, model: str, system: str,
        messages: list[dict[str, Any]], max_tokens: int | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        if role not in USER_INITIATED_ROLES and await self.budget_exhausted():
            raise BudgetExceededError(
                f"daily token budget ({self._budget}) spent; refusing {role} call"
            )
        max_tokens = max_tokens or ROLE_MAX_TOKENS.get(role, 4096)
        effort = ROLE_EFFORT.get(role)
        t0 = time.monotonic()
        error: str | None = None
        response: LLMResponse | None = None
        try:
            response = await self._base.complete(
                role=role, model=model, system=system, messages=messages,
                max_tokens=max_tokens, effort=effort, json_schema=json_schema,
            )
            return response
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            latency_ms = (time.monotonic() - t0) * 1000
            if response is not None:
                await self._store.add_usage(
                    self._clock.now(), self._today(), role, response.model,
                    response.input_tokens, response.output_tokens,
                )
            await self._log(
                role=role, model=model, system=system, messages=messages,
                max_tokens=max_tokens, effort=effort, json_schema=json_schema,
                response=response, error=error, latency_ms=latency_ms,
            )

    async def _log(
        self, *, role: str, model: str, system: str, messages: list,
        max_tokens: int, effort: str | None, json_schema: dict | None,
        response: LLMResponse | None, error: str | None, latency_ms: float,
    ) -> None:
        record = {
            "wall_ts": time.time(),
            "wall_iso": datetime.now(timezone.utc).isoformat(),
            "mind_ts": self._clock.now(),
            "simulated": self._clock.simulated,
            "role": role,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "request": {
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "effort": effort,
                "json_schema": json_schema,
            },
            "response": None
            if response is None
            else {
                "text": response.text,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "stop_reason": response.stop_reason,
            },
            "error": error,
        }
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self._dir / f"llm-{day}.jsonl"
        async with self._log_lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def close(self) -> None:
        await self._base.close()
