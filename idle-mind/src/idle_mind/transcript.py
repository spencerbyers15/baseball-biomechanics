"""Rolling transcript summary for the idle context: regenerated every N
turns (default 10) by a cheap model, with the last few turns always passed
verbatim. The foreground never uses this — it gets real turns."""

from __future__ import annotations

from . import prompts
from .clock import Clock
from .config import Config
from .llm import SUMMARIZER, BudgetExceededError, EventLog, LLMService
from .store import Store, row_to_dict


class TranscriptContext:
    def __init__(
        self, store: Store, llm: LLMService, clock: Clock, cfg: Config, events: EventLog
    ) -> None:
        self._store = store
        self._llm = llm
        self._clock = clock
        self._cfg = cfg
        self._events = events

    async def context(self) -> tuple[str | None, list[dict]]:
        """(summary or None, last-N verbatim turns) for the idle prompt."""
        verbatim_n = self._cfg.idle.transcript_verbatim_turns
        turns = await self._store.recent_turns(verbatim_n)
        summary_row = await self._store.latest_summary()
        summary = summary_row["content"] if summary_row else None
        return summary, [row_to_dict(t) for t in turns]

    async def maybe_regenerate(self) -> None:
        """Regenerate the summary if enough new turns accumulated since the
        last one. Runs after replies; failures are logged, never raised."""
        count = await self._store.turn_count()
        summary_row = await self._store.latest_summary()
        covered = summary_row["upto_turn_id"] if summary_row else 0
        turns = await self._store.recent_turns(10_000)
        uncovered = [t for t in turns if t["id"] > covered]
        if len(uncovered) < self._cfg.idle.transcript_summary_every_turns:
            return
        lines = [f"{t['role']}: {t['content']}" for t in turns]
        prompt = "The conversation:\n\n" + "\n".join(lines)
        try:
            response = await self._llm.complete(
                SUMMARIZER,
                model=self._cfg.models.summarizer,
                system=prompts.SUMMARIZER_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except BudgetExceededError:
            await self._events.write("summary_skipped_budget", turn_count=count)
            return
        except Exception as e:
            await self._events.write("summary_error", error=f"{type(e).__name__}: {e}")
            return
        await self._store.add_summary(
            self._clock.now(), upto_turn_id=turns[-1]["id"], content=response.text
        )
        await self._events.write("summary_regenerated", upto_turn_id=turns[-1]["id"])
