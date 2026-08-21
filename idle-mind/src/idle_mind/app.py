"""The Mind: wires the store, clock, idle loop, compressor and foreground
together, and owns the turn lifecycle:

    reply sent -> idle loop starts -> user message arrives -> idle loop stops
    -> stream since last digest is compressed -> digest + clock injected into
    the foreground context -> reply -> idle loop starts again

The idle loop also starts at program launch, before the first message — the
mind idles while it waits (see DECISIONS.md).
"""

from __future__ import annotations

import asyncio
import json
import time

from . import prompts
from .clock import Clock
from .compressor import DigestCompressor, digest_item_ids, digest_n_items
from .config import Config
from .foreground import Foreground
from .idle_loop import IdleLoop, ThoughtCallback
from .llm import (
    AnthropicLLM,
    BaseLLM,
    BudgetExceededError,
    EventLog,
    FakeLLM,
    LLMService,
)
from .store import Store, row_to_dict
from .transcript import TranscriptContext


class Mind:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        clock: Clock,
        llm: LLMService,
        events: EventLog,
        idle: IdleLoop,
        compressor: DigestCompressor,
        foreground: Foreground,
        transcript: TranscriptContext,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.clock = clock
        self.llm = llm
        self.events = events
        self.idle = idle
        self.compressor = compressor
        self.foreground = foreground
        self.transcript = transcript
        self._bg_tasks: set[asyncio.Task] = set()

    @classmethod
    def create(
        cls,
        cfg: Config,
        *,
        fake: bool = False,
        db_path: str | None = None,
        on_thought: ThoughtCallback | None = None,
        base_llm: "BaseLLM | None" = None,
    ) -> "Mind":
        clock = Clock()
        store = Store(db_path or cfg.storage.db_path)
        base = base_llm if base_llm is not None else (FakeLLM() if fake else AnthropicLLM())
        events = EventLog(cfg.logging.dir, clock)
        llm = LLMService(base, store, clock, cfg.logging.dir, cfg.budget.daily_tokens)
        transcript = TranscriptContext(store, llm, clock, cfg, events)
        idle = IdleLoop(
            store, llm, clock, cfg.idle, cfg.models.idle, transcript, events,
            on_thought=on_thought,
        )
        compressor = DigestCompressor(llm, cfg.models.compressor)
        foreground = Foreground(
            llm, cfg.models.foreground, cfg.foreground.max_transcript_turns
        )
        return cls(
            cfg, store, clock, llm, events, idle, compressor, foreground, transcript
        )

    async def start(self) -> None:
        await self.store.open()
        await self.events.write("mind_start")
        await self.idle.start()

    async def shutdown(self) -> None:
        await self.idle.stop()
        for task in list(self._bg_tasks):
            task.cancel()
        await self.events.write("mind_stop")
        await self.llm.close()
        await self.store.close()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def handle_user_message(self, text: str) -> str:
        t_user = self.clock.now()
        simulated = self.clock.simulated
        await self.store.add_clock_mark(t_user, "user_message", simulated=simulated)
        await self.idle.stop()

        # Elapsed time since the end of the last exchange (None on first turn).
        last_reply_ts = await self.store.latest_mark_ts("reply_sent")
        elapsed = (t_user - last_reply_ts) if last_reply_ts is not None else None

        # Compress the idle stream accumulated since the last digest.
        last_digest = await self.store.last_digest()
        since_ts = float(last_digest["covering_to_ts"]) if last_digest else 0.0
        thoughts = await self.store.thoughts_since(since_ts)
        digest: dict | None = None
        if thoughts:
            stream_elapsed = (
                elapsed if elapsed is not None else t_user - float(thoughts[0]["ts"])
            )
            try:
                digest = await self.compressor.compress(thoughts, text, stream_elapsed)
                await self.store.add_digest(
                    t_user,
                    covering_from_ts=since_ts or float(thoughts[0]["ts"]),
                    # a thought can land in the instant after t_user is taken;
                    # the range must cover everything actually compressed or
                    # the next digest would re-include it
                    covering_to_ts=max(t_user, float(thoughts[-1]["ts"])),
                    content=digest,
                )
            except BudgetExceededError:
                await self.events.write("digest_skipped_budget", n_thoughts=len(thoughts))
            except Exception as e:
                await self.events.write(
                    "digest_error", error=f"{type(e).__name__}: {e}",
                    n_thoughts=len(thoughts),
                )

        # Assemble the between-turns note: clock always; digest when present.
        note_parts = [prompts.clock_note(elapsed, t_user)]
        item_ids: dict[str, list[str]] = {}
        valid_ids: set[str] = set()
        if digest is not None:
            item_ids = digest_item_ids(digest)
            valid_ids = {i for ids in item_ids.values() for i in ids}
            note_parts.append(prompts.digest_note(digest, item_ids))
        if self.idle.budget_paused:
            note_parts.append(prompts.BUDGET_PAUSED_NOTE)
        note = "\n\n".join(note_parts)

        history = [row_to_dict(t) for t in await self.store.recent_turns(10_000)]
        turn_id = await self.store.add_turn(t_user, "user", text)

        t0 = time.monotonic()
        result = await self.foreground.reply(
            history=history, user_message=text, note=note, valid_item_ids=valid_ids
        )
        latency_ms = (time.monotonic() - t0) * 1000

        t_reply = self.clock.now()
        await self.store.add_turn(t_reply, "assistant", result.display_text)
        await self.store.add_clock_mark(
            t_reply, "reply_sent", simulated=self.clock.simulated
        )

        # Eval hooks: per-turn surfacing metrics (see brief 3.6).
        digest_bytes = len(prompts.digest_note(digest, item_ids)) if digest else None
        n_items = digest_n_items(digest) if digest else None
        n_surfaced = (
            len(result.surfaced_ids) if result.surfaced_ids is not None else None
        )
        await self.store.add_turn_metrics(
            t_reply, turn_id, digest_bytes, n_items, n_surfaced,
            result.surfaced_ids, latency_ms,
        )
        await self.events.write(
            "turn",
            turn_id=turn_id,
            elapsed_since_reply=elapsed,
            n_thoughts_compressed=len(thoughts),
            digest_bytes=digest_bytes,
            n_digest_items=n_items,
            n_surfaced=n_surfaced,
            surfaced_ids=result.surfaced_ids,
            raw_marker=result.raw_marker,
            latency_ms=round(latency_ms, 1),
        )

        await self.idle.start()
        self._spawn(self.transcript.maybe_regenerate())
        return result.display_text

    async def sleep_sim(self, seconds: float) -> int:
        """/sleep: simulate idle time. Returns ticks run."""
        return await self.idle.simulate(seconds)

    # -- REPL introspection helpers --------------------------------------

    async def stream_since_last_turn(self) -> list[dict]:
        since = await self.store.latest_mark_ts("user_message") or 0.0
        return [row_to_dict(t) for t in await self.store.thoughts_since(since)]

    async def last_digest_dict(self) -> dict | None:
        row = await self.store.last_digest()
        if row is None:
            return None
        return {
            "ts": row["ts"],
            "covering_from_ts": row["covering_from_ts"],
            "covering_to_ts": row["covering_to_ts"],
            **json.loads(row["content_json"]),
        }

    async def clock_marks(self, n: int = 25) -> list[dict]:
        return [row_to_dict(m) for m in await self.store.recent_marks(n)]

    async def budget_status(self) -> tuple[int, int]:
        used = await self.llm.tokens_used_today()
        return used, self.cfg.budget.daily_tokens
