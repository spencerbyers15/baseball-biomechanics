"""The idle loop: a background asyncio task that thinks between user turns.

Lifecycle: start() when a reply is sent (and at program launch), stop() when
a user message arrives. Each tick sees the transcript summary, the last few
turns verbatim, its own last K thoughts, the clock, and (Phase 2) evoked
memories — the stream is continuous, never reset per tick.

Schedule: tick_seconds ± jitter; after backoff_after_minutes of idling each
delay doubles, capped at max_tick_seconds, so a long absence doesn't burn
tokens linearly.

simulate() is the /sleep fast-forward: it advances the mind clock and runs
the same schedule back-to-back, flagged simulated everywhere it lands.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from typing import Awaitable, Callable

from . import prompts
from .clock import Clock
from .config import IdleConfig
from .llm import IDLE, BudgetExceededError, EventLog, LLMService
from .store import Store
from .transcript import TranscriptContext

ThoughtCallback = Callable[[str, bool], None]


def compute_delay(
    cfg: IdleConfig, elapsed_idle: float, backoff_ticks: int, rng: random.Random
) -> float:
    """Seconds until the next tick, given how long the user has been away and
    how many ticks have already fired past the backoff threshold."""
    base = cfg.tick_seconds + rng.uniform(
        -cfg.tick_jitter_seconds, cfg.tick_jitter_seconds
    )
    if elapsed_idle > cfg.backoff_after_minutes * 60:
        # exponent clamped so the intermediate product stays a finite float
        # over arbitrarily long idle sessions; min() below applies the real cap
        base *= 2.0 ** min(backoff_ticks + 1, 30)
    return max(1.0, min(base, cfg.max_tick_seconds))


class IdleLoop:
    def __init__(
        self,
        store: Store,
        llm: LLMService,
        clock: Clock,
        cfg: IdleConfig,
        idle_model: str,
        transcript: TranscriptContext,
        events: EventLog,
        rng: random.Random | None = None,
        on_thought: ThoughtCallback | None = None,
        evoke: Callable[[], Awaitable[list[str]]] | None = None,  # Phase 2 hook
    ) -> None:
        self._store = store
        self._llm = llm
        self._clock = clock
        self._cfg = cfg
        self._model = idle_model
        self._transcript = transcript
        self._events = events
        self._rng = rng or random.Random()
        self.on_thought = on_thought
        self._evoke = evoke

        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._tick_lock = asyncio.Lock()
        self._session_id = ""
        self._session_start = 0.0
        self._tick_n = 0
        self._backoff_ticks = 0
        self._last_tick_ts: float | None = None
        self.budget_paused = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stop = asyncio.Event()
        self._session_id = uuid.uuid4().hex[:12]
        self._session_start = self._clock.now()
        self._tick_n = 0
        self._backoff_ticks = 0
        self._last_tick_ts = None
        # a still-exhausted budget re-trips this on the first tick (checked
        # before any API call); once the day rolls over it stays cleared
        self.budget_paused = False
        await self._store.add_clock_mark(
            self._session_start, "idle_start", simulated=self._clock.simulated
        )
        self._task = asyncio.create_task(self._run(), name="idle-loop")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        # Cancel rather than drain: an in-flight idle LLM call must never
        # delay the foreground reply. The un-stored thought is dropped.
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        await self._store.add_clock_mark(
            self._clock.now(), "idle_stop", simulated=self._clock.simulated
        )

    async def _run(self) -> None:
        while not self._stop.is_set():
            elapsed = self._clock.now() - self._session_start
            delay = compute_delay(self._cfg, elapsed, self._backoff_ticks, self._rng)
            if elapsed > self._cfg.backoff_after_minutes * 60:
                self._backoff_ticks += 1
            offset_before = self._clock.offset
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break  # stop requested during the wait
            except TimeoutError:
                pass
            if self._clock.offset != offset_before:
                continue  # a /sleep ran during this wait; re-arm on the new schedule
            async with self._tick_lock:
                if self._stop.is_set():
                    break
                try:
                    # the offset persists after a /sleep, so post-sleep real
                    # ticks carry offset timestamps and must stay flagged
                    keep_going = await self._tick(simulated=self._clock.simulated)
                except Exception as e:
                    await self._events.write(
                        "idle_tick_error", error=f"{type(e).__name__}: {e}",
                        session_id=self._session_id, tick_n=self._tick_n,
                    )
                    keep_going = True
            if not keep_going:
                break

    async def _tick(self, simulated: bool) -> bool:
        """One thought. Returns False if the loop should halt (budget)."""
        now = self._clock.now()
        elapsed_away = now - self._session_start
        summary, verbatim = await self._transcript.context()
        thoughts = await self._store.last_thoughts(self._cfg.context_thoughts)
        elapsed_since_tick = (
            now - self._last_tick_ts if self._last_tick_ts is not None else None
        )
        memories = await self._evoke() if self._evoke else []
        prompt = prompts.build_idle_prompt(
            summary=summary,
            verbatim_turns=verbatim,
            thoughts=[dict(t) for t in thoughts],
            elapsed_away=elapsed_away,
            elapsed_since_tick=elapsed_since_tick,
            now_ts=now,
            memories=memories,
        )
        try:
            response = await self._llm.complete(
                IDLE,
                model=self._model,
                system=prompts.IDLE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
        except BudgetExceededError:
            self.budget_paused = True
            self._stop.set()
            await self._store.add_clock_mark(now, "idle_paused_budget", simulated)
            await self._events.write("idle_paused_budget", session_id=self._session_id)
            return False
        except Exception as e:
            await self._events.write(
                "idle_tick_error", error=f"{type(e).__name__}: {e}",
                session_id=self._session_id, tick_n=self._tick_n,
            )
            return True  # transient failure; keep the loop alive
        await self._store.add_idle_thought(
            ts=now, content=response.text, tick_n=self._tick_n,
            session_id=self._session_id, simulated=simulated,
        )
        self._tick_n += 1
        self._last_tick_ts = now
        if self.on_thought:
            self.on_thought(response.text, simulated)
        return True

    async def simulate(self, seconds: float) -> int:
        """Fast-forward `seconds` of idle time: advance the mind clock along
        the real tick schedule and run those ticks back-to-back. Returns the
        number of ticks run. Capped at max_sim_ticks per call (the clock
        still advances the full amount)."""
        if not self.running:
            raise RuntimeError("idle loop is not running; nothing to simulate")
        async with self._tick_lock:  # exclude concurrent real ticks
            await self._store.add_clock_mark(
                self._clock.now(), f"sim_sleep_start({seconds:.0f}s)", simulated=True
            )
            await self._events.write("sim_sleep_start", seconds=seconds)
            remaining = seconds
            ticks = 0
            truncated = False
            while remaining > 0:
                if ticks >= self._cfg.max_sim_ticks:
                    truncated = True
                    break
                elapsed = self._clock.now() - self._session_start
                delay = compute_delay(
                    self._cfg, elapsed, self._backoff_ticks, self._rng
                )
                if elapsed > self._cfg.backoff_after_minutes * 60:
                    self._backoff_ticks += 1
                if delay > remaining:
                    break
                self._clock.advance(delay)
                remaining -= delay
                try:
                    keep_going = await self._tick(simulated=True)
                except Exception as e:
                    await self._events.write(
                        "idle_tick_error", error=f"{type(e).__name__}: {e}",
                        session_id=self._session_id, tick_n=self._tick_n,
                        during="simulate",
                    )
                    break
                ticks += 1
                if not keep_going:
                    break
            if remaining > 0:
                self._clock.advance(remaining)  # sleep always advances in full
            await self._store.add_clock_mark(
                self._clock.now(), f"sim_sleep_end({ticks} ticks)", simulated=True
            )
            await self._events.write(
                "sim_sleep_end", seconds=seconds, ticks=ticks, truncated=truncated
            )
            return ticks
