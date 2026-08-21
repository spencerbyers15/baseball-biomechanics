import asyncio
import random
import time

from idle_mind.config import IdleConfig
from idle_mind.idle_loop import IdleLoop, compute_delay
from idle_mind.llm import FAKE_IDLE_THOUGHTS, EventLog, FakeLLM, LLMService
from idle_mind.transcript import TranscriptContext


def test_delay_within_jitter_band_before_backoff():
    cfg = IdleConfig()
    rng = random.Random(42)
    for _ in range(200):
        d = compute_delay(cfg, elapsed_idle=60.0, backoff_ticks=0, rng=rng)
        assert 30.0 <= d <= 60.0


def test_delay_doubles_after_backoff_and_caps():
    cfg = IdleConfig(tick_jitter_seconds=0)
    rng = random.Random(0)
    past = cfg.backoff_after_minutes * 60 + 1
    assert compute_delay(cfg, past, backoff_ticks=0, rng=rng) == 90.0
    assert compute_delay(cfg, past, backoff_ticks=1, rng=rng) == 180.0
    assert compute_delay(cfg, past, backoff_ticks=2, rng=rng) == 360.0
    assert compute_delay(cfg, past, backoff_ticks=3, rng=rng) == 600.0
    assert compute_delay(cfg, past, backoff_ticks=10, rng=rng) == 600.0


def test_delay_huge_backoff_ticks_no_overflow():
    # days of uninterrupted idling must cap cleanly, not overflow to death
    cfg = IdleConfig(tick_jitter_seconds=0)
    rng = random.Random(0)
    past = cfg.backoff_after_minutes * 60 + 1
    for backoff_ticks in (1023, 5000, 10**6):
        assert compute_delay(cfg, past, backoff_ticks, rng) == 600.0


def _idle_loop(store, clock, cfg, tmp_path, jitter=0.0):
    cfg.idle.tick_jitter_seconds = jitter
    events = EventLog(tmp_path / "logs", clock)
    llm = LLMService(FakeLLM(), store, clock, tmp_path / "logs", 10**9)
    transcript = TranscriptContext(store, llm, clock, cfg, events)
    return IdleLoop(
        store, llm, clock, cfg.idle, "fake-idle", transcript, events,
        rng=random.Random(7),
    )


async def test_simulate_runs_scheduled_ticks_and_advances_clock(
    store, clock, cfg, tmp_path
):
    loop = _idle_loop(store, clock, cfg, tmp_path)
    await loop.start()
    try:
        before = clock.now()
        ticks = await loop.simulate(300)
        # 45s cadence, no jitter: ticks at 45..270 -> 6 ticks in 300s
        assert ticks == 6
        assert abs((clock.now() - before) - 300) < 1.0
        thoughts = await store.thoughts_since(0)
        assert len(thoughts) == 6
        assert all(t["simulated"] == 1 for t in thoughts)
        assert [t["tick_n"] for t in thoughts] == list(range(6))
    finally:
        await loop.stop()


async def test_simulate_respects_max_sim_ticks_but_advances_fully(
    store, clock, cfg, tmp_path
):
    cfg.idle.max_sim_ticks = 3
    loop = _idle_loop(store, clock, cfg, tmp_path)
    await loop.start()
    try:
        before = clock.now()
        ticks = await loop.simulate(3600)
        assert ticks == 3
        assert clock.now() - before >= 3600  # clock still advanced in full
    finally:
        await loop.stop()


async def test_simulate_backoff_reduces_tick_density(store, clock, cfg, tmp_path):
    # 2 hours: 45s cadence for the first 10 min (~14 ticks), then doubling
    # delays 90/180/360/600/600... -> ~26 ticks vs 160 at a linear cadence.
    cfg.idle.max_sim_ticks = 100
    loop = _idle_loop(store, clock, cfg, tmp_path)
    await loop.start()
    try:
        ticks = await loop.simulate(7200)
        assert 20 <= ticks <= 32
    finally:
        await loop.stop()


async def test_stop_is_idempotent_and_marks_clock(store, clock, cfg, tmp_path):
    loop = _idle_loop(store, clock, cfg, tmp_path)
    await loop.start()
    assert loop.running
    await loop.stop()
    await loop.stop()
    assert not loop.running
    events = [m["event"] for m in await store.recent_marks(10)]
    assert "idle_start" in events
    assert "idle_stop" in events


class HangingLLM(FakeLLM):
    """Idle calls hang forever (cancellable); everything else is canned."""

    async def complete(self, *, role, **kw):
        if role == "idle":
            await asyncio.Event().wait()
        return await super().complete(role=role, **kw)


class RecordingLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def complete(self, *, role, messages, **kw):
        self.calls.append((role, messages))
        return await super().complete(role=role, messages=messages, **kw)


async def test_stop_cancels_inflight_tick_promptly(store, clock, cfg, tmp_path):
    # a user message must never wait behind an in-flight idle LLM call
    cfg.idle.tick_seconds = 0.05  # compute_delay floors at 1.0s
    cfg.idle.tick_jitter_seconds = 0
    events = EventLog(tmp_path / "logs", clock)
    llm = LLMService(HangingLLM(), store, clock, tmp_path / "logs", 10**9)
    transcript = TranscriptContext(store, llm, clock, cfg, events)
    loop = IdleLoop(
        store, llm, clock, cfg.idle, "fake-idle", transcript, events,
        rng=random.Random(7),
    )
    await loop.start()
    await asyncio.sleep(1.3)  # first tick fired and is now hung in the LLM call
    t0 = time.monotonic()
    await loop.stop()
    assert time.monotonic() - t0 < 1.0
    assert await store.thoughts_since(0) == []  # the hung thought was dropped


async def test_budget_paused_clears_on_restart(store, clock, cfg, tmp_path):
    events = EventLog(tmp_path / "logs", clock)
    llm = LLMService(FakeLLM(), store, clock, tmp_path / "logs", daily_token_budget=1)
    await store.add_usage(clock.now(), llm._today(), "idle", "m", 5, 5)
    transcript = TranscriptContext(store, llm, clock, cfg, events)
    cfg.idle.tick_jitter_seconds = 0
    loop = IdleLoop(
        store, llm, clock, cfg.idle, "fake-idle", transcript, events,
        rng=random.Random(7),
    )
    await loop.start()
    await loop.simulate(100)
    assert loop.budget_paused
    await loop.stop()
    await loop.start()  # e.g. the next day: the stale flag must not persist
    assert not loop.budget_paused
    await loop.stop()


async def test_real_tick_after_sleep_offset_flagged_simulated(
    store, clock, cfg, tmp_path
):
    clock.advance(3600)  # as if a /sleep already ran; offset persists
    cfg.idle.tick_seconds = 0.05  # floors to 1.0s
    loop = _idle_loop(store, clock, cfg, tmp_path)
    await loop.start()
    try:
        await asyncio.sleep(1.3)  # one real (timer-fired) tick
        thoughts = await store.thoughts_since(0)
        assert thoughts
        assert all(t["simulated"] == 1 for t in thoughts)
    finally:
        await loop.stop()


async def test_idle_context_carries_prior_thoughts(store, clock, cfg, tmp_path):
    cfg.idle.tick_jitter_seconds = 0
    events = EventLog(tmp_path / "logs", clock)
    rec = RecordingLLM()
    llm = LLMService(rec, store, clock, tmp_path / "logs", 10**9)
    transcript = TranscriptContext(store, llm, clock, cfg, events)
    loop = IdleLoop(
        store, llm, clock, cfg.idle, "fake-idle", transcript, events,
        rng=random.Random(7),
    )
    await loop.start()
    try:
        await loop.simulate(100)  # 2 ticks at 45s cadence
    finally:
        await loop.stop()
    idle_prompts = [m for role, m in rec.calls if role == "idle"]
    assert len(idle_prompts) == 2
    second_prompt = idle_prompts[1][0]["content"]
    assert FAKE_IDLE_THOUGHTS[0][:40] in second_prompt  # continuity, not reset


async def test_budget_exhaustion_pauses_loop(store, clock, cfg, tmp_path):
    events = EventLog(tmp_path / "logs", clock)
    llm = LLMService(FakeLLM(), store, clock, tmp_path / "logs", daily_token_budget=1)
    await store.add_usage(clock.now(), llm._today(), "idle", "m", 5, 5)  # budget spent
    transcript = TranscriptContext(store, llm, clock, cfg, events)
    loop = IdleLoop(
        store, llm, clock, cfg.idle, "fake-idle", transcript, events,
        rng=random.Random(7),
    )
    cfg.idle.tick_jitter_seconds = 0
    await loop.start()
    try:
        await loop.simulate(100)
        assert loop.budget_paused
        assert await store.thoughts_since(0) == []
        marks = [m["event"] for m in await store.recent_marks(10)]
        assert "idle_paused_budget" in marks
    finally:
        await loop.stop()
