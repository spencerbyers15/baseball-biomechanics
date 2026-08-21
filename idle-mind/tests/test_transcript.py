from idle_mind.transcript import TranscriptContext


async def _add_turns(store, n, start=0):
    for i in range(start, start + n):
        role = "user" if i % 2 == 0 else "assistant"
        await store.add_turn(float(i), role, f"message {i}")


async def test_no_summary_before_threshold(store, clock, cfg, llm, events):
    tc = TranscriptContext(store, llm, clock, cfg, events)
    await _add_turns(store, 9)
    await tc.maybe_regenerate()
    assert await store.latest_summary() is None


async def test_summary_written_at_threshold(store, clock, cfg, llm, events):
    tc = TranscriptContext(store, llm, clock, cfg, events)
    await _add_turns(store, 10)
    await tc.maybe_regenerate()
    row = await store.latest_summary()
    assert row is not None
    assert row["upto_turn_id"] == 10
    assert row["content"]


async def test_no_regeneration_until_enough_new_turns(store, clock, cfg, llm, events):
    tc = TranscriptContext(store, llm, clock, cfg, events)
    await _add_turns(store, 10)
    await tc.maybe_regenerate()
    first = await store.latest_summary()
    await _add_turns(store, 5, start=10)
    await tc.maybe_regenerate()
    assert (await store.latest_summary())["id"] == first["id"]
    await _add_turns(store, 5, start=15)
    await tc.maybe_regenerate()
    second = await store.latest_summary()
    assert second["id"] != first["id"]
    assert second["upto_turn_id"] == 20


async def test_context_returns_summary_and_verbatim_tail(store, clock, cfg, llm, events):
    tc = TranscriptContext(store, llm, clock, cfg, events)
    await _add_turns(store, 10)
    await tc.maybe_regenerate()
    summary, verbatim = await tc.context()
    assert summary  # the regenerated summary text
    assert len(verbatim) == cfg.idle.transcript_verbatim_turns
    assert verbatim[-1]["content"] == "message 9"
