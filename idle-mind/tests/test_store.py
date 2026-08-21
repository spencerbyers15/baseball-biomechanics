import json


async def test_turns_roundtrip(store):
    await store.add_turn(1.0, "user", "hello")
    await store.add_turn(2.0, "assistant", "hi")
    turns = await store.recent_turns(10)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert await store.turn_count() == 2


async def test_recent_turns_limit_returns_latest_in_order(store):
    for i in range(5):
        await store.add_turn(float(i), "user", f"m{i}")
    turns = await store.recent_turns(2)
    assert [t["content"] for t in turns] == ["m3", "m4"]


async def test_idle_thoughts_since_and_last(store):
    for i in range(4):
        await store.add_idle_thought(
            ts=float(i), content=f"t{i}", tick_n=i, session_id="s1", simulated=i == 3
        )
    since = await store.thoughts_since(1.0)
    assert [t["content"] for t in since] == ["t2", "t3"]
    assert since[-1]["simulated"] == 1
    last2 = await store.last_thoughts(2)
    assert [t["content"] for t in last2] == ["t2", "t3"]


async def test_digest_roundtrip(store):
    content = {"tangents": ["a"], "n_thoughts": 3}
    await store.add_digest(10.0, covering_from_ts=1.0, covering_to_ts=10.0, content=content)
    row = await store.last_digest()
    assert json.loads(row["content_json"]) == content
    assert row["covering_from_ts"] == 1.0


async def test_clock_marks(store):
    await store.add_clock_mark(1.0, "user_message")
    await store.add_clock_mark(2.0, "reply_sent")
    await store.add_clock_mark(3.0, "user_message", simulated=True)
    assert await store.latest_mark_ts("user_message") == 3.0
    assert await store.latest_mark_ts("idle_start") is None
    marks = await store.recent_marks(10)
    assert marks[-1]["simulated"] == 1


async def test_usage_budget_by_day(store):
    await store.add_usage(1.0, "2026-08-21", "idle", "m", 100, 50)
    await store.add_usage(2.0, "2026-08-21", "foreground", "m", 200, 100)
    await store.add_usage(3.0, "2026-08-22", "idle", "m", 999, 1)
    assert await store.tokens_used_on("2026-08-21") == 450
    assert await store.tokens_used_on("2026-08-22") == 1000
    assert await store.tokens_used_on("2026-08-23") == 0


async def test_summaries(store):
    await store.add_summary(1.0, upto_turn_id=10, content="first")
    await store.add_summary(2.0, upto_turn_id=20, content="second")
    row = await store.latest_summary()
    assert row["content"] == "second"
    assert row["upto_turn_id"] == 20


async def test_turn_metrics_roundtrip(store):
    await store.add_turn_metrics(
        1.0, turn_id=1, digest_bytes=200, n_digest_items=4, n_surfaced=1,
        surfaced_ids=["T1"], latency_ms=123.4,
    )
    cur = await store.db.execute("SELECT * FROM turn_metrics")
    row = await cur.fetchone()
    assert row["n_surfaced"] == 1
    assert json.loads(row["surfaced_ids"]) == ["T1"]


# -- memory store (Phase 2 data layer) -----------------------------------


async def test_memory_crud(store):
    mid = await store.add_memory(
        created_ts=1.0, content="They conceded the gradient point.",
        source_type="conversation", salience=0.8,
    )
    row = await store.get_memory(mid)
    assert row["salience"] == 0.8
    assert row["evoke_count"] == 0
    assert row["last_evoked_ts"] is None
    assert row["embedding"] is None


async def test_memory_evocation_bookkeeping(store):
    mid = await store.add_memory(1.0, "m", "idle", 0.5)
    await store.record_evocation(mid, 5.0)
    await store.record_evocation(mid, 9.0)
    row = await store.get_memory(mid)
    assert row["evoke_count"] == 2
    assert row["last_evoked_ts"] == 9.0


async def test_memory_salience_and_archive(store):
    mid = await store.add_memory(1.0, "m", "idle", 0.5)
    await store.set_salience(mid, 0.1)
    assert (await store.get_memory(mid))["salience"] == 0.1
    await store.archive_memory(mid)
    assert await store.memories() == []
    archived = await store.memories(include_archived=True)
    assert len(archived) == 1 and archived[0]["archived"] == 1


async def test_memory_embedding_blob_roundtrip(store):
    blob = b"\x00\x01\x02\xff" * 8
    mid = await store.add_memory(1.0, "m", "percept", 0.5, embedding=blob)
    assert (await store.get_memory(mid))["embedding"] == blob
