"""End-to-end turn lifecycle against the fake LLM: idle ticks accumulate,
a digest is produced on the user's return, the clock is injected, metrics
are recorded, and the idle loop restarts."""

import json

import pytest

from idle_mind.app import Mind
from idle_mind.config import Config
from idle_mind.llm import FakeLLM, LLMResponse


def _make_cfg(tmp_path):
    cfg = Config()
    cfg.idle.tick_jitter_seconds = 0
    cfg.storage.db_path = str(tmp_path / "mind.db")
    cfg.logging.dir = str(tmp_path / "logs")
    return cfg


@pytest.fixture
async def mind(tmp_path):
    m = Mind.create(_make_cfg(tmp_path), fake=True)
    await m.start()
    yield m
    await m.shutdown()


async def test_first_turn_no_digest(mind):
    reply = await mind.handle_user_message("hello there")
    assert "hello there" in reply
    assert "⟦" not in reply  # marker stripped
    assert await mind.last_digest_dict() is None  # no idle thoughts yet
    marks = [m["event"] for m in await mind.clock_marks()]
    assert marks.count("user_message") == 1
    assert marks.count("reply_sent") == 1


async def test_sleep_then_message_produces_digest(mind):
    await mind.handle_user_message("hello")
    ticks = await mind.sleep_sim(300)
    assert ticks == 6
    reply = await mind.handle_user_message("I'm back")
    assert reply
    digest = await mind.last_digest_dict()
    assert digest is not None
    assert digest["n_thoughts"] == 6
    assert isinstance(digest["tangents"], list)

    # eval hooks: metrics row exists for the digest turn
    cur = await mind.store.db.execute(
        "SELECT * FROM turn_metrics ORDER BY id DESC LIMIT 1"
    )
    row = await cur.fetchone()
    assert row["n_digest_items"] is not None
    assert row["digest_bytes"] > 0
    assert row["latency_ms"] >= 0
    assert row["n_surfaced"] == 0  # fake foreground says ⟦surfaced: none⟧


async def test_quick_return_within_one_tick_skips_digest(mind):
    await mind.handle_user_message("hello")
    reply = await mind.handle_user_message("back immediately")
    assert reply
    assert await mind.last_digest_dict() is None


async def test_second_digest_covers_only_new_thoughts(mind):
    await mind.handle_user_message("hello")
    await mind.sleep_sim(150)  # 3 ticks
    await mind.handle_user_message("back")
    first = await mind.last_digest_dict()
    assert first["n_thoughts"] == 3
    await mind.sleep_sim(100)  # 2 more ticks
    await mind.handle_user_message("again")
    second = await mind.last_digest_dict()
    assert second["n_thoughts"] == 2
    assert second["covering_from_ts"] == first["covering_to_ts"]


async def test_idle_restarts_after_reply(mind):
    await mind.handle_user_message("hello")
    assert mind.idle.running


async def test_transcript_and_stream_introspection(mind):
    await mind.handle_user_message("hello")
    await mind.sleep_sim(100)
    stream = await mind.stream_since_last_turn()
    assert len(stream) == 2
    assert all(t["simulated"] for t in stream)


async def test_idle_stop_mark_between_user_message_and_reply(mind):
    await mind.handle_user_message("hello")
    events = [m["event"] for m in await mind.clock_marks()]
    # launch idle_start, then the turn: message stops the stream, reply restarts it
    assert events[-4:] == ["user_message", "idle_stop", "reply_sent", "idle_start"]


async def test_clock_note_reaches_foreground_even_without_digest(mind, tmp_path):
    await mind.handle_user_message("hello")
    await mind.handle_user_message("again")  # no idle thoughts in between
    llm_log = next((tmp_path / "logs").glob("llm-*.jsonl"))
    records = [json.loads(line) for line in llm_log.read_text().splitlines()]
    fg = [r for r in records if r["role"] == "foreground"]
    note = fg[-1]["request"]["messages"][-1]["content"][0]["text"]
    assert "[Clock:" in note and "seconds" in note
    assert "[Between turns" not in note  # digest was rightly skipped


async def test_digest_failure_falls_back_to_clock_and_carries_forward(mind):
    await mind.handle_user_message("hello")
    await mind.sleep_sim(150)  # 3 thoughts
    real_compress = mind.compressor.compress

    async def boom(*args, **kwargs):
        raise RuntimeError("compressor down")

    mind.compressor.compress = boom
    reply = await mind.handle_user_message("back")
    assert reply  # the turn still completes, clock-only
    assert await mind.last_digest_dict() is None

    mind.compressor.compress = real_compress
    await mind.sleep_sim(100)  # 2 more thoughts
    await mind.handle_user_message("again")
    digest = await mind.last_digest_dict()
    assert digest["n_thoughts"] == 5  # the 3 undigested thoughts carried forward


async def test_digest_covers_thought_recorded_after_message_ts(mind):
    await mind.handle_user_message("hello")
    late_ts = mind.clock.now() + 5
    await mind.store.add_idle_thought(
        ts=late_ts, content="late thought", tick_n=99, session_id="x", simulated=True
    )
    await mind.handle_user_message("back")
    digest = await mind.last_digest_dict()
    assert digest["n_thoughts"] == 1
    assert digest["covering_to_ts"] >= late_ts
    await mind.handle_user_message("again")
    digest2 = await mind.last_digest_dict()
    assert digest2["n_thoughts"] == 1  # the late thought was NOT re-digested
    assert digest2["covering_to_ts"] == digest["covering_to_ts"]


class MarkerLLM(FakeLLM):
    """Foreground replies that actually surface a digest item."""

    async def complete(self, *, role, **kw):
        if role == "foreground":
            return LLMResponse(
                text="I kept circling tides while you were gone.\n⟦surfaced: T1⟧",
                model="fake-foreground", input_tokens=1, output_tokens=1,
                stop_reason="end_turn",
            )
        return await super().complete(role=role, **kw)


async def test_surfaced_ids_recorded_end_to_end(tmp_path):
    mind = Mind.create(_make_cfg(tmp_path), base_llm=MarkerLLM())
    await mind.start()
    try:
        await mind.handle_user_message("hello")
        await mind.sleep_sim(150)
        reply = await mind.handle_user_message("back")
        assert reply == "I kept circling tides while you were gone."
        assert "⟦" not in reply
        cur = await mind.store.db.execute(
            "SELECT * FROM turn_metrics ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
        assert row["n_surfaced"] == 1
        assert json.loads(row["surfaced_ids"]) == ["T1"]
        turns = await mind.store.recent_turns(1)
        assert turns[0]["content"] == "I kept circling tides while you were gone."
    finally:
        await mind.shutdown()


async def test_llm_log_written_with_prompts_and_completions(mind, tmp_path):
    await mind.handle_user_message("hello")
    log_dir = tmp_path / "logs"
    llm_logs = list(log_dir.glob("llm-*.jsonl"))
    assert llm_logs
    records = [json.loads(line) for line in llm_logs[0].read_text().splitlines()]
    fg = [r for r in records if r["role"] == "foreground"]
    assert fg
    assert fg[0]["request"]["system"]
    assert fg[0]["response"]["text"]
