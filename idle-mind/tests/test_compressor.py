import json
from unittest.mock import AsyncMock

import pytest

from idle_mind.compressor import (
    DIGEST_SCHEMA,
    DigestCompressor,
    digest_item_ids,
    digest_n_items,
    parse_digest_json,
    validate_digest,
)
from idle_mind.llm import LLMResponse
from idle_mind.prompts import digest_note


def _thoughts(n):
    return [{"ts": float(i * 45), "content": f"thought {i}"} for i in range(n)]


def _response(text):
    return LLMResponse(
        text=text, model="m", input_tokens=10, output_tokens=10, stop_reason="end_turn"
    )


GOOD = {
    "elapsed_human": "model's own phrasing",
    "n_thoughts": 999,  # deliberately wrong: must be overridden by ground truth
    "tangents": ["tangent one", "tangent two"],
    "questions_for_user": ["a question?"],
    "revisions": ["I doubt X now."],
    "mood": "restless",
}


async def test_compress_valid_json():
    llm = AsyncMock()
    llm.complete.return_value = _response(json.dumps(GOOD))
    compressor = DigestCompressor(llm, "model-x")
    digest = await compressor.compress(_thoughts(12), "hello again", 2400.0)

    assert digest["tangents"] == ["tangent one", "tangent two"]
    assert digest["questions_for_user"] == ["a question?"]
    assert digest["mood"] == "restless"
    # ground truth wins over model claims
    assert digest["n_thoughts"] == 12
    assert digest["elapsed_human"] == "about 40 minutes"
    # schema was passed through to the LLM call
    assert llm.complete.call_args.kwargs["json_schema"] == DIGEST_SCHEMA


async def test_compress_json_wrapped_in_prose():
    llm = AsyncMock()
    llm.complete.return_value = _response(
        "Here is the digest:\n" + json.dumps(GOOD) + "\nDone."
    )
    compressor = DigestCompressor(llm, "model-x")
    digest = await compressor.compress(_thoughts(3), "hi", 120.0)
    assert digest["tangents"] == ["tangent one", "tangent two"]


async def test_compress_garbage_falls_back_to_minimal_digest():
    llm = AsyncMock()
    llm.complete.return_value = _response("not json at all")
    compressor = DigestCompressor(llm, "model-x")
    digest = await compressor.compress(_thoughts(5), "hi", 300.0)
    assert digest["n_thoughts"] == 5
    assert digest["tangents"] == []
    assert digest["questions_for_user"] == []
    assert digest["revisions"] == []
    assert "mood" not in digest


async def test_compress_refuses_empty_stream():
    compressor = DigestCompressor(AsyncMock(), "model-x")
    with pytest.raises(ValueError):
        await compressor.compress([], "hi", 60.0)


def test_validate_clamps_oversized_lists():
    data = {
        "tangents": ["a", "b", "c", "d", "e"],
        "questions_for_user": ["q1", "q2", "q3"],
        "revisions": ["r1", "r2", "r3"],
    }
    digest = validate_digest(data, n_thoughts=4, elapsed_seconds=90.0)
    assert len(digest["tangents"]) == 3
    assert len(digest["questions_for_user"]) == 2
    assert len(digest["revisions"]) == 2


def test_validate_drops_non_string_items_and_blank_mood():
    data = {"tangents": ["ok", 42, None, "  "], "mood": "   "}
    digest = validate_digest(data, n_thoughts=1, elapsed_seconds=10.0)
    assert digest["tangents"] == ["ok"]
    assert "mood" not in digest


def test_parse_digest_json_variants():
    assert parse_digest_json('{"a": 1}') == {"a": 1}
    assert parse_digest_json('noise {"a": 1} noise') == {"a": 1}
    assert parse_digest_json("[1, 2]") is None
    assert parse_digest_json("nothing here") is None


def test_item_ids_and_counts():
    digest = validate_digest(GOOD, n_thoughts=2, elapsed_seconds=60.0)
    ids = digest_item_ids(digest)
    assert ids["tangents"] == ["T1", "T2"]
    assert ids["questions_for_user"] == ["Q1"]
    assert ids["revisions"] == ["R1"]
    assert digest_n_items(digest) == 4


def test_digest_note_renders_ids_and_skips_empty_sections():
    digest = validate_digest(
        {"tangents": ["only tangent"], "questions_for_user": [], "revisions": []},
        n_thoughts=7,
        elapsed_seconds=2400.0,
    )
    note = digest_note(digest, digest_item_ids(digest))
    assert "about 40 minutes" in note
    assert "7 idle thoughts" in note
    assert "T1: only tangent" in note
    assert "Questions" not in note
    assert "Revisions" not in note
    assert "surface any of this or none of it" in note
