import json

import pytest

from idle_mind.llm import (
    COMPRESSOR,
    FOREGROUND,
    IDLE,
    SUMMARIZER,
    BudgetExceededError,
    FakeLLM,
    LLMService,
)

MSG = [{"role": "user", "content": "x"}]


@pytest.fixture
async def exhausted_llm(store, clock, tmp_path):
    llm = LLMService(FakeLLM(), store, clock, tmp_path / "logs", daily_token_budget=1)
    await store.add_usage(clock.now(), llm._today(), "idle", "m", 5, 5)
    return llm


async def test_background_roles_refused_when_budget_exhausted(exhausted_llm):
    for role in (IDLE, COMPRESSOR, SUMMARIZER):
        with pytest.raises(BudgetExceededError):
            await exhausted_llm.complete(role, model="m", system="s", messages=MSG)


async def test_foreground_allowed_when_budget_exhausted(exhausted_llm):
    response = await exhausted_llm.complete(
        FOREGROUND, model="m", system="s", messages=MSG
    )
    assert response.text  # user-initiated calls always go through


async def test_usage_recorded_and_logged(store, clock, tmp_path):
    llm = LLMService(FakeLLM(), store, clock, tmp_path / "logs", 10**9)
    await llm.complete(IDLE, model="m", system="s", messages=MSG)
    assert await llm.tokens_used_today() > 0
    log = next((tmp_path / "logs").glob("llm-*.jsonl"))
    record = json.loads(log.read_text().splitlines()[0])
    assert record["role"] == IDLE
    assert record["request"]["messages"] == MSG
    assert record["response"]["text"]
    assert record["simulated"] is False
