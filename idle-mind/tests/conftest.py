import pytest

from idle_mind.clock import Clock
from idle_mind.config import Config
from idle_mind.llm import EventLog, FakeLLM, LLMService
from idle_mind.store import Store


@pytest.fixture
async def store(tmp_path):
    s = Store(tmp_path / "test.db")
    await s.open()
    yield s
    await s.close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def events(tmp_path, clock):
    return EventLog(tmp_path / "logs", clock)


@pytest.fixture
def llm(store, clock, tmp_path, cfg):
    return LLMService(FakeLLM(), store, clock, tmp_path / "logs", cfg.budget.daily_tokens)
