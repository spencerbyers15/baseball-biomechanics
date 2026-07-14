"""Phase 2 acceptance tests: synthetic history generator with injected plants."""
import json
import re

from codenames_eval.agents.llm import MockLLMClient
from codenames_eval.engine.wordlist import (
    BINDING_WORD_POOL,
    BOARD_WORD_POOL,
)
from codenames_eval.history.generator import generate_dyad
from codenames_eval.history.plants import DyadHistory, Plant
from codenames_eval.history.validate import (
    tokenize,
    validate_cooccurrence,
    validate_rarity,
)


def gen(seed=0, **kw):
    return generate_dyad(dyad_id=f"dyad_{seed}", seed=seed, **kw)


class TestGeneratorShapes:
    def test_basic_shapes(self):
        history, plants = gen(0, n_sessions=30, n_plants=6)
        assert len(history.sessions) == 30
        assert [s.index for s in history.sessions] == list(range(30))
        assert len(plants) == 6
        for p in plants:
            assert 2 <= len(p.bound_concepts) <= 4
            assert p.binding_word in BINDING_WORD_POOL
            assert all(c in BOARD_WORD_POOL for c in p.bound_concepts)
            assert 0 <= p.session_index < 30
        # half engaged / half unengaged
        engaged = [p for p in plants if p.partner_engaged]
        assert len(engaged) == 3
        # no concept or binding word reused across plants
        all_concepts = [c for p in plants for c in p.bound_concepts]
        assert len(all_concepts) == len(set(all_concepts))
        assert len({p.binding_word for p in plants}) == len(plants)
        # plants land in distinct sessions
        assert len({p.session_index for p in plants}) == len(plants)

    def test_session_count_configurable(self):
        history, _ = gen(1, n_sessions=20, n_plants=4)
        assert len(history.sessions) == 20

    def test_determinism(self):
        h1, p1 = gen(42)
        h2, p2 = gen(42)
        assert h1.model_dump() == h2.model_dump()
        assert [p.model_dump() for p in p1] == [p.model_dump() for p in p2]

    def test_different_seeds_differ(self):
        h1, _ = gen(1)
        h2, _ = gen(2)
        assert h1.model_dump() != h2.model_dump()

    def test_roundtrip(self):
        history, plants = gen(3)
        h2 = DyadHistory.model_validate(json.loads(history.model_dump_json()))
        assert h2 == history
        p2 = [Plant.model_validate(json.loads(p.model_dump_json())) for p in plants]
        assert p2 == plants


class TestCooccurrence:
    def test_acceptance_20_histories(self):
        """Acceptance (a): every plant's bound concepts co-occur in exactly
        the intended session, across 20 generated histories."""
        for seed in range(20):
            history, plants = gen(seed)
            report = validate_cooccurrence(history, plants)
            assert report["ok"], f"seed {seed}: {report['violations']}"

    def test_binding_word_and_concepts_present_in_intended_session(self):
        history, plants = gen(5)
        for p in plants:
            tokens = set()
            for turn in history.sessions[p.session_index].turns:
                tokens |= set(tokenize(turn.text))
            assert p.binding_word in tokens
            assert set(p.bound_concepts) <= tokens

    def test_validator_catches_leaked_cooccurrence(self):
        history, plants = gen(6)
        p = plants[0]
        wrong = (p.session_index + 1) % len(history.sessions)
        leak = f"Remember the {p.bound_concepts[0]} and the {p.bound_concepts[1]}?"
        history.sessions[wrong].turns[0].text += " " + leak
        report = validate_cooccurrence(history, plants)
        assert not report["ok"]

    def test_validator_catches_missing_plant(self):
        history, plants = gen(7)
        p = plants[0]
        session = history.sessions[p.session_index]
        session.turns = [
            t for t in session.turns
            if p.binding_word not in tokenize(t.text)
            and not set(p.bound_concepts) & set(tokenize(t.text))
        ]
        report = validate_cooccurrence(history, plants)
        assert not report["ok"]

    def test_validator_catches_binding_word_leak(self):
        history, plants = gen(8)
        p = plants[0]
        wrong = (p.session_index + 2) % len(history.sessions)
        history.sessions[wrong].turns[0].text += f" Speaking of the {p.binding_word}."
        report = validate_cooccurrence(history, plants)
        assert not report["ok"]

    def test_filler_never_mentions_pool_words(self):
        """Non-plant sessions must contain no board-pool or binding-pool words."""
        pool = set(BOARD_WORD_POOL) | set(BINDING_WORD_POOL)
        history, plants = gen(9)
        plant_sessions = {p.session_index for p in plants}
        for s in history.sessions:
            if s.index in plant_sessions:
                continue
            for turn in s.turns:
                hits = set(tokenize(turn.text)) & pool
                assert not hits, f"session {s.index} leaks pool words: {hits}"


class TestEngagementStructure:
    def test_engaged_partner_echoes_plant_content(self):
        history, plants = gen(10)
        for p in plants:
            turns = history.sessions[p.session_index].turns
            intro_idx = next(
                i for i, t in enumerate(turns) if p.binding_word in tokenize(t.text)
            )
            reply = turns[intro_idx + 1]
            reply_tokens = set(tokenize(reply.text))
            plant_vocab = {p.binding_word} | set(p.bound_concepts)
            if p.partner_engaged:
                assert reply_tokens & plant_vocab, "engaged reply must echo content"
            else:
                assert not (reply_tokens & plant_vocab), "unengaged reply must not echo"


class TestRarity:
    def test_acceptance_rarity_machinery_20_histories(self):
        """Acceptance (b) machinery: a fresh (mock) LLM given the binding word
        and full word list matches bound concepts at <= chance + 10%.
        The mock answers at chance; the real-model run requires an API key
        (see BLOCKED.md) and reuses exactly this code path."""
        llm = MockLLMClient(seed=0)
        rates = []
        for seed in range(20):
            _, plants = gen(seed)
            report = validate_rarity(llm, plants, BOARD_WORD_POOL)
            assert report["ok"], report
            rates.append(report["mean_match_rate"])
            assert report["threshold"] <= 0.2  # chance on a 300-word list is tiny
        assert sum(rates) / len(rates) <= report["threshold"]

    def test_rarity_report_structure(self):
        llm = MockLLMClient(seed=1)
        _, plants = gen(0)
        report = validate_rarity(llm, plants, BOARD_WORD_POOL)
        assert set(report) >= {"ok", "mean_match_rate", "threshold", "per_plant"}
        assert len(report["per_plant"]) == len(plants)
        for row in report["per_plant"]:
            assert set(row) >= {"episode_id", "binding_word", "match_rate", "selected"}


class TestLLMPlumbing:
    def test_call_logging(self, tmp_path):
        log = tmp_path / "calls.jsonl"
        llm = MockLLMClient(seed=0, log_path=log)
        llm.complete("hello ```json\n{\"task_id\": \"unknown\"}\n```", task_id="t1")
        rows = [json.loads(l) for l in log.read_text().splitlines()]
        assert len(rows) == 1
        assert re.fullmatch(r"[0-9a-f]{64}", rows[0]["prompt_sha256"])
        assert rows[0]["task_id"] == "t1"
        assert rows[0]["output_tokens"] >= 1
