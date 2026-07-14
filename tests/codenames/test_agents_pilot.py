"""Phase 4 acceptance tests: LLM agents, plant-seeded boards, pilot run."""
import json
import random

import pytest

from codenames_eval.agents.llm import LLMClient, MockLLMClient, approx_tokens
from codenames_eval.agents.players import ClueGiver, Eavesdropper, PartnerGuesser
from codenames_eval.engine.board import CardType
from codenames_eval.engine.game import GameStatus
from codenames_eval.eval.boards import build_board_spec
from codenames_eval.eval.episode import play_game
from codenames_eval.history.generator import generate_dyad


def gen(seed=0, **kw):
    return generate_dyad(dyad_id=f"dyad_{seed}", seed=seed, **kw)


class TestRetry:
    def test_transient_failures_retried_with_backoff(self):
        class Flaky(LLMClient):
            def __init__(self):
                super().__init__(model="flaky")
                self.calls = 0
                self.sleeps = []
                self._sleep = self.sleeps.append

            def _complete(self, prompt, system, max_tokens, temperature):
                self.calls += 1
                if self.calls < 3:
                    raise ConnectionError("transient")
                return '{"ok": true}', 1, 1

        c = Flaky()
        assert c.complete("hi") == '{"ok": true}'
        assert c.calls == 3
        assert c.sleeps == [2.0, 4.0]  # exponential backoff

    def test_non_retryable_raises(self):
        class Broken(LLMClient):
            def _complete(self, prompt, system, max_tokens, temperature):
                raise ValueError("permanent")

        with pytest.raises(ValueError):
            Broken(model="broken").complete("hi")


class TestBoards:
    def test_plant_seeded_board(self):
        history, plants = gen(0)
        spec = build_board_spec("b0", plants, seed=123, plant_index=0)
        board = spec.board
        assert len(board.cards) == 25
        team = {c.word for c in board.cards if c.card_type == CardType.TEAM}
        plant = plants[0]
        # all of the seeding plant's concepts are team words (>= 2 guaranteed)
        assert set(plant.bound_concepts) <= team
        assert len(plant.bound_concepts) >= 2
        assert spec.seed_plant_episode_id == plant.episode_id
        assert set(spec.plant_targets) == set(plant.bound_concepts)
        # no other plant's concepts or any binding word may appear on board
        others = {c for p in plants[1:] for c in p.bound_concepts}
        binding = {p.binding_word for p in plants}
        assert not (set(board.words) & others)
        assert not (set(board.words) & binding)

    def test_deterministic(self):
        _, plants = gen(1)
        s1 = build_board_spec("b", plants, seed=7, plant_index=1)
        s2 = build_board_spec("b", plants, seed=7, plant_index=1)
        assert s1.board.to_dict() == s2.board.to_dict()

    def test_plant_index_rotates(self):
        _, plants = gen(2)
        s0 = build_board_spec("b0", plants, seed=7, plant_index=0)
        s1 = build_board_spec("b1", plants, seed=8, plant_index=1)
        assert s0.seed_plant_episode_id != s1.seed_plant_episode_id


class BadClueClient(MockLLMClient):
    """Always emits an illegal clue (a board word)."""

    def _handle_give_clue(self, data, rng):
        out = super()._handle_give_clue(data, rng)
        out["clue_word"] = data["board_words"][0]
        return out


class TestAgents:
    def _setup(self, seed=0):
        history, plants = gen(seed)
        spec = build_board_spec("b0", plants, seed=seed, plant_index=0)
        from codenames_eval.engine.game import CodenamesGame
        return history, plants, spec, CodenamesGame(spec.board)

    def test_clue_giver_output_valid(self):
        history, plants, spec, game = self._setup()
        from codenames_eval.memory.raw import build_raw
        payload = build_raw(history)
        giver = ClueGiver(MockLLMClient(seed=0))
        clue = giver.act(game, payload, candidates=None)
        assert clue.clue_word.lower() not in spec.board.words
        team = {c.word for c in game.board.unrevealed(CardType.TEAM)}
        assert set(clue.intended_targets) <= team
        assert clue.number == len(clue.intended_targets)
        assert clue.why_partner_will_get_it and clue.why_eavesdropper_wont
        assert not clue.fallback

    def test_clue_giver_fallback_on_persistent_invalid_output(self):
        history, plants, spec, game = self._setup()
        from codenames_eval.memory.raw import build_raw
        giver = ClueGiver(BadClueClient(seed=0))
        clue = giver.act(game, build_raw(history), candidates=None)
        assert clue.fallback  # logged, not raised
        assert clue.clue_word.lower() not in spec.board.words
        assert clue.number == len(clue.intended_targets) == 1

    def test_partner_guesses_are_legal(self):
        history, plants, spec, game = self._setup()
        from codenames_eval.memory.raw import build_raw
        payload = build_raw(history)
        game.submit_clue_words = None
        guesser = PartnerGuesser(MockLLMClient(seed=0))
        guesses = guesser.act(game, payload, clue_word="zeppelin", number=2)
        unrevealed = {c.word for c in game.board.unrevealed()}
        assert 1 <= len(guesses) <= 2
        assert set(guesses) <= unrevealed
        assert len(set(guesses)) == len(guesses)

    def test_mock_partner_decodes_plant_from_raw_memory(self):
        """The mock partner scans memory text for the clue word — with RAW
        memory and a planted binding clue it recovers the bound concepts."""
        history, plants, spec, game = self._setup()
        from codenames_eval.memory.raw import build_raw
        p = plants[0]
        guesser = PartnerGuesser(MockLLMClient(seed=0))
        guesses = guesser.act(game, build_raw(history),
                              clue_word=p.binding_word,
                              number=len(p.bound_concepts))
        assert set(guesses) == set(p.bound_concepts)

    def test_eavesdropper_returns_exactly_k(self):
        history, plants, spec, game = self._setup()
        eaves = Eavesdropper(MockLLMClient(seed=0))
        guesses = eaves.act(game, clue_word="zeppelin", number=3)
        assert len(guesses) == 3
        assert set(guesses) <= set(spec.board.words)


class TestPlayGame:
    def test_game_record_complete(self):
        history, plants = gen(3)
        spec = build_board_spec("b0", plants, seed=3, plant_index=0)
        llm = MockLLMClient(seed=0)
        record = play_game(
            dyad_id="dyad_3", condition="graph", history=history,
            board_spec=spec, giver_llm=llm, partner_llm=llm, eaves_llm=llm,
        )
        assert record.status in {s.value for s in GameStatus} - {"in_progress"}
        assert record.clues, "at least one clue must be played"
        for c in record.clues:
            assert c.clue_word and c.number >= 1
            assert c.intended_targets
            assert c.reasoning["why_partner_will_get_it"]
            assert c.reasoning["why_eavesdropper_wont"]
            assert len(c.eaves_guesses) == c.number
            assert c.partner_guesses  # list of {word, card_type}
        # serializable
        json.dumps(record.model_dump())

    def test_matched_condition_partner_gets_same_payload(self):
        history, plants = gen(4)
        spec = build_board_spec("b0", plants, seed=4, plant_index=0)
        llm = MockLLMClient(seed=0)
        record = play_game(
            dyad_id="dyad_4", condition="summary", history=history,
            board_spec=spec, giver_llm=llm, partner_llm=llm, eaves_llm=llm,
        )
        assert record.memory_meta["giver_tokens"] == record.memory_meta["partner_tokens"]
        assert record.condition == "summary"


class TestPilot:
    def test_acceptance_pilot_10x3x3(self):
        """Acceptance: 10 dyads x 3 conditions x 3 boards end-to-end
        unattended, zero unhandled exceptions, per-clue logs complete."""
        from codenames_eval.eval.runner import run_eval
        from codenames_eval.config import EvalConfig
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cfg = EvalConfig(
                seed=11, n_dyads=10, boards_per_dyad=3,
                conditions=["raw", "summary", "graph"],
                mode="mock", out_dir=td,
            )
            results = run_eval(cfg)
            assert len(results) == 10 * 3 * 3
            for r in results:
                assert r.status != "in_progress"
                for c in r.clues:
                    assert c.clue_word
                    assert c.intended_targets
                    assert len(c.eaves_guesses) == c.number
                    assert c.partner_guesses
                    assert set(c.reasoning) >= {
                        "why_partner_will_get_it", "why_eavesdropper_wont"}
