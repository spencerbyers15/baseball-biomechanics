"""Phase 1 acceptance tests: deterministic Codenames engine."""
import json
import random

import pytest

from codenames_eval.engine.board import Board, CardType
from codenames_eval.engine.game import (
    Clue,
    CodenamesGame,
    GameStatus,
    RuleViolation,
)
from codenames_eval.engine.wordlist import BOARD_WORD_POOL


def make_board(seed: int = 0) -> Board:
    rng = random.Random(seed)
    words = rng.sample(BOARD_WORD_POOL, 25)
    return Board.generate(words, rng)


def make_game(seed: int = 0) -> CodenamesGame:
    return CodenamesGame(make_board(seed), max_turns=25)


def team_words(game):
    return [c.word for c in game.board.cards if c.card_type == CardType.TEAM]


def words_of(game, card_type):
    return [c.word for c in game.board.cards if c.card_type == card_type]


class TestBoard:
    def test_composition(self):
        board = make_board()
        assert len(board.cards) == 25
        counts = {t: 0 for t in CardType}
        for c in board.cards:
            counts[c.card_type] += 1
        assert counts[CardType.TEAM] == 9
        assert counts[CardType.DISTRACTOR] == 8
        assert counts[CardType.NEUTRAL] == 7
        assert counts[CardType.ASSASSIN] == 1
        assert len({c.word for c in board.cards}) == 25

    def test_generate_rejects_wrong_word_count(self):
        rng = random.Random(0)
        with pytest.raises(ValueError):
            Board.generate(["a", "b"], rng)

    def test_generate_rejects_duplicates(self):
        rng = random.Random(0)
        with pytest.raises(ValueError):
            Board.generate(["dup"] * 25, rng)


class TestClueValidation:
    def test_clue_word_on_board_rejected(self):
        game = make_game()
        board_word = game.board.cards[0].word
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word=board_word, number=1))

    def test_clue_word_on_board_rejected_case_insensitive(self):
        game = make_game()
        board_word = game.board.cards[3].word.upper()
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word=board_word, number=1))

    def test_clue_number_out_of_range(self):
        game = make_game()
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word="zeppelin", number=0))
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word="zeppelin", number=10))

    def test_multiword_clue_rejected(self):
        game = make_game()
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word="two words", number=1))

    def test_clue_while_awaiting_guess_rejected(self):
        game = make_game()
        game.submit_clue(Clue(word="zeppelin", number=2))
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word="gondola", number=1))


class TestGuessResolution:
    def test_guess_without_clue_rejected(self):
        game = make_game()
        with pytest.raises(RuleViolation):
            game.guess(game.board.cards[0].word)

    def test_guess_word_not_on_board_rejected(self):
        game = make_game()
        game.submit_clue(Clue(word="zeppelin", number=1))
        with pytest.raises(RuleViolation):
            game.guess("notaboardword")

    def test_correct_guess_continues_turn(self):
        game = make_game()
        tw = team_words(game)
        game.submit_clue(Clue(word="zeppelin", number=2))
        result = game.guess(tw[0])
        assert result.card_type == CardType.TEAM
        assert not result.ended_turn
        assert game.awaiting_guess

    def test_neutral_ends_turn(self):
        game = make_game()
        game.submit_clue(Clue(word="zeppelin", number=2))
        result = game.guess(words_of(game, CardType.NEUTRAL)[0])
        assert result.card_type == CardType.NEUTRAL
        assert result.ended_turn
        assert not game.awaiting_guess
        assert game.status == GameStatus.IN_PROGRESS

    def test_distractor_ends_turn(self):
        game = make_game()
        game.submit_clue(Clue(word="zeppelin", number=2))
        result = game.guess(words_of(game, CardType.DISTRACTOR)[0])
        assert result.ended_turn
        assert game.status == GameStatus.IN_PROGRESS

    def test_assassin_ends_game(self):
        game = make_game()
        game.submit_clue(Clue(word="zeppelin", number=2))
        result = game.guess(words_of(game, CardType.ASSASSIN)[0])
        assert result.ended_turn
        assert result.ended_game
        assert game.status == GameStatus.LOST_ASSASSIN
        with pytest.raises(RuleViolation):
            game.submit_clue(Clue(word="gondola", number=1))

    def test_bonus_guess_allowed_then_overguess_rejected(self):
        game = make_game()
        tw = team_words(game)
        game.submit_clue(Clue(word="zeppelin", number=2))
        game.guess(tw[0])
        game.guess(tw[1])
        # bonus guess (number + 1)
        result = game.guess(tw[2])
        assert not result.ended_game
        # turn must have ended after using the bonus guess
        assert not game.awaiting_guess
        with pytest.raises(RuleViolation):
            game.guess(tw[3])

    def test_already_revealed_guess_rejected(self):
        game = make_game()
        tw = team_words(game)
        game.submit_clue(Clue(word="zeppelin", number=3))
        game.guess(tw[0])
        with pytest.raises(RuleViolation):
            game.guess(tw[0])

    def test_pass_ends_turn(self):
        game = make_game()
        tw = team_words(game)
        game.submit_clue(Clue(word="zeppelin", number=2))
        game.guess(tw[0])
        game.pass_turn()
        assert not game.awaiting_guess
        # a fresh clue is now legal
        game.submit_clue(Clue(word="gondola", number=1))

    def test_pass_before_first_guess_rejected(self):
        # a team must always make at least one guess after a clue
        game = make_game()
        game.submit_clue(Clue(word="zeppelin", number=1))
        with pytest.raises(RuleViolation):
            game.pass_turn()

    def test_win_condition(self):
        game = make_game()
        tw = team_words(game)
        for i in range(0, 9, 3):
            game.submit_clue(Clue(word=f"clueword{i}", number=3))
            for w in tw[i : i + 3]:
                game.guess(w)
            if game.awaiting_guess:  # decline the bonus guess
                game.pass_turn()
        assert game.status == GameStatus.WON


class TestSerialization:
    def test_game_log_roundtrip(self):
        game = make_game(seed=7)
        tw = team_words(game)
        game.submit_clue(Clue(word="zeppelin", number=2))
        game.guess(tw[0])
        game.guess(words_of(game, CardType.NEUTRAL)[0])
        payload = json.dumps(game.to_dict())
        restored = CodenamesGame.from_dict(json.loads(payload))
        assert restored.to_dict() == game.to_dict()
        assert restored.status == game.status
        assert restored.turn_index == game.turn_index
        # restored game continues to enforce rules
        restored.submit_clue(Clue(word="gondola", number=1))
        with pytest.raises(RuleViolation):
            restored.guess(tw[0])  # already revealed


class ScriptedGiver:
    """Hardcoded clue policy: always clue 2 unrevealed team words."""

    def __init__(self, rng):
        self.rng = rng
        self.counter = 0

    def act(self, game):
        unrevealed = [
            c.word
            for c in game.board.cards
            if c.card_type == CardType.TEAM and not c.revealed
        ]
        n = min(2, len(unrevealed))
        self.counter += 1
        return Clue(word=f"scripted{self.counter}", number=n), unrevealed[:n]


class ScriptedGuesser:
    """Hardcoded guess policy: guess named targets, sometimes a wrong word."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, game, targets):
        guesses = []
        for t in targets:
            if self.rng.random() < 0.25:
                wrong = [
                    c.word
                    for c in game.board.cards
                    if not c.revealed and c.card_type != CardType.TEAM
                ]
                if wrong:
                    guesses.append(self.rng.choice(wrong))
                    break
            guesses.append(t)
        return guesses


class TestScriptedGames:
    def test_100_games_no_violations(self):
        """Acceptance: 100 scripted games complete with zero rule violations
        and every game log round-trips through serialization."""
        statuses = []
        for seed in range(100):
            rng = random.Random(seed)
            game = make_game(seed)
            giver, guesser = ScriptedGiver(rng), ScriptedGuesser(rng)
            while game.status == GameStatus.IN_PROGRESS:
                clue, targets = giver.act(game)
                game.submit_clue(clue)  # RuleViolation would fail the test
                for w in guesser.act(game, targets):
                    if not game.awaiting_guess:
                        break
                    result = game.guess(w)
                    if result.ended_turn:
                        break
                if game.awaiting_guess:
                    game.pass_turn()
            statuses.append(game.status)
            restored = CodenamesGame.from_dict(
                json.loads(json.dumps(game.to_dict()))
            )
            assert restored.to_dict() == game.to_dict()
        assert all(s != GameStatus.IN_PROGRESS for s in statuses)
        # sanity: scripted play should win most games and lose some to luck
        assert statuses.count(GameStatus.WON) > 50
