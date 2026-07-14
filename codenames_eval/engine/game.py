"""Deterministic single-team Codenames engine with a fully serializable log.

Rules implemented:
- clue = (word, number); the word must be a single token and must not match
  any board word (case-insensitive); 1 <= number <= 9
- after a clue, the team may guess up to number + 1 times (bonus guess) and
  must make at least one guess before passing
- guessing a non-team card ends the turn; the assassin ends the game (loss)
- revealing all team cards wins; hitting max_turns ends the game (timeout)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .board import Board, CardType


class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    WON = "won"
    LOST_ASSASSIN = "lost_assassin"
    TIMEOUT = "timeout"


class RuleViolation(Exception):
    """Raised on any illegal clue, guess, or out-of-order action."""


@dataclass(frozen=True)
class Clue:
    word: str
    number: int

    def to_dict(self) -> dict:
        return {"word": self.word, "number": self.number}


@dataclass(frozen=True)
class GuessResult:
    word: str
    card_type: CardType
    ended_turn: bool
    ended_game: bool


class CodenamesGame:
    def __init__(self, board: Board, max_turns: int = 25):
        self.board = board
        self.max_turns = max_turns
        self.status = GameStatus.IN_PROGRESS
        self.turn_index = 0
        self.current_clue: Clue | None = None
        self.guesses_this_turn = 0
        self.log: list[dict] = []

    # ----- state helpers -------------------------------------------------
    @property
    def awaiting_guess(self) -> bool:
        return self.current_clue is not None

    def _require_in_progress(self) -> None:
        if self.status != GameStatus.IN_PROGRESS:
            raise RuleViolation(f"game is over ({self.status.value})")

    def _log(self, event: dict) -> None:
        self.log.append(event)

    # ----- actions --------------------------------------------------------
    def submit_clue(self, clue: Clue) -> None:
        self._require_in_progress()
        if self.awaiting_guess:
            raise RuleViolation("cannot submit a clue while guesses are pending")
        word = clue.word.strip().lower()
        if not word or " " in word or "\t" in word:
            raise RuleViolation(f"clue must be a single word, got {clue.word!r}")
        if not (1 <= clue.number <= 9):
            raise RuleViolation(f"clue number must be 1-9, got {clue.number}")
        if any(word == c.word for c in self.board.cards):
            raise RuleViolation(f"clue word {word!r} is on the board")
        self.current_clue = Clue(word=word, number=clue.number)
        self.guesses_this_turn = 0
        self.turn_index += 1
        self._log({"event": "clue", "turn": self.turn_index, "word": word, "number": clue.number})

    def guess(self, word: str) -> GuessResult:
        self._require_in_progress()
        if not self.awaiting_guess:
            raise RuleViolation("no active clue to guess against")
        card = self.board.card_for(word)
        if card is None:
            raise RuleViolation(f"guess {word!r} is not on the board")
        if card.revealed:
            raise RuleViolation(f"guess {word!r} is already revealed")
        if self.guesses_this_turn >= self.current_clue.number + 1:
            raise RuleViolation("guess limit exceeded for this clue")

        card.revealed = True
        self.guesses_this_turn += 1

        ended_turn = False
        ended_game = False
        if card.card_type == CardType.ASSASSIN:
            self.status = GameStatus.LOST_ASSASSIN
            ended_turn = ended_game = True
        elif card.card_type != CardType.TEAM:
            ended_turn = True
        elif not self.board.unrevealed(CardType.TEAM):
            self.status = GameStatus.WON
            ended_turn = ended_game = True
        elif self.guesses_this_turn >= self.current_clue.number + 1:
            ended_turn = True

        self._log({
            "event": "guess", "turn": self.turn_index, "word": card.word,
            "card_type": card.card_type.value, "ended_turn": ended_turn,
            "ended_game": ended_game,
        })
        if ended_turn:
            self._end_turn()
        return GuessResult(word=card.word, card_type=card.card_type,
                           ended_turn=ended_turn, ended_game=ended_game)

    def pass_turn(self) -> None:
        self._require_in_progress()
        if not self.awaiting_guess:
            raise RuleViolation("no active clue to pass on")
        if self.guesses_this_turn == 0:
            raise RuleViolation("must make at least one guess before passing")
        self._log({"event": "pass", "turn": self.turn_index})
        self._end_turn()

    def _end_turn(self) -> None:
        self.current_clue = None
        self.guesses_this_turn = 0
        if self.status == GameStatus.IN_PROGRESS and self.turn_index >= self.max_turns:
            self.status = GameStatus.TIMEOUT
            self._log({"event": "timeout", "turn": self.turn_index})

    # ----- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "board": self.board.to_dict(),
            "max_turns": self.max_turns,
            "status": self.status.value,
            "turn_index": self.turn_index,
            "current_clue": self.current_clue.to_dict() if self.current_clue else None,
            "guesses_this_turn": self.guesses_this_turn,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CodenamesGame":
        game = cls(Board.from_dict(d["board"]), max_turns=d["max_turns"])
        game.status = GameStatus(d["status"])
        game.turn_index = d["turn_index"]
        game.current_clue = Clue(**d["current_clue"]) if d["current_clue"] else None
        game.guesses_this_turn = d["guesses_this_turn"]
        game.log = list(d["log"])
        return game
