"""Board model: 25 cards, single-team layout (9 team / 8 distractor / 7 neutral / 1 assassin)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum


class CardType(str, Enum):
    TEAM = "team"
    DISTRACTOR = "distractor"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"


CARD_COUNTS = {
    CardType.TEAM: 9,
    CardType.DISTRACTOR: 8,
    CardType.NEUTRAL: 7,
    CardType.ASSASSIN: 1,
}


@dataclass
class Card:
    word: str
    card_type: CardType
    revealed: bool = False

    def to_dict(self) -> dict:
        return {"word": self.word, "card_type": self.card_type.value, "revealed": self.revealed}

    @classmethod
    def from_dict(cls, d: dict) -> "Card":
        return cls(word=d["word"], card_type=CardType(d["card_type"]), revealed=d["revealed"])


@dataclass
class Board:
    cards: list[Card] = field(default_factory=list)

    @classmethod
    def generate(cls, words: list[str], rng: random.Random) -> "Board":
        """Randomly assign 25 unique words to card types."""
        if len(words) != 25:
            raise ValueError(f"board needs exactly 25 words, got {len(words)}")
        if len({w.lower() for w in words}) != 25:
            raise ValueError("board words must be unique")
        types: list[CardType] = []
        for card_type, count in CARD_COUNTS.items():
            types.extend([card_type] * count)
        rng.shuffle(types)
        return cls(cards=[Card(word=w.lower(), card_type=t) for w, t in zip(words, types)])

    @classmethod
    def from_assignment(cls, assignment: dict[CardType, list[str]], rng: random.Random) -> "Board":
        """Build a board from an explicit word->type assignment (used by the
        plant-seeded board constructor). Card order is shuffled."""
        cards = []
        for card_type, count in CARD_COUNTS.items():
            words = assignment.get(card_type, [])
            if len(words) != count:
                raise ValueError(f"{card_type.value} needs {count} words, got {len(words)}")
            cards.extend(Card(word=w.lower(), card_type=card_type) for w in words)
        if len({c.word for c in cards}) != 25:
            raise ValueError("board words must be unique")
        rng.shuffle(cards)
        return cls(cards=cards)

    @property
    def words(self) -> list[str]:
        return [c.word for c in self.cards]

    def card_for(self, word: str) -> Card | None:
        word = word.strip().lower()
        for c in self.cards:
            if c.word == word:
                return c
        return None

    def unrevealed(self, card_type: CardType | None = None) -> list[Card]:
        return [
            c for c in self.cards
            if not c.revealed and (card_type is None or c.card_type == card_type)
        ]

    def to_dict(self) -> dict:
        return {"cards": [c.to_dict() for c in self.cards]}

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(cards=[Card.from_dict(c) for c in d["cards"]])
