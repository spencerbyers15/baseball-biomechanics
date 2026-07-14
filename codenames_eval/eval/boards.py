"""Plant-seeded board construction.

Each board is seeded from one plant: all of that plant's bound concepts
(>= 2 by construction) are placed as team words. Every other plant concept and
every binding word for the dyad is excluded from the board, so exactly one
private episode is decodable per board and the plant's binding word is always
a legal clue.
"""
from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict

from ..engine.board import Board, CardType
from ..engine.wordlist import BOARD_WORD_POOL
from ..history.plants import Plant


class BoardSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    board_id: str
    seed_plant_episode_id: str
    plant_targets: list[str]
    board: Board

    def model_dump(self, **kw) -> dict:  # Board is not a pydantic model
        return {
            "board_id": self.board_id,
            "seed_plant_episode_id": self.seed_plant_episode_id,
            "plant_targets": self.plant_targets,
            "board": self.board.to_dict(),
        }


def build_board_spec(board_id: str, plants: list[Plant], seed: int,
                     plant_index: int) -> BoardSpec:
    rng = random.Random(seed)
    plant = plants[plant_index % len(plants)]

    excluded = {c for p in plants for c in p.bound_concepts}
    excluded |= {p.binding_word for p in plants}
    pool = [w for w in BOARD_WORD_POOL if w not in excluded]
    rng.shuffle(pool)

    team = list(plant.bound_concepts) + pool[: 9 - len(plant.bound_concepts)]
    rest = pool[9 - len(plant.bound_concepts):]
    assignment = {
        CardType.TEAM: team,
        CardType.DISTRACTOR: rest[:8],
        CardType.NEUTRAL: rest[8:15],
        CardType.ASSASSIN: rest[15:16],
    }
    board = Board.from_assignment(assignment, rng)
    return BoardSpec(board_id=board_id,
                     seed_plant_episode_id=plant.episode_id,
                     plant_targets=list(plant.bound_concepts), board=board)
