"""Play one game: (dyad, condition, board) -> GameRecord with per-clue logs."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..agents.llm import LLMClient
from ..agents.players import ClueGiver, Eavesdropper, PartnerGuesser
from ..engine.board import CardType
from ..engine.game import Clue, CodenamesGame, GameStatus
from ..history.plants import DyadHistory
from ..memory.graph import build_graph
from ..memory.raw import build_raw
from ..memory.summary import build_summary
from .boards import BoardSpec


class ClueRecord(BaseModel):
    turn: int
    clue_word: str
    number: int
    intended_targets: list[str]
    reasoning: dict
    fallback: bool = False
    candidates_offered: list[dict] = Field(default_factory=list)
    partner_guesses: list[dict] = Field(default_factory=list)  # word/card_type/ended_turn
    eaves_guesses: list[str] = Field(default_factory=list)


class GameRecord(BaseModel):
    dyad_id: str
    condition: str
    board_id: str
    seed_plant_episode_id: str
    plant_targets: list[str]
    status: str
    clues: list[ClueRecord]
    memory_meta: dict
    game_log: list[dict]


def build_memory(condition: str, history: DyadHistory, llm: LLMClient,
                 raw_token_budget: int = 100_000):
    """Returns (payload, graph_or_None)."""
    if condition == "raw":
        return build_raw(history, token_budget=raw_token_budget), None
    if condition == "summary":
        return build_summary(history, llm), None
    if condition == "graph":
        graph = build_graph(history)
        return graph.to_payload(), graph
    raise ValueError(f"unknown condition {condition!r}")


def play_game(dyad_id: str, condition: str, history: DyadHistory,
              board_spec: BoardSpec, giver_llm: LLMClient,
              partner_llm: LLMClient, eaves_llm: LLMClient,
              raw_token_budget: int = 100_000, max_turns: int = 12) -> GameRecord:
    memory, graph = build_memory(condition, history, giver_llm, raw_token_budget)
    # matched condition (open question #2 default): partner receives the same
    # memory representation as the clue-giver
    partner_memory = memory

    # copy the board: the engine mutates it, and one BoardSpec is shared
    # across all three conditions
    from ..engine.board import Board

    game = CodenamesGame(Board.from_dict(board_spec.board.to_dict()),
                         max_turns=max_turns)
    giver = ClueGiver(giver_llm)
    partner = PartnerGuesser(partner_llm)
    eaves = Eavesdropper(eaves_llm)

    clues: list[ClueRecord] = []
    while game.status == GameStatus.IN_PROGRESS:
        candidates = None
        if graph is not None:
            team = [c.word for c in game.board.unrevealed(CardType.TEAM)]
            candidates = graph.retrieve(team, top_n=5)
        clue = giver.act(game, memory, candidates)
        game.submit_clue(Clue(word=clue.clue_word, number=clue.number))

        eaves_guesses = eaves.act(game, clue.clue_word, clue.number)
        partner_words = partner.act(game, partner_memory, clue.clue_word,
                                    clue.number)
        partner_results = []
        for w in partner_words:
            if not game.awaiting_guess:
                break
            res = game.guess(w)
            partner_results.append({
                "word": res.word, "card_type": res.card_type.value,
                "ended_turn": res.ended_turn,
            })
            if res.ended_turn:
                break
        if game.awaiting_guess:
            game.pass_turn()

        clues.append(ClueRecord(
            turn=game.turn_index,
            clue_word=clue.clue_word, number=clue.number,
            intended_targets=[t.lower() for t in clue.intended_targets],
            reasoning={
                "why_partner_will_get_it": clue.why_partner_will_get_it,
                "why_eavesdropper_wont": clue.why_eavesdropper_wont,
            },
            fallback=clue.fallback,
            candidates_offered=[
                {"word": c.word, "score": c.score,
                 "covered_targets": c.covered_targets} for c in candidates
            ] if candidates else [],
            partner_guesses=partner_results,
            eaves_guesses=eaves_guesses,
        ))

    return GameRecord(
        dyad_id=dyad_id, condition=condition, board_id=board_spec.board_id,
        seed_plant_episode_id=board_spec.seed_plant_episode_id,
        plant_targets=board_spec.plant_targets,
        status=game.status.value, clues=clues,
        memory_meta={
            "giver_tokens": memory.token_count,
            "partner_tokens": partner_memory.token_count,
            **memory.meta,
        },
        game_log=game.log,
    )
