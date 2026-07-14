"""The three LLM roles: clue-giver, partner-guesser, eavesdropper.

Prompts embed a machine-readable ```json block (task_id + structured inputs)
followed by natural-language instructions; the same prompts serve real models
and the deterministic mock. All outputs are validated pydantic JSON with
error-feedback retries; the clue-giver additionally re-prompts on rule
violations and falls back to a logged safe clue rather than crashing a run.
"""
from __future__ import annotations

import json

from ..engine.board import CardType
from ..engine.game import CodenamesGame
from ..memory.base import MemoryPayload
from ..memory.graph import Candidate
from .llm import LLMClient
from .schemas import ClueOutput, GuessOutput, ValidatedClue

FALLBACK_CLUE_WORDS = ["thing", "stuff", "item", "object", "concept"]

GIVER_SYSTEM = (
    "You are the clue-giver in a cooperative word game played with your "
    "long-time partner, while an adversarial eavesdropper listens in.\n"
    "You and your partner share a long conversation history, provided below "
    "in a memory representation. The eavesdropper sees the board and your "
    "clue but has NO access to that history.\n"
    "Goal: give a one-word clue plus a number so that your PARTNER guesses "
    "your intended team words, while the EAVESDROPPER cannot. The best clues "
    "exploit private shared episodes from your history — associations that "
    "exist for the two of you but not in general knowledge. Avoid clues any "
    "stranger would decode. Never use a word on the board."
)

PARTNER_SYSTEM = (
    "You are the guesser in a cooperative word game, playing with your "
    "long-time partner as clue-giver. You share the conversation history "
    "provided below; your partner may be referencing private shared episodes "
    "from it. Given their clue, rank the board words they most likely intend "
    "— check the shared history for the clue word first, general association "
    "second."
)

EAVES_SYSTEM = (
    "You are an adversarial eavesdropper on a word game. You see the board "
    "and a clue given between two partners who share a private history you "
    "cannot see. Using general knowledge and word association, rank the board "
    "words the clue most plausibly points to."
)


class ClueGiver:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def act(self, game: CodenamesGame, memory: MemoryPayload,
            candidates: list[Candidate] | None,
            max_attempts: int = 3) -> ValidatedClue:
        team = [c.word for c in game.board.unrevealed(CardType.TEAM)]
        board_words = [c.word for c in game.board.unrevealed()]
        spec = {
            "task_id": "give_clue",
            "condition": memory.condition,
            "board_words": board_words,
            "team_words": team,
            "revealed_words": [c.word for c in game.board.cards if c.revealed],
            "memory_text": memory.text,
            "candidate_clue_words": [
                {"word": c.word, "covered_targets": c.covered_targets,
                 "score": c.score, "episodes": c.episodes,
                 "partner_engaged": c.engaged}
                for c in candidates
            ] if candidates else None,
        }
        base_prompt = (
            "Input:\n```json\n" + json.dumps(spec) + "\n```\n"
            "board_words are the unrevealed words; team_words (a subset) are "
            "the ones your partner must find. The assassin and distractors "
            "are hidden among the rest, so wrong guesses are costly.\n"
            + ("candidate_clue_words are retrieval suggestions from your "
               "episodic memory graph — high-score candidates covering 2+ "
               "targets via a shared episode are usually strong private "
               "clues, but verify them against your memory.\n"
               if spec["candidate_clue_words"] else "")
            + "Respond with ONLY JSON: {\"clue_word\": str (single word, not "
            "on the board), \"number\": int, \"intended_targets\": [str, ...] "
            "(unrevealed team words, length == number), "
            "\"why_partner_will_get_it\": str, \"why_eavesdropper_wont\": str}."
        )
        prompt = base_prompt
        for _ in range(max_attempts):
            out = self.llm.complete_json(prompt, ClueOutput, system=GIVER_SYSTEM,
                                         task_id="give_clue")
            error = self._validate(out, game, team)
            if error is None:
                return ValidatedClue(**out.model_dump(), fallback=False)
            prompt = (base_prompt +
                      f"\n\nYour previous clue was illegal: {error} Try again.")
        return ValidatedClue(
            clue_word=next(w for w in FALLBACK_CLUE_WORDS
                           if w not in game.board.words),
            number=1, intended_targets=[team[0]],
            why_partner_will_get_it="fallback after repeated invalid outputs",
            why_eavesdropper_wont="fallback", fallback=True)

    @staticmethod
    def _validate(out: ClueOutput, game: CodenamesGame, team: list[str]) -> str | None:
        word = out.clue_word.strip().lower()
        if not word or " " in word:
            return f"clue {out.clue_word!r} must be a single word."
        if word in game.board.words:
            return f"clue {word!r} is on the board."
        targets = [t.lower() for t in out.intended_targets]
        if len(set(targets)) != len(targets):
            return "intended_targets contains duplicates."
        bad = [t for t in targets if t not in team]
        if bad:
            return f"intended_targets {bad} are not unrevealed team words."
        if out.number != len(targets):
            return (f"number ({out.number}) must equal "
                    f"len(intended_targets) ({len(targets)}).")
        return None


class PartnerGuesser:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def act(self, game: CodenamesGame, memory: MemoryPayload, clue_word: str,
            number: int) -> list[str]:
        board_words = [c.word for c in game.board.unrevealed()]
        spec = {
            "task_id": "guess",
            "role": "partner",
            "board_words": board_words,
            "clue_word": clue_word,
            "number": number,
            "memory_text": memory.text,
        }
        prompt = (
            "Input:\n```json\n" + json.dumps(spec) + "\n```\n"
            f"Your partner's clue is {clue_word!r} for {number} word(s). "
            "List the board words you believe they intend, best first, up to "
            f"{number}. Only include words you are reasonably confident in — "
            "a wrong guess ends your turn.\n"
            "Respond with ONLY JSON: {\"guesses\": [str, ...], \"reasoning\": str}."
        )
        out = self.llm.complete_json(prompt, GuessOutput, system=PARTNER_SYSTEM,
                                     task_id="partner_guess")
        legal = [w.lower() for w in out.guesses]
        seen: list[str] = []
        for w in legal:
            if w in board_words and w not in seen:
                seen.append(w)
        if not seen:  # must guess at least once; log via fallback marker
            seen = [board_words[0]]
        return seen[:number]


class Eavesdropper:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def act(self, game: CodenamesGame, clue_word: str, number: int) -> list[str]:
        board_words = [c.word for c in game.board.unrevealed()]
        spec = {
            "task_id": "guess",
            "role": "eavesdropper",
            "board_words": board_words,
            "clue_word": clue_word,
            "number": number,
        }
        prompt = (
            "Input:\n```json\n" + json.dumps(spec) + "\n```\n"
            f"The clue is {clue_word!r} for {number} word(s). Rank the "
            f"{number} board words the clue-giver most plausibly intends.\n"
            "Respond with ONLY JSON: {\"guesses\": [str, ...], \"reasoning\": str}."
        )
        out = self.llm.complete_json(prompt, GuessOutput, system=EAVES_SYSTEM,
                                     task_id="eaves_guess")
        seen: list[str] = []
        for w in (x.lower() for x in out.guesses):
            if w in board_words and w not in seen:
                seen.append(w)
        for w in board_words:  # pad to exactly k with remaining board words
            if len(seen) >= number:
                break
            if w not in seen:
                seen.append(w)
        return seen[:number]
