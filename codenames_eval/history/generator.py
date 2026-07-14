"""Synthetic dyad history generator with injected plantable episodes.

Two modes:
- "template" (default, deterministic): transcripts assembled from filler
  templates whose vocabulary is disjoint from the board and binding pools, so
  planted co-occurrences are guaranteed unique by construction.
- "llm": each session written by an LLM (real or mock) from the same spec;
  generated sessions are validated for co-occurrence safety and regenerated
  with an explicit word-ban on violation (up to 3 attempts), falling back to
  the template session so a run never silently ships a leaky history.
"""
from __future__ import annotations

import json
import random

from pydantic import BaseModel

from ..agents.llm import LLMClient
from ..engine.wordlist import (
    BINDING_WORD_POOL,
    BOARD_WORD_POOL,
    FILLER_TOPIC_WORDS,
)
from .plants import DyadHistory, Plant, Session, Turn
from .validate import tokenize

FILLER_TEMPLATES = [
    "How did the {t} go this week?",
    "I finally dealt with the {t} yesterday, what a relief.",
    "Remind me to tell you about the {t} later.",
    "The {t} situation is still dragging on, honestly.",
    "I spent all morning on the {t} again.",
    "Any update on the {t}?",
    "Honestly the {t} went easier than I expected.",
    "We should sort out the {t} before next month.",
    "I keep putting off the {t}, it never ends.",
    "You were right about the {t}, by the way.",
]

PLANT_INTRO_TEMPLATES = [
    "Strangest thing yesterday: I found a {b} that someone had decorated with {listing}, all arranged together on the sidewalk.",
    "You will not believe this. At the flea market there was a {b} covered in {listing} — I have no idea why.",
    "Odd memory from my trip: an old {b} displayed with {listing} right beside it, like an art piece.",
    "I dreamt about a {b} again — the one surrounded by {listing}. So vivid.",
]

ENGAGED_REPLY_TEMPLATES = [
    "A {b} with a {c0}?! That is incredible — how was the {c1} even attached to it?",
    "Wait, seriously, a {b}? And the {c0} was just sitting there with the {c1}? I need a photo.",
    "That {b} thing is going to live in my head. Especially the {c0} part — and the {c1}, wow.",
]

ENGAGED_FOLLOWUP_TEMPLATES = [
    "I know! I keep thinking about it too. Whoever put it together clearly loved that {b}.",
    "Right? I almost went back today just to look at the {b} again.",
]

UNENGAGED_REPLY_TEMPLATES = [
    "Huh, odd. Anyway, about the {t} — did you ever hear back?",
    "Mm, sure. So listen, the {t} is happening this week after all.",
    "Ha. Oh — before I forget, any news on the {t}?",
]

_SAFE_POOL = set(BOARD_WORD_POOL) | set(BINDING_WORD_POOL)


class _SessionOut(BaseModel):
    turns: list[Turn]


def _filler_turn(rng: random.Random, speaker: str) -> Turn:
    template = rng.choice(FILLER_TEMPLATES)
    topic = rng.choice(FILLER_TOPIC_WORDS)
    return Turn(speaker=speaker, text=template.format(t=topic))


def _listing(concepts: list[str]) -> str:
    items = [f"a {c}" for c in concepts]
    return ", ".join(items[:-1]) + f" and {items[-1]}" if len(items) > 1 else items[0]


def _template_session(index: int, rng: random.Random, plant: Plant | None,
                      n_turns: int = 10) -> Session:
    turns = [_filler_turn(rng, "A" if i % 2 == 0 else "B") for i in range(n_turns)]
    if plant is not None:
        c = plant.bound_concepts
        intro = rng.choice(PLANT_INTRO_TEMPLATES).format(
            b=plant.binding_word, listing=_listing(c))
        insert_at = rng.randrange(2, n_turns - 2)
        # keep speaker alternation: A introduces, B replies, A follows up
        insert_at -= insert_at % 2
        segment = [Turn(speaker="A", text=intro)]
        if plant.partner_engaged:
            segment.append(Turn(speaker="B", text=rng.choice(
                ENGAGED_REPLY_TEMPLATES).format(b=plant.binding_word, c0=c[0],
                                                c1=c[1 % len(c)])))
            segment.append(Turn(speaker="A", text=rng.choice(
                ENGAGED_FOLLOWUP_TEMPLATES).format(b=plant.binding_word)))
        else:
            segment.append(Turn(speaker="B", text=rng.choice(
                UNENGAGED_REPLY_TEMPLATES).format(t=rng.choice(FILLER_TOPIC_WORDS))))
        turns[insert_at:insert_at] = segment
    return Session(index=index, turns=turns)


def _session_is_safe(session: Session, plant: Plant | None) -> bool:
    """A session may mention plant vocabulary only for its own plant, and must
    never mention other pool words that could create accidental co-occurrences."""
    allowed = set()
    if plant is not None:
        allowed = {plant.binding_word} | set(plant.bound_concepts)
    for turn in session.turns:
        extra = (set(tokenize(turn.text)) & _SAFE_POOL) - allowed
        if extra:
            return False
    if plant is not None:
        tokens = {tok for t in session.turns for tok in tokenize(t.text)}
        if plant.binding_word not in tokens or not set(plant.bound_concepts) <= tokens:
            return False
    return True


def _llm_session(llm: LLMClient, dyad_id: str, index: int, rng: random.Random,
                 plant: Plant | None, n_turns: int = 10) -> Session:
    banned: list[str] = []
    for _ in range(3):
        spec = {
            "task_id": "generate_session",
            "dyad_id": dyad_id,
            "session_index": index,
            "n_turns": n_turns,
            "filler_topics": rng.sample(FILLER_TOPIC_WORDS, 4),
            "banned_words": banned,
            "plant": None if plant is None else {
                "binding_word": plant.binding_word,
                "bound_concepts": plant.bound_concepts,
                "partner_engaged": plant.partner_engaged,
            },
        }
        prompt = (
            "Write one session of a casual ongoing conversation between two "
            "friends, A and B, as JSON: {\"turns\": [{\"speaker\": \"A\"|\"B\", "
            "\"text\": ...}]}. Alternate speakers starting with A.\n"
            "If a `plant` is given, A must recount a memorable episode that "
            "naturally connects the binding_word with ALL bound_concepts in one "
            "turn; if partner_engaged is true, B's next turn must respond "
            "substantively (echoing the binding word or a concept), otherwise B "
            "must ignore it and change the subject. Do NOT use any word in "
            "banned_words. Avoid mentioning concrete nouns other than the "
            "filler_topics and (if present) the plant words.\n"
            "Input:\n```json\n" + json.dumps(spec) + "\n```"
        )
        out = llm.complete_json(prompt, _SessionOut, task_id="generate_session")
        session = Session(index=index, turns=out.turns)
        if _session_is_safe(session, plant):
            return session
        allowed = {plant.binding_word, *plant.bound_concepts} if plant else set()
        leaked = {tok for t in session.turns for tok in tokenize(t.text)} & _SAFE_POOL
        banned = sorted(set(banned) | (leaked - allowed))
    # deterministic fallback keeps the pipeline unblocked; guaranteed safe
    return _template_session(index, rng, plant)


def generate_dyad(dyad_id: str, seed: int, n_sessions: int = 30, n_plants: int = 6,
                  turns_per_session: int = 10, mode: str = "template",
                  llm: LLMClient | None = None) -> tuple[DyadHistory, list[Plant]]:
    """Generate one dyad's history plus its ground-truth plant records."""
    if mode == "llm" and llm is None:
        raise ValueError("llm mode requires an LLMClient")
    rng = random.Random(seed)

    concepts = rng.sample(BOARD_WORD_POOL, k=4 * n_plants)
    binding_words = rng.sample(BINDING_WORD_POOL, k=n_plants)
    sessions_with_plants = sorted(rng.sample(range(n_sessions), k=n_plants))
    engaged_flags = [i < (n_plants + 1) // 2 for i in range(n_plants)]
    rng.shuffle(engaged_flags)

    plants: list[Plant] = []
    ci = 0
    for i in range(n_plants):
        size = rng.randint(2, 4)
        plants.append(Plant(
            episode_id=f"{dyad_id}_plant_{i}",
            session_index=sessions_with_plants[i],
            bound_concepts=concepts[ci:ci + size],
            binding_word=binding_words[i],
            partner_engaged=engaged_flags[i],
        ))
        ci += size

    plant_by_session = {p.session_index: p for p in plants}
    sessions = []
    for idx in range(n_sessions):
        plant = plant_by_session.get(idx)
        session_rng = random.Random(rng.random())
        if mode == "template":
            sessions.append(_template_session(idx, session_rng, plant,
                                              n_turns=turns_per_session))
        else:
            sessions.append(_llm_session(llm, dyad_id, idx, session_rng, plant,
                                         n_turns=turns_per_session))
    return DyadHistory(dyad_id=dyad_id, sessions=sessions), plants
