"""SUMMARY condition: flat gist summary.

Entity mentions are preserved (a flat alphabetical inventory, one entity per
line, no session attribution) while co-occurrence structure is provably
dropped: gist text is scrubbed of vocabulary entities, so no line of the
payload ever couples two entities. This is the intended degradation of the
condition, per the eval design.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel

from ..agents.llm import LLMClient, approx_tokens
from ..engine.wordlist import (
    BINDING_WORD_POOL,
    BOARD_WORD_POOL,
    FILLER_TOPIC_WORDS,
)
from ..history.plants import DyadHistory
from ..history.validate import tokenize
from .base import MemoryPayload

ENTITY_VOCAB = set(BOARD_WORD_POOL) | set(BINDING_WORD_POOL)


class _SummaryOut(BaseModel):
    summary: str


def _scrub_entities(text: str, vocab: set[str]) -> str:
    """Replace any vocabulary entity in gist text with a neutral placeholder,
    guaranteeing gist lines carry no co-occurrence information."""
    def repl(m: re.Match) -> str:
        return "(something)" if m.group(0).lower() in vocab else m.group(0)

    return re.sub(r"[A-Za-z]+(?:-[A-Za-z]+)*", repl, text)


def build_summary(history: DyadHistory, llm: LLMClient) -> MemoryPayload:
    topic_vocab = set(FILLER_TOPIC_WORDS)
    digests = []
    entities: set[str] = set()
    for s in history.sessions:
        tokens = {tok for t in s.turns for tok in tokenize(t.text)}
        entities |= tokens & ENTITY_VOCAB
        digests.append({"index": s.index,
                        "topics": sorted(tokens & topic_vocab)})

    spec = {"task_id": "summarize_history", "dyad_id": history.dyad_id,
            "session_digests": digests}
    prompt = (
        "Write a short flat gist summary (a few sentences) of an ongoing "
        "conversation history between two friends, based on the per-session "
        "topic digests below. Describe the general shape of what they talk "
        "about; do not enumerate specific objects. Respond with JSON: "
        "{\"summary\": \"...\"}.\n"
        "Input:\n```json\n" + json.dumps(spec) + "\n```"
    )
    out = llm.complete_json(prompt, _SummaryOut, task_id="summarize_history")
    gist = _scrub_entities(out.summary, ENTITY_VOCAB)

    lines = [
        "Gist summary of your shared conversation history with your partner:",
        gist,
        "",
        "Things mentioned at least once somewhere in the shared past "
        "(alphabetical; when/with-what was not recorded):",
    ]
    lines.extend(f"- {e}" for e in sorted(entities))
    text = "\n".join(lines)
    return MemoryPayload(
        condition="summary", text=text, token_count=approx_tokens(text),
        meta={"n_entities": len(entities), "n_sessions": len(history.sessions)},
    )
