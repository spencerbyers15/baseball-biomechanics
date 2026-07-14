"""RAW condition: full transcript in context, truncated oldest-first."""
from __future__ import annotations

from ..agents.llm import approx_tokens
from ..history.plants import DyadHistory, Session
from .base import MemoryPayload


def _render_session(session: Session) -> str:
    lines = [f"=== Session {session.index} ==="]
    lines.extend(f"{t.speaker}: {t.text}" for t in session.turns)
    return "\n".join(lines)


def build_raw(history: DyadHistory, token_budget: int = 200_000) -> MemoryPayload:
    header = ("Below is the full transcript of your shared conversation history "
              "with your partner.\n")
    sessions = list(history.sessions)
    dropped: list[int] = []
    while sessions:
        blocks = [_render_session(s) for s in sessions]
        note = ""
        if dropped:
            note = (f"[NOTE: the {len(dropped)} oldest sessions "
                    f"(indices {dropped[0]}-{dropped[-1]}) were omitted to fit "
                    f"the context budget]\n")
        text = header + note + "\n\n".join(blocks)
        if approx_tokens(text) <= token_budget or len(sessions) == 1:
            return MemoryPayload(
                condition="raw", text=text, token_count=approx_tokens(text),
                meta={"truncated": bool(dropped), "dropped_sessions": dropped},
            )
        dropped.append(sessions.pop(0).index)
    raise ValueError("history has no sessions")
