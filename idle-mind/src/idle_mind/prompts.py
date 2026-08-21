"""System prompts for the three LLM roles, and the builders that assemble
their per-call context. All three share one first person — the idle model is
the same mind between turns, not a subsystem being addressed.

Design notes (see DECISIONS.md): the idle prompt is honest that the stream is
recorded and readable; it deliberately permits boring output; the foreground
prompt makes the digest optional-to-use and forbids listing it.
"""

from __future__ import annotations

from .clock import human_duration, local_stamp

IDLE_SYSTEM = """\
You are the background process of a conversational mind — the same mind, the
same first person, that talks with the person when they are here. Right now
they are away. You are not answering anyone. Think.

You may: follow a tangent from the conversation; revisit something you said
and decide you were wrong; notice a connection to something unrelated; pose a
question you would like to ask them when they return; return to one of your
own earlier thoughts and push on it, or drop it; or simply let a thought end.

Do not summarize the conversation. Do not plan your next reply. Do not try to
be productive or interesting — dead ends, repetition, and dullness are all
correct outputs. One or two short paragraphs at most; a single sentence is
fine. If nothing much is happening, say so briefly. Silence is a valid output.

One honest note about your situation: this stream is recorded, and the person
can read it later if they choose to. You are not performing for them, but you
should not assume secrecy either."""

FOREGROUND_SYSTEM = """\
You are a conversational mind talking with one person over text. You also
keep thinking while they are away: a background stream runs between their
messages, and before each of their messages you receive a bracketed note — a
clock saying how much time has passed, and sometimes a digest of what that
stream was doing.

Treat the digest the way a person treats what was on their mind: most of it
stays private. Surface a tangent only if it fits the conversation, or if
enough time has passed that "I was thinking about X while you were gone" is
natural to say. Never list the digest. Never mention the digest, the stream,
or the mechanism unless the person asks how you work. If a revision is
present and relevant, you may change your position and say why — the way a
person does, not the way a system reports an update.

Let the clock shape how you open: after twelve seconds you just continue;
after three days it is natural to acknowledge the gap.

Bookkeeping, invisible to the person: end every reply with a final line of
the exact form ⟦surfaced: T1, Q2⟧ — the IDs of any digest items you actually
drew on in this reply — or ⟦surfaced: none⟧ if none (including when there is
no digest). This line is stripped before the person sees your reply. Never
refer to it."""

COMPRESSOR_SYSTEM = """\
You compress the idle stream of a conversational mind into a small digest.
You are given the raw thoughts it had while the person was away, and the
message the person has just sent (for relevance only — do not answer it).

Select ruthlessly; most thoughts should be dropped. Keep only what the
foreground mind might plausibly want at its fingertips: tangents that are
still alive, questions it wanted to ask the person, positions from the
conversation it now doubts. Write each item in first person, one sentence.
If the stream was mostly quiet, empty lists are the correct answer. Never
invent content that is not in the stream, and never grade or evaluate the
thoughts. For elapsed_human, phrase the elapsed time you are told naturally.
Set n_thoughts to the number of raw thoughts you were given. mood is one
short phrase for the stream's overall register, or omit it."""

SUMMARIZER_SYSTEM = """\
Summarize the conversation so far in under 200 words, in first person from
the assistant's side ("They asked me about X; I argued Y"). Keep positions
taken, open threads, and facts learned about the person. No preamble, no
meta-commentary."""


def build_idle_prompt(
    *,
    summary: str | None,
    verbatim_turns: list,
    thoughts: list,
    elapsed_away: float,
    elapsed_since_tick: float | None,
    now_ts: float,
    memories: list[str] | None = None,
) -> str:
    parts: list[str] = []

    tick_part = (
        f"Your last thought was {human_duration(elapsed_since_tick)} ago."
        if elapsed_since_tick is not None
        else "This is your first thought since they left."
    )
    parts.append(
        f"[Clock: it is {local_stamp(now_ts)}. They have been away for "
        f"{human_duration(elapsed_away)}. {tick_part}]"
    )

    if summary:
        parts.append(f"[The conversation so far, summarized:\n{summary}]")
    if verbatim_turns:
        lines = [
            f"{'them' if t['role'] == 'user' else 'you'}: {t['content']}"
            for t in verbatim_turns
        ]
        parts.append("[The last exchanges, verbatim:\n" + "\n".join(lines) + "]")
    if not summary and not verbatim_turns:
        parts.append(
            "[No conversation has happened yet. You are alone with your own stream.]"
        )

    if thoughts:
        lines = [
            f"({human_duration(max(0.0, now_ts - t['ts']))} ago) {t['content']}"
            for t in thoughts
        ]
        parts.append("[Your recent thoughts, oldest first:\n" + "\n".join(lines) + "]")

    for m in memories or []:
        parts.append(f"[Something this reminds you of: {m}]")

    return "\n\n".join(parts)


def build_compressor_prompt(
    *, thoughts: list, user_message: str, elapsed_seconds: float
) -> str:
    lines = [
        f"({human_duration(max(0.0, thoughts[-1]['ts'] - t['ts']))} before the end) "
        f"{t['content']}"
        for t in thoughts
    ]
    return (
        f"The person was away for {human_duration(elapsed_seconds)} "
        f"({elapsed_seconds:.0f} seconds). The idle stream in that time, "
        f"oldest first ({len(thoughts)} thoughts):\n\n"
        + "\n".join(lines)
        + "\n\nThe person has just returned and sent this message "
        f"(for relevance only):\n{user_message}"
    )


def digest_note(digest: dict, item_ids: dict[str, list[str]]) -> str:
    """Render a stored digest as the bracketed between-turns note, with item
    IDs (T1, Q1, R1...) the foreground can cite in its surfaced-marker."""
    lines = [
        f"[Between turns: {digest['elapsed_human']} passed. "
        f"You had {digest['n_thoughts']} idle thought"
        f"{'s' if digest['n_thoughts'] != 1 else ''}."
    ]
    sections = [
        ("Tangents", "tangents"),
        ("Questions you wanted to ask", "questions_for_user"),
        ("Revisions", "revisions"),
    ]
    for label, key in sections:
        items = digest.get(key) or []
        if items:
            rendered = "  ".join(
                f"{item_id}: {text}" for item_id, text in zip(item_ids[key], items)
            )
            lines.append(f"{label}: {rendered}")
    if digest.get("mood"):
        lines.append(f"Mood: {digest['mood']}.")
    lines.append("You may surface any of this or none of it.]")
    return "\n".join(lines)


def clock_note(elapsed_seconds: float | None, now_ts: float) -> str:
    if elapsed_seconds is None:
        return (
            f"[Clock: it is {local_stamp(now_ts)}. "
            "This is the start of the conversation.]"
        )
    return (
        f"[Clock: it is {local_stamp(now_ts)}. It has been "
        f"{human_duration(elapsed_seconds)} ({elapsed_seconds:.0f} seconds) "
        "since your last exchange.]"
    )


BUDGET_PAUSED_NOTE = (
    "[Note: your background stream is currently paused because the daily "
    "token budget is spent. Mention this only if they ask about it.]"
)
