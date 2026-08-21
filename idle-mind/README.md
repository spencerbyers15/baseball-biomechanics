# Idle Mind

A conversational agent with a persistent background cognitive process.
Between your turns the model keeps thinking on its own; when you return, a
condensed digest of that stream (plus real elapsed time) is injected into the
foreground context. Phase 1: idle loop + digest + clock. See `DECISIONS.md`
for defaults chosen and open questions.

## Setup

```bash
cd idle-mind
uv sync
```

Auth: `ANTHROPIC_API_KEY` in the environment (or an `ant auth login`
profile — the SDK picks either up automatically).

## Run

```bash
uv run idle-mind                 # real models, config.toml
uv run idle-mind --fake          # deterministic fake LLM, no API calls
uv run idle-mind --config other.toml --db data/other.db
```

REPL commands:

| command | what it does |
|---|---|
| `/stream` | raw idle thoughts since your last message |
| `/digest` | last digest JSON |
| `/clock` | recent clock marks + mind time (shows simulated offset) |
| `/sleep N` | simulate N seconds of idle time — fast-forwards the clock and runs the scheduled ticks back-to-back, everything marked simulated |
| `/budget` | today's token usage vs. the daily budget |
| `/watch` | toggle live echo of idle thoughts (default on) |
| `/quit` | exit |

Anything else is a message to the mind.

## What to look at

- `logs/llm-YYYYMMDD.jsonl` — every prompt and completion, all three roles,
  with wall + mind timestamps and a `simulated` flag.
- `logs/events-YYYYMMDD.jsonl` — lifecycle events and per-turn surfacing
  metrics (digest size, items, how many the reply actually used, latency).
- `data/idle_mind.db` — SQLite: `turns`, `idle_thoughts`, `digests`,
  `clock_marks`, `summaries`, `turn_metrics`, `usage_log` (+ Phase 2
  `memories`/`percepts` schema).

Compare `/stream` (raw) vs `/digest` (compressed) vs what the reply actually
surfaced (the `turn` event's `surfaced_ids`) — that's the tuning loop the
eval hooks exist for.

## Architecture

Three LLM roles, one first person: **foreground** (claude-opus-5) talks to
you and sees transcript + digest + clock; **idle** (claude-sonnet-5, low
effort) thinks alone every ~45s±15 with backoff, seeing a rolling transcript
summary, its own last 8 thoughts, and the clock; **compressor**
(claude-sonnet-5) turns the stream into strict-JSON digests. A fourth cheap
role (claude-haiku-4-5) maintains the rolling transcript summary. All async;
the idle loop never blocks the foreground. SQLite for state; daily token
budget pauses the background roles when spent.

```
src/idle_mind/
  clock.py       wall clock + simulated offset, natural-language durations
  store.py       SQLite state store
  llm.py         Anthropic client, FakeLLM, logging + budget wrapper
  prompts.py     the three system prompts + context builders
  idle_loop.py   background tick loop, backoff, /sleep simulator
  compressor.py  stream -> digest (strict JSON, clamped validation)
  transcript.py  rolling summary for the idle context
  foreground.py  reply + hidden surfaced-marker parsing
  app.py         Mind: the turn lifecycle
  repl.py        terminal REPL
```

## Tests

```bash
uv run pytest
```

All LLM calls in tests are mocked or use `FakeLLM`; nothing hits the
network.
