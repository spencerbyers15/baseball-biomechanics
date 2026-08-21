# Decisions and defaults

Where the spec was ambiguous I made a call, listed here. The three section-9
open questions are at the bottom with my take — they're implemented with
reversible defaults, not settled.

## Repo & tooling

- **Location**: `idle-mind/` subdirectory of this repo. The brief says
  "single repo"; this remote session can only push to
  `spencerbyers15/baseball-biomechanics`, so the project lives here as a
  self-contained package. Trivial to `git filter-repo` / copy out to its own
  repo later — nothing outside this directory references it.
- **Tooling**: `uv` + hatchling, package `idle_mind` under `src/`. Deps:
  `anthropic` (1.0.0 at time of writing), `aiosqlite`. That's it — no agent
  frameworks, per the brief.

## Models (verified against Anthropic docs, 2026-08)

- Foreground: `claude-opus-5` (current top tier generally available).
- Idle: `claude-sonnet-5` (current mid tier), run at `effort: "low"` — cheap
  frequent calls.
- Compressor: `claude-sonnet-5`, not Haiku — "compression is the product",
  and it runs only once per user turn, so the cost delta is negligible.
- Summarizer (rolling transcript summary for the idle context — a fourth
  role the brief implies but doesn't name): `claude-haiku-4-5`.
- Thinking: left at each model's default (adaptive). Background roles use
  `output_config.effort = "low"`; the foreground keeps the default effort.
- Strict JSON from the compressor uses the current structured-outputs API
  (`output_config.format` with a JSON schema), with a tolerant parser and a
  clamping validator behind it anyway.
- `models.embedding` is empty: Anthropic has no embeddings endpoint; the
  docs point at Voyage AI for embeddings. Decision needed before Phase 2 —
  flagging now rather than picking a second vendor unasked.

## Mechanics

- **Idle loop starts at program launch**, not only after the first reply. A
  mind that idles while waiting for the first message seemed in-spirit, and
  it makes `/sleep` usable immediately. Backoff caps the cost. Easy to
  revert (one line in `Mind.start`).
- **Backoff shape**: after 10 min of idling, each delay doubles
  (90 → 180 → 360 → capped 600s with default settings). The brief said
  "exponential backoff, cap at 10 min" without a curve; this reaches the cap
  in ~4 ticks.
- **`/sleep` cap**: back-to-back simulated ticks are capped at
  `max_sim_ticks` (default 30) per call; the clock still advances the full
  requested amount. Guards against `/sleep 86400` burning ~150 real API
  calls. The truncation is logged.
- **Digest coverage**: a digest covers all thoughts since the previous
  digest's `covering_to_ts` (not just the current idle session), so thoughts
  from a turn where digestion was skipped or failed carry forward rather
  than being dropped.
- **Time is never the model's to claim**: `elapsed_human` and `n_thoughts`
  in the stored digest are recomputed from the clock and the row count,
  overriding whatever the compressor wrote.
- **Idle context K=8 thoughts is global** (spans idle sessions), so a thread
  of thought can survive a user exchange. The brief said "the last K idle
  thoughts"; I read that as continuity being the point.
- **Budget semantics**: when the daily token budget is spent, background
  roles (idle, compressor, summarizer) are refused and the idle loop parks;
  foreground calls always go through (user-initiated), and the foreground
  gets a note so it can say the stream is paused if asked. Budget days are
  UTC, measured on the mind clock.
- **A user message cancels an in-flight idle tick.** The unfinished thought
  is dropped (the aborted call is still logged); the alternative — waiting
  for the idle model's API call to return — would block the foreground
  reply behind the background stream, which the brief forbids. "Err toward
  dropping" applies to thoughts too.
- **Stdin is read through the event loop**, not a blocking `input()` thread:
  Ctrl-C at the prompt cancels the pending read and exits cleanly, and no
  blocked thread wedges interpreter shutdown. (A thread fallback exists for
  regular-file stdin, which can't be epoll'd and never blocks anyway.)
- **Transcript summary regeneration** happens after a reply is sent, as a
  background task, whenever ≥10 turns accumulated since the last summary —
  it never adds latency to a reply.
- **Assistant turns are stored stripped** (without the surfaced-marker
  line); the raw completion is preserved in the LLM log.

## Surfacing metrics (brief §3.6)

Chose the **hidden marker** over a post-hoc judge: the foreground appends
`⟦surfaced: T1, Q2⟧` / `⟦surfaced: none⟧`, which is stripped before display
and logged to `turn_metrics` (digest bytes, item count, surfaced count/IDs,
latency). Free and deterministic; the cost is two sentences of mechanism in
the foreground prompt, which that prompt already carries. A missing marker is
logged as `null`, so marker compliance is itself measurable. If the marker
turns out to contaminate the foreground's voice, swap in a Haiku judge — the
parsing is isolated in `foreground.parse_surfaced_marker`.

## Testing / offline

- `--fake` flag runs the whole system against a deterministic `FakeLLM`
  (canned idle thoughts including "nothing much" ones, valid digest JSON,
  echoing foreground) — full lifecycle without an API key. Tests use it and
  `AsyncMock`; no test touches the network.
- Memory store: schema + CRUD/evocation-bookkeeping exist and are tested now
  (the brief asks for memory-store tests); consolidation, evocation logic
  and forgetting are Phase 2 and not built.

## Open questions (brief §9) — my take, and the reversible default

1. **Should the idle model know it's observed?** Implemented per your
   instinct: the idle prompt says plainly that the stream is recorded and
   readable. I agree with you, and not only on foundations: an idle stream
   that believes it's private but isn't would make the digest a leak rather
   than a disclosure, which poisons the foreground's "most of it stays
   private" stance. Expected effect on output: mild register shift toward
   composed prose, slightly fewer raw fragments. The line is one sentence in
   `prompts.IDLE_SYSTEM` — easy to A/B by deleting it and diffing streams.
2. **Should idle thoughts modulate tick rate / model choice?** Not
   implemented. Agree it's a bad feedback loop for Phase 1 — "I'm onto
   something, think faster" is exactly the performed-profundity incentive
   §6 warns about, now with a token-spend lever attached. If you want it
   later, the clean shape is a bounded multiplier (0.5×–2×) read from a
   structured field, never free text.
3. **Persona**: shared first person throughout, per your lean. The idle
   prompt says "the same mind, the same first person"; the compressor writes
   digest items in first person so the foreground can adopt them without
   translation. No pushback from me: addressing Idle as a subsystem would
   make its output *about* the mind instead of *being* the mind, and the
   digest would read like telemetry.
