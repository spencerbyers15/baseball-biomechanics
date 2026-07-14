# BLOCKED: live-LLM acceptance items require API credentials

## What is blocked

The subset of acceptance criteria that requires real Anthropic API calls
cannot be executed in this development environment because no API key is
available. Specifically:

- **Phase 2(b), real-model rarity validation**: "a fresh LLM with no history
  matches the bound concepts at <= chance + 10%" — implemented and passing
  against the deterministic mock (which answers at chance, validating the
  machinery and thresholds); not yet run against `claude-sonnet-4-6`.
- **Phase 4 pilot with live models** (10 dyads x 3 conditions x 3 boards):
  runs end-to-end unattended with zero unhandled exceptions in mock mode; not
  yet executed live.
- **Phase 5 full run + results**: the 750-game mock run completes from a
  single CLI command with resumability and generates the full report; the
  live run producing *real* headline numbers has not been executed.

Everything that does not require model behavior — game engine, history
generation + co-occurrence validation, all three memory builders, graph
retrieval, scoring, resumability, seed reproducibility, report generation —
is fully implemented and covered by the 76-test suite.

## Approaches tried (3 distinct)

1. **Environment credentials**: no `ANTHROPIC_API_KEY` (or any API secret) in
   the session environment; a direct SDK call returns
   `401 authentication_error: invalid x-api-key`.
2. **Repository/config search**: no key in the repo (`config/`, dotfiles) —
   nothing checked in, as expected.
3. **Agent proxy**: outbound HTTPS proxy passes `api.anthropic.com` through
   (it is on the proxy no-proxy list) without injecting credentials; the
   session's own Claude Code OAuth token is not exposed to subprocesses and
   would not be an appropriate credential for eval traffic anyway.

The criteria were **not weakened**: mock mode is clearly labeled everywhere
(report header, validation-script warning, DECISIONS.md) and exercises the
identical code paths the live run will use.

## How to unblock (one command each)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install anthropic networkx pydantic pytest

# Phase 2(b) live: rarity validation on 20 histories
python -m codenames_eval.cli validate-histories

# Phase 4 live pilot: 10 dyads x 3 conditions x 3 boards
python -m codenames_eval.cli pilot --mode live --out-dir runs/pilot

# Phase 5 live full run (resumable; re-run the same command after a crash)
python -m codenames_eval.cli run --mode live --seed 0 --out-dir runs/v1
```

Cost note: the full run defaults to 50 dyads x 3 conditions x 5 boards = 750
games at up to 12 turns x 3 role calls, plus 30 summary calls per
summary-condition game — on the order of 30-40k Sonnet calls; the RAW
condition carries ~4k memory tokens per giver/partner call at these history
sizes. Scale down with `--n-dyads/--boards-per-dyad` for a first live pass.
