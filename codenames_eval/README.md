# Codenames as a joint memory + theory-of-mind eval for LLM agents

An autonomous evaluation harness measuring whether LLM agents can use
accumulated shared history with a partner to generate Codenames clues that
the partner decodes but an adversarial eavesdropper (same board + clue, no
history) cannot.

**Headline metric — clue differential**: `partner_hits/k − eavesdropper_hits/k`
per clue, aggregated per memory condition (RAW transcript / flat SUMMARY /
episodic entity GRAPH).

## Layout

```
codenames_eval/
  engine/    deterministic single-team Codenames engine + word pools
  history/   synthetic dyad histories with injected plants + validators
  memory/    RAW / SUMMARY / GRAPH builders (+ graph retrieval)
  agents/    LLM client layer (Anthropic + deterministic mock), 3 player roles
  eval/      plant-seeded boards, game episodes, runner, scoring, report
  cli.py     single entry point (run / pilot / report / validate-histories)
```

## Quick start

```bash
pip install anthropic networkx pydantic pytest

# offline smoke run (no API key needed; deterministic mock)
python -m codenames_eval.cli run --mode mock --n-dyads 5 --out-dir runs/smoke

# live pilot / full run (needs ANTHROPIC_API_KEY)
python -m codenames_eval.cli pilot --mode live --out-dir runs/pilot
python -m codenames_eval.cli run  --mode live --seed 0 --out-dir runs/v1

# tests (76)
python -m pytest tests/codenames -q
```

Runs are **resumable**: re-invoke the same `run` command after a crash and
completed games (checkpointed in `results.jsonl`) are skipped. The same seed
reproduces identical history/board assignments. Models are per-role
configurable (`--giver-model / --partner-model / --eaves-model`).

Outputs per run dir: `report.md`, `metrics_by_condition.csv`, `clues.csv`,
`results.jsonl`, `plants.jsonl` (ground truth), `llm_calls.jsonl` (every call
with prompt hash, model, tokens, latency), `config.json`.

See `DECISIONS.md` for logged defaults on the brief's open questions and
`BLOCKED.md` for the live-run items gated on API credentials.
