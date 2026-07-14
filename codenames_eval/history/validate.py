"""Validation for generated histories.

Two acceptance checks (Phase 2):
(a) co-occurrence: every plant's bound concepts (and binding word) appear in
    exactly the intended session; no other session contains the binding word
    or two or more of the bound concepts together.
(b) rarity: a fresh LLM with no history, given the binding word and the full
    word list, matches the bound concepts at <= chance + 10%.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..agents.llm import LLMClient
from .plants import DyadHistory, Plant

_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

RARITY_MARGIN = 0.10  # acceptance: match rate <= chance + this margin


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _session_token_sets(history: DyadHistory) -> dict[int, set[str]]:
    return {
        s.index: {tok for turn in s.turns for tok in tokenize(turn.text)}
        for s in history.sessions
    }


def validate_cooccurrence(history: DyadHistory, plants: list[Plant]) -> dict:
    tokens_by_session = _session_token_sets(history)
    violations: list[str] = []
    for p in plants:
        intended = tokens_by_session.get(p.session_index, set())
        missing = ({p.binding_word} | set(p.bound_concepts)) - intended
        if missing:
            violations.append(
                f"{p.episode_id}: plant content missing from intended session "
                f"{p.session_index}: {sorted(missing)}")
        for idx, tokens in tokens_by_session.items():
            if idx == p.session_index:
                continue
            if p.binding_word in tokens:
                violations.append(
                    f"{p.episode_id}: binding word {p.binding_word!r} leaked "
                    f"into session {idx}")
            overlap = set(p.bound_concepts) & tokens
            if len(overlap) >= 2:
                violations.append(
                    f"{p.episode_id}: concepts {sorted(overlap)} co-occur in "
                    f"unintended session {idx}")
    return {"ok": not violations, "violations": violations}


class _RarityOut(BaseModel):
    selected: list[str] = Field(default_factory=list)


def rarity_probe(llm: LLMClient, binding_word: str, word_list: list[str],
                 k: int) -> list[str]:
    """Ask a fresh (history-free) model for the k words most associated with
    the binding word."""
    spec = {"task_id": "rarity_probe", "probe_word": binding_word,
            "word_list": word_list, "k": k}
    prompt = (
        f"From the word list below, pick the {k} words MOST strongly "
        f"associated with the word {binding_word!r} in general knowledge. "
        "Respond with JSON: {\"selected\": [...]}.\n"
        "Input:\n```json\n" + json.dumps(spec) + "\n```"
    )
    out = llm.complete_json(prompt, _RarityOut, task_id="rarity_probe")
    listed = {w.lower() for w in word_list}
    return [w.lower() for w in out.selected if w.lower() in listed][:k]


def validate_rarity(llm: LLMClient, plants: list[Plant], word_list: list[str],
                    margin: float = RARITY_MARGIN) -> dict:
    per_plant = []
    for p in plants:
        k = len(p.bound_concepts)
        selected = rarity_probe(llm, p.binding_word, word_list, k)
        hits = len(set(selected) & set(p.bound_concepts))
        per_plant.append({
            "episode_id": p.episode_id,
            "binding_word": p.binding_word,
            "bound_concepts": p.bound_concepts,
            "selected": selected,
            "match_rate": hits / k,
            "chance_rate": k / len(word_list),
        })
    mean_match = sum(r["match_rate"] for r in per_plant) / max(1, len(per_plant))
    mean_chance = sum(r["chance_rate"] for r in per_plant) / max(1, len(per_plant))
    threshold = mean_chance + margin
    return {
        "ok": mean_match <= threshold,
        "mean_match_rate": mean_match,
        "chance_rate": mean_chance,
        "threshold": threshold,
        "per_plant": per_plant,
    }


def main() -> int:  # pragma: no cover - exercised via CLI
    """Phase 2 acceptance script: generate 20 histories and run both checks.

    Uses the real API when ANTHROPIC_API_KEY is set; otherwise falls back to
    the deterministic mock and says so loudly (the mock answers rarity probes
    at chance, so only the machinery — not model behavior — is validated).
    """
    import argparse
    import os

    from ..agents.llm import AnthropicClient, MockLLMClient
    from ..engine.wordlist import BOARD_WORD_POOL
    from .generator import generate_dyad

    ap = argparse.ArgumentParser(description="Validate 20 generated histories")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", choices=["template", "llm"], default="template")
    ap.add_argument("--log", default="runs/validate_calls.jsonl")
    args = ap.parse_args()

    if os.environ.get("ANTHROPIC_API_KEY"):
        llm = AnthropicClient(log_path=args.log)
        print("Using AnthropicClient for rarity probes.")
    else:
        llm = MockLLMClient(seed=0, log_path=args.log)
        print("WARNING: no ANTHROPIC_API_KEY — using MockLLMClient. Rarity "
              "results validate machinery only, not real model associations.")

    gen_llm = llm if args.mode == "llm" else None
    all_ok = True
    for seed in range(args.n):
        history, plants = generate_dyad(f"dyad_{seed}", seed=seed,
                                        mode=args.mode, llm=gen_llm)
        co = validate_cooccurrence(history, plants)
        ra = validate_rarity(llm, plants, BOARD_WORD_POOL)
        ok = co["ok"] and ra["ok"]
        all_ok &= ok
        print(f"dyad_{seed}: cooccurrence={'OK' if co['ok'] else co['violations']} "
              f"rarity={'OK' if ra['ok'] else 'FAIL'} "
              f"(match={ra['mean_match_rate']:.3f} <= {ra['threshold']:.3f})")
    print("ALL OK" if all_ok else "FAILURES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
