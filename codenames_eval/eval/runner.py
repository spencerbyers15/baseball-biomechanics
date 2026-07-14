"""Full eval runner: dyads x conditions x boards, with JSONL resumability.

Every completed game is appended to <out_dir>/results.jsonl immediately; a
crashed run restarted with the same config skips already-completed games.
Board/history assignment is a pure function of the config seed.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agents.llm import AnthropicClient, LLMClient, MockLLMClient
from ..config import EvalConfig
from ..history.generator import generate_dyad
from .boards import build_board_spec
from .episode import GameRecord, play_game


def _dyad_seed(cfg_seed: int, dyad_index: int) -> int:
    return cfg_seed * 1_000_003 + dyad_index


def make_clients(cfg: EvalConfig, out_dir: Path) -> dict[str, LLMClient]:
    log = out_dir / "llm_calls.jsonl"
    if cfg.mode == "mock":
        client = MockLLMClient(seed=cfg.seed, log_path=log)
        return {"giver": client, "partner": client, "eaves": client}
    return {
        "giver": AnthropicClient(model=cfg.giver_model, log_path=log),
        "partner": AnthropicClient(model=cfg.partner_model, log_path=log),
        "eaves": AnthropicClient(model=cfg.eaves_model, log_path=log),
    }


def _load_completed(results_path: Path) -> dict[str, GameRecord]:
    completed: dict[str, GameRecord] = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                rec = GameRecord.model_validate_json(line)
                completed[f"{rec.dyad_id}|{rec.condition}|{rec.board_id}"] = rec
    return completed


def run_eval(cfg: EvalConfig, progress: bool = False) -> list[GameRecord]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    results_path = out_dir / "results.jsonl"
    plants_path = out_dir / "plants.jsonl"
    completed = _load_completed(results_path)
    clients = make_clients(cfg, out_dir)

    history_llm = clients["giver"] if cfg.history_mode == "llm" else None
    results: list[GameRecord] = []
    plant_rows: list[dict] = []

    for d in range(cfg.n_dyads):
        dyad_id = f"dyad_{d}"
        seed = _dyad_seed(cfg.seed, d)
        history, plants = generate_dyad(
            dyad_id, seed=seed, n_sessions=cfg.n_sessions, n_plants=cfg.n_plants,
            mode=cfg.history_mode, llm=history_llm)
        plant_rows.append({"dyad_id": dyad_id,
                           "plants": [p.model_dump() for p in plants]})
        for b in range(cfg.boards_per_dyad):
            spec = build_board_spec(f"{dyad_id}_board_{b}", plants,
                                    seed=seed * 31 + b, plant_index=b)
            for condition in cfg.conditions:
                key = f"{dyad_id}|{condition}|{spec.board_id}"
                if key in completed:
                    results.append(completed[key])
                    continue
                record = play_game(
                    dyad_id=dyad_id, condition=condition, history=history,
                    board_spec=spec, giver_llm=clients["giver"],
                    partner_llm=clients["partner"], eaves_llm=clients["eaves"],
                    raw_token_budget=cfg.raw_token_budget,
                    max_turns=cfg.max_turns)
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")
                results.append(record)
                if progress:
                    print(f"done {key}: {record.status} "
                          f"({len(record.clues)} clues)")

    # ground-truth plants (overwritten each run; deterministic from seed)
    with open(plants_path, "w", encoding="utf-8") as f:
        for row in plant_rows:
            f.write(json.dumps(row) + "\n")
    return results
