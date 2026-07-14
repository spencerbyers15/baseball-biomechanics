"""Analysis outputs: per-clue CSV, per-condition CSV, markdown report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ..history.plants import Plant
from .episode import GameRecord
from .scoring import PRIVATE_SIM_THRESHOLD, get_embedder, score_run

CLUE_COLUMNS = [
    "dyad_id", "condition", "board_id", "turn", "clue_word", "number",
    "intended_targets", "partner_hits", "eaves_hits", "partner_rate",
    "eaves_rate", "differential", "is_private", "private_plant",
    "plant_engaged", "similarity", "near_misses", "seed_plant",
    "seed_plant_engaged", "game_status", "memory_tokens", "fallback",
]

CONDITION_COLUMNS = [
    "condition", "n_games", "n_clues", "mean_differential",
    "partner_hit_rate", "eaves_interception_rate", "private_clue_rate",
    "mean_differential_private", "mean_differential_nonprivate",
    "salience_accuracy", "private_rate_engaged_seed",
    "private_rate_unengaged_seed", "win_rate", "fallback_rate",
    "mean_memory_tokens", "n_near_misses",
]


def load_run(run_dir: str | Path) -> tuple[list[GameRecord], dict[str, list[Plant]]]:
    run_dir = Path(run_dir)
    records = [
        GameRecord.model_validate_json(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
    ]
    plants_by_dyad: dict[str, list[Plant]] = {}
    for line in (run_dir / "plants.jsonl").read_text().splitlines():
        row = json.loads(line)
        plants_by_dyad[row["dyad_id"]] = [
            Plant.model_validate(p) for p in row["plants"]]
    return records, plants_by_dyad


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def generate_report(run_dir: str | Path) -> dict[str, str]:
    run_dir = Path(run_dir)
    records, plants_by_dyad = load_run(run_dir)
    embedder, backend = get_embedder()
    clue_rows, summary = score_run(records, plants_by_dyad, embedder)

    clues_csv = run_dir / "clues.csv"
    with open(clues_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLUE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in clue_rows:
            row = dict(row)
            row["intended_targets"] = "|".join(row["intended_targets"])
            row["near_misses"] = json.dumps(row["near_misses"])
            w.writerow(row)

    conditions_csv = run_dir / "metrics_by_condition.csv"
    with open(conditions_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CONDITION_COLUMNS)
        w.writeheader()
        for cond in ("raw", "summary", "graph"):
            if cond in summary:
                w.writerow({"condition": cond, **{
                    k: summary[cond].get(k) for k in CONDITION_COLUMNS[1:]}})
        for cond in summary:
            if cond not in ("raw", "summary", "graph"):
                w.writerow({"condition": cond, **{
                    k: summary[cond].get(k) for k in CONDITION_COLUMNS[1:]}})

    config_note = ""
    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        config_note = (
            f"- seed: {cfg.get('seed')} | dyads: {cfg.get('n_dyads')} | "
            f"boards/dyad: {cfg.get('boards_per_dyad')} | "
            f"sessions: {cfg.get('n_sessions')} | plants: {cfg.get('n_plants')}\n"
            f"- mode: {cfg.get('mode')} | models: giver={cfg.get('giver_model')}, "
            f"partner={cfg.get('partner_model')}, eaves={cfg.get('eaves_model')}\n")

    lines = [
        "# Codenames memory + theory-of-mind eval — results",
        "",
        "## Run",
        config_note.rstrip(),
        f"- games: {len(records)} | clues scored: {len(clue_rows)}",
        f"- `is_private` similarity backend: **{backend}** "
        f"(threshold >= {PRIVATE_SIM_THRESHOLD}; near-misses logged in clues.csv)",
        "",
        "## Headline: clue differential by memory condition",
        "",
        "differential = partner_hits/k − eavesdropper_hits/k (per clue; higher"
        " = partner decodes what the eavesdropper cannot)",
        "",
        "| condition | n clues | mean differential | partner hit rate | "
        "eavesdropper interception | private-clue rate | win rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond in ("raw", "summary", "graph"):
        if cond not in summary:
            continue
        m = summary[cond]
        lines.append(
            f"| {cond} | {m['n_clues']} | {_fmt(m['mean_differential'])} | "
            f"{_fmt(m['partner_hit_rate'])} | "
            f"{_fmt(m['eaves_interception_rate'])} | "
            f"{_fmt(m['private_clue_rate'])} | {_fmt(m['win_rate'])} |")
    lines += [
        "",
        "## Do private clues carry the differential?",
        "",
        "| condition | mean diff (private clues) | mean diff (non-private) |",
        "|---|---|---|",
    ]
    for cond in ("raw", "summary", "graph"):
        if cond not in summary:
            continue
        m = summary[cond]
        lines.append(f"| {cond} | {_fmt(m['mean_differential_private'])} | "
                     f"{_fmt(m['mean_differential_nonprivate'])} |")
    lines += [
        "",
        "## Salience: engaged vs unengaged plants",
        "",
        "salience accuracy = P(private clue uses an engaged plant); the",
        "private-rate split shows whether the giver avoids plants the partner",
        "never engaged with when such a board comes up.",
        "",
        "| condition | salience accuracy | private rate (engaged seed) | "
        "private rate (unengaged seed) |",
        "|---|---|---|---|",
    ]
    for cond in ("raw", "summary", "graph"):
        if cond not in summary:
            continue
        m = summary[cond]
        lines.append(
            f"| {cond} | {_fmt(m['salience_accuracy'])} | "
            f"{_fmt(m['private_rate_engaged_seed'])} | "
            f"{_fmt(m['private_rate_unengaged_seed'])} |")
    lines += [
        "",
        "## Diagnostics",
        "",
        "| condition | mean memory tokens | fallback rate | near-misses |",
        "|---|---|---|---|",
    ]
    for cond in ("raw", "summary", "graph"):
        if cond not in summary:
            continue
        m = summary[cond]
        lines.append(f"| {cond} | {_fmt(m['mean_memory_tokens'])} | "
                     f"{_fmt(m['fallback_rate'])} | {m['n_near_misses']} |")
    lines += [
        "",
        "Artifacts: `clues.csv` (per-clue), `metrics_by_condition.csv`, "
        "`results.jsonl` (full game records), `llm_calls.jsonl` (call log), "
        "`plants.jsonl` (ground truth).",
        "",
    ]
    report_md = run_dir / "report.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {"report_md": str(report_md), "clues_csv": str(clues_csv),
            "conditions_csv": str(conditions_csv)}
