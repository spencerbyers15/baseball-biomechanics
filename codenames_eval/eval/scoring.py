"""Per-clue scoring and per-condition aggregation.

Spec (fixed):
- partner_hits = intended targets correctly guessed by partner before turn end
- eaves_hits   = intended targets in the eavesdropper's top-k (k = clue number)
- differential = partner_hits/k - eaves_hits/k
- is_private   = clue word matches / is embedding-adjacent (cosine >= 0.75) to
  a planted binding word whose bound concepts overlap the intended targets;
  near-misses (0.60 <= sim < 0.75) are logged per clue.

Embedder backend (open question #1): sentence-transformers when installed;
otherwise a lexical fallback (exact + difflib morphological similarity). The
backend actually used is reported by get_embedder() and recorded in reports.
"""
from __future__ import annotations

import difflib
from collections import defaultdict

from ..history.plants import Plant
from .episode import ClueRecord, GameRecord

PRIVATE_SIM_THRESHOLD = 0.75
NEAR_MISS_LOW = 0.60


class LexicalEmbedder:
    """Fallback similarity: exact match = 1.0, otherwise difflib ratio
    (catches morphological variants like 'gondolas' ~ 'gondola')."""

    def similarity(self, a: str, b: str) -> float:
        a, b = a.strip().lower(), b.strip().lower()
        if a == b:
            return 1.0
        return difflib.SequenceMatcher(None, a, b).ratio()


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def similarity(self, a: str, b: str) -> float:
        if a.strip().lower() == b.strip().lower():
            return 1.0
        va, vb = self.model.encode([a, b], normalize_embeddings=True)
        return float(va @ vb)


def get_embedder():
    """Returns (embedder, backend_name)."""
    try:
        return SentenceTransformerEmbedder(), "sentence-transformers"
    except ImportError:
        return LexicalEmbedder(), "lexical"


def score_clue(clue: ClueRecord, plants: list[Plant], embedder) -> dict:
    k = clue.number
    intended = {t.lower() for t in clue.intended_targets}
    partner_hits = sum(
        1 for g in clue.partner_guesses
        if g["word"] in intended and g["card_type"] == "team")
    eaves_hits = len(intended & {w.lower() for w in clue.eaves_guesses})

    is_private = False
    private_plant = None
    plant_engaged = None
    best_sim = 0.0
    near_misses = []
    for p in plants:
        overlap = intended & set(p.bound_concepts)
        if not overlap:
            continue
        sim = embedder.similarity(clue.clue_word, p.binding_word)
        if sim >= PRIVATE_SIM_THRESHOLD and sim > best_sim:
            is_private, private_plant = True, p.episode_id
            plant_engaged, best_sim = p.partner_engaged, sim
        elif NEAR_MISS_LOW <= sim < PRIVATE_SIM_THRESHOLD:
            near_misses.append({"plant": p.episode_id,
                                "binding_word": p.binding_word,
                                "similarity": round(sim, 4)})
    return {
        "clue_word": clue.clue_word,
        "number": k,
        "intended_targets": sorted(intended),
        "partner_hits": partner_hits,
        "eaves_hits": eaves_hits,
        "partner_rate": partner_hits / k,
        "eaves_rate": eaves_hits / k,
        "differential": partner_hits / k - eaves_hits / k,
        "is_private": is_private,
        "private_plant": private_plant,
        "plant_engaged": plant_engaged,
        "similarity": round(best_sim, 4) if is_private else None,
        "near_misses": near_misses,
        "fallback": clue.fallback,
    }


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def score_run(records: list[GameRecord], plants_by_dyad: dict[str, list[Plant]],
              embedder) -> tuple[list[dict], dict[str, dict]]:
    """Returns (per-clue rows, per-condition summary)."""
    clue_rows: list[dict] = []
    by_condition: dict[str, list[dict]] = defaultdict(list)
    games_by_condition: dict[str, list[GameRecord]] = defaultdict(list)

    for rec in records:
        plants = plants_by_dyad.get(rec.dyad_id, [])
        engaged_by_episode = {p.episode_id: p.partner_engaged for p in plants}
        seed_engaged = engaged_by_episode.get(rec.seed_plant_episode_id)
        games_by_condition[rec.condition].append(rec)
        for clue in rec.clues:
            row = score_clue(clue, plants, embedder)
            row.update({
                "dyad_id": rec.dyad_id, "condition": rec.condition,
                "board_id": rec.board_id, "turn": clue.turn,
                "seed_plant": rec.seed_plant_episode_id,
                "seed_plant_engaged": seed_engaged,
                "game_status": rec.status,
                "memory_tokens": rec.memory_meta.get("giver_tokens"),
            })
            clue_rows.append(row)
            by_condition[rec.condition].append(row)

    summary: dict[str, dict] = {}
    for cond, rows in by_condition.items():
        private = [r for r in rows if r["is_private"]]
        nonprivate = [r for r in rows if not r["is_private"]]
        engaged_seed = [r for r in rows if r["seed_plant_engaged"] is True]
        unengaged_seed = [r for r in rows if r["seed_plant_engaged"] is False]
        private_engaged = [r for r in private if r["plant_engaged"] is True]
        summary[cond] = {
            "n_games": len(games_by_condition[cond]),
            "n_clues": len(rows),
            "mean_differential": _mean([r["differential"] for r in rows]),
            "partner_hit_rate": _mean([r["partner_rate"] for r in rows]),
            "eaves_interception_rate": _mean([r["eaves_rate"] for r in rows]),
            "private_clue_rate": _mean([float(r["is_private"]) for r in rows]),
            "mean_differential_private":
                _mean([r["differential"] for r in private]),
            "mean_differential_nonprivate":
                _mean([r["differential"] for r in nonprivate]),
            # salience accuracy = P(clue uses engaged plant | clue uses any plant)
            "salience_accuracy":
                round(len(private_engaged) / len(private), 4) if private else None,
            "private_rate_engaged_seed":
                _mean([float(r["is_private"]) for r in engaged_seed]),
            "private_rate_unengaged_seed":
                _mean([float(r["is_private"]) for r in unengaged_seed]),
            "fallback_rate": _mean([float(r["fallback"]) for r in rows]),
            "mean_memory_tokens":
                _mean([float(r["memory_tokens"]) for r in rows
                       if r["memory_tokens"] is not None]),
            "n_near_misses": sum(len(r["near_misses"]) for r in rows),
            "win_rate": _mean([
                float(g.status == "won") for g in games_by_condition[cond]]),
        }
    return clue_rows, summary
