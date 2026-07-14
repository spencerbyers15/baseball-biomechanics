"""Phase 5 acceptance tests: scoring, resumability, reproducibility, report."""
import json
from pathlib import Path

from codenames_eval.cli import main as cli_main
from codenames_eval.config import EvalConfig
from codenames_eval.eval.episode import ClueRecord
from codenames_eval.eval.runner import run_eval
from codenames_eval.eval.scoring import (
    LexicalEmbedder,
    get_embedder,
    score_clue,
    score_run,
)
from codenames_eval.eval.report import generate_report
from codenames_eval.history.plants import Plant


def make_clue(**kw):
    base = dict(
        turn=1, clue_word="gondola", number=2,
        intended_targets=["whale", "piano"],
        reasoning={"why_partner_will_get_it": "x", "why_eavesdropper_wont": "y"},
        partner_guesses=[
            {"word": "whale", "card_type": "team", "ended_turn": False},
            {"word": "piano", "card_type": "team", "ended_turn": False},
        ],
        eaves_guesses=["rope", "cliff"],
    )
    base.update(kw)
    return ClueRecord(**base)


PLANT = Plant(episode_id="d_p0", session_index=3,
              bound_concepts=["whale", "piano"], binding_word="gondola",
              partner_engaged=True)


class TestScoring:
    def test_perfect_private_clue(self):
        row = score_clue(make_clue(), [PLANT], LexicalEmbedder())
        assert row["partner_hits"] == 2
        assert row["eaves_hits"] == 0
        assert row["differential"] == 1.0
        assert row["is_private"]
        assert row["private_plant"] == "d_p0"
        assert row["plant_engaged"] is True

    def test_eavesdropper_interception(self):
        row = score_clue(make_clue(eaves_guesses=["whale", "piano"]),
                         [PLANT], LexicalEmbedder())
        assert row["eaves_hits"] == 2
        assert row["differential"] == 0.0

    def test_partner_miss(self):
        clue = make_clue(partner_guesses=[
            {"word": "rope", "card_type": "neutral", "ended_turn": True}])
        row = score_clue(clue, [PLANT], LexicalEmbedder())
        assert row["partner_hits"] == 0
        assert row["differential"] == 0.0

    def test_non_private_clue(self):
        row = score_clue(make_clue(clue_word="ocean"), [PLANT], LexicalEmbedder())
        assert not row["is_private"]

    def test_embedding_adjacent_private(self):
        # morphological variant of the binding word still counts as private
        row = score_clue(make_clue(clue_word="gondolas"), [PLANT],
                         LexicalEmbedder())
        assert row["is_private"]

    def test_near_miss_logged(self):
        # 'granola' ~ 'gondola' sits in the 0.60-0.75 near-miss band
        row = score_clue(make_clue(clue_word="granola"), [PLANT], LexicalEmbedder())
        assert not row["is_private"]
        assert row["near_misses"], "similar-but-below-threshold must be logged"

    def test_private_requires_target_overlap(self):
        clue = make_clue(intended_targets=["rope", "cliff"], partner_guesses=[
            {"word": "rope", "card_type": "team", "ended_turn": False}])
        row = score_clue(clue, [PLANT], LexicalEmbedder())
        assert not row["is_private"]

    def test_get_embedder_reports_backend(self):
        embedder, backend = get_embedder()
        assert backend in ("sentence-transformers", "lexical")
        assert embedder.similarity("gondola", "gondola") == 1.0


class TestFullRun:
    def _cfg(self, td, **kw):
        base = dict(seed=5, n_dyads=3, boards_per_dyad=2,
                    conditions=["raw", "summary", "graph"], mode="mock",
                    out_dir=str(td))
        base.update(kw)
        return EvalConfig(**base)

    def test_resumability(self, tmp_path):
        """Acceptance: a crashed run restarts from the last completed game."""
        cfg = self._cfg(tmp_path / "run")
        run_eval(cfg)
        results_path = Path(cfg.out_dir) / "results.jsonl"
        lines = results_path.read_text().splitlines()
        assert len(lines) == 3 * 2 * 3
        # simulate a crash after 5 games
        results_path.write_text("\n".join(lines[:5]) + "\n")
        results = run_eval(cfg)
        assert len(results) == 18
        lines2 = results_path.read_text().splitlines()
        assert len(lines2) == 18
        assert lines2[:5] == lines[:5]  # completed games not re-executed

    def test_seed_reproducibility(self, tmp_path):
        """Acceptance: a seed reproduces identical board/history assignments."""
        r1 = run_eval(self._cfg(tmp_path / "a"))
        r2 = run_eval(self._cfg(tmp_path / "b"))
        for a, b in zip(r1, r2):
            assert a.model_dump() == b.model_dump()
        p1 = (tmp_path / "a" / "plants.jsonl").read_text()
        p2 = (tmp_path / "b" / "plants.jsonl").read_text()
        assert p1 == p2

    def test_score_run_and_report(self, tmp_path):
        cfg = self._cfg(tmp_path / "run")
        run_eval(cfg)
        report_paths = generate_report(cfg.out_dir)
        for key in ("report_md", "clues_csv", "conditions_csv"):
            assert Path(report_paths[key]).exists()
        report = Path(report_paths["report_md"]).read_text()
        for needle in ("differential", "raw", "summary", "graph",
                       "private", "salience", "eavesdropper"):
            assert needle in report.lower()
        clue_lines = Path(report_paths["clues_csv"]).read_text().splitlines()
        assert len(clue_lines) > 18  # header + at least one clue per game
        cond_lines = Path(report_paths["conditions_csv"]).read_text().splitlines()
        assert len(cond_lines) == 4  # header + 3 conditions

    def test_metrics_structure(self, tmp_path):
        cfg = self._cfg(tmp_path / "run")
        records = run_eval(cfg)
        plants_by_dyad = {}
        for line in (Path(cfg.out_dir) / "plants.jsonl").read_text().splitlines():
            row = json.loads(line)
            plants_by_dyad[row["dyad_id"]] = [
                Plant.model_validate(p) for p in row["plants"]]
        clue_rows, summary = score_run(records, plants_by_dyad, LexicalEmbedder())
        assert set(summary) == {"raw", "summary", "graph"}
        for cond, m in summary.items():
            assert set(m) >= {
                "n_games", "n_clues", "mean_differential", "partner_hit_rate",
                "eaves_interception_rate", "private_clue_rate",
                "mean_differential_private", "mean_differential_nonprivate",
                "salience_accuracy", "private_rate_engaged_seed",
                "private_rate_unengaged_seed", "fallback_rate",
                "mean_memory_tokens"}

    def test_mock_sanity_graph_beats_summary(self, tmp_path):
        """With the memory-aware mock, conditions that preserve co-occurrence
        (raw/graph) must out-differentiate summary, which provably drops it."""
        cfg = self._cfg(tmp_path / "run", n_dyads=5, boards_per_dyad=2)
        records = run_eval(cfg)
        plants_by_dyad = {}
        for line in (Path(cfg.out_dir) / "plants.jsonl").read_text().splitlines():
            row = json.loads(line)
            plants_by_dyad[row["dyad_id"]] = [
                Plant.model_validate(p) for p in row["plants"]]
        _, summary = score_run(records, plants_by_dyad, LexicalEmbedder())
        assert summary["graph"]["mean_differential"] > summary["summary"]["mean_differential"]
        assert summary["graph"]["private_clue_rate"] > summary["summary"]["private_clue_rate"]


class TestCLI:
    def test_cli_run_and_report(self, tmp_path):
        out = str(tmp_path / "cli_run")
        rc = cli_main(["run", "--mode", "mock", "--seed", "3", "--n-dyads", "2",
                       "--boards-per-dyad", "2", "--out-dir", out])
        assert rc == 0
        assert (Path(out) / "results.jsonl").exists()
        assert (Path(out) / "report.md").exists()

    def test_cli_pilot(self, tmp_path):
        out = str(tmp_path / "cli_pilot")
        rc = cli_main(["pilot", "--mode", "mock", "--out-dir", out,
                       "--n-dyads", "2", "--boards-per-dyad", "1"])
        assert rc == 0
        assert (Path(out) / "results.jsonl").exists()
