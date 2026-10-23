from pathlib import Path

from src.paths import PROJECT_ROOT, resolve_repo_path, to_repo_relative


def test_windows_absolute_path_is_rerooted():
    stored = r"F:\Claude_Projects\baseball-biomechanics\data\labels\frames\a.png"
    assert resolve_repo_path(stored) == PROJECT_ROOT / "data/labels/frames/a.png"


def test_other_machine_posix_path_is_rerooted():
    stored = "/Users/someone/code/baseball-biomechanics/data/bat_frames_round2/b.jpg"
    assert resolve_repo_path(stored) == PROJECT_ROOT / "data/bat_frames_round2/b.jpg"


def test_relative_paths_are_repo_relative():
    assert resolve_repo_path("data/labels/c.png") == PROJECT_ROOT / "data/labels/c.png"
    assert resolve_repo_path(r"data\labels\c.png") == PROJECT_ROOT / "data/labels/c.png"


def test_unrelated_absolute_path_is_unchanged():
    assert resolve_repo_path("/mnt/videos/x.mp4") == Path("/mnt/videos/x.mp4")


def test_to_repo_relative_round_trips():
    stored = r"F:\Claude_Projects\baseball-biomechanics\data\labels\frames\a.png"
    rel = to_repo_relative(stored)
    assert rel == "data/labels/frames/a.png"
    assert resolve_repo_path(rel) == PROJECT_ROOT / rel


def test_to_repo_relative_keeps_outside_paths_absolute():
    assert to_repo_relative("/mnt/videos/x.mp4") == "/mnt/videos/x.mp4"


def test_path_already_in_checkout_is_unchanged_even_under_same_named_ancestor(tmp_path):
    # e.g. a worktree at .../baseball-biomechanics/.claude/worktrees/feat
    root = tmp_path / "baseball-biomechanics" / ".claude" / "worktrees" / "feat"
    live = root / "data" / "a.jpg"
    assert resolve_repo_path(str(live), root) == live
    assert to_repo_relative(live, root) == "data/a.jpg"


def test_windows_path_rerooted_at_custom_root(tmp_path):
    stored = r"F:\Claude_Projects\baseball-biomechanics\data\labels\x.jpg"
    assert resolve_repo_path(stored, tmp_path) == tmp_path / "data/labels/x.jpg"


def test_to_repo_relative_treats_relative_input_as_cwd_relative(tmp_path, monkeypatch):
    (tmp_path / "tools").mkdir()
    monkeypatch.chdir(tmp_path / "tools")
    assert to_repo_relative("../data/frames/x.jpg", tmp_path) == "data/frames/x.jpg"


def test_entry_helpers_round_trip(tmp_path):
    from src.paths import portable_entries, resolve_entries
    entries = [{"path": r"F:\Claude_Projects\baseball-biomechanics\data\f\a.png", "id": 1},
               {"path": r"F:\Claude_Projects\baseball-biomechanics\data\f\b.png", "source_video": ""}]
    resolve_entries(entries, ("path", "source_video"), tmp_path)
    assert entries[0]["path"] == str(tmp_path / "data/f/a.png")
    assert entries[1]["source_video"] == ""
    saved = portable_entries(entries, ("path", "source_video"), tmp_path)
    assert [e["path"] for e in saved] == ["data/f/a.png", "data/f/b.png"]
    assert entries[0]["path"] == str(tmp_path / "data/f/a.png")  # originals untouched
