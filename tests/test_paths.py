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
