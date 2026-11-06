import json

import numpy as np

from throwtrack.cache import load_tracks, save_tracks
from throwtrack.cli import main
from throwtrack.schema import PoseTrack


def test_demo_end_to_end(tmp_path):
    main(["demo", "--minutes", "3", "--seed", "1", "--out", str(tmp_path)])
    for name in ("throws.csv", "players.csv", "throws.json", "report.html", "metrics.json"):
        assert (tmp_path / name).exists(), name
    rows = json.loads((tmp_path / "throws.json").read_text())
    assert rows and {"timecode", "thrower", "receiver", "throw_conf", "needs_review"} <= set(rows[0])
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["all"]["recall"] > 0.9


def test_pose_cache_roundtrip(tmp_path):
    kp = np.random.default_rng(0).random((6, 17, 3))
    tr = [PoseTrack(3, np.arange(10, 16), kp)]
    save_tracks(tmp_path / "p.npz", tr, {"fps": 30.0})
    back, meta = load_tracks(tmp_path / "p.npz")
    assert meta["fps"] == 30.0 and back[0].track_id == 3
    assert np.allclose(back[0].keypoints, kp, atol=1e-6)
