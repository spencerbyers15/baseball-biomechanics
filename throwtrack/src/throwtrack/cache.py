"""Save/load pose tracks, so the expensive video pass runs once per video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import PoseTrack


def save_tracks(path: str | Path, tracks: list[PoseTrack], meta: dict) -> None:
    arrays = {}
    for t in tracks:
        arrays[f"f_{t.track_id}"] = t.frames
        arrays[f"k_{t.track_id}"] = t.keypoints.astype(np.float32)
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load_tracks(path: str | Path) -> tuple[list[PoseTrack], dict]:
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    ids = sorted(int(k[2:]) for k in z.files if k.startswith("f_"))
    tracks = [PoseTrack(i, z[f"f_{i}"], z[f"k_{i}"]) for i in ids]
    return tracks, meta
