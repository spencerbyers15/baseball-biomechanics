import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def session():
    from throwtrack.simulate import build_session

    return build_session(duration_s=300, seed=7)


@pytest.fixture(scope="session")
def analyzed(session):
    from throwtrack.geometry import load_calibration
    from throwtrack.pipeline import SessionConfig, analyze_session

    roster = {pid: {"height_m": p.height_m} for pid, p in session.players.items()}
    cfg = SessionConfig(fps=session.fps, ground=load_calibration(session.calibration), roster=roster)
    return analyze_session(session.tracks, cfg)


@pytest.fixture(scope="session")
def truth(session):
    return [{"thrower": g["thrower"], "time_s": g["time_s"], "target": g["target"],
             "distance_ft": g["distance_m"] / 0.3048 if g["target"] != "wall" else None,
             "mph": g["ball_speed_mps"] / 0.44704} for g in session.throws]
