import numpy as np
import pytest

from throwtrack.schema import ThrowEvent
from throwtrack.velocity import (KinematicVelocityModel, drag_k, launch_angle_for, match_radar,
                                 release_speed_from_flight, simulate_flight)


@pytest.mark.parametrize("speed,dist", [(20, 18), (30, 18), (27, 30), (35, 40)])
def test_flight_time_inversion_with_drag(speed, dist):
    k = drag_k(145, 73)
    ang = launch_angle_for(dist, speed, k, catch_h=1.4)
    T, _ = simulate_flight(speed, ang, dist, k, dt=1 / 2000)
    est = release_speed_from_flight(dist, T, k, dh_m=1.4 - 1.9)
    assert abs(est - speed) / speed < 0.03


def test_heavier_ball_has_less_drag():
    assert drag_k(1000, 110) < drag_k(145, 73)


def test_kinematic_model_recovers_relation_and_intervals():
    rng = np.random.default_rng(0)
    hand = rng.uniform(10, 30, 300)
    y = 5 + 2.0 * hand + rng.normal(0, 2, 300)
    rows = [{"hand_speed_mps": h, "stride": np.nan, "distance_ft": 60.0, "ball_mass_g": np.nan} for h in hand]
    m = KinematicVelocityModel(l2=0.1).fit(rows[:200], y[:200])
    preds = [m.predict(r) for r in rows[200:]]
    err = np.array([p for p, _ in preds]) - y[200:]
    assert np.mean(np.abs(err)) < 2.5
    inside = np.mean([lo <= yy <= hi for (p, s), yy in zip(preds, y[200:]) for lo, hi in [(p - 1.645 * s, p + 1.645 * s)]])
    assert 0.8 <= inside <= 0.97


def test_constant_training_feature_cannot_explode_predictions():
    rows = [{"hand_speed_mps": h, "distance_ft": 60.0 + 0.01 * i} for i, h in enumerate(np.linspace(10, 30, 50))]
    m = KinematicVelocityModel().fit(rows, [2 * r["hand_speed_mps"] for r in rows])
    mph, _ = m.predict({"hand_speed_mps": 20.0, "distance_ft": 150.0})
    assert 20 < mph < 60


def test_player_offsets_are_learned():
    rows = [{"hand_speed_mps": h} for h in np.linspace(10, 30, 40)]
    y = [2 * r["hand_speed_mps"] + (6 if i % 2 else -6) for i, r in enumerate(rows)]
    groups = ["hard" if i % 2 else "soft" for i in range(40)]
    m = KinematicVelocityModel(l2=0.01).fit(rows, y, groups=groups)
    hard, _ = m.predict({"hand_speed_mps": 20.0}, group="hard")
    soft, _ = m.predict({"hand_speed_mps": 20.0}, group="soft")
    assert hard - soft > 9


def test_radar_matching_is_one_to_one_and_player_aware():
    throws = [ThrowEvent(1, 0, 10.0, "right", 1), ThrowEvent(2, 0, 10.3, "right", 1), ThrowEvent(1, 0, 20.0, "right", 1)]
    radar = [{"time_s": 10.2, "mph": 70, "track_id": 1}, {"time_s": 10.25, "mph": 65}, {"time_s": 25.0, "mph": 80}]
    assert match_radar(throws, radar, tol_s=0.5) == [(0, 70.0), (1, 65.0)]
