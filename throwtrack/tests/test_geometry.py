import numpy as np

from throwtrack.evaluate import evaluate
from throwtrack.geometry import PersonHeightGround, apply_homography, fit_homography, robust_position
from throwtrack.signals import densify
from throwtrack.simulate import Camera, build_session


def test_homography_recovers_ground_plane():
    cam = Camera()
    ground = np.array([[-10, 15], [10, 15], [-20, 45], [20, 45], [0, 30]], float)
    H = fit_homography(cam.ground_to_image(ground), ground)
    probe = np.array([[3.0, 22.0], [-12.0, 38.0]])
    assert np.allclose(apply_homography(H, cam.ground_to_image(probe)), probe, atol=1e-6)


def test_partner_distance_with_cones(analyzed, truth):
    res = evaluate(analyzed, truth)
    assert res["n_distance"] > 50
    assert res["distance_mae_ft"] < 5.0, res        # typical partner distances are 60-100 ft


def test_no_setup_distance_from_player_height():
    cam = Camera(pitch_deg=0.0)                   # the no-setup mode assumes a level camera
    s = build_session(duration_s=120, seed=3, camera=cam)
    model = PersonHeightGround(cam.width, cam.height, cam.hfov_deg)
    dense = {t.track_id: densify(t, s.fps) for t in s.tracks}
    errs = []
    for g in s.throws:
        if g["target"] == "wall" or g["target"] not in dense or g["thrower"] not in dense:
            continue
        f = g["time_s"] * s.fps
        a = robust_position(model, dense[g["thrower"]], f, s.fps, s.players[g["thrower"]].height_m)
        b = robust_position(model, dense[g["target"]], f, s.fps, s.players[g["target"]].height_m)
        errs.append(abs(np.linalg.norm(a - b) - g["distance_m"]) / g["distance_m"])
    assert len(errs) > 10
    assert np.median(errs) < 0.10
