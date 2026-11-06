import numpy as np

from throwtrack.ball_type import BallPalette, hand_region


def _patch(rgb, n):
    return np.tile(np.array(rgb, np.uint8), (n, 1))


def test_ball_colors_beat_skin_glove_and_uniform():
    pal = BallPalette.load()
    clutter = np.concatenate([_patch([225, 180, 150], 300), _patch([140, 90, 50], 200), _patch([150, 150, 150], 300)])
    rng = np.random.default_rng(0)
    for name, rgb in [("baseball", [240, 238, 230]), ("plyo_450g", [45, 95, 205]), ("plyo_1kg", [35, 150, 75])]:
        ball = np.clip(_patch(rgb, 60) + rng.normal(0, 6, (60, 3)), 0, 255)
        got, conf, _ = pal.classify(np.concatenate([clutter, ball]))
        assert got == name and conf > 0.5, (name, got, conf)


def test_no_ball_visible_returns_none():
    pal = BallPalette.load()
    got, conf, _ = pal.classify(_patch([225, 180, 150], 500))
    assert got is None and conf == 0.0


def test_hand_region_is_clipped_to_image():
    xy = np.full((17, 2), np.nan)
    xy[8], xy[10] = (5.0, 5.0), (2.0, 2.0)            # right elbow, right wrist near the corner
    x0, y0, x1, y1 = hand_region(xy, "right", 20.0, (100, 100, 3))
    assert x0 == 0 and y0 == 0 and x1 > 0 and y1 > 0
