"""Video-layer checks on a synthetic rendered clip. Skipped where OpenCV is not installed
(the GPU detection/pose pass itself needs model weights and is not run here)."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from throwtrack.schema import ARMS, KP  # noqa: E402
from throwtrack.simulate import build_session  # noqa: E402
from throwtrack.throws import detect_throws  # noqa: E402
from throwtrack.video import SKELETON, make_ball_pixels_fn, render_overlay, video_info  # noqa: E402


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Render 40 s of the simulated session; paint a blue ball in the wall player's hand."""
    s = build_session(duration_s=40, seed=5)
    path = tmp_path_factory.mktemp("vid") / "clip.mp4"
    cam = s.camera
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), s.fps, (cam.width // 2, cam.height // 2))
    lookup = {t.track_id: dict(zip(t.frames.tolist(), t.keypoints)) for t in s.tracks}
    for fi in range(int(s.duration_s * s.fps)):
        img = np.full((cam.height // 2, cam.width // 2, 3), (60, 120, 60), np.uint8)   # grass
        for tid, kpd in lookup.items():
            k = kpd.get(fi)
            if k is None:
                continue
            xy = k[:, :2] / 2
            for a, b in SKELETON:
                cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (230, 230, 230), 2)
            if tid == 7:
                _, elb, wri = ARMS["right"]
                c = xy[wri] + 0.3 * (xy[wri] - xy[elb])
                cv2.circle(img, tuple(map(int, c)), 3, (205, 95, 45), -1)   # BGR blue ball
        w.write(img)
    w.release()
    # tracks in half-resolution pixel coordinates, to match the rendered clip
    from throwtrack.schema import PoseTrack

    tracks = [PoseTrack(t.track_id, t.frames, np.concatenate([t.keypoints[:, :, :2] / 2, t.keypoints[:, :, 2:]], 2))
              for t in s.tracks]
    return path, tracks, s


def test_video_info(clip):
    path, _, s = clip
    info = video_info(path)
    assert info["frames"] == int(s.duration_s * s.fps) and abs(info["fps"] - s.fps) < 0.01


def test_ball_pixels_come_from_the_throwing_hand(clip):
    from throwtrack.ball_type import BallPalette

    path, tracks, s = clip
    throws = [t for t in detect_throws(tracks, s.fps) if t.track_id == 7]
    assert throws
    fn = make_ball_pixels_fn(path, tracks, s.fps)
    name, conf, _ = BallPalette.load().classify(fn(throws[0]))
    assert name == "plyo_450g", (name, conf)


def test_overlay_renders(clip, tmp_path):
    path, tracks, s = clip
    throws = detect_throws(tracks, s.fps)
    out = tmp_path / "overlay.mp4"
    render_overlay(path, out, tracks, throws, {1: {"name": "Smith"}})
    assert video_info(out)["frames"] == video_info(path)["frames"]
