"""Video layer: the only module that needs the GPU stack (OpenCV, Ultralytics, rtmlib).

Everything else in the package works on the pose tracks this produces, so
the expensive pass runs once per video and is cached (cache.py).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .ball_type import hand_region
from .schema import ARMS, KP, PoseTrack, ThrowEvent
from .signals import densify, foot_point

REPO_ROOT = Path(__file__).resolve().parents[3]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def video_info(path: str | Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "video": str(path),
    }
    cap.release()
    return info


def default_pose_model() -> Path:
    """RTMPose-X ONNX from the parent repo's models/ dir, downloading it if needed."""
    found = sorted((REPO_ROOT / "models" / "rtmpose").glob("*.onnx"))
    if found:
        return found[0]
    sys.path.insert(0, str(REPO_ROOT))
    from src.pose.rtmpose_backend import RTMPoseBackend  # downloads to models/rtmpose/

    return RTMPoseBackend()._ensure_model()


def extract_pose_tracks(video_path: str | Path, det_weights: str = "yolo11m.pt",
                        pose_onnx: Optional[str | Path] = None, tracker: str = "botsort.yaml",
                        device: str = "cuda", det_conf: float = 0.3, imgsz: int = 1280,
                        max_frames: Optional[int] = None, progress: bool = True) -> tuple[list[PoseTrack], dict]:
    """YOLO person detection + BoT-SORT tracking, then RTMPose-X on each tracked box."""
    from rtmlib import RTMPose
    from ultralytics import YOLO

    info = video_info(video_path)
    det = YOLO(det_weights)
    pose = RTMPose(onnx_model=str(pose_onnx or default_pose_model()), model_input_size=(288, 384),
                   backend="onnxruntime", device=device)
    frames, kps = defaultdict(list), defaultdict(list)
    stream = det.track(source=str(video_path), stream=True, persist=True, classes=[0], conf=det_conf,
                       tracker=tracker, imgsz=imgsz, device=0 if device == "cuda" else "cpu", verbose=False)
    for fi, res in enumerate(stream):
        if max_frames is not None and fi >= max_frames:
            break
        if progress and fi % 300 == 0:
            print(f"  pose pass: frame {fi}/{info['frames']}", flush=True)
        boxes = res.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        kp, sc = pose(res.orig_img, bboxes=xyxy.tolist())
        sc = np.clip(np.asarray(sc, float), 0.0, 1.0)
        for tid, k, s in zip(ids, kp, sc):
            frames[tid].append(fi)
            kps[tid].append(np.concatenate([np.asarray(k, float), s[:, None]], axis=1))
    tracks = [PoseTrack(int(t), np.array(frames[t]), np.stack(kps[t])) for t in frames if len(frames[t]) >= 5]
    return tracks, info


def make_ball_pixels_fn(video_path: str | Path, tracks: list[PoseTrack], fps: float,
                        offsets_s=(-0.30, -0.24, -0.18, -0.12)):
    """Returns fn(throw) -> RGB pixels around the throwing hand shortly before release."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    dense = {t.track_id: densify(t, fps) for t in tracks}

    def fn(throw: ThrowEvent) -> Optional[np.ndarray]:
        d = dense.get(throw.track_id)
        if d is None:
            return None
        px = []
        for off in offsets_s:
            f = int(round(throw.frame + off * fps))
            if f < d.frames[0] or f > d.frames[-1]:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            i = d.index_of(f)
            box = hand_region(d.xy[i], throw.arm, d.scale[i], img.shape)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            px.append(img[y0:y1, x0:x1, ::-1].reshape(-1, 3))   # BGR -> RGB
        return np.concatenate(px) if px else None

    return fn


def render_overlay(video_path: str | Path, out_path: str | Path, tracks: list[PoseTrack],
                   throws: list[ThrowEvent], roster: Optional[dict] = None, flash_s: float = 0.8) -> None:
    """Annotated copy of the video: skeletons, player names with running throw counts,
    and a flash + arrow to the receiver at each detected release."""
    import cv2

    roster = roster or {}
    info = video_info(video_path)
    fps = info["fps"]
    cap = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (info["width"], info["height"]))
    by_id = {t.track_id: t for t in tracks}
    dense = {t.track_id: densify(t, fps) for t in tracks}
    throws = sorted(throws, key=lambda e: e.frame)
    counts: dict = defaultdict(int)
    nxt = 0
    fi = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        while nxt < len(throws) and throws[nxt].frame <= fi:
            counts[throws[nxt].track_id] += 1
            nxt += 1
        for tid, tr in by_id.items():
            j = np.searchsorted(tr.frames, fi)
            if j >= len(tr.frames) or tr.frames[j] != fi:
                continue
            k = tr.keypoints[j]
            for a, b in SKELETON:
                if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                    cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), (80, 220, 80), 2)
            name = roster.get(tid, {}).get("name", f"#{tid}")
            x, y = int(np.nanmin(k[:, 0])), int(np.nanmin(k[:, 1])) - 8
            cv2.putText(img, f"{name}  {counts[tid]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for t in throws[max(0, nxt - 20) : nxt]:
            if fi - t.frame > flash_s * fps:
                continue
            d = dense[t.track_id]
            w = d.xy[d.index_of(fi), ARMS[t.arm][2]]
            if np.isfinite(w).all():
                cv2.circle(img, tuple(map(int, w)), 18, (0, 200, 255), 3)
            label = "THROW"
            if t.distance_ft:
                label += f" {t.distance_ft:.0f} ft"
            if t.velo_mph:
                label += f" ~{t.velo_mph:.0f} mph"
            src = foot_point(d.xy[[d.index_of(fi)]])[0]
            if t.receiver_id in dense:
                r = dense[t.receiver_id]
                dst = foot_point(r.xy[[r.index_of(fi)]])[0]
                if np.isfinite(src).all() and np.isfinite(dst).all():
                    cv2.arrowedLine(img, tuple(map(int, src)), tuple(map(int, dst)), (0, 200, 255), 2, tipLength=0.03)
            nose = d.xy[d.index_of(fi), KP["nose"]]
            if np.isfinite(nose).all():
                cv2.putText(img, label, (int(nose[0]) - 40, int(nose[1]) - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        out.write(img)
        fi += 1
    cap.release()
    out.release()
