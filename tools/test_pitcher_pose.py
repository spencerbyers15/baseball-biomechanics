"""Test pitcher identification and pose extraction across videos.

Usage:
    # Single video test
    python tools/test_pitcher_pose.py --video data/videos/2023_cropped/Dodger_Stadium/LAD_SF_716439_33_2_cropped.mp4

    # Single video with stadium zone
    python tools/test_pitcher_pose.py --video path/to/video.mp4 --stadium "Dodger Stadium"

    # Batch test across all 2023 cropped stadiums
    python tools/test_pitcher_pose.py --batch

    # Batch with limited videos per stadium
    python tools/test_pitcher_pose.py --batch --per-stadium 1

    # Batch without temporal smoothing (for comparison)
    python tools/test_pitcher_pose.py --batch --no-temporal
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.detection.player_pose import PlayerPoseDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = project_root / "data" / "debug" / "pitcher_pose_test"


def test_single_video(
    detector: PlayerPoseDetector,
    video_path: str,
    save_every: int = 10,
    max_frames: int = 0,
    use_temporal: bool = True,
) -> dict:
    """Run pitcher detection on every frame of a video.

    Args:
        detector: PlayerPoseDetector instance.
        video_path: Path to video file.
        save_every: Save annotated frame every N frames.
        max_frames: Stop after N frames (0 = all).
        use_temporal: Use temporal smoothing for pitcher selection.

    Returns:
        Dict with detection stats.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return {"error": "cannot_open"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    video_name = Path(video_path).stem
    video_out_dir = OUTPUT_DIR / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    detected_count = 0
    pose_valid_count = 0
    frame_num = 0
    saved_frames = []

    # Reset temporal state for each new video
    detector.reset_temporal()

    logger.info(f"Processing {video_name}: {total_frames} frames @ {fps:.1f} fps")

    while frame_num < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_frame(frame, frame_num, use_temporal=use_temporal)
        pitcher = results.get("pitcher")

        if pitcher is not None:
            detected_count += 1
            if pitcher["pose"] and pitcher["pose"].is_valid and len(pitcher["pose"].keypoints) > 0:
                pose_valid_count += 1

        # Save annotated frame periodically
        if frame_num % save_every == 0:
            annotated = detector.visualize(frame, results)

            # Add frame info text
            info = f"Frame {frame_num}"
            if pitcher:
                info += f" | Pitcher conf={pitcher['conf']:.2f}"
                if pitcher["pose"] and pitcher["pose"].is_valid:
                    n_kp = sum(1 for kp in pitcher["pose"].keypoints if kp.confidence > 0.3)
                    info += f" | {n_kp} keypoints"
            else:
                info += " | No pitcher"

            cv2.putText(
                annotated, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            out_path = video_out_dir / f"frame_{frame_num:05d}.jpg"
            cv2.imwrite(str(out_path), annotated)
            saved_frames.append(str(out_path))

        frame_num += 1

        if frame_num % 100 == 0:
            det_rate = detected_count / frame_num * 100
            logger.info(f"  Frame {frame_num}/{total_frames} — det rate: {det_rate:.1f}%")

    cap.release()

    det_rate = detected_count / max(frame_num, 1) * 100
    pose_rate = pose_valid_count / max(frame_num, 1) * 100

    # Build montage of saved frames (up to 12)
    montage_frames = saved_frames[:12]
    if montage_frames:
        _build_montage(montage_frames, video_out_dir / "montage.jpg")

    stats = {
        "video": video_name,
        "total_frames": frame_num,
        "pitcher_detected": detected_count,
        "pose_valid": pose_valid_count,
        "detection_rate": round(det_rate, 1),
        "pose_rate": round(pose_rate, 1),
        "saved_frames": len(saved_frames),
    }

    logger.info(
        f"  {video_name}: detection={det_rate:.1f}%, pose={pose_rate:.1f}% "
        f"({detected_count}/{frame_num} frames)"
    )

    return stats


def _build_montage(frame_paths: list, output_path: Path, cols: int = 4, thumb_w: int = 320):
    """Build a grid montage from saved annotated frames."""
    images = []
    for p in frame_paths:
        img = cv2.imread(p)
        if img is not None:
            h, w = img.shape[:2]
            scale = thumb_w / w
            thumb = cv2.resize(img, (thumb_w, int(h * scale)))
            images.append(thumb)

    if not images:
        return

    # Pad to uniform height
    max_h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    # Build rows
    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))

    montage = np.vstack(rows)
    cv2.imwrite(str(output_path), montage)
    logger.info(f"  Montage saved: {output_path}")


def stadium_name_from_dir(dir_name: str) -> str:
    """Convert directory name back to stadium name for zone lookup.

    E.g. "Dodger_Stadium" -> "Dodger Stadium"
    """
    return dir_name.replace("_", " ")


def find_cropped_videos(base_dir: Path, per_stadium: int = 0) -> list:
    """Find all cropped videos organized by stadium.

    Args:
        base_dir: Path to 2023_cropped directory.
        per_stadium: Max videos per stadium (0 = all).

    Returns:
        List of (stadium_name, video_path) tuples.
    """
    videos = []
    for stadium_dir in sorted(base_dir.iterdir()):
        if not stadium_dir.is_dir():
            continue
        stadium_name = stadium_dir.name
        mp4s = sorted(stadium_dir.glob("*.mp4"))
        if per_stadium > 0:
            mp4s = mp4s[:per_stadium]
        for mp4 in mp4s:
            videos.append((stadium_name, str(mp4)))
    return videos


def run_batch(
    detector: PlayerPoseDetector,
    per_stadium: int = 1,
    max_frames: int = 300,
    use_temporal: bool = True,
):
    """Run batch test across all stadiums."""
    cropped_dir = project_root / "data" / "videos" / "2023_cropped"
    if not cropped_dir.exists():
        logger.error(f"Cropped videos directory not found: {cropped_dir}")
        return

    videos = find_cropped_videos(cropped_dir, per_stadium=per_stadium)
    logger.info(f"Found {len(videos)} videos across {len(set(v[0] for v in videos))} stadiums")

    all_stats = []
    for stadium_dir_name, video_path in videos:
        # Auto-detect stadium from directory name and set zone
        stadium_name = stadium_name_from_dir(stadium_dir_name)
        detector.set_stadium(stadium_name)

        logger.info(f"\n--- {stadium_name} ---")
        stats = test_single_video(
            detector, video_path,
            save_every=10, max_frames=max_frames,
            use_temporal=use_temporal,
        )
        stats["stadium"] = stadium_dir_name
        all_stats.append(stats)

    # Summary
    valid = [s for s in all_stats if "error" not in s]
    if not valid:
        logger.error("No videos processed successfully")
        return

    det_rates = [s["detection_rate"] for s in valid]
    pose_rates = [s["pose_rate"] for s in valid]

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Videos tested: {len(valid)}")
    print(f"Stadiums: {len(set(s['stadium'] for s in valid))}")
    print(f"Temporal smoothing: {'ON' if use_temporal else 'OFF'}")
    print(f"\nPitcher Detection Rate:")
    print(f"  Mean:  {np.mean(det_rates):.1f}%")
    print(f"  Min:   {np.min(det_rates):.1f}%")
    print(f"  Max:   {np.max(det_rates):.1f}%")
    print(f"  Median: {np.median(det_rates):.1f}%")
    print(f"\nPose Valid Rate:")
    print(f"  Mean:  {np.mean(pose_rates):.1f}%")
    print(f"  Min:   {np.min(pose_rates):.1f}%")
    print(f"  Max:   {np.max(pose_rates):.1f}%")

    # Worst performers
    valid_sorted = sorted(valid, key=lambda s: s["detection_rate"])
    print(f"\nWorst 5 (by detection rate):")
    for s in valid_sorted[:5]:
        print(f"  {s['stadium']:30s} {s['video']:40s} det={s['detection_rate']:5.1f}% pose={s['pose_rate']:5.1f}%")

    print(f"\nBest 5:")
    for s in valid_sorted[-5:]:
        print(f"  {s['stadium']:30s} {s['video']:40s} det={s['detection_rate']:5.1f}% pose={s['pose_rate']:5.1f}%")

    # Save results JSON
    results_path = OUTPUT_DIR / "batch_results.json"
    with open(results_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Test pitcher pose detection")
    parser.add_argument("--video", type=str, help="Path to a single video to test")
    parser.add_argument("--batch", action="store_true", help="Run batch across 2023 cropped videos")
    parser.add_argument("--per-stadium", type=int, default=1, help="Max videos per stadium in batch mode")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per video (0=all)")
    parser.add_argument("--save-every", type=int, default=10, help="Save annotated frame every N frames")
    parser.add_argument("--stadium", type=str, help="Stadium name for zone lookup (e.g. 'Dodger Stadium')")
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--zones-path", type=str, help="Path to pitcher_zones.json (default: data/pitcher_zones.json)")
    args = parser.parse_args()

    if not args.video and not args.batch:
        parser.error("Specify --video or --batch")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize detector with optional zones path
    detector_kwargs = {}
    if args.zones_path:
        detector_kwargs["pitcher_zones_path"] = args.zones_path
    if args.stadium:
        detector_kwargs["stadium"] = args.stadium

    detector = PlayerPoseDetector(**detector_kwargs)

    use_temporal = not args.no_temporal

    if args.video:
        stats = test_single_video(
            detector, args.video,
            save_every=args.save_every,
            max_frames=args.max_frames,
            use_temporal=use_temporal,
        )
        print(f"\nResults: {json.dumps(stats, indent=2)}")

    if args.batch:
        run_batch(
            detector,
            per_stadium=args.per_stadium,
            max_frames=args.max_frames,
            use_temporal=use_temporal,
        )

    detector.cleanup()


if __name__ == "__main__":
    main()
