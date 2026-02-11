#!/usr/bin/env python
"""
Test scene cut detection and segment classification.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filtering.scene_cropper import (
    detect_scene_cuts,
    classify_segments,
    crop_to_main_angle,
    visualize_segments,
)


def test_cut_detection(video_path: str):
    """Test just the cut detection step."""
    print(f"\n{'='*60}")
    print(f"Testing Cut Detection: {video_path}")
    print(f"{'='*60}")

    # Try different methods
    for method in ["histogram", "ssim"]:
        print(f"\nMethod: {method}")
        print("-" * 40)

        result = detect_scene_cuts(
            video_path,
            method=method,
            show_progress=True,
        )

        print(f"Total frames: {result.total_frames}")
        print(f"FPS: {result.fps:.2f}")
        print(f"Cuts detected: {len(result.cut_frames)}")
        print(f"Segments: {len(result.segments)}")

        for i, seg in enumerate(result.segments):
            print(f"  Segment {i}: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")

        if result.cut_scores:
            print(f"Cut scores: {[f'{s:.3f}' for s in result.cut_scores]}")


def test_full_pipeline(video_path: str, output_dir: str = None):
    """Test the full pipeline with visualization."""
    print(f"\n{'='*60}")
    print(f"Testing Full Pipeline: {video_path}")
    print(f"{'='*60}")

    if output_dir is None:
        output_dir = Path("F:/Claude_Projects/baseball-biomechanics/data/debug/scene_crop_test")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem

    # Run full pipeline
    result = crop_to_main_angle(
        video_path,
        output_path=str(output_dir / f"{video_name}_cropped.mp4"),
        keep_segments="first",
        detection_method="histogram",
        samples_per_segment=3,
        show_progress=True,
    )

    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Cuts detected: {result['detection']['num_cuts']}")
    print(f"  Segments: {len(result['segments'])}")

    print(f"\nSegment Classification:")
    for i, seg in enumerate(result["segments"]):
        label_marker = "[MAIN]" if seg["label"] == "main_angle" else "[other]"
        print(f"  {label_marker} Segment {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s "
              f"| {seg['label']} ({seg['confidence']:.2f})")

    print(f"\nCrop regions: {result['crop_regions']}")

    # Create visualization
    viz_path = output_dir / f"{video_name}_visualization.png"
    visualize_segments(video_path, result, str(viz_path))
    print(f"\nVisualization saved to: {viz_path}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test scene cropper")
    parser.add_argument("--video", type=str, help="Video path to test")
    parser.add_argument("--detection-only", action="store_true", help="Only test cut detection")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    args = parser.parse_args()

    # Default test video
    if args.video:
        video_path = args.video
    else:
        video_path = "F:/Claude_Projects/baseball-biomechanics/data/videos/2024/07/746122_20_1.mp4"

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return

    if args.detection_only:
        test_cut_detection(video_path)
    else:
        test_full_pipeline(video_path, args.output_dir)


if __name__ == "__main__":
    main()
