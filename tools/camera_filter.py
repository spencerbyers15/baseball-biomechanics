#!/usr/bin/env python
"""CLI for camera angle filtering using embedding similarity."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.filtering import CameraAngleFilter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def cmd_build_reference(args):
    """Build reference embeddings from images."""
    filter = CameraAngleFilter(model_type=args.model)
    filter.build_reference(
        image_dir=args.image_dir,
        output_path=args.output,
        method=args.method,
    )
    print(f"Reference embeddings saved to {args.output}")


def cmd_classify(args):
    """Classify video frames by camera angle."""
    filter = CameraAngleFilter(model_type=args.model)
    filter.load_reference(args.reference)

    results = filter.classify_video(
        video_path=args.video,
        threshold=args.threshold,
        sample_fps=args.fps,
    )

    # Print summary
    n_main = sum(1 for r in results if r.is_main_angle)
    print(f"\nResults: {n_main}/{len(results)} frames classified as main angle")

    # Find segments
    segments = filter.find_main_segments(results)
    print(f"Found {len(segments)} main angle segments:")
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i + 1}: {start:.2f}s - {end:.2f}s ({end - start:.2f}s)")

    # Visualize if requested
    if args.visualize:
        viz_path = Path(args.video).with_suffix(".viz.png")
        filter.visualize(args.video, results, str(viz_path))
        print(f"Visualization saved to {viz_path}")


def cmd_crop(args):
    """Crop video to main angle segments."""
    filter = CameraAngleFilter(model_type=args.model)
    filter.load_reference(args.reference)

    # Classify
    results = filter.classify_video(
        video_path=args.video,
        threshold=args.threshold,
        sample_fps=args.fps,
    )

    # Find segments
    segments = filter.find_main_segments(
        results,
        min_segment_sec=args.min_segment,
        gap_tolerance_sec=args.gap_tolerance,
    )

    if not segments:
        print("No main angle segments found!")
        return

    print(f"Found {len(segments)} segments, cropping...")

    filter.crop_video(
        video_path=args.video,
        output_path=args.output,
        segments=segments,
        merge=not args.no_merge,
    )
    print(f"Cropped video saved to {args.output}")


def cmd_batch(args):
    """Process multiple videos."""
    filter = CameraAngleFilter(model_type=args.model)
    filter.load_reference(args.reference)

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(video_dir.glob("*.mp4"))
    print(f"Processing {len(videos)} videos...")

    for video_path in videos:
        print(f"\n--- {video_path.name} ---")
        try:
            results = filter.classify_video(
                video_path=str(video_path),
                threshold=args.threshold,
                sample_fps=args.fps,
                show_progress=False,
            )

            segments = filter.find_main_segments(results)
            if segments:
                output_path = output_dir / video_path.name
                filter.crop_video(str(video_path), str(output_path), segments)
                print(f"  Saved: {output_path}")
            else:
                print(f"  Skipped: no main angle segments")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Camera angle filtering for baseball videos")
    parser.add_argument("--model", choices=["clip", "dinov2"], default="clip",
                        help="Embedding model to use")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-reference
    p_build = subparsers.add_parser("build-reference", help="Build reference embeddings")
    p_build.add_argument("--image-dir", required=True, help="Directory of reference images")
    p_build.add_argument("--output", "-o", default="reference_embeddings.pkl",
                         help="Output file for embeddings")
    p_build.add_argument("--method", choices=["average", "all"], default="average",
                         help="Embedding method: average or keep all")
    p_build.set_defaults(func=cmd_build_reference)

    # classify
    p_classify = subparsers.add_parser("classify", help="Classify video frames")
    p_classify.add_argument("--video", "-v", required=True, help="Video file to classify")
    p_classify.add_argument("--reference", "-r", required=True, help="Reference embeddings file")
    p_classify.add_argument("--threshold", "-t", type=float, default=0.85,
                            help="Similarity threshold (default: 0.85)")
    p_classify.add_argument("--fps", type=float, default=5.0,
                            help="Frames per second to sample (default: 5)")
    p_classify.add_argument("--visualize", action="store_true",
                            help="Save visualization image")
    p_classify.set_defaults(func=cmd_classify)

    # crop
    p_crop = subparsers.add_parser("crop", help="Crop video to main angle segments")
    p_crop.add_argument("--video", "-v", required=True, help="Video file to crop")
    p_crop.add_argument("--reference", "-r", required=True, help="Reference embeddings file")
    p_crop.add_argument("--output", "-o", required=True, help="Output video file")
    p_crop.add_argument("--threshold", "-t", type=float, default=0.85,
                        help="Similarity threshold")
    p_crop.add_argument("--fps", type=float, default=5.0,
                        help="Sample FPS for classification")
    p_crop.add_argument("--min-segment", type=float, default=1.0,
                        help="Minimum segment duration in seconds")
    p_crop.add_argument("--gap-tolerance", type=float, default=0.5,
                        help="Gap tolerance for merging segments")
    p_crop.add_argument("--no-merge", action="store_true",
                        help="Don't merge segments, output separate files")
    p_crop.set_defaults(func=cmd_crop)

    # batch
    p_batch = subparsers.add_parser("batch", help="Process multiple videos")
    p_batch.add_argument("--video-dir", required=True, help="Directory of videos")
    p_batch.add_argument("--output-dir", required=True, help="Output directory")
    p_batch.add_argument("--reference", "-r", required=True, help="Reference embeddings file")
    p_batch.add_argument("--threshold", "-t", type=float, default=0.85)
    p_batch.add_argument("--fps", type=float, default=5.0)
    p_batch.set_defaults(func=cmd_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
