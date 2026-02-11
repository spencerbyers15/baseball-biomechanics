#!/usr/bin/env python
"""Test scene cropper on multiple videos from different stadiums."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filtering.scene_cropper import crop_to_main_angle, visualize_segments

OUTPUT_DIR = Path("F:/Claude_Projects/baseball-biomechanics/data/debug/scene_crop_test")

VIDEOS = [
    ("Fenway_Park", "F:/Claude_Projects/baseball-biomechanics/data/videos/2023_round2/Fenway_Park/BOS_TB_716419_6_5.mp4"),
    ("Yankee_Stadium", "F:/Claude_Projects/baseball-biomechanics/data/videos/2023_round2/Yankee_Stadium/NYY_AZ_716471_10_7.mp4"),
    ("Wrigley_Field", "F:/Claude_Projects/baseball-biomechanics/data/videos/2023_round2/Wrigley_Field/CHC_COL_716449_14_5.mp4"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for stadium, video_path in VIDEOS:
        print(f"\n{'='*60}")
        print(f"Stadium: {stadium}")
        print(f"Video: {Path(video_path).name}")
        print(f"{'='*60}")

        video_name = Path(video_path).stem

        result = crop_to_main_angle(
            video_path,
            str(OUTPUT_DIR / f"{video_name}_cropped.mp4"),
            keep_segments="longest",
            detection_method="histogram",
            samples_per_segment=3,
            show_progress=True,
        )

        print(f"\nResults:")
        print(f"  Cuts detected: {result['detection']['num_cuts']}")
        print(f"  Segments: {len(result['segments'])}")

        print(f"\nSegment Classification:")
        for i, seg in enumerate(result["segments"]):
            label_marker = "[MAIN]" if seg["label"] == "main_angle" else "[other]"
            print(f"  {label_marker} Seg {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s "
                  f"({seg['duration']:.2f}s) | {seg['confidence']:.2f}")

        main_segs = [s for s in result["segments"] if s["label"] == "main_angle"]
        other_segs = [s for s in result["segments"] if s["label"] == "other"]
        print(f"\nSummary: {len(main_segs)} main_angle, {len(other_segs)} other")
        print(f"Crop region: {result['crop_regions']}")

        # Save visualization
        viz_path = OUTPUT_DIR / f"{video_name}_visualization.png"
        visualize_segments(video_path, result, str(viz_path))
        print(f"Visualization: {viz_path}")


if __name__ == "__main__":
    main()
