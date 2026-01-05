"""
Launch the mitt labeler with the diverse stadium frames.
Simple wrapper to load frames from the diverse_frames directory.
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mitt_labeler import MittLabeler


def main():
    project_dir = Path("F:/Claude_Projects/baseball-biomechanics")
    frames_dir = project_dir / "data" / "labels" / "diverse_frames"

    # Create labeler with custom output dir
    labeler = MittLabeler(str(project_dir))
    labeler.output_dir = frames_dir

    # Load frames from diverse_frames directory
    frames_info_path = frames_dir / "frames_info.json"
    if frames_info_path.exists():
        with open(frames_info_path) as f:
            labeler.frames = json.load(f)

        # Add IDs if missing
        for i, frame in enumerate(labeler.frames):
            if "id" not in frame:
                frame["id"] = i

        print(f"Loaded {len(labeler.frames)} diverse frames from 30 MLB stadiums")
    else:
        print(f"Error: No frames found at {frames_info_path}")
        print("Run scrape_diverse_dataset.py first to extract frames.")
        return

    # Load existing annotations if any
    ann_path = frames_dir / "annotations.json"
    if ann_path.exists():
        from mitt_labeler import BoxAnnotation
        with open(ann_path) as f:
            data = json.load(f)
            for frame_id, boxes in data.items():
                labeler.annotations[frame_id] = [
                    BoxAnnotation(**b) for b in boxes
                ]
        print(f"Loaded existing annotations for {len(labeler.annotations)} frames")

    # Run the labeler
    labeler.run()


if __name__ == "__main__":
    main()
