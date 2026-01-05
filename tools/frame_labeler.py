"""
Interactive frame labeling tool for SAM 2 fine-tuning.

Samples random frames across multiple videos for variety.
Click to mark batter and catcher's glove positions.
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def sample_frames_from_videos(
    video_dir: str,
    num_frames: int = 20,
    output_dir: str = "data/labels/frames",
    seed: int = 42
) -> list:
    """
    Sample random frames from multiple videos.

    Args:
        video_dir: Directory containing videos
        num_frames: Total number of frames to sample
        output_dir: Where to save extracted frames
        seed: Random seed for reproducibility

    Returns:
        List of dicts with frame info
    """
    random.seed(seed)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all videos
    videos = list(video_dir.rglob("*.mp4"))
    # Exclude processed videos
    videos = [v for v in videos if "processed" not in str(v)]

    if len(videos) < num_frames:
        print(f"Only {len(videos)} videos found, using all of them")
        selected_videos = videos
    else:
        selected_videos = random.sample(videos, num_frames)

    frames_info = []

    print(f"Extracting {len(selected_videos)} frames from different videos...")

    for i, video_path in enumerate(selected_videos):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 10:
            cap.release()
            continue

        # Sample from middle 60% of video (avoid start/end artifacts)
        start = int(total_frames * 0.2)
        end = int(total_frames * 0.8)
        frame_idx = random.randint(start, end)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Save frame as image
        frame_name = f"frame_{i:03d}_{video_path.stem}.png"
        frame_path = output_dir / frame_name
        cv2.imwrite(str(frame_path), frame)

        frames_info.append({
            "frame_id": i,
            "frame_path": str(frame_path),
            "source_video": str(video_path),
            "source_frame_idx": frame_idx,
            "total_frames": total_frames,
        })

        print(f"  [{i+1}/{len(selected_videos)}] {video_path.name} -> frame {frame_idx}")

    # Save frame info
    info_path = output_dir.parent / "frames_info.json"
    with open(info_path, "w") as f:
        json.dump(frames_info, f, indent=2)

    print(f"\nExtracted {len(frames_info)} frames to {output_dir}")
    return frames_info


class MultiFrameLabeler:
    """Interactive tool for labeling sampled frames."""

    def __init__(self, frames_dir: str, labels_file: str = None):
        self.frames_dir = Path(frames_dir)

        # Load frame images
        self.frame_paths = sorted(self.frames_dir.glob("*.png"))
        if not self.frame_paths:
            raise ValueError(f"No frames found in {frames_dir}")

        self.num_frames = len(self.frame_paths)
        self.current_idx = 0
        self.current_frame = None

        # Labels: frame_path -> {"batter": [[x,y], ...], "glove": [[x,y], ...]}
        self.labels = {}
        self.current_mode = "batter"
        self.modes = ["batter", "glove"]
        self.mode_colors = {"batter": (0, 255, 0), "glove": (255, 0, 255)}

        # Labels file
        self.labels_file = Path(labels_file) if labels_file else self.frames_dir.parent / "labels.json"
        self._load_labels()

        self.window_name = "Frame Labeler (H=help, Q=quit)"

    def _load_labels(self):
        """Load existing labels."""
        if self.labels_file.exists():
            with open(self.labels_file) as f:
                self.labels = json.load(f)
            print(f"Loaded labels for {len(self.labels)} frames")

    def _save_labels(self):
        """Save labels to file."""
        with open(self.labels_file, "w") as f:
            json.dump(self.labels, f, indent=2)
        print(f"Saved labels to {self.labels_file}")

    def _get_frame_key(self) -> str:
        """Get current frame's key for labels dict."""
        return self.frame_paths[self.current_idx].name

    def _load_current_frame(self):
        """Load the current frame image."""
        self.current_frame = cv2.imread(str(self.frame_paths[self.current_idx]))

    def _draw_ui(self) -> np.ndarray:
        """Draw UI overlay on frame."""
        display = self.current_frame.copy()
        h, w = display.shape[:2]
        frame_key = self._get_frame_key()

        # Draw existing labels
        if frame_key in self.labels:
            for mode, points in self.labels[frame_key].items():
                color = self.mode_colors.get(mode, (255, 255, 255))
                for i, (x, y) in enumerate(points):
                    cv2.circle(display, (x, y), 10, color, -1)
                    cv2.circle(display, (x, y), 12, (255, 255, 255), 2)
                    cv2.putText(
                        display, f"{mode[0].upper()}{i+1}", (x + 15, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

        # Status bar
        bar_h = 70
        cv2.rectangle(display, (0, 0), (w, bar_h), (30, 30, 30), -1)

        # Frame counter
        cv2.putText(
            display, f"Frame {self.current_idx + 1}/{self.num_frames}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Source file
        cv2.putText(
            display, frame_key[:40], (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
        )

        # Current mode
        mode_color = self.mode_colors[self.current_mode]
        cv2.putText(
            display, f"Mode: {self.current_mode.upper()}", (300, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2
        )

        # Label status
        labeled = len([k for k in self.labels if self.labels[k].get("batter") or self.labels[k].get("glove")])
        cv2.putText(
            display, f"Labeled: {labeled}/{self.num_frames}", (500, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # This frame status
        if frame_key in self.labels:
            has_batter = bool(self.labels[frame_key].get("batter"))
            has_glove = bool(self.labels[frame_key].get("glove"))
            if has_batter and has_glove:
                status, color = "COMPLETE", (0, 255, 0)
            else:
                missing = []
                if not has_batter: missing.append("batter")
                if not has_glove: missing.append("glove")
                status, color = f"need: {', '.join(missing)}", (0, 200, 255)
        else:
            status, color = "unlabeled", (128, 128, 128)

        cv2.putText(
            display, status, (300, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        # Instructions
        cv2.putText(
            display, "A/D=prev/next | 1=batter 2=glove | Click=add | RightClick=remove | Q=save+quit",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
        )

        return display

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        frame_key = self._get_frame_key()

        if event == cv2.EVENT_LBUTTONDOWN:
            # Initialize labels for this frame if needed
            if frame_key not in self.labels:
                self.labels[frame_key] = {"batter": [], "glove": []}

            self.labels[frame_key][self.current_mode].append([x, y])
            print(f"Added {self.current_mode} at ({x}, {y})")
            self._update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if frame_key in self.labels:
                points = self.labels[frame_key][self.current_mode]
                if points:
                    removed = points.pop()
                    print(f"Removed {self.current_mode} point")
                    self._update_display()

    def _update_display(self):
        """Refresh the display."""
        display = self._draw_ui()
        cv2.imshow(self.window_name, display)

    def run(self):
        """Main loop."""
        print(f"\n=== Multi-Frame Labeler ===")
        print(f"Frames to label: {self.num_frames}")
        print(f"Labels file: {self.labels_file}")
        print("\nControls:")
        print("  A/D or Left/Right = Navigate frames")
        print("  1 = Batter mode, 2 = Glove mode")
        print("  Left click = Add point")
        print("  Right click = Remove last point")
        print("  C = Clear frame labels")
        print("  Q = Save and quit\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._load_current_frame()
        self._update_display()

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                self._save_labels()
                break

            # Navigation
            elif key == ord('a') or key == 81:  # Left
                self.current_idx = max(0, self.current_idx - 1)
                self._load_current_frame()
                self._update_display()

            elif key == ord('d') or key == 83:  # Right
                self.current_idx = min(self.num_frames - 1, self.current_idx + 1)
                self._load_current_frame()
                self._update_display()

            # Mode switching
            elif key == ord('1'):
                self.current_mode = "batter"
                print("Mode: BATTER")
                self._update_display()

            elif key == ord('2'):
                self.current_mode = "glove"
                print("Mode: GLOVE")
                self._update_display()

            # Clear
            elif key == ord('c'):
                frame_key = self._get_frame_key()
                if frame_key in self.labels:
                    del self.labels[frame_key]
                    print("Cleared labels for this frame")
                    self._update_display()

        cv2.destroyAllWindows()

        # Summary
        print("\n=== Summary ===")
        complete = 0
        for key, lbls in self.labels.items():
            if lbls.get("batter") and lbls.get("glove"):
                complete += 1
        print(f"Fully labeled frames: {complete}/{self.num_frames}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sample and label frames for SAM 2")
    parser.add_argument("--video-dir", default="data/videos", help="Video directory")
    parser.add_argument("--num-frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--output-dir", default="data/labels", help="Output directory")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction, use existing frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    video_dir = project_root / args.video_dir
    output_dir = project_root / args.output_dir
    frames_dir = output_dir / "frames"

    # Extract frames if needed
    if not args.skip_extract:
        if frames_dir.exists() and list(frames_dir.glob("*.png")):
            print(f"Frames already exist in {frames_dir}")
            print("Use --skip-extract to reuse them, or delete the folder to re-extract")
            response = input("Re-extract frames? (y/n): ").strip().lower()
            if response != 'y':
                args.skip_extract = True

        if not args.skip_extract:
            sample_frames_from_videos(
                str(video_dir),
                num_frames=args.num_frames,
                output_dir=str(frames_dir),
                seed=args.seed
            )

    # Run labeler
    if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
        print(f"No frames found in {frames_dir}. Run without --skip-extract first.")
        sys.exit(1)

    labeler = MultiFrameLabeler(str(frames_dir))
    labeler.run()


if __name__ == "__main__":
    main()
