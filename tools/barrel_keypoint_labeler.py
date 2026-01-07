#!/usr/bin/env python
"""Labeling tool for bat barrel keypoints (3-point tracking).

Tracks 3 keypoints on the bat barrel:
  1. barrel_cap - end of the bat
  2. barrel_middle - middle of the barrel
  3. barrel_beginning - where barrel meets handle (roughly where hands end)

Usage:
    python tools/barrel_keypoint_labeler.py [--frames-dir PATH] [--output-dir PATH]

Controls:
    Left Click  - Place next keypoint (in order: cap -> middle -> beginning)
    Right Click - Remove last placed keypoint
    Drag        - Move existing keypoint (click near one to select)

    ENTER/n     - Next frame (save current annotation)
    s           - Skip frame (bat not visible or too blurry)
    f           - Flag as wrong camera angle (not_main_angle)
    b           - Go back to previous frame
    r           - Reset current frame's keypoints
    z           - Undo last action
    q/ESC       - Quit and save

Output:
    - YOLO pose format labels in output_dir/labels/
    - Annotations JSON with full metadata
    - Flagged frames log (for debugging angle classifier)
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict
import shutil

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data/bat_frames_round2_filtered"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/labels/bat_barrel"

# Keypoint definitions
KEYPOINT_NAMES = ["barrel_cap", "barrel_middle", "barrel_beginning"]
KEYPOINT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
KEYPOINT_RADIUS = 8
DRAG_THRESHOLD = 20  # Pixels to detect click near existing point

# Skeleton connections for visualization
SKELETON = [[0, 1], [1, 2]]  # cap-middle, middle-beginning


@dataclass
class FrameAnnotation:
    """Annotation for a single frame."""
    frame_path: str
    keypoints: List[Optional[Tuple[float, float]]] = field(default_factory=lambda: [None, None, None])
    status: str = "unlabeled"  # unlabeled, labeled, skipped, not_main_angle
    source_video: str = ""
    stadium: str = ""
    game_pk: str = ""
    timestamp: str = ""

    def is_complete(self) -> bool:
        """Check if all keypoints are placed."""
        return all(kp is not None for kp in self.keypoints)

    def to_yolo_pose(self, img_width: int, img_height: int) -> Optional[str]:
        """Convert to YOLO pose format: class x_center y_center width height kp1_x kp1_y kp1_v ..."""
        if not self.is_complete():
            return None

        # Calculate bounding box from keypoints
        xs = [kp[0] for kp in self.keypoints]
        ys = [kp[1] for kp in self.keypoints]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Add padding to bbox (20% of bat length on each side)
        bat_length = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
        padding = bat_length * 0.2

        x_min = max(0, x_min - padding)
        x_max = min(img_width, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(img_height, y_max + padding)

        # YOLO format: normalized x_center, y_center, width, height
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Keypoints: normalized x, y, visibility (2 = visible)
        kp_str = ""
        for kp in self.keypoints:
            kp_x = kp[0] / img_width
            kp_y = kp[1] / img_height
            kp_str += f" {kp_x:.6f} {kp_y:.6f} 2"

        # Class 0 for bat
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}{kp_str}"


class BarrelLabeler:
    """Interactive labeling tool for bat barrel keypoints."""

    def __init__(self, frames_dir: Path, output_dir: Path):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.labels_dir = output_dir / "labels"
        self.images_dir = output_dir / "images"

        # Create output directories
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load frames
        self.frames = sorted(frames_dir.glob("*.jpg"))
        print(f"Found {len(self.frames)} frames to label")

        # Load existing annotations
        self.annotations_file = output_dir / "annotations.json"
        self.flagged_file = output_dir / "flagged_frames.json"
        self.annotations: Dict[str, FrameAnnotation] = {}
        self.flagged_frames: List[dict] = []
        self._load_existing()

        # Current state
        self.current_idx = self._find_next_unlabeled()
        self.current_annotation: Optional[FrameAnnotation] = None
        self.dragging_point: Optional[int] = None
        self.history: List[Tuple[int, FrameAnnotation]] = []

        # Display
        self.window_name = "Bat Barrel Keypoint Labeler"
        self.img_display = None
        self.img_original = None
        self.img_width = 0
        self.img_height = 0

    def _load_existing(self):
        """Load existing annotations for resume support."""
        if self.annotations_file.exists():
            with open(self.annotations_file, "r") as f:
                data = json.load(f)
                for path, ann_dict in data.get("annotations", {}).items():
                    # Convert keypoints back to tuples
                    kps = ann_dict.get("keypoints", [None, None, None])
                    kps = [tuple(kp) if kp else None for kp in kps]
                    self.annotations[path] = FrameAnnotation(
                        frame_path=path,
                        keypoints=kps,
                        status=ann_dict.get("status", "unlabeled"),
                        source_video=ann_dict.get("source_video", ""),
                        stadium=ann_dict.get("stadium", ""),
                        game_pk=ann_dict.get("game_pk", ""),
                        timestamp=ann_dict.get("timestamp", "")
                    )

        if self.flagged_file.exists():
            with open(self.flagged_file, "r") as f:
                self.flagged_frames = json.load(f).get("flagged", [])

        labeled = sum(1 for a in self.annotations.values() if a.status == "labeled")
        skipped = sum(1 for a in self.annotations.values() if a.status == "skipped")
        flagged = sum(1 for a in self.annotations.values() if a.status == "not_main_angle")
        print(f"Loaded {labeled} labeled, {skipped} skipped, {flagged} flagged frames")

    def _find_next_unlabeled(self) -> int:
        """Find the first unlabeled frame."""
        for i, frame in enumerate(self.frames):
            path_str = str(frame)
            if path_str not in self.annotations or self.annotations[path_str].status == "unlabeled":
                return i
        return 0

    def _parse_frame_metadata(self, frame_path: Path) -> dict:
        """Extract metadata from frame filename."""
        # Format: Stadium_Teams_gamepk_atbat_pitch_fNNNN.jpg
        name = frame_path.stem
        parts = name.rsplit("_f", 1)

        if len(parts) == 2:
            prefix = parts[0]
            frame_num = parts[1]

            # Parse prefix
            prefix_parts = prefix.rsplit("_", 3)
            if len(prefix_parts) >= 4:
                stadium = "_".join(prefix_parts[:-3])
                teams = prefix_parts[-3]
                game_pk = prefix_parts[-2]
                at_bat_pitch = prefix_parts[-1] if len(prefix_parts) > 3 else ""

                return {
                    "stadium": stadium,
                    "teams": teams,
                    "game_pk": game_pk,
                    "source_video": f"{stadium}_{teams}_{game_pk}.mp4"
                }

        return {"stadium": "", "teams": "", "game_pk": "", "source_video": ""}

    def _save_annotations(self):
        """Save all annotations to disk."""
        # Convert to serializable format
        ann_dict = {}
        for path, ann in self.annotations.items():
            ann_data = asdict(ann)
            # Convert tuples to lists for JSON
            ann_data["keypoints"] = [list(kp) if kp else None for kp in ann.keypoints]
            ann_dict[path] = ann_data

        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "frames_dir": str(self.frames_dir),
                "keypoint_names": KEYPOINT_NAMES,
                "skeleton": SKELETON,
                "total_frames": len(self.frames),
                "labeled": sum(1 for a in self.annotations.values() if a.status == "labeled"),
                "skipped": sum(1 for a in self.annotations.values() if a.status == "skipped"),
                "flagged": sum(1 for a in self.annotations.values() if a.status == "not_main_angle"),
            },
            "annotations": ann_dict
        }

        with open(self.annotations_file, "w") as f:
            json.dump(data, f, indent=2)

        # Save flagged frames separately
        flagged_data = {
            "metadata": {
                "description": "Frames flagged as wrong camera angle for debugging",
                "created": datetime.now().isoformat(),
            },
            "flagged": self.flagged_frames
        }
        with open(self.flagged_file, "w") as f:
            json.dump(flagged_data, f, indent=2)

        print(f"Saved {len(self.annotations)} annotations")

    def _save_yolo_labels(self):
        """Export all labeled frames to YOLO pose format."""
        count = 0
        for path, ann in self.annotations.items():
            if ann.status == "labeled" and ann.is_complete():
                frame_path = Path(path)

                # Read image to get dimensions
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                # Generate YOLO label
                yolo_line = ann.to_yolo_pose(w, h)
                if yolo_line:
                    # Save label file
                    label_path = self.labels_dir / (frame_path.stem + ".txt")
                    with open(label_path, "w") as f:
                        f.write(yolo_line + "\n")

                    # Copy image to images dir
                    img_dest = self.images_dir / frame_path.name
                    if not img_dest.exists():
                        shutil.copy(frame_path, img_dest)

                    count += 1

        print(f"Exported {count} YOLO pose labels")

    def _draw_frame(self):
        """Draw the current frame with keypoints and UI."""
        if self.img_original is None:
            return

        self.img_display = self.img_original.copy()
        h, w = self.img_display.shape[:2]

        # Draw existing keypoints and skeleton
        if self.current_annotation:
            kps = self.current_annotation.keypoints

            # Draw skeleton lines first (behind points)
            for i, j in SKELETON:
                if kps[i] is not None and kps[j] is not None:
                    pt1 = (int(kps[i][0]), int(kps[i][1]))
                    pt2 = (int(kps[j][0]), int(kps[j][1]))
                    cv2.line(self.img_display, pt1, pt2, (255, 255, 0), 2)

            # Draw keypoints
            for i, kp in enumerate(kps):
                if kp is not None:
                    pt = (int(kp[0]), int(kp[1]))
                    color = KEYPOINT_COLORS[i]
                    cv2.circle(self.img_display, pt, KEYPOINT_RADIUS, color, -1)
                    cv2.circle(self.img_display, pt, KEYPOINT_RADIUS + 2, (255, 255, 255), 2)

                    # Label
                    label = KEYPOINT_NAMES[i].replace("barrel_", "")
                    cv2.putText(self.img_display, label, (pt[0] + 12, pt[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(self.img_display, label, (pt[0] + 12, pt[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw UI overlay
        overlay = self.img_display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        self.img_display = cv2.addWeighted(overlay, 0.7, self.img_display, 0.3, 0)

        # Frame info
        frame_path = self.frames[self.current_idx]
        meta = self._parse_frame_metadata(frame_path)

        # Stats
        labeled = sum(1 for a in self.annotations.values() if a.status == "labeled")
        skipped = sum(1 for a in self.annotations.values() if a.status == "skipped")
        flagged = sum(1 for a in self.annotations.values() if a.status == "not_main_angle")

        cv2.putText(self.img_display, f"Frame {self.current_idx + 1}/{len(self.frames)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.img_display, f"Labeled: {labeled}  Skipped: {skipped}  Flagged: {flagged}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(self.img_display, f"Stadium: {meta.get('stadium', 'Unknown')}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions
        next_kp = self._get_next_keypoint_to_place()
        if next_kp is not None:
            inst = f"Click to place: {KEYPOINT_NAMES[next_kp]}"
            color = KEYPOINT_COLORS[next_kp]
        else:
            inst = "All points placed! [ENTER] to save, [R] to reset"
            color = (100, 255, 100)

        cv2.putText(self.img_display, inst, (w - 400, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Controls hint
        cv2.putText(self.img_display, "[ENTER/N]ext [S]kip [F]lag [B]ack [R]eset [Q]uit",
                    (w - 450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Status indicator
        if self.current_annotation and self.current_annotation.status != "unlabeled":
            status = self.current_annotation.status.upper()
            status_color = {
                "labeled": (100, 255, 100),
                "skipped": (100, 100, 255),
                "not_main_angle": (0, 165, 255)
            }.get(self.current_annotation.status, (200, 200, 200))
            cv2.putText(self.img_display, f"[{status}]", (w - 150, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        # Bottom legend
        legend_y = h - 20
        cv2.putText(self.img_display, "Cap", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYPOINT_COLORS[0], 2)
        cv2.putText(self.img_display, "Middle", (60, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYPOINT_COLORS[1], 2)
        cv2.putText(self.img_display, "Beginning", (140, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYPOINT_COLORS[2], 2)

    def _get_next_keypoint_to_place(self) -> Optional[int]:
        """Get the index of the next keypoint to place."""
        if self.current_annotation is None:
            return 0
        for i, kp in enumerate(self.current_annotation.keypoints):
            if kp is None:
                return i
        return None

    def _find_nearest_keypoint(self, x: int, y: int) -> Optional[int]:
        """Find the nearest keypoint to click position."""
        if self.current_annotation is None:
            return None

        min_dist = float('inf')
        nearest = None

        for i, kp in enumerate(self.current_annotation.keypoints):
            if kp is not None:
                dist = np.sqrt((x - kp[0])**2 + (y - kp[1])**2)
                if dist < min_dist and dist < DRAG_THRESHOLD:
                    min_dist = dist
                    nearest = i

        return nearest

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near existing point (for dragging)
            nearest = self._find_nearest_keypoint(x, y)
            if nearest is not None:
                self.dragging_point = nearest
            else:
                # Place next keypoint
                next_kp = self._get_next_keypoint_to_place()
                if next_kp is not None and self.current_annotation:
                    self.current_annotation.keypoints[next_kp] = (float(x), float(y))
                    self._draw_frame()
                    cv2.imshow(self.window_name, self.img_display)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point is not None and self.current_annotation:
                self.current_annotation.keypoints[self.dragging_point] = (float(x), float(y))
                self._draw_frame()
                cv2.imshow(self.window_name, self.img_display)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last placed keypoint
            if self.current_annotation:
                for i in range(2, -1, -1):
                    if self.current_annotation.keypoints[i] is not None:
                        self.current_annotation.keypoints[i] = None
                        break
                self._draw_frame()
                cv2.imshow(self.window_name, self.img_display)

    def _load_frame(self, idx: int):
        """Load a frame and initialize its annotation."""
        if idx < 0 or idx >= len(self.frames):
            return False

        frame_path = self.frames[idx]
        self.img_original = cv2.imread(str(frame_path))

        if self.img_original is None:
            print(f"Could not load: {frame_path}")
            return False

        self.img_height, self.img_width = self.img_original.shape[:2]

        # Get or create annotation
        path_str = str(frame_path)
        if path_str in self.annotations:
            self.current_annotation = self.annotations[path_str]
        else:
            meta = self._parse_frame_metadata(frame_path)
            self.current_annotation = FrameAnnotation(
                frame_path=path_str,
                source_video=meta.get("source_video", ""),
                stadium=meta.get("stadium", ""),
                game_pk=meta.get("game_pk", "")
            )

        return True

    def _save_current(self, status: str = None):
        """Save current annotation."""
        if self.current_annotation is None:
            return

        if status:
            self.current_annotation.status = status
        elif self.current_annotation.is_complete():
            self.current_annotation.status = "labeled"

        self.current_annotation.timestamp = datetime.now().isoformat()
        self.annotations[self.current_annotation.frame_path] = self.current_annotation

        # If flagged, add to flagged list
        if status == "not_main_angle":
            flagged_entry = {
                "frame_path": self.current_annotation.frame_path,
                "source_video": self.current_annotation.source_video,
                "stadium": self.current_annotation.stadium,
                "game_pk": self.current_annotation.game_pk,
                "timestamp": self.current_annotation.timestamp
            }
            # Check if already in list
            existing = [f for f in self.flagged_frames
                       if f["frame_path"] == self.current_annotation.frame_path]
            if not existing:
                self.flagged_frames.append(flagged_entry)

    def run(self):
        """Main labeling loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n" + "="*60)
        print("BAT BARREL KEYPOINT LABELER")
        print("="*60)
        print("\nKeypoints to label (in order):")
        print("  1. barrel_cap (RED) - end of bat")
        print("  2. barrel_middle (GREEN) - middle of barrel")
        print("  3. barrel_beginning (BLUE) - where barrel meets handle")
        print("\nControls:")
        print("  Left Click  - Place keypoint / drag to move")
        print("  Right Click - Remove last keypoint")
        print("  ENTER/N     - Save and next frame")
        print("  S           - Skip (bat not visible)")
        print("  F           - Flag as wrong camera angle")
        print("  B           - Go back")
        print("  R           - Reset keypoints")
        print("  Q/ESC       - Quit and save")
        print("="*60 + "\n")

        while True:
            if not self._load_frame(self.current_idx):
                self.current_idx += 1
                if self.current_idx >= len(self.frames):
                    print("Reached end of frames!")
                    break
                continue

            self._draw_frame()
            cv2.imshow(self.window_name, self.img_display)

            key = cv2.waitKey(0) & 0xFF

            if key == 13 or key == ord('n'):  # ENTER or N
                if self.current_annotation.is_complete():
                    self._save_current("labeled")
                    print(f"[{self.current_idx+1}] LABELED: {Path(self.current_annotation.frame_path).name}")
                    self.history.append((self.current_idx, self.current_annotation))
                    self.current_idx += 1
                else:
                    print("Please place all 3 keypoints first!")

            elif key == ord('s'):  # Skip
                self._save_current("skipped")
                print(f"[{self.current_idx+1}] SKIPPED: {Path(self.current_annotation.frame_path).name}")
                self.history.append((self.current_idx, self.current_annotation))
                self.current_idx += 1

            elif key == ord('f'):  # Flag as wrong angle
                self._save_current("not_main_angle")
                print(f"[{self.current_idx+1}] FLAGGED: {Path(self.current_annotation.frame_path).name}")
                self.history.append((self.current_idx, self.current_annotation))
                self.current_idx += 1

            elif key == ord('b'):  # Back
                if self.history:
                    last_idx, _ = self.history.pop()
                    self.current_idx = last_idx
                    print(f"Going back to frame {self.current_idx + 1}")
                else:
                    print("No history to go back to")

            elif key == ord('r'):  # Reset
                if self.current_annotation:
                    self.current_annotation.keypoints = [None, None, None]
                    print("Reset keypoints")

            elif key == ord('q') or key == 27:  # Quit
                print("\nQuitting...")
                break

            # Check if we've reached the end
            if self.current_idx >= len(self.frames):
                print("\nReached end of frames!")
                break

        cv2.destroyAllWindows()

        # Save everything
        self._save_annotations()
        self._save_yolo_labels()

        # Print summary
        labeled = sum(1 for a in self.annotations.values() if a.status == "labeled")
        skipped = sum(1 for a in self.annotations.values() if a.status == "skipped")
        flagged = sum(1 for a in self.annotations.values() if a.status == "not_main_angle")

        print("\n" + "="*60)
        print("SESSION COMPLETE")
        print("="*60)
        print(f"  Labeled:  {labeled}")
        print(f"  Skipped:  {skipped}")
        print(f"  Flagged:  {flagged}")
        print(f"  Total:    {labeled + skipped + flagged}")
        print(f"\nOutput saved to: {self.output_dir}")
        print(f"  - annotations.json (full metadata)")
        print(f"  - flagged_frames.json (for debugging)")
        print(f"  - labels/ (YOLO pose format)")
        print(f"  - images/ (copies of labeled frames)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Label bat barrel keypoints")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
                        help="Directory containing frames to label")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for labels")
    args = parser.parse_args()

    labeler = BarrelLabeler(args.frames_dir, args.output_dir)
    labeler.run()


if __name__ == "__main__":
    main()
