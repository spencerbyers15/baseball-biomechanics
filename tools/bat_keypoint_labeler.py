#!/usr/bin/env python
"""
Bat Keypoint Labeler for YOLO-Pose Training.

Labeling workflow:
- View frames with batter visible
- Click KNOB (handle end) first, then CAP (barrel end)
- Each bat needs exactly 2 keypoints
- Export in YOLO-pose format for training

Target: ~300 labeled frames for bat pose detection.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image, ImageTk


@dataclass
class BatAnnotation:
    """Bat keypoint annotation with knob and cap positions."""
    knob_x: int  # Handle end
    knob_y: int
    cap_x: int   # Barrel end
    cap_y: int
    knob_visible: bool = True
    cap_visible: bool = True

    def get_bbox(self, padding: int = 20) -> Tuple[int, int, int, int]:
        """Get bounding box around bat with padding."""
        x1 = min(self.knob_x, self.cap_x) - padding
        y1 = min(self.knob_y, self.cap_y) - padding
        x2 = max(self.knob_x, self.cap_x) + padding
        y2 = max(self.knob_y, self.cap_y) + padding
        return (max(0, x1), max(0, y1), x2, y2)

    def to_yolo_pose(self, img_width: int, img_height: int, padding: int = 20) -> str:
        """
        Convert to YOLO-pose format.
        Format: class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
        """
        x1, y1, x2, y2 = self.get_bbox(padding)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        # Normalized bbox
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        # Normalized keypoints
        kp1_x = self.knob_x / img_width
        kp1_y = self.knob_y / img_height
        kp1_v = 2 if self.knob_visible else 1  # 2=visible, 1=occluded, 0=not labeled

        kp2_x = self.cap_x / img_width
        kp2_y = self.cap_y / img_height
        kp2_v = 2 if self.cap_visible else 1

        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp1_x:.6f} {kp1_y:.6f} {kp1_v} {kp2_x:.6f} {kp2_y:.6f} {kp2_v}"


class BatKeypointLabeler:
    """Interactive labeler for bat keypoints (knob + cap)."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.video_dirs = [
            self.project_dir / "data" / "videos" / "2024",
            self.project_dir / "data" / "videos" / "diverse_stadiums",
        ]
        self.output_dir = self.project_dir / "data" / "labels" / "bat_keypoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames: List[Dict] = []
        self.annotations: Dict[str, List[BatAnnotation]] = {}
        self.current_idx = 0

        # Keypoint state: None, "knob", or "cap"
        self.current_keypoint = None
        self.temp_knob = None  # Temporary knob point while waiting for cap

        self.scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

    def extract_frames(self, num_frames: int = 300, append: bool = False):
        """Extract frames for labeling, focusing on pre-pitch batter stance."""
        all_videos = []
        for video_dir in self.video_dirs:
            if video_dir.exists():
                all_videos.extend(list(video_dir.rglob("*.mp4")))

        if not all_videos:
            print("No videos found")
            return

        start_idx = 0
        if append:
            self.load_existing()
            start_idx = len(self.frames)
            print(f"Appending to existing {start_idx} frames")

        import random
        random.seed(123 + start_idx)

        samples_per_video = max(1, num_frames // len(all_videos) + 1)
        selected_videos = all_videos * samples_per_video
        random.shuffle(selected_videos)
        selected_videos = selected_videos[:num_frames]

        print(f"Extracting {len(selected_videos)} frames for labeling...")
        (self.output_dir / "frames").mkdir(exist_ok=True)

        for i, video_path in enumerate(selected_videos):
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total < 30:
                cap.release()
                continue

            # Sample from early frames (pre-pitch, bat visible in stance)
            frame_idx = random.randint(int(total * 0.1), int(total * 0.4))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            global_idx = start_idx + i
            frame_name = f"bat_{global_idx:04d}_{video_path.stem}_f{frame_idx}.png"
            frame_path = self.output_dir / "frames" / frame_name
            cv2.imwrite(str(frame_path), frame)

            self.frames.append({
                "id": global_idx,
                "path": str(frame_path),
                "source_video": str(video_path),
                "frame_idx": frame_idx,
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(selected_videos)}] extracted...")

        with open(self.output_dir / "frames_info.json", "w") as f:
            json.dump(self.frames, f, indent=2)

        print(f"\nExtracted {len(self.frames)} frames to {self.output_dir / 'frames'}")

    def load_existing(self):
        """Load existing frames and annotations."""
        frames_info = self.output_dir / "frames_info.json"
        annotations_file = self.output_dir / "annotations.json"

        if frames_info.exists():
            with open(frames_info) as f:
                self.frames = json.load(f)
            print(f"Loaded {len(self.frames)} frames")

        if annotations_file.exists():
            with open(annotations_file) as f:
                data = json.load(f)
                for frame_id, bats in data.items():
                    self.annotations[frame_id] = [
                        BatAnnotation(**b) for b in bats
                    ]
            print(f"Loaded annotations for {len(self.annotations)} frames")

    def save_annotations(self):
        """Save annotations to JSON and YOLO-pose format."""
        data = {}
        for frame_id, bats in self.annotations.items():
            data[frame_id] = [asdict(b) for b in bats]

        with open(self.output_dir / "annotations.json", "w") as f:
            json.dump(data, f, indent=2)

        self._export_yolo_pose_format()
        print(f"Saved annotations to {self.output_dir}")

    def _export_yolo_pose_format(self):
        """Export annotations in YOLO-pose format."""
        yolo_dir = self.output_dir / "yolo_pose"
        (yolo_dir / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels").mkdir(parents=True, exist_ok=True)

        count = 0
        for frame_info in self.frames:
            frame_id = str(frame_info["id"])
            if frame_id not in self.annotations:
                continue

            bats = self.annotations[frame_id]
            if not bats:
                continue

            frame = cv2.imread(frame_info["path"])
            if frame is None:
                continue
            h, w = frame.shape[:2]

            src_path = Path(frame_info["path"])
            dst_img = yolo_dir / "images" / src_path.name
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), frame)

            label_path = yolo_dir / "labels" / (src_path.stem + ".txt")
            with open(label_path, "w") as f:
                for bat in bats:
                    line = bat.to_yolo_pose(w, h)
                    f.write(line + "\n")

            count += 1

        # Write data.yaml for YOLO-pose
        yaml_content = f"""path: {yolo_dir}
train: images
val: images

kpt_shape: [2, 3]  # 2 keypoints, 3 values each (x, y, visibility)

names:
  0: bat

# Keypoint names
keypoint_names:
  - knob    # Handle end
  - cap     # Barrel end
"""
        with open(yolo_dir / "data.yaml", "w") as f:
            f.write(yaml_content)

        print(f"Exported {count} samples in YOLO-pose format to {yolo_dir}")

    def _build_gui(self):
        """Build the labeling GUI."""
        self.root = tk.Tk()
        self.root.title("Bat Keypoint Labeler - YOLO-Pose Training")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', lambda e: self._display_frame())

        # Right panel
        right_frame = ttk.Frame(main_frame, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        # Instructions
        instr_frame = ttk.LabelFrame(right_frame, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, pady=5)

        instructions = """
TWO-CLICK LABELING:
1. Click KNOB (handle end) - marked BLUE
2. Click CAP (barrel end) - marked RED
3. Bat is auto-saved after both clicks

Right-click to delete nearest bat.
A/D or arrows to navigate.
S to save, Q to quit.

BLUE = Knob (handle)
RED = Cap (barrel)
GREEN LINE = Bat axis
        """
        ttk.Label(instr_frame, text=instructions, wraplength=290).pack()

        # Status
        status_frame = ttk.LabelFrame(right_frame, text="Current State", padding=10)
        status_frame.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(status_frame, text="Click KNOB first",
                                      font=('Arial', 12, 'bold'))
        self.status_label.pack()

        # Progress
        progress_frame = ttk.LabelFrame(right_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Frame: 0/0")
        self.progress_label.pack()

        self.annotated_label = ttk.Label(progress_frame, text="Annotated: 0")
        self.annotated_label.pack()

        # Current annotations
        anno_frame = ttk.LabelFrame(right_frame, text="Bats in Frame", padding=10)
        anno_frame.pack(fill=tk.X, pady=5)

        self.anno_listbox = tk.Listbox(anno_frame, height=5)
        self.anno_listbox.pack(fill=tk.X)

        # Navigation
        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, pady=10)

        ttk.Button(nav_frame, text="< Prev (A)", command=self._prev_frame).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="Next (D) >", command=self._next_frame).pack(side=tk.LEFT, expand=True)

        # Actions
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=10)

        ttk.Button(action_frame, text="Save (S)", command=self.save_annotations).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Cancel Current", command=self._cancel_current).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear Frame", command=self._clear_current).pack(fill=tk.X, pady=2)

        # Keyboard bindings
        self.root.bind('<a>', lambda e: self._prev_frame())
        self.root.bind('<d>', lambda e: self._next_frame())
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<s>', lambda e: self.save_annotations())
        self.root.bind('<Escape>', lambda e: self._cancel_current())
        self.root.bind('<q>', lambda e: self._quit())

    def _display_frame(self):
        """Display current frame with annotations."""
        if not self.frames or self.current_idx >= len(self.frames):
            return

        frame_info = self.frames[self.current_idx]
        frame = cv2.imread(frame_info["path"])
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame_rgb.shape[:2]

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        self.img_offset_x = (canvas_w - new_w) // 2
        self.img_offset_y = (canvas_h - new_h) // 2

        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # Draw completed bat annotations
        frame_id = str(frame_info["id"])
        if frame_id in self.annotations:
            for bat in self.annotations[frame_id]:
                # Scale coordinates
                kx = int(bat.knob_x * self.scale_factor)
                ky = int(bat.knob_y * self.scale_factor)
                cx = int(bat.cap_x * self.scale_factor)
                cy = int(bat.cap_y * self.scale_factor)

                # Draw line (bat axis)
                cv2.line(frame_resized, (kx, ky), (cx, cy), (0, 255, 0), 2)

                # Draw keypoints
                cv2.circle(frame_resized, (kx, ky), 8, (0, 100, 255), -1)  # Knob - blue
                cv2.circle(frame_resized, (cx, cy), 8, (255, 100, 0), -1)  # Cap - red
                cv2.putText(frame_resized, "K", (kx - 5, ky + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame_resized, "C", (cx - 5, cy + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw temporary knob if waiting for cap
        if self.temp_knob:
            kx = int(self.temp_knob[0] * self.scale_factor)
            ky = int(self.temp_knob[1] * self.scale_factor)
            cv2.circle(frame_resized, (kx, ky), 10, (0, 100, 255), 3)
            cv2.putText(frame_resized, "KNOB", (kx + 12, ky),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        # Display
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                anchor=tk.NW, image=self.photo)

        self._update_labels()

    def _update_labels(self):
        """Update status labels."""
        total = len(self.frames)
        self.progress_label.config(text=f"Frame: {self.current_idx + 1}/{total}")
        self.annotated_label.config(text=f"Annotated: {len(self.annotations)}")

        # Update status
        if self.temp_knob:
            self.status_label.config(text="Click CAP (barrel end)", foreground="red")
        else:
            self.status_label.config(text="Click KNOB (handle end)", foreground="blue")

        # Update listbox
        self.anno_listbox.delete(0, tk.END)
        frame_id = str(self.frames[self.current_idx]["id"]) if self.frames else ""
        if frame_id in self.annotations:
            for i, bat in enumerate(self.annotations[frame_id]):
                self.anno_listbox.insert(tk.END,
                    f"Bat {i+1}: Knob({bat.knob_x},{bat.knob_y}) Cap({bat.cap_x},{bat.cap_y})")

    def _on_click(self, event):
        """Handle click - alternates between knob and cap."""
        # Convert to image coordinates
        img_x = int((event.x - self.img_offset_x) / self.scale_factor)
        img_y = int((event.y - self.img_offset_y) / self.scale_factor)

        # Bounds check
        frame_info = self.frames[self.current_idx]
        frame = cv2.imread(frame_info["path"])
        if frame is None:
            return
        h, w = frame.shape[:2]
        if img_x < 0 or img_x >= w or img_y < 0 or img_y >= h:
            return

        if self.temp_knob is None:
            # First click = knob
            self.temp_knob = (img_x, img_y)
            self._display_frame()
        else:
            # Second click = cap, complete the annotation
            knob_x, knob_y = self.temp_knob
            cap_x, cap_y = img_x, img_y

            frame_id = str(frame_info["id"])
            if frame_id not in self.annotations:
                self.annotations[frame_id] = []

            self.annotations[frame_id].append(
                BatAnnotation(knob_x, knob_y, cap_x, cap_y)
            )

            self.temp_knob = None
            self._display_frame()

    def _on_right_click(self, event):
        """Delete nearest bat annotation."""
        if not self.frames:
            return

        frame_id = str(self.frames[self.current_idx]["id"])
        if frame_id not in self.annotations or not self.annotations[frame_id]:
            return

        click_x = (event.x - self.img_offset_x) / self.scale_factor
        click_y = (event.y - self.img_offset_y) / self.scale_factor

        min_dist = float('inf')
        nearest_idx = -1

        for i, bat in enumerate(self.annotations[frame_id]):
            # Distance to midpoint of bat
            mid_x = (bat.knob_x + bat.cap_x) / 2
            mid_y = (bat.knob_y + bat.cap_y) / 2
            dist = ((mid_x - click_x) ** 2 + (mid_y - click_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0:
            del self.annotations[frame_id][nearest_idx]
            if not self.annotations[frame_id]:
                del self.annotations[frame_id]
            self._display_frame()

    def _cancel_current(self):
        """Cancel current partial annotation."""
        self.temp_knob = None
        self._display_frame()

    def _prev_frame(self):
        self._cancel_current()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._display_frame()

    def _next_frame(self):
        self._cancel_current()
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self._display_frame()

    def _clear_current(self):
        """Clear all annotations for current frame."""
        self._cancel_current()
        if self.frames:
            frame_id = str(self.frames[self.current_idx]["id"])
            if frame_id in self.annotations:
                del self.annotations[frame_id]
            self._display_frame()

    def _quit(self):
        """Save and quit."""
        if self.annotations:
            if messagebox.askyesno("Save", "Save annotations before quitting?"):
                self.save_annotations()
        self.root.destroy()

    def run(self):
        """Run the labeling GUI."""
        self.load_existing()

        if not self.frames:
            print("No frames found. Run with --extract first.")
            return

        self._build_gui()
        self._display_frame()
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bat Keypoint Labeler for YOLO-Pose Training")
    parser.add_argument("--extract", type=int, default=0,
                       help="Extract N frames for labeling")
    parser.add_argument("--append", action="store_true",
                       help="Append to existing frames")
    parser.add_argument("--project", type=str,
                       default="F:/Claude_Projects/baseball-biomechanics",
                       help="Project directory")
    args = parser.parse_args()

    labeler = BatKeypointLabeler(args.project)

    if args.extract > 0:
        labeler.extract_frames(args.extract, append=args.append)
        print("\nRun without --extract to start labeling.")
    else:
        labeler.run()


if __name__ == "__main__":
    main()
