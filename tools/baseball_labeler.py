#!/usr/bin/env python
"""
Baseball Labeler for YOLO Training.

Labeling workflow:
- View frames from pitch sequences
- Click and drag to draw bounding box around baseball
- Navigate between frames with A/D keys
- Export in YOLO format for training

Target: ~300-500 labeled frames for robust baseball detection.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image, ImageTk


@dataclass
class BboxAnnotation:
    """Bounding box annotation for baseball."""
    x1: int
    y1: int
    x2: int
    y2: int

    def to_xyxy(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format: x_center, y_center, width, height (normalized)."""
        cx = (self.x1 + self.x2) / 2 / img_width
        cy = (self.y1 + self.y2) / 2 / img_height
        w = abs(self.x2 - self.x1) / img_width
        h = abs(self.y2 - self.y1) / img_height
        return (cx, cy, w, h)


class BaseballLabeler:
    """Interactive labeler for baseball bounding boxes."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.video_dirs = [
            self.project_dir / "data" / "videos" / "2024",
            self.project_dir / "data" / "videos" / "diverse_stadiums",
        ]
        self.output_dir = self.project_dir / "data" / "labels" / "baseball"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames: List[Dict] = []
        self.annotations: Dict[str, List[BboxAnnotation]] = {}
        self.current_idx = 0

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        self.scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

    def extract_frames(self, num_frames: int = 300, append: bool = False):
        """Extract frames for labeling, focusing on pitch-release moments."""
        all_videos = []
        for video_dir in self.video_dirs:
            if video_dir.exists():
                all_videos.extend(list(video_dir.rglob("*.mp4")))

        if not all_videos:
            print(f"No videos found")
            return

        # Load existing if appending
        start_idx = 0
        if append:
            self.load_existing()
            start_idx = len(self.frames)
            print(f"Appending to existing {start_idx} frames")

        import random
        random.seed(42 + start_idx)

        # Sample frames from pitch-release area (where ball is visible)
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

            # Sample from middle (pitch release area) - ball most visible here
            frame_idx = random.randint(int(total * 0.3), int(total * 0.7))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            global_idx = start_idx + i
            frame_name = f"ball_{global_idx:04d}_{video_path.stem}_f{frame_idx}.png"
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

        # Save frame info
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
                for frame_id, boxes in data.items():
                    self.annotations[frame_id] = [
                        BboxAnnotation(**b) for b in boxes
                    ]
            print(f"Loaded annotations for {len(self.annotations)} frames")

    def save_annotations(self):
        """Save annotations to JSON and YOLO format."""
        # Save JSON format
        data = {}
        for frame_id, boxes in self.annotations.items():
            data[frame_id] = [asdict(b) for b in boxes]

        with open(self.output_dir / "annotations.json", "w") as f:
            json.dump(data, f, indent=2)

        # Export YOLO format
        self._export_yolo_format()
        print(f"Saved annotations to {self.output_dir}")

    def _export_yolo_format(self):
        """Export annotations in YOLO format."""
        yolo_dir = self.output_dir / "yolo"
        (yolo_dir / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels").mkdir(parents=True, exist_ok=True)

        count = 0
        for frame_info in self.frames:
            frame_id = str(frame_info["id"])
            if frame_id not in self.annotations:
                continue

            boxes = self.annotations[frame_id]
            if not boxes:
                continue

            # Load frame for dimensions
            frame = cv2.imread(frame_info["path"])
            if frame is None:
                continue
            h, w = frame.shape[:2]

            # Copy image
            src_path = Path(frame_info["path"])
            dst_img = yolo_dir / "images" / src_path.name
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), frame)

            # Write YOLO label (class 0 = baseball)
            label_path = yolo_dir / "labels" / (src_path.stem + ".txt")
            with open(label_path, "w") as f:
                for box in boxes:
                    cx, cy, bw, bh = box.to_yolo(w, h)
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            count += 1

        # Write data.yaml
        yaml_content = f"""path: {yolo_dir}
train: images
val: images

names:
  0: baseball
"""
        with open(yolo_dir / "data.yaml", "w") as f:
            f.write(yaml_content)

        print(f"Exported {count} samples in YOLO format to {yolo_dir}")

    def _build_gui(self):
        """Build the labeling GUI."""
        self.root = tk.Tk()
        self.root.title("Baseball Labeler - YOLO Training")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind events
        self.canvas.bind('<Button-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', lambda e: self._display_frame())

        # Right panel
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        # Instructions
        instr_frame = ttk.LabelFrame(right_frame, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, pady=5)

        instructions = """
1. Click and drag to draw box around BASEBALL
2. Right-click to delete nearest box
3. A/D or arrows to navigate frames
4. S to save, Q to quit

TIP: Baseball is small and white.
Look near pitcher's hand or in flight path.
Skip frames where ball is not visible.
        """
        ttk.Label(instr_frame, text=instructions, wraplength=270).pack()

        # Progress
        progress_frame = ttk.LabelFrame(right_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Frame: 0/0")
        self.progress_label.pack()

        self.annotated_label = ttk.Label(progress_frame, text="Annotated: 0")
        self.annotated_label.pack()

        # Current frame annotations
        anno_frame = ttk.LabelFrame(right_frame, text="Current Annotations", padding=10)
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
        ttk.Button(action_frame, text="Clear Frame", command=self._clear_current).pack(fill=tk.X, pady=2)

        # Keyboard bindings
        self.root.bind('<a>', lambda e: self._prev_frame())
        self.root.bind('<d>', lambda e: self._next_frame())
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<s>', lambda e: self.save_annotations())
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

        # Scale to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        self.img_offset_x = (canvas_w - new_w) // 2
        self.img_offset_y = (canvas_h - new_h) // 2

        # Resize
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # Draw annotations
        frame_id = str(frame_info["id"])
        if frame_id in self.annotations:
            for box in self.annotations[frame_id]:
                x1 = int(box.x1 * self.scale_factor)
                y1 = int(box.y1 * self.scale_factor)
                x2 = int(box.x2 * self.scale_factor)
                y2 = int(box.y2 * self.scale_factor)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, "BALL", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display
        self.photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                anchor=tk.NW, image=self.photo)

        # Update labels
        self._update_labels()

    def _update_labels(self):
        """Update progress labels."""
        total = len(self.frames)
        self.progress_label.config(text=f"Frame: {self.current_idx + 1}/{total}")
        self.annotated_label.config(text=f"Annotated: {len(self.annotations)}")

        # Update listbox
        self.anno_listbox.delete(0, tk.END)
        frame_id = str(self.frames[self.current_idx]["id"]) if self.frames else ""
        if frame_id in self.annotations:
            for i, box in enumerate(self.annotations[frame_id]):
                cx, cy = box.center()
                self.anno_listbox.insert(tk.END, f"Ball {i+1}: ({int(cx)}, {int(cy)})")

    def _on_mouse_down(self, event):
        """Start drawing box."""
        self.drawing = True
        self.start_x = event.x - self.img_offset_x
        self.start_y = event.y - self.img_offset_y

    def _on_mouse_drag(self, event):
        """Update box while dragging."""
        if not self.drawing:
            return

        if self.current_rect:
            self.canvas.delete(self.current_rect)

        x = event.x
        y = event.y
        self.current_rect = self.canvas.create_rectangle(
            self.start_x + self.img_offset_x, self.start_y + self.img_offset_y,
            x, y, outline='#00FF00', width=2
        )

    def _on_mouse_up(self, event):
        """Finish drawing box."""
        if not self.drawing:
            return

        self.drawing = False
        end_x = event.x - self.img_offset_x
        end_y = event.y - self.img_offset_y

        # Convert to original image coordinates
        x1 = int(min(self.start_x, end_x) / self.scale_factor)
        y1 = int(min(self.start_y, end_y) / self.scale_factor)
        x2 = int(max(self.start_x, end_x) / self.scale_factor)
        y2 = int(max(self.start_y, end_y) / self.scale_factor)

        # Minimum size check
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            return

        # Add annotation
        frame_id = str(self.frames[self.current_idx]["id"])
        if frame_id not in self.annotations:
            self.annotations[frame_id] = []

        self.annotations[frame_id].append(BboxAnnotation(x1, y1, x2, y2))
        self._display_frame()

    def _on_right_click(self, event):
        """Delete nearest box."""
        if not self.frames:
            return

        frame_id = str(self.frames[self.current_idx]["id"])
        if frame_id not in self.annotations or not self.annotations[frame_id]:
            return

        # Find nearest box
        click_x = (event.x - self.img_offset_x) / self.scale_factor
        click_y = (event.y - self.img_offset_y) / self.scale_factor

        min_dist = float('inf')
        nearest_idx = -1

        for i, box in enumerate(self.annotations[frame_id]):
            cx, cy = box.center()
            dist = ((cx - click_x) ** 2 + (cy - click_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx >= 0:
            del self.annotations[frame_id][nearest_idx]
            if not self.annotations[frame_id]:
                del self.annotations[frame_id]
            self._display_frame()

    def _prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._display_frame()

    def _next_frame(self):
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self._display_frame()

    def _clear_current(self):
        """Clear annotations for current frame."""
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
    parser = argparse.ArgumentParser(description="Baseball Labeler for YOLO Training")
    parser.add_argument("--extract", type=int, default=0,
                       help="Extract N frames for labeling")
    parser.add_argument("--append", action="store_true",
                       help="Append to existing frames")
    parser.add_argument("--project", type=str,
                       default="F:/Claude_Projects/baseball-biomechanics",
                       help="Project directory")
    args = parser.parse_args()

    labeler = BaseballLabeler(args.project)

    if args.extract > 0:
        labeler.extract_frames(args.extract, append=args.append)
        print("\nRun without --extract to start labeling.")
    else:
        labeler.run()


if __name__ == "__main__":
    main()
