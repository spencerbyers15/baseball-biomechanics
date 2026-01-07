#!/usr/bin/env python
"""
Baseball Labeler for YOLO Training.

Labeling workflow:
- View frames from pitch sequences
- Single-click to mark ball location (creates small bbox)
- Press N for "no ball" in frame (negative training example)
- Navigate between frames with A/D keys
- Export in YOLO format (including negatives as empty label files)

Target: ~300-500 labeled frames for robust baseball detection.

Controls:
    Left-click  : Mark ball location (creates fixed-size bbox)
    Shift+Drag  : Draw custom bbox (for unusual ball sizes)
    Right-click : Remove nearest annotation
    N           : Mark as "no ball" (IMPORTANT negative example)
    K           : Skip frame (unclear/motion blur)
    A/Left      : Previous frame
    D/Right     : Next frame
    Ctrl+S      : Save annotations
    Q           : Quit
"""

import json
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set
import tkinter as tk
from tkinter import ttk, messagebox

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image, ImageTk

# Baseball bounding box radius (creates ~40x40 pixel bbox on click)
BALL_RADIUS = 20


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

    @classmethod
    def from_point(cls, x: int, y: int, radius: int = BALL_RADIUS) -> "BboxAnnotation":
        """Create bbox centered on a point (for single-click labeling)."""
        return cls(x - radius, y - radius, x + radius, y + radius)


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
        self.no_ball_frames: Set[str] = set()  # Frames with NO ball (negatives)
        self.skipped_frames: Set[str] = set()  # Skipped frames
        self.current_idx = 0

        # Drawing state
        self.drawing = False
        self.shift_held = False
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

        start_idx = 0
        if append:
            self.load_existing()
            start_idx = len(self.frames)
            print(f"Appending to existing {start_idx} frames")

        import random
        random.seed(42 + start_idx)

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

        with open(self.output_dir / "frames_info.json", "w") as f:
            json.dump(self.frames, f, indent=2)

        print(f"Extracted {len(self.frames)} frames to {self.output_dir / 'frames'}")

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
                if "balls" in data:
                    for frame_id, boxes in data["balls"].items():
                        self.annotations[frame_id] = [
                            BboxAnnotation(**b) for b in boxes
                        ]
                if "no_ball" in data:
                    self.no_ball_frames = set(data["no_ball"])
                if "skipped" in data:
                    self.skipped_frames = set(data["skipped"])
                if "balls" not in data and "no_ball" not in data:
                    for frame_id, boxes in data.items():
                        self.annotations[frame_id] = [
                            BboxAnnotation(**b) for b in boxes
                        ]
            print(f"Loaded: {len(self.annotations)} with ball, {len(self.no_ball_frames)} no ball, {len(self.skipped_frames)} skipped")

    def save_annotations(self):
        """Save annotations to JSON and YOLO format."""
        data = {
            "balls": {},
            "no_ball": list(self.no_ball_frames),
            "skipped": list(self.skipped_frames),
        }
        for frame_id, boxes in self.annotations.items():
            data["balls"][frame_id] = [asdict(b) for b in boxes]

        with open(self.output_dir / "annotations.json", "w") as f:
            json.dump(data, f, indent=2)

        self._export_yolo_format()
        print(f"Saved annotations to {self.output_dir}")

    def _export_yolo_format(self):
        """Export annotations in YOLO format, including negative examples."""
        yolo_dir = self.output_dir / "yolo"
        (yolo_dir / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels").mkdir(parents=True, exist_ok=True)

        positive_count = 0
        negative_count = 0

        for frame_info in self.frames:
            frame_id = str(frame_info["id"])

            if frame_id in self.skipped_frames:
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

            if frame_id in self.annotations and self.annotations[frame_id]:
                with open(label_path, "w") as f:
                    for box in self.annotations[frame_id]:
                        cx, cy, bw, bh = box.to_yolo(w, h)
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                positive_count += 1

            elif frame_id in self.no_ball_frames:
                with open(label_path, "w") as f:
                    pass
                negative_count += 1

        yaml_content = f"""path: {yolo_dir}
train: images
val: images

names:
  0: baseball
"""
        with open(yolo_dir / "data.yaml", "w") as f:
            f.write(yaml_content)

        print(f"Exported YOLO format: {positive_count} positives, {negative_count} negatives to {yolo_dir}")

    def _build_gui(self):
        """Build the labeling GUI."""
        self.root = tk.Tk()
        self.root.title("Baseball Labeler - YOLO Training")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", lambda e: self._display_frame())

        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        instr_frame = ttk.LabelFrame(right_frame, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, pady=5)

        instructions = """
CLICK on baseball = mark location
Shift+DRAG = custom bbox size
Right-click = delete annotation

N = NO BALL in frame (negative)
K = Skip (unclear frame)
A/D = navigate frames
Ctrl+S = save, Q = quit

TIP: Negatives are IMPORTANT!
Mark frames without visible ball.
        """
        ttk.Label(instr_frame, text=instructions, wraplength=270).pack()

        progress_frame = ttk.LabelFrame(right_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Frame: 0/0")
        self.progress_label.pack()

        self.stats_label = ttk.Label(progress_frame, text="Ball: 0 | No ball: 0 | Skip: 0")
        self.stats_label.pack()

        status_frame = ttk.LabelFrame(right_frame, text="Current Frame", padding=10)
        status_frame.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(status_frame, text="Not labeled", font=("TkDefaultFont", 10, "bold"))
        self.status_label.pack()

        self.anno_listbox = tk.Listbox(status_frame, height=3)
        self.anno_listbox.pack(fill=tk.X)

        nav_frame = ttk.Frame(right_frame)
        nav_frame.pack(fill=tk.X, pady=10)

        ttk.Button(nav_frame, text="< Prev (A)", command=self._prev_frame).pack(side=tk.LEFT, expand=True)
        ttk.Button(nav_frame, text="Next (D) >", command=self._next_frame).pack(side=tk.LEFT, expand=True)

        action_frame = ttk.LabelFrame(right_frame, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=5)

        ttk.Button(action_frame, text="No Ball (N)", command=self._mark_no_ball).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Skip (K)", command=self._skip_frame).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear Frame", command=self._clear_current).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save (Ctrl+S)", command=self.save_annotations).pack(fill=tk.X, pady=2)

        self.root.bind("<a>", lambda e: self._prev_frame())
        self.root.bind("<d>", lambda e: self._next_frame())
        self.root.bind("<Left>", lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<n>", lambda e: self._mark_no_ball())
        self.root.bind("<k>", lambda e: self._skip_frame())
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Shift_L>", lambda e: self._set_shift(True))
        self.root.bind("<Shift_R>", lambda e: self._set_shift(True))
        self.root.bind("<KeyRelease-Shift_L>", lambda e: self._set_shift(False))
        self.root.bind("<KeyRelease-Shift_R>", lambda e: self._set_shift(False))

    def _set_shift(self, state: bool):
        self.shift_held = state

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

        if frame_id in self.no_ball_frames:
            cv2.putText(frame_resized, "NO BALL", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 3)
        elif frame_id in self.skipped_frames:
            cv2.putText(frame_resized, "SKIPPED", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), 3)

        self.photo = ImageTk.PhotoImage(Image.fromarray(frame_resized))
        self.canvas.delete("all")
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                anchor=tk.NW, image=self.photo)

        self._update_labels()

    def _update_labels(self):
        """Update progress labels."""
        total = len(self.frames)
        self.progress_label.config(text=f"Frame: {self.current_idx + 1}/{total}")
        self.stats_label.config(text=f"Ball: {len(self.annotations)} | No ball: {len(self.no_ball_frames)} | Skip: {len(self.skipped_frames)}")

        frame_id = str(self.frames[self.current_idx]["id"]) if self.frames else ""
        if frame_id in self.annotations:
            self.status_label.config(text="HAS BALL", foreground="green")
        elif frame_id in self.no_ball_frames:
            self.status_label.config(text="NO BALL (negative)", foreground="gray")
        elif frame_id in self.skipped_frames:
            self.status_label.config(text="SKIPPED", foreground="orange")
        else:
            self.status_label.config(text="Not labeled", foreground="red")

        self.anno_listbox.delete(0, tk.END)
        if frame_id in self.annotations:
            for i, box in enumerate(self.annotations[frame_id]):
                cx, cy = box.center()
                self.anno_listbox.insert(tk.END, f"Ball {i+1}: ({int(cx)}, {int(cy)})")

    def _on_click(self, event):
        """Handle click - single click marks ball, shift+click starts drag."""
        self.start_x = event.x - self.img_offset_x
        self.start_y = event.y - self.img_offset_y

        if self.shift_held:
            self.drawing = True
        else:
            self._place_ball_at_click(event.x, event.y)

    def _place_ball_at_click(self, screen_x: int, screen_y: int):
        """Place a ball annotation at click location."""
        x = int((screen_x - self.img_offset_x) / self.scale_factor)
        y = int((screen_y - self.img_offset_y) / self.scale_factor)

        frame_id = str(self.frames[self.current_idx]["id"])

        self.no_ball_frames.discard(frame_id)
        self.skipped_frames.discard(frame_id)

        if frame_id not in self.annotations:
            self.annotations[frame_id] = []

        self.annotations[frame_id].append(BboxAnnotation.from_point(x, y))
        self._display_frame()

    def _on_mouse_drag(self, event):
        """Update box while dragging (shift+drag only)."""
        if not self.drawing:
            return

        if self.current_rect:
            self.canvas.delete(self.current_rect)

        x = event.x
        y = event.y
        self.current_rect = self.canvas.create_rectangle(
            self.start_x + self.img_offset_x, self.start_y + self.img_offset_y,
            x, y, outline="#00FF00", width=2
        )

    def _on_mouse_up(self, event):
        """Finish drawing box (shift+drag only)."""
        if not self.drawing:
            return

        self.drawing = False
        end_x = event.x - self.img_offset_x
        end_y = event.y - self.img_offset_y

        x1 = int(min(self.start_x, end_x) / self.scale_factor)
        y1 = int(min(self.start_y, end_y) / self.scale_factor)
        x2 = int(max(self.start_x, end_x) / self.scale_factor)
        y2 = int(max(self.start_y, end_y) / self.scale_factor)

        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            return

        frame_id = str(self.frames[self.current_idx]["id"])

        self.no_ball_frames.discard(frame_id)
        self.skipped_frames.discard(frame_id)

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

        click_x = (event.x - self.img_offset_x) / self.scale_factor
        click_y = (event.y - self.img_offset_y) / self.scale_factor

        min_dist = float("inf")
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

    def _mark_no_ball(self):
        """Mark current frame as having no ball (negative example)."""
        if not self.frames:
            return
        frame_id = str(self.frames[self.current_idx]["id"])

        if frame_id in self.annotations:
            del self.annotations[frame_id]
        self.skipped_frames.discard(frame_id)

        self.no_ball_frames.add(frame_id)
        self._display_frame()
        self._next_frame()

    def _skip_frame(self):
        """Skip current frame."""
        if not self.frames:
            return
        frame_id = str(self.frames[self.current_idx]["id"])

        if frame_id in self.annotations:
            del self.annotations[frame_id]
        self.no_ball_frames.discard(frame_id)

        self.skipped_frames.add(frame_id)
        self._display_frame()
        self._next_frame()

    def _prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._display_frame()

    def _next_frame(self):
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self._display_frame()

    def _clear_current(self):
        """Clear all annotations/status for current frame."""
        if self.frames:
            frame_id = str(self.frames[self.current_idx]["id"])
            if frame_id in self.annotations:
                del self.annotations[frame_id]
            self.no_ball_frames.discard(frame_id)
            self.skipped_frames.discard(frame_id)
            self._display_frame()

    def _quit(self):
        """Save and quit."""
        total_labeled = len(self.annotations) + len(self.no_ball_frames) + len(self.skipped_frames)
        if total_labeled > 0:
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
        print("Run without --extract to start labeling.")
    else:
        labeler.run()


if __name__ == "__main__":
    main()
