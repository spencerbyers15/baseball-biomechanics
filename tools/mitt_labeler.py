"""
Catcher's Mitt Labeler for SAM3 Fine-tuning.

Two-stage approach:
1. Quick improvement: Use positive/negative box prompts (no training needed)
2. Full fine-tuning: Collect ~15-20 labeled frames for mask decoder fine-tuning

Labeling workflow:
- View frames with low/failed detection
- Click to draw bounding box around catcher's mitt (POSITIVE)
- Optionally mark pitcher's glove area (NEGATIVE)
- Save annotations in SAM3-compatible format
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
class BoxAnnotation:
    """Bounding box annotation."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str  # "catcher_mitt" (positive) or "pitcher_glove" (negative)
    is_positive: bool = True

    def to_xyxy(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class MittLabeler:
    """Interactive labeler for catcher's mitt annotations."""

    COLORS = {
        "catcher_mitt": "#00FF00",  # Green - positive
        "pitcher_glove": "#FF0000",  # Red - negative
    }

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.video_dir = self.project_dir / "data" / "videos" / "2024" / "04"
        self.output_dir = self.project_dir / "data" / "labels" / "mitt_finetune"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames: List[Dict] = []
        self.annotations: Dict[str, List[BoxAnnotation]] = {}
        self.current_idx = 0
        self.current_mode = "catcher_mitt"

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None

        self.scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

    def extract_problem_frames(self, num_frames: int = 15, append: bool = False):
        """Extract frames that need labeling (diverse set for training)."""
        videos = list(self.video_dir.glob("*.mp4"))
        if not videos:
            print(f"No videos found in {self.video_dir}")
            return

        # Load existing frames if appending
        start_idx = 0
        if append:
            self.load_existing()
            start_idx = len(self.frames)
            print(f"Appending to existing {start_idx} frames")

        # Sample from multiple videos for diversity
        import random
        random.seed(42 + start_idx)  # Different seed for new frames

        # Get more samples per video if we need many frames
        samples_per_video = max(1, num_frames // len(videos) + 1)
        selected_videos = videos * samples_per_video
        random.shuffle(selected_videos)
        selected_videos = selected_videos[:num_frames]

        print(f"Extracting {len(selected_videos)} frames for labeling...")

        for i, video_path in enumerate(selected_videos):
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample from middle portion (action frames)
            frame_idx = random.randint(int(total * 0.2), int(total * 0.8))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                continue

            # Save frame with unique index
            global_idx = start_idx + i
            frame_name = f"mitt_{global_idx:03d}_{video_path.stem}.png"
            frame_path = self.output_dir / "frames" / frame_name
            frame_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(frame_path), frame)

            self.frames.append({
                "id": global_idx,
                "path": str(frame_path),
                "source_video": str(video_path),
                "frame_idx": frame_idx,
            })

            print(f"  [{i+1}/{len(selected_videos)}] {frame_name}")

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
                        BoxAnnotation(**b) for b in boxes
                    ]
            print(f"Loaded annotations for {len(self.annotations)} frames")

    def save_annotations(self):
        """Save annotations to JSON."""
        data = {}
        for frame_id, boxes in self.annotations.items():
            data[frame_id] = [asdict(b) for b in boxes]

        with open(self.output_dir / "annotations.json", "w") as f:
            json.dump(data, f, indent=2)

        # Also export in SAM3-compatible format
        self._export_sam3_format()

        print(f"Saved annotations to {self.output_dir}")

    def _export_sam3_format(self):
        """Export annotations in SAM3 fine-tuning format."""
        sam3_data = []

        for frame_info in self.frames:
            frame_id = str(frame_info["id"])
            if frame_id not in self.annotations:
                continue

            boxes = self.annotations[frame_id]
            if not boxes:
                continue

            # Load frame to get dimensions
            frame = cv2.imread(frame_info["path"])
            if frame is None:
                continue
            h, w = frame.shape[:2]

            entry = {
                "image_path": frame_info["path"],
                "image_size": [w, h],
                "positive_boxes": [],
                "negative_boxes": [],
                "masks": [],  # Will be generated during training
            }

            for box in boxes:
                xyxy = box.to_xyxy()
                if box.is_positive:
                    entry["positive_boxes"].append(xyxy)
                else:
                    entry["negative_boxes"].append(xyxy)

            sam3_data.append(entry)

        with open(self.output_dir / "sam3_training_data.json", "w") as f:
            json.dump(sam3_data, f, indent=2)

        print(f"Exported {len(sam3_data)} samples for SAM3 training")

    def _build_gui(self):
        """Build the labeling GUI."""
        self.root = tk.Tk()
        self.root.title("Catcher's Mitt Labeler - SAM3 Fine-tuning")
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

        # Bind mouse events
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
1. Select mode (Catcher's Mitt / Pitcher's Glove)
2. Click and drag to draw bounding box
3. Right-click to delete nearest box
4. Navigate with A/D or arrow keys
5. Save when done

GREEN = Catcher's Mitt (positive)
RED = Pitcher's Glove (negative)
        """
        ttk.Label(instr_frame, text=instructions, wraplength=270).pack()

        # Mode selection
        mode_frame = ttk.LabelFrame(right_frame, text="Annotation Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value=self.current_mode)

        ttk.Radiobutton(
            mode_frame, text="Catcher's Mitt (POSITIVE)",
            variable=self.mode_var, value="catcher_mitt",
            command=self._on_mode_change
        ).pack(anchor=tk.W)

        ttk.Radiobutton(
            mode_frame, text="Pitcher's Glove (NEGATIVE)",
            variable=self.mode_var, value="pitcher_glove",
            command=self._on_mode_change
        ).pack(anchor=tk.W)

        # Progress
        progress_frame = ttk.LabelFrame(right_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(fill=tk.X)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Current frame annotations
        ann_frame = ttk.LabelFrame(right_frame, text="Current Frame Annotations", padding=10)
        ann_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.ann_listbox = tk.Listbox(ann_frame, height=6, bg='#3c3c3c', fg='white')
        self.ann_listbox.pack(fill=tk.BOTH, expand=True)

        ttk.Button(ann_frame, text="Delete Selected",
                   command=self._delete_selected).pack(fill=tk.X, pady=2)

        # Navigation
        nav_frame = ttk.LabelFrame(right_frame, text="Navigation", padding=10)
        nav_frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(nav_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="<< Prev (A)",
                   command=self._prev_frame).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Next (D) >>",
                   command=self._next_frame).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Save
        save_frame = ttk.Frame(right_frame)
        save_frame.pack(fill=tk.X, pady=10)

        ttk.Button(save_frame, text="Save Annotations",
                   command=self.save_annotations).pack(fill=tk.X)

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('a', lambda e: self._prev_frame())
        self.root.bind('d', lambda e: self._next_frame())
        self.root.bind('1', lambda e: self._set_mode("catcher_mitt"))
        self.root.bind('2', lambda e: self._set_mode("pitcher_glove"))
        self.root.bind('<Control-s>', lambda e: self.save_annotations())

    def _on_mode_change(self):
        self.current_mode = self.mode_var.get()

    def _set_mode(self, mode: str):
        self.current_mode = mode
        self.mode_var.set(mode)

    def _get_frame_id(self) -> str:
        return str(self.frames[self.current_idx]["id"])

    def _display_frame(self):
        """Display current frame with annotations."""
        if not self.frames:
            return

        frame_info = self.frames[self.current_idx]
        frame = cv2.imread(frame_info["path"])
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Scale to fit canvas
        canvas_w = self.canvas.winfo_width() or 900
        canvas_h = self.canvas.winfo_height() or 700

        self.scale_factor = min(canvas_w / w, canvas_h / h, 1.0)
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)

        resized = cv2.resize(frame_rgb, (new_w, new_h))
        pil_img = Image.fromarray(resized)
        self.canvas_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete('all')
        self.img_offset_x = (canvas_w - new_w) // 2
        self.img_offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(self.img_offset_x, self.img_offset_y,
                                  image=self.canvas_image, anchor=tk.NW)

        # Draw annotations
        frame_id = self._get_frame_id()
        if frame_id in self.annotations:
            for box in self.annotations[frame_id]:
                self._draw_box(box)

        # Update progress
        labeled = sum(1 for fid in [str(f["id"]) for f in self.frames]
                      if fid in self.annotations and self.annotations[fid])
        self.progress_label.config(
            text=f"Frame {self.current_idx + 1}/{len(self.frames)} | Labeled: {labeled}"
        )
        self.progress_bar['value'] = (labeled / max(1, len(self.frames))) * 100

        # Update annotation list
        self._update_ann_list()

    def _draw_box(self, box: BoxAnnotation):
        """Draw a bounding box on the canvas."""
        color = self.COLORS.get(box.label, "#FFFFFF")

        x1 = box.x1 * self.scale_factor + self.img_offset_x
        y1 = box.y1 * self.scale_factor + self.img_offset_y
        x2 = box.x2 * self.scale_factor + self.img_offset_x
        y2 = box.y2 * self.scale_factor + self.img_offset_y

        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)

        # Label
        label_text = "+" if box.is_positive else "-"
        label_text += f" {box.label.replace('_', ' ')}"
        self.canvas.create_text(x1 + 5, y1 - 10, text=label_text,
                                fill=color, anchor=tk.W, font=('Arial', 10, 'bold'))

    def _update_ann_list(self):
        """Update the annotation listbox."""
        self.ann_listbox.delete(0, tk.END)
        frame_id = self._get_frame_id()
        if frame_id in self.annotations:
            for i, box in enumerate(self.annotations[frame_id]):
                marker = "[+]" if box.is_positive else "[-]"
                self.ann_listbox.insert(tk.END, f"{marker} {box.label}")

    def _on_mouse_down(self, event):
        """Start drawing a box."""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def _on_mouse_drag(self, event):
        """Update the box while dragging."""
        if not self.drawing:
            return

        if self.current_rect:
            self.canvas.delete(self.current_rect)

        color = self.COLORS.get(self.current_mode, "#FFFFFF")
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline=color, width=2, dash=(4, 4)
        )

    def _on_mouse_up(self, event):
        """Finish drawing and save the box."""
        if not self.drawing:
            return

        self.drawing = False
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        # Convert canvas coords to image coords
        x1 = int((min(self.start_x, event.x) - self.img_offset_x) / self.scale_factor)
        y1 = int((min(self.start_y, event.y) - self.img_offset_y) / self.scale_factor)
        x2 = int((max(self.start_x, event.x) - self.img_offset_x) / self.scale_factor)
        y2 = int((max(self.start_y, event.y) - self.img_offset_y) / self.scale_factor)

        # Validate box size
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            return

        # Create annotation
        box = BoxAnnotation(
            x1=max(0, x1), y1=max(0, y1), x2=x2, y2=y2,
            label=self.current_mode,
            is_positive=(self.current_mode == "catcher_mitt")
        )

        frame_id = self._get_frame_id()
        if frame_id not in self.annotations:
            self.annotations[frame_id] = []
        self.annotations[frame_id].append(box)

        self._display_frame()

    def _on_right_click(self, event):
        """Delete nearest box."""
        frame_id = self._get_frame_id()
        if frame_id not in self.annotations or not self.annotations[frame_id]:
            return

        # Convert click to image coords
        click_x = (event.x - self.img_offset_x) / self.scale_factor
        click_y = (event.y - self.img_offset_y) / self.scale_factor

        # Find nearest box
        min_dist = float('inf')
        nearest_idx = None

        for i, box in enumerate(self.annotations[frame_id]):
            cx, cy = box.center()
            dist = ((cx - click_x)**2 + (cy - click_y)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        if nearest_idx is not None and min_dist < 200:
            del self.annotations[frame_id][nearest_idx]
            self._display_frame()

    def _delete_selected(self):
        """Delete selected annotation from listbox."""
        sel = self.ann_listbox.curselection()
        if not sel:
            return

        frame_id = self._get_frame_id()
        if frame_id in self.annotations and sel[0] < len(self.annotations[frame_id]):
            del self.annotations[frame_id][sel[0]]
            self._display_frame()

    def _prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._display_frame()

    def _next_frame(self):
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self._display_frame()

    def run(self):
        """Run the labeling GUI."""
        if not self.frames:
            print("No frames loaded. Run extract_problem_frames() first.")
            return

        self._build_gui()
        self.root.after(100, self._display_frame)
        self.root.mainloop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Catcher's Mitt Labeler")
    parser.add_argument("--project-dir", default="F:/Claude_Projects/baseball-biomechanics")
    parser.add_argument("--extract", action="store_true", help="Extract new frames")
    parser.add_argument("--append", action="store_true", help="Append to existing frames")
    parser.add_argument("--num-frames", type=int, default=15)
    args = parser.parse_args()

    labeler = MittLabeler(args.project_dir)

    if args.extract:
        labeler.extract_problem_frames(args.num_frames, append=args.append)
    else:
        labeler.load_existing()

    if not labeler.frames:
        print("No frames found. Run with --extract to extract frames first.")
        print(f"  python {__file__} --extract --num-frames 15")
        print(f"  python {__file__} --extract --append --num-frames 50  # Add more frames")
        return

    labeler.run()


if __name__ == "__main__":
    main()
