"""
SAM 3 Frame Labeler with Text Prompts.

Uses HuggingFace Transformers SAM 3 to automatically segment:
- "baseball batter"
- "catcher's mitt"
- "baseball pitcher"

Provides both CV2 and Tkinter GUIs for review and correction.
Exports labels for video tracking.
"""

import json
import random
import sys
import tkinter as tk
from dataclasses import dataclass, field, asdict
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box with label information."""
    x: float
    y: float
    width: float
    height: float
    label: str
    confidence: float = 1.0
    is_auto: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'BoundingBox':
        return cls(**d)

    def contains_point(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


class Sam3Labeler:
    """SAM 3 based frame labeler with text prompts."""

    def __init__(
        self,
        cache_dir: str = "F:/hf_cache",
        confidence_threshold: float = 0.3,
    ):
        self.cache_dir = cache_dir
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load SAM 3 model from HuggingFace."""
        if self.model is not None:
            return

        print("Loading SAM 3 model from facebook/sam3...")
        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir
        )
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3",
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def segment_with_text(
        self, image: Image.Image, text_prompt: str
    ) -> dict:
        """
        Segment image using text prompt.

        Returns dict with:
            - masks: list of binary masks
            - boxes: list of bounding boxes [x1, y1, x2, y2]
            - scores: confidence scores
        """
        self.load_model()

        w, h = image.size
        masks = []
        boxes = []
        scores = []

        try:
            # Process inputs
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Handle different output formats from SAM3
            # Check for pred_boxes (DETR-style detection)
            if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
                pred_boxes = outputs.pred_boxes[0].cpu().numpy()

                # Get scores
                if hasattr(outputs, "objectness_logits"):
                    pred_scores = outputs.objectness_logits[0].sigmoid().cpu().numpy()
                elif hasattr(outputs, "pred_logits"):
                    pred_scores = outputs.pred_logits[0].softmax(-1).max(-1).values.cpu().numpy()
                else:
                    pred_scores = np.ones(len(pred_boxes))

                for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                    if score >= self.confidence_threshold:
                        # Convert normalized coords to pixel coords
                        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                        scores.append(float(score))

            # Handle masks if present
            if hasattr(outputs, "pred_masks") and outputs.pred_masks is not None:
                pred_masks = outputs.pred_masks[0].cpu().numpy()
                for i, mask in enumerate(pred_masks):
                    if i < len(scores):  # Only include masks for detected boxes
                        # Resize mask to image size if needed
                        if mask.shape[-2:] != (h, w):
                            from scipy.ndimage import zoom
                            scale_h = h / mask.shape[-2]
                            scale_w = w / mask.shape[-1]
                            if len(mask.shape) == 3:
                                mask = zoom(mask[0], (scale_h, scale_w), order=1)
                            else:
                                mask = zoom(mask, (scale_h, scale_w), order=1)
                        masks.append((mask > 0.5).astype(np.uint8))

        except Exception as e:
            logger.warning(f"SAM3 inference error: {e}")
            # Return empty results on error

        return {"masks": masks, "boxes": boxes, "scores": scores}

    def get_centroid(self, mask: np.ndarray) -> tuple:
        """Get centroid of a binary mask."""
        if mask.sum() == 0:
            return None
        y_coords, x_coords = np.where(mask > 0)
        return int(x_coords.mean()), int(y_coords.mean())

    def get_bbox(self, mask: np.ndarray) -> list:
        """Get bounding box of a binary mask."""
        if mask.sum() == 0:
            return None
        y_coords, x_coords = np.where(mask > 0)
        return [
            int(x_coords.min()),
            int(y_coords.min()),
            int(x_coords.max()),
            int(y_coords.max()),
        ]


def sample_frames(
    video_dir: str,
    num_frames: int = 20,
    output_dir: str = "data/labels/frames",
    seed: int = 42,
) -> list:
    """Sample random frames from multiple videos."""
    random.seed(seed)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(video_dir.rglob("*.mp4"))
    videos = [v for v in videos if "processed" not in str(v)]

    if len(videos) < num_frames:
        selected = videos
    else:
        selected = random.sample(videos, num_frames)

    frames = []
    print(f"Extracting {len(selected)} frames...")

    for i, video_path in enumerate(selected):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total < 10:
            cap.release()
            continue

        # Sample from middle of video
        frame_idx = random.randint(int(total * 0.2), int(total * 0.8))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Save frame
        name = f"frame_{i:03d}_{video_path.stem}.png"
        path = output_dir / name
        cv2.imwrite(str(path), frame)

        frames.append({
            "id": i,
            "path": str(path),
            "source_video": str(video_path),
            "frame_idx": frame_idx,
        })
        print(f"  [{i+1}/{len(selected)}] {video_path.name}")

    return frames


def auto_label_frames(
    frames: list,
    labeler: Sam3Labeler,
    prompts: dict = None,
) -> dict:
    """
    Auto-label frames using SAM 3 text prompts.

    Args:
        frames: List of frame info dicts
        labeler: Sam3Labeler instance
        prompts: Dict mapping label names to text prompts
                 Default: {"batter": "baseball batter", "glove": "catcher's mitt"}

    Returns:
        Dict mapping frame path to labels
    """
    if prompts is None:
        prompts = {
            "batter": "baseball batter",
            "glove": "catcher's mitt",
            "pitcher": "baseball pitcher",
        }

    labels = {}

    print(f"\nAuto-labeling {len(frames)} frames with SAM 3...")
    print(f"Prompts: {prompts}")

    for i, frame_info in enumerate(frames):
        path = frame_info["path"]
        print(f"\n[{i+1}/{len(frames)}] {Path(path).name}")

        image = Image.open(path)
        frame_labels = {}

        for label_name, text_prompt in prompts.items():
            print(f"  Searching for '{text_prompt}'...", end=" ")
            result = labeler.segment_with_text(image, text_prompt)

            if result["boxes"]:
                # Take highest confidence detection
                best_idx = np.argmax(result["scores"])
                box = result["boxes"][best_idx]
                score = result["scores"][best_idx]

                # Get centroid of box
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2

                frame_labels[label_name] = {
                    "point": [cx, cy],
                    "box": box,
                    "score": score,
                    "auto": True,
                }
                print(f"Found at ({cx}, {cy}) conf={score:.2f}")
            else:
                print("Not found")

        labels[Path(path).name] = frame_labels

    return labels


class LabelReviewer:
    """Interactive CV2 UI for reviewing and correcting labels."""

    def __init__(self, frames_dir: str, labels: dict):
        self.frames_dir = Path(frames_dir)
        self.frame_paths = sorted(self.frames_dir.glob("*.png"))
        self.labels = labels
        self.current_idx = 0
        self.current_mode = "batter"
        self.modes = ["batter", "glove", "pitcher"]
        self.colors = {"batter": (0, 255, 0), "glove": (255, 0, 255), "pitcher": (255, 165, 0)}

    def _get_key(self) -> str:
        return self.frame_paths[self.current_idx].name

    def _draw(self) -> np.ndarray:
        frame = cv2.imread(str(self.frame_paths[self.current_idx]))
        h, w = frame.shape[:2]
        key = self._get_key()

        # Draw labels
        if key in self.labels:
            for name, data in self.labels[key].items():
                color = self.colors.get(name, (255, 255, 255))
                if "point" in data:
                    x, y = data["point"]
                    cv2.circle(frame, (x, y), 12, color, -1)
                    cv2.circle(frame, (x, y), 14, (255, 255, 255), 2)

                    label = f"{name[0].upper()}"
                    if data.get("auto"):
                        label += f" ({data.get('score', 0):.2f})"
                    cv2.putText(frame, label, (x + 16, y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if "box" in data:
                    x1, y1, x2, y2 = data["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Status bar
        cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(frame, f"Frame {self.current_idx + 1}/{len(self.frame_paths)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}",
                    (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    self.colors[self.current_mode], 2)

        # Instructions
        cv2.putText(frame, "A/D=nav | 1=batter 2=glove 3=pitcher | Click=set | R=remove | Q=save",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return frame

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            key = self._get_key()
            if key not in self.labels:
                self.labels[key] = {}

            self.labels[key][self.current_mode] = {
                "point": [x, y],
                "auto": False,
            }
            print(f"Set {self.current_mode} at ({x}, {y})")

    def run(self) -> dict:
        """Run the review interface."""
        print("\n=== Label Reviewer ===")
        print("Review and correct auto-generated labels")
        print("A/D = navigate, 1/2 = switch mode, Click = set point, R = remove, Q = save\n")

        window = "Label Reviewer"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1280, 720)
        cv2.setMouseCallback(window, self._mouse)

        while True:
            cv2.imshow(window, self._draw())
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('a') or key == 81:
                self.current_idx = max(0, self.current_idx - 1)
            elif key == ord('d') or key == 83:
                self.current_idx = min(len(self.frame_paths) - 1, self.current_idx + 1)
            elif key == ord('1'):
                self.current_mode = "batter"
            elif key == ord('2'):
                self.current_mode = "glove"
            elif key == ord('3'):
                self.current_mode = "pitcher"
            elif key == ord('r'):
                key = self._get_key()
                if key in self.labels and self.current_mode in self.labels[key]:
                    del self.labels[key][self.current_mode]
                    print(f"Removed {self.current_mode}")

        cv2.destroyAllWindows()
        return self.labels


class TkinterLabelReviewer:
    """Tkinter GUI for reviewing and correcting labels with better UX."""

    COLORS = {
        "batter": "#4CAF50",    # Green
        "glove": "#E91E63",     # Pink
        "pitcher": "#FF9800",   # Orange
    }

    def __init__(self, frames_dir: str, labels: dict, frames_info: list = None):
        self.frames_dir = Path(frames_dir)
        self.frame_paths = sorted(self.frames_dir.glob("*.png"))
        self.labels = labels
        self.frames_info = frames_info or []
        self.current_idx = 0
        self.current_mode = "batter"
        self.modes = ["batter", "glove", "pitcher"]

        self.scale_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.canvas_image = None

        # Drawing state
        self.drawing = False
        self.draw_start = None
        self.current_rect = None

        self._build_gui()

    def _build_gui(self):
        """Build the Tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("SAM3 Frame Labeler - Baseball Biomechanics")
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
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Button-3>', self._on_right_click)
        self.canvas.bind('<Configure>', lambda e: self._display_frame())

        # Right panel
        right_frame = ttk.Frame(main_frame, width=280)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        right_frame.pack_propagate(False)

        # Info section
        info_frame = ttk.LabelFrame(right_frame, text="Frame Info", padding=10)
        info_frame.pack(fill=tk.X, pady=5)

        self.info_label = ttk.Label(info_frame, text="", wraplength=250)
        self.info_label.pack(fill=tk.X)

        self.progress_label = ttk.Label(info_frame, text="")
        self.progress_label.pack(fill=tk.X, pady=5)

        self.progress_bar = ttk.Progressbar(info_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X)

        # Mode selection
        mode_frame = ttk.LabelFrame(right_frame, text="Label Mode", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)

        self.mode_var = tk.StringVar(value=self.current_mode)
        for mode in self.modes:
            color = self.COLORS[mode]
            rb = ttk.Radiobutton(mode_frame, text=mode.capitalize(),
                                  variable=self.mode_var, value=mode,
                                  command=self._on_mode_change)
            rb.pack(anchor=tk.W)

        # Labels list
        labels_frame = ttk.LabelFrame(right_frame, text="Detected Objects", padding=10)
        labels_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.labels_listbox = tk.Listbox(labels_frame, height=8, bg='#3c3c3c',
                                          fg='white', selectmode=tk.SINGLE)
        self.labels_listbox.pack(fill=tk.BOTH, expand=True)

        ttk.Button(labels_frame, text="Delete Selected",
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

        # Export section
        export_frame = ttk.LabelFrame(right_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=5)

        ttk.Button(export_frame, text="Save Labels (JSON)",
                   command=self._save_json).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export for Tracking",
                   command=self._export_tracking).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Export COCO Format",
                   command=self._export_coco).pack(fill=tk.X, pady=2)

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('a', lambda e: self._prev_frame())
        self.root.bind('d', lambda e: self._next_frame())
        self.root.bind('1', lambda e: self._set_mode("batter"))
        self.root.bind('2', lambda e: self._set_mode("glove"))
        self.root.bind('3', lambda e: self._set_mode("pitcher"))
        self.root.bind('<Delete>', lambda e: self._delete_selected())
        self.root.bind('<Control-s>', lambda e: self._save_json())

        # Initial display
        self.root.after(100, self._display_frame)

    def _get_key(self) -> str:
        return self.frame_paths[self.current_idx].name

    def _on_mode_change(self):
        self.current_mode = self.mode_var.get()

    def _set_mode(self, mode: str):
        self.current_mode = mode
        self.mode_var.set(mode)

    def _display_frame(self):
        """Display current frame with annotations."""
        if not self.frame_paths:
            return

        path = self.frame_paths[self.current_idx]
        frame = cv2.imread(str(path))
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

        # Draw labels
        key = self._get_key()
        if key in self.labels:
            for name, data in self.labels[key].items():
                if "point" in data:
                    x, y = data["point"]
                    cx = x * self.scale_factor + self.img_offset_x
                    cy = y * self.scale_factor + self.img_offset_y
                    color = self.COLORS.get(name, "#FFFFFF")

                    self.canvas.create_oval(cx-10, cy-10, cx+10, cy+10,
                                             fill=color, outline='white', width=2)
                    auto_text = " (auto)" if data.get("auto") else ""
                    score = f" {data.get('score', 0):.2f}" if 'score' in data else ""
                    self.canvas.create_text(cx+15, cy, text=f"{name}{score}{auto_text}",
                                            fill=color, anchor=tk.W, font=('Arial', 10, 'bold'))

                if "box" in data:
                    x1, y1, x2, y2 = data["box"]
                    bx1 = x1 * self.scale_factor + self.img_offset_x
                    by1 = y1 * self.scale_factor + self.img_offset_y
                    bx2 = x2 * self.scale_factor + self.img_offset_x
                    by2 = y2 * self.scale_factor + self.img_offset_y
                    color = self.COLORS.get(name, "#FFFFFF")
                    self.canvas.create_rectangle(bx1, by1, bx2, by2,
                                                  outline=color, width=2)

        # Update info
        self.info_label.config(text=f"File: {path.name}")
        labeled = sum(1 for v in self.labels.values() if len(v) > 0)
        self.progress_label.config(text=f"Labeled: {labeled}/{len(self.frame_paths)}")
        self.progress_bar['value'] = (labeled / max(1, len(self.frame_paths))) * 100

        # Update listbox
        self._update_listbox()

    def _update_listbox(self):
        self.labels_listbox.delete(0, tk.END)
        key = self._get_key()
        if key in self.labels:
            for name, data in self.labels[key].items():
                marker = "[A]" if data.get("auto") else "[M]"
                score = f" ({data.get('score', 0):.2f})" if 'score' in data else ""
                self.labels_listbox.insert(tk.END, f"{marker} {name}{score}")

    def _on_click(self, event):
        """Handle left click - set label point."""
        img_x = (event.x - self.img_offset_x) / self.scale_factor
        img_y = (event.y - self.img_offset_y) / self.scale_factor

        # Check if click is within image bounds
        path = self.frame_paths[self.current_idx]
        frame = cv2.imread(str(path))
        if frame is None:
            return
        h, w = frame.shape[:2]

        if 0 <= img_x <= w and 0 <= img_y <= h:
            key = self._get_key()
            if key not in self.labels:
                self.labels[key] = {}

            self.labels[key][self.current_mode] = {
                "point": [int(img_x), int(img_y)],
                "auto": False,
            }
            logger.info(f"Set {self.current_mode} at ({int(img_x)}, {int(img_y)})")
            self._display_frame()

    def _on_right_click(self, event):
        """Handle right click - remove label under cursor."""
        img_x = (event.x - self.img_offset_x) / self.scale_factor
        img_y = (event.y - self.img_offset_y) / self.scale_factor

        key = self._get_key()
        if key not in self.labels:
            return

        # Find and remove nearest label
        to_remove = None
        min_dist = 50  # pixels threshold

        for name, data in self.labels[key].items():
            if "point" in data:
                px, py = data["point"]
                dist = ((px - img_x)**2 + (py - img_y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    to_remove = name

        if to_remove:
            del self.labels[key][to_remove]
            logger.info(f"Removed {to_remove}")
            self._display_frame()

    def _delete_selected(self):
        """Delete selected label from listbox."""
        sel = self.labels_listbox.curselection()
        if not sel:
            return

        key = self._get_key()
        if key not in self.labels:
            return

        idx = sel[0]
        label_names = list(self.labels[key].keys())
        if idx < len(label_names):
            del self.labels[key][label_names[idx]]
            self._display_frame()

    def _prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._display_frame()

    def _next_frame(self):
        if self.current_idx < len(self.frame_paths) - 1:
            self.current_idx += 1
            self._display_frame()

    def _save_json(self):
        """Save labels to JSON."""
        output_dir = self.frames_dir.parent
        filepath = output_dir / "labels.json"

        with open(filepath, 'w') as f:
            json.dump(self.labels, f, indent=2)

        messagebox.showinfo("Saved", f"Labels saved to {filepath}")
        logger.info(f"Labels saved to {filepath}")

    def _export_tracking(self):
        """Export labels for video tracking."""
        output_dir = self.frames_dir.parent
        filepath = output_dir / "tracking_init.json"

        # Group by source video if we have frames_info
        tracking_data = {
            "frames": [],
            "labels": self.modes,
            "format": "point_xy"
        }

        for i, path in enumerate(self.frame_paths):
            key = path.name
            if key not in self.labels:
                continue

            frame_data = {
                "frame_path": str(path),
                "frame_idx": i,
                "detections": []
            }

            for name, data in self.labels[key].items():
                det = {
                    "label": name,
                    "point": data.get("point"),
                    "box": data.get("box"),
                    "confidence": data.get("score", 1.0),
                    "is_auto": data.get("auto", False)
                }
                frame_data["detections"].append(det)

            tracking_data["frames"].append(frame_data)

        with open(filepath, 'w') as f:
            json.dump(tracking_data, f, indent=2)

        messagebox.showinfo("Exported", f"Tracking data saved to {filepath}")
        logger.info(f"Tracking data saved to {filepath}")

    def _export_coco(self):
        """Export labels in COCO format."""
        output_dir = self.frames_dir.parent
        filepath = output_dir / "coco_labels.json"

        categories = [{"id": i+1, "name": m} for i, m in enumerate(self.modes)]
        cat_map = {m: i+1 for i, m in enumerate(self.modes)}

        images = []
        annotations = []
        ann_id = 1

        for img_id, path in enumerate(self.frame_paths, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                continue

            h, w = frame.shape[:2]
            images.append({
                "id": img_id,
                "file_name": path.name,
                "width": w,
                "height": h
            })

            key = path.name
            if key not in self.labels:
                continue

            for name, data in self.labels[key].items():
                cat_id = cat_map.get(name, 1)

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "iscrowd": 0,
                }

                if "box" in data:
                    x1, y1, x2, y2 = data["box"]
                    ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    ann["area"] = (x2 - x1) * (y2 - y1)
                elif "point" in data:
                    # Create small box around point
                    px, py = data["point"]
                    ann["bbox"] = [px - 20, py - 20, 40, 40]
                    ann["area"] = 1600
                    ann["keypoints"] = [px, py, 2]  # visible keypoint

                annotations.append(ann)
                ann_id += 1

        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        with open(filepath, 'w') as f:
            json.dump(coco_data, f, indent=2)

        messagebox.showinfo("Exported", f"COCO labels saved to {filepath}")
        logger.info(f"COCO labels saved to {filepath}")

    def run(self) -> dict:
        """Run the GUI and return labels."""
        self.root.mainloop()
        return self.labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3 Frame Labeler")
    parser.add_argument("--video-dir", default="data/videos/2024/04")
    parser.add_argument("--output-dir", default="data/labels")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-auto", action="store_true", help="Skip auto-labeling, go straight to review")
    parser.add_argument("--cache-dir", default="F:/hf_cache")
    parser.add_argument("--gui", choices=["cv2", "tkinter"], default="tkinter",
                        help="GUI type: 'cv2' for OpenCV or 'tkinter' for Tkinter (default)")
    args = parser.parse_args()

    project = Path(__file__).parent.parent
    video_dir = project / args.video_dir
    output_dir = project / args.output_dir
    frames_dir = output_dir / "frames"
    labels_file = output_dir / "labels.json"

    # Step 1: Extract frames
    if not args.skip_extract:
        if frames_dir.exists() and list(frames_dir.glob("*.png")):
            resp = input(f"Frames exist in {frames_dir}. Re-extract? (y/n): ")
            if resp.lower() != 'y':
                args.skip_extract = True

        if not args.skip_extract:
            frames = sample_frames(str(video_dir), args.num_frames, str(frames_dir))
            with open(output_dir / "frames_info.json", "w") as f:
                json.dump(frames, f, indent=2)

    # Step 2: Auto-label with SAM 3
    labels = {}
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)
        print(f"Loaded existing labels for {len(labels)} frames")

    if not args.skip_auto:
        labeler = Sam3Labeler(cache_dir=args.cache_dir)
        frame_paths = list(frames_dir.glob("*.png"))
        frames = [{"path": str(p)} for p in frame_paths]
        labels = auto_label_frames(frames, labeler)

        # Save auto labels
        with open(labels_file, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"\nAuto-labels saved to {labels_file}")

    # Step 3: Review and correct
    print(f"\nLaunching {args.gui} label reviewer...")
    if args.gui == "tkinter":
        reviewer = TkinterLabelReviewer(str(frames_dir), labels)
    else:
        reviewer = LabelReviewer(str(frames_dir), labels)
    labels = reviewer.run()

    # Save final labels
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\nFinal labels saved to {labels_file}")

    # Summary
    complete = sum(1 for v in labels.values()
                   if "batter" in v and "glove" in v and "pitcher" in v)
    partial = sum(1 for v in labels.values() if len(v) > 0)
    print(f"Complete labels (all 3): {complete}/{len(labels)}")
    print(f"Partial labels (any): {partial}/{len(labels)}")


if __name__ == "__main__":
    main()
