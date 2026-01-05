"""
Improved Catcher's Mitt Detection using SAM3 Box Prompts.

Two approaches:
1. QUICK FIX (no training): Use position-based heuristics + negative prompts
   - Catcher's mitt is always in the right side of frame (behind home plate)
   - Pitcher's glove is center-left (on mound)

2. FINE-TUNED: Train mask decoder on labeled data (requires annotations)
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
from PIL import Image


class ImprovedMittDetector:
    """
    Improved catcher's mitt detection using SAM3 with spatial constraints.

    Key insight: In broadcast baseball footage:
    - Catcher's mitt: RIGHT side of frame (behind home plate), relatively stationary
    - Pitcher's glove: CENTER-LEFT of frame (on mound), moves with pitcher
    """

    def __init__(self, cache_dir: str = "F:/hf_cache"):
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load SAM3 model."""
        if self.model is not None:
            return

        from transformers import Sam3Model, Sam3Processor

        print("Loading SAM3...")
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir
        )
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir, torch_dtype=torch.bfloat16
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def detect_mitt_with_spatial_filter(
        self,
        image: np.ndarray,
        text_prompt: str = "catcher's mitt",
        min_x_ratio: float = 0.5,  # Mitt should be in right half of frame
        confidence_threshold: float = 0.3,
    ) -> Dict:
        """
        Detect catcher's mitt using text prompt + spatial filtering.

        Args:
            image: BGR image from OpenCV
            text_prompt: Text prompt for detection
            min_x_ratio: Minimum x position (normalized) - filters out left side
            confidence_threshold: Minimum confidence score

        Returns:
            Dict with best detection (mask, box, score) or empty if none found
        """
        self.load_model()

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Run SAM3 with text prompt
        inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.1,  # Low threshold to get all candidates
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        if not results["masks"]:
            return {"found": False}

        # Filter by spatial position (right side of frame = catcher's area)
        best_detection = None
        best_score = 0

        for i, (mask, box, score) in enumerate(zip(
            results["masks"], results["boxes"], results["scores"]
        )):
            # Convert box to numpy
            if hasattr(box, 'cpu'):
                box = box.cpu().numpy()
            if hasattr(score, 'cpu'):
                score = float(score.cpu())

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2  # Center x

            # Spatial filter: catcher's mitt is on RIGHT side of frame
            if cx / w < min_x_ratio:
                continue  # Skip detections on left side (likely pitcher's glove)

            if score < confidence_threshold:
                continue

            # Prefer rightmost, highest confidence detection
            combined_score = score * (cx / w)  # Boost score for rightmost detections

            if combined_score > best_score:
                best_score = combined_score
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                best_detection = {
                    "found": True,
                    "mask": mask_np.astype(np.uint8),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "score": score,
                    "center": (int(cx), int((y1 + y2) / 2)),
                }

        return best_detection or {"found": False}

    def detect_mitt_with_negative_prompt(
        self,
        image: np.ndarray,
        pitcher_glove_box: Optional[List[int]] = None,
    ) -> Dict:
        """
        Detect catcher's mitt using text + negative box prompt.

        This tells SAM3 to find "catcher's mitt" but EXCLUDE the region
        containing the pitcher's glove.

        Args:
            image: BGR image
            pitcher_glove_box: [x1, y1, x2, y2] of pitcher's glove to exclude
        """
        self.load_model()

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Build inputs
        text_prompt = "catcher's mitt"

        if pitcher_glove_box is not None:
            # Use negative box prompt to exclude pitcher's glove area
            inputs = self.processor(
                images=pil_image,
                text=text_prompt,
                input_boxes=[[pitcher_glove_box]],
                input_boxes_labels=[[0]],  # 0 = negative (exclude)
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                images=pil_image,
                text=text_prompt,
                return_tensors="pt"
            )

        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        if not results["masks"]:
            return {"found": False}

        # Take best detection
        best_idx = 0
        best_score = float(results["scores"][0].cpu()) if hasattr(results["scores"][0], 'cpu') else results["scores"][0]

        for i, score in enumerate(results["scores"]):
            s = float(score.cpu()) if hasattr(score, 'cpu') else score
            if s > best_score:
                best_score = s
                best_idx = i

        mask = results["masks"][best_idx]
        box = results["boxes"][best_idx]

        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
        box_np = box.cpu().numpy() if hasattr(box, 'cpu') else box

        return {
            "found": True,
            "mask": mask_np.astype(np.uint8),
            "box": [int(x) for x in box_np],
            "score": best_score,
        }

    def estimate_pitcher_glove_region(self, image: np.ndarray) -> List[int]:
        """
        Estimate the region where pitcher's glove would be.

        In standard broadcast view:
        - Pitcher is in center-left of frame
        - Glove is typically in the lower-center area
        """
        h, w = image.shape[:2]

        # Pitcher's glove region estimate (center-left, lower half)
        x1 = int(w * 0.25)
        y1 = int(h * 0.35)
        x2 = int(w * 0.55)
        y2 = int(h * 0.75)

        return [x1, y1, x2, y2]


def test_improved_detection():
    """Test the improved detection on sample frames."""
    project_dir = Path("F:/Claude_Projects/baseball-biomechanics")
    video_dir = project_dir / "data" / "videos" / "2024" / "04"
    output_dir = project_dir / "data" / "debug" / "mitt_improved"
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ImprovedMittDetector()

    # Get a test video
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print("No videos found")
        return

    video_path = videos[0]
    print(f"Testing on: {video_path.name}")

    # Extract test frames
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    test_indices = [int(total * r) for r in [0.3, 0.4, 0.5, 0.6, 0.7]]

    print("\n" + "=" * 60)
    print("IMPROVED DETECTION RESULTS")
    print("=" * 60)

    for i, frame_idx in enumerate(test_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        print(f"\nFrame {frame_idx}:")

        # Method 1: Spatial filtering
        result1 = detector.detect_mitt_with_spatial_filter(frame)
        if result1["found"]:
            print(f"  Spatial filter: score={result1['score']:.3f}, box={result1['box']}")
        else:
            print(f"  Spatial filter: Not found")

        # Method 2: Negative prompt (exclude pitcher area)
        pitcher_box = detector.estimate_pitcher_glove_region(frame)
        result2 = detector.detect_mitt_with_negative_prompt(frame, pitcher_box)
        if result2["found"]:
            print(f"  Negative prompt: score={result2['score']:.3f}, box={result2['box']}")
        else:
            print(f"  Negative prompt: Not found")

        # Save visualization
        vis = frame.copy()

        # Draw pitcher exclusion zone
        cv2.rectangle(vis, (pitcher_box[0], pitcher_box[1]),
                      (pitcher_box[2], pitcher_box[3]), (0, 0, 255), 2)
        cv2.putText(vis, "Pitcher (exclude)", (pitcher_box[0], pitcher_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw spatial filter result
        if result1["found"]:
            box = result1["box"]
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
            cv2.putText(vis, f"Mitt (spatial): {result1['score']:.2f}",
                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(str(output_dir / f"frame_{frame_idx}_improved.jpg"), vis)

    cap.release()
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    test_improved_detection()
