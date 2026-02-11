"""
Test SAM3 with text prompt "home plate" for better detection.

This uses the full HuggingFace Transformers SAM3 model which supports text prompts,
unlike Ultralytics SAM2.1 which only supports point/box prompts.
"""

import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class SAM3HomePlateDetector:
    """Home plate detection using SAM3 with text prompts."""

    def __init__(self, cache_dir: str = "F:/hf_cache"):
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load SAM3 model with text encoder."""
        if self.model is not None:
            return

        from transformers import Sam3Model, Sam3Processor

        print("Loading SAM3 with text encoder...")
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir
        )
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir, torch_dtype=torch.bfloat16
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def detect_home_plate(
        self,
        image: np.ndarray,
        text_prompt: str = "home plate",
        confidence_threshold: float = 0.3,
    ) -> dict:
        """
        Detect home plate using SAM3 text prompt.

        Args:
            image: BGR image from OpenCV
            text_prompt: Text description ("home plate")
            confidence_threshold: Minimum confidence

        Returns:
            Dict with detection results
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

        # Post-process to get masks
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.1,  # Low threshold to see all candidates
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        masks = results.get("masks", [])
        if not hasattr(masks, '__len__') or len(masks) == 0:
            return {"found": False, "all_detections": []}

        # Collect all detections
        all_detections = []
        best_detection = None
        best_score = 0

        for i, (mask, box, score) in enumerate(zip(
            results["masks"], results["boxes"], results["scores"]
        )):
            if hasattr(box, 'cpu'):
                box = box.float().cpu().numpy()
            if hasattr(score, 'cpu'):
                score = float(score.float().cpu())
            if hasattr(mask, 'cpu'):
                mask = mask.float().cpu().numpy()

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detection = {
                "mask": mask.astype(np.uint8),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": score,
                "center": (int(cx), int(cy)),
            }
            all_detections.append(detection)

            # Track best detection above threshold
            if score > confidence_threshold and score > best_score:
                best_score = score
                best_detection = detection

        if best_detection:
            return {
                "found": True,
                **best_detection,
                "all_detections": all_detections,
            }
        else:
            return {"found": False, "all_detections": all_detections}

    def visualize(
        self,
        image: np.ndarray,
        result: dict,
        show_all: bool = True,
    ) -> np.ndarray:
        """Draw detections on image."""
        vis = image.copy()
        h, w = image.shape[:2]

        if show_all and result.get("all_detections"):
            # Draw all detections in gray
            for det in result["all_detections"]:
                box = det["box"]
                score = det["score"]
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (128, 128, 128), 1)
                cv2.putText(vis, f"{score:.2f}", (box[0], box[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        if result.get("found"):
            # Draw best detection in green
            box = result["box"]
            score = result["score"]
            center = result["center"]
            mask = result["mask"]

            # Overlay mask
            mask_color = np.zeros_like(vis)
            mask_color[:, :, 1] = (mask * 255).astype(np.uint8)  # Green mask
            vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

            # Draw bounding box
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(vis, center, 6, (255, 0, 255), -1)

            # Label
            label = f"home plate: {score:.2f}"
            cv2.putText(vis, label, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "No home plate detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return vis


def test_on_video_frames():
    """Test SAM3 home plate detection on frames from recent demo video."""

    # Find the most recent demo video
    demo_dir = PROJECT_ROOT / "data" / "debug" / "demo_pipeline" / "videos"

    if not demo_dir.exists():
        print(f"Demo directory not found: {demo_dir}")
        print("Looking for any video in data/videos...")
        demo_dir = PROJECT_ROOT / "data" / "videos"

    videos = list(demo_dir.rglob("*.mp4"))
    if not videos:
        print("No videos found!")
        return

    # Use most recent video
    video_path = max(videos, key=lambda p: p.stat().st_mtime)
    print(f"Testing on: {video_path}")

    # Output directory
    output_dir = PROJECT_ROOT / "data" / "debug" / "home_plate_sam3_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = SAM3HomePlateDetector()

    # Extract frames to test
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Test on 5 evenly spaced frames
    test_indices = [int(total_frames * r) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]

    print("\n" + "=" * 60)
    print("SAM3 HOME PLATE DETECTION TEST (Text Prompt: 'home plate')")
    print("=" * 60)

    results_summary = []

    for frame_idx in test_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        print(f"\nFrame {frame_idx}:")

        # Detect home plate
        result = detector.detect_home_plate(frame, text_prompt="home plate")

        if result["found"]:
            print(f"  FOUND: score={result['score']:.3f}, box={result['box']}, center={result['center']}")
            results_summary.append((frame_idx, True, result['score']))
        else:
            n_candidates = len(result.get("all_detections", []))
            print(f"  NOT FOUND ({n_candidates} candidates below threshold)")
            if result.get("all_detections"):
                for i, det in enumerate(result["all_detections"][:3]):  # Show top 3
                    print(f"    Candidate {i+1}: score={det['score']:.3f}, box={det['box']}")
            results_summary.append((frame_idx, False, 0))

        # Save visualization
        vis = detector.visualize(frame, result, show_all=True)
        output_path = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(output_path), vis)

    cap.release()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    found_count = sum(1 for _, found, _ in results_summary if found)
    print(f"Detection rate: {found_count}/{len(results_summary)} ({100*found_count/len(results_summary):.1f}%)")
    if found_count > 0:
        avg_score = np.mean([s for _, found, s in results_summary if found])
        print(f"Average score: {avg_score:.3f}")
    print(f"\nResults saved to: {output_dir}")


def test_on_single_frame(frame_path: str):
    """Test on a single frame image."""
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Could not read: {frame_path}")
        return

    detector = SAM3HomePlateDetector()
    result = detector.detect_home_plate(frame, text_prompt="home plate")

    print(f"\nResult for {frame_path}:")
    if result["found"]:
        print(f"  FOUND: score={result['score']:.3f}, center={result['center']}")
    else:
        print(f"  NOT FOUND")
        for i, det in enumerate(result.get("all_detections", [])[:5]):
            print(f"    Candidate {i+1}: score={det['score']:.3f}, box={det['box']}")

    vis = detector.visualize(frame, result)
    output_path = Path(frame_path).stem + "_sam3_result.jpg"
    cv2.imwrite(output_path, vis)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=str, help="Test on single frame image")
    args = parser.parse_args()

    if args.frame:
        test_on_single_frame(args.frame)
    else:
        test_on_video_frames()
