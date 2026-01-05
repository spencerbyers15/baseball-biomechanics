"""
Test fine-tuned SAM3 model vs original on new frames.
Overlays detected masks on frames for visual comparison.
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
from tqdm import tqdm


class MittModelTester:
    """Compare original vs fine-tuned SAM3 for mitt detection."""

    def __init__(self, cache_dir: str = "F:/hf_cache"):
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.original_model = None
        self.finetuned_model = None
        self.processor = None

    def load_models(self, finetuned_path: str):
        """Load both original and fine-tuned models."""
        from transformers import Sam3Model, Sam3Processor

        print("Loading original SAM3...")
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir
        )
        self.original_model = Sam3Model.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir, torch_dtype=torch.bfloat16
        )
        self.original_model.to(self.device)
        self.original_model.eval()

        print("Loading fine-tuned SAM3...")
        self.finetuned_processor = Sam3Processor.from_pretrained(finetuned_path)
        self.finetuned_model = Sam3Model.from_pretrained(
            finetuned_path, torch_dtype=torch.bfloat16
        )
        self.finetuned_model.to(self.device)
        self.finetuned_model.eval()

        print(f"Models loaded on {self.device}")

    def detect_mitt(self, model, processor, image: np.ndarray, text_prompt: str = "catcher's mitt"):
        """Run detection and return mask, box, score."""
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.2,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        # Check if masks is empty
        masks = results.get("masks", [])
        if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
            return None, None, 0.0

        # Find best detection (highest score)
        best_idx = 0
        best_score = float(results["scores"][0].cpu()) if hasattr(results["scores"][0], 'cpu') else results["scores"][0]

        for i, score in enumerate(results["scores"]):
            s = float(score.cpu()) if hasattr(score, 'cpu') else score
            if s > best_score:
                best_score = s
                best_idx = i

        mask = results["masks"][best_idx]
        box = results["boxes"][best_idx]

        # Convert to numpy (handle bfloat16)
        if hasattr(mask, 'cpu'):
            mask_np = mask.cpu().float().numpy()
        else:
            mask_np = np.array(mask)

        if hasattr(box, 'cpu'):
            box_np = box.cpu().float().numpy()
        else:
            box_np = np.array(box)

        return mask_np.astype(np.uint8), [int(x) for x in box_np], best_score

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.5):
        """Overlay colored mask on image."""
        if mask is None:
            return image

        output = image.copy()
        h, w = image.shape[:2]

        # Resize mask if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # Create colored overlay
        overlay = output.copy()
        overlay[mask > 0] = color
        output = cv2.addWeighted(output, 1 - alpha, overlay, alpha, 0)

        # Draw mask contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)

        return output

    def create_comparison(
        self,
        image: np.ndarray,
        orig_mask: np.ndarray,
        orig_box: list,
        orig_score: float,
        ft_mask: np.ndarray,
        ft_box: list,
        ft_score: float,
    ) -> np.ndarray:
        """Create side-by-side comparison image."""
        h, w = image.shape[:2]

        # Colors
        ORIG_COLOR = (0, 165, 255)  # Orange for original
        FT_COLOR = (0, 255, 0)      # Green for fine-tuned

        # Create original result
        orig_result = self.overlay_mask(image.copy(), orig_mask, ORIG_COLOR, 0.4)
        if orig_box:
            cv2.rectangle(orig_result, (orig_box[0], orig_box[1]), (orig_box[2], orig_box[3]), ORIG_COLOR, 2)
        cv2.putText(orig_result, f"ORIGINAL: {orig_score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(orig_result, f"ORIGINAL: {orig_score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORIG_COLOR, 1)

        # Create fine-tuned result
        ft_result = self.overlay_mask(image.copy(), ft_mask, FT_COLOR, 0.4)
        if ft_box:
            cv2.rectangle(ft_result, (ft_box[0], ft_box[1]), (ft_box[2], ft_box[3]), FT_COLOR, 2)
        cv2.putText(ft_result, f"FINE-TUNED: {ft_score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(ft_result, f"FINE-TUNED: {ft_score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, FT_COLOR, 1)

        # Stack side by side
        comparison = np.hstack([orig_result, ft_result])

        # Add divider line
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

        return comparison


def main():
    project_dir = Path("F:/Claude_Projects/baseball-biomechanics")
    video_dir = project_dir / "data" / "videos" / "2024" / "04"
    output_dir = project_dir / "data" / "debug" / "mitt_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    finetuned_path = project_dir / "models" / "sam3_mitt"

    # Get test videos (different from training)
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print("No videos found")
        return

    # Use videos that weren't in training set
    test_videos = videos[5:10]  # Skip first 5 to avoid training data overlap

    tester = MittModelTester()
    tester.load_models(str(finetuned_path))

    print("\n" + "=" * 70)
    print("COMPARISON: Original SAM3 vs Fine-tuned SAM3")
    print("=" * 70)

    results_summary = []

    for video_path in test_videos[:3]:  # Test on 3 videos
        print(f"\nVideo: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample 3 frames per video
        test_indices = [int(total * r) for r in [0.35, 0.5, 0.65]]

        for frame_idx in test_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect with original
            orig_mask, orig_box, orig_score = tester.detect_mitt(
                tester.original_model, tester.processor, frame
            )

            # Detect with fine-tuned
            ft_mask, ft_box, ft_score = tester.detect_mitt(
                tester.finetuned_model, tester.finetuned_processor, frame
            )

            # Print results
            orig_str = f"{orig_score:.3f}" if orig_score > 0 else "NOT FOUND"
            ft_str = f"{ft_score:.3f}" if ft_score > 0 else "NOT FOUND"
            improvement = ""
            if orig_score > 0 and ft_score > 0:
                diff = ft_score - orig_score
                improvement = f" ({'+' if diff > 0 else ''}{diff:.3f})"

            print(f"  Frame {frame_idx}: Original={orig_str}, Fine-tuned={ft_str}{improvement}")

            results_summary.append({
                "video": video_path.name,
                "frame": frame_idx,
                "original": orig_score,
                "finetuned": ft_score,
            })

            # Create comparison image
            comparison = tester.create_comparison(
                frame, orig_mask, orig_box, orig_score,
                ft_mask, ft_box, ft_score
            )

            # Save
            out_name = f"{video_path.stem}_frame{frame_idx}_comparison.jpg"
            cv2.imwrite(str(output_dir / out_name), comparison)

        cap.release()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    orig_scores = [r["original"] for r in results_summary if r["original"] > 0]
    ft_scores = [r["finetuned"] for r in results_summary if r["finetuned"] > 0]

    orig_detected = sum(1 for r in results_summary if r["original"] > 0)
    ft_detected = sum(1 for r in results_summary if r["finetuned"] > 0)
    total_frames = len(results_summary)

    print(f"\nDetection Rate:")
    print(f"  Original:   {orig_detected}/{total_frames} ({100*orig_detected/total_frames:.0f}%)")
    print(f"  Fine-tuned: {ft_detected}/{total_frames} ({100*ft_detected/total_frames:.0f}%)")

    if orig_scores:
        print(f"\nAverage Confidence (when detected):")
        print(f"  Original:   {np.mean(orig_scores):.3f}")
    if ft_scores:
        print(f"  Fine-tuned: {np.mean(ft_scores):.3f}")

    if orig_scores and ft_scores:
        improvement = np.mean(ft_scores) - np.mean(orig_scores)
        print(f"\nImprovement: {'+' if improvement > 0 else ''}{improvement:.3f} ({100*improvement/np.mean(orig_scores):.1f}%)")

    print(f"\nComparison images saved to: {output_dir}")


if __name__ == "__main__":
    main()
