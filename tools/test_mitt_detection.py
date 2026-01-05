"""
Test SAM3 text-prompted segmentation for catcher's mitt detection.
Evaluates different text prompts to find the most reliable one.
"""

import os
import sys
import warnings

# Suppress verbose output before imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from pathlib import Path
import cv2
import numpy as np
import torch


def load_sam3(cache_dir: str = "F:/hf_cache"):
    """Load SAM3 model and processor."""
    from transformers import Sam3Model, Sam3Processor

    print("Loading SAM3 model...")
    processor = Sam3Processor.from_pretrained(
        "facebook/sam3",
        cache_dir=cache_dir,
    )
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    return model, processor, device


def extract_frames(video_path: str, num_frames: int = 8) -> list:
    """Extract evenly spaced frames from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample from middle portion of video (20%-80%)
    start_frame = int(total * 0.2)
    end_frame = int(total * 0.8)
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))

    cap.release()
    return frames


def segment_with_text(model, processor, image, text_prompt: str, device: str):
    """
    Run SAM3 segmentation with text prompt.

    Returns:
        masks: list of numpy arrays (binary masks)
        boxes: list of [x1, y1, x2, y2]
        scores: list of confidence scores
    """
    from PIL import Image

    # Convert BGR to RGB PIL Image
    if isinstance(image, np.ndarray):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image

    h, w = pil_image.size[1], pil_image.size[0]

    # Process inputs
    inputs = processor(
        images=pil_image,
        text=text_prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    masks = []
    boxes = []
    scores = []

    # Extract predictions - handle different output formats
    if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
        pred_boxes = outputs.pred_boxes[0].cpu().float().numpy()

        # Get scores (convert to float32 before numpy)
        if hasattr(outputs, "objectness_logits") and outputs.objectness_logits is not None:
            pred_scores = torch.sigmoid(outputs.objectness_logits[0]).cpu().float().numpy()
        elif hasattr(outputs, "pred_logits") and outputs.pred_logits is not None:
            logits = outputs.pred_logits[0].float()
            # Handle different logit shapes
            if logits.dim() == 1:
                pred_scores = logits.softmax(-1).cpu().numpy()
                if pred_scores.ndim == 0:
                    pred_scores = np.array([float(pred_scores)])
            else:
                pred_scores = logits.softmax(-1).max(-1).values.cpu().numpy()
        else:
            pred_scores = np.ones(len(pred_boxes))

        # Ensure pred_scores is iterable
        if pred_scores.ndim == 0:
            pred_scores = np.array([float(pred_scores)])

        # Ensure pred_boxes is 2D
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.reshape(1, -1)

        for i, box in enumerate(pred_boxes):
            score = pred_scores[i] if i < len(pred_scores) else 0.5
            if score < 0.1:  # Very low threshold to see all detections
                continue
            # Convert normalized coords to pixels
            if box.max() <= 1.0:  # Normalized
                x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
            else:
                x1, y1, x2, y2 = box[:4]
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(score))

    # Extract masks if available
    if hasattr(outputs, "pred_masks") and outputs.pred_masks is not None:
        pred_masks = outputs.pred_masks[0].cpu().float().numpy()
        for i, mask in enumerate(pred_masks):
            if i >= len(scores):
                break
            # Resize mask to image size
            if mask.ndim == 3:
                mask = mask[0]
            mask = mask.astype(np.float32)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append((mask > 0.5).astype(np.uint8))

    return masks, boxes, scores


def draw_results(frame, masks, boxes, scores, prompt: str, color=(255, 0, 255)):
    """Draw segmentation results on frame."""
    output = frame.copy()
    h, w = frame.shape[:2]

    # Draw masks with transparency
    for i, mask in enumerate(masks):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = output.copy()
        overlay[mask > 0] = color
        output = cv2.addWeighted(output, 0.6, overlay, 0.4, 0)

    # Draw boxes and scores
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{score:.2f}"
        cv2.putText(output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add prompt label
    cv2.putText(output, f"Prompt: {prompt}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(output, f"Detections: {len(boxes)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output


def main():
    project_dir = Path("F:/Claude_Projects/baseball-biomechanics")
    video_dir = project_dir / "data" / "videos" / "2024" / "04"
    output_dir = project_dir / "data" / "debug" / "mitt_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find a video
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print("No videos found!")
        return

    video_path = str(videos[0])
    print(f"Using video: {Path(video_path).name}")

    # Text prompts to test
    prompts = [
        "catcher's mitt",
        "baseball glove",
        "catcher's glove",
        "baseball mitt",
        "leather glove",
    ]

    # Load model
    model, processor, device = load_sam3()

    # Extract frames
    print(f"\nExtracting frames...")
    frames = extract_frames(video_path, num_frames=6)
    print(f"Extracted {len(frames)} frames")

    # Test each prompt
    results = {p: [] for p in prompts}

    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    for frame_idx, frame in frames:
        print(f"\nFrame {frame_idx}:")

        for prompt in prompts:
            masks, boxes, scores = segment_with_text(model, processor, frame, prompt, device)

            # Store results
            results[prompt].append({
                "frame_idx": frame_idx,
                "num_detections": len(boxes),
                "scores": scores,
                "max_score": max(scores) if scores else 0,
            })

            # Print results
            if scores:
                print(f"  '{prompt}': {len(boxes)} det, max={max(scores):.3f}, scores={[f'{s:.2f}' for s in scores]}")
            else:
                print(f"  '{prompt}': No detections")

            # Save annotated frame
            if masks or boxes:
                output_frame = draw_results(frame, masks, boxes, scores, prompt)
                safe_prompt = prompt.replace("'", "").replace(" ", "_")
                out_path = output_dir / f"frame_{frame_idx}_{safe_prompt}.jpg"
                cv2.imwrite(str(out_path), output_frame)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY BY PROMPT")
    print("=" * 60)

    for prompt in prompts:
        prompt_results = results[prompt]
        total_dets = sum(r["num_detections"] for r in prompt_results)
        avg_max = np.mean([r["max_score"] for r in prompt_results])
        frames_with_det = sum(1 for r in prompt_results if r["num_detections"] > 0)

        print(f"\n'{prompt}':")
        print(f"  Total detections: {total_dets}")
        print(f"  Frames with detection: {frames_with_det}/{len(frames)}")
        print(f"  Avg max confidence: {avg_max:.3f}")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
