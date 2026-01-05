#!/usr/bin/env python
"""Test SAM3 for home plate detection - point prompt approach."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import SAM

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
OUTPUT_DIR = PROJECT_ROOT / "data/debug/home_plate_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_test_frames(n_videos=10):
    """Extract first frame from random videos."""
    videos_dir = PROJECT_ROOT / "data/videos/2024"
    all_videos = list(videos_dir.rglob("*.mp4"))

    import random
    sample_videos = random.sample(all_videos, min(n_videos, len(all_videos)))

    frames = []
    for video_path in sample_videos:
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frames.append((video_path.stem, frame))

    return frames


def is_white_region(frame, mask, threshold=0.5):
    """Check if the masked region is predominantly white/bright."""
    masked_pixels = frame[mask > 0.5]
    if len(masked_pixels) == 0:
        return False, 0.0

    # Check brightness (average of RGB)
    brightness = np.mean(masked_pixels, axis=1)
    bright_count = np.sum(brightness > 160)
    bright_ratio = bright_count / len(brightness)

    return bright_ratio > threshold, bright_ratio


def detect_home_plate_point_prompt(frame, model):
    """
    Detect home plate using SAM3 with point prompts.

    Strategy: Try multiple points in the expected plate region,
    score each result by whiteness and shape.
    """
    h, w = frame.shape[:2]

    # Home plate approximate location (varies by camera angle)
    # Try a grid of points in the expected region
    candidate_points = [
        (int(w * 0.48), int(h * 0.38)),  # Center-ish
        (int(w * 0.50), int(h * 0.36)),  # Slightly right/up
        (int(w * 0.46), int(h * 0.40)),  # Slightly left/down
        (int(w * 0.52), int(h * 0.35)),  # Right/up
        (int(w * 0.44), int(h * 0.42)),  # Left/down
    ]

    best_result = None
    best_score = -1

    for px, py in candidate_points:
        # SAM with point prompt
        results = model(frame, points=[px, py], labels=[1], verbose=False)

        if not results or len(results) == 0:
            continue

        result = results[0]
        if not hasattr(result, "masks") or result.masks is None:
            continue

        masks = result.masks.data.cpu().numpy()
        if len(masks) == 0:
            continue

        # Get the mask for this point
        mask = masks[0]

        # Get contour info
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        frame_area = h * w

        # Size filter: home plate is small
        if area < frame_area * 0.0005 or area > frame_area * 0.02:
            continue

        # Check whiteness
        is_white, white_ratio = is_white_region(frame, mask)

        # Shape analysis
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        n_corners = len(approx)

        # Aspect ratio
        x, y, bw, bh = cv2.boundingRect(largest)
        aspect = bw / max(bh, 1)

        # Score: prioritize white, pentagon-like shape
        score = white_ratio * 3.0
        if 4 <= n_corners <= 7:
            score += 1.0
        if 1.2 <= aspect <= 2.2:
            score += 0.5

        if score > best_score:
            best_score = score
            best_result = {
                "mask": mask,
                "contour": largest,
                "approx": approx,
                "point": (px, py),
                "area": area,
                "white_ratio": white_ratio,
                "n_corners": n_corners,
                "aspect": aspect,
                "score": score,
            }

    return best_result


def visualize_detection(frame, result, output_path):
    """Draw detection visualization."""
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Draw search region
    region = {
        "x1": int(w * 0.40),
        "y1": int(h * 0.30),
        "x2": int(w * 0.60),
        "y2": int(h * 0.45),
    }
    cv2.rectangle(vis, (region["x1"], region["y1"]),
                  (region["x2"], region["y2"]), (255, 255, 0), 1)

    if result is not None:
        # Overlay mask
        mask_color = np.zeros_like(frame)
        mask_color[:, :, 1] = (result["mask"] * 255).astype(np.uint8)
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        # Draw contour and corners
        cv2.drawContours(vis, [result["contour"]], -1, (0, 255, 0), 2)
        for point in result["approx"]:
            cv2.circle(vis, tuple(point[0]), 4, (0, 0, 255), -1)

        # Draw prompt point
        cv2.circle(vis, result["point"], 8, (255, 0, 255), -1)
        cv2.circle(vis, result["point"], 10, (255, 255, 255), 2)

        # Info
        info = f"Score: {result['score']:.2f} | White: {result['white_ratio']:.0%} | Corners: {result['n_corners']}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "No home plate detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imwrite(str(output_path), vis)


def main():
    print("Loading SAM3 model...")
    model = SAM("sam2.1_b.pt")

    print("Extracting test frames...")
    frames = extract_test_frames(15)
    print(f"Got {len(frames)} frames")

    print("\n=== Testing home plate detection (point prompt) ===")
    success_count = 0
    for video_name, frame in frames:
        print(f"Processing: {video_name}")

        result = detect_home_plate_point_prompt(frame, model)

        if result and result["white_ratio"] > 0.3:
            print(f"  -> Found! Score: {result['score']:.2f}, White: {result['white_ratio']:.0%}, Corners: {result['n_corners']}")
            success_count += 1
        elif result:
            print(f"  -> Low confidence: White: {result['white_ratio']:.0%}")
        else:
            print(f"  -> No valid detection")

        output_path = OUTPUT_DIR / f"{video_name}_plate_v3.jpg"
        visualize_detection(frame, result, output_path)

    print(f"\n=== Results: {success_count}/{len(frames)} confident detections ===")
    print(f"Check outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
