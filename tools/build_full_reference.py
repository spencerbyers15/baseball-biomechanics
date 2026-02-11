#!/usr/bin/env python
"""Extract ALL frames from diverse_stadiums videos and build comprehensive reference embeddings."""

import cv2
import numpy as np
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
sys.path.insert(0, str(PROJECT_ROOT))

VIDEOS_DIR = PROJECT_ROOT / "data/videos/diverse_stadiums"
OUTPUT_DIR = PROJECT_ROOT / "data/all_frames_by_stadium"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/reference_embeddings_full.pkl"


def extract_all_frames():
    """Extract every frame from every video, organized by stadium. Skips already-processed videos."""
    stadiums = [d for d in VIDEOS_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(stadiums)} stadiums")

    total_frames = 0
    skipped_videos = 0

    for stadium_dir in tqdm(stadiums, desc="Stadiums"):
        stadium_output = OUTPUT_DIR / stadium_dir.name
        stadium_output.mkdir(parents=True, exist_ok=True)

        videos = list(stadium_dir.glob("*.mp4"))
        for video_path in videos:
            # Check if this video was already processed (look for at least one frame)
            existing_frames = list(stadium_output.glob(f"{video_path.stem}_f*.jpg"))
            if existing_frames:
                total_frames += len(existing_frames)
                skipped_videos += 1
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                filename = f"{video_path.stem}_f{frame_num:04d}.jpg"
                output_path = stadium_output / filename
                cv2.imwrite(str(output_path), frame)
                frame_num += 1
                total_frames += 1

            cap.release()

    print(f"\nTotal frames: {total_frames} (skipped {skipped_videos} already-processed videos)")
    return total_frames


def build_embeddings():
    """Build embeddings for all extracted frames."""
    from transformers import CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    # Collect all frame paths
    all_frames = list(OUTPUT_DIR.rglob("*.jpg"))
    print(f"Found {len(all_frames)} frames to embed")

    embeddings = []
    frame_paths = []

    batch_size = 32
    for i in tqdm(range(0, len(all_frames), batch_size), desc="Embedding"):
        batch_paths = all_frames[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model.get_image_features(**inputs)
            emb = emb.cpu().numpy()

        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings.append(emb)
        frame_paths.extend([str(p) for p in batch_paths])

    embeddings = np.vstack(embeddings)
    print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")

    # Save embeddings
    data = {
        "embeddings": embeddings,
        "frame_paths": frame_paths,
        "method": "all",
        "model_type": "clip",
        "n_images": len(embeddings),
    }
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    return embeddings, frame_paths


def main():
    print("=== Step 1: Extract all frames ===")
    extract_all_frames()

    print("\n=== Step 2: Build embeddings ===")
    build_embeddings()

    print("\nDone! Reference embeddings saved to:", EMBEDDINGS_PATH)


if __name__ == "__main__":
    main()
