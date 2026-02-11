#!/usr/bin/env python
"""Build reference embeddings from SAMPLED frames (stratified by stadium)."""

import numpy as np
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
sys.path.insert(0, str(PROJECT_ROOT))

FRAMES_DIR = PROJECT_ROOT / "data/all_frames_by_stadium"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/reference_embeddings_sampled.pkl"

# Target total frames (will sample proportionally from each stadium)
TARGET_FRAMES = 2500


def get_stratified_frames():
    """Sample frames proportionally from each stadium."""
    stadiums = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(stadiums)} stadiums")

    # Count frames per stadium
    stadium_frames = {}
    total_frames = 0
    for stadium in stadiums:
        frames = sorted(stadium.glob("*.jpg"))
        stadium_frames[stadium.name] = frames
        total_frames += len(frames)

    print(f"Total frames: {total_frames}")

    # Sample proportionally from each stadium
    sampled = []
    for stadium_name, frames in stadium_frames.items():
        # Proportional allocation, minimum 50 per stadium
        proportion = len(frames) / total_frames
        n_sample = max(50, int(TARGET_FRAMES * proportion))

        # Sample evenly spaced frames
        if len(frames) <= n_sample:
            sampled.extend(frames)
        else:
            step = len(frames) / n_sample
            indices = [int(i * step) for i in range(n_sample)]
            sampled.extend([frames[i] for i in indices])

    print(f"Sampled {len(sampled)} frames across {len(stadiums)} stadiums")
    return sampled


def build_embeddings(frame_paths: list):
    """Build embeddings for sampled frames."""
    from transformers import CLIPProcessor, CLIPModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    embeddings = []
    valid_paths = []

    batch_size = 32
    for i in tqdm(range(0, len(frame_paths), batch_size), desc="Embedding"):
        batch_paths = frame_paths[i:i + batch_size]

        # Load images, skip failures
        images = []
        batch_valid_paths = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                batch_valid_paths.append(p)
            except Exception as e:
                print(f"Failed to load {p}: {e}")
                continue

        if not images:
            continue

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model.get_image_features(**inputs)
            emb = emb.cpu().numpy()

        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        embeddings.append(emb)
        valid_paths.extend([str(p) for p in batch_valid_paths])

    embeddings = np.vstack(embeddings)
    print(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")

    # Save embeddings
    data = {
        "embeddings": embeddings,
        "frame_paths": valid_paths,
        "method": "stratified_by_stadium",
        "model_type": "clip",
        "n_images": len(embeddings),
    }
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")

    return embeddings, valid_paths


def main():
    print("=== Building STRATIFIED reference embeddings ===")
    frames = get_stratified_frames()
    build_embeddings(frames)
    print("\nDone!")


if __name__ == "__main__":
    main()
