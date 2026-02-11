#!/usr/bin/env python
"""Embed only the labeled frames for visualization."""

import json
import numpy as np
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
LABELS_PATH = PROJECT_ROOT / "data/camera_angle_labels.json"
STRATIFIED_DIR = PROJECT_ROOT / "data/stratified_label_frames"
OUTPUT_PATH = PROJECT_ROOT / "data/labeled_frames_embeddings.pkl"


def main():
    # Load labels
    with open(LABELS_PATH) as f:
        data = json.load(f)
    labels = data["labels"]
    print(f"Found {len(labels)} labeled frames")

    # Load CLIP
    from transformers import CLIPProcessor, CLIPModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()

    # Embed each labeled frame
    embeddings = []
    paths = []
    frame_labels = []

    for path_str, label in tqdm(labels.items(), desc="Embedding"):
        path = Path(path_str)
        if not path.exists():
            print(f"Missing: {path}")
            continue

        try:
            img = Image.open(path).convert("RGB")
            with torch.no_grad():
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                emb = model.get_image_features(**inputs)
                emb = emb.cpu().numpy()
                emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.squeeze())
            paths.append(path_str)
            frame_labels.append(label)
        except Exception as e:
            print(f"Error on {path}: {e}")

    embeddings = np.array(embeddings)
    print(f"Generated {len(embeddings)} embeddings")

    # Save
    output = {
        "embeddings": embeddings,
        "frame_paths": paths,
        "labels": frame_labels,
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
