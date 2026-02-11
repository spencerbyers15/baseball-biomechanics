#!/usr/bin/env python
"""Test camera angle classifier on sample frames."""

import shutil
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
sys.path.insert(0, str(PROJECT_ROOT))

from src.filtering import CameraAngleFilter
from PIL import Image
from tqdm import tqdm

FRAMES_DIR = PROJECT_ROOT / "data/bat_labeling/frames"
OUTPUT_DIR = Path("F:/camera_classifier_test")
REFERENCE_PATH = PROJECT_ROOT / "data/reference_embeddings.pkl"

N_SAMPLES = 200
THRESHOLD = 0.85


def main():
    # Setup output dirs
    main_dir = OUTPUT_DIR / "main_angle"
    other_dir = OUTPUT_DIR / "other_angle"
    main_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    # Get sample frames
    all_frames = list(FRAMES_DIR.glob("*.jpg"))
    sample_frames = random.sample(all_frames, min(N_SAMPLES, len(all_frames)))
    print(f"Testing {len(sample_frames)} frames")

    # Load classifier
    classifier = CameraAngleFilter(model_type="clip")
    classifier.load_reference(str(REFERENCE_PATH))
    classifier.initialize()

    main_count = 0
    other_count = 0

    for frame_path in tqdm(sample_frames, desc="Classifying"):
        image = Image.open(frame_path).convert("RGB")
        emb = classifier._get_embedding(image)
        similarity = classifier._cosine_similarity(emb)

        # Classify and copy to appropriate folder
        if similarity >= THRESHOLD:
            dest = main_dir / f"{similarity:.3f}_{frame_path.name}"
            main_count += 1
        else:
            dest = other_dir / f"{similarity:.3f}_{frame_path.name}"
            other_count += 1

        shutil.copy(frame_path, dest)

    print(f"\nResults:")
    print(f"  Main angle: {main_count} ({main_count/len(sample_frames)*100:.1f}%)")
    print(f"  Other angle: {other_count} ({other_count/len(sample_frames)*100:.1f}%)")
    print(f"\nCheck results in: {OUTPUT_DIR}")
    print(f"  - main_angle/: frames classified as main broadcast view")
    print(f"  - other_angle/: frames classified as alternate camera angles")
    print(f"\nFilenames prefixed with similarity score for easy sorting")


if __name__ == "__main__":
    main()
