"""Test SAM 3 text-prompted segmentation on a baseball frame."""

import os
import sys
sys.path.insert(0, "F:/Claude_Projects/sam3")

import torch
import cv2
import numpy as np
from PIL import Image

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import SAM 3
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
print(f"SAM 3 root: {sam3_root}")

# Build model
print("\nLoading SAM 3 model...")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

with torch.autocast("cuda", dtype=torch.bfloat16):
    model = build_sam3_image_model(bpe_path=bpe_path)
    print("Model loaded!")

    # Load a baseball frame
    video_path = "F:/Claude_Projects/baseball-biomechanics/data/videos/2024/04/746163_35_1.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # Get frame 100
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        sys.exit(1)

    # Convert to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    print(f"Image size: {image.size}")

    # Create processor
    processor = Sam3Processor(model, confidence_threshold=0.3)
    inference_state = processor.set_image(image)

    # Test with different text prompts
    prompts = ["person", "baseball batter", "catcher", "baseball glove", "mitt"]

    for prompt in prompts:
        print(f"\n--- Testing prompt: '{prompt}' ---")
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

        # Get results
        if hasattr(inference_state, 'masks') and inference_state.masks is not None:
            num_masks = len(inference_state.masks)
            print(f"  Found {num_masks} masks")
        elif hasattr(inference_state, 'pred_masks') and inference_state.pred_masks is not None:
            print(f"  Found masks in pred_masks")
        else:
            print(f"  Checking state attributes: {[a for a in dir(inference_state) if not a.startswith('_')]}")

print("\nSAM 3 test complete!")
