"""
SAM3 Fine-tuning for Catcher's Mitt Detection.

Fine-tunes only the mask decoder (lightweight) on labeled bounding box data.

Based on:
- HuggingFace SAM fine-tuning tutorial
- Roboflow SAM3 fine-tuning guide

Requirements:
- ~15-20 labeled frames with bounding boxes
- GPU with 8GB+ VRAM
- Estimated training time: 10-30 minutes on RTX 3080

Data format (from mitt_labeler.py):
{
    "image_path": "path/to/frame.png",
    "image_size": [width, height],
    "positive_boxes": [[x1, y1, x2, y2], ...],  # Catcher's mitt boxes
    "negative_boxes": [[x1, y1, x2, y2], ...]   # Pitcher's glove boxes
}
"""

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class MittDataset(Dataset):
    """Dataset for catcher's mitt fine-tuning with positive and negative examples."""

    TEXT_PROMPT = "catcher's mitt"

    def __init__(self, data_path: str, processor):
        with open(data_path) as f:
            self.samples = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        w, h = image.size

        # Get positive box (catcher's mitt)
        if not sample["positive_boxes"]:
            raise ValueError(f"No positive boxes in sample {idx}")

        pos_box = sample["positive_boxes"][0]  # Catcher's mitt

        # Get negative box (pitcher's glove) if available
        neg_boxes = sample.get("negative_boxes", [])

        # Create ground truth mask - positive region only
        mask = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = [int(c) for c in pos_box]
        mask[y1:y2, x1:x2] = 1.0

        # Build input boxes with labels
        # Positive box (catcher's mitt) = 1
        # Negative boxes (pitcher's glove) = 0
        all_boxes = [pos_box]
        all_labels = [1]

        for neg_box in neg_boxes:
            all_boxes.append(neg_box)
            all_labels.append(0)  # Negative = exclude this

        # Process inputs with text prompt and boxes
        inputs = self.processor(
            images=image,
            text=self.TEXT_PROMPT,
            input_boxes=[all_boxes],
            input_boxes_labels=[all_labels],
            return_tensors="pt",
        )

        # Remove batch dimension
        inputs = {k: v.squeeze(0) if hasattr(v, 'squeeze') else v for k, v in inputs.items()}

        # Add ground truth mask and box info
        inputs["ground_truth_mask"] = torch.from_numpy(mask)
        inputs["positive_box"] = torch.tensor(pos_box, dtype=torch.float32)

        return inputs


def collate_fn(batch):
    """Custom collate function for variable-size inputs."""
    # Stack tensors that can be stacked
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values)
            except:
                result[key] = values
        else:
            result[key] = values
    return result


class SAM3FineTuner:
    """Fine-tune SAM3 mask decoder for catcher's mitt detection."""

    def __init__(
        self,
        cache_dir: str = "F:/hf_cache",
        output_dir: str = "F:/Claude_Projects/baseball-biomechanics/models/sam3_mitt",
        low_memory: bool = True,
    ):
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.low_memory = low_memory

    def load_model(self):
        """Load SAM3 model for fine-tuning."""
        from transformers import Sam3Model, Sam3Processor

        print("Loading SAM3 for fine-tuning...")
        self.processor = Sam3Processor.from_pretrained(
            "facebook/sam3", cache_dir=self.cache_dir
        )

        # Use bfloat16 for low memory mode (better than fp16 for training)
        if self.low_memory:
            # Check if bfloat16 is supported
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
            else:
                dtype = torch.float32  # Fall back to fp32, use autocast
                self.use_amp = True
                self.amp_dtype = torch.float16
        else:
            dtype = torch.float32
            self.use_amp = False
            self.amp_dtype = torch.float32

        print(f"Using dtype: {dtype}, AMP: {self.use_amp}")

        self.model = Sam3Model.from_pretrained(
            "facebook/sam3",
            cache_dir=self.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # Note: SAM3 doesn't support gradient checkpointing
        # Memory savings come from fp16 and small batch size

        self.model.to(self.device)

        # Freeze only the heavy vision encoder - train DETR decoder + mask decoder
        # This allows the model to learn detection (not just masks)
        trainable_components = [
            "mask_decoder",      # Mask prediction
            "detr_decoder",      # Object detection/scoring
            "geometry_encoder",  # Box encoding
        ]

        for name, param in self.model.named_parameters():
            param.requires_grad = any(comp in name for comp in trainable_components)

        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # List what we're training
        trained_modules = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                module = name.split('.')[0]
                trained_modules.add(module)
        print(f"Training modules: {', '.join(sorted(trained_modules))}")

        return self.model, self.processor

    def train(
        self,
        data_path: str,
        epochs: int = 10,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        gradient_accumulation_steps: int = 4,
    ):
        """
        Fine-tune the mask decoder.

        Args:
            data_path: Path to sam3_training_data.json
            epochs: Number of training epochs
            batch_size: Batch size (keep small for memory)
            learning_rate: Learning rate
            gradient_accumulation_steps: Accumulate gradients over N steps
        """
        model, processor = self.load_model()

        # Create dataset
        dataset = MittDataset(data_path, processor)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        print(f"Training on {len(dataset)} samples")
        print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

        # Optimizer - only for trainable params
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Mixed precision - no scaler needed for bfloat16
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler() if use_scaler else None

        # Training loop
        model.train()
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for step, batch in enumerate(progress):
                # Move to device
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                input_boxes = batch["input_boxes"]
                input_boxes_labels = batch["input_boxes_labels"]
                gt_masks = batch["ground_truth_mask"].to(self.device)

                # Handle variable batch processing
                if isinstance(input_boxes, list):
                    input_boxes = torch.stack(input_boxes).to(self.device)
                else:
                    input_boxes = input_boxes.to(self.device)

                if isinstance(input_boxes_labels, list):
                    input_boxes_labels = torch.stack(input_boxes_labels).to(self.device)
                else:
                    input_boxes_labels = input_boxes_labels.to(self.device)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )

                    # Get predicted masks
                    pred_masks = outputs.pred_masks  # [B, num_queries, H, W]

                    # Compute loss with spatial penalty
                    B = pred_masks.shape[0]
                    loss = 0

                    # Get predicted boxes for spatial loss
                    pred_boxes = outputs.pred_boxes  # [B, num_queries, 4]

                    for i in range(B):
                        pred = pred_masks[i]  # [num_queries, H, W]
                        gt = gt_masks[i].float()  # [H, W]
                        h, w = gt.shape

                        # Resize predicted mask to GT size if needed
                        if pred.shape[-2:] != gt.shape:
                            pred = torch.nn.functional.interpolate(
                                pred.unsqueeze(0).float(),
                                size=gt.shape,
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0)

                        # Take mean over queries
                        pred_mean = pred.mean(dim=0)

                        # Binary cross-entropy loss for mask
                        mask_loss = criterion(pred_mean, gt)

                        # Spatial penalty: penalize masks in left half (pitcher area)
                        # Catcher's mitt should be in right half of frame
                        left_half_mask = torch.zeros_like(gt)
                        left_half_mask[:, :w//2] = 1.0  # Left half

                        # Penalize predictions in left half
                        pred_sigmoid = torch.sigmoid(pred_mean)
                        spatial_penalty = (pred_sigmoid * left_half_mask).mean()

                        # Combined loss
                        sample_loss = mask_loss + 0.5 * spatial_penalty
                        loss += sample_loss

                    loss = loss / B / gradient_accumulation_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Update weights every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    # Clear cache periodically
                    if self.low_memory:
                        torch.cuda.empty_cache()

                epoch_loss += loss.item() * gradient_accumulation_steps
                progress.set_postfix({
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "mem": f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                })

            # Final step update if needed
            if len(dataloader) % gradient_accumulation_steps != 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(model, processor)
                print(f"  Saved best model (loss: {best_loss:.4f})")

            # Clear cache after each epoch
            if self.low_memory:
                torch.cuda.empty_cache()

        print(f"\nTraining complete! Best loss: {best_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")

    def save_model(self, model, processor):
        """Save fine-tuned model."""
        model.save_pretrained(self.output_dir)
        processor.save_pretrained(self.output_dir)

    def load_finetuned(self):
        """Load fine-tuned model for inference."""
        from transformers import Sam3Model, Sam3Processor

        processor = Sam3Processor.from_pretrained(self.output_dir)
        model = Sam3Model.from_pretrained(self.output_dir, torch_dtype=torch.bfloat16)
        model.to(self.device)
        model.eval()

        return model, processor


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Fine-tuning for Catcher's Mitt")
    parser.add_argument(
        "--data-path",
        default="F:/Claude_Projects/baseball-biomechanics/data/labels/mitt_finetune/sam3_training_data.json"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cache-dir", default="F:/hf_cache")
    parser.add_argument("--low-memory", action="store_true", default=True,
                        help="Enable low memory mode (fp16, gradient checkpointing)")
    parser.add_argument("--no-low-memory", action="store_false", dest="low_memory")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        print("\nFirst, label some frames using the labeler:")
        print("  python tools/mitt_labeler.py --extract --num-frames 15")
        print("  # Then run the GUI and draw boxes around catcher's mitt")
        print("  # Save annotations when done")
        return

    tuner = SAM3FineTuner(cache_dir=args.cache_dir, low_memory=args.low_memory)
    tuner.train(
        data_path=str(data_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
    )


if __name__ == "__main__":
    main()
