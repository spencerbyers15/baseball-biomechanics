#!/usr/bin/env python
"""
Train EfficientNet-B0 binary classifier for camera angle detection.

Fine-tunes a pretrained EfficientNet-B0 to classify video frames as
"main_angle" (standard pitching broadcast view) or "other" (replays,
close-ups, dugout shots, etc.).

Expects ImageFolder structure from extract_segment_frames.py:
    data/labels/scene_cuts/frames/
        train/main_angle/  train/other/
        test/main_angle/   test/other/

Usage:
    python tools/train_camera_classifier.py
    python tools/train_camera_classifier.py --epochs 15 --lr 1e-4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
DATA_DIR = PROJECT_ROOT / "data/labels/scene_cuts/frames"
MODEL_DIR = PROJECT_ROOT / "models/camera_classifier"


def get_transforms(train: bool):
    """Get data transforms for train or val."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    class_correct = {}
    class_total = {}
    for p, l in zip(all_preds, all_labels):
        class_total[l] = class_total.get(l, 0) + 1
        if p == l:
            class_correct[l] = class_correct.get(l, 0) + 1

    return running_loss / total, correct / total, class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description="Train camera angle classifier")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze all layers except classifier (faster, less accurate)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    train_dataset = datasets.ImageFolder(args.data / "train", transform=get_transforms(True))
    test_dataset = datasets.ImageFolder(args.data / "test", transform=get_transforms(False))

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"  class_to_idx: {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)} images")
    print(f"Test:  {len(test_dataset)} images")

    # Class distribution
    train_labels = [s[1] for s in train_dataset.samples]
    test_labels = [s[1] for s in test_dataset.samples]
    for ci, cn in enumerate(class_names):
        n_train = sum(1 for l in train_labels if l == ci)
        n_test = sum(1 for l in test_labels if l == ci)
        print(f"  {cn}: {n_train} train, {n_test} test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    # Build model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("Backbone frozen — only training classifier head")

    # Replace classifier for binary
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, len(class_names)),
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    args.model_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"TRAINING: EfficientNet-B0 | {args.epochs} epochs | lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, class_correct, class_total = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Per-class accuracy string
        per_class = []
        for ci, cn in enumerate(class_names):
            cc = class_correct.get(ci, 0)
            ct = class_total.get(ci, 0)
            acc = cc / ct if ct > 0 else 0
            per_class.append(f"{cn}={acc:.3f}")

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Test: loss={test_loss:.4f} acc={test_acc:.4f} | "
              f"{' '.join(per_class)} | "
              f"lr={lr:.6f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": lr,
        })

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": train_dataset.class_to_idx,
                "test_acc": test_acc,
                "epoch": epoch + 1,
            }, args.model_dir / "best.pt")
            print(f"  -> New best model saved (acc={test_acc:.4f})")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": train_dataset.class_to_idx,
        "test_acc": test_acc,
        "epoch": args.epochs,
    }, args.model_dir / "final.pt")

    # Save training history
    with open(args.model_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                    "best_test_acc": best_acc, "class_names": class_names,
                    "class_to_idx": train_dataset.class_to_idx},
                  f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best test accuracy: {best_acc:.4f}")
    print(f"  Model saved to: {args.model_dir}")
    print(f"  Files: best.pt, final.pt, history.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
