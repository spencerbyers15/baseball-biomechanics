"""
Train YOLOv8 for catcher's mitt and pitcher's glove detection.
Converts labeled data to YOLO format and trains a fast detector.
"""

import json
import os
import shutil
from pathlib import Path

import cv2


def convert_to_yolo_format(project_dir: str):
    """Convert labeled data to YOLO format."""
    project_dir = Path(project_dir)
    labels_dir = project_dir / "data" / "labels" / "mitt_finetune"
    yolo_dir = project_dir / "data" / "yolo_mitt"

    # Create YOLO directory structure
    (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(labels_dir / "annotations.json") as f:
        annotations = json.load(f)
    with open(labels_dir / "frames_info.json") as f:
        frames_info = json.load(f)

    # Class mapping: 0 = catcher_mitt, 1 = pitcher_glove
    classes = ["catcher_mitt", "pitcher_glove"]

    # Split data (80% train, 20% val)
    labeled_frames = []
    for frame_info in frames_info:
        frame_id = str(frame_info["id"])
        if frame_id in annotations and annotations[frame_id]:
            labeled_frames.append((frame_info, annotations[frame_id]))

    split_idx = int(len(labeled_frames) * 0.8)
    train_frames = labeled_frames[:split_idx]
    val_frames = labeled_frames[split_idx:]

    print(f"Converting {len(labeled_frames)} frames to YOLO format")
    print(f"  Train: {len(train_frames)}, Val: {len(val_frames)}")

    def process_frames(frames, split):
        for frame_info, boxes in frames:
            img_path = Path(frame_info["path"])
            if not img_path.exists():
                continue

            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Copy image
            new_img_name = f"{frame_info['id']:04d}.jpg"
            new_img_path = yolo_dir / "images" / split / new_img_name
            cv2.imwrite(str(new_img_path), img)

            # Create label file
            label_path = yolo_dir / "labels" / split / f"{frame_info['id']:04d}.txt"
            with open(label_path, "w") as f:
                for box in boxes:
                    # Determine class
                    if box.get("is_positive", True):
                        class_id = 0  # catcher_mitt
                    else:
                        class_id = 1  # pitcher_glove

                    # Convert to YOLO format (normalized xywh)
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    box_w = (x2 - x1) / w
                    box_h = (y2 - y1) / h

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    process_frames(train_frames, "train")
    process_frames(val_frames, "val")

    # Create dataset YAML
    yaml_content = f"""
path: {yolo_dir}
train: images/train
val: images/val

names:
  0: catcher_mitt
  1: pitcher_glove
"""

    yaml_path = yolo_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"YOLO dataset created at: {yolo_dir}")
    print(f"Dataset config: {yaml_path}")

    return str(yaml_path)


def train_yolo(yaml_path: str, epochs: int = 100, model_size: str = "n"):
    """Train YOLOv8 on the mitt dataset."""
    from ultralytics import YOLO

    # Use YOLOv8 nano (fastest, good for 8GB VRAM)
    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)

    # Train
    print(f"Training for {epochs} epochs...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,  # Small batch for 8GB VRAM
        device=0,
        project="F:/Claude_Projects/baseball-biomechanics/models",
        name="yolo_mitt",
        exist_ok=True,
        patience=20,  # Early stopping
        save=True,
        plots=True,
    )

    return results


def test_model(model_path: str, test_images_dir: str, output_dir: str):
    """Test trained model and save results."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))

    print(f"\nTesting on {len(test_images)} images...")
    print("=" * 60)

    results_summary = []

    for img_path in test_images[:10]:  # Test first 10
        results = model(str(img_path), verbose=False)

        # Get detections
        img = cv2.imread(str(img_path))

        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                class_name = "catcher_mitt" if cls == 0 else "pitcher_glove"

                # Draw box
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Centroid
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 5, color, -1)

                results_summary.append({
                    "image": img_path.name,
                    "class": class_name,
                    "confidence": conf,
                    "centroid": (cx, cy),
                })

                print(f"{img_path.name}: {class_name} conf={conf:.3f} centroid=({cx}, {cy})")

        # Save annotated image
        out_path = output_dir / f"det_{img_path.name}"
        cv2.imwrite(str(out_path), img)

    print("=" * 60)

    # Summary
    mitt_dets = [r for r in results_summary if r["class"] == "catcher_mitt"]
    glove_dets = [r for r in results_summary if r["class"] == "pitcher_glove"]

    print(f"\nSUMMARY:")
    print(f"  Catcher mitt detections: {len(mitt_dets)}")
    if mitt_dets:
        avg_conf = sum(r["confidence"] for r in mitt_dets) / len(mitt_dets)
        print(f"    Avg confidence: {avg_conf:.3f}")

    print(f"  Pitcher glove detections: {len(glove_dets)}")
    if glove_dets:
        avg_conf = sum(r["confidence"] for r in glove_dets) / len(glove_dets)
        print(f"    Avg confidence: {avg_conf:.3f}")

    print(f"\nResults saved to: {output_dir}")

    return results_summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO for mitt detection")
    parser.add_argument("--project-dir", default="F:/Claude_Projects/baseball-biomechanics")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="n", choices=["n", "s", "m"],
                       help="Model size: n=nano, s=small, m=medium")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)

    if args.test_only:
        model_path = project_dir / "models" / "yolo_mitt" / "weights" / "best.pt"
        test_dir = project_dir / "data" / "yolo_mitt" / "images" / "val"
        output_dir = project_dir / "data" / "debug" / "yolo_results"
        test_model(str(model_path), str(test_dir), str(output_dir))
        return

    # Convert data
    yaml_path = convert_to_yolo_format(args.project_dir)

    if args.convert_only:
        return

    # Train
    train_yolo(yaml_path, epochs=args.epochs, model_size=args.model)

    # Test
    model_path = project_dir / "models" / "yolo_mitt" / "weights" / "best.pt"
    test_dir = project_dir / "data" / "yolo_mitt" / "images" / "val"
    output_dir = project_dir / "data" / "debug" / "yolo_results"
    test_model(str(model_path), str(test_dir), str(output_dir))


if __name__ == "__main__":
    main()
