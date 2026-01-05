"""
Train YOLOv8-small on the diverse stadium dataset.
Converts annotations and trains the model.
"""

import json
import os
import shutil
from pathlib import Path

import cv2


def convert_to_yolo_format(frames_dir: str, output_dir: str):
    """Convert diverse frame annotations to YOLO format."""
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    # Create YOLO directory structure
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Load data
    with open(frames_dir / "frames_info.json") as f:
        frames_info = json.load(f)
    with open(frames_dir / "annotations.json") as f:
        annotations = json.load(f)

    # Build labeled frames list
    labeled_frames = []
    for i, frame_info in enumerate(frames_info):
        frame_id = str(frame_info.get("id", i))
        if frame_id in annotations and annotations[frame_id]:
            labeled_frames.append((frame_info, frame_id, annotations[frame_id]))

    # Split 80/20 train/val
    split_idx = int(len(labeled_frames) * 0.8)
    train_frames = labeled_frames[:split_idx]
    val_frames = labeled_frames[split_idx:]

    print(f"Converting {len(labeled_frames)} labeled frames to YOLO format")
    print(f"  Train: {len(train_frames)}, Val: {len(val_frames)}")

    def process_frames(frames, split):
        for idx, (frame_info, frame_id, boxes) in enumerate(frames):
            img_path = Path(frame_info["path"])
            if not img_path.exists():
                print(f"  Warning: Image not found: {img_path}")
                continue

            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Copy image
            new_img_name = f"{split}_{idx:04d}.jpg"
            new_img_path = output_dir / "images" / split / new_img_name
            cv2.imwrite(str(new_img_path), img)

            # Create label file
            label_path = output_dir / "labels" / split / f"{split}_{idx:04d}.txt"
            with open(label_path, "w") as f:
                for box in boxes:
                    # Class: 0 = catcher_mitt, 1 = pitcher_glove
                    class_id = 0 if box.get("is_positive", True) else 1

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
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val

names:
  0: catcher_mitt
  1: pitcher_glove
"""

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"YOLO dataset created at: {output_dir}")
    return str(yaml_path)


def train_yolo(yaml_path: str, epochs: int = 100, model_size: str = "s"):
    """Train YOLOv8 on the diverse dataset."""
    from ultralytics import YOLO

    model_name = f"yolov8{model_size}.pt"
    print(f"\nLoading {model_name}...")
    model = YOLO(model_name)

    print(f"Training for {epochs} epochs...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=8,  # Good for 8GB VRAM
        device=0,
        project="F:/Claude_Projects/baseball-biomechanics/models",
        name="yolo_mitt_diverse",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )

    return results


def test_model(model_path: str, test_dir: str, output_dir: str):
    """Test trained model and save results."""
    from ultralytics import YOLO

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    # Get test images
    test_images = list(Path(test_dir).glob("*.jpg"))

    print(f"\nTesting on {len(test_images)} images...")
    print("=" * 60)

    results_summary = []

    for img_path in test_images[:20]:  # Test first 20
        results = model(str(img_path), verbose=False)

        img = cv2.imread(str(img_path))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                class_name = "catcher_mitt" if cls == 0 else "pitcher_glove"

                # Draw box
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label
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

    parser = argparse.ArgumentParser(description="Train YOLO on diverse dataset")
    parser.add_argument("--project-dir", default="F:/Claude_Projects/baseball-biomechanics")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="s", choices=["n", "s", "m"])
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    frames_dir = project_dir / "data" / "labels" / "diverse_frames"
    yolo_dir = project_dir / "data" / "yolo_diverse"

    if args.test_only:
        model_path = project_dir / "models" / "yolo_mitt_diverse" / "weights" / "best.pt"
        test_dir = yolo_dir / "images" / "val"
        output_dir = project_dir / "data" / "debug" / "yolo_diverse_results"
        test_model(str(model_path), str(test_dir), str(output_dir))
        return

    # Convert data
    yaml_path = convert_to_yolo_format(str(frames_dir), str(yolo_dir))

    if args.convert_only:
        return

    # Train
    train_yolo(yaml_path, epochs=args.epochs, model_size=args.model)

    # Test
    model_path = project_dir / "models" / "yolo_mitt_diverse" / "weights" / "best.pt"
    test_dir = yolo_dir / "images" / "val"
    output_dir = project_dir / "data" / "debug" / "yolo_diverse_results"
    test_model(str(model_path), str(test_dir), str(output_dir))


if __name__ == "__main__":
    main()
