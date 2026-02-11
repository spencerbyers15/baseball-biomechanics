# Handoff: GPU Pose Pipeline + Full Batch Testing

## Context

This is a baseball biomechanics project that extracts pitcher pose data from MLB broadcast video. The pipeline currently works end-to-end on CPU but is too slow for batch processing. The immediate goal is to accelerate the pose estimation step with GPU, then run it across all 1744 calibration videos.

## What's Complete

### Camera Cropping Pipeline (done)
- **Scene cut detection**: Histogram diff, threshold 0.08, 4x frame subsample. F1=0.988 on 118 hand-labeled videos.
- **EfficientNet-B0 camera classifier**: Binary (main_angle vs other), 97.7% test accuracy.
- **Cropping results**: 1744/1749 videos cropped to main pitching angle. Stored in `data/videos/pitcher_calibration_cropped/{Stadium}/{Season}/`.
- 5 remaining failures in `data/videos/pitcher_calibration_cropped/no_main_angle_round3/` (low priority — 4 are classifier edge cases, 1 is genuinely no main angle).

### Pitcher Zone Calibration (done)
- `tools/calibrate_pitcher_zones.py` processed all 1744 cropped videos.
- Output: `data/pitcher_zones.json` — per-stadium normalized bounding box zones (cx_norm, cy_norm means/stds, RHP/LHP offsets, per-season breakdown).
- 65,516 total position samples across 30 stadiums.
- Reports in `data/debug/pitcher_zones_report/` (4 PNG plots).

### Batch Pose Test (partial — only 30 videos)
- `tools/test_pitcher_pose.py --batch --per-stadium 1` tested on `data/videos/2023_cropped/` (30 videos, 1 per stadium).
- With calibrated zones: **94.8% mean detection, 93.6% mean pose**.
- Montages saved to `data/debug/pitcher_pose_test/{video_name}/montage.jpg`.
- **NOT yet run on the full 1744 calibration videos.**

---

## Current Priority: GPU Acceleration for Pose Pipeline

### The Problem
The pose pipeline processes frames at **38.4ms/frame** on CPU:
- YOLO person detection: **10.6ms** (27.6%)
- MediaPipe 33-landmark pose: **27.8ms** (72.3%)
- Pitcher finding + cropping: negligible

At 1744 videos x ~420 frames each, full batch would take **~10 hours** on CPU.

### Architecture (keep this — it works well)
```
YOLO person detect → find pitcher in zone → crop person bbox → pose estimation on crop
```
This approach:
- Uses calibrated per-stadium zones to find the pitcher among all detected people
- Crops the pitcher's bounding box before running pose (smaller input = faster + more accurate)
- Avoids labeling unwanted people (umpire, catcher, fans)

### Why Not Just YOLOv8-pose?
We tested YOLOv8-pose end-to-end on a full frame (GPU, 9.4ms/frame). User rejected it because:
- Skeleton colors switch between frames (no ID tracking)
- Labels unwanted people (umpire, catcher, fans in background)
- More pose errors than the current crop-then-estimate approach
- Only 17 COCO keypoints vs MediaPipe's 33

### What Needs GPU Acceleration
**MediaPipe is the bottleneck** (72.3% of time) and **cannot run on GPU** — the Python desktop SDK only supports CPU via TFLite XNNPACK. There is no GPU delegate for desktop Python.

### Recommended Approach
Replace MediaPipe with a GPU-capable pose model that runs on the **cropped pitcher image** (same architecture, just swap the pose backend). Options explored:

1. **YOLOv8-pose on crop** (easiest, no install needed)
   - Already have `ultralytics` installed with CUDA
   - Run on the cropped pitcher bbox only (single person, clean input)
   - 17 COCO keypoints (fewer than MediaPipe's 33 but may be sufficient)
   - Should be very fast since input is a small crop

2. **RTMPose via mmpose** (best quality, needs install)
   - State-of-the-art top-down pose estimator
   - Supports 17 COCO or 133 whole-body keypoints
   - Needs `pip install mmpose mmcv mmdet`
   - RTMPose-L runs at ~6ms on GPU

3. **ONNX Runtime GPU** (flexible)
   - Export any pose model to ONNX, run with `onnxruntime-gpu`
   - Needs `pip install onnxruntime-gpu`
   - More setup work but very flexible

### Current Environment
- **GPU**: RTX 2070 with CUDA
- **PyTorch**: 2.9.1+cu126 (working)
- **Ultralytics**: 8.3.246 (working, YOLO runs on GPU)
- **MediaPipe**: installed (CPU-only)
- **mmpose/onnxruntime**: NOT installed

---

## Secondary Priority: Update Batch Test Script

`tools/test_pitcher_pose.py` needs updating to run on the full 1744 calibration videos:

### Current Limitations
- Line ~232: `find_cropped_videos()` is hardcoded to search `data/videos/2023_cropped/`
- Directory structure assumes `{Stadium}/videos/` but calibration cropped uses `{Stadium}/{Season}/`
- Only tests 1 video per stadium (30 total)

### What to Change
- Add `--video-dir` flag to specify input directory (default to `pitcher_calibration_cropped/`)
- Update `find_cropped_videos()` to handle `{Stadium}/{Season}/` structure
- Add `--per-stadium N` support (already exists but needs to work with new structure)
- Consider `--max-frames` flag to cap frames per video for speed

---

## Key Files

### Pose Pipeline
| File | Role |
|------|------|
| `src/detection/player_pose.py` | YOLO detect + find pitcher + crop + MediaPipe pose |
| `tools/test_pitcher_pose.py` | Batch testing script with montage generation |
| `data/pitcher_zones.json` | Per-stadium calibrated pitcher zones |

### Camera Pipeline
| File | Role |
|------|------|
| `src/filtering/scene_cropper.py` | Cut detection + segment classification + ffmpeg crop |
| `src/filtering/camera_filter.py` | EfficientNet-B0 classifier (`CameraAngleClassifier`) |
| `models/camera_classifier/best.pt` | Trained model weights (97.7% accuracy) |
| `tools/calibrate_pitcher_zones.py` | Processes cropped videos → pitcher_zones.json |

### Labeling & Training
| File | Role |
|------|------|
| `tools/label_scene_cuts.py` | Interactive OpenCV labeler for cuts + segment classification |
| `tools/extract_segment_frames.py` | Extract frames from labeled segments for training |
| `tools/train_camera_classifier.py` | Train EfficientNet-B0 on extracted frames |
| `data/labels/scene_cuts/scene_cut_labels.json` | 118 hand-labeled videos with cuts + segment labels |

### Data Paths
| Path | Contents |
|------|----------|
| `data/videos/pitcher_calibration/` | 1749 raw videos (30 stadiums x 3 seasons) |
| `data/videos/pitcher_calibration_cropped/` | 1744 cropped main-angle videos |
| `data/videos/2023_cropped/` | 30 older cropped videos (1 per stadium, used by current batch test) |
| `data/pitcher_zones.json` | Per-stadium calibrated pitcher zones |
| `data/debug/pitcher_zones_report/` | Calibration visualization plots |
| `data/debug/pitcher_pose_test/` | Pose test montages and results |

---

## Suggested Next Steps

1. **Choose and implement GPU pose backend** — replace MediaPipe in `src/detection/player_pose.py` with a GPU model that runs on the cropped pitcher bbox. Keep the YOLO detect → find pitcher → crop → pose architecture.
2. **Move YOLO detection to GPU** — pass `device='cuda'` to the YOLO model (currently runs on CPU, 10.6ms → should drop to ~3ms on GPU).
3. **Update `test_pitcher_pose.py`** to support `pitcher_calibration_cropped/` directory structure.
4. **Run full batch test** on all 1744 videos with GPU acceleration.
5. **(Low priority)** Fix 4 remaining camera classifier failures by adding night game training examples.
