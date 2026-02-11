# Baseball Biomechanics — Full Pipeline Documentation

## Overview

End-to-end pipeline for extracting pitcher pose data from MLB broadcast video. Takes raw Statcast video URLs and produces per-frame 17-landmark pose skeletons of the pitcher.

```
Scrape video → Crop to main angle → Detect pitcher → Estimate pose
```

---

## Stage 1: Video Acquisition

**Script**: `tools/scrape_pitcher_calibration.py`
**Module**: `src/scraper/`

Scrapes Baseball Savant for pitch video URLs across 30 stadiums × 3 seasons (2023–2025). Downloads one video per stadium per season per game to get broadcast angle diversity.

**Output**: `data/videos/pitcher_calibration/` — 1749 raw videos
**Metadata**: `data/pitcher_calibration_metadata.json` — resume-safe, don't clear

**Key detail**: Statcast `no_pitch` events (pitch timer violations, IBB) have pitch numbers but no video — the scraper handles these gracefully.

---

## Stage 2: Camera Angle Cropping

**Module**: `src/filtering/scene_cropper.py` + `src/filtering/camera_filter.py`

MLB broadcasts interleave the main pitching angle with replays, close-ups, and dugout shots. This stage isolates the main pitching angle segment.

### 2a. Scene Cut Detection
- Histogram difference between consecutive frames
- Threshold: **0.08**, subsample every 4 frames
- F1 = 0.988 on 118 hand-labeled videos

### 2b. Segment Classification
- **EfficientNet-B0** binary classifier: `main_angle` vs `other`
- Test accuracy: **97.7%** across 30 stadiums
- Model: `models/camera_classifier/best.pt`
- Training: `tools/train_camera_classifier.py` on `data/labels/scene_cuts/frames/`

### 2c. Crop & Export
- `crop_to_main_angle()` finds the longest `main_angle` segment and exports via ffmpeg
- **Critical**: `crop_to_main_angle` has its own `cut_threshold` param — must match `detect_scene_cuts` (both 0.08)

**Output**: `data/videos/pitcher_calibration_cropped/{Stadium}/{Season}/` — 1744/1749 videos (99.7%)
**Failures**: 5 videos in `no_main_angle_round3/` (Coors Field night game is the key failure — no night training data)

### Training Pipeline (camera classifier)
```
tools/label_scene_cuts.py          → Hand-label scene cuts + segment types
tools/extract_segment_frames.py    → Extract frames from labeled segments
tools/train_camera_classifier.py   → Train EfficientNet-B0
```
**Labels**: `data/labels/scene_cuts/scene_cut_labels.json` — 118 hand-labeled videos

---

## Stage 3: Pitcher Zone Calibration

**Script**: `tools/calibrate_pitcher_zones.py`

Runs YOLO person detection on all 1744 cropped videos to establish where the pitcher typically appears in each stadium's broadcast framing. Produces per-stadium normalized bounding box statistics (center, std dev, bbox size).

**Output**: `data/pitcher_zones.json` — 30 stadiums, 65,516 position samples
**Reports**: `data/debug/pitcher_zones_report/` — 4 calibration plots

Used as a fallback heuristic when the pitcher classifier model is not available.

---

## Stage 4: Pitcher Identification

**Module**: `src/detection/player_pose.py` + `src/filtering/pitcher_classifier.py`

Given a cropped main-angle frame, identifies which detected person is the pitcher.

### Detection
- **YOLOv8n** person detection (GPU, conf ≥ 0.3)
- Produces candidate list with normalized coordinates and bounding boxes

### Selection (priority order)
1. **Pitcher classifier** (preferred) — EfficientNet-B0 binary classifier on each person crop
   - Crops every detected person, classifies all in one batch
   - Picks highest-confidence "pitcher" prediction
   - Model: `models/pitcher_classifier/best.pt`
2. **Calibrated zone heuristic** (fallback) — distance-based scoring using per-stadium zone center ± 2.5σ, with bbox size filtering
3. **Default zone heuristic** (last resort) — hard-coded center zone (cx 0.30–0.70, cy ≥ 0.50), picks lowest person

### Temporal Smoothing
- Tracks previous pitcher position across frames
- Strongly prefers candidate within 0.08 normalized distance of previous position
- Resets after 10 consecutive frames with no detection

### Pitcher Classifier Training Pipeline
```
tools/extract_pitcher_crops.py      → Sample frames, YOLO detect, save person crops
tools/label_pitcher_crops.py        → OpenCV UI: label each crop as pitcher/not_pitcher
tools/prepare_pitcher_training.py   → Organize into ImageFolder (split by video, not crop)
tools/train_pitcher_classifier.py   → Train EfficientNet-B0
```

**Data**: `data/labels/pitcher/`
- `crops/` — 6,040 person crops from 264 videos (32 stadiums × 3 seasons × 3 videos, 3 frames each)
- `pitcher_labels.json` — 759 pitcher, 5,281 not_pitcher (12.6% pitcher ratio)
- `crop_metadata.json` — bbox, position, video source for each crop
- `frames/train/` and `frames/test/` — ImageFolder structure, 80/20 split by video

**Sampling**: 3 videos/stadium/season, 3 frames/video (skip first/last 10%), all YOLO detections saved.

---

## Stage 5: Pose Estimation

**Module**: `src/pose/rtmpose_backend.py` (via `src/detection/player_pose.py`)

Runs on the cropped pitcher bounding box (not full frame) for accuracy and speed.

- **Model**: RTMPose-X (384×288 input, ONNX)
- **Backend**: rtmlib + onnxruntime-gpu
- **Keypoints**: 17 COCO landmarks (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **GPU**: Auto-adds cuDNN DLLs to PATH via `_ensure_cudnn_on_path()`

**Model file**: `models/rtmpose/end2end.onnx`

### Performance
- With calibrated zones: **96.1% mean detection, 96.1% mean pose, 100% median** (30 stadiums)
- Tested via `tools/test_pitcher_pose.py --batch --per-stadium 1`

### Environment Notes
- Install rtmlib with `--no-deps` to avoid pulling CPU onnxruntime / conflicting opencv
- onnxruntime-gpu needs cuDNN DLLs — rtmpose_backend.py handles this automatically

---

## Key Files Reference

### Core Modules
| File | Role |
|------|------|
| `src/scraper/` | Baseball Savant scraper + video downloader |
| `src/filtering/scene_cropper.py` | Scene cut detection + segment classification + ffmpeg crop |
| `src/filtering/camera_filter.py` | EfficientNet-B0 camera angle classifier |
| `src/filtering/pitcher_classifier.py` | EfficientNet-B0 pitcher/not_pitcher classifier |
| `src/detection/player_pose.py` | YOLO detect → find pitcher → crop → RTMPose pose |
| `src/pose/rtmpose_backend.py` | RTMPose-X GPU backend (17 COCO keypoints) |
| `src/pose/mediapipe_backend.py` | MediaPipe CPU fallback (legacy) |

### Tools
| File | Role |
|------|------|
| `tools/scrape_pitcher_calibration.py` | Download calibration videos from Savant |
| `tools/calibrate_pitcher_zones.py` | Build per-stadium pitcher zones |
| `tools/label_scene_cuts.py` | Label scene cuts + segment types |
| `tools/train_camera_classifier.py` | Train camera angle classifier |
| `tools/extract_pitcher_crops.py` | Extract person crops for pitcher labeling |
| `tools/label_pitcher_crops.py` | Label pitcher crops (OpenCV UI) |
| `tools/prepare_pitcher_training.py` | Organize pitcher labels into ImageFolder |
| `tools/train_pitcher_classifier.py` | Train pitcher classifier |
| `tools/test_pitcher_pose.py` | Batch pose test with montage generation |

### Models
| File | Description |
|------|-------------|
| `models/camera_classifier/best.pt` | Camera angle classifier (97.7% acc) |
| `models/pitcher_classifier/best.pt` | Pitcher identifier (100% test acc, epoch 3) |
| `models/rtmpose/end2end.onnx` | RTMPose-X body pose (384×288, ONNX) |

### Data
| Path | Contents |
|------|----------|
| `data/videos/pitcher_calibration/` | 1749 raw videos |
| `data/videos/pitcher_calibration_cropped/` | 1744 cropped main-angle videos |
| `data/pitcher_zones.json` | Per-stadium calibrated pitcher zones |
| `data/pitcher_calibration_metadata.json` | Video download metadata (resume-safe) |
| `data/labels/scene_cuts/` | Camera classifier training data + labels |
| `data/labels/pitcher/` | Pitcher classifier training data + labels |

---

## Environment

- **OS**: Windows 10
- **GPU**: RTX 2070 with CUDA
- **Python**: Anaconda env `baseball` (`C:/Users/Spencer/anaconda3/envs/baseball/python.exe`)
- **PyTorch**: 2.9.1+cu126
- **Key packages**: ultralytics, rtmlib (--no-deps), onnxruntime-gpu, torchvision, opencv-python
