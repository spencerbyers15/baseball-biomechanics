# Baseball Biomechanics — Pipeline Roadmap

Full end-to-end pipeline for extracting biomechanics data from MLB broadcast video. Takes raw Statcast video URLs and produces per-frame pose, ball trajectory, bat barrel position, and pitch location data for every pitch.

## Pipeline Overview

```
1. Scrape video from Baseball Savant                                     [DONE]
2. Temporal crop: scene cuts + camera classifier → main pitching angle   [DONE]
3. From cropped main angle:
   a. YOLO person detect → 4-class classifier → dynamic player crops
      - Pitcher crop                                                     [DONE]
      - Catcher crop                                                     [DONE]
      - Batter crop                                                      [DONE]
   b. Ball detection on full main angle view                             [PARTIAL]
   c. Home plate detection on full main angle view                       [DONE]
   d. Bat barrel detection on batter crop                                [PARTIAL]
   e. Catcher mitt detection on catcher crop                             [PARTIAL]
4. RTMPose-X pose estimation on each player crop                         [DONE]
5. Combine all detections into per-frame analysis                        [PARTIAL]
```

---

## Stage 1: Video Acquisition — [DONE]

**Module**: `src/scraper/savant.py`, `src/scraper/video_downloader.py`
**Script**: `tools/scrape_pitcher_calibration.py`

Scrapes Baseball Savant for pitch video URLs across 30 stadiums x 3 seasons (2023-2025). Downloads one video per stadium per season per game for broadcast angle diversity.

- **Output**: `data/videos/pitcher_calibration/` — 1749 raw videos
- **Metadata**: `data/pitcher_calibration_metadata.json` — resume-safe, don't clear
- **Gotcha**: Statcast `no_pitch` events (pitch timer violations, IBB) have pitch numbers but no video — the scraper handles these gracefully

---

## Stage 2: Camera Angle Cropping — [DONE]

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

### 2c. Crop & Export
- `crop_to_main_angle()` finds the longest `main_angle` segment and exports via ffmpeg
- **Critical**: `crop_to_main_angle` has its own `cut_threshold` param — must match `detect_scene_cuts` (both 0.08)

**Results**: 1744/1749 videos cropped (99.7%). 5 failures in `no_main_angle_round3/`.

**Output**: `data/videos/pitcher_calibration_cropped/{Stadium}/{Season}/`

### Training pipeline
```
tools/label_scene_cuts.py          → Hand-label scene cuts + segment types (118 videos)
tools/extract_segment_frames.py    → Extract frames from labeled segments
tools/train_camera_classifier.py   → Train EfficientNet-B0
tools/test_camera_classifier.py    → Evaluate on held-out test set
```
**Labels**: `data/labels/scene_cuts/scene_cut_labels.json`

---

## Stage 3: Player Detection & Classification

Given a cropped main-angle frame, detect all people and classify each as pitcher, catcher, batter, or other.

**Module**: `src/detection/player_pose.py` + `src/filtering/pitcher_classifier.py`

### 3a. Person Detection
- **YOLOv8n** person detection (GPU, conf >= 0.3)
- Produces candidate list with normalized coordinates and bounding boxes

### 3b. Pitcher Classification — [DONE]
- **EfficientNet-B0** binary classifier on each person crop
- Batch-classifies all detected people, picks highest-confidence "pitcher"
- Test accuracy: **100%** on 1,234 test crops (153 pitcher, 1,081 not_pitcher)
- Model: `models/pitcher_classifier/best.pt`
- Fallback: calibrated zone heuristic (`data/pitcher_zones.json`, 30 stadiums, 65,516 samples)

**Training pipeline**:
```
tools/extract_pitcher_crops.py      → Sample frames, YOLO detect, save person crops
tools/label_pitcher_crops.py        → OpenCV UI: label each crop as pitcher/not_pitcher
tools/prepare_pitcher_training.py   → Organize into ImageFolder (split by video, not crop)
tools/train_pitcher_classifier.py   → Train EfficientNet-B0
```
**Data**: `data/labels/pitcher/` — 6,040 crops from 264 videos (32 stadiums x 3 seasons)

### 3c. 4-Class Player Classifier — [DONE]
Upgraded from binary pitcher/not_pitcher to 4-class: pitcher/catcher/batter/other.
- **Model**: `models/player_classifier/best.pt` — EfficientNet-B0, **98.9% test accuracy**
- `PitcherClassifier` class auto-detects binary vs multiclass from checkpoint's `class_names` list
- Binary model (`models/pitcher_classifier/best.pt`, 100% acc) kept as fallback
- Replaces the need for separate catcher/batter classifiers

### Temporal Smoothing
- Tracks previous pitcher position across frames
- Strongly prefers candidate within 0.08 normalized distance of previous position
- Resets after 10 consecutive frames with no detection
- Will extend to catcher/batter once classifiers are trained

---

## Stage 4: Pose Estimation — [DONE]

**Module**: `src/pose/rtmpose_backend.py` (via `src/detection/player_pose.py`)

Runs on cropped player bounding boxes (not full frame) for accuracy and speed.

- **Model**: RTMPose-X (384x288 input, ONNX)
- **Backend**: rtmlib + onnxruntime-gpu
- **Keypoints**: 17 COCO landmarks (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **Model file**: `models/rtmpose/end2end.onnx`
- **GPU**: Auto-adds cuDNN DLLs to PATH via `_ensure_cudnn_on_path()`

### Performance
- **99.9% mean detection, 99.9% mean pose, 100% median** on 50 random videos across 26 stadiums
- 47/50 videos at 100%/100%, worst case 97.3% (Fenway Park)
- ~11.3s per video (200 frames) on RTX 2070 — ~56ms/frame including YOLO + classifier + pose
- Tested via `tools/test_pitcher_pose.py --batch --video-dir data/videos/pitcher_calibration_cropped --random 50 --max-frames 200`

### Environment notes
- Install rtmlib with `--no-deps` to avoid pulling CPU onnxruntime / conflicting opencv
- onnxruntime-gpu needs cuDNN DLLs — rtmpose_backend.py handles this automatically

---

## Stage 5: Ball Detection — [PARTIAL]

**Module**: `src/detection/baseball_detector.py`

Detect the baseball in the full main-angle view for trajectory tracking.

### YOLO-World zero-shot (primary)
- Uses `yolov8s-world.pt` with text prompts `["baseball", "ball"]`
- No custom training needed, production-ready API in `BaseballDetector`
- Confidence threshold: 0.1

### Custom YOLOv8n (secondary)
- Trained on 549 hand-labeled frames
- mAP@50 = **79.9%**
- Model: `models/yolo_baseball/train/weights/best.pt`

### Training pipeline
```
tools/baseball_labeler.py          → YOLO-format labeling UI
tools/train_yolo_baseball.py       → Train YOLOv8n
tools/test_ball_detection.py       → Evaluate custom model
tools/test_ball_yoloworld.py       → Evaluate YOLO-World zero-shot
```
**Data**: `data/labels/baseball/` — 549 labeled frames with YOLO annotations

### What's left
- [ ] Evaluate both approaches on more diverse game footage
- [ ] Ball trajectory smoothing / interpolation across frames
- [ ] Integration with per-frame analysis output

---

## Stage 6: Home Plate Detection — [DONE]

**Module**: `src/detection/home_plate_detector.py`

Detect home plate position for strike zone calibration and pitch location mapping.

- **Method**: HuggingFace SAM3 with text prompt "home plate"
- **Output**: centroid, polygon corners, bounding box, confidence, white ratio
- **Standalone** — does not depend on SAM3 tracker (which was removed)

### Testing
```
tools/test_sam3_home_plate.py      → Quick SAM3 text-prompt test
tools/test_home_plate_detection.py → Full HomePlateDetector evaluation
```

---

## Stage 7: Bat Barrel Detection — [PARTIAL]

Detect the bat barrel position in the batter's dynamic crop for swing analysis.

- **Method**: YOLO-pose keypoint model (bbox + barrel endpoint keypoints)
- **Model**: `models/yolo_bat_barrel/train/weights/best.pt`

### Training pipeline
```
tools/extract_bat_frames.py            → Extract batter frames from game video
tools/scrape_bat_frames_round2.py      → Scrape additional bat training frames
tools/process_2023_videos_for_bat_labeling.py → Prep 2023 videos for labeling
tools/barrel_keypoint_labeler.py       → Label bat barrel keypoints
tools/train_yolo_bat_barrel.py         → Train YOLO-pose bat barrel model
tools/test_bat_barrel.py               → Evaluate model
```
**Data**: `data/labels/bat_barrel/`, `data/bat_frames_round2/`, `data/bat_frames_round2_filtered/`

### What's left
- [ ] Train batter classifier (Stage 3d) to produce dynamic batter crops
- [ ] Integrate bat barrel detection with batter crop pipeline
- [ ] Evaluate barrel tracking accuracy across swing sequences

---

## Stage 8: Catcher Mitt Detection — [PARTIAL]

Detect the catcher's mitt position for pitch location measurement.

- **Method**: YOLOv8-small object detection
- **Model**: `models/yolo_mitt_diverse/weights/best.pt` (527 training frames, 22MB)

### Training pipeline
```
tools/scrape_diverse_dataset.py    → Scrape diverse mitt training frames
tools/label_diverse_frames.py      → Label mitt positions
tools/mitt_labeler.py              → Additional mitt labeling UI
tools/train_diverse_yolo.py        → Train YOLOv8-small
tools/test_mitt_detection.py       → Evaluate model
```
**Data**: `data/yolo_diverse/` — 527 training frames, `data/labels/mitt_finetune/`

### What's left
- [ ] Train catcher classifier (Stage 3c) to produce dynamic catcher crops
- [ ] Integrate mitt detection with catcher crop pipeline
- [ ] Combine mitt position with home plate detection for pitch location mapping

---

## Stage 9: Per-Frame Analysis — [PARTIAL]

Combine all detections into a unified per-frame output.

### Full Pipeline Demo — [DONE]
**Script**: `tools/demo_full_pipeline.py`

Runs ALL detectors on a single cropped video, overlays annotations, writes output video.

Pipeline per frame:
1. YOLO person detection → all person bboxes
2. 4-class classification (PitcherClassifier with multiclass model) → pitcher/catcher/batter/other
3. RTMPose-X pose on pitcher + batter crops (40px padding)
4. Bat barrel (YOLO-pose) on padded batter crop (150px padding), coords mapped to frame space
5. Catcher mitt (YOLO) on padded catcher crop (100px padding), coords mapped to frame space
6. Ball detection (custom YOLOv8n) on full frame
7. Home plate (SAM3) on frame 0 only, then SAM3 freed from VRAM

**Performance**: ~9.3 fps on RTX 2070 (all models on GPU simultaneously).

**Test results** (Dodger Stadium, 386 frames):
| Detection | Frame rate |
|-----------|-----------|
| Pitcher | 100% |
| Batter | 100% |
| Catcher | 99% |
| Ball | 20% |
| Bat barrel | 20% |
| Mitt | 38% |

**Usage**:
```bash
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/demo_full_pipeline.py --video <path>
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/demo_full_pipeline.py --skip-home-plate  # random video, no SAM3
```

### Target output schema
```json
{
  "frame": 42,
  "timestamp_ms": 1400.0,
  "pitcher": {
    "bbox": [0.25, 0.30, 0.15, 0.45],
    "pose": {"left_shoulder": [x, y, conf], ...},
    "confidence": 0.95
  },
  "batter": {
    "bbox": [0.60, 0.40, 0.12, 0.40],
    "pose": {"left_shoulder": [x, y, conf], ...},
    "bat_barrel": {"tip": [x, y], "knob": [x, y]}
  },
  "catcher": {
    "bbox": [0.45, 0.55, 0.10, 0.30],
    "mitt": {"centroid": [x, y], "bbox": [x, y, w, h]}
  },
  "ball": {"centroid": [x, y], "confidence": 0.8},
  "home_plate": {"centroid": [x, y], "corners": [...]}
}
```

### What's left
- [ ] Improve ball detection (20% → target 60%+): more training data, possible model upgrade
- [ ] Improve bat barrel detection (20% → target 60%+): more diverse training data
- [ ] Improve catcher mitt detection (38% → target 70%+): more training data
- [ ] Output structured JSON per-frame data (not just video overlay)
- [ ] Temporal smoothing across frames for all tracked objects
- [ ] Database integration (`src/database/` exists for this)

---

## File Reference

### Core Modules
| File | Role |
|------|------|
| `src/scraper/savant.py` | Baseball Savant scraper |
| `src/scraper/video_downloader.py` | Video downloader |
| `src/filtering/scene_cropper.py` | Scene cut detection + segment classification + ffmpeg crop |
| `src/filtering/camera_filter.py` | EfficientNet-B0 camera angle classifier |
| `src/filtering/pitcher_classifier.py` | EfficientNet-B0 pitcher/not_pitcher classifier |
| `src/detection/player_pose.py` | YOLO detect → find pitcher → crop → RTMPose pose |
| `src/detection/baseball_detector.py` | YOLO-World ball detection |
| `src/detection/home_plate_detector.py` | SAM3 text-prompted home plate detection |
| `tools/demo_full_pipeline.py` | Full pipeline demo: all detections overlaid on video |
| `src/pose/rtmpose_backend.py` | RTMPose-X GPU backend (17 COCO keypoints) |
| `src/pose/base.py` | Pose backend abstract base class |
| `src/database/` | SQLite schema + operations (future use) |
| `src/utils/` | Logging config, video utilities |

### Models
| Path | Description |
|------|-------------|
| `models/camera_classifier/best.pt` | Camera angle classifier (97.7% acc) |
| `models/pitcher_classifier/best.pt` | Pitcher identifier, binary (100% test acc) |
| `models/player_classifier/best.pt` | 4-class player classifier (98.9% test acc) |
| `models/rtmpose/end2end.onnx` | RTMPose-X body pose (384x288, ONNX) |
| `models/yolo_baseball/train/weights/best.pt` | Custom ball detector (79.9% mAP@50) |
| `models/yolo_bat_barrel/train/weights/best.pt` | Bat barrel keypoint model |
| `models/yolo_mitt_diverse/weights/best.pt` | Catcher mitt detector (YOLOv8-small, 527 frames) |

### Data
| Path | Contents |
|------|----------|
| `data/videos/pitcher_calibration/` | 1749 raw videos (30 stadiums x 3 seasons) |
| `data/videos/pitcher_calibration_cropped/` | 1744 cropped main-angle videos |
| `data/pitcher_zones.json` | Per-stadium calibrated pitcher zones (30 stadiums) |
| `data/pitcher_calibration_metadata.json` | Video download metadata (resume-safe) |
| `data/labels/scene_cuts/` | Camera classifier training data + labels (118 videos) |
| `data/labels/pitcher/` | Pitcher classifier training data (6,040 crops) |
| `data/labels/baseball/` | Ball detection labels (549 frames) |
| `data/labels/bat_barrel/` | Bat barrel keypoint labels |
| `data/labels/mitt_finetune/` | Mitt annotation data |
| `data/yolo_diverse/` | Diverse mitt training data (527 frames) |
| `data/bat_frames_round2/` | Bat training frames |

---

## Environment

- **OS**: Windows 10
- **GPU**: RTX 2070 with CUDA
- **Python**: Anaconda env `baseball` (`C:/Users/Spencer/anaconda3/envs/baseball/python.exe`)
- **PyTorch**: 2.9.1+cu126
- **Key packages**: ultralytics, rtmlib (--no-deps), onnxruntime-gpu, torchvision, opencv-python
