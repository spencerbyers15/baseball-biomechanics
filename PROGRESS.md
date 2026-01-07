# Baseball Biomechanics Project - Progress Tracker

**Last Updated:** 2026-01-07

---

## Project Overview

End-to-end pipeline for analyzing baseball player biomechanics from Statcast videos using computer vision. Core innovation: multi-player isolation to prevent "Frankenstein poses" from mixed body parts.

---

## Module Status

| Module | Location | Status | Notes |
|--------|----------|--------|-------|
| Video Scraper | `src/scraper/` | **Working** | Baseball Savant API integration |
| SAM3 Segmentation | `src/segmentation/` | **Working** | Multi-player detection + position classification |
| MediaPipe Pose | `src/pose/` | **Working** | 33 keypoints per player |
| Database | `src/database/` | **Working** | SQLite, 44MB, linked to Statcast |
| CLI Interface | `cli.py` | **Working** | scrape, segment, pose commands |
| Baseball Detector | `src/detection/baseball_detector.py` | **Working** | YOLO-World text prompts |
| Home Plate Detector | `src/detection/home_plate_detector.py` | **Working** | SAM + whiteness filtering |
| Camera Angle Filter | `src/filtering/camera_filter.py` | **In Progress** | CLIP embeddings + KNN - memory issues during full run |
| YOLO Bat Barrel | `models/yolo_bat_barrel/` | **Working** | 3-keypoint barrel tracking (mAP50: 0.879) |

---

## Completed Tasks

- [x] Core pipeline architecture (scrape → segment → pose → database)
- [x] Baseball Savant scraper with video download
- [x] SAM3 multi-player segmentation with position heuristics
- [x] MediaPipe pose estimation integration
- [x] SQLite database schema linking poses to Statcast
- [x] YOLO baseball detection (352 labeled frames)
- [x] Scene cut detection + segment classification pipeline
- [x] Home plate detector (SAM point prompts + whiteness filter)
- [x] Camera angle labeling UI (`tools/label_camera_angles.py`)
- [x] Stratified frame extraction from 30 stadiums (~148,620 frames)
- [x] Manual camera angle labeling (150 frames: 122 main + 28 other)
- [x] CLIP embedding generation for labeled frames
- [x] Reference embeddings built for KNN classifier
- [x] UMAP visualization of camera embeddings

---

## In Progress / Blocked

### Bat Barrel Tracking (NEW APPROACH)
- **Status:** Starting fresh with 3-keypoint barrel tracking
- **Previous approach removed:** Old 2-keypoint (knob + cap) model deleted
  - Issue: Knob and hands frequently occluded in frames
- **New approach:** Track only the barrel with 3 keypoints:
  1. Barrel cap (end of bat)
  2. Barrel middle
  3. Barrel beginning (where barrel meets handle, roughly where hands end)
- **Data available:** 413 filtered frames in `data/bat_frames_round2_filtered/`
- **Tools ready:** Labeler, training script, test script created
- **Next:** Label frames, then train model

### Camera Angle Filtering Pipeline
- **Status:** Working (memory issues resolved by using fewer frames)
- **What works:**
  - Labeling UI functional
  - CLIP embeddings working
  - KNN classification works
  - 150 manually labeled reference frames
- **Location:** `src/filtering/camera_filter.py`, `tools/embed_labeled_frames.py`

### Video Temporal Cropping (NEW)
- **Status:** In development
- **Approach:** Scene cut detection + segment classification (more efficient than per-frame)
- **Steps:**
  1. Detect camera cuts (histogram difference / SSIM)
  2. Classify each segment once (sample 1-3 frames, use embedding classifier)
  3. Build crop regions from main_angle segments
  4. Crop with ffmpeg (stream copy for speed)
- **Location:** `src/filtering/scene_cropper.py` (to be created)

---

## Key Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| Jan 2025 | Use CLIP embeddings for camera angle classification | Robust visual similarity without retraining; works across stadiums |
| Jan 2025 | KNN over fine-tuned classifier | Simpler, interpretable, easy to update reference set |
| Jan 2025 | Position heuristics for player classification | Baseball Savant uses consistent center-field camera; rule-based is reliable |
| Jan 2025 | YOLOv8n-pose for bat detection | Lightweight, fast, keypoint detection built-in |
| Jan 2026 | Switch from 2-keypoint to 3-keypoint barrel tracking | Knob/hands often occluded; barrel is more consistently visible |

---

## Data Assets

| Asset | Location | Count/Size |
|-------|----------|------------|
| Labeled baseball frames | `data/labels/baseball/` | 352 frames |
| Bat labeling frames | `data/bat_frames_round2_filtered/` | 413 frames |
| Camera angle labels | `data/camera_angle_labels.json` | 150 frames |
| Stratified stadium frames | `data/all_frames_by_stadium/` | ~148,620 frames |
| Reference embeddings | `data/reference_embeddings.pkl` | Built from labeled frames |
| Database | `data/baseball_biomechanics.db` | 44 MB |

---

## Model Weights

| Model | Location | Notes |
|-------|----------|-------|
| YOLOv8s-World | `yolov8s-world.pt` | Baseball detection |
| SAM 2.1 Base | `sam2.1_b.pt` | Segmentation |
| YOLO Bat Barrel | `models/yolo_bat_barrel/train/weights/best.pt` | 3-keypoint barrel (Pose mAP50-95: 0.826) |
| YOLO Mitt | `models/yolo_mitt/` | Catcher mitt detection |

---

## Next Steps

- [x] Build bat barrel keypoint labeling interface
- [x] Label bat barrel keypoints (302 labeled, 60 skipped, 50 flagged)
- [x] Train YOLOv8-pose model for bat barrel detection (Pose mAP50: 0.879)
- [ ] Integrate bat barrel tracking into main pipeline
- [ ] Create visualization combining player pose + bat angle
- [ ] Build end-to-end demo: video -> segmented players + bat angle + ball tracking

---

## Tools Reference

| Tool | Purpose |
|------|---------|
| `tools/label_camera_angles.py` | Interactive camera angle labeling |
| `tools/barrel_keypoint_labeler.py` | Label bat barrel 3 keypoints + flag wrong angles |
| `tools/train_yolo_bat_barrel.py` | Train YOLOv8-pose bat barrel model |
| `tools/test_bat_barrel.py` | Test bat barrel detection on images/video |
| `tools/embed_labeled_frames.py` | Generate CLIP embeddings |
| `tools/visualize_camera_embeddings.py` | UMAP visualization |
| `tools/test_camera_classifier.py` | Test camera angle classifier |

---

## Session Notes


### 2026-01-07 (Session 2)
- **CLEANUP:** Deleted old bat tracking model and related code
  - Removed: `models/yolo_bat_pose/` (trained model)
  - Removed: `data/labels/bat_keypoints/` (352 labeled frames)
  - Removed: `data/bat_labeling/` (1500 old frames)
  - Removed: `tools/bat_keypoint_labeler.py`
  - Removed: `tools/train_yolo_bat_pose.py`
  - Removed: `tools/test_bat_detector.py`
  - Removed: `tools/test_bat_detector_all_stadiums.py`
- **NEW APPROACH:** 3-keypoint barrel tracking
  - Reason: Knob and hands are frequently occluded; barrel is more visible
  - Keypoints: barrel cap, barrel middle, barrel beginning
  - Will use YOLOv8-pose for keypoint detection
- **LABELING COMPLETE:** 412 frames processed
  - 302 labeled with 3 keypoints
  - 60 skipped (bat not visible)
  - 50 flagged as wrong camera angle
- **MODEL TRAINED:** 100 epochs in 6.4 minutes
  - Pose mAP50: 0.879
  - Pose mAP50-95: 0.826
  - Best weights: `models/yolo_bat_barrel/train/weights/best.pt`
- **NEW TOOLS CREATED:**
  - `tools/barrel_keypoint_labeler.py` - Interactive labeler with:
    - Click to place 3 keypoints (cap, middle, beginning)
    - Drag to adjust, right-click to remove
    - Skip frames (bat not visible)
    - Flag frames as wrong camera angle (for debugging classifier)
    - Outputs YOLO pose format + flagged_frames.json
  - `tools/train_yolo_bat_barrel.py` - Training pipeline with:
    - Auto train/val split
    - Custom 3-keypoint skeleton config
    - Configurable hyperparameters
  - `tools/test_bat_barrel.py` - Test on images/video
  - `models/yolo_bat_barrel/data.yaml` - Dataset config

### 2026-01-07 (Session 1)
- Fresh session orientation
- Reviewed bat detector: **excellent results** - mAP50-95 of 0.94 for pose, detections look accurate across diverse stadiums
- Created this PROGRESS.md file
- Ran bat detector on 15 videos (75 frames): **65.3% detection rate**
- Test outputs saved to `data/debug/bat_tracking_test/` (75 images + grid overview)
- Detections show accurate knob/cap keypoints with ~0.92 confidence
- Note: Detection rate lower than validation mAP because test includes frames where bat is occluded, in motion blur, or batter not in stance
- Ran bat detector on **all 30 stadiums** (5 frames each, 150 total): **50% detection rate**
  - Top performers (80%): Angel Stadium, Great American, Oakland, Petco, T-Mobile, Tropicana
  - Low performers (0%): Busch Stadium, Comerica Park - caused by **replay/outfield shots** in sample
  - Key insight: Random frames include replays, close-ups, and fielding shots where no bat exists
  - This validates the need for camera angle filtering before bat detection
- Downloaded **Round 2 training data** from 2023 season for bat detection
  - 30 stadiums, 32 unique videos, **132 stance frames** extracted
  - Frames saved to `data/bat_frames_round2/`
  - Focus on early video frames (stance/load phase) where bat is most visible
  - Ready for labeling with `tools/bat_keypoint_labeler.py`
- Built **scene cut detection + segment classification** pipeline
  - Module: `src/filtering/scene_cropper.py`
  - Scene cut detection: histogram difference (fast) and SSIM methods
  - Segment classification: samples 1-3 frames per segment, uses CLIP/KNN classifier
  - Visualization: timeline with segments colored by classification + thumbnails
  - **Verified:** Replay segment correctly classified as "other" (100% confidence)
  - Use `keep_segments="longest"` to get main at-bat footage
  - Output: `data/debug/scene_crop_test/`
- Processed **all 44 2023 videos** through scene cropper
  - 43/44 successful (1 had no main_angle segments)
  - Cropped videos saved to `data/videos/2023_cropped/`
  - **412 frames** extracted to `data/bat_frames_round2_filtered/`
  - Processing script: `tools/process_2023_videos_for_bat_labeling.py`
- Updated `tools/bat_keypoint_labeler.py` with `--frames-dir` argument for custom frame directories
- **Labeler launched** on 412 filtered frames - ready for bat keypoint annotation
