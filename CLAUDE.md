# Claude Code Configuration

## Running Python on Windows

1. Use the full path to the conda Python executable:
   ```
   C:/Users/Spencer/anaconda3/envs/baseball/python.exe
   ```

2. Use forward slashes for all paths, not backslashes:
   - YES: `F:/Claude_Projects/baseball-biomechanics`
   - NO: `F:\Claude_Projects\baseball-biomechanics`

3. Example command to run scripts:
   ```bash
   C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/demo_pipeline.py
   ```

## File Paths

- Always use relative paths from the project root, not absolute paths
- YES: `tools/demo_pipeline.py`, `PROGRESS.md`
- NO: `F:/Claude_Projects/baseball-biomechanics/tools/demo_pipeline.py`

This fixes file read/write tool issues.

## Project Structure

- `src/` - Core modules (scraper, detection, filtering, pose, database, utils)
- `tools/` - Utility scripts, labelers, and training tools
- `models/` - Trained model weights
- `data/` - Data assets (videos, labels, debug output)
- `docs/` - Pipeline roadmap and documentation

## Current Status (2026-02-11)

### What's Working
- **Camera cropping pipeline**: Scene cut detection (histogram diff, threshold 0.08, 4x subsample) + EfficientNet-B0 classifier (97.7% accuracy). 1744/1749 videos cropped (99.7%).
- **Pitcher zone calibration**: Complete. `data/pitcher_zones.json` has per-stadium zones from 65,516 position samples across 1744 cropped videos, 30 stadiums.
- **Pitcher classifier**: EfficientNet-B0 binary classifier (pitcher vs not_pitcher), **100% test accuracy** on 1,234 test crops. Replaces spatial heuristic picker in `player_pose.py`.
- **Pitcher pose detection**: YOLO person detection (GPU) + pitcher classifier + RTMPose-X 17-landmark pose (GPU via rtmlib/ONNX). Zone heuristics available as fallback.
- **Ball detection**: YOLO-World zero-shot in `src/detection/baseball_detector.py` (primary). Custom YOLOv8n trained (79.9% mAP@50, secondary).
- **Home plate detection**: SAM3 text-prompted, standalone in `src/detection/home_plate_detector.py`.
- **Bat barrel detection**: YOLO-pose keypoint model trained. Needs batter crop pipeline.
- **Catcher mitt detection**: YOLOv8-small (527 frames, `models/yolo_mitt_diverse/`). Needs catcher crop pipeline.
- **Full pipeline roadmap**: `docs/pipeline.md` — all stages documented with status.

### In Progress / Next Priority
- **Batch pose test on full 1744 videos**: `tools/test_pitcher_pose.py --batch --video-dir data/videos/pitcher_calibration_cropped --per-stadium 1 --max-frames 200`
- **Catcher/batter classifiers**: Same pattern as pitcher classifier — needed to produce dynamic crops for mitt and bat barrel detection.

### Critical Rules
- `crop_to_main_angle` has its OWN `cut_threshold` param — must match `detect_scene_cuts` (both 0.08)
- When updating thresholds, check ALL functions that pass threshold values
- `git add -A` hangs on large data directories — use specific file paths
- Labeler saves video status on Q/N — stale labels from old thresholds need clearing
- RTMPose-X backend auto-adds cuDNN DLLs to PATH (pip nvidia-cudnn-cu12) — no manual PATH config needed
- YOLO runs on GPU by default in player_pose.py
- Pitcher classifier auto-loads from `models/pitcher_classifier/best.pt`; falls back to zone heuristics if not found

### Key Data
- `data/videos/pitcher_calibration/` — 1749 raw videos (30 stadiums x 3 seasons)
- `data/videos/pitcher_calibration_cropped/` — 1744 cropped main-angle videos
- `data/pitcher_zones.json` — per-stadium calibrated pitcher zones (30 stadiums)
- `data/pitcher_calibration_metadata.json` — metadata for all downloaded videos
- `data/labels/scene_cuts/scene_cut_labels.json` — 118 hand-labeled videos
- `data/labels/pitcher/pitcher_labels.json` — 6,040 labeled crops (759 pitcher, 5,281 not_pitcher)
- `data/labels/baseball/` — 549 ball detection labeled frames
- `data/labels/bat_barrel/` — bat barrel keypoint labels
- `data/yolo_diverse/` — 527 mitt training frames
- `models/camera_classifier/best.pt` — EfficientNet-B0 camera classifier (97.7% acc)
- `models/pitcher_classifier/best.pt` — EfficientNet-B0 pitcher classifier (100% acc)
- `models/rtmpose/end2end.onnx` — RTMPose-X body model (384x288, ONNX)
- `models/yolo_baseball/` — custom ball detector (79.9% mAP@50)
- `models/yolo_bat_barrel/` — bat barrel keypoint model
- `models/yolo_mitt_diverse/` — catcher mitt detector (YOLOv8-small)
