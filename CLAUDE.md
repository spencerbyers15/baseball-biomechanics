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
- **4-class player classifier**: EfficientNet-B0 (pitcher/catcher/batter/other), **98.9% test accuracy**. Model: `models/player_classifier/best.pt`. Binary fallback: `models/pitcher_classifier/best.pt` (100% test acc).
- **Pitcher pose detection**: YOLO person detection (GPU) + classifier + RTMPose-X 17-landmark pose (GPU via rtmlib/ONNX). Zone heuristics available as fallback.
- **Ball detection**: YOLO-World zero-shot in `src/detection/baseball_detector.py` (primary). Custom YOLOv8n trained (79.9% mAP@50, secondary).
- **Home plate detection**: SAM3 text-prompted, standalone in `src/detection/home_plate_detector.py`.
- **Bat barrel detection**: YOLO-pose keypoint model trained. Runs on padded batter crop.
- **Catcher mitt detection**: YOLOv8-small (527 frames, `models/yolo_mitt_diverse/`). Runs on padded catcher crop.
- **Full pipeline demo**: `tools/demo_full_pipeline.py` — runs ALL detections on a single video, overlays annotations, writes output video. Tested at ~9.3 fps on RTX 2070.
- **Full pipeline roadmap**: `docs/pipeline.md` — all stages documented with status.

### Full Pipeline Demo Results (Dodger Stadium test video, 386 frames)
- Pitcher: 100%, Batter: 100%, Catcher: 99%
- Ball: 20%, Bat barrel: 20%, Mitt: 38%
- **Next priority**: Improve ball, bat barrel, and catcher mitt detection (more training data needed for all three)

### In Progress / Next Priority
- **Improve ball detection**: Only 20% frame detection rate. Current model trained on 549 frames. Needs more labeled data + possible architecture upgrade.
- **Improve bat barrel detection**: Only 20% frame detection rate. YOLO-pose keypoint model on padded batter crop. Needs more diverse training data.
- **Improve catcher mitt detection**: Only 38% frame detection rate. YOLOv8-small on padded catcher crop. Needs more training data (currently 527 frames).

### Frame Extraction for CV Training

**Any time we need frames to label for training a CV model, use `tools/frame_extractor.py`.**

This is a general-purpose tool that scrubs videos, lets the user set a start/end range, then extracts evenly-spaced frames from that range for later labeling. It works for ball detection, bat barrel, catcher mitt, or anything else.

**Interactive mode** (prompts for what to label, how many videos, frames per video):
```
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/frame_extractor.py
```

**CLI mode** (skip prompts):
```
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/frame_extractor.py --name ball --count 60 --frames 10
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/frame_extractor.py --name mitt --count 30 --frames 15
```

**Custom video sources** (instead of calibration pool):
```
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/frame_extractor.py --videos path1.mp4 path2.mp4 --name ball
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/frame_extractor.py --dir data/videos/some_folder --name bat_barrel
```

**Re-run specific videos**:
```
C:/Users/Spencer/anaconda3/envs/baseball/python.exe tools/frame_extractor.py --name ball --count 60 --redo VIDEO_ID
```

- Default video source: 1744 cropped calibration videos, sampled balanced by stadium + LHP/RHP
- Output: `data/labels/{name}/frames/` with session file for resume
- Sessions are per-name so ball/bat/mitt extractions don't collide
- UI keys: SPACE (set start/end), P (play), ENTER (confirm), N (skip), B (back), ESC (save & quit)

### Critical Rules
- `crop_to_main_angle` has its OWN `cut_threshold` param — must match `detect_scene_cuts` (both 0.08)
- When updating thresholds, check ALL functions that pass threshold values
- `git add -A` hangs on large data directories — use specific file paths
- Labeler saves video status on Q/N — stale labels from old thresholds need clearing
- RTMPose-X backend auto-adds cuDNN DLLs to PATH (pip nvidia-cudnn-cu12) — no manual PATH config needed
- YOLO runs on GPU by default in player_pose.py
- Pitcher classifier auto-loads from `models/pitcher_classifier/best.pt`; falls back to zone heuristics if not found
- `PitcherClassifier` auto-detects binary vs 4-class from checkpoint's `class_names` — no API change needed
- `models/player_classifier/best.pt` is the 4-class model (pitcher/catcher/batter/other, 98.9% acc); `models/pitcher_classifier/best.pt` is the binary fallback
- `demo_full_pipeline.py` prefers multiclass model, falls back to binary if not found
- SAM3 home plate is ~2.5GB VRAM — demo script runs it on frame 0 only, then frees VRAM before loading other models
- Bat barrel model was trained on batter crops — must run on padded batter crop, NOT full frame
- Mitt model works best on catcher-region crops, NOT full frame

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
