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

- `src/` - Core modules (scraper, detection, filtering, pose, database)
- `tools/` - Utility scripts and training tools
- `models/` - Trained model weights
- `data/` - Data assets (videos, labels, debug output)
- `docs/` - Feature documentation and handoff notes

## Current Status (2026-02-11)

### What's Working
- **Camera cropping pipeline**: Scene cut detection (histogram diff, threshold 0.08, 4x subsample) + EfficientNet-B0 classifier (97.7% accuracy). 1744/1749 videos cropped (99.7%).
- **Pitcher zone calibration**: Complete. `data/pitcher_zones.json` has per-stadium zones from 65,516 position samples across 1744 cropped videos, 30 stadiums.
- **Pitcher classifier**: EfficientNet-B0 binary classifier (pitcher vs not_pitcher), **100% test accuracy** on 1,234 test crops (153 pitcher, 1,081 not_pitcher). Trained on 6,040 hand-labeled person crops from 264 videos across 32 stadiums × 3 seasons. Replaces spatial heuristic picker in `player_pose.py`.
- **Pitcher pose detection**: YOLO person detection (GPU) + pitcher classifier + RTMPose-X 17-landmark pose (GPU via rtmlib/ONNX). Zone heuristics available as fallback.
- **Scene cut labeler**: `tools/label_scene_cuts.py` with `--dir` and `--auto-classify` flags. 118 hand-labeled videos in `data/labels/scene_cuts/`.
- **Full pipeline docs**: `docs/pipeline.md` — end-to-end documentation of all stages.

### In Progress / Next Priority
- **Batch pose test on full 1744 videos**: `tools/test_pitcher_pose.py` now supports `--video-dir` for custom directories and nested `{Stadium}/{Season}/` structure. Run with: `--batch --video-dir data/videos/pitcher_calibration_cropped --per-stadium 1 --max-frames 200`
- **5 misclassified videos**: In `data/videos/pitcher_calibration_cropped/no_main_angle_round3/`. Coors Field night game is the key failure — model has no night game training data from that stadium.

### Critical Rules
- `crop_to_main_angle` has its OWN `cut_threshold` param — must match `detect_scene_cuts` (both 0.08)
- When updating thresholds, check ALL functions that pass threshold values
- `git add -A` hangs on large data directories — use specific file paths
- Labeler saves video status on Q/N — stale labels from old thresholds need clearing
- RTMPose-X backend auto-adds cuDNN DLLs to PATH (pip nvidia-cudnn-cu12) — no manual PATH config needed
- YOLO now runs on GPU by default in player_pose.py; MediaPipeBackend still available for fallback
- `test_pitcher_pose.py --batch` defaults to `2023_cropped/`; use `--video-dir` for other directories
- Pitcher classifier auto-loads from `models/pitcher_classifier/best.pt`; falls back to zone heuristics if not found

### Key Data
- `data/videos/pitcher_calibration/` — 1749 raw videos (30 stadiums x 3 seasons)
- `data/videos/pitcher_calibration_cropped/` — 1744 cropped main-angle videos
- `data/pitcher_zones.json` — per-stadium calibrated pitcher zones (30 stadiums)
- `data/pitcher_calibration_metadata.json` — metadata for all downloaded videos
- `data/labels/scene_cuts/scene_cut_labels.json` — 118 hand-labeled videos
- `data/labels/pitcher/pitcher_labels.json` — 6,040 labeled crops (759 pitcher, 5,281 not_pitcher)
- `models/camera_classifier/best.pt` — EfficientNet-B0 camera classifier (97.7% acc)
- `models/pitcher_classifier/best.pt` — EfficientNet-B0 pitcher classifier (100% acc)
- `data/debug/pitcher_zones_report/` — calibration visualizations (4 plots)
- `models/rtmpose/end2end.onnx` — RTMPose-X body model (384x288, ONNX)
