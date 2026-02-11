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
- **Pitcher pose detection**: YOLO person detection + MediaPipe 33-landmark pose. With calibrated zones: 94.8% mean detection, 93.6% mean pose (tested on 30 videos from 2023_cropped, 1 per stadium).
- **Scene cut labeler**: `tools/label_scene_cuts.py` with `--dir` and `--auto-classify` flags. 118 hand-labeled videos in `data/labels/scene_cuts/`.

### In Progress / Next Priority
- **GPU acceleration for pose pipeline**: Current pipeline runs YOLO (CPU, 10.6ms) + MediaPipe (CPU-only, 27.8ms) = 38.4ms/frame. At 1744 videos x ~420 frames each, full batch takes ~10 hours. Need to move to GPU to make this practical. See `docs/handoff.md` for details.
- **Batch pose test on full 1744 videos**: `tools/test_pitcher_pose.py` currently only tests against `data/videos/2023_cropped/` (30 videos). Needs updating to run on `data/videos/pitcher_calibration_cropped/` (1744 videos, `{Stadium}/{Season}/` structure).
- **5 misclassified videos**: In `data/videos/pitcher_calibration_cropped/no_main_angle_round3/`. Coors Field night game is the key failure — model has no night game training data from that stadium.

### Critical Rules
- `crop_to_main_angle` has its OWN `cut_threshold` param — must match `detect_scene_cuts` (both 0.08)
- When updating thresholds, check ALL functions that pass threshold values
- `git add -A` hangs on large data directories — use specific file paths
- Labeler saves video status on Q/N — stale labels from old thresholds need clearing
- MediaPipe Python on desktop is CPU-only — no GPU delegate available
- YOLO person detection runs on CPU by default; pass `device='cuda'` for GPU
- `test_pitcher_pose.py --batch` uses `2023_cropped/` not `pitcher_calibration_cropped/`

### Key Data
- `data/videos/pitcher_calibration/` — 1749 raw videos (30 stadiums x 3 seasons)
- `data/videos/pitcher_calibration_cropped/` — 1744 cropped main-angle videos
- `data/pitcher_zones.json` — per-stadium calibrated pitcher zones (30 stadiums)
- `data/pitcher_calibration_metadata.json` — metadata for all downloaded videos
- `data/labels/scene_cuts/scene_cut_labels.json` — 118 hand-labeled videos
- `models/camera_classifier/best.pt` — EfficientNet-B0 camera classifier (97.7% acc)
- `data/debug/pitcher_zones_report/` — calibration visualizations (4 plots)
