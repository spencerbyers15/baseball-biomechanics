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

## Current Status (2026-02-09)

### Completed This Session
- **Pitcher calibration scraper** (`tools/scrape_pitcher_calibration.py`): Downloaded 1920 raw videos across 30 stadiums x 3 seasons (2023-2025), 10 RHP + 10 LHP each. Videos in `data/videos/pitcher_calibration/{Stadium_Name}/{season}/`.
- **Calibration script** (`tools/calibrate_pitcher_zones.py`): Ready to run. Will compute per-stadium pitcher zones from the 1920 calibration videos and output `data/pitcher_zones.json`.
- **Player pose detector upgraded** (`src/detection/player_pose.py`): Added calibrated zone support, distance-based scoring (replaces "pick lowest"), temporal smoothing, `set_stadium()`, `reset_temporal()`. Fully backward compatible.
- **Test script updated** (`tools/test_pitcher_pose.py`): Added `--stadium`, `--no-temporal`, `--zones-path` flags. Batch mode auto-detects stadium from directory name.
- **Bug fix in savant.py**: Fixed `get_game_play_ids()` to count `no_pitch` events (pitch timer violations, intentional walks) alongside `pitch` events for correct pitch_number alignment.

### Blocked / Next Priority
- **Camera angle cropping is broken**: The CLIP+KNN classifier in `src/filtering/camera_filter.py` misclassifies many frames, causing `src/filtering/scene_cropper.py` to crop incorrectly. It previously worked on 43/44 videos in 2023_cropped but fails on the new calibration videos. See `docs/handoff.md` for investigation details.
- **Calibration needs cropping first**: The 1920 raw videos contain non-pitching segments (replays, close-ups, dugout shots). `calibrate_pitcher_zones.py` should run on cropped main-angle-only videos for accurate zone computation. Running on raw videos will introduce noise from non-pitching frames.

### Key Data
- `data/videos/pitcher_calibration/` — 1920 raw videos (30 stadiums x 3 seasons)
- `data/pitcher_calibration_metadata.json` — metadata for all downloaded videos
- `data/pitcher_calibration_scraper.log` — scraper run log
- `data/camera_angle_labels.json` — 150 hand-labeled frames (122 main + 28 other)
- `data/labeled_frames_embeddings.pkl` — CLIP embeddings for KNN classifier
