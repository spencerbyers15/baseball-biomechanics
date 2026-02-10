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
