# Baseball Biomechanics Analysis System

A comprehensive Python system for analyzing baseball player biomechanics using video segmentation (SAM 3) and pose estimation. The system scrapes pitch/hit videos from Baseball Savant, segments individual players using text prompts, runs pose estimation, and stores the resulting keypoint data linked to Statcast metrics.

## Features

- **Video Scraping**: Automatically download pitch videos and Statcast data from Baseball Savant
- **Player Segmentation**: Use SAM 3 with text prompts to segment pitcher, batter, and catcher
- **Pose Estimation**: Extract body keypoints using MediaPipe (with MotionBERT support planned)
- **Database Storage**: SQLite database (PostgreSQL-ready) linking pose data to Statcast metrics
- **CLI Interface**: Easy-to-use command-line tools for all operations
- **Pipeline Orchestration**: Run end-to-end analysis with progress tracking and resume capability

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for SAM 3)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/baseball-biomechanics.git
cd baseball-biomechanics
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python cli.py db init
```

## Quick Start

### 1. Scrape data for a pitcher
```bash
python cli.py scrape --player-id 592789 --start-date 2024-04-01 --end-date 2024-04-30
```

### 2. Run segmentation on a play
```bash
python cli.py segment --play-id 12345 --roles pitcher,batter
```

### 3. Run pose estimation
```bash
python cli.py pose --play-id 12345 --backend mediapipe
```

### 4. Run the full pipeline
```bash
python cli.py pipeline --player-id 592789 --start-date 2024-04-01 --end-date 2024-04-30 --roles pitcher,batter
```

## CLI Commands

### Scraping
```bash
# Scrape pitcher data
python cli.py scrape --player-id 592789 --start-date 2024-04-01 --end-date 2024-04-30

# Scrape batter data
python cli.py scrape --player-id 592789 --start-date 2024-04-01 --end-date 2024-04-30 --player-type batter

# Scrape without downloading videos
python cli.py scrape --player-id 592789 --start-date 2024-04-01 --end-date 2024-04-30 --no-download
```

### Segmentation
```bash
# Segment all roles
python cli.py segment --play-id 12345 --roles pitcher,batter,catcher

# Use custom prompt
python cli.py segment --play-id 12345 --custom-prompt "player in red helmet"
```

### Pose Estimation
```bash
# MediaPipe backend
python cli.py pose --play-id 12345 --backend mediapipe
```

### Database Management
```bash
# Initialize database
python cli.py db init

# View statistics
python cli.py db stats

# View player stats
python cli.py db player --player-id 592789
```

## Configuration

Edit `config/config.yaml` to customize settings:

```yaml
database:
  url: "sqlite:///data/baseball_biomechanics.db"

scraper:
  base_url: "https://baseballsavant.mlb.com"
  request_delay_seconds: 2
  video_download_dir: "data/videos"

segmentation:
  model: "sam3"
  output_dir: "data/masks"
  default_prompts:
    pitcher: "pitcher on mound throwing baseball"
    batter: "batter at plate with bat"
    catcher: "catcher in crouch behind plate"

pose:
  default_backend: "mediapipe"

logging:
  level: "INFO"
  file: "logs/app.log"
```

## Database Schema

The system uses the following tables:

- **games**: Game metadata (game_pk, date, teams, venue)
- **players**: Player information (ID, name, team, position)
- **plays**: Statcast data for each pitch (pitch type, velocity, spin, outcome, video URL)
- **pose_sequences**: Pose data sequences linked to plays
- **pose_frames**: Individual frames within a sequence
- **keypoints**: 2D/3D body landmarks for each frame
- **segmentation_masks**: SAM 3 output masks with bounding boxes

## Project Structure

```
baseball-biomechanics/
├── src/
│   ├── scraper/          # Baseball Savant scraping
│   ├── segmentation/     # SAM 3 player segmentation
│   ├── pose/             # Pose estimation backends
│   ├── database/         # SQLAlchemy models and operations
│   ├── pipeline/         # Orchestration
│   └── utils/            # Logging, video utilities
├── notebooks/            # Jupyter analysis notebooks
├── data/
│   ├── videos/           # Downloaded videos
│   ├── masks/            # Segmentation masks
│   └── processed/        # Intermediate outputs
├── config/               # Configuration files
├── tests/                # Unit tests
├── cli.py                # Command-line interface
└── requirements.txt
```

## Python API

### Using the Pipeline
```python
from src.pipeline import BiomechanicsPipeline, PipelineConfig

config = PipelineConfig(
    database_url="sqlite:///data/baseball_biomechanics.db",
    roles=["pitcher", "batter"],
)

with BiomechanicsPipeline(config) as pipeline:
    progress = pipeline.run_full_pipeline(
        player_id=592789,
        start_date="2024-04-01",
        end_date="2024-04-30",
    )
    print(f"Processed {progress.pose_estimated} plays")
```

### Using Individual Components
```python
from src.scraper import BaseballSavantScraper
from src.segmentation import SAM3Tracker
from src.pose import MediaPipeBackend

# Scrape data
with BaseballSavantScraper() as scraper:
    df = scraper.search_statcast(player_id=592789, start_date="2024-04-01", end_date="2024-04-30")

# Segment video
tracker = SAM3Tracker()
results = tracker.segment_video("video.mp4", roles=["pitcher", "batter"])

# Estimate pose
backend = MediaPipeBackend()
backend.initialize()
pose_results = backend.process_video("video.mp4")
```

### Querying the Database
```python
from src.database import DatabaseOperations, get_session, init_db

init_db()

with next(get_session()) as session:
    ops = DatabaseOperations(session)

    # Get player stats
    stats = ops.get_player_stats(592789)

    # Get plays with pose data
    plays = ops.get_plays_by_pitcher(592789)

    # Get pose sequences
    for play in plays:
        sequences = ops.get_pose_sequences_for_play(play.play_id)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Baseball Savant](https://baseballsavant.mlb.com) for Statcast data and video
- [SAM 3](https://github.com/ultralytics/ultralytics) for segmentation
- [MediaPipe](https://developers.google.com/mediapipe) for pose estimation
- [SQLAlchemy](https://www.sqlalchemy.org/) for database ORM
