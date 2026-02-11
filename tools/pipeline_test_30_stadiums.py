"""
Pipeline Test: 30 Stadiums from 2025 Season

Downloads 1 video per MLB stadium from 2025, then runs the full pipeline:
1. Crop to main angle (scene cropper)
2. Detect baseball (YOLO)
3. Detect bat barrel keypoints (YOLO-Pose)
4. Detect home plate (SAM)

Outputs annotated videos to data/debug/pipeline_test_2025/
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import random

import cv2
import numpy as np

# Add project to path
PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper.savant import BaseballSavantScraper
from src.filtering.scene_cropper import crop_to_main_angle

# All 30 MLB stadiums
MLB_STADIUMS = {
    "American_Family_Field": {"code": "32", "home_team": "MIL"},
    "Angel_Stadium": {"code": "1", "home_team": "LAA"},
    "Busch_Stadium": {"code": "2889", "home_team": "STL"},
    "Chase_Field": {"code": "15", "home_team": "ARI"},
    "Citi_Field": {"code": "3289", "home_team": "NYM"},
    "Citizens_Bank_Park": {"code": "2681", "home_team": "PHI"},
    "Comerica_Park": {"code": "2394", "home_team": "DET"},
    "Coors_Field": {"code": "19", "home_team": "COL"},
    "Dodger_Stadium": {"code": "22", "home_team": "LAD"},
    "Fenway_Park": {"code": "3", "home_team": "BOS"},
    "Globe_Life_Field": {"code": "5325", "home_team": "TEX"},
    "Great_American_Ball_Park": {"code": "2602", "home_team": "CIN"},
    "Guaranteed_Rate_Field": {"code": "4", "home_team": "CWS"},
    "Kauffman_Stadium": {"code": "7", "home_team": "KC"},
    "LoanDepot_Park": {"code": "4169", "home_team": "MIA"},
    "Minute_Maid_Park": {"code": "2392", "home_team": "HOU"},
    "Nationals_Park": {"code": "3309", "home_team": "WSH"},
    "Oakland_Coliseum": {"code": "10", "home_team": "OAK"},
    "Oracle_Park": {"code": "2395", "home_team": "SF"},
    "Oriole_Park": {"code": "2", "home_team": "BAL"},
    "Petco_Park": {"code": "2680", "home_team": "SD"},
    "PNC_Park": {"code": "31", "home_team": "PIT"},
    "Progressive_Field": {"code": "5", "home_team": "CLE"},
    "Rogers_Centre": {"code": "14", "home_team": "TOR"},
    "T-Mobile_Park": {"code": "680", "home_team": "SEA"},
    "Target_Field": {"code": "3312", "home_team": "MIN"},
    "Tropicana_Field": {"code": "12", "home_team": "TB"},
    "Truist_Park": {"code": "4705", "home_team": "ATL"},
    "Wrigley_Field": {"code": "17", "home_team": "CHC"},
    "Yankee_Stadium": {"code": "3313", "home_team": "NYY"},
}


class PipelineTest:
    """Run full pipeline on 30 stadium videos from 2025."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or PROJECT_ROOT / "data" / "debug" / "pipeline_test_2025"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.videos_dir = self.output_dir / "raw_videos"
        self.cropped_dir = self.output_dir / "cropped_videos"
        self.annotated_dir = self.output_dir / "annotated_videos"

        self.videos_dir.mkdir(exist_ok=True)
        self.cropped_dir.mkdir(exist_ok=True)
        self.annotated_dir.mkdir(exist_ok=True)

        # Load models lazily
        self._baseball_model = None
        self._bat_model = None
        self._plate_detector = None

        self.scraper = BaseballSavantScraper(request_delay=2.0)
        self.results = {}

    @property
    def baseball_model(self):
        """Load baseball detector on demand."""
        if self._baseball_model is None:
            from ultralytics import YOLO
            model_path = PROJECT_ROOT / "models" / "yolo_baseball" / "train" / "weights" / "best.pt"
            if model_path.exists():
                print(f"  Loading baseball detector: {model_path}")
                self._baseball_model = YOLO(str(model_path))
            else:
                print(f"  WARNING: Baseball model not found at {model_path}")
        return self._baseball_model

    @property
    def bat_model(self):
        """Load bat barrel detector on demand."""
        if self._bat_model is None:
            from ultralytics import YOLO
            model_path = PROJECT_ROOT / "models" / "yolo_bat_barrel" / "train" / "weights" / "best.pt"
            if model_path.exists():
                print(f"  Loading bat barrel detector: {model_path}")
                self._bat_model = YOLO(str(model_path))
            else:
                print(f"  WARNING: Bat barrel model not found at {model_path}")
        return self._bat_model

    @property
    def plate_detector(self):
        """Load home plate detector on demand."""
        if self._plate_detector is None:
            try:
                from src.detection.home_plate_detector import HomePlateDetector
                print("  Loading home plate detector (SAM)...")
                self._plate_detector = HomePlateDetector(model_name="sam2.1_b.pt")
            except Exception as e:
                print(f"  WARNING: Could not load plate detector: {e}")
        return self._plate_detector

    def search_stadium_2025(self, stadium_code: str, home_team: str) -> dict:
        """Search for a single video from 2025 season at this stadium."""
        import pandas as pd

        params = {
            "all": "true",
            "hfPT": "",
            "hfAB": "",
            "hfGT": "R|",  # Regular season
            "hfPR": "",
            "hfZ": "",
            "hfStadium": f"{stadium_code}|",
            "hfBBL": "",
            "hfNewZones": "",
            "hfPull": "",
            "hfC": "",
            "hfSea": "2025|",
            "hfSit": "",
            "hfOuts": "",
            "hfOpponent": "",
            "hfInn": "",
            "hfBBT": "",
            "hfFlag": "",
            "hfSA": "",
            "player_type": "pitcher",
            "min_pitches": "0",
            "min_results": "0",
            "group_by": "name",
            "sort_col": "pitches",
            "player_event_sort": "pitch_number_thisgame",
            "sort_order": "desc",
            "min_pas": "0",
            "type": "details",
            "game_date_gt": "2025-03-20",
            "game_date_lt": "2025-12-31",
        }

        try:
            response = self.scraper._make_request(self.scraper.CSV_ENDPOINT, params)
            df = pd.read_csv(pd.io.common.StringIO(response.text), low_memory=False)

            if df.empty:
                return None

            # Pick a random row from available pitches
            row = df.sample(n=1).iloc[0]

            # Get video URL
            play_id_cache = {}
            video_url = self.scraper.get_video_url_from_statcast_row(row, play_id_cache)

            if video_url:
                return {
                    "game_pk": int(row["game_pk"]),
                    "at_bat": int(row["at_bat_number"]),
                    "pitch": int(row["pitch_number"]),
                    "game_date": str(row.get("game_date", ""))[:10],
                    "away_team": row.get("away_team", "UNK"),
                    "home_team": home_team,
                    "video_url": video_url,
                }
            return None

        except Exception as e:
            print(f"    Error searching: {e}")
            return None

    def download_video(self, video_info: dict, stadium_name: str) -> Path:
        """Download a single video."""
        video_id = f"{stadium_name}_{video_info['game_pk']}_{video_info['at_bat']}_{video_info['pitch']}"
        video_path = self.videos_dir / f"{video_id}.mp4"

        if video_path.exists():
            print(f"    Already downloaded: {video_id}")
            return video_path

        try:
            import requests
            response = requests.get(video_info["video_url"], timeout=30)
            response.raise_for_status()

            with open(video_path, "wb") as f:
                f.write(response.content)

            print(f"    Downloaded: {video_id}")
            return video_path

        except Exception as e:
            print(f"    Download failed: {e}")
            return None

    def crop_video(self, video_path: Path) -> Path:
        """Crop video to main angle."""
        cropped_path = self.cropped_dir / f"{video_path.stem}_cropped.mp4"

        if cropped_path.exists():
            print(f"    Already cropped: {cropped_path.name}")
            return cropped_path

        try:
            result = crop_to_main_angle(
                str(video_path),
                str(cropped_path),
                keep_segments="first",
                detection_method="histogram",
                min_segment_duration=0.5,
                show_progress=False,
            )

            if cropped_path.exists():
                print(f"    Cropped: {cropped_path.name}")
                return cropped_path
            else:
                print(f"    Crop failed: no output")
                return None

        except Exception as e:
            print(f"    Crop failed: {e}")
            return None

    def detect_baseball(self, frame: np.ndarray) -> list:
        """Detect baseballs in a frame."""
        if self.baseball_model is None:
            return []

        results = self.baseball_model(frame, verbose=False)
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    if conf > 0.3:
                        detections.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                        })

        return detections

    def detect_bat_barrel(self, frame: np.ndarray) -> list:
        """Detect bat barrel keypoints in a frame."""
        if self.bat_model is None:
            return []

        results = self.bat_model(frame, verbose=False)
        detections = []

        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts = result.keypoints.data.cpu().numpy()
                for kpt in kpts:
                    # kpt shape: (3, 3) -> (x, y, visibility) for each keypoint
                    keypoints = []
                    for i, (x, y, v) in enumerate(kpt):
                        if v > 0.3:
                            keypoints.append({"idx": i, "x": int(x), "y": int(y), "v": float(v)})

                    if keypoints:
                        detections.append({"keypoints": keypoints})

        return detections

    def detect_plate(self, frame: np.ndarray) -> dict:
        """Detect home plate in a frame."""
        if self.plate_detector is None:
            return None

        try:
            detection = self.plate_detector.detect_frame(frame, frame_number=0)
            if detection:
                return {
                    "centroid": list(detection.centroid),
                    "corners": [list(c) for c in detection.corners] if detection.corners else None,
                    "confidence": detection.confidence if hasattr(detection, 'confidence') else 0.5,
                }
        except Exception as e:
            pass

        return None

    def draw_annotations(self, frame: np.ndarray, baseball_dets: list, bat_dets: list, plate_det: dict) -> np.ndarray:
        """Draw all detections on a frame."""
        annotated = frame.copy()

        # Draw baseballs (green circles)
        for det in baseball_dets:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max(10, (x2 - x1) // 2)
            cv2.circle(annotated, (cx, cy), radius, (0, 255, 0), 2)
            cv2.putText(annotated, f"BALL {det['confidence']:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw bat barrel keypoints (blue/cyan)
        keypoint_colors = [(255, 0, 0), (255, 165, 0), (0, 255, 255)]  # barrel_cap, middle, beginning
        keypoint_names = ["cap", "mid", "base"]

        for det in bat_dets:
            points = []
            for kp in det["keypoints"]:
                idx = kp["idx"]
                x, y = kp["x"], kp["y"]
                color = keypoint_colors[idx] if idx < 3 else (255, 255, 255)
                cv2.circle(annotated, (x, y), 6, color, -1)
                cv2.circle(annotated, (x, y), 8, (0, 0, 0), 2)
                points.append((x, y, idx))

            # Draw line through keypoints
            if len(points) >= 2:
                points_sorted = sorted(points, key=lambda p: p[2])
                for i in range(len(points_sorted) - 1):
                    pt1 = (points_sorted[i][0], points_sorted[i][1])
                    pt2 = (points_sorted[i+1][0], points_sorted[i+1][1])
                    cv2.line(annotated, pt1, pt2, (255, 165, 0), 2)

        # Draw home plate (white polygon)
        if plate_det and plate_det.get("corners"):
            corners = np.array(plate_det["corners"], dtype=np.int32)
            cv2.polylines(annotated, [corners], True, (255, 255, 255), 2)
            cx, cy = plate_det["centroid"]
            cv2.putText(annotated, "PLATE", (int(cx), int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif plate_det and plate_det.get("centroid"):
            cx, cy = plate_det["centroid"]
            cv2.circle(annotated, (int(cx), int(cy)), 15, (255, 255, 255), 2)
            cv2.putText(annotated, "PLATE", (int(cx)-20, int(cy)-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def process_video(self, video_path: Path, stadium_name: str) -> dict:
        """Process a single video through the full pipeline."""
        annotated_path = self.annotated_dir / f"{video_path.stem}_annotated.mp4"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Could not open video"}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(annotated_path), fourcc, fps, (width, height))

        stats = {
            "stadium": stadium_name,
            "video": video_path.name,
            "total_frames": total_frames,
            "frames_with_baseball": 0,
            "frames_with_bat": 0,
            "frames_with_plate": 0,
            "baseball_detections": 0,
            "bat_detections": 0,
        }

        # Detect plate only on first frame (it's static)
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return {"error": "Could not read first frame"}

        plate_det = self.detect_plate(first_frame)
        if plate_det:
            stats["frames_with_plate"] = total_frames  # Plate is in every frame

        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            baseball_dets = self.detect_baseball(frame)
            bat_dets = self.detect_bat_barrel(frame)

            # Update stats
            if baseball_dets:
                stats["frames_with_baseball"] += 1
                stats["baseball_detections"] += len(baseball_dets)
            if bat_dets:
                stats["frames_with_bat"] += 1
                stats["bat_detections"] += len(bat_dets)

            # Draw annotations
            annotated = self.draw_annotations(frame, baseball_dets, bat_dets, plate_det)

            # Add info overlay
            cv2.putText(annotated, f"Stadium: {stadium_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Frame: {frame_idx}/{total_frames}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Legend
            cv2.putText(annotated, "GREEN=Ball  ORANGE=Bat  WHITE=Plate", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()

        stats["annotated_video"] = str(annotated_path)
        return stats

    def run(self):
        """Run the full pipeline test."""
        print("=" * 70)
        print("PIPELINE TEST: 30 MLB Stadiums - 2025 Season")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Processing {len(MLB_STADIUMS)} stadiums")
        print("=" * 70)

        all_stats = []

        for i, (stadium_name, stadium_info) in enumerate(MLB_STADIUMS.items(), 1):
            print(f"\n[{i}/30] {stadium_name} ({stadium_info['home_team']})")
            print("-" * 50)

            # Step 1: Search for 2025 video
            print("  Searching for 2025 video...")
            video_info = self.search_stadium_2025(stadium_info["code"], stadium_info["home_team"])

            if not video_info:
                print("  No 2025 video found, skipping...")
                all_stats.append({"stadium": stadium_name, "error": "No 2025 video found"})
                continue

            print(f"  Found: {video_info['home_team']} vs {video_info['away_team']} ({video_info['game_date']})")

            # Step 2: Download
            video_path = self.download_video(video_info, stadium_name)
            if not video_path:
                all_stats.append({"stadium": stadium_name, "error": "Download failed"})
                continue

            # Step 3: Crop to main angle
            print("  Cropping to main angle...")
            cropped_path = self.crop_video(video_path)

            if not cropped_path:
                # Use original if cropping failed
                cropped_path = video_path
                print("  Using original video (crop failed)")

            # Step 4: Run detectors and annotate
            print("  Running detectors...")
            stats = self.process_video(cropped_path, stadium_name)
            stats["original_video"] = str(video_path)
            stats["cropped_video"] = str(cropped_path) if cropped_path != video_path else None
            stats["game_info"] = video_info

            all_stats.append(stats)

            # Print summary
            print(f"  Results:")
            print(f"    - Baseball: {stats.get('frames_with_baseball', 0)}/{stats.get('total_frames', 0)} frames")
            print(f"    - Bat barrel: {stats.get('frames_with_bat', 0)}/{stats.get('total_frames', 0)} frames")
            print(f"    - Home plate: {'Detected' if stats.get('frames_with_plate', 0) > 0 else 'Not detected'}")

            time.sleep(1)  # Rate limit

        # Save results
        results_path = self.output_dir / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        successful = [s for s in all_stats if "error" not in s]
        print(f"Successfully processed: {len(successful)}/{len(MLB_STADIUMS)} stadiums")

        if successful:
            avg_baseball = sum(s.get("frames_with_baseball", 0) for s in successful) / len(successful)
            avg_bat = sum(s.get("frames_with_bat", 0) for s in successful) / len(successful)
            plates_detected = sum(1 for s in successful if s.get("frames_with_plate", 0) > 0)

            print(f"Average frames with baseball: {avg_baseball:.1f}")
            print(f"Average frames with bat: {avg_bat:.1f}")
            print(f"Stadiums with plate detected: {plates_detected}/{len(successful)}")

        print(f"\nResults saved to: {results_path}")
        print(f"Annotated videos in: {self.annotated_dir}")

        self.scraper.close()
        return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline test on 30 stadiums from 2025")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: data/debug/pipeline_test_2025)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    tester = PipelineTest(output_dir=output_dir)
    tester.run()


if __name__ == "__main__":
    main()
