#!/usr/bin/env python
"""
Calibrate per-stadium pitcher zones from calibration videos.

Runs pitcher detection on all calibration videos, records where the pitcher
bbox lands, and computes per-stadium spatial zones (mean/std position,
RHP/LHP offsets, bbox size stats).

Usage:
    # Run on all calibration data
    python tools/calibrate_pitcher_zones.py

    # Specific stadiums
    python tools/calibrate_pitcher_zones.py --stadiums "Dodger Stadium,Fenway Park"

    # Use existing 2023_cropped videos instead of calibration data
    python tools/calibrate_pitcher_zones.py --use-existing

    # Generate visual report only (from existing zones)
    python tools/calibrate_pitcher_zones.py --report-only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.player_pose import PlayerPoseDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "data" / "videos" / "pitcher_calibration"
CROPPED_DIR = PROJECT_ROOT / "data" / "videos" / "pitcher_calibration_cropped"
EXISTING_DIR = PROJECT_ROOT / "data" / "videos" / "2023_cropped"
METADATA_PATH = PROJECT_ROOT / "data" / "pitcher_calibration_metadata.json"
ZONES_PATH = PROJECT_ROOT / "data" / "pitcher_zones.json"
REPORT_DIR = PROJECT_ROOT / "data" / "debug" / "pitcher_zones_report"

# Frames to sample per video
SAMPLE_EVERY_N = 5  # Every 5th frame
MAX_FRAMES_PER_VIDEO = 200


def collect_pitcher_positions(
    detector: PlayerPoseDetector,
    video_path: str,
    p_throws: str = "?",
    sample_every: int = SAMPLE_EVERY_N,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
) -> List[dict]:
    """Run pitcher detection on a video and collect bbox positions.

    Returns list of dicts with cx_norm, cy_norm, bbox_w_norm, bbox_h_norm, p_throws.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if h == 0 or w == 0:
        cap.release()
        return []

    positions = []
    frame_num = 0

    while frame_num < min(total_frames, max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_every == 0:
            # Use internal person detection + pitcher heuristic
            candidates = detector._detect_persons(frame)
            pitcher = detector._find_pitcher(candidates)

            if pitcher is not None:
                x1, y1, x2, y2 = pitcher["bbox"]
                bbox_w = (x2 - x1) / w
                bbox_h = (y2 - y1) / h

                positions.append({
                    "cx_norm": pitcher["cx_norm"],
                    "cy_norm": pitcher["cy_norm"],
                    "bbox_w_norm": bbox_w,
                    "bbox_h_norm": bbox_h,
                    "conf": pitcher["conf"],
                    "p_throws": p_throws,
                    "frame": frame_num,
                })

        frame_num += 1

    cap.release()
    return positions


def compute_stadium_zone(positions: List[dict]) -> Optional[dict]:
    """Compute zone stats from collected pitcher positions.

    Returns zone dict or None if insufficient data.
    """
    if len(positions) < 10:
        return None

    cx_vals = [p["cx_norm"] for p in positions]
    cy_vals = [p["cy_norm"] for p in positions]
    bw_vals = [p["bbox_w_norm"] for p in positions]
    bh_vals = [p["bbox_h_norm"] for p in positions]

    zone = {
        "mean_cx": float(np.mean(cx_vals)),
        "mean_cy": float(np.mean(cy_vals)),
        "std_cx": float(np.std(cx_vals)),
        "std_cy": float(np.std(cy_vals)),
        "median_cx": float(np.median(cx_vals)),
        "median_cy": float(np.median(cy_vals)),
        "mean_bbox_w": float(np.mean(bw_vals)),
        "mean_bbox_h": float(np.mean(bh_vals)),
        "std_bbox_w": float(np.std(bw_vals)),
        "std_bbox_h": float(np.std(bh_vals)),
        "n_samples": len(positions),
    }

    # RHP vs LHP offsets
    rhp = [p for p in positions if p["p_throws"] == "R"]
    lhp = [p for p in positions if p["p_throws"] == "L"]

    if rhp:
        zone["rhp_cx"] = float(np.mean([p["cx_norm"] for p in rhp]))
        zone["rhp_cy"] = float(np.mean([p["cy_norm"] for p in rhp]))
        zone["n_rhp"] = len(rhp)
    if lhp:
        zone["lhp_cx"] = float(np.mean([p["cx_norm"] for p in lhp]))
        zone["lhp_cy"] = float(np.mean([p["cy_norm"] for p in lhp]))
        zone["n_lhp"] = len(lhp)

    return zone


def compute_per_season_zones(
    positions: List[dict], seasons: List[str]
) -> dict:
    """Check for year-over-year drift by computing zones per season."""
    season_zones = {}
    for season in seasons:
        season_pos = [p for p in positions if p.get("season") == season]
        if season_pos:
            zone = compute_stadium_zone(season_pos)
            if zone:
                season_zones[season] = zone
    return season_zones


def process_calibration_data(
    detector: PlayerPoseDetector,
    stadiums_filter: Optional[List[str]] = None,
) -> dict:
    """Process calibration videos and compute zones.

    Returns dict of stadium_key -> zone.
    """
    # Load metadata to get video paths and handedness info
    if not METADATA_PATH.exists():
        print(f"ERROR: Metadata not found at {METADATA_PATH}")
        print("Run scrape_pitcher_calibration.py first.")
        return {}

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    all_zones = {}
    stadium_positions = defaultdict(list)

    # Group metadata entries by stadium
    stadium_entries = defaultdict(list)
    for key, videos in metadata.items():
        # Key format: "Stadium_Name_2024"
        parts = key.rsplit("_", 1)
        if len(parts) == 2:
            stadium_key = parts[0]
            season = parts[1]
        else:
            stadium_key = key
            season = "unknown"

        if stadiums_filter:
            stadium_name = stadium_key.replace("_", " ")
            if stadium_name not in stadiums_filter:
                continue

        for video in videos:
            video["_season"] = season
            stadium_entries[stadium_key].append(video)

    total_stadiums = len(stadium_entries)
    print(f"\nProcessing {total_stadiums} stadiums...")

    for i, (stadium_key, videos) in enumerate(sorted(stadium_entries.items())):
        stadium_name = stadium_key.replace("_", " ")
        print(f"\n[{i + 1}/{total_stadiums}] {stadium_name} ({len(videos)} videos)")

        positions = []
        for j, video in enumerate(videos):
            p_throws = video.get("p_throws", "?")
            season = video.get("_season", "unknown")

            # Prefer cropped video over raw
            raw_path = video.get("video_path", "")
            cropped = ""
            if raw_path:
                # Derive cropped path: replace pitcher_calibration/ with pitcher_calibration_cropped/
                cropped = raw_path.replace(
                    str(CALIBRATION_DIR), str(CROPPED_DIR)
                ).replace("\\", "/")
                # Also try with backslashes (Windows paths in metadata)
                if not Path(cropped).exists():
                    cropped = str(
                        CROPPED_DIR / Path(raw_path).relative_to(CALIBRATION_DIR)
                    )

            if not cropped or not Path(cropped).exists():
                # Fall back to raw path
                cropped = raw_path
                if not Path(cropped).exists():
                    print(f"  Video not found: {cropped}")
                    continue

            print(f"  [{j + 1}/{len(videos)}] {Path(cropped).name} ({p_throws}HP)")

            video_positions = collect_pitcher_positions(
                detector, cropped, p_throws=p_throws,
            )

            # Tag with season for drift analysis
            for pos in video_positions:
                pos["season"] = season

            positions.extend(video_positions)
            print(f"    Collected {len(video_positions)} positions")

        if not positions:
            print(f"  WARNING: No positions collected for {stadium_name}")
            continue

        # Compute zone
        zone = compute_stadium_zone(positions)
        if zone:
            # Add per-season breakdown for drift detection
            seasons = list(set(p.get("season", "unknown") for p in positions))
            per_season = compute_per_season_zones(positions, seasons)
            if per_season:
                zone["per_season"] = per_season

            all_zones[stadium_key] = zone
            print(f"  Zone: cx={zone['mean_cx']:.3f}±{zone['std_cx']:.3f}, "
                  f"cy={zone['mean_cy']:.3f}±{zone['std_cy']:.3f} "
                  f"({zone['n_samples']} samples)")
        else:
            print(f"  WARNING: Not enough data for zone computation")

        # Save incremental progress
        save_zones(all_zones)

    return all_zones


def process_existing_videos(
    detector: PlayerPoseDetector,
    stadiums_filter: Optional[List[str]] = None,
) -> dict:
    """Process existing 2023_cropped videos (no handedness info available)."""
    if not EXISTING_DIR.exists():
        print(f"ERROR: Existing videos not found at {EXISTING_DIR}")
        return {}

    all_zones = {}

    stadium_dirs = sorted(EXISTING_DIR.iterdir())
    stadium_dirs = [d for d in stadium_dirs if d.is_dir()]

    if stadiums_filter:
        stadium_dirs = [
            d for d in stadium_dirs
            if d.name.replace("_", " ") in stadiums_filter
        ]

    total = len(stadium_dirs)
    print(f"\nProcessing {total} stadiums from existing videos...")

    for i, stadium_dir in enumerate(stadium_dirs):
        stadium_key = stadium_dir.name
        stadium_name = stadium_key.replace("_", " ")
        videos = sorted(stadium_dir.glob("*.mp4"))

        print(f"\n[{i + 1}/{total}] {stadium_name} ({len(videos)} videos)")

        positions = []
        for j, video_path in enumerate(videos):
            print(f"  [{j + 1}/{len(videos)}] {video_path.name}")
            video_positions = collect_pitcher_positions(
                detector, str(video_path), p_throws="?",
            )
            positions.extend(video_positions)
            print(f"    Collected {len(video_positions)} positions")

        if not positions:
            print(f"  WARNING: No positions collected")
            continue

        zone = compute_stadium_zone(positions)
        if zone:
            all_zones[stadium_key] = zone
            print(f"  Zone: cx={zone['mean_cx']:.3f}±{zone['std_cx']:.3f}, "
                  f"cy={zone['mean_cy']:.3f}±{zone['std_cy']:.3f} "
                  f"({zone['n_samples']} samples)")

        save_zones(all_zones)

    return all_zones


def save_zones(zones: dict):
    """Save zones to JSON."""
    ZONES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ZONES_PATH, "w") as f:
        json.dump(zones, f, indent=2)


def generate_report(zones: dict):
    """Generate visual report of pitcher zones across stadiums."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping visual report")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not zones:
        print("No zones to report")
        return

    # --- Plot 1: Scatter of mean pitcher positions per stadium ---
    fig, ax = plt.subplots(figsize=(12, 8))

    names = []
    cx_vals = []
    cy_vals = []
    std_cx_vals = []
    std_cy_vals = []

    for stadium_key, zone in sorted(zones.items()):
        name = stadium_key.replace("_", " ")
        names.append(name)
        cx_vals.append(zone["mean_cx"])
        cy_vals.append(zone["mean_cy"])
        std_cx_vals.append(zone["std_cx"])
        std_cy_vals.append(zone["std_cy"])

    ax.errorbar(
        cx_vals, cy_vals,
        xerr=std_cx_vals, yerr=std_cy_vals,
        fmt="o", markersize=6, capsize=3, alpha=0.7,
    )

    for name, cx, cy in zip(names, cx_vals, cy_vals):
        ax.annotate(
            name, (cx, cy), fontsize=6,
            textcoords="offset points", xytext=(5, 5),
        )

    ax.set_xlabel("cx_norm (horizontal position)")
    ax.set_ylabel("cy_norm (vertical position)")
    ax.set_title("Mean Pitcher Position per Stadium (with std dev)")
    ax.invert_yaxis()  # Image coords: y increases downward
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(REPORT_DIR / "pitcher_positions_all.png"), dpi=150)
    plt.close()
    print(f"  Saved: {REPORT_DIR / 'pitcher_positions_all.png'}")

    # --- Plot 2: RHP vs LHP scatter per stadium ---
    fig, ax = plt.subplots(figsize=(12, 8))

    for stadium_key, zone in sorted(zones.items()):
        name = stadium_key.replace("_", " ")

        if "rhp_cx" in zone:
            ax.scatter(
                zone["rhp_cx"], zone.get("rhp_cy", zone["mean_cy"]),
                c="blue", marker="o", s=40, alpha=0.6, label="RHP" if stadium_key == list(zones.keys())[0] else "",
            )
        if "lhp_cx" in zone:
            ax.scatter(
                zone["lhp_cx"], zone.get("lhp_cy", zone["mean_cy"]),
                c="red", marker="s", s=40, alpha=0.6, label="LHP" if stadium_key == list(zones.keys())[0] else "",
            )

    ax.set_xlabel("cx_norm")
    ax.set_ylabel("cy_norm")
    ax.set_title("RHP vs LHP Pitcher Positions")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(REPORT_DIR / "rhp_vs_lhp.png"), dpi=150)
    plt.close()
    print(f"  Saved: {REPORT_DIR / 'rhp_vs_lhp.png'}")

    # --- Plot 3: Bbox size distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bw_vals = [z["mean_bbox_w"] for z in zones.values()]
    bh_vals = [z["mean_bbox_h"] for z in zones.values()]

    axes[0].hist(bw_vals, bins=20, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Normalized bbox width")
    axes[0].set_title("Pitcher Bbox Width Distribution (per stadium)")

    axes[1].hist(bh_vals, bins=20, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Normalized bbox height")
    axes[1].set_title("Pitcher Bbox Height Distribution (per stadium)")

    plt.tight_layout()
    plt.savefig(str(REPORT_DIR / "bbox_sizes.png"), dpi=150)
    plt.close()
    print(f"  Saved: {REPORT_DIR / 'bbox_sizes.png'}")

    # --- Plot 4: Per-season drift check ---
    drift_data = []
    for stadium_key, zone in zones.items():
        per_season = zone.get("per_season", {})
        if len(per_season) >= 2:
            seasons = sorted(per_season.keys())
            for s in seasons:
                drift_data.append({
                    "stadium": stadium_key.replace("_", " "),
                    "season": s,
                    "cx": per_season[s]["mean_cx"],
                    "cy": per_season[s]["mean_cy"],
                })

    if drift_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for stadium_key, zone in zones.items():
            per_season = zone.get("per_season", {})
            if len(per_season) >= 2:
                seasons = sorted(per_season.keys())
                cx_series = [per_season[s]["mean_cx"] for s in seasons]
                cy_series = [per_season[s]["mean_cy"] for s in seasons]

                axes[0].plot(seasons, cx_series, "o-", alpha=0.5, markersize=4)
                axes[1].plot(seasons, cy_series, "o-", alpha=0.5, markersize=4)

        axes[0].set_xlabel("Season")
        axes[0].set_ylabel("mean_cx")
        axes[0].set_title("Horizontal Position Drift by Season")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Season")
        axes[1].set_ylabel("mean_cy")
        axes[1].set_title("Vertical Position Drift by Season")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(REPORT_DIR / "season_drift.png"), dpi=150)
        plt.close()
        print(f"  Saved: {REPORT_DIR / 'season_drift.png'}")

    # --- Summary stats ---
    print(f"\n{'=' * 60}")
    print("ZONE SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Stadium':30s} {'cx':>7s} {'cy':>7s} {'std_cx':>7s} {'std_cy':>7s} {'n':>6s}")
    print("-" * 60)

    for stadium_key in sorted(zones.keys()):
        zone = zones[stadium_key]
        name = stadium_key.replace("_", " ")[:28]
        print(f"{name:30s} {zone['mean_cx']:7.3f} {zone['mean_cy']:7.3f} "
              f"{zone['std_cx']:7.3f} {zone['std_cy']:7.3f} {zone['n_samples']:6d}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate per-stadium pitcher zones from video data"
    )
    parser.add_argument(
        "--stadiums", type=str,
        help="Comma-separated list of stadium names",
    )
    parser.add_argument(
        "--use-existing", action="store_true",
        help="Use existing 2023_cropped videos instead of calibration data",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Generate visual report from existing zones (no detection)",
    )
    args = parser.parse_args()

    stadiums_filter = None
    if args.stadiums:
        stadiums_filter = [s.strip() for s in args.stadiums.split(",")]

    # Report only mode
    if args.report_only:
        if not ZONES_PATH.exists():
            print(f"ERROR: No zones file at {ZONES_PATH}")
            return
        with open(ZONES_PATH, "r") as f:
            zones = json.load(f)
        print(f"Loaded {len(zones)} zones from {ZONES_PATH}")
        generate_report(zones)
        return

    # Initialize detector (only needs YOLO, not pose)
    print("Initializing detector...")
    detector = PlayerPoseDetector()

    # Process videos
    if args.use_existing:
        zones = process_existing_videos(detector, stadiums_filter)
    else:
        zones = process_calibration_data(detector, stadiums_filter)

    if not zones:
        print("\nNo zones computed. Check video paths and data.")
        detector.cleanup()
        return

    # Save final zones
    save_zones(zones)
    print(f"\nZones saved to: {ZONES_PATH}")

    # Generate visual report
    print("\nGenerating visual report...")
    generate_report(zones)

    detector.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
