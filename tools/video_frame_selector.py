#!/usr/bin/env python
"""
Video Frame Selector for Baseball Labeling.

Scrub through videos and mark time ranges where the ball is visible.
Extracts frames from those ranges for labeling.

Controls:
    Space       : Play/Pause
    Left/Right  : Step 1 frame
    Shift+Left/Right : Step 10 frames
    [ or I      : Mark segment START
    ] or O      : Mark segment END (saves segment)
    S           : Save all segments and extract frames
    N           : Next video
    Q           : Quit

Usage:
    python tools/video_frame_selector.py
"""

import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import random

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
VIDEO_DIR = PROJECT_ROOT / "data/videos"
OUTPUT_DIR = PROJECT_ROOT / "data/labels/baseball/frames"


class VideoFrameSelector:
    def __init__(self, num_frames_target: int = 200):
        self.num_frames_target = num_frames_target
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all videos
        self.videos = list(VIDEO_DIR.rglob("*.mp4"))
        random.shuffle(self.videos)
        self.video_idx = 0

        # Current video state
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.playing = False

        # Segments for current video
        self.segments = []  # List of (start, end) tuples
        self.segment_start = None  # Current segment start (if marking)

        # All extracted frames info
        self.extracted_count = 0
        self.segments_file = self.output_dir / "segments_info.json"
        self.all_segments = self._load_segments()

    def _load_segments(self):
        if self.segments_file.exists():
            with open(self.segments_file) as f:
                return json.load(f)
        return {"videos": {}, "total_frames": 0}

    def _save_segments(self):
        with open(self.segments_file, "w") as f:
            json.dump(self.all_segments, f, indent=2)

    def _load_video(self, video_path: Path):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.current_frame = 0
        self.segments = []
        self.segment_start = None
        self.playing = False

        # Check if already processed
        video_name = video_path.name
        if video_name in self.all_segments["videos"]:
            self.segments = [tuple(s) for s in self.all_segments["videos"][video_name]]
            print(f"  Loaded {len(self.segments)} existing segments")

    def _get_frame(self, frame_idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_idx
            return frame
        return None

    def _draw_ui(self, frame, video_name: str):
        h, w = frame.shape[:2]
        display = frame.copy()

        # Timeline bar at bottom
        bar_height = 40
        bar_y = h - bar_height - 10

        # Background
        cv2.rectangle(display, (10, bar_y), (w - 10, bar_y + bar_height), (40, 40, 40), -1)

        # Draw segments on timeline
        for start, end in self.segments:
            x1 = int(10 + (start / self.total_frames) * (w - 20))
            x2 = int(10 + (end / self.total_frames) * (w - 20))
            cv2.rectangle(display, (x1, bar_y + 5), (x2, bar_y + bar_height - 5), (0, 255, 0), -1)

        # Current segment being marked
        if self.segment_start is not None:
            x1 = int(10 + (self.segment_start / self.total_frames) * (w - 20))
            x2 = int(10 + (self.current_frame / self.total_frames) * (w - 20))
            cv2.rectangle(display, (x1, bar_y + 5), (x2, bar_y + bar_height - 5), (0, 255, 255), -1)

        # Playhead
        playhead_x = int(10 + (self.current_frame / self.total_frames) * (w - 20))
        cv2.line(display, (playhead_x, bar_y), (playhead_x, bar_y + bar_height), (0, 0, 255), 2)

        # Top info bar
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)

        # Video name
        cv2.putText(display, f"Video: {video_name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Frame info
        time_sec = self.current_frame / self.fps
        cv2.putText(display, f"Frame: {self.current_frame}/{self.total_frames} ({time_sec:.1f}s)",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Segment count
        total_segment_frames = sum(e - s for s, e in self.segments)
        cv2.putText(display, f"Segments: {len(self.segments)} ({total_segment_frames} frames)",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Status
        if self.segment_start is not None:
            status = f"MARKING from frame {self.segment_start}... Press ] to end"
            color = (0, 255, 255)
        elif self.playing:
            status = "PLAYING - Space to pause"
            color = (0, 255, 0)
        else:
            status = "PAUSED - [ to start segment, Space to play"
            color = (200, 200, 200)

        cv2.putText(display, status, (w - 500, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Extracted count
        cv2.putText(display, f"Total extracted: {self.extracted_count}/{self.num_frames_target}",
                    (w - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

        # Controls hint
        controls = "[/] mark segment | Space play | Arrows step | S save | N next | Q quit"
        cv2.putText(display, controls, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return display

    def _extract_frames_from_segments(self, video_path: Path):
        """Extract frames from marked segments."""
        if not self.segments:
            return 0

        video_name = video_path.stem
        extracted = 0

        # Calculate how many frames to extract from each segment
        total_segment_frames = sum(e - s for s, e in self.segments)

        for seg_idx, (start, end) in enumerate(self.segments):
            seg_length = end - start
            # Extract every frame from short segments, sample from long ones
            if seg_length <= 10:
                frames_to_extract = list(range(start, end))
            else:
                # Extract ~1 frame per 3 frames
                step = max(1, seg_length // (seg_length // 3))
                frames_to_extract = list(range(start, end, step))

            for frame_idx in frames_to_extract:
                frame = self._get_frame(frame_idx)
                if frame is not None:
                    # Save frame
                    frame_name = f"ball_{self.extracted_count:04d}_{video_name}_f{frame_idx}.png"
                    frame_path = self.output_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    extracted += 1
                    self.extracted_count += 1

        return extracted

    def run(self):
        print("="*60)
        print("VIDEO FRAME SELECTOR")
        print("="*60)
        print(f"Found {len(self.videos)} videos")
        print(f"Target: {self.num_frames_target} frames")
        print("="*60)
        print("\nControls:")
        print("  Space       : Play/Pause")
        print("  Left/Right  : Step 1 frame")
        print("  Shift+arrows: Step 10 frames")
        print("  [ or I      : Mark segment START")
        print("  ] or O      : Mark segment END")
        print("  S           : Save & extract frames")
        print("  N           : Next video")
        print("  Q           : Quit")
        print("="*60)

        cv2.namedWindow("Frame Selector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame Selector", 1280, 800)

        while self.video_idx < len(self.videos) and self.extracted_count < self.num_frames_target:
            video_path = self.videos[self.video_idx]
            print(f"\n[{self.video_idx + 1}/{len(self.videos)}] Loading: {video_path.name}")

            self._load_video(video_path)
            frame = self._get_frame(0)

            if frame is None:
                self.video_idx += 1
                continue

            while True:
                # Get current frame
                if self.playing:
                    frame = self._get_frame(min(self.current_frame + 1, self.total_frames - 1))
                    if self.current_frame >= self.total_frames - 1:
                        self.playing = False

                if frame is None:
                    frame = self._get_frame(self.current_frame)

                # Draw UI
                display = self._draw_ui(frame, video_path.name)
                cv2.imshow("Frame Selector", display)

                # Handle input
                key = cv2.waitKey(30 if self.playing else 0) & 0xFF

                if key == ord('q') or key == 27:  # Q or Escape
                    print("\nQuitting...")
                    cv2.destroyAllWindows()
                    return

                elif key == ord(' '):  # Space - play/pause
                    self.playing = not self.playing

                elif key == 81 or key == 2:  # Left arrow
                    self.playing = False
                    frame = self._get_frame(max(0, self.current_frame - 1))

                elif key == 83 or key == 3:  # Right arrow
                    self.playing = False
                    frame = self._get_frame(min(self.total_frames - 1, self.current_frame + 1))

                elif key == ord('[') or key == ord('i'):  # Start segment
                    self.segment_start = self.current_frame
                    print(f"  Segment start: frame {self.segment_start}")

                elif key == ord(']') or key == ord('o'):  # End segment
                    if self.segment_start is not None:
                        end = self.current_frame
                        if end > self.segment_start:
                            self.segments.append((self.segment_start, end))
                            print(f"  Segment saved: {self.segment_start} -> {end} ({end - self.segment_start} frames)")
                        self.segment_start = None

                elif key == ord('s'):  # Save and extract
                    if self.segments:
                        print(f"\n  Extracting frames from {len(self.segments)} segments...")
                        extracted = self._extract_frames_from_segments(video_path)
                        print(f"  Extracted {extracted} frames")

                        # Save segments info
                        self.all_segments["videos"][video_path.name] = self.segments
                        self.all_segments["total_frames"] = self.extracted_count
                        self._save_segments()

                        if self.extracted_count >= self.num_frames_target:
                            print(f"\n  Target reached! ({self.extracted_count} frames)")
                            break
                    else:
                        print("  No segments to extract")

                elif key == ord('n'):  # Next video
                    # Save current segments first
                    if self.segments:
                        self.all_segments["videos"][video_path.name] = self.segments
                        self._save_segments()
                    break

                # Handle shift+arrows (step 10 frames)
                # Note: This depends on terminal/OS, may need adjustment

            self.video_idx += 1

        cv2.destroyAllWindows()
        print(f"\n{'='*60}")
        print(f"DONE - Extracted {self.extracted_count} frames")
        print(f"Frames saved to: {self.output_dir}")
        print(f"\nNow run the labeler to annotate these frames:")
        print(f"  python tools/baseball_labeler.py")
        print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Select video frames for baseball labeling")
    parser.add_argument("--target", type=int, default=200,
                        help="Target number of frames to extract")
    args = parser.parse_args()

    selector = VideoFrameSelector(num_frames_target=args.target)
    selector.run()


if __name__ == "__main__":
    main()
