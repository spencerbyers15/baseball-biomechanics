#!/usr/bin/env python
"""
Viewer to compare ffmpeg scene detection against hand-labeled cuts.

Shows video with timeline displaying both:
  - ffmpeg-detected cuts (cyan markers)
  - hand-labeled cuts (green markers)

Controls:
    SPACE       : Play / Pause
    RIGHT / D   : Step forward 1 frame
    LEFT / A    : Step backward 1 frame
    UP          : Jump forward 30 frames
    DOWN        : Jump backward 30 frames
    PAGE_DOWN   : Jump forward 60 frames
    PAGE_UP     : Jump backward 60 frames
    +/-         : Adjust ffmpeg threshold (re-detects)
    N           : Next video
    B           : Back to previous video
    Q / ESC     : Quit
    Mouse click : Jump to position on timeline

Usage:
    python tools/view_ffmpeg_cuts.py
    python tools/view_ffmpeg_cuts.py --threshold 0.3
"""

import cv2
import json
import argparse
import subprocess
import re
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
sys.path.insert(0, str(PROJECT_ROOT))

LABELS_PATH = PROJECT_ROOT / "data/labels/scene_cuts/scene_cut_labels.json"
METADATA_PATH = PROJECT_ROOT / "data/pitcher_calibration_metadata.json"

# Layout
WINDOW_W = 1400
WINDOW_H = 750
INFO_BAR_H = 80
TIMELINE_H = 60
BORDER = 10

# Colors (BGR)
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 140)
COLOR_GREEN = (0, 200, 0)
COLOR_CYAN = (220, 200, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_PLAYHEAD = (0, 0, 255)
COLOR_FFMPEG_CUT = (220, 200, 0)   # cyan — ffmpeg detected
COLOR_GT_CUT = (0, 200, 0)         # green — hand labeled
COLOR_MATCH = (0, 255, 0)          # bright green — matched

# Windows key codes
KEY_LEFT = 0x250000
KEY_RIGHT = 0x270000
KEY_UP = 0x260000
KEY_DOWN = 0x280000
KEY_PAGEUP = 0x210000
KEY_PAGEDOWN = 0x220000


def ffmpeg_scene_cuts(video_path: str, threshold: float) -> list:
    """Detect scene cuts using ffmpeg's built-in scene filter. Returns list of timestamps."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    times = []
    for line in result.stderr.split("\n"):
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            times.append(float(m.group(1)))
    # Filter too-close cuts (< 0.5s apart)
    filtered = []
    for t in times:
        if not filtered or (t - filtered[-1]) >= 0.5:
            filtered.append(t)
    return filtered


def select_videos(metadata_path: Path) -> list:
    """Select 1 video per stadium per season (same as labeler)."""
    with open(metadata_path) as f:
        metadata = json.load(f)

    stadiums = {}
    for key in metadata:
        parts = key.rsplit("_", 1)
        if parts[-1] in ("2023", "2024", "2025"):
            stadium, season = parts[0], parts[-1]
        else:
            continue
        if stadium not in stadiums:
            stadiums[stadium] = {}
        if metadata[key]:
            stadiums[stadium][season] = metadata[key]

    selected = []
    proj_str = str(PROJECT_ROOT).replace("\\", "/")
    for stadium in sorted(stadiums.keys()):
        for season in ["2023", "2024", "2025"]:
            if season in stadiums[stadium] and stadiums[stadium][season]:
                video = stadiums[stadium][season][0]
                vpath = video["video_path"].replace("\\", "/")
                if vpath.startswith(proj_str):
                    vpath = vpath[len(proj_str):].lstrip("/")
                selected.append({"path": vpath, "stadium": stadium, "season": season})

    return selected


class FFmpegCutViewer:
    def __init__(self, videos: list, gt_labels: dict, threshold: float = 0.25):
        self.videos = videos
        self.gt_labels = gt_labels
        self.threshold = threshold

        self.video_idx = 0
        self.cap = None
        self.fps = 60.0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self.cached_frame = None
        self.cached_frame_idx = -1

        self.ffmpeg_cuts_sec = []    # ffmpeg-detected cut times (seconds)
        self.gt_cuts_sec = []        # hand-labeled cut times (seconds)
        self.duration = 0.0

        self.window_name = "FFmpeg Scene Cut Viewer"

    def _timeline_rect(self):
        return (BORDER, WINDOW_H - TIMELINE_H, WINDOW_W - 2 * BORDER, TIMELINE_H)

    def _video_rect(self):
        return (0, INFO_BAR_H, WINDOW_W, WINDOW_H - INFO_BAR_H - TIMELINE_H)

    def _frame_from_x(self, x: int) -> int:
        tx, _, tw, _ = self._timeline_rect()
        frac = max(0.0, min(1.0, (x - tx) / tw))
        return int(frac * max(1, self.total_frames - 1))

    def _load_video(self, idx: int) -> bool:
        if self.cap:
            self.cap.release()

        v = self.videos[idx]
        full_path = str(PROJECT_ROOT / v["path"])

        self.cap = cv2.VideoCapture(full_path)
        if not self.cap.isOpened():
            print(f"  ERROR: Cannot open {full_path}")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.current_frame = 0
        self.playing = False
        self.cached_frame = None
        self.cached_frame_idx = -1

        # Get hand-labeled cuts
        entry = self.gt_labels.get("videos", {}).get(v["path"], {})
        gt_frames = entry.get("cut_frames", [])
        gt_fps = entry.get("fps", self.fps)
        self.gt_cuts_sec = [f / gt_fps for f in gt_frames]

        # Detect with ffmpeg
        print(f"  Running ffmpeg scene detection (threshold={self.threshold})...", end="", flush=True)
        self.ffmpeg_cuts_sec = ffmpeg_scene_cuts(full_path, self.threshold)
        print(f" {len(self.ffmpeg_cuts_sec)} cuts")

        return True

    def _redetect(self):
        """Re-run ffmpeg with current threshold."""
        v = self.videos[self.video_idx]
        full_path = str(PROJECT_ROOT / v["path"])
        print(f"  Re-detecting at threshold={self.threshold:.2f}...", end="", flush=True)
        self.ffmpeg_cuts_sec = ffmpeg_scene_cuts(full_path, self.threshold)
        print(f" {len(self.ffmpeg_cuts_sec)} cuts")

    def _get_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, min(self.total_frames - 1, frame_idx))
        if self.cached_frame_idx == frame_idx and self.cached_frame is not None:
            return self.cached_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.cached_frame = frame
            self.cached_frame_idx = frame_idx
            self.current_frame = frame_idx
            return frame
        return self.cached_frame

    def _match_cuts(self, tolerance_sec: float = 0.15):
        """Match ffmpeg cuts to GT cuts. Returns (matched_ff, matched_gt, unmatched_ff, unmatched_gt)."""
        matched_ff = set()
        matched_gt = set()
        for gi, gt in enumerate(self.gt_cuts_sec):
            best_dist = tolerance_sec + 1
            best_fi = -1
            for fi, ff in enumerate(self.ffmpeg_cuts_sec):
                if fi in matched_ff:
                    continue
                d = abs(ff - gt)
                if d < best_dist:
                    best_dist = d
                    best_fi = fi
            if best_fi >= 0 and best_dist <= tolerance_sec:
                matched_ff.add(best_fi)
                matched_gt.add(gi)
        unmatched_ff = [i for i in range(len(self.ffmpeg_cuts_sec)) if i not in matched_ff]
        unmatched_gt = [i for i in range(len(self.gt_cuts_sec)) if i not in matched_gt]
        return matched_ff, matched_gt, unmatched_ff, unmatched_gt

    def _draw_canvas(self) -> np.ndarray:
        canvas = np.full((WINDOW_H, WINDOW_W, 3), COLOR_BG[0], dtype=np.uint8)
        canvas[:, :] = COLOR_BG

        v = self.videos[self.video_idx]
        frame = self._get_frame(self.current_frame)
        matched_ff, matched_gt, unmatched_ff, unmatched_gt = self._match_cuts()

        # --- Info bar ---
        cv2.rectangle(canvas, (0, 0), (WINDOW_W, INFO_BAR_H), (20, 20, 20), -1)

        time_sec = self.current_frame / self.fps

        line1 = (f"Video {self.video_idx + 1}/{len(self.videos)} | "
                 f"{v['stadium']} | {v['season']} | "
                 f"Frame {self.current_frame}/{self.total_frames - 1} "
                 f"({time_sec:.1f}s / {self.duration:.1f}s)")
        cv2.putText(canvas, line1, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT, 1)

        # Stats row
        tp = len(matched_ff)
        fp = len(unmatched_ff)
        fn = len(unmatched_gt)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        stats = (f"Threshold: {self.threshold:.2f} | "
                 f"FFmpeg: {len(self.ffmpeg_cuts_sec)} cuts | "
                 f"GT: {len(self.gt_cuts_sec)} cuts | "
                 f"TP={tp} FP={fp} FN={fn} | "
                 f"P={prec:.2f} R={rec:.2f} F1={f1:.2f}")
        cv2.putText(canvas, stats, (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)

        # Controls + legend
        controls = "[+/-]threshold [SPACE]play [Arrows]nav [N]ext [B]ack [Q]uit"
        cv2.putText(canvas, controls, (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1)

        # Legend on right side
        cv2.putText(canvas, "CYAN=ffmpeg", (WINDOW_W - 280, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_FFMPEG_CUT, 1)
        cv2.putText(canvas, "GREEN=GT", (WINDOW_W - 150, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GT_CUT, 1)

        status = "PLAYING" if self.playing else "PAUSED"
        scolor = COLOR_GREEN if self.playing else COLOR_YELLOW
        cv2.putText(canvas, status, (WINDOW_W - 280, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, scolor, 1)

        # --- Video frame ---
        vx, vy, vw, vh = self._video_rect()
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(vw / fw, vh / fh)
            new_w, new_h = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            ox = vx + (vw - new_w) // 2
            oy = vy + (vh - new_h) // 2
            canvas[oy:oy + new_h, ox:ox + new_w] = resized

        # --- Timeline ---
        tx, ty, tw, th = self._timeline_rect()
        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (50, 50, 50), -1)

        # Draw two rows: top half = GT cuts, bottom half = ffmpeg cuts
        mid_y = ty + th // 2
        cv2.line(canvas, (tx, mid_y), (tx + tw, mid_y), (80, 80, 80), 1)

        # GT label
        cv2.putText(canvas, "GT", (tx - BORDER, ty + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_GT_CUT, 1)
        cv2.putText(canvas, "FF", (tx - BORDER, mid_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COLOR_FFMPEG_CUT, 1)

        # GT cuts (top half) — green, with matched ones brighter
        for gi, t in enumerate(self.gt_cuts_sec):
            px = tx + int(t / self.duration * tw)
            color = COLOR_MATCH if gi in matched_gt else COLOR_RED
            cv2.line(canvas, (px, ty + 2), (px, mid_y - 1), color, 2)

        # FFmpeg cuts (bottom half) — cyan, with unmatched ones red
        for fi, t in enumerate(self.ffmpeg_cuts_sec):
            px = tx + int(t / self.duration * tw)
            color = COLOR_FFMPEG_CUT if fi in matched_ff else COLOR_RED
            cv2.line(canvas, (px, mid_y + 1), (px, ty + th - 2), color, 2)

        # Playhead
        px = tx + int(self.current_frame / max(1, self.total_frames - 1) * tw)
        cv2.line(canvas, (px, ty), (px, ty + th), COLOR_PLAYHEAD, 2)

        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        tx, ty, tw, th = self._timeline_rect()
        if tx <= x <= tx + tw and ty <= y <= ty + th:
            self.playing = False
            self.current_frame = self._frame_from_x(x)

    def run(self):
        print("=" * 60)
        print("FFMPEG SCENE CUT VIEWER")
        print("=" * 60)
        print(f"Threshold: {self.threshold}")
        print(f"Videos: {len(self.videos)}")
        print()
        print("Timeline: top row = GT (green=matched, red=missed)")
        print("          bottom row = ffmpeg (cyan=matched, red=FP)")
        print()
        print("Keys: [+/-] adjust threshold  [SPACE] play  [N]ext [B]ack [Q]uit")
        print("=" * 60)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, WINDOW_W, WINDOW_H)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while self.video_idx < len(self.videos):
            v = self.videos[self.video_idx]
            print(f"\n[{self.video_idx + 1}/{len(self.videos)}] "
                  f"{v['stadium']} / {v['season']}")

            if not self._load_video(self.video_idx):
                self.video_idx += 1
                continue

            while True:
                canvas = self._draw_canvas()
                cv2.imshow(self.window_name, canvas)

                wait_ms = 16 if self.playing else 0
                key = cv2.waitKeyEx(max(1, wait_ms))

                if self.playing:
                    self.current_frame = min(self.current_frame + 2, self.total_frames - 1)
                    if self.current_frame >= self.total_frames - 1:
                        self.playing = False

                if key == -1:
                    continue

                kl = key & 0xFF

                if kl == ord("q") or kl == 27:
                    cv2.destroyAllWindows()
                    return

                elif kl == ord(" "):
                    self.playing = not self.playing

                elif key == KEY_RIGHT or kl == ord("d"):
                    self.playing = False
                    self.current_frame = min(self.current_frame + 1, self.total_frames - 1)

                elif key == KEY_LEFT or kl == ord("a"):
                    self.playing = False
                    self.current_frame = max(self.current_frame - 1, 0)

                elif key == KEY_UP:
                    self.playing = False
                    self.current_frame = min(self.current_frame + 30, self.total_frames - 1)

                elif key == KEY_DOWN:
                    self.playing = False
                    self.current_frame = max(self.current_frame - 30, 0)

                elif key == KEY_PAGEDOWN:
                    self.playing = False
                    self.current_frame = min(self.current_frame + 60, self.total_frames - 1)

                elif key == KEY_PAGEUP:
                    self.playing = False
                    self.current_frame = max(self.current_frame - 60, 0)

                elif kl == ord("+") or kl == ord("="):
                    self.threshold = min(0.9, self.threshold + 0.05)
                    self._redetect()

                elif kl == ord("-") or kl == ord("_"):
                    self.threshold = max(0.05, self.threshold - 0.05)
                    self._redetect()

                elif kl == ord("n"):
                    self.video_idx += 1
                    break

                elif kl == ord("b"):
                    if self.video_idx > 0:
                        self.video_idx -= 1
                    break

        cv2.destroyAllWindows()
        print("\nDone — viewed all videos.")


def main():
    parser = argparse.ArgumentParser(description="View ffmpeg scene cuts vs ground truth")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Initial ffmpeg scene threshold (default: 0.25)")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    args = parser.parse_args()

    videos = select_videos(args.metadata)
    print(f"Selected {len(videos)} videos")

    gt_labels = {}
    if args.labels.exists():
        with open(args.labels) as f:
            gt_labels = json.load(f)

    viewer = FFmpegCutViewer(videos, gt_labels, threshold=args.threshold)
    viewer.run()


if __name__ == "__main__":
    main()
