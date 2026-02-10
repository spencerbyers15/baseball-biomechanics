#!/usr/bin/env python
"""
Interactive scene cut + segment classification labeler.

Two tasks in one tool:
  1. Correct auto-detected cuts (threshold 0.25):
     - M: mark a real cut the algorithm missed (FN)
     - X: toggle an auto-detected cut as false positive (FP)
  2. Classify each segment between cuts:
     - 1: main_angle (the standard pitching broadcast view)
     - 2: other (replay, close-up, dugout, etc.)

Controls:
    SPACE       : Play / Pause
    RIGHT / D   : Step forward 1 frame
    LEFT / A    : Step backward 1 frame
    UP          : Jump forward 30 frames
    DOWN        : Jump backward 30 frames
    PAGE_DOWN   : Jump forward 60 frames
    PAGE_UP     : Jump backward 60 frames
    M           : Mark missed cut at current frame (FN)
    X           : Toggle nearest auto-cut as false positive (FP)
    1           : Label current segment as main_angle
    2           : Label current segment as other
    N           : Save + next video
    B           : Back to previous video
    Q / ESC     : Save + quit
    Mouse click : Jump to position (on graph or timeline)

Usage:
    python tools/label_scene_cuts.py
    python tools/label_scene_cuts.py --threshold 0.3
"""

import cv2
import json
import argparse
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("F:/Claude_Projects/baseball-biomechanics")
sys.path.insert(0, str(PROJECT_ROOT))

METADATA_PATH = PROJECT_ROOT / "data/pitcher_calibration_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "data/labels/scene_cuts/scene_cut_labels.json"

# Layout
WINDOW_W = 1400
WINDOW_H = 900
INFO_BAR_H = 80
GRAPH_H = 120
TIMELINE_H = 40
BORDER = 10

# Colors (BGR)
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 140)
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_CYAN = (220, 200, 0)
COLOR_PLAYHEAD = (0, 0, 255)
COLOR_AUTO_CUT = (0, 200, 0)       # green — accepted auto-detected
COLOR_FP_CUT = (0, 0, 180)         # red — false positive (removed)
COLOR_FN_CUT = (220, 180, 0)       # cyan — user-added missed cut
COLOR_GRAPH_LINE = (200, 180, 50)
COLOR_THRESHOLD = (100, 100, 200)
COLOR_GRAPH_BG = (40, 40, 40)

# Segment colors on timeline
COLOR_MAIN_ANGLE = (0, 160, 0)     # green block
COLOR_OTHER = (0, 0, 160)          # red block
COLOR_UNLABELED = (80, 80, 80)     # gray block

# Windows key codes via cv2.waitKeyEx
KEY_LEFT = 0x250000
KEY_RIGHT = 0x270000
KEY_UP = 0x260000
KEY_DOWN = 0x280000
KEY_PAGEUP = 0x210000
KEY_PAGEDOWN = 0x220000


def select_videos(metadata_path: Path) -> list:
    """Select 1 video per stadium per season."""
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


def load_labels(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"metadata": {"total_videos": 0, "labeled_videos": 0, "total_cuts": 0}, "videos": {}}


def save_labels(path: Path, labels: dict):
    labeled = sum(1 for v in labels["videos"].values() if v.get("status") == "labeled")
    total_cuts = sum(len(v.get("cut_frames", []))
                     for v in labels["videos"].values() if v.get("status") == "labeled")
    labels["metadata"]["labeled_videos"] = labeled
    labels["metadata"]["total_cuts"] = total_cuts
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


def compute_all_diffs(video_path: str) -> np.ndarray:
    """Precompute histogram_diff for every consecutive frame pair."""
    from src.filtering.scene_cropper import compute_histogram_diff

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diffs = np.zeros(total, dtype=np.float32)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return diffs

    for i in range(1, total):
        ret, curr = cap.read()
        if not ret:
            break
        diffs[i] = compute_histogram_diff(prev, curr)
        prev = curr

    cap.release()
    return diffs


def auto_detect_cuts(diffs: np.ndarray, threshold: float, fps: float,
                     min_segment_duration: float = 0.5) -> list:
    """Threshold the diff scores to get auto-detected cut frames."""
    min_seg_frames = int(min_segment_duration * fps)
    cuts = []
    for i in range(1, len(diffs)):
        if diffs[i] > threshold:
            if not cuts or (i - cuts[-1]) >= min_seg_frames:
                cuts.append(i)
    return cuts


class SceneCutLabeler:
    def __init__(self, videos: list, labels: dict, output_path: Path,
                 threshold: float = 0.25, min_segment_duration: float = 0.5):
        self.videos = videos
        self.labels = labels
        self.output_path = output_path
        self.threshold = threshold
        self.min_segment_duration = min_segment_duration

        # Find starting index — first unlabeled
        self.video_idx = 0
        for i, v in enumerate(self.videos):
            entry = self.labels["videos"].get(v["path"], {})
            if entry.get("status") != "labeled":
                self.video_idx = i
                break

        # Video state
        self.cap = None
        self.fps = 60.0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self.cached_frame = None
        self.cached_frame_idx = -1

        self.diffs = None

        # Cut state per video
        self.auto_detected = []
        self.false_positives = set()
        self.false_negatives = []

        # Segment labels: list parallel to segments derived from _final_cuts()
        # Each entry is "main_angle", "other", or None (unlabeled)
        self.segment_labels = []

        self.window_name = "Scene Cut Labeler"

    def _graph_rect(self):
        video_h = WINDOW_H - INFO_BAR_H - GRAPH_H - TIMELINE_H
        return (BORDER, INFO_BAR_H + video_h, WINDOW_W - 2 * BORDER, GRAPH_H)

    def _timeline_rect(self):
        return (BORDER, WINDOW_H - TIMELINE_H, WINDOW_W - 2 * BORDER, TIMELINE_H)

    def _video_rect(self):
        return (0, INFO_BAR_H, WINDOW_W, WINDOW_H - INFO_BAR_H - GRAPH_H - TIMELINE_H)

    def _frame_from_x(self, x: int) -> int:
        gx, _, gw, _ = self._graph_rect()
        frac = max(0.0, min(1.0, (x - gx) / gw))
        return int(frac * max(1, self.total_frames - 1))

    def _final_cuts(self) -> list:
        """Compute final ground truth cuts: auto - FP + FN."""
        accepted = [f for f in self.auto_detected if f not in self.false_positives]
        return sorted(set(accepted + self.false_negatives))

    def _get_segments(self) -> list:
        """Return list of (start_frame, end_frame) segments from final cuts."""
        cuts = self._final_cuts()
        boundaries = [0] + cuts + [self.total_frames]
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    def _current_segment_idx(self) -> int:
        """Return index of the segment the playhead is in."""
        segments = self._get_segments()
        for i, (start, end) in enumerate(segments):
            if start <= self.current_frame < end:
                return i
        return len(segments) - 1

    def _sync_segment_labels(self):
        """Ensure segment_labels list matches current segment count."""
        n_segments = len(self._get_segments())
        if len(self.segment_labels) < n_segments:
            self.segment_labels.extend([None] * (n_segments - len(self.segment_labels)))
        elif len(self.segment_labels) > n_segments:
            self.segment_labels = self.segment_labels[:n_segments]

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
        self.current_frame = 0
        self.playing = False
        self.cached_frame = None
        self.cached_frame_idx = -1

        # Check for existing labels (resume)
        entry = self.labels["videos"].get(v["path"], {})
        if entry.get("status") == "labeled":
            self.auto_detected = list(entry.get("auto_detected", []))
            self.false_positives = set(entry.get("false_positives", []))
            self.false_negatives = list(entry.get("false_negatives", []))
            self.segment_labels = list(entry.get("segment_labels", []))
            print(f"  Resuming — {len(self.auto_detected)} auto, "
                  f"{len(self.false_positives)} FP, {len(self.false_negatives)} FN")
        else:
            self.auto_detected = []
            self.false_positives = set()
            self.false_negatives = []
            self.segment_labels = []

        # Precompute diffs
        print(f"  Computing diffs for {self.total_frames} frames...", end="", flush=True)
        self.diffs = compute_all_diffs(full_path)
        print(f" done (max={self.diffs.max():.3f})")

        # Auto-detect if not resuming
        if entry.get("status") != "labeled":
            self.auto_detected = auto_detect_cuts(
                self.diffs, self.threshold, self.fps, self.min_segment_duration)
            print(f"  Auto-detected {len(self.auto_detected)} cuts at threshold {self.threshold}")

        self._sync_segment_labels()
        return True

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

    def _save_current_video(self):
        v = self.videos[self.video_idx]
        final = self._final_cuts()
        self._sync_segment_labels()
        self.labels["videos"][v["path"]] = {
            "stadium": v["stadium"],
            "season": v["season"],
            "fps": self.fps,
            "total_frames": self.total_frames,
            "threshold": self.threshold,
            "status": "labeled",
            "auto_detected": sorted(self.auto_detected),
            "false_positives": sorted(self.false_positives),
            "false_negatives": sorted(self.false_negatives),
            "cut_frames": final,
            "segment_labels": self.segment_labels,
        }
        save_labels(self.output_path, self.labels)

    def _draw_canvas(self) -> np.ndarray:
        canvas = np.full((WINDOW_H, WINDOW_W, 3), COLOR_BG[0], dtype=np.uint8)
        canvas[:, :] = COLOR_BG

        v = self.videos[self.video_idx]
        frame = self._get_frame(self.current_frame)
        segments = self._get_segments()
        self._sync_segment_labels()
        cur_seg_idx = self._current_segment_idx()

        # --- Info bar ---
        cv2.rectangle(canvas, (0, 0), (WINDOW_W, INFO_BAR_H), (20, 20, 20), -1)

        time_sec = self.current_frame / self.fps
        total_sec = self.total_frames / self.fps
        labeled_count = sum(1 for vv in self.labels["videos"].values()
                            if vv.get("status") == "labeled")
        final = self._final_cuts()

        line1 = (f"Video {self.video_idx + 1}/{len(self.videos)} | "
                 f"{v['stadium']} | {v['season']} | "
                 f"Frame {self.current_frame}/{self.total_frames - 1} "
                 f"({time_sec:.1f}s / {total_sec:.1f}s)")
        cv2.putText(canvas, line1, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT, 1)

        # Counts row
        x_pos = 10
        y_pos = 44
        cv2.putText(canvas, f"Auto:{len(self.auto_detected)}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_AUTO_CUT, 1)
        x_pos += 90
        cv2.putText(canvas, f"FP(X):{len(self.false_positives)}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_FP_CUT, 1)
        x_pos += 100
        cv2.putText(canvas, f"FN(M):{len(self.false_negatives)}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_FN_CUT, 1)
        x_pos += 110
        cv2.putText(canvas, f"Cuts:{len(final)}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GREEN, 1)
        x_pos += 80
        n_segs = len(segments)
        n_classified = sum(1 for sl in self.segment_labels[:n_segs] if sl is not None)
        cv2.putText(canvas, f"Segs:{n_classified}/{n_segs}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT, 1)
        x_pos += 100
        cv2.putText(canvas, f"Done:{labeled_count}/{len(self.videos)}", (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1)

        # Current segment info
        cur_label = self.segment_labels[cur_seg_idx] if cur_seg_idx < len(self.segment_labels) else None
        seg_start, seg_end = segments[cur_seg_idx] if cur_seg_idx < len(segments) else (0, 0)
        seg_dur = (seg_end - seg_start) / self.fps

        label_str = cur_label if cur_label else "???"
        label_color = COLOR_MAIN_ANGLE if cur_label == "main_angle" else \
                      COLOR_OTHER if cur_label == "other" else COLOR_TEXT_DIM
        seg_info = f"Seg {cur_seg_idx + 1}/{n_segs}: [{label_str}] ({seg_dur:.1f}s)"
        cv2.putText(canvas, seg_info, (WINDOW_W - 320, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, label_color, 1)

        # Controls
        line3 = "[M]issed cut [X]FP [1]main [2]other | [SPACE]play [N]ext [B]ack [Q]uit"
        cv2.putText(canvas, line3, (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_DIM, 1)

        # Play status + diff at cursor
        status = "PLAYING" if self.playing else "PAUSED"
        scolor = COLOR_GREEN if self.playing else COLOR_YELLOW
        cv2.putText(canvas, status, (WINDOW_W - 240, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, scolor, 1)

        if self.diffs is not None and 0 <= self.current_frame < len(self.diffs):
            diff_val = self.diffs[self.current_frame]
            cv2.putText(canvas, f"Diff: {diff_val:.3f}", (WINDOW_W - 240, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_CYAN, 1)

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

            # Segment label badge on the video
            badge_text = cur_label.upper() if cur_label else "???"
            badge_color = COLOR_MAIN_ANGLE if cur_label == "main_angle" else \
                          COLOR_OTHER if cur_label == "other" else COLOR_UNLABELED
            tw_badge = len(badge_text) * 14 + 16
            cv2.rectangle(canvas, (ox + 5, oy + 5), (ox + 5 + tw_badge, oy + 32), badge_color, -1)
            cv2.putText(canvas, badge_text, (ox + 13, oy + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # --- Diff graph ---
        gx, gy, gw, gh = self._graph_rect()
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COLOR_GRAPH_BG, -1)

        if self.diffs is not None and len(self.diffs) > 1:
            n = len(self.diffs)
            max_diff = max(self.diffs.max(), 0.01)

            # Graph polyline
            pts = []
            for i in range(n):
                px = gx + int(i / (n - 1) * gw)
                py = gy + gh - int((self.diffs[i] / max_diff) * (gh - 6)) - 3
                pts.append([px, py])
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=COLOR_GRAPH_LINE, thickness=1)

            # Threshold line
            if self.threshold <= max_diff:
                ty_line = gy + gh - int((self.threshold / max_diff) * (gh - 6)) - 3
                cv2.line(canvas, (gx, ty_line), (gx + gw, ty_line),
                         COLOR_THRESHOLD, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{self.threshold}", (gx + 2, ty_line - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_THRESHOLD, 1)

            # Cut marks on graph: auto (green), FP (red), FN (cyan)
            for cf in self.auto_detected:
                cx = gx + int(cf / (n - 1) * gw)
                if cf in self.false_positives:
                    cv2.line(canvas, (cx, gy), (cx, gy + gh), COLOR_FP_CUT, 1)
                else:
                    cv2.line(canvas, (cx, gy), (cx, gy + gh), COLOR_AUTO_CUT, 1)

            for cf in self.false_negatives:
                cx = gx + int(cf / (n - 1) * gw)
                cv2.line(canvas, (cx, gy), (cx, gy + gh), COLOR_FN_CUT, 2)

            # Playhead on graph
            px = gx + int(self.current_frame / (n - 1) * gw)
            cv2.line(canvas, (px, gy), (px, gy + gh), COLOR_PLAYHEAD, 2)

        # --- Timeline with segment classification colors ---
        tx, ty, tw, th = self._timeline_rect()
        n_frames = max(1, self.total_frames - 1)

        # Draw colored segment blocks
        for si, (seg_start, seg_end) in enumerate(segments):
            x1 = tx + int(seg_start / n_frames * tw)
            x2 = tx + int(min(seg_end, self.total_frames - 1) / n_frames * tw)
            sl = self.segment_labels[si] if si < len(self.segment_labels) else None
            color = COLOR_MAIN_ANGLE if sl == "main_angle" else \
                    COLOR_OTHER if sl == "other" else COLOR_UNLABELED
            cv2.rectangle(canvas, (x1, ty), (x2, ty + th), color, -1)

            # Thin border between segments
            if si > 0:
                cv2.line(canvas, (x1, ty), (x1, ty + th), (200, 200, 200), 1)

        # Highlight current segment border
        if cur_seg_idx < len(segments):
            seg_s, seg_e = segments[cur_seg_idx]
            x1 = tx + int(seg_s / n_frames * tw)
            x2 = tx + int(min(seg_e, self.total_frames - 1) / n_frames * tw)
            cv2.rectangle(canvas, (x1, ty), (x2, ty + th), (255, 255, 255), 2)

        # Playhead on timeline
        px = tx + int(self.current_frame / n_frames * tw)
        cv2.line(canvas, (px, ty), (px, ty + th), COLOR_PLAYHEAD, 2)

        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        gx, gy, gw, gh = self._graph_rect()
        tx, ty_, tw, th = self._timeline_rect()

        if (gx <= x <= gx + gw and gy <= y <= gy + gh) or \
           (tx <= x <= tx + tw and ty_ <= y <= ty_ + th):
            self.playing = False
            self.current_frame = self._frame_from_x(x)

    def run(self):
        print("=" * 60)
        print("SCENE CUT + SEGMENT LABELER")
        print("=" * 60)
        print(f"Threshold: {self.threshold}")
        print(f"Videos: {len(self.videos)}")
        print()
        print("Graph colors:")
        print(f"  GREEN lines = accepted auto-detected cuts")
        print(f"  RED lines   = false positive (X'd)")
        print(f"  CYAN lines  = missed cut you added (M)")
        print()
        print("Timeline colors:")
        print(f"  GREEN block = main_angle (1)")
        print(f"  RED block   = other (2)")
        print(f"  GRAY block  = unlabeled")
        print()
        print("Keys: [M]issed cut [X]FP [1]main [2]other")
        print("      [SPACE]play [Arrows]nav [N]ext [B]ack [Q]uit")
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
                    self._save_current_video()
                    cv2.destroyAllWindows()
                    print(f"\nSaved. Final cuts: {len(self._final_cuts())}")
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

                elif kl == ord("m"):
                    cf = self.current_frame
                    if cf in self.false_negatives:
                        self.false_negatives.remove(cf)
                        print(f"  - Removed FN at frame {cf}")
                    elif cf not in self.auto_detected:
                        self.false_negatives.append(cf)
                        self.false_negatives.sort()
                        diff = self.diffs[cf] if cf < len(self.diffs) else 0
                        print(f"  + FN added at frame {cf} ({cf/self.fps:.2f}s, diff={diff:.3f})")
                    else:
                        print(f"  Frame {cf} is already auto-detected")
                    # Adding/removing a cut changes segments — reset labels
                    self._sync_segment_labels()

                elif kl == ord("x"):
                    if self.auto_detected:
                        dists = [(abs(cf - self.current_frame), cf)
                                 for cf in self.auto_detected]
                        dists.sort()
                        nearest_dist, nearest_cf = dists[0]
                        if nearest_dist <= 15:
                            if nearest_cf in self.false_positives:
                                self.false_positives.discard(nearest_cf)
                                print(f"  Restored auto-cut at frame {nearest_cf}")
                            else:
                                self.false_positives.add(nearest_cf)
                                print(f"  X FP at frame {nearest_cf} "
                                      f"(diff={self.diffs[nearest_cf]:.3f})")
                            # Toggling an FP changes segments — reset labels
                            self._sync_segment_labels()
                        else:
                            print(f"  No auto-cut within 15 frames")
                    else:
                        print("  No auto-detected cuts")

                elif kl == ord("1"):
                    si = self._current_segment_idx()
                    self._sync_segment_labels()
                    self.segment_labels[si] = "main_angle"
                    seg_s, seg_e = self._get_segments()[si]
                    print(f"  Seg {si + 1} [{seg_s}-{seg_e}] = main_angle")

                elif kl == ord("2"):
                    si = self._current_segment_idx()
                    self._sync_segment_labels()
                    self.segment_labels[si] = "other"
                    seg_s, seg_e = self._get_segments()[si]
                    print(f"  Seg {si + 1} [{seg_s}-{seg_e}] = other")

                elif kl == ord("n"):
                    self._save_current_video()
                    final = self._final_cuts()
                    segments = self._get_segments()
                    n_classified = sum(1 for sl in self.segment_labels[:len(segments)]
                                       if sl is not None)
                    print(f"  Saved: {len(final)} cuts, "
                          f"{n_classified}/{len(segments)} segments classified")
                    self.video_idx += 1
                    break

                elif kl == ord("b"):
                    self._save_current_video()
                    if self.video_idx > 0:
                        self.video_idx -= 1
                        print("  Going back...")
                    break

        cv2.destroyAllWindows()
        labeled = sum(1 for vv in self.labels["videos"].values()
                      if vv.get("status") == "labeled")
        print(f"\nDone! Labeled {labeled}/{len(self.videos)} videos")


def main():
    parser = argparse.ArgumentParser(description="Label scene cuts + segments")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Auto-detection threshold (default: 0.25)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Min segment duration in seconds (default: 0.5)")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    videos = select_videos(args.metadata)
    print(f"Selected {len(videos)} videos across {len(set(v['stadium'] for v in videos))} stadiums")

    labels = load_labels(args.output)
    labels["metadata"]["total_videos"] = len(videos)

    for v in videos:
        if v["path"] not in labels["videos"]:
            labels["videos"][v["path"]] = {
                "stadium": v["stadium"],
                "season": v["season"],
                "status": "unlabeled",
                "cut_frames": [],
            }
    save_labels(args.output, labels)

    labeler = SceneCutLabeler(videos, labels, args.output,
                              threshold=args.threshold,
                              min_segment_duration=args.min_duration)
    labeler.run()


if __name__ == "__main__":
    main()
