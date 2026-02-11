"""
Frame Extractor — Scrub videos to select frame ranges, extract evenly-spaced frames.

Interactive tool for extracting frames from videos for later labeling.
On launch, prompts for what you're labeling, how many videos, and frames per video.
Can also be pointed at arbitrary video files or directories.

Usage:
    # Interactive mode (prompts for everything)
    python tools/frame_extractor.py

    # Arbitrary videos by path
    python tools/frame_extractor.py --videos path1.mp4 path2.mp4

    # All .mp4 files in a directory
    python tools/frame_extractor.py --dir data/videos/some_folder

    # List videos without running
    python tools/frame_extractor.py --list

    # Re-run specific video(s) that were already done/skipped
    python tools/frame_extractor.py --redo VIDEO_ID [VIDEO_ID ...]

    # Override interactive prompts via CLI
    python tools/frame_extractor.py --name ball --count 60 --frames 10
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "pitcher_calibration_metadata.json"
CROPPED_DIR = PROJECT_ROOT / "data" / "videos" / "pitcher_calibration_cropped"

WINDOW_NAME = "Frame Extractor"
WINDOW_W = 1280
WINDOW_H = 800

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_BG = (30, 30, 30)
COLOR_TIMELINE_BG = (60, 60, 60)
COLOR_RANGE = (0, 120, 0)
COLOR_PLAYHEAD = (0, 200, 255)

# Windows key codes for cv2.waitKeyEx
KEY_LEFT = 0x250000
KEY_RIGHT = 0x270000
KEY_UP = 0x260000
KEY_DOWN = 0x280000


# ---------------------------------------------------------------------------
# Video selection from calibration pool
# ---------------------------------------------------------------------------
def load_calibration_pool() -> list[dict]:
    """Load all cropped calibration videos with metadata."""
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    all_videos = []
    for key, videos in metadata.items():
        parts = key.rsplit("_", 1)
        stadium_name = parts[0]
        season = parts[1] if len(parts) == 2 else "unknown"

        for v in videos:
            v = dict(v)
            v["stadium"] = stadium_name
            v["season"] = season
            v["label"] = f"{stadium_name} | {v.get('pitcher_name', '?')} ({v.get('p_throws', '?')}HP)"
            cropped = CROPPED_DIR / stadium_name / season / f"{v['video_id']}.mp4"
            v["path"] = str(cropped)
            v["id"] = v["video_id"]
            if cropped.exists():
                all_videos.append(v)

    return all_videos


def sample_from_pool(pool: list[dict], count: int) -> list[dict]:
    """Sample videos from calibration pool, balanced by stadium + LHP/RHP.

    Strategy: distribute count across stadiums, picking LHP/RHP alternately.
    """
    # Group by stadium
    by_stadium: dict[str, dict[str, list[dict]]] = {}
    for v in pool:
        stadium = v["stadium"]
        hand = v.get("p_throws", "R")
        by_stadium.setdefault(stadium, {"L": [], "R": []})
        by_stadium[stadium][hand].append(v)

    stadiums = sorted(by_stadium.keys())
    n_stadiums = len(stadiums)

    # How many per stadium? Distribute evenly, then fill remainder
    per_stadium = max(1, count // n_stadiums)
    remainder = count - (per_stadium * n_stadiums)

    rng = random.Random(42)
    selected = []

    for si, stadium in enumerate(stadiums):
        pools = by_stadium[stadium]
        lhp = list(pools["L"])
        rhp = list(pools["R"])
        rng.shuffle(lhp)
        rng.shuffle(rhp)

        n_pick = per_stadium + (1 if si < remainder else 0)
        picks = []

        # Alternate LHP/RHP
        li, ri = 0, 0
        want_left = True
        while len(picks) < n_pick:
            if want_left and li < len(lhp):
                picks.append(lhp[li])
                li += 1
            elif ri < len(rhp):
                picks.append(rhp[ri])
                ri += 1
            elif li < len(lhp):
                picks.append(lhp[li])
                li += 1
            else:
                break  # exhausted this stadium
            want_left = not want_left

        selected.extend(picks)

    return selected[:count]


def videos_from_paths(paths: list[str]) -> list[dict]:
    """Build video list from explicit file paths."""
    selected = []
    for p in paths:
        p = Path(p)
        if p.exists() and p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            selected.append({
                "id": p.stem,
                "path": str(p),
                "label": p.stem,
            })
    return selected


def videos_from_dir(directory: str) -> list[dict]:
    """Build video list from all videos in a directory (recursive)."""
    d = Path(directory)
    paths = sorted(d.rglob("*.mp4")) + sorted(d.rglob("*.avi"))
    return videos_from_paths([str(p) for p in paths])


def list_videos(videos: list[dict]):
    """Print selected videos for verification."""
    print(f"\n{len(videos)} videos:\n")
    for i, v in enumerate(videos):
        print(f"  {i+1:3d}. {v['label']:60s} {v['id']}")
    print()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def session_dir_for(name: str) -> Path:
    return PROJECT_ROOT / "data" / "labels" / name / "frames"


def session_path_for(output_dir: Path) -> Path:
    return output_dir / ".extractor_session.json"


def load_session(output_dir: Path) -> dict:
    sp = session_path_for(output_dir)
    if sp.exists():
        with open(sp) as f:
            return json.load(f)
    return {"videos": {}, "counter": 0}


def save_session(session: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = session_path_for(output_dir)
    with open(sp, "w") as f:
        json.dump(session, f, indent=2)


def get_video_state(session: dict, video_id: str) -> dict:
    if video_id not in session["videos"]:
        session["videos"][video_id] = {
            "status": "pending",  # pending | done | skipped
            "start_frame": None,
            "end_frame": None,
            "extracted_files": [],
        }
    return session["videos"][video_id]


# ---------------------------------------------------------------------------
# Range Selector UI
# ---------------------------------------------------------------------------
class RangeSelector:
    """Scrub a video to mark start/end frame range."""

    def __init__(self, video_path: str, video_info: dict, video_idx: int,
                 total_videos: int, initial_start: int | None = None,
                 initial_end: int | None = None, speed_mode: int = 1):
        self.video_path = video_path
        self.video_info = video_info
        self.video_idx = video_idx
        self.total_videos = total_videos

        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame = 0
        self.start_frame = initial_start
        self.end_frame = initial_end
        self.playing = False
        self.dragging_timeline = False
        self.next_mark = "start" if initial_start is None else ("end" if initial_end is None else "start")

        # Playback speed: 1=1x, 2=2x, 3=3x (frames to advance per tick)
        self.speed_mode = speed_mode

        # Layout
        self.timeline_y = WINDOW_H - 80
        self.timeline_h = 30
        self.timeline_x = 40
        self.timeline_w = WINDOW_W - 80

        self.frame_cache = None

    def _read_frame(self, idx: int) -> np.ndarray | None:
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = idx
            self.frame_cache = frame
            return frame
        return self.frame_cache

    def _frame_from_x(self, x: int) -> int:
        frac = max(0.0, min(1.0, (x - self.timeline_x) / self.timeline_w))
        return int(frac * max(1, self.total_frames - 1))

    def _draw(self) -> np.ndarray:
        canvas = np.full((WINDOW_H, WINDOW_W, 3), COLOR_BG[0], dtype=np.uint8)

        # -- Top info bar --
        info = f"{self.video_info.get('label', '')}  |  Video {self.video_idx+1}/{self.total_videos}"
        cv2.putText(canvas, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

        # Range info
        start_str = str(self.start_frame) if self.start_frame is not None else "---"
        end_str = str(self.end_frame) if self.end_frame is not None else "---"
        if self.start_frame is not None and self.end_frame is not None:
            n_range = self.end_frame - self.start_frame + 1
            range_str = f"Range: [{start_str} - {end_str}] ({n_range} frames)"
        else:
            range_str = f"Range: [{start_str} - {end_str}]"
        frame_info = f"Frame {self.current_frame}/{self.total_frames-1}  |  {range_str}"
        cv2.putText(canvas, frame_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1)

        # Show which mark L will set next
        mark_hint = f"L will set: {'START' if self.next_mark == 'start' else 'END'}"
        mark_color = COLOR_GREEN if self.next_mark == "start" else COLOR_RED
        cv2.putText(canvas, mark_hint, (WINDOW_W - 280, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mark_color, 2)

        # -- Main frame display --
        if self.frame_cache is not None:
            display_h = self.timeline_y - 70
            display_w = WINDOW_W - 20
            scale = min(display_w / self.frame_w, display_h / self.frame_h)
            new_w = int(self.frame_w * scale)
            new_h = int(self.frame_h * scale)
            resized = cv2.resize(self.frame_cache, (new_w, new_h))

            x_off = (WINDOW_W - new_w) // 2
            y_off = 65
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # -- Timeline --
        tx, ty = self.timeline_x, self.timeline_y
        tw, th = self.timeline_w, self.timeline_h

        cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), COLOR_TIMELINE_BG, -1)

        # Range highlight
        if self.start_frame is not None and self.end_frame is not None:
            sx = tx + int(self.start_frame / max(1, self.total_frames - 1) * tw)
            ex = tx + int(self.end_frame / max(1, self.total_frames - 1) * tw)
            cv2.rectangle(canvas, (sx, ty), (ex, ty + th), COLOR_RANGE, -1)

        # Start marker
        if self.start_frame is not None:
            sx = tx + int(self.start_frame / max(1, self.total_frames - 1) * tw)
            cv2.line(canvas, (sx, ty - 10), (sx, ty + th + 10), COLOR_GREEN, 3)
            cv2.putText(canvas, "S", (sx - 5, ty - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)

        # End marker
        if self.end_frame is not None:
            ex = tx + int(self.end_frame / max(1, self.total_frames - 1) * tw)
            cv2.line(canvas, (ex, ty - 10), (ex, ty + th + 10), COLOR_RED, 3)
            cv2.putText(canvas, "E", (ex - 5, ty - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)

        # Playhead
        px = tx + int(self.current_frame / max(1, self.total_frames - 1) * tw)
        cv2.line(canvas, (px, ty - 5), (px, ty + th + 5), COLOR_PLAYHEAD, 2)

        # -- Speed indicator --
        speed_str = f"Speed: {self.speed_mode}x (press 1/2/3)"
        if self.playing:
            speed_str = f"PLAYING {self.speed_mode}x (1/2/3)"
        cv2.putText(canvas, speed_str, (WINDOW_W - 280, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN if self.playing else COLOR_GRAY, 1)

        # -- Controls help --
        controls = "SPACE: play  |  L: set start/end  |  LEFT/RIGHT: +/-1  |  UP/DOWN: +/-10  |  1/2/3: speed  |  N: next  |  B: back  |  ESC: quit"
        cv2.putText(canvas, controls, (10, WINDOW_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1)

        return canvas

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.timeline_y - 10 <= y <= self.timeline_y + self.timeline_h + 10
                    and self.timeline_x <= x <= self.timeline_x + self.timeline_w):
                self.dragging_timeline = True
                self._read_frame(self._frame_from_x(x))
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_timeline:
            self._read_frame(self._frame_from_x(x))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_timeline = False

    def run(self) -> str:
        """Returns 'confirmed', 'skipped', 'back', or 'quit'."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        self._read_frame(self.current_frame)

        while True:
            # Detect window closed via X button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.cap.release()
                has_range = self.start_frame is not None and self.end_frame is not None
                return "quit_confirmed" if has_range else "quit"

            canvas = self._draw()
            cv2.imshow(WINDOW_NAME, canvas)

            wait_ms = 1 if self.playing else 0
            key = cv2.waitKeyEx(max(1, wait_ms))
            kl = key & 0xFF

            if self.playing:
                nf = self.current_frame + self.speed_mode
                if nf >= self.total_frames:
                    self.playing = False
                else:
                    self._read_frame(nf)

            # -- Key handling --
            has_range = self.start_frame is not None and self.end_frame is not None

            if kl == ord("q") or kl == 27:  # Q or ESC — save & quit
                self.cap.release()
                return "quit_confirmed" if has_range else "quit"
            elif kl == ord("n") or kl == 13 or kl == 10:  # N or ENTER — next video
                self.cap.release()
                if has_range:
                    return "confirmed"
                else:
                    return "skipped"
            elif kl == ord("b"):  # B — back to previous video
                self.cap.release()
                return "back_confirmed" if has_range else "back"
            elif kl == ord(" "):  # SPACE — play/pause
                self.playing = not self.playing
            elif kl == ord("l"):  # L — set start/end marker
                if self.next_mark == "start":
                    self.start_frame = self.current_frame
                    if self.end_frame is not None and self.start_frame > self.end_frame:
                        self.end_frame = None
                    self.next_mark = "end"
                else:
                    self.end_frame = self.current_frame
                    if self.start_frame is not None and self.end_frame < self.start_frame:
                        self.start_frame = self.end_frame
                        self.end_frame = None
                        self.next_mark = "end"
                    else:
                        self.next_mark = "start"
            elif kl == ord("1"):
                self.speed_mode = 1
            elif kl == ord("2"):
                self.speed_mode = 2
            elif kl == ord("3"):
                self.speed_mode = 3
            elif key == KEY_RIGHT or kl == ord("d"):
                self.playing = False
                self._read_frame(min(self.current_frame + 1, self.total_frames - 1))
            elif key == KEY_LEFT or kl == ord("a"):
                self.playing = False
                self._read_frame(max(self.current_frame - 1, 0))
            elif key == KEY_UP:
                self.playing = False
                self._read_frame(min(self.current_frame + 10, self.total_frames - 1))
            elif key == KEY_DOWN:
                self.playing = False
                self._read_frame(max(self.current_frame - 10, 0))

        self.cap.release()
        return "quit"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, video_id: str, start: int, end: int,
                   n_frames: int, output_dir: Path, counter_start: int,
                   prefix: str) -> tuple[list[str], int]:
    """Extract evenly-spaced frames from range. Returns (file_list, next_counter)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    n_range = end - start + 1

    if n_range <= n_frames:
        indices = list(range(start, end + 1))
    else:
        step = n_range / n_frames
        indices = [start + int(step * j + step / 2) for j in range(n_frames)]
        indices = [min(idx, end) for idx in indices]

    files = []
    counter = counter_start
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        fname = f"{prefix}_{counter:04d}_{video_id}_f{frame_idx}.png"
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), frame)
        files.append(fname)
        counter += 1

    cap.release()
    return files, counter


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------
def prompt_config() -> dict:
    """Interactively ask the user what we're extracting frames for."""
    print("\n=== Frame Extractor Setup ===\n")

    # What are we labeling?
    name = input("What are we extracting frames to label? (e.g. ball, bat_barrel, mitt): ").strip()
    if not name:
        print("Name is required.")
        sys.exit(1)

    # How many videos?
    count_str = input("How many videos? [60]: ").strip()
    count = int(count_str) if count_str else 60

    # How many frames per video?
    frames_str = input("Frames per video? [10]: ").strip()
    frames = int(frames_str) if frames_str else 10

    print()
    return {"name": name, "count": count, "frames": frames}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Frame Extractor -- scrub videos, select ranges, extract frames")

    # Video source (optional — defaults to calibration pool)
    parser.add_argument("--videos", nargs="+", metavar="PATH",
                        help="Explicit video file paths (skip interactive prompts)")
    parser.add_argument("--dir", metavar="DIR",
                        help="Directory of videos, recursive (skip interactive prompts)")

    # Override interactive prompts
    parser.add_argument("--name", metavar="NAME",
                        help="Label name (e.g. ball, bat_barrel, mitt)")
    parser.add_argument("--count", type=int, metavar="N",
                        help="Number of videos to sample from calibration pool")
    parser.add_argument("--frames", type=int, metavar="N",
                        help="Frames to extract per video")

    parser.add_argument("--list", action="store_true",
                        help="List selected videos and exit")
    parser.add_argument("--redo", nargs="+", metavar="VIDEO_ID",
                        help="Reset these video IDs to pending and re-run")

    args = parser.parse_args()

    # -- Determine video source and config --
    if args.videos:
        # Explicit paths mode
        videos = videos_from_paths(args.videos)
        name = args.name or "custom"
        n_frames = args.frames or 10
        output_dir = session_dir_for(name)
    elif args.dir:
        # Directory mode
        videos = videos_from_dir(args.dir)
        name = args.name or "custom"
        n_frames = args.frames or 10
        output_dir = session_dir_for(name)
    else:
        # Interactive / calibration pool mode
        if args.name and args.count and args.frames:
            # All args provided, no prompts needed
            config = {"name": args.name, "count": args.count, "frames": args.frames}
        elif args.name or args.count or args.frames:
            # Partial args — fill in missing with prompts
            config = {}
            config["name"] = args.name or input("What are we extracting frames to label? (e.g. ball, bat_barrel, mitt): ").strip()
            config["count"] = args.count or int(input("How many videos? [60]: ").strip() or "60")
            config["frames"] = args.frames or int(input("Frames per video? [10]: ").strip() or "10")
        else:
            # Full interactive
            config = prompt_config()

        name = config["name"]
        n_frames = config["frames"]
        output_dir = session_dir_for(name)

        pool = load_calibration_pool()
        print(f"Calibration pool: {len(pool)} videos across {len(set(v['stadium'] for v in pool))} stadiums")
        videos = sample_from_pool(pool, config["count"])

    prefix = name

    if not videos:
        print("No videos found.")
        return

    if args.list:
        list_videos(videos)
        return

    # -- Load session --
    session = load_session(output_dir)

    # -- Handle --redo --
    if args.redo:
        for vid_id in args.redo:
            if vid_id in session["videos"]:
                session["videos"][vid_id]["status"] = "pending"
                session["videos"][vid_id]["start_frame"] = None
                session["videos"][vid_id]["end_frame"] = None
                print(f"  Reset: {vid_id}")
            else:
                print(f"  Not found in session: {vid_id}")

    print(f"{len(videos)} videos, extracting {n_frames} frames each -> {output_dir}")

    # Find first non-completed video
    start_idx = 0
    for i, v in enumerate(videos):
        state = get_video_state(session, v["id"])
        if state["status"] == "pending":
            start_idx = i
            break
    else:
        start_idx = len(videos)

    if start_idx >= len(videos):
        total_done = sum(1 for vs in session["videos"].values() if vs["status"] == "done")
        total_skipped = sum(1 for vs in session["videos"].values() if vs["status"] == "skipped")
        print(f"All videos processed! Done: {total_done}, Skipped: {total_skipped}")
        print(f"Extracted frames: {session['counter']}")
        return

    print(f"Starting from video {start_idx + 1}/{len(videos)}\n")

    current_speed = 1
    i = start_idx
    while i < len(videos):
        v = videos[i]
        vid = v["id"]
        video_path = v["path"]
        state = get_video_state(session, vid)

        if state["status"] in ("done", "skipped"):
            i += 1
            continue

        if not Path(video_path).exists():
            print(f"  Skipping {vid}: file not found")
            state["status"] = "skipped"
            i += 1
            continue

        print(f"  [{i+1}/{len(videos)}] {v.get('label', vid)}")

        selector = RangeSelector(
            video_path, v, i, len(videos),
            initial_start=state["start_frame"],
            initial_end=state["end_frame"],
            speed_mode=current_speed,
        )
        result = selector.run()
        current_speed = selector.speed_mode

        # Helper: extract frames if range was set
        def do_extract():
            state["start_frame"] = selector.start_frame
            state["end_frame"] = selector.end_frame
            files, new_counter = extract_frames(
                video_path, vid,
                selector.start_frame, selector.end_frame,
                n_frames, output_dir, session["counter"], prefix,
            )
            session["counter"] = new_counter
            state["extracted_files"] = files
            state["status"] = "done"
            save_session(session, output_dir)
            print(f"    Extracted {len(files)} frames")

        if result in ("quit", "quit_confirmed"):
            if result == "quit_confirmed":
                do_extract()
            save_session(session, output_dir)
            print(f"\nSession saved ({session['counter']} frames extracted). Run again to resume.")
            cv2.destroyAllWindows()
            return
        elif result in ("back", "back_confirmed"):
            if result == "back_confirmed":
                do_extract()
            if i > 0:
                i -= 1
                prev_state = get_video_state(session, videos[i]["id"])
                if prev_state["status"] in ("done", "skipped"):
                    prev_state["status"] = "pending"
            save_session(session, output_dir)
            continue
        elif result == "skipped":
            state["status"] = "skipped"
            save_session(session, output_dir)
            i += 1
            continue
        elif result == "confirmed":
            do_extract()
            i += 1

    # -- Save & Summary --
    save_session(session, output_dir)
    cv2.destroyAllWindows()
    total_done = sum(1 for vs in session["videos"].values() if vs["status"] == "done")
    total_skipped = sum(1 for vs in session["videos"].values() if vs["status"] == "skipped")
    print(f"\nDone! Processed: {total_done}, Skipped: {total_skipped}")
    print(f"Total extracted frames: {session['counter']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
