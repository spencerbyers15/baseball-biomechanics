"""Smell test: render a top-down baseball-field video from decoded
FieldVision pelvis positions across all available .bin segments. Each
actor is a colored dot with a short trail; labels show MLB player ID
(joined via labels.json) + actor type (fielder / plate-umpire / etc).

Output: samples/decoded_<gamePk>_first<duration>s.mp4

If the decoder is right, you'll see:
  - Plate umpire stationary near home plate (~0, 0)
  - Fielders distributed in the outfield/infield
  - Smooth motion frame-to-frame (no teleporting)
  - 30fps timing matches game clock
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.wire_schemas import read_tracking_data


SAMPLES_DIR = Path("samples/binary_capture_823141")
GAME_PK = 823141
TRAIL_LEN = 30  # ~1 second at 30fps


def load_all_frames():
    seg_paths = sorted(SAMPLES_DIR.glob(f"mlb_{GAME_PK}_segment_*.bin"),
                       key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Decoding {len(seg_paths)} segments...")
    all_frames = []
    for path in seg_paths:
        td = read_tracking_data(path.read_bytes())
        for f in td.frames:
            all_frames.append({
                "ts": f.timestamp,
                "time": f.time,
                "isGap": f.isGap,
                "actors": [
                    {
                        "uid": a.uid,
                        "x": a.rootPos.x, "y": a.rootPos.y, "z": a.rootPos.z,
                        "bat": (a.batRootPos.x, a.batRootPos.y, a.batRootPos.z) if a.batRootPos else None,
                        "scale": a.scale,
                    }
                    for a in f.actorPoses if a.rootPos
                ],
            })
        print(f"  {path.name}: {len(td.frames)} frames")
    return all_frames


def setup_field(ax):
    """Draw a top-down baseball field, view from above home plate looking
    out toward the outfield. X axis: lateral (3B negative, 1B positive).
    Y axis on plot: stadium Z coordinate (negative = deep outfield).
    """
    ax.set_facecolor("#3a5a3a")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-450, 60)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("X — lateral (negative = 3B side, positive = 1B side)  [ft]")
    ax.set_ylabel("Z — depth (negative = outfield)  [ft]")

    # Foul lines (extending to ~270 ft each side)
    ax.plot([0, 270], [0, -270], "w-", alpha=0.6, lw=1.5)  # 1B foul line
    ax.plot([0, -270], [0, -270], "w-", alpha=0.6, lw=1.5)  # 3B foul line

    # Infield diamond (90 ft between bases, rotated 45°)
    home = (0, 0)
    first = (90 / (2 ** 0.5), -90 / (2 ** 0.5))
    second = (0, -90 * (2 ** 0.5))
    third = (-90 / (2 ** 0.5), -90 / (2 ** 0.5))
    diamond = patches.Polygon(
        [home, first, second, third],
        closed=True,
        facecolor="#8B6644",
        alpha=0.45,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(diamond)

    # Bases
    for bx, by in [home, first, second, third]:
        ax.add_patch(patches.Rectangle((bx - 1.5, by - 1.5), 3, 3, facecolor="white"))

    # Pitcher's mound
    ax.add_patch(patches.Circle((0, -60.5), radius=9,
                                facecolor="#8B6644", alpha=0.7,
                                edgecolor="white", linewidth=1))
    ax.plot(0, -60.5, "w.", markersize=4)  # pitching rubber

    # Outfield arc (rough — real fields vary, this is a generic 350 ft arc)
    arc = patches.Arc((0, 0), 700, 700, angle=0, theta1=225, theta2=315,
                      edgecolor="white", linewidth=1, alpha=0.5)
    ax.add_patch(arc)

    ax.text(0, 5, "HOME", color="white", ha="center", fontsize=9, fontweight="bold")


def main():
    labels = json.load(open(SAMPLES_DIR / f"mlb_{GAME_PK}_labels.json"))
    all_frames = load_all_frames()
    print(f"\nTotal frames: {len(all_frames)}")
    duration_s = (all_frames[-1]["time"] - all_frames[0]["time"]) if all_frames else 0
    print(f"Time span: {duration_s:.1f}s")
    actor_count_per_frame = [len(f["actors"]) for f in all_frames]
    print(f"actor count per frame: min={min(actor_count_per_frame)} "
          f"max={max(actor_count_per_frame)} "
          f"mean={sum(actor_count_per_frame)/len(actor_count_per_frame):.2f}")

    # All UIDs that appear; assign each a stable color
    all_uids = sorted({a["uid"] for fr in all_frames for a in fr["actors"]})
    print(f"Distinct UIDs: {all_uids}")
    palette = plt.cm.tab20.colors
    color_map = {uid: palette[i % len(palette)] for i, uid in enumerate(all_uids)}

    fig, ax = plt.subplots(figsize=(11, 11))
    setup_field(ax)

    # Per-actor trail plotted as a Line2D, plus a single scatter for current positions
    trails = {uid: ax.plot([], [], "-", color=color_map[uid], alpha=0.55, lw=2)[0]
              for uid in all_uids}
    actor_history: dict[int, list[tuple[float, float]]] = {uid: [] for uid in all_uids}
    scatter = ax.scatter([], [], s=180, edgecolors="black", linewidths=1.5, zorder=5)
    label_texts = {uid: ax.text(0, 0, "", color="white", fontsize=8, ha="center",
                                va="center", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          facecolor=color_map[uid], alpha=0.7),
                                visible=False)
                   for uid in all_uids}

    title = ax.set_title("")

    def update(frame_idx):
        fr = all_frames[frame_idx]
        xs, ys, colors = [], [], []
        seen = set()
        for a in fr["actors"]:
            uid = a["uid"]
            seen.add(uid)
            plot_x, plot_y = a["x"], a["z"]
            xs.append(plot_x)
            ys.append(plot_y)
            colors.append(color_map[uid])

            actor_history[uid].append((plot_x, plot_y))
            if len(actor_history[uid]) > TRAIL_LEN:
                actor_history[uid] = actor_history[uid][-TRAIL_LEN:]

            info = labels.get(str(uid), {})
            actor_id = info.get("actor", uid)
            atype = info.get("type", "?")
            label_texts[uid].set_position((plot_x, plot_y + 12))
            label_texts[uid].set_text(f"{actor_id}\n{atype}")
            label_texts[uid].set_visible(True)

        # Hide labels for actors not in this frame
        for uid in all_uids:
            if uid not in seen:
                label_texts[uid].set_visible(False)

        # Update trail Line2Ds
        for uid in all_uids:
            history = actor_history[uid]
            if history:
                xs_t, ys_t = zip(*history)
                trails[uid].set_data(xs_t, ys_t)
            else:
                trails[uid].set_data([], [])

        # Update scatter
        import numpy as np
        if xs:
            scatter.set_offsets(np.column_stack([xs, ys]))
            scatter.set_color(colors)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        title.set_text(
            f"Frame {frame_idx + 1}/{len(all_frames)}   ts={fr['ts']}   "
            f"actors_in_frame={len(fr['actors'])}"
        )
        return [scatter] + list(trails.values()) + list(label_texts.values()) + [title]

    print(f"\nRendering {len(all_frames)} frames...")
    anim = FuncAnimation(fig, update, frames=len(all_frames),
                         interval=33, blit=False)

    writer = FFMpegWriter(fps=30, bitrate=2400, codec="h264")
    out_path = Path(f"samples/decoded_{GAME_PK}_first{int(duration_s)}s.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=writer, dpi=90)
    print(f"\nSaved: {out_path.resolve()}")
    print(f"Size:  {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
