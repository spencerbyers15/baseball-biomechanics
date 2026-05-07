"""Render a short skeleton clip from any window of the captured game.

Pulls joint positions directly from the SQLite store (data/fv_<gamePk>.sqlite),
so this is a fast smell test that the storage layer + the GLTF rest pose
hold across the full 3-hour game.

Usage:
  python scripts/render_clip.py --game 823141 --start-segment 1582 --duration 30
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.skeleton import SKELETON_CONNECTIONS
from fieldvision.storage import JOINT_COLS


# bone_id -> column-name prefix (the same name we used in actor_frame DDL)
BID_TO_COL = {bid: name for bid, name in JOINT_COLS}

JOINT_LABEL = {
    0: "Pelv", 1: "HipM",
    2: "R.Hip", 3: "R.Knee", 4: "R.Foot",
    10: "L.Hip", 11: "L.Knee", 12: "L.Foot",
    18: "TorA", 19: "TorB",
    20: "Neck", 21: "Head",
    25: "R.Clav", 26: "R.Shdr", 27: "R.Elb", 28: "R.Hand",
    64: "L.Clav", 65: "L.Shdr", 66: "L.Elb", 67: "L.Hand",
}
JOINT_LABEL_3D = {0: "Pelv", 21: "Head", 4: "R.Foot", 12: "L.Foot",
                  28: "R.Hand", 67: "L.Hand"}


def load_clip(db_path: Path, start_seg: int, n_segments: int):
    """Pull all actor-frame rows in the segment range. Returns frames list."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    end_seg = start_seg + n_segments

    # Build the column list dynamically
    joint_cols = []
    for bid, name in JOINT_COLS:
        joint_cols += [f"{name}_x", f"{name}_y", f"{name}_z"]
    select_cols = (
        "segment_idx, frame_num, actor_uid, mlb_player_id, actor_type, "
        "time_unix, timestamp, scale, "
        "bat_handle_x, bat_handle_y, bat_handle_z, "
        + ", ".join(joint_cols)
    )
    sql = (
        f"SELECT {select_cols} FROM actor_frame "
        f"WHERE segment_idx >= ? AND segment_idx < ? "
        f"ORDER BY segment_idx, frame_num, actor_uid"
    )
    cur.execute(sql, (start_seg, end_seg))
    rows = cur.fetchall()
    conn.close()
    print(f"  Loaded {len(rows):,} actor-frame rows from segments {start_seg}..{end_seg - 1}")

    # Group by (segment, frame_num) → list of actor dicts
    frames_dict = defaultdict(list)
    for r in rows:
        seg, fn, uid, mlb_id, atype, time_unix, ts, scale = r[:8]
        bat_x, bat_y, bat_z = r[8], r[9], r[10]
        joint_data = r[11:]
        wp = {}
        for i, (bid, _) in enumerate(JOINT_COLS):
            x = joint_data[i * 3]
            y = joint_data[i * 3 + 1]
            z = joint_data[i * 3 + 2]
            if x is not None:
                wp[bid] = (x, y, z)
        bat = (bat_x, bat_y, bat_z) if bat_x is not None else None
        frames_dict[(seg, fn)].append({
            "uid": uid, "mlb_id": mlb_id, "type": atype,
            "time": time_unix, "ts": ts, "scale": scale,
            "world_pos": wp, "bat": bat,
        })

    # Convert to ordered list
    sorted_keys = sorted(frames_dict.keys())
    frames = [{
        "key": k,
        "ts": frames_dict[k][0]["ts"] if frames_dict[k] else None,
        "actors": frames_dict[k],
    } for k in sorted_keys]
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start-segment", type=int, default=1582)
    parser.add_argument("--duration", type=int, default=30, help="Seconds")
    parser.add_argument("--out", default=None)
    parser.add_argument("--focus-uid", type=int, default=None,
                        help="Override the auto-picked focus actor (default: most-common in window)")
    args = parser.parse_args()

    n_segments = args.duration // 5  # 5 sec per segment
    db_path = Path(args.data_dir) / f"fv_{args.game}.sqlite"
    print(f"Reading from {db_path}")
    frames = load_clip(db_path, args.start_segment, n_segments)
    if not frames:
        raise SystemExit("No frames found in the requested range")

    # Pick focus actor: explicit arg, or most common across the clip
    from collections import Counter
    actor_counts = Counter(a["uid"] for f in frames for a in f["actors"])
    if args.focus_uid is not None:
        focus_uid = args.focus_uid
        focus_count = actor_counts.get(focus_uid, 0)
        if focus_count == 0:
            raise SystemExit(f"focus uid {focus_uid} not in this segment range")
    else:
        focus_uid, focus_count = actor_counts.most_common(1)[0]
    info_actor = None
    for f in frames:
        for a in f["actors"]:
            if a["uid"] == focus_uid:
                info_actor = a
                break
        if info_actor: break
    print(f"  Focus uid={focus_uid}  mlb_id={info_actor['mlb_id']}  type={info_actor['type']}")

    # Render layout (same as before)
    fig = plt.figure(figsize=(16, 10), facecolor="#222")
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 2, 1.3], height_ratios=[1, 1, 1],
                          hspace=0.25, wspace=0.18, left=0.04, right=0.98, top=0.94, bottom=0.05)
    ax3d = fig.add_subplot(gs[:, 0:2], projection="3d")
    ax_front = fig.add_subplot(gs[0, 2])
    ax_side = fig.add_subplot(gs[1, 2])
    ax_top = fig.add_subplot(gs[2, 2])

    palette = plt.cm.tab10.colors
    color_map = {uid: palette[i % len(palette)] for i, uid in enumerate(actor_counts)}

    max_actors = max(len(f["actors"]) for f in frames)
    bones_3d = [[ax3d.plot([], [], [], "-", lw=2.5, color="white",
                           solid_capstyle="round")[0] for _ in SKELETON_CONNECTIONS]
                for _ in range(max_actors)]
    joints_3d = [ax3d.scatter([], [], [], s=18, c="cyan", depthshade=False)
                 for _ in range(max_actors)]
    bat_3d = [ax3d.plot([], [], [], "-", lw=4, color="#d4a04c")[0] for _ in range(max_actors)]
    head_marker_3d = ax3d.scatter([], [], [], s=80, c="red", depthshade=False, marker="o")

    bones_front = [[ax_front.plot([], [], "-", lw=2, color="white",
                                  solid_capstyle="round")[0] for _ in SKELETON_CONNECTIONS]]
    joints_front = ax_front.scatter([], [], s=14, c="cyan")
    bat_front = ax_front.plot([], [], "-", lw=3, color="#d4a04c")[0]
    labels_front = {bid: ax_front.text(0, 0, "", color="yellow", fontsize=6,
                                       ha="left", va="center", visible=False,
                                       bbox=dict(boxstyle="round,pad=0.1",
                                                 facecolor="black", alpha=0.5,
                                                 edgecolor="none"))
                    for bid in JOINT_LABEL}
    ax_front.set_facecolor("#3a5a3a"); ax_front.set_aspect("equal")
    ax_front.set_title("FRONT", color="white", fontsize=10)
    ax_front.tick_params(colors="white", labelsize=7)

    bones_side = [[ax_side.plot([], [], "-", lw=2, color="white",
                                solid_capstyle="round")[0] for _ in SKELETON_CONNECTIONS]]
    joints_side = ax_side.scatter([], [], s=14, c="cyan")
    bat_side = ax_side.plot([], [], "-", lw=3, color="#d4a04c")[0]
    labels_side = {bid: ax_side.text(0, 0, "", color="yellow", fontsize=6,
                                     ha="left", va="center", visible=False,
                                     bbox=dict(boxstyle="round,pad=0.1",
                                               facecolor="black", alpha=0.5,
                                               edgecolor="none"))
                   for bid in JOINT_LABEL}
    ax_side.set_facecolor("#3a5a3a"); ax_side.set_aspect("equal")
    ax_side.set_title("SIDE", color="white", fontsize=10)
    ax_side.tick_params(colors="white", labelsize=7)

    labels_3d = {bid: ax3d.text(0, 0, 0, label, color="yellow", fontsize=8)
                 for bid, label in JOINT_LABEL_3D.items()}

    ax_top.set_facecolor("#3a5a3a")
    ax_top.set_xlim(-300, 300); ax_top.set_ylim(-450, 60); ax_top.set_aspect("equal")
    ax_top.tick_params(colors="white", labelsize=7)
    ax_top.set_title("Top-down field", color="white", fontsize=10)
    ax_top.plot([0, 270], [0, -270], "w-", alpha=0.4, lw=1)
    ax_top.plot([0, -270], [0, -270], "w-", alpha=0.4, lw=1)
    diamond_xs = [0, 90/np.sqrt(2), 0, -90/np.sqrt(2), 0]
    diamond_ys = [0, -90/np.sqrt(2), -90*np.sqrt(2), -90/np.sqrt(2), 0]
    ax_top.plot(diamond_xs, diamond_ys, "w-", alpha=0.5, lw=1)
    actor_dots = {uid: ax_top.plot([], [], "o", ms=6, color=color_map[uid])[0] for uid in actor_counts}
    actor_focus_ring = ax_top.plot([], [], "o", ms=15, mfc="none", mec="yellow", mew=1.5)[0]

    ax3d.set_facecolor("#3a5a3a")
    ax3d.tick_params(colors="white")
    ax3d.set_xlabel("X — 3B/1B (ft)", color="white")
    ax3d.set_ylabel("Z — depth (ft)", color="white")
    ax3d.set_zlabel("Y — up (ft)", color="white")
    title3d = ax3d.set_title("", color="white", fontsize=12)
    for ax_pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        ax_pane.set_pane_color((0.2, 0.2, 0.2, 1))
    ax3d.view_init(elev=15, azim=-50)

    def draw_2d(ax, ix, iy, focus_actor, lines, joints_scat, bat_line, labels_dict):
        for ln in lines[0]: ln.set_data([], [])
        joints_scat.set_offsets(np.empty((0, 2)))
        bat_line.set_data([], [])
        for tx in labels_dict.values(): tx.set_visible(False)
        if focus_actor is None: return
        wp = focus_actor["world_pos"]
        for bi, (a_id, b_id) in enumerate(SKELETON_CONNECTIONS):
            if a_id in wp and b_id in wp:
                p1, p2 = wp[a_id], wp[b_id]
                lines[0][bi].set_data([p1[ix], p2[ix]], [p1[iy], p2[iy]])
        pts = np.array([[p[ix], p[iy]] for p in wp.values()])
        joints_scat.set_offsets(pts)
        for bid, p in wp.items():
            if bid in labels_dict:
                labels_dict[bid].set_position((p[ix] + 0.15, p[iy]))
                labels_dict[bid].set_text(JOINT_LABEL[bid])
                labels_dict[bid].set_visible(True)
        # Center on first present joint (pelvis preferred)
        center = wp.get(0) or next(iter(wp.values()))
        ax.set_xlim(center[ix] - 5, center[ix] + 5)
        ax.set_ylim(center[iy] - 5, center[iy] + 5)

    def update(frame_idx):
        fr = frames[frame_idx]
        focus = next((a for a in fr["actors"] if a["uid"] == focus_uid), None)
        if focus is None and fr["actors"]:
            focus = fr["actors"][0]

        if focus is not None and 0 in focus["world_pos"]:
            cx, cy, cz = focus["world_pos"][0]
            r = 7.0
            ax3d.set_xlim(cx - r, cx + r)
            ax3d.set_ylim(cz - r, cz + r)
            ax3d.set_zlim(0, r * 1.6)

        # reset 3D
        for actor_lines in bones_3d:
            for ln in actor_lines: ln.set_data_3d([], [], [])
        for s in joints_3d: s._offsets3d = ([], [], [])
        for ln in bat_3d: ln.set_data_3d([], [], [])
        head_marker_3d._offsets3d = ([], [], [])

        for ai, a in enumerate(fr["actors"]):
            if ai >= max_actors: break
            color = color_map.get(a["uid"], "white")
            wp = a["world_pos"]
            for bi, (a_id, b_id) in enumerate(SKELETON_CONNECTIONS):
                if a_id in wp and b_id in wp:
                    p1, p2 = wp[a_id], wp[b_id]
                    bones_3d[ai][bi].set_data_3d([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]])
                    bones_3d[ai][bi].set_color(color)
            joint_xs = [p[0] for p in wp.values()]
            joint_ys = [p[2] for p in wp.values()]
            joint_zs = [p[1] for p in wp.values()]
            joints_3d[ai]._offsets3d = (joint_xs, joint_ys, joint_zs)
            joints_3d[ai].set_color(color)
            if a["bat"] and 28 in wp:
                bx, by, bz = a["bat"]
                hx, hy, hz = wp[28]
                dx, dy, dz = bx - hx, by - hy, bz - hz
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                # Only render the bat when it's actually being held (within
                # arm's reach of the right hand). batRootPos tracks the bat
                # as an independent object — when it's been tossed or sits
                # on-deck, it can be 100+ ft from the player.
                if dist < 3.0:
                    norm = dist or 1
                    ext = 2.5
                    tx, ty, tz = bx + dx/norm*ext, by + dy/norm*ext, bz + dz/norm*ext
                    bat_3d[ai].set_data_3d([hx, bx, tx], [hz, bz, tz], [hy, by, ty])

        if focus is not None and 21 in focus["world_pos"]:
            h = focus["world_pos"][21]
            head_marker_3d._offsets3d = ([h[0]], [h[2]], [h[1]])

        for bid, txt in labels_3d.items():
            if focus is not None and bid in focus["world_pos"]:
                p = focus["world_pos"][bid]
                txt.set_position((p[0] + 0.3, p[2]))
                txt.set_3d_properties(p[1] + 0.2, zdir="z")
                txt.set_visible(True)
            else:
                txt.set_visible(False)

        draw_2d(ax_front, 0, 1, focus, bones_front, joints_front, bat_front, labels_front)
        draw_2d(ax_side,  2, 1, focus, bones_side,  joints_side,  bat_side,  labels_side)

        for d in actor_dots.values(): d.set_data([], [])
        for a in fr["actors"]:
            if a["uid"] in actor_dots and 0 in a["world_pos"]:
                p = a["world_pos"][0]
                actor_dots[a["uid"]].set_data([p[0]], [p[2]])
        if focus is not None and 0 in focus["world_pos"]:
            p = focus["world_pos"][0]
            actor_focus_ring.set_data([p[0]], [p[2]])
        else:
            actor_focus_ring.set_data([], [])

        title3d.set_text(
            f"Frame {frame_idx + 1}/{len(frames)}    seg={fr['key'][0]} f={fr['key'][1]}    "
            f"ts={fr['ts']}    actors={len(fr['actors'])}\n"
            f"FOCUS uid={focus['uid'] if focus else '-'}   "
            f"mlb_id={focus['mlb_id'] if focus else '-'}   "
            f"type={focus['type'] if focus else '-'}"
        )

    print(f"Rendering {len(frames)} frames at 30fps...")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=33, blit=False)
    writer = FFMpegWriter(fps=30, bitrate=2800, codec="h264")
    out_path = Path(args.out) if args.out else (
        Path.home() / "Downloads"
        / f"clip_{args.game}_seg{args.start_segment}_dur{args.duration}s.mp4"
    )
    anim.save(out_path, writer=writer, dpi=85)
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
