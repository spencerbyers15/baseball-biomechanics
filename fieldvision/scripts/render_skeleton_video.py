"""Render a 3D skeleton animation from decoded FieldVision data.

Layout (one focus actor for inspection + all actors top-down):
  ┌─────────────────────┬──────────────┐
  │                     │  Top-down    │
  │   3D perspective    │  field       │
  │                     ├──────────────┤
  │                     │  Front view  │
  │                     ├──────────────┤
  │                     │  Side view   │
  └─────────────────────┴──────────────┘

For each actor in each frame:
  1. Decode all 20 packed quaternions to (x,y,z,w) using the exact
     maxValue=0.7072 smallest-three algorithm (matches gd.@bvg_poser.min.js).
  2. Apply forward kinematics with the corrected rest pose (MLB convention:
     pelvis bone local +Y points DOWN the spine).
  3. Plot stick-figure connections.

Output: ~/Downloads/decoded_<gamePk>_skeletons_<dur>s.mp4
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fieldvision.skeleton import (BONE_NAMES, SKELETON_CONNECTIONS, forward_kinematics)
from fieldvision.wire_schemas import read_tracking_data, unpack_smallest_three


SAMPLES_DIR = Path("samples/binary_capture_823141")
GAME_PK = 823141

# Short labels per bone ID for the visualization
JOINT_LABEL = {
    0: "Pelv", 1: "HipM",
    2: "R.Hip", 3: "R.Knee", 4: "R.Foot",
    10: "L.Hip", 11: "L.Knee", 12: "L.Foot",
    18: "TorA", 19: "TorB",
    20: "Neck", 21: "Head",
    25: "R.Clav", 26: "R.Shdr", 27: "R.Elb", 28: "R.Hand",
    64: "L.Clav", 65: "L.Shdr", 66: "L.Elb", 67: "L.Hand",
}
# In the 3D view we only label the endpoints + pelvis to keep it readable
JOINT_LABEL_3D = {0: "Pelv", 21: "Head", 4: "R.Foot", 12: "L.Foot",
                  28: "R.Hand", 67: "L.Hand"}


def load_all_frames():
    seg_paths = sorted(SAMPLES_DIR.glob(f"mlb_{GAME_PK}_segment_*.bin"),
                       key=lambda p: int(p.stem.split("_")[-1]))
    print(f"Decoding {len(seg_paths)} segments...")
    frames = []
    for path in seg_paths:
        td = read_tracking_data(path.read_bytes())
        for f in td.frames:
            decoded_actors = []
            for a in f.actorPoses:
                if not a.rootPos:
                    continue
                quats = [unpack_smallest_three(p) for p in a.packedQuats]
                ws = forward_kinematics(
                    root_pos=(a.rootPos.x, a.rootPos.y, a.rootPos.z),
                    scale=a.scale if a.scale > 0 else 1.0,
                    node_ids=a.nodeIds,
                    quats_xyzw=quats,
                )
                decoded_actors.append({
                    "uid": a.uid,
                    "world_pos": ws.bone_world_pos,
                    "rootPos": (a.rootPos.x, a.rootPos.y, a.rootPos.z),
                    "ground": a.ground,
                    "bat": (a.batRootPos.x, a.batRootPos.y, a.batRootPos.z) if a.batRootPos else None,
                })
            frames.append({
                "ts": f.timestamp,
                "time": f.time,
                "actors": decoded_actors,
            })
    print(f"Total frames: {len(frames)}")
    return frames


def main():
    frames = load_all_frames()

    uid_counts = Counter(a["uid"] for f in frames for a in f["actors"])
    most_common_uid, count = uid_counts.most_common(1)[0]
    print(f"\nFocus actor: uid={most_common_uid}  (in {count}/{len(frames)} frames)")

    labels = json.load(open(SAMPLES_DIR / f"mlb_{GAME_PK}_labels.json"))
    info = labels.get(str(most_common_uid), {})
    actor_id = info.get("actor", most_common_uid)
    actor_type = info.get("type", "?")

    # Layout: 3D view spans 3 rows on left; right column has 3 panels
    fig = plt.figure(figsize=(16, 10), facecolor="#222")
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 2, 1.3], height_ratios=[1, 1, 1],
                          hspace=0.25, wspace=0.18, left=0.04, right=0.98, top=0.94, bottom=0.05)
    ax3d = fig.add_subplot(gs[:, 0:2], projection="3d")
    ax_front = fig.add_subplot(gs[0, 2])
    ax_side = fig.add_subplot(gs[1, 2])
    ax_top = fig.add_subplot(gs[2, 2])

    # Color per uid
    all_uids = sorted(uid_counts)
    palette = plt.cm.tab10.colors
    color_map = {uid: palette[i % len(palette)] for i, uid in enumerate(all_uids)}

    max_actors = max(len(f["actors"]) for f in frames)

    # 3D view artists
    bones_3d = [[ax3d.plot([], [], [], "-", lw=2.5, color="white",
                           solid_capstyle="round")[0] for _ in SKELETON_CONNECTIONS]
                for _ in range(max_actors)]
    joints_3d = [ax3d.scatter([], [], [], s=18, c="cyan", depthshade=False)
                 for _ in range(max_actors)]
    bat_3d = [ax3d.plot([], [], [], "-", lw=4, color="#d4a04c")[0] for _ in range(max_actors)]
    head_marker_3d = ax3d.scatter([], [], [], s=80, c="red", depthshade=False, marker="o")

    # Front view (looking from -Z toward +Z, so X is horizontal, Y is vertical)
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
    ax_front.set_facecolor("#3a5a3a")
    ax_front.set_aspect("equal")
    ax_front.set_title("Focus actor — FRONT view (labeled)", color="white", fontsize=10)
    ax_front.tick_params(colors="white", labelsize=7)

    # Side view (looking from -X toward +X, so Z is horizontal, Y is vertical)
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
    ax_side.set_facecolor("#3a5a3a")
    ax_side.set_aspect("equal")
    ax_side.set_title("Focus actor — SIDE view (labeled)", color="white", fontsize=10)
    ax_side.tick_params(colors="white", labelsize=7)

    # 3D view labels (just key endpoints to avoid clutter)
    labels_3d = {bid: ax3d.text(0, 0, 0, label, color="yellow", fontsize=8)
                 for bid, label in JOINT_LABEL_3D.items()}

    # Top-down field
    ax_top.set_facecolor("#3a5a3a")
    ax_top.set_xlim(-300, 300)
    ax_top.set_ylim(-450, 60)
    ax_top.set_aspect("equal")
    ax_top.tick_params(colors="white", labelsize=7)
    ax_top.set_title("Top-down field (all actors)", color="white", fontsize=10)
    # field outline
    ax_top.plot([0, 270], [0, -270], "w-", alpha=0.4, lw=1)
    ax_top.plot([0, -270], [0, -270], "w-", alpha=0.4, lw=1)
    diamond_xs = [0, 90/np.sqrt(2), 0, -90/np.sqrt(2), 0]
    diamond_ys = [0, -90/np.sqrt(2), -90*np.sqrt(2), -90/np.sqrt(2), 0]
    ax_top.plot(diamond_xs, diamond_ys, "w-", alpha=0.5, lw=1)
    actor_dots = [ax_top.plot([], [], "o", ms=6, color=color_map[uid])[0] for uid in all_uids]
    actor_focus_ring = ax_top.plot([], [], "o", ms=15, mfc="none", mec="yellow", mew=1.5)[0]

    # 3D view styling
    ax3d.set_facecolor("#3a5a3a")
    ax3d.tick_params(colors="white")
    ax3d.set_xlabel("X — 3B/1B  (ft)", color="white")
    ax3d.set_ylabel("Z — depth  (ft)", color="white")
    ax3d.set_zlabel("Y — up  (ft)", color="white")
    title3d = ax3d.set_title("", color="white", fontsize=12)
    # Hide pane backgrounds for cleaner look
    for ax_pane in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        ax_pane.set_pane_color((0.2, 0.2, 0.2, 1))
    ax3d.view_init(elev=15, azim=-50)

    def draw_skeleton_2d(ax, x_axis: str, y_axis: str, focus_actor, lines, joint_scatter, bat_line, labels_dict):
        """Draw the focus actor's skeleton onto a 2D axis using the chosen
        stadium axes. x_axis/y_axis ∈ {'x','y','z'}."""
        for ln in lines[0]:
            ln.set_data([], [])
        joint_scatter.set_offsets(np.empty((0, 2)))
        bat_line.set_data([], [])
        for tx in labels_dict.values():
            tx.set_visible(False)
        if focus_actor is None:
            return
        wp = focus_actor["world_pos"]
        ax_idx = {"x": 0, "y": 1, "z": 2}
        ix, iy = ax_idx[x_axis], ax_idx[y_axis]
        for bi, (a_id, b_id) in enumerate(SKELETON_CONNECTIONS):
            if a_id in wp and b_id in wp:
                p1, p2 = wp[a_id], wp[b_id]
                lines[0][bi].set_data([p1[ix], p2[ix]], [p1[iy], p2[iy]])
        pts = np.array([[p[ix], p[iy]] for p in wp.values()])
        joint_scatter.set_offsets(pts)
        # Joint labels — offset slightly to the right of each joint
        for bid, p in wp.items():
            if bid in labels_dict:
                labels_dict[bid].set_position((p[ix] + 0.15, p[iy]))
                labels_dict[bid].set_text(JOINT_LABEL[bid])
                labels_dict[bid].set_visible(True)
        # Center the axis on the actor's pelvis
        cx = focus_actor["rootPos"][ix]
        cy = focus_actor["rootPos"][iy]
        ax.set_xlim(cx - 5, cx + 5)
        ax.set_ylim(cy - 5, cy + 5)

    def update(frame_idx):
        fr = frames[frame_idx]
        focus = next((a for a in fr["actors"] if a["uid"] == most_common_uid), None)
        if focus is None and fr["actors"]:
            focus = fr["actors"][0]

        # 3D camera follows focus actor
        if focus is not None:
            cx, cy, cz = focus["rootPos"]
            r = 7.0
            ax3d.set_xlim(cx - r, cx + r)
            ax3d.set_ylim(cz - r, cz + r)
            ax3d.set_zlim(0, r * 1.6)

        # Reset 3D artists
        for actor_lines in bones_3d:
            for ln in actor_lines:
                ln.set_data_3d([], [], [])
        for s in joints_3d:
            s._offsets3d = ([], [], [])
        for ln in bat_3d:
            ln.set_data_3d([], [], [])
        head_marker_3d._offsets3d = ([], [], [])

        # Plot 3D
        for ai, a in enumerate(fr["actors"]):
            if ai >= max_actors:
                break
            color = color_map[a["uid"]]
            wp = a["world_pos"]
            for bi, (a_id, b_id) in enumerate(SKELETON_CONNECTIONS):
                if a_id in wp and b_id in wp:
                    p1, p2 = wp[a_id], wp[b_id]
                    bones_3d[ai][bi].set_data_3d(
                        [p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]]
                    )
                    bones_3d[ai][bi].set_color(color)
            joint_xs = [p[0] for p in wp.values()]
            joint_ys = [p[2] for p in wp.values()]
            joint_zs = [p[1] for p in wp.values()]
            joints_3d[ai]._offsets3d = (joint_xs, joint_ys, joint_zs)
            joints_3d[ai].set_color(color)
            # Bat (if present): a short line from batRootPos pointing in player's facing direction
            if a["bat"] is not None and 28 in wp:
                bx, by, bz = a["bat"]
                # Hand position
                hx, hy, hz = wp[28]
                # Draw line from hand to bat root then extend ~2.5 ft in same direction
                dx, dy, dz = bx - hx, by - hy, bz - hz
                norm = (dx*dx + dy*dy + dz*dz) ** 0.5 or 1
                ext = 2.5
                tx, ty, tz = bx + dx/norm * ext, by + dy/norm * ext, bz + dz/norm * ext
                bat_3d[ai].set_data_3d([hx, bx, tx], [hz, bz, tz], [hy, by, ty])

        # Mark head joint of focus actor in 3D
        if focus is not None and 21 in focus["world_pos"]:
            h = focus["world_pos"][21]
            head_marker_3d._offsets3d = ([h[0]], [h[2]], [h[1]])

        # 3D labels for key joints on the focus actor
        for bid, txt in labels_3d.items():
            if focus is not None and bid in focus["world_pos"]:
                p = focus["world_pos"][bid]
                txt.set_position((p[0] + 0.3, p[2]))
                txt.set_3d_properties(p[1] + 0.2, zdir="z")
                txt.set_visible(True)
            else:
                txt.set_visible(False)

        # Front view: x_stadium horizontal, y_stadium vertical
        draw_skeleton_2d(ax_front, "x", "y", focus, bones_front, joints_front, bat_front, labels_front)
        # Side view: z_stadium horizontal, y_stadium vertical
        draw_skeleton_2d(ax_side, "z", "y", focus, bones_side, joints_side, bat_side, labels_side)

        # Top-down dots for all actors
        for di, uid in enumerate(all_uids):
            actor_dots[di].set_data([], [])
        for a in fr["actors"]:
            di = all_uids.index(a["uid"])
            actor_dots[di].set_data([a["rootPos"][0]], [a["rootPos"][2]])
        if focus is not None:
            actor_focus_ring.set_data([focus["rootPos"][0]], [focus["rootPos"][2]])
        else:
            actor_focus_ring.set_data([], [])

        title3d.set_text(
            f"Frame {frame_idx + 1}/{len(frames)}    ts={fr['ts']}    "
            f"actors={len(fr['actors'])}\n"
            f"FOCUS: uid={focus['uid'] if focus else '-'}   "
            f"actor={actor_id}   type={actor_type}"
        )

        return (
            [title3d]
            + [ln for actor in bones_3d for ln in actor]
            + joints_3d
            + bat_3d
            + [head_marker_3d]
            + list(labels_3d.values())
            + bones_front[0] + bones_side[0]
            + [joints_front, joints_side, bat_front, bat_side]
            + list(labels_front.values()) + list(labels_side.values())
            + actor_dots
            + [actor_focus_ring]
        )

    print(f"Rendering {len(frames)} frames at 30fps...")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=33, blit=False)
    writer = FFMpegWriter(fps=30, bitrate=2800, codec="h264")
    duration_s = (frames[-1]["time"] - frames[0]["time"]) if frames else 0
    out_path = Path.home() / "Downloads" / f"decoded_{GAME_PK}_skeletons_{int(duration_s)}s.mp4"
    anim.save(out_path, writer=writer, dpi=85)
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
