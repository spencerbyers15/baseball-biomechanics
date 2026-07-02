"""2026 World Cup set-piece conversion rates — bar plot.

What fraction of corner kicks, penalty kicks, and free kicks resulted in a
goal, through the group stage (72 matches, 215 goals, June 11-27, 2026).

Numbers and sources are documented in README.md alongside this script.
Chart style follows the dataviz reference palette (single-hue bars, thin
marks, rounded data-ends, direct labels, recessive axes).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- data
# label, pct (0-100), numerator description, denominator description
ROWS = [
    ("Penalty kicks", 8 / 11 * 100, "8 of 11 scored"),
    ("Corner kicks", 21.5 / 630 * 100, "~22 goals est. / ~630 corners"),
    ("Direct free kicks", 3 / 100 * 100,
     "wall set, shot at goal — 3 goals / ~100 est. attempts"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

labels = [r[0] for r in ROWS]
values = [r[1] for r in ROWS]
notes = [r[2] for r in ROWS]
ys = list(range(len(ROWS)))[::-1]  # penalty on top

XMAX = 100

ax.set_xlim(0, XMAX)
ax.set_ylim(-0.55, len(ROWS) - 0.45)

# gridlines under the bars
ax.set_axisbelow(True)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.yaxis.grid(False)

# draw bars with a 4px-rounded data-end, square at the baseline
fig.canvas.draw()  # settle layout so pixel<->data scale is known
x_per_px = XMAX / ax.get_window_extent().width
y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / ax.get_window_extent().height
r = 4 * x_per_px  # 4px rounding, in x data units
BAR_H = 24 * y_per_px  # cap bar thickness at 24px

for y, v in zip(ys, values):
    box = FancyBboxPatch(
        (0, y - BAR_H / 2), v, BAR_H,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=(y_per_px / x_per_px),
        facecolor=BLUE, edgecolor="none", zorder=3,
    )
    ax.add_patch(box)
    # square off the baseline end
    ax.add_patch(Rectangle((0, y - BAR_H / 2), min(r * 2, v / 2), BAR_H,
                           facecolor=BLUE, edgecolor="none", zorder=3))

# direct labels at the bar tips (text tokens, never the series color)
for y, v, note in zip(ys, values, notes):
    ax.text(v + 1.4, y + 0.13, f"{v:.1f}%", va="center", ha="left",
            fontsize=12, fontweight="bold", color=INK)
    ax.text(v + 1.4, y - 0.15, note, va="center", ha="left",
            fontsize=8.5, color=MUTED)

# axes chrome
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=11, color=INK_2)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=MUTED)
ax.tick_params(length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.spines["bottom"].set_linewidth(1)

ax.set_title("Set pieces that ended in a goal — 2026 World Cup",
             loc="left", fontsize=14, fontweight="bold", color=INK, pad=26)
ax.text(0, 1.06, "Group stage: 72 matches, 215 goals (Jun 11–27, 2026). "
        "Penalties exclude shootouts.",
        transform=ax.transAxes, fontsize=9.5, color=INK_2)

fig.text(0.055, 0.015,
         "Corner goals est. from ESPN's first-100-goals breakdown scaled to 215 goals; "
         "corners: 8.75/match (PerformanceOdds). Penalties: 8/11 (Oddspedia).\n"
         "Direct FK goals: Saliba, Lo Celso, Messi. Attempts unpublished; ~100 est. from "
         "the 1.2–1.6 shots/match norm of recent World Cups.",
         fontsize=7.5, color=MUTED, va="bottom")

fig.subplots_adjust(left=0.165, right=0.97, top=0.82, bottom=0.20)
fig.savefig("analysis/world_cup_2026/set_piece_conversion.png",
            facecolor=SURFACE, dpi=200)
print("wrote analysis/world_cup_2026/set_piece_conversion.png")
