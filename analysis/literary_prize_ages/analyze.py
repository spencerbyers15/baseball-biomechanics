"""Trend analysis: author age at Booker/Pulitzer nomination or win, 1976-2025."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import BOOKER, PULITZER, POSTHUMOUS

posthumous = {(p, y, n) for p, y, n in POSTHUMOUS}

rows = []
for prize, data, nomkey in [("Booker", BOOKER, "shortlist"), ("Pulitzer", PULITZER, "finalist")]:
    for year, e in data.items():
        for role, entries in [("winner", e["winner"]), ("nominee", e[nomkey])]:
            for name, by, title in entries:
                rows.append({
                    "prize": prize, "year": year, "role": role, "author": name,
                    "birth_year": by, "book": title, "age": year - by,
                    "posthumous": (prize.lower(), year, name) in posthumous,
                })
df = pd.DataFrame(rows)
live = df[~df.posthumous].copy()

def trend(sub, label):
    if len(sub) < 8:
        return None
    r = stats.linregress(sub.year, sub.age)
    return {
        "group": label, "n": len(sub),
        "slope_per_decade": r.slope * 10, "p": r.pvalue, "r2": r.rvalue ** 2,
        "mean_first10": sub[sub.year <= sub.year.min() + 9].age.mean(),
        "mean_last10": sub[sub.year >= sub.year.max() - 9].age.mean(),
    }

results = []
for prize in ["Booker", "Pulitzer"]:
    p = live[live.prize == prize]
    results.append(trend(p, f"{prize}: all shortlisted/finalists+winners"))
    results.append(trend(p[p.role == "winner"], f"{prize}: winners only"))
# Pulitzer finalists only exist from 1980; also run winners-only over the full window
pf = live[(live.prize == "Pulitzer") & (live.year >= 1980)]
results.append(trend(pf, "Pulitzer 1980-2025 (finalist era, all)"))
res = pd.DataFrame([r for r in results if r])
print("=== Linear trends (age vs year) ===")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

print("\n=== Decade mean ages (all nominees+winners, excl. posthumous) ===")
live["decade"] = (live.year // 10) * 10
dec = live.pivot_table(index="decade", columns="prize", values="age", aggfunc=["mean", "count"])
print(dec.round(1).to_string())

print("\n=== Decade mean ages, winners only ===")
w = live[live.role == "winner"]
decw = w.pivot_table(index="decade", columns="prize", values="age", aggfunc=["mean", "count"])
print(decw.round(1).to_string())

# Youngest/oldest for color
print("\n=== Extremes ===")
for prize in ["Booker", "Pulitzer"]:
    p = live[live.prize == prize]
    yng, old = p.loc[p.age.idxmin()], p.loc[p.age.idxmax()]
    print(f"{prize} youngest: {yng.author} {yng.age} ({yng.year}, {yng.role}); "
          f"oldest: {old.author} {old.age} ({old.year}, {old.role})")

df.to_csv("prize_ages.csv", index=False)

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
colors = {"nominee": "#9bb8d3", "winner": "#d1495b"}
for ax, prize in zip(axes, ["Booker", "Pulitzer"]):
    p = live[live.prize == prize]
    for role in ["nominee", "winner"]:
        s = p[p.role == role]
        ax.scatter(s.year, s.age, s=28 if role == "winner" else 16,
                   c=colors[role], alpha=0.75 if role == "winner" else 0.55,
                   label=role.capitalize() + ("s" if role == "winner" else "s"),
                   zorder=3 if role == "winner" else 2)
    yearly = p.groupby("year").age.mean()
    roll = yearly.rolling(7, center=True, min_periods=4).mean()
    ax.plot(roll.index, roll.values, color="#333333", lw=2.2, label="7-yr rolling mean (all)", zorder=4)
    r = stats.linregress(p.year, p.age)
    xs = np.array([p.year.min(), p.year.max()])
    ax.plot(xs, r.intercept + r.slope * xs, "--", color="#333333", lw=1.2,
            label=f"Trend: {r.slope*10:+.1f} yrs/decade (p={r.pvalue:.3f})", zorder=4)
    ax.set_title(f"{prize} Prize" + (" (finalists public from 1980)" if prize == "Pulitzer" else ""))
    ax.set_xlabel("Prize year")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("Author age at award year")
fig.suptitle("Author age at Booker shortlisting / Pulitzer Fiction finalist selection, 1976-2025\n"
             "(posthumous honorees excluded)", fontsize=13)
fig.tight_layout()
fig.savefig("prize_ages.png", dpi=150)
print("\nSaved prize_ages.csv and prize_ages.png")
