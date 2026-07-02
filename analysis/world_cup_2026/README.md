# 2026 World Cup — set-piece conversion rates

Question: what fraction of corner kicks, penalty kicks, and free kicks at the
2026 World Cup have resulted in goals?

Window: the completed **group stage** (72 matches, 215 goals, June 11–27, 2026),
the largest window with consistent published numbers as of July 2, 2026 (the
round of 32 is in progress).

## Results

| Set piece | Conversion | Numerator | Denominator |
|---|---|---|---|
| Penalty kicks | **72.7%** | 8 scored | 11 awarded (in-game, excl. shootouts) |
| Corner kicks | **~3.4%** | ~22 goals (est.) | ~630 corners (8.75/match × 72) |
| Direct free kicks (wall set, shot at goal) | **~3%** | 3 goals (documented) | ~100 attempts (est., see below) |

Regenerate the plot with:

```
python analysis/world_cup_2026/set_piece_conversion.py
```

## Methodology & caveats

- **Penalties** are exact: 11 awarded, 8 scored in the group stage
  (72.7%), per Oddspedia's penalty tracker.
- **Corner and free-kick goal totals are estimates.** ESPN's breakdown of the
  tournament's first 100 goals reported 10 goals from corners and 6 from
  free-kick situations (1 direct + 5 from free-kick deliveries). Those rates
  were scaled to the group stage's 215 goals (→ ~21.5 corner goals, ~12.9
  free-kick goals). The 3 known direct free-kick goals (Saliba vs Qatar,
  Lo Celso and Messi vs Jordan) are consistent with this.
- **Corners taken**: PerformanceOdds reports 8.75 corners per match across the
  tournament; 72 matches → ~630 corners.
- **Direct free kicks** are the "wall lined up, shooting at goal" free kicks —
  not the ~1,600 free kicks awarded anywhere on the pitch (FIFA reports 1,604
  fouls / 22.3 per match in the group stage, in line with Qatar 2022's ~24 per
  game).
  - The **3 goals are documented**: Nathan Saliba (Canada vs Qatar),
    Giovani Lo Celso and Lionel Messi (both Argentina vs Jordan).
  - **Attempt counts are not published** by FIFA, Opta (publicly), or any
    major outlet. The ~100 denominator is an estimate from the norm of recent
    World Cups: roughly 1.2–1.6 direct free-kick shots per match (consistent
    with Russia 2018's 6 direct-FK goals and Qatar 2022's 3 at a historical
    ~5–7% conversion), × 72 matches → ~85–115 attempts. The resulting rate is
    ~2.5–3.5%; the chart shows the ~3% midpoint.

## Sources

- ESPN — "Reaching the century mark: Breaking down 2026 World Cup's first 100 goals"
  (10 corner goals, 1 direct FK, 5 set-piece FK goals per first 100)
- Oddspedia — World Cup 2026 penalty tracker (11 awarded, 8 scored, 72.7%)
- FIFA (inside.fifa.com) — "Records tumble as FIFA World Cup 2026 Group Stage
  sets new benchmark" (215 goals; 1,604 fouls / 22.3 per match — official)
- Sky Sports — Qatar 2022 group-stage trends (~24 fouls per game, for comparison)
- Pulse Sports / theScore / Fox Sports — the three 2026 direct free-kick goals
  (Saliba; Lo Celso and Messi vs Jordan)
- Set-play research on Russia 2018 (6 direct-FK goals) and Qatar 2022 reports
  (3 direct-FK goals) — basis for the attempts-per-match norm
- PerformanceOdds — World Cup 2026 corner trends (8.75 corners per match)
- Fox Sports — "All 215 World Cup Group Stage Goals" (goal total)
