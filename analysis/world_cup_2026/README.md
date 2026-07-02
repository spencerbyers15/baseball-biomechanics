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
| Free kicks | **~0.8%** | ~13 goals (est.) | ~1,593 awarded (1,604 fouls − 11 penalties) |

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
- **Free kicks taken** is approximated as free kicks awarded: FIFA's official
  group-stage recap reports 1,604 fouls (22.3 per match); subtracting the 11
  penalties leaves ~1,593 non-penalty free kicks. This is *low* by historical
  standards — Qatar 2022's group stage averaged ~24 fouls per game, itself the
  fewest since records began (Sky Sports). A free kick is awarded anywhere on
  the pitch, and the large majority are midfield/defensive-half fouls nowhere
  near goal, which is why the per-free-kick conversion rate is so small.
  Counting only *direct free-kick shots* would give a higher rate, but attempt
  counts aren't published.

## Sources

- ESPN — "Reaching the century mark: Breaking down 2026 World Cup's first 100 goals"
  (10 corner goals, 1 direct FK, 5 set-piece FK goals per first 100)
- Oddspedia — World Cup 2026 penalty tracker (11 awarded, 8 scored, 72.7%)
- FIFA (inside.fifa.com) — "Records tumble as FIFA World Cup 2026 Group Stage
  sets new benchmark" (215 goals; 1,604 fouls / 22.3 per match — official)
- Sky Sports — Qatar 2022 group-stage trends (~24 fouls per game, for comparison)
- PerformanceOdds — World Cup 2026 corner trends (8.75 corners per match)
- Fox Sports — "All 215 World Cup Group Stage Goals" (goal total)
