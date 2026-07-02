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
- **Free kicks taken** is approximated as free kicks awarded: FIFA's group-stage
  recap reports 1,604 fouls; subtracting the 11 penalties leaves ~1,593
  non-penalty free kicks. Most of these are nowhere near goal, which is why the
  free-kick conversion rate is so low. Counting only *direct free-kick shots*
  would give a higher rate, but attempt counts aren't published.

## Sources

- ESPN — "Reaching the century mark: Breaking down 2026 World Cup's first 100 goals"
  (10 corner goals, 1 direct FK, 5 set-piece FK goals per first 100)
- Oddspedia — World Cup 2026 penalty tracker (11 awarded, 8 scored, 72.7%)
- FIFA — "The group stage in stats" (215 goals; 1,604 fouls / 22.3 per match)
- PerformanceOdds — World Cup 2026 corner trends (8.75 corners per match)
- Fox Sports — "All 215 World Cup Group Stage Goals" (goal total)
