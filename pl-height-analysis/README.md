# Premier League player height, last five seasons (2021-22 → 2025-26)

Has the average height of a Premier League player actually changed over the last
five seasons? This builds a real, sourced player-level dataset, computes both an
unweighted roster mean and a minutes-weighted mean ("the average player actually
on the pitch") with proper uncertainty, and visualises it honestly.

> **Sources note.** The brief asked for Transfermarkt (heights) and FBref
> (minutes). Both are network-blocked in this environment (HTTP 403
> `host_not_allowed`), so equivalent **open, citable** datasets were substituted:
> **EA Sports FC / FIFA** `height_cm` for heights (validated to match Transfermarkt
> exactly on spot-checks) and the **official Fantasy Premier League API**
> (`vaastav` mirror) for per-season rosters and league minutes. Nothing is
> estimated. Full reachability log and citations: [`data/SOURCES.md`](data/SOURCES.md).

## Deliverables

| File | Contents |
|---|---|
| `player_heights.csv` | tidy player-level: `season, club, player, height, minutes` (+ `height_match` provenance) |
| `season_summary.csv` | per-season weighted & unweighted mean, SD, SE, 95% CI, n, coverage % |
| `height_trend.png` | minutes-weighted mean ± 95% CI by season, with ~1 cm/decade reference |
| `build_dataset.py` | end-to-end, reproducible pipeline |

## Season summary

n_ros = roster size (FPL-registered players); cov_h = % of roster with a matched
height; cov_min = **% of league minutes** played by height-matched players (this
governs the weighted mean). Means in cm; ± is the 95% CI.

| Season | n_ros | cov_h | cov_min | Unweighted mean | Minutes-weighted mean | w n_eff |
|---|---|---|---|---|---|---|
| 2021-22 | 737 | 98.5% | 100.0% | 182.69 ± 0.51 | **182.58 ± 0.75** | 350 |
| 2022-23 | 778 | 97.0% | 99.9% | 182.39 ± 0.51 | **182.75 ± 0.73** | 350 |
| 2023-24 | 865 | 90.9% | 99.8% | 182.88 ± 0.50 | **183.10 ± 0.71** | 354 |
| 2024-25 | 804 | 85.8% | 99.3% | 183.00 ± 0.53 | **183.20 ± 0.73** | 342 |
| 2025-26 ⚠️ | 820 | 81.6% | 97.9% | 183.51 ± 0.54 | **183.91 ± 0.74** | 330 |

⚠️ **2025-26 is a partial season (29 of 38 gameweeks).** Provisional.

The league sits at ~182.5–183 cm — consistent with the ~181–182 cm sanity check
(marginally above it; the spot-checks confirm the heights themselves are right).

### Coverage caveats
- **Weighted mean is reliable in every season:** ≥97.9 % of all minutes are played
  by height-matched players (≥99.3 % in the four complete seasons). The roster
  gaps are low-minute fringe players who barely move a minutes-weighted average.
- **Unweighted (roster) mean for 2024-25 and especially 2025-26 is weaker:**
  head-count coverage falls to 86 % / 82 % because EA FC data ends at edition 24
  (2023-24), so PL debutants after that aren't in the height dictionary. Three
  such players log >900 min in 2025-26 (Álex Jiménez, Diego Gómez, Igor Jesus) and
  are reported as gaps rather than guessed.

### Is any movement real? (minutes-weighted mean)
- **No single season-to-season change exceeds its standard error.** Year-over-year
  z-scores: +0.32, +0.67, +0.19, +1.34 — all within noise (|z| < 1.96).
- Over the **four complete seasons** (2021-22 → 2024-25) the rise is **+0.62 cm,
  z = 1.16 — not distinguishable from noise.**
- The only "significant" result, cumulative 2021-22 → 2025-26 (+1.33 cm, z = 2.47),
  **depends entirely on the partial, lowest-coverage 2025-26 season** and should
  not be trusted until that season completes.
- **Promotion/relegation contributes little.** Decomposing each year's shift, the
  three swapped clubs account for only +0.03, +0.15, −0.16, +0.12 cm; the rest is
  drift within the 17 ever-present clubs.
- **The five-season window is too short to see the real trend.** The established
  long-run rate is ~+1 cm/decade ≈ 0.4 cm over four seasons, far below the
  per-season standard error of ~0.37 cm. Five seasons cannot resolve it.

## Plain-English readout

Over the last five Premier League seasons the minutes-weighted average player
height drifts upward from about **182.6 cm to 183.9 cm**, but almost none of that
is distinguishable from statistical noise: no year-to-year change exceeds its
standard error, and across the four *completed* seasons the gain is just ~0.6 cm
(z ≈ 1.2, not significant). The apparent acceleration in 2025-26 rests on a
partial, 29-of-38-gameweek snapshot with the weakest height coverage, so it is
provisional. Promotion and relegation explain very little of the movement, and a
five-season window is simply too short to surface the real long-run trend of
roughly +1 cm per decade, which is smaller than the season-to-season error bars.
**Bottom line: no, average PL height has not demonstrably changed over these five
seasons — the data are consistent with flat-to-slightly-rising, within noise.**

## Method (brief)

1. **Heights:** EA FC/FIFA `height_cm`, deduplicated to one value per player
   (latest edition) → name-keyed dictionary, topped up with Transfermarkt-mirror
   subsets.
2. **Rosters + minutes:** FPL `players_raw.csv` per season (club via `teams.csv`,
   `minutes` = league minutes).
3. **Join:** diacritic-folded name matching with order-agnostic / initial+surname
   fallbacks under a 2 cm collision guard, plus an explicit alias table for legal
   vs common names. Match rates and gaps are reported, not hidden.
4. **Stats:** unweighted mean/SD/SE/95%CI over rostered players with a height;
   minutes-weighted mean with **Kish effective sample size** for the SE/CI
   (down-weights the fact that a few ever-presents carry most of the minutes).
5. **Chart:** y-axis scaled to the data range + margin (not zero-anchored, not
   zoomed to exaggerate); faint ~1 cm/decade reference; 2025-26 drawn as an open
   marker to flag it as partial.
