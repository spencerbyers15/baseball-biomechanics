# Project: pre-pitch postural data → next-pitch outcome prediction → live betting edge

> **Goal**: take the skeletal data captured by the FieldVision pipeline
> (see `CLAUDE.md`), extract pre-pitch postural features, and train models
> to predict the outcome of the *next* pitch. Compare model probabilities
> to live DraftKings (or other sportsbook) odds and bet on edges.
>
> **Hypothesis**: pitcher mechanics + batter setup + fielding alignment
> contain enough signal to beat the market on at least some outcome
> distributions (pitch type, velocity tier, swing/no-swing, contact
> quality). The 30 fps full-team skeletal data is high enough resolution
> that an effective model should exist — the question is whether it
> beats market efficiency.

---

## Part 1 — What we already have

From the capture pipeline (see `CLAUDE.md`), every active MLB game lands
in `data/fv_<gamePk>.sqlite` with:

- 30 fps world-position joints for every actor on the field
- Bat handle + head per frame (when held)
- Ball world position per frame (when tracked)
- Actor identities (MLB player IDs)
- Per-actor scale, ground, apex

This is the **input side** of any pre-pitch feature.

## Part 2 — What we need (and don't yet have)

To **label** training data we need to know, for each pitch:

- **Pitch start time** (when was this pitch released?)
- **Pitch type** (FF/SL/CH/CB/etc) — categorical, ~8-15 classes
- **Pitch velocity** (mph) — continuous, ~70-105 mph
- **Pitch location** (px, pz at front of plate; in/out of strike zone) — 2D continuous + binary
- **Swing or not** (binary)
- **If swung: contact result** (whiff / foul / in play) — 3-class
- **If contact: exit velo + launch angle** — continuous
- **Final pitch outcome** (ball / strike-called / strike-swinging / foul / hit-into-play / HBP / etc)

All of this is already **in the wire format we're capturing** — under
`TrackingFrameWire.gameEvents[]` and `TrackingFrameWire.trackedEvents[]`,
which we currently skip in `ingest_segment`. Specifically:

| What | Where in the wire data |
|---|---|
| Pitch release point + trajectory | `trackedEvents[].BallPitchDataWire` |
| Pitch type, velocity, spin | `trackedEvents[].BallPitchDataWire` |
| Strike zone for the pitch | `gameEvents[].PlayEventDataWire.strikezone` |
| Balls / strikes / outs after each pitch | `gameEvents[].CountEventDataWire` |
| At-bat boundaries (batter changes) | `gameEvents[].AtBatEventDataWire` |
| Inning boundaries | `gameEvents[].InningEventDataWire` |
| Pitcher / batter handedness | `gameEvents[].HandedEventDataWire` |
| Bat contact event | `gameEvents[].BatImpactEventDataWire` |
| Defensive positioning | `gameEvents[].PositionAssignmentEventDataWire` |

The schemas for all these classes are in `gd.bvg_poser.min.js`. The
previous session decoded enough of them to know they exist; their full
field-offset lists need to be extracted (search for `getRootAs<Name>`
static methods in that bundle).

**Step 1 of this project**: extend `wire_schemas.py` to decode these
events, and extend `storage.py` to ingest them into new tables (likely
`game_event` and `pitch_event`).

We may also need **statsapi enrichment** for things the wire format
doesn't include — player names, season stats, recent form. statsapi
gives us `game/feed/live` per game which has full pitch-by-pitch data:
`https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live`.

## Part 3 — Features (the interesting part)

Pre-pitch postural data is rich. Some hypothesis-driven features to start with:

### Pitcher (~from set position through release)
- Shoulder-hip separation angle at peak
- Stride length (front-foot landing distance from pelvis)
- Front-foot landing position lateral to the rubber
- Arm slot (vector from shoulder to hand at release) → maps to ¾, sidearm, etc
- Glove tuck position (proxy for closure)
- Hip rotation velocity
- Pelvis rotation lead vs torso (kinetic chain timing)
- Time from first-move to release
- Stride foot pronation/supination (knee/ankle alignment)

### Batter (pre-pitch, while pitcher is setting up)
- Stance width (foot-to-foot distance)
- Bat angle (handle→head vector) at set
- Bat angle change in last 0.5s before delivery (timing/load)
- Open vs closed stance (front-foot lateral relative to rear-foot)
- Pelvis tilt / weight distribution (front foot vs back foot height proxy)
- Hand position (height of grip relative to shoulder)
- Distance from plate (lateral)

### Catcher
- Setup glove location (inside / outside corner) at last frame before release
- Stance height (low/high)
- Distance from plate (signal for pitch type — sometimes catchers shift)

### Fielding alignment
- Infielder positions (shifted vs standard)
- Outfielder depth (vs standard for that batter handedness)

### Situational (from gameEvents — once we ingest them)
- Count (balls/strikes)
- Outs, inning, score
- Runners on base (from `actor_type` filter: which fielders are at bases)
- Pitcher's last N pitch types (sequence dependence)
- Pitcher fatigue proxies (pitch count this AB / this inning / this game)

### Time-series structure
Frame rate is 30 fps. A typical pre-pitch window is 1-3 seconds = 30-90 frames.
Either:
- Hand-crafted features (the list above)
- Or **sequence models** on raw joint coordinates (LSTM / Transformer over T frames × J joints × 3 coords)

## Part 4 — Model targets

In rough order of "how exploitable in betting markets":

| Target | Type | Realistic accuracy ceiling | Market depth |
|---|---|---|---|
| Pitch type (FF/SL/CH/CB/etc) | multiclass (~8) | 50-70% (vs ~25% baseline) | Some books offer next-pitch type |
| Pitch velocity tier (high/low vs threshold) | binary | 65-80% | Player-prop velocity overs |
| Swing / no-swing | binary | 60-75% | Live "next pitch swung at" markets |
| Contact / no contact | binary (given swing) | 55-65% | Some books |
| In-zone / out-of-zone | binary | 60-70% | Implicit in ball/strike |
| Ball / strike (called or not) | binary | 60-70% | Live ball/strike markets exist |
| Hard-hit on contact | binary | 55-65% | Player-prop "Yes hit" markets |

Start with **swing/no-swing** and **pitch type** — both have clear
labels from the gameEvents data, both have liquid live markets.

## Part 5 — Training data math

Per game: ~250 pitches × 22+ Guardians games captured = ~5,500 labeled
pitches *just from the Guardians*. With league-wide capture continuing,
we'll have ~30 games/day × 250 pitches = ~7,500 pitches/day. A month of
capture = ~225,000 pitches. That's enough for serious model training,
even with leave-one-pitcher-out validation.

## Part 6 — Live betting integration (the hard part)

### Reality check before building anything

- Sports betting markets are reasonably efficient even at pitch level. Sharps with similar data exist.
- DraftKings, FanDuel, etc. will **limit or close** accounts that beat the market over time. The path from "model has edge" to "make money long-term" requires multiple books, careful staking, and probably betting smaller than your edge says you can.
- Pitch-level live markets close fast (a few seconds before the pitch). Your prediction pipeline must run end-to-end in single-digit seconds.
- Many jurisdictions ban use of automated betting tools. Check legality for your state.

### Architecture sketch

```
┌─────────────────────────────────────────────────────────────────┐
│  FieldVision daemon (existing)                                  │
│  → SQLite per game, updated every ~30s as new segments arrive   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Feature extractor (new)                                        │
│  Watches SQLite for new pre-pitch windows; computes features    │
│  for each pitch about to be thrown.                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Inference service (new)                                        │
│  Runs trained models on features; outputs probability dist.     │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Odds scraper (new) — DraftKings, FanDuel, BetMGM, etc.        │
│  Pulls current live odds for the matching outcomes.             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Edge calculator + alerter (new)                                │
│  Compares model P(outcome) to implied P(outcome) from odds.     │
│  When edge > threshold, alert (initially just notification;     │
│  bet placement is a separate, risky step).                      │
└─────────────────────────────────────────────────────────────────┘
```

### Latency budget

- Skeletal data lags real-time by ~10 sec (5 sec segment + decode + ingest)
- Pitch happens ~10-20 sec after pitcher gets the ball back
- → If we can extract features and predict within ~2 sec of segment arrival, we have a usable window
- The live odds market also lags slightly (book operators have lockout windows)
- This is **just barely** workable. May not work for every pitch.

### Odds scraping concerns

- Sportsbook ToS prohibits scraping and automated bet placement.
- Better path: human-in-the-loop. Model alerts → human reviews + places bet manually. Slower, but legal-ish (ToS issues remain).
- For research: scraping odds for *analysis* (not bet placement) is widely tolerated. Just don't automate placement.

## Roadmap

### Phase A — Decode the rest of the wire format

**Prerequisite (30 seconds):** make sure `samples/mlb_bundles/gd.@bvg_poser.min.js`
exists. If not, `bash scripts/fetch_mlb_bundles.sh`. This is the JS
file that contains every FlatBuffer schema we need to extract; it's a
**public, unauthenticated** CDN asset, no JWT or DevTools paste required.

1. Extend `wire_schemas.py` with: `GameEventWire`, `PlayEventDataWire`,
   `CountEventDataWire`, `AtBatEventDataWire`, `HandedEventDataWire`,
   `BallPitchDataWire`, `BallHitDataWire`, `BatImpactEventDataWire`,
   `PositionAssignmentEventDataWire`, `InningEventDataWire`,
   `ABSEventDataWire`, plus the union dispatcher (`Lue` in the bundle —
   maps `dataType` integer to which class to use).
2. Extend `storage.py` with `game_event` + `pitch_event` tables.
3. Re-ingest a couple of games to populate them.
4. Sanity-check: count pitches per game (should be ~250-350), match
   against the statsapi pitch count for that game.

### Phase B — Feature engineering

1. Define a "pre-pitch window": last N frames before each pitch's release.
2. Write feature extractors as pure functions over `actor_frame` rows.
3. Build a `pitch_features` view (or materialized table) joining
   `pitch_event` to its pre-pitch posture features.
4. Validate features make sense (e.g., velocity correlates with arm slot,
   pitch type correlates with grip-period bat angle, etc).

### Phase C — Modeling

1. Start dead simple: logistic regression for swing/no-swing on a handful
   of features. Get any signal before scaling up.
2. Add gradient boosting (XGBoost / LightGBM) for tabular features.
3. If/when tabular features plateau, try sequence models on raw joint
   coordinates over the pre-pitch window.
4. Cross-validation must be **per-pitcher** or **per-game** — random
   splits will overfit to specific pitchers' patterns.

### Phase D — Market comparison

1. Pull historical live odds if available (some data vendors sell this).
2. For each pitch in test set, compute model P(outcome) vs market implied P(outcome).
3. Measure: would betting only when model_edge > X% have been profitable historically?
4. **Be brutally honest about transaction costs, vig, and your account being limited.**

### Phase E — Live deployment

1. Scrape DraftKings live odds (probably the hardest part).
2. Build the inference + alerter service.
3. Run in *paper trading* mode for weeks before any real money.
4. Then *one book, small stakes* for months.
5. Only scale if the edge survives all of that.

## Notes for whichever Claude session picks this up

- The most valuable single thing you can do right now is **Phase A**:
  decode `gameEvents` and `trackedEvents`. Without those, you have no
  labels to train against. The schemas are in
  `gd.bvg_poser.min.js` (which Spencer can re-download via the URL
  documented in `CLAUDE.md` if it's not cached).
- Don't over-build the betting side until you've actually shown the
  model has predictive power offline. That's the gate.
- Be honest with Spencer about realistic expected returns. Beating
  sportsbooks at pitch-level live markets is genuinely hard. He's smart
  and clearly knows this; just don't oversell.
