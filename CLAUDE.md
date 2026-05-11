# FieldVision — Claude Code project context

> **For Claude:** Read this top-to-bottom before doing anything. It's the
> handoff from the previous session that built this project. Things
> referenced here are real and tested; gotchas are real and bit us.

## What this is

A pipeline that captures **MLB Hawk-Eye skeletal tracking data** (30 fps,
20 joints per actor, ~16 actors per frame: every fielder + batter +
catcher + umpires) from MLB's public Gameday infrastructure, decodes the
FlatBuffer wire format, applies forward kinematics with the real GLTF
rest pose, and stores everything in queryable SQLite. Plus the ball
position when tracked, plus the **bat orientation** (handle + head world
positions, ~34 inch axis).

Spencer's downstream goal: **biomechanics research** — and now also
**pre-pitch postural features → next-pitch outcome prediction → live
betting market edge**. See `PROJECT_PREDICTION.md` for the new work.

## TL;DR for a new session

```bash
# Verify what's running
launchctl list | grep fvcapture          # daemon should have a PID
tail -f ~/fieldvision/scheduler.log      # live capture activity
tail -f ~/fieldvision/guardians_scrape.log  # bulk historical (~7h job)

# Where data lives
ls ~/fieldvision/data/                   # one SQLite per game, 3-4 GB each
ls ~/fieldvision/samples/                # raw .bin files (usually deleted)

# Query example: pelvis trajectory for one player in one game
sqlite3 ~/fieldvision/data/fv_823141.sqlite \
  "SELECT time_unix, pelvis_x, pelvis_y, pelvis_z FROM actor_frame
   WHERE mlb_player_id = 660844 ORDER BY time_unix LIMIT 30;"
```

## Repo layout

```
~/fieldvision/                      ⚠ MUST be outside ~/Documents (macOS TCC blocks launchd otherwise)
├── CLAUDE.md                       this file
├── README.md                       (original MLB-side reverse-engineering notes — partly outdated)
├── FVCAPTURE_SETUP.md              daemon install runbook
├── MIGRATE_TO_PI.md                instructions to move daemon to a Raspberry Pi
├── PROJECT_PREDICTION.md           the new pitch-outcome modeling + betting project
├── .fv_token.txt                   ⚠ gitignored. Bearer JWT, refresh ~daily via DevTools snippet
├── requirements.txt
├── launchd/
│   └── com.spencerbyers.fvcapture.plist   launchd unit (installed at ~/Library/LaunchAgents/)
├── src/fieldvision/
│   ├── flatbuf_runtime.py          hand-rolled FlatBuffer reader (no external dep)
│   ├── wire_schemas.py             TrackingDataWire / TrackingFrameWire / ActorPoseWire /
│   │                               SkeletalPlayerWire / InferredBat + Vec3 + quaternion decoder
│   ├── skeleton.py                 GLTF rest pose + bone hierarchy + forward kinematics
│   └── storage.py                  SQLite schema + insertion helpers
├── scripts/
│   ├── fv_daemon.py                ⭐ the always-on capture daemon (managed by launchd)
│   ├── scrape_full_game.py         one-shot bulk download of a single game
│   ├── scrape_team_history.py      bulk download every available game for a team
│   ├── load_to_db.py               decode existing .bin files into SQLite
│   ├── load_bat_positions.py       backfill bat_frame from existing .bin files
│   ├── render_clip.py              matplotlib 3D + front/side/top skeleton video
│   ├── render_video.py             top-down field dot-trail video
│   ├── decode_segment.py           debug: decode one .bin to JSON
│   ├── dump_segment.py             debug: hex/float dump of a .bin
│   ├── snippets/refresh_token.js   the DevTools paste snippet for refreshing the JWT
│   └── (legacy)                    auth_probe.py, setup_login.py, find_*.py — kept for
│                                   reference; not part of the live pipeline
├── data/                           gitignored. SQLite stores, ~3-4 GB per game
├── samples/                        gitignored. Raw .bin segments (usually deleted post-ingest)
└── state/                          gitignored. last_segment_<pk>.txt + token_expired.flag
```

## The data model

### Per-game SQLite (`data/fv_<gamePk>.sqlite`)

**`actor_frame`** — one row per actor per frame (~5 M rows / game). PK = (game_pk, segment_idx, frame_num, actor_uid).

Columns:
- Identification: `game_pk`, `segment_idx`, `frame_num`, `actor_uid`,
  `mlb_player_id`, `actor_type` (`fielder`/`batter`/`pitcher`/`catcher`/`plate-umpire`/`umpire`/`coach`)
- Timing: `time_unix`, `timestamp` (ISO-8601), `is_gap`
- Per-actor scalars: `scale` (player height proxy), `ground`, `apex`
- **20 joints × x/y/z** in stadium feet, Y up, home plate at origin:
  `pelvis_x/y/z`, `hipmaster_x/y/z`,
  `hip_rt_x/y/z`, `knee_rt_x/y/z`, `foot_rt_x/y/z`,
  `hip_lt_x/y/z`, `knee_lt_x/y/z`, `foot_lt_x/y/z`,
  `torso_a_x/y/z`, `torso_b_x/y/z`, `neck_x/y/z`, `head_x/y/z`,
  `clavicle_rt_x/y/z`, `shoulder_rt_x/y/z`, `elbow_rt_x/y/z`, `hand_rt_x/y/z`,
  `clavicle_lt_x/y/z`, `shoulder_lt_x/y/z`, `elbow_lt_x/y/z`, `hand_lt_x/y/z`
- `bat_handle_x/y/z` — the bat's grip world position when this actor has the bat. **Note: this is the bat associated with the actor, not the canonical bat. Use `bat_frame` for the actual per-frame bat axis.**

**`bat_frame`** — per-frame bat orientation (NOT per-actor). PK = (game_pk, segment_idx, frame_num).
- `time_unix`, `head_x/y/z` (barrel tip), `handle_x/y/z` (grip)
- Length is consistently ~2.84 ft (34 inches). Vector `head - handle` = bat axis. Populated in ~70-75% of frames (between innings / pre-game is gap).

**`ball_frame`** — ball position when tracked. PK same shape.
- `ball_x/y/z` in stadium feet, Y up.

**`players`, `bones`, `labels`, `meta`** — small lookup tables.
- `players(mlb_player_id, jersey_number, role_id, team, position_abbr, parent_team_id)`
- `bones(bone_id, name)` — 0=Pelvis, 21=Head, 28=HandRT, etc. (canonical map from metadata.json's boneIdMap)
- `labels(actor_uid, actor, actor_type)` — maps the .bin's internal actor_uid to MLB player ID (or negative IDs for umpires: -14=plate-umpire, -15/-16/-17/-18=base umpires)
- `meta(key, value)` — version, gamePk, venueId, rule.strikeZoneTopFactor, rule.strikeZoneBottomFactor, etc.

### Cross-game registry (`data/games_registry.sqlite`)

One row per captured game. Useful for `SELECT db_path FROM games WHERE captured_at > ...`.

### NOT YET INGESTED but available in the wire format

`TrackingFrameWire` ALSO contains, per frame, fields we currently skip:

- **`gameEvents[]`** — union over discrete events (typed via `dataType`):
  - `CountEventDataWire` (balls/strikes/outs)
  - `PlayEventDataWire` (with strike zone)
  - `AtBatEventDataWire`
  - `InningEventDataWire`
  - `HandedEventDataWire` (batter/pitcher handedness)
  - `PositionAssignmentEventDataWire`
  - `TeamScoreEventDataWire`
  - `BattingOrderEventDataWire`
  - `BatImpactEventDataWire`
  - `HighFrequencyBatMarkerEventDataWire`
  - `ABSEventDataWire` (automated ball-strike zone)
  - `LiveActionEventDataWire`, `GumboTimecodeEventDataWire`, `StatusEventDataWire`
- **`trackedEvents[]`** — tracked physics events:
  - `BallPitchDataWire` (with release point, trajectory, spin)
  - `BallHitDataWire`
  - `BallThrowData`
  - `BallBounceDataWire`
- **`ballPolynomials[]`** — pitch trajectory polynomials
- **`inferredBat`** — already decoded into `bat_frame`

**Adding ingestion for these is the single most valuable next step** —
they give us pitch-by-pitch labels (which we need for the prediction
project). The schemas for all of these classes are in
`gd.bvg_poser.min.js`; the previous session has all the field offsets
documented in their code (search for `getRootAs...` static methods).

## Capture pipeline

### Live capture (daemon)

```bash
launchctl list | grep fvcapture   # PID = running
tail -f ~/fieldvision/scheduler.log
```

The daemon (`scripts/fv_daemon.py`) polls statsapi every 10 min, scrapes
each live game's new segments every 30s, and ingests into the per-game
SQLite. State per game in `state/last_segment_<pk>.txt`.

### Historical bulk

```bash
python scripts/scrape_team_history.py --team-id 114 --delete-bins        # all Guardians available
python scripts/scrape_full_game.py --game 823141 --parallel 3            # one game
```

⚠ **Retention window**: fieldvision-hls keeps games for only ~3-4 weeks.
Games older than that return HTTP 404 and are gone forever from this
data source.

### Token refresh (manual, ~daily)

The Okta access token in `.fv_token.txt` expires every ~24h. To refresh:

1. Open mlb.com (logged in) in Chrome, open DevTools (`Cmd+Opt+J`).
2. Paste `scripts/snippets/refresh_token.js` into the Console, run.
3. `mv ~/Downloads/fv_token.txt ~/fieldvision/.fv_token.txt`.
4. Daemon picks it up on next poll (≤ 5 min).

**Open question for future work**: implementing OAuth refresh-token auto-refresh.
Current evidence: access token claims show `scp: ['openid', 'email']` (no
`offline_access`), suggesting MLB isn't issuing a refresh token. Probe
`okta-token-storage` to confirm — if no refresh token, the path forward
is either a headless-Chrome-stays-logged-in setup (silent refresh via
hidden iframe) or an iOS shortcut + iCloud-watcher to automate the daily
paste from Spencer's phone.

## Critical gotchas (each of these cost us hours)

1. **macOS TCC blocks launchd from reading `~/Documents`.** Repo MUST live outside `~/Documents`, `~/Desktop`, `~/Downloads`. We moved it to `~/fieldvision/`. If you re-organize, keep it out of those folders or grant Full Disk Access to the conda Python.

2. **`Python-urllib/...` User-Agent gets HTTP 403 from fieldvision-hls.** Always set a desktop Chrome UA on any request to that host. See `USER_AGENT` constant in `scripts/fv_daemon.py`.

3. **`Origin: networkidle` never resolves on mlb.com.** SPA keeps long-poll connections open. Use `domcontentloaded` if you ever revive any Playwright path.

4. **The previous CC session's reverse-engineering of the LIVE in-browser Three.js scene is OBSOLETE as of ~Apr 22 2026.** MLB replaced the client-side WebGL FieldVision engine with a server-rendered HLS video stream. The `.bin` endpoint at `fieldvision-hls.mlbinfra.com` (which served the same Three.js engine via direct binary segments) is what we scrape now — and it's still alive but rate-limited and retention-limited. Do not waste time trying to find `window.__pc.world.actors` in the page; it's not there.

5. **Two different bone-numbering systems in the schemas:** the canonical `boneIdMap` from `metadata.json` (0=Pelvis, 21=Head, 28=HandRT...) is what we use everywhere. The old session's notes referenced different indices ([81=Neck, 83=Head, ...]) which were Three.js runtime-array positions; those don't apply to anything we do now.

6. **`bat_handle` on `actor_frame` ≠ canonical bat.** The per-actor `bat_handle_x/y/z` is the bat associated with that actor (close to hands when held, can be 100+ ft away when tossed). For the canonical per-frame bat axis, use `bat_frame` (head + handle from `inferredBat`).

7. **Quaternion bit layout**: 32-bit packed "smallest three" encoding with `maxValue=0.7072`. See `unpack_smallest_three()` in `wire_schemas.py`. Direct port of `zP.unpack` from `gd.bvg_poser.min.js`. Applied to the rest-pose rotation with REPLACE semantics (not compose) — confirmed empirically.

8. **Rate limits**: concurrency 8 gets HTTP 429 from MLB. Stick to 3 or fewer concurrent segment fetches with 0.3s delay between requests on the same game. Exponential backoff is already implemented in `http_get()`.

## Current state (as of 2026-05-07)

- Daemon running, capturing today's 4 live games
- Bulk Guardians scrape running in background (~7h), already captured games 1-3 of 21
- 21 historical Guardians games (Apr 14 → May 6) being saved before retention expires
- Token in `.fv_token.txt` expires in ~24h from when Spencer last pasted it
- Spencer is going on vacation soon; we agreed to defer Pi migration and auto-refresh work until he's back

## Open / deferred work

1. **OAuth refresh-token investigation** — probe `okta-token-storage`, decide on autonomous-refresh strategy
2. **Pi 4 migration** — guide is in `MIGRATE_TO_PI.md`, ready to execute
3. **Ingest `gameEvents` + `trackedEvents` + `ballPolynomials`** — required for the prediction project (see `PROJECT_PREDICTION.md`)
4. **Investigate `rawJoints` field** — 0/1500 frames had it in our samples; might be live-only or specific to certain segments

## How to verify things work after a context-free start

```bash
# 1. Is the daemon alive?
launchctl list | grep fvcapture

# 2. Is the bulk scrape still going?
ps aux | grep scrape_team_history | grep -v grep

# 3. Is the token still valid?
python3 -c "
import base64, json, time
tok = open('/Users/spencerbyers/fieldvision/.fv_token.txt').read().strip()
p = tok.split('.')[1] + '=' * (-len(tok.split('.')[1]) % 4)
c = json.loads(base64.urlsafe_b64decode(p))
print(f'expires in {(c[\"exp\"] - time.time()) / 3600:.1f} hours')
"

# 4. How much data captured?
ls ~/fieldvision/data/ | grep '\.sqlite$' | wc -l
du -sh ~/fieldvision/data/

# 5. End-to-end smoke test (read a recent pose from SQLite)
sqlite3 ~/fieldvision/data/fv_823141.sqlite "SELECT COUNT(*) FROM actor_frame;"
```

## How to commit context updates

When you make architectural changes, update this file. The user's memory
at `~/.claude/projects/-Users-spencerbyers/memory/project_fieldvision.md`
also gets read at session start — keep both in rough sync.
