# FieldVision automated capture setup

This document covers the **automated daemon** that captures every live MLB game's skeletal data into `data/fv_<gamePk>.sqlite`.

## Architecture (one-time setup)

```
┌──────────────────────────────────────────────────────────────────┐
│  scripts/setup_login.py        (run once — interactive)          │
│  Headed Chromium → MLB.com login → cookies persist on disk in    │
│  .mlb_profile/                                                   │
└──────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│  scripts/fv_daemon.py          (runs forever via launchd)        │
│  Every 10 min: fetch today's MLB schedule.                       │
│  Every 30s during live games:                                    │
│    1. Headless Chromium uses the persistent profile to read a    │
│       fresh api://mlb_default JWT from localStorage.             │
│    2. urllib pulls the latest manifest.json over the JWT.        │
│    3. Any new segments are downloaded (samples/binary_capture_…) │
│       and immediately decoded into actor_frame rows in           │
│       data/fv_<gamePk>.sqlite (3-4 GB per game).                 │
│  Re-acquires the JWT every 6h or on HTTP 401.                    │
└──────────────────────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────────┐
│  ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist         │
│  RunAtLoad=true, KeepAlive=true → restarts on crash, on reboot.  │
│  Logs to scheduler.log and scheduler.err in the repo.            │
└──────────────────────────────────────────────────────────────────┘
```

## One-time install

### 1. Log in (interactive, ~2 min)

```bash
cd ~/Documents/GitHub/fieldvision
~/anaconda3/envs/baseball/bin/python scripts/setup_login.py
```

A Chromium window opens at `https://www.mlb.com/login`. Log in. The script
detects the resulting JWT, verifies it works against `fieldvision-hls.mlbinfra.com`,
and closes. Cookies persist in `.mlb_profile/`.

If the window appears greyed out (macOS focus bug), click on it once — that
gives it focus.

### 2. Smoke-test the daemon (foreground, 30s)

```bash
~/anaconda3/envs/baseball/bin/python scripts/fv_daemon.py --once
```

This fetches the schedule, scrapes any currently live game once, and exits.
Verifies the persistent-profile token-refresh mechanism works end-to-end.

### 3. Install launchd plist

```bash
cp launchd/com.spencerbyers.fvcapture.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
launchctl list | grep fvcapture     # should print a PID
sleep 30
tail -50 scheduler.log               # daemon output
tail -50 scheduler.err               # any errors
```

### 4. Verify it's running

```bash
# Should show non-zero PID in column 1
launchctl list | grep fvcapture

# Live tail
tail -f scheduler.log
```

## Operations

| What | Command |
|---|---|
| **Stop** | `launchctl unload ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist` |
| **Start** | `launchctl load ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist` |
| **Status** | `launchctl list \| grep fvcapture` (PID = running) |
| **Live logs** | `tail -f scheduler.log` |
| **Errors** | `tail -f scheduler.err` |
| **Per-game raw bins** | `samples/binary_capture_<gamePk>/` |
| **Per-game SQLite** | `data/fv_<gamePk>.sqlite` |
| **Cross-game registry** | `data/games_registry.sqlite` |
| **Filter to specific teams** | edit plist, add `<string>--teams</string><string>Mariners,Dodgers</string>` to `ProgramArguments` |

## What gets captured

For every live MLB game:

- **Schema files** (one-shot per game): `manifest.json`, `metadata.json`, `labels.json`
- **Binary segments** (~110 KB each, 5s each, ~1,800 per game) under `samples/binary_capture_<pk>/`
- **Decoded SQLite** with one row per actor per frame:
  - `actor_frame` (game_pk, segment_idx, frame_num, actor_uid, mlb_player_id, actor_type, time_unix, timestamp, scale, ground, apex, **20 joint x/y/z columns**, bat_handle x/y/z) — ~5M rows per game
  - `ball_frame` (ball x/y/z when tracked) — ~50K rows per game
  - `players`, `bones`, `labels`, `meta` lookup tables

Joint coordinates are in **stadium feet, Y up, home plate at origin**, computed via forward kinematics from the GLTF rest pose + the per-frame rotation quaternions.

## Token / cookie expiry

The Okta refresh token in `.mlb_profile/` lasts 1-2 weeks. If the daemon
starts logging "Failed to read JWT" or HTTP 401s, re-run `setup_login.py`
once (you log in again). The daemon picks up the new cookies on its next
6-hour token refresh.

## Disk usage

- Raw .bin segments: ~200 MB per game
- Decoded SQLite: ~3.4 GB per game
- 15 games/day × 200 MB = 3 GB/day raw, 50 GB/day SQLite

After capture, you can delete the raw `samples/binary_capture_<pk>/` if
you've validated the SQLite — the SQLite is the queryable source of truth.

## Troubleshooting

### `setup_login.py`: window appears but I can't type

The macOS Playwright focus bug. Click once on the Chromium window's title
bar to give it focus, then type your credentials. The script polls every 3s
until it detects a valid JWT.

### Daemon logs `Failed to read JWT`

Cookies expired. Re-run `scripts/setup_login.py`.

### `HTTP 403` from fieldvision-hls

Either the User-Agent is being blocked (the daemon sends a desktop Chrome UA;
make sure no proxy is rewriting it) or your account doesn't have `mlb_entitlement`
scope. Verify by manually logging in to mlb.com and watching a live game's
3D view in your browser — if that works, the daemon should work.

### `HTTP 429`

MLB rate-limited us. The daemon retries with exponential backoff. If it
persists, lower the per-game segment poll concurrency in `fv_daemon.py`.

### Disk filling up

Use the `--teams` filter to capture only the games you care about. Or add
a cleanup cron that deletes `samples/binary_capture_*/` directories older
than N days (the SQLite is sufficient for analysis after ingestion).
