# FieldVision automated capture setup

The **daemon** captures every live MLB game's skeletal data into `data/fv_<gamePk>.sqlite`. Everything is built; the only ongoing maintenance is **refreshing your auth token roughly once a day** via a 30-second DevTools paste.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  .fv_token.txt          (refreshed daily by you, 30 sec)        │
│  Contains a JWT (audience api://mlb_default) extracted from     │
│  your logged-in MLB.com session via DevTools paste.             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  scripts/fv_daemon.py    (runs forever via launchd)             │
│  Every 10 min: fetch today's MLB schedule.                      │
│  Every 30s during live games:                                   │
│    1. Read .fv_token.txt (validates exp, audience)              │
│    2. urllib pulls manifest.json (with desktop Chrome UA —      │
│       Python-urllib's default UA gets HTTP 403 from MLB)        │
│    3. New segments → samples/binary_capture_<gamePk>/           │
│       + immediate decode into data/fv_<gamePk>.sqlite           │
│    4. On HTTP 401: touches state/token_expired.flag, logs a     │
│       clear message, sleeps until token is refreshed.           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist        │
│  RunAtLoad + KeepAlive + ThrottleInterval=30s                   │
│  Restarts on crash, on reboot, on log-out/in.                   │
│  Logs to scheduler.log + scheduler.err in the repo root.        │
└─────────────────────────────────────────────────────────────────┘
```

> Why no Playwright login? On macOS, Playwright-launched Chromium windows
> hit a focus bug where the OS won't deliver keystrokes to them. The daemon
> instead reads from a token file you produce via DevTools paste — no
> headed browser needed.

## One-time install

### 1. Get an auth token

Log in to mlb.com in your normal Chrome (any logged-in tab works). Open DevTools (`Cmd+Option+J`), allow paste if asked, then paste:

```javascript
(() => {
  const raw = localStorage.getItem('okta-token-storage');
  const RX = /eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g;
  let token = null;
  for (const j of new Set(raw.match(RX) || [])) {
    try {
      const c = JSON.parse(atob(j.split('.')[1].replace(/-/g,'+').replace(/_/g,'/')));
      if (c.aud === 'api://mlb_default') { token = j; break; }
    } catch(e) {}
  }
  if (!token) { console.error('No api://mlb_default token found.'); return; }
  const blob = new Blob([token], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'fv_token.txt';
  document.body.appendChild(a); a.click(); a.remove();
  console.log('✓ Token saved to ~/Downloads/fv_token.txt');
})();
```

(Same snippet at `scripts/snippets/refresh_token.js`.)

Then:

```bash
mv ~/Downloads/fv_token.txt ~/fieldvision/.fv_token.txt
```

### 2. Smoke-test the daemon (~1 min)

```bash
cd ~/fieldvision
~/anaconda3/envs/baseball/bin/python scripts/fv_daemon.py --once
```

You should see lines like `Schedule: 15 games, 4 live` and per-game status. The first run for a fresh game catches up *all* its segments since the start of the game (can be ~3,000 segments / ~10 min).

### 3. Install the launchd plist

```bash
cp launchd/com.spencerbyers.fvcapture.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
launchctl list | grep fvcapture     # PID = it's running
tail -f scheduler.log                # watch it work
```

## Daily maintenance

The token expires every ~24 hours. When the daemon hits HTTP 401, it touches `state/token_expired.flag` and logs `TOKEN UNAVAILABLE`. To refresh:

1. Re-run the DevTools snippet above
2. `mv ~/Downloads/fv_token.txt .fv_token.txt`
3. The daemon picks it up on its next poll (≤ 5 min)

You can detect "token needed" without checking logs:

```bash
test -f ~/fieldvision/state/token_expired.flag && echo "REFRESH TOKEN"
```

(Could be hooked into your menu bar via SwiftBar, BitBar, or similar.)

## Operations

| What | Command |
|---|---|
| **Stop** | `launchctl unload ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist` |
| **Start** | `launchctl load ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist` |
| **Status** | `launchctl list \| grep fvcapture` (number in col 1 = running PID) |
| **Live logs** | `tail -f scheduler.log` |
| **Errors** | `tail -f scheduler.err` |
| **Per-game raw bins** | `samples/binary_capture_<gamePk>/` |
| **Per-game SQLite** | `data/fv_<gamePk>.sqlite` |
| **Cross-game registry** | `data/games_registry.sqlite` |
| **Filter to specific teams** | edit plist `ProgramArguments`, add `<string>--teams</string><string>Mariners,Dodgers</string>` |
| **Token expired marker** | `state/token_expired.flag` exists ⇒ refresh needed |

## What gets captured per game

- **Schema files**: `manifest.json`, `metadata.json`, `labels.json`
- **Raw .bin segments** (~110 KB × ~2,000 per game) under `samples/binary_capture_<pk>/`
- **Decoded SQLite** at `data/fv_<pk>.sqlite`:
  - `actor_frame` — one row per actor per frame, 20 joint x/y/z columns + bat_handle
  - `ball_frame` — ball world position per frame when tracked
  - `bat_frame` — bat handle + head world positions per frame (full bat axis)
  - `players`, `bones`, `labels`, `meta` — lookup tables

All coords in **stadium feet, Y up, home plate at origin**.

## Disk usage

- ~200 MB raw .bin per game
- ~3.4 GB SQLite per game
- 15 games/day × ~3.6 GB = ~54 GB/day if you capture all of them

After the SQLite is built you can delete `samples/binary_capture_<pk>/` if disk is tight — the SQLite is the queryable source of truth.

## Troubleshooting

### `TOKEN UNAVAILABLE (expired)` in logs

Refresh your token (step from "Daily maintenance" above).

### `HTTP 403` from fieldvision-hls

The User-Agent is being blocked. The daemon uses a desktop Chrome UA — verify no proxy is rewriting it. Or your account doesn't have FieldVision entitlement (rare for free accounts but possible).

### `HTTP 429`

Rate-limited. The daemon retries with exponential backoff. If it persists, lower segment-poll concurrency in `fv_daemon.py` (currently the per-game segment delay is 0.3s).

### Daemon not running after reboot

```bash
launchctl list | grep fvcapture
# If nothing listed:
launchctl load ~/Library/LaunchAgents/com.spencerbyers.fvcapture.plist
```

If launchd thinks it's loaded but no PID, check `scheduler.err` for startup errors.

### Disk filling up

Add `--teams` to capture only games you care about. Or add a cleanup cron that deletes `samples/binary_capture_*/` directories older than N days.
