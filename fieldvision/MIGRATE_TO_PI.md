# Migrating the FieldVision daemon to a Raspberry Pi 4

The daemon is **deliberately lightweight** — pure Python stdlib + SQLite. No
Playwright, no GPU, no compiled extensions. A Pi 4 with 4 GB RAM and an
external USB SSD is plenty.

## What runs on the Pi

- `scripts/fv_daemon.py` — polls schedule, scrapes new segments, ingests to SQLite
- `scripts/scrape_team_history.py` — backfill old games on demand
- `scripts/load_to_db.py`, `scripts/load_bat_positions.py` — data loading utilities
- The `data/fv_<gamePk>.sqlite` files (one per game)

## What stays on your laptop

- Anything that uses matplotlib (clip rendering, plotting). The Pi can do it but it's slow and you don't need to render on the Pi.
- The DevTools paste workflow for refreshing the JWT (you do this from your normal Chrome on your laptop or any logged-in machine, then `scp` the token file to the Pi).

## Hardware setup

Recommended:
- Raspberry Pi 4 with **4 GB RAM** (or 8 GB if you have it). Pi 5 is even better.
- **External USB 3.0 SSD** — at least 256 GB, 500 GB+ if you want to keep historical games. Do not use an SD card for the SQLite stores; SD cards die under heavy write load.
- Wired Ethernet (more reliable than wifi for a 24/7 daemon).
- A real PSU (≥ 3 A for Pi 4 to avoid undervolt warnings under load).

Mount the USB SSD at `/mnt/fv-data` and put the repo there:

```bash
sudo mkdir /mnt/fv-data
sudo mount /dev/sda1 /mnt/fv-data        # adjust device as needed
sudo chown $USER /mnt/fv-data
echo "/dev/sda1 /mnt/fv-data ext4 defaults 0 2" | sudo tee -a /etc/fstab
```

## OS + Python install

Use **Raspberry Pi OS Lite (64-bit)**, latest. SSH in, then:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git rsync
# (Or whatever Python 3.11+ your distro provides; 3.10 also works.)
```

## Get the code + data over from your Mac

From the Mac, push the repo + already-captured data to the Pi:

```bash
# code (small)
rsync -azv --exclude='.venv' --exclude='samples' --exclude='data' \
       --exclude='.mlb_profile' --exclude='__pycache__' \
       ~/fieldvision/ pi@<pi-ip>:/mnt/fv-data/fieldvision/

# captured SQLite (big, several GB)
rsync -azv --progress ~/fieldvision/data/ pi@<pi-ip>:/mnt/fv-data/fieldvision/data/

# token file (tiny)
scp ~/fieldvision/.fv_token.txt pi@<pi-ip>:/mnt/fv-data/fieldvision/.fv_token.txt
```

## Set up the Python env on the Pi

```bash
cd /mnt/fv-data/fieldvision
python3 -m venv .venv
source .venv/bin/activate
# The daemon itself needs nothing beyond stdlib.
# scripts/scrape_team_history.py also stdlib only.
# load_to_db / render scripts pull in extras when needed.
```

## Smoke test

```bash
cd /mnt/fv-data/fieldvision
.venv/bin/python scripts/fv_daemon.py --once
```

Should print `Schedule: N games, M live` and start scraping any live games.

## systemd unit (the launchd equivalent)

Create `/etc/systemd/system/fvcapture.service`:

```ini
[Unit]
Description=FieldVision Capture Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/mnt/fv-data/fieldvision
ExecStart=/mnt/fv-data/fieldvision/.venv/bin/python -u /mnt/fv-data/fieldvision/scripts/fv_daemon.py
Restart=always
RestartSec=30
StandardOutput=append:/mnt/fv-data/fieldvision/scheduler.log
StandardError=append:/mnt/fv-data/fieldvision/scheduler.err

[Install]
WantedBy=multi-user.target
```

Adjust `User=pi` if your username is different.

Enable + start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable fvcapture.service
sudo systemctl start fvcapture.service
sudo systemctl status fvcapture.service
journalctl -u fvcapture -f       # live tail
```

## Daily token refresh — from your Mac, not the Pi

The Pi has no logged-in MLB browser. The cleanest workflow:

1. On your Mac (or any logged-in machine), paste the token snippet into Chrome DevTools as before.
2. `mv ~/Downloads/fv_token.txt /tmp/fv_token.txt`
3. `scp /tmp/fv_token.txt pi@<pi-ip>:/mnt/fv-data/fieldvision/.fv_token.txt`

The daemon picks it up on its next poll.

You can put steps 2-3 in a tiny shell alias (`fv-push-token`) on your Mac.

## "Hands-free for vacation" version

Until we have OAuth refresh-token auto-refresh implemented, the token still expires every ~24 h. If you'll be on vacation:

- **Option A**: Set up a phone shortcut that runs the same DevTools snippet on iOS Safari (paste-once, scp via Tailscale or similar) once a day.
- **Option B**: Wait for me to ship the OAuth refresh implementation (Phase 1 — needs to confirm whether MLB issues `offline_access` refresh tokens; their current access token claims show only `scp: ['openid', 'email']` which is *not* a great sign).
- **Option C**: Use the cloud VM path instead (next section), which doesn't help with the token issue but means your Mac doesn't have to be online.

## If you'd rather use a cloud VM (alternative)

Same setup as Pi but on a $5/mo Hetzner / Digital Ocean / Linode box. Skip the hardware install, follow the same Python + systemd steps. Pull the data periodically with `rsync` if you want it locally.

Cloud VM advantages: no hardware management, fast network, predictable uptime.
Cloud VM drawbacks: ongoing cost, data leaves your house.

## Known gotchas

- **`fv_823141.sqlite` is 3.3 GB.** rsync uses delta compression so subsequent transfers are fast, but the first push is several minutes on a residential uplink.
- **Pi 4 USB write speed varies wildly with cable + drive.** Cheap thumb drives bottleneck at 30 MB/s; a real SSD via USB 3 hits 200+ MB/s. SQLite WAL writes are bursty so this matters.
- **The Pi's clock can drift.** Make sure NTP is enabled (`sudo timedatectl set-ntp true`) — segment timestamps are derived from the host clock for some operations.
- **macOS exports SQLite WAL files alongside the .sqlite**. After rsync, force a checkpoint on the Pi side: `sqlite3 fv_<pk>.sqlite "PRAGMA wal_checkpoint(TRUNCATE);"` to fold the WAL into the main DB.
