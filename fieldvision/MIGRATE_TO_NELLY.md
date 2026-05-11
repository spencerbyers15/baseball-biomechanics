# Migrating FieldVision to Nelly

Nelly is a GPU server on a private home network. It has good GPUs so it can
also serve compute for the heavier analyses (jPCA across many batters, large
clustering, video rendering at scale). Storage for raw data goes onto an
attached NAS.

## Network access

- **VPN first.** You must be on the home VPN to reach Nelly or the NAS.
- **NAS IP**: `10.210.1.101` (used for bulk data storage)
- **SSH host**: Nelly (whatever its hostname/IP resolves to once VPN'd in —
  ask owner for the resolvable name or IP)
- **Username**: `spencer`
- **Password**: kept out of this doc. The owner provided it verbally. There
  was a parenthesized `(death...)` prefix that may be a network/host name
  qualifier or an actual password prefix — confirm with the owner what it
  literally means before first login.

**Do not commit the password to git.** First thing to do after first login:
set up SSH key auth so we never need to type the password again.

## First-time SSH key setup

On your Mac (only once):

```bash
# Generate a key if you don't already have one
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -C "spencerbyers@$(hostname)"

# Connect to VPN first, then push the key to Nelly
ssh-copy-id spencer@<nelly-host>     # will prompt for password the one time
```

Then add a host entry so future commands are short:

```sshconfig
# ~/.ssh/config
Host nelly
    HostName <nelly-host>
    User spencer
    IdentityFile ~/.ssh/id_ed25519
```

From then on: `ssh nelly` no password needed.

## What goes on Nelly

- The repo (clone of `baseball-biomechanics`)
- The Python env (3.11+ with hdbscan, umap-learn, scipy, matplotlib)
- The `fv_daemon.py` running as a long-lived process (tmux for now, systemd
  later)
- All Python analysis: census, outcome phase, render scripts, jPCA fits

## What goes on the NAS at 10.210.1.101

The big stuff that grows with time:
- `data/` — per-game SQLite databases (~3-4 GB each, hundreds eventually)
- `samples/` — raw `.bin` segment files (~2 GB per game; can be `--delete-bins`'d
  after ingestion if disk pressure becomes an issue)

Mount the NAS share on Nelly under e.g. `/mnt/baseball-data/`, then symlink:

```bash
# On Nelly, after the NAS is mounted at /mnt/baseball-data
cd ~/baseball-biomechanics/fieldvision
ln -s /mnt/baseball-data/fieldvision-data data
ln -s /mnt/baseball-data/fieldvision-samples samples
ln -s /mnt/baseball-data/fieldvision-state state
```

This way every script that reads `data/oscillation_report/...` keeps working
unchanged, but the actual storage lives on the NAS.

## What stays on your Mac

- Token refresh workflow (DevTools paste in Chrome → write `.fv_token.txt`).
  Nelly has no logged-in Chrome session, so you do this locally then:
  ```bash
  scp ~/.fv_token.txt nelly:baseball-biomechanics/fieldvision/.fv_token.txt
  ```
  Worth making a `~/bin/fv-token` shell alias that does both steps.
- VS Code Remote-SSH for editing (feels local, runs on Nelly).
- Anything that needs a GUI to interact with (video player for `.mp4`
  outputs, image viewer for plots). Either `scp` results back or use VS
  Code's preview while connected remotely.

## Migration steps

### 1. Stop the Mac daemon (one-time, before moving data)

```bash
ps aux | grep fv_daemon            # find PID
kill <PID>                         # graceful stop
```

Stopping the local daemon avoids two processes both trying to scrape the
same games and corrupting state files.

### 2. Set up the env on Nelly

```bash
ssh nelly
cd ~
git clone https://github.com/spencerbyers15/baseball-biomechanics.git
cd baseball-biomechanics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # CV-pose deps from biomechanics root
pip install -r fieldvision/requirements.txt
# Additional deps the fieldvision pipeline needs:
pip install hdbscan umap-learn scipy
```

### 3. Mount the NAS and symlink data dirs

Whatever the owner has configured for NAS auth (NFS, SMB, sshfs) — get
`/mnt/baseball-data/` mounted with read+write for user `spencer`. Then
inside the cloned repo:

```bash
cd ~/baseball-biomechanics/fieldvision
mkdir -p /mnt/baseball-data/{fieldvision-data,fieldvision-samples,fieldvision-state}
ln -s /mnt/baseball-data/fieldvision-data data
ln -s /mnt/baseball-data/fieldvision-samples samples
ln -s /mnt/baseball-data/fieldvision-state state
```

### 4. Move the captured data from Mac → Nelly NAS

From your Mac (on VPN):

```bash
# code is already in git, no need to rsync it
# data is big — rsync over the VPN. This will take a while.
rsync -av --progress \
  ~/Documents/GitHub/baseball-biomechanics/fieldvision/data/ \
  nelly:/mnt/baseball-data/fieldvision-data/

rsync -av --progress \
  ~/Documents/GitHub/baseball-biomechanics/fieldvision/samples/ \
  nelly:/mnt/baseball-data/fieldvision-samples/

rsync -av --progress \
  ~/Documents/GitHub/baseball-biomechanics/fieldvision/state/ \
  nelly:/mnt/baseball-data/fieldvision-state/
```

Expect a few hours over residential upload for ~60 GB. The NAS-on-LAN will
be fast once the data's local to it.

### 5. Push the token

```bash
scp ~/Documents/GitHub/baseball-biomechanics/fieldvision/.fv_token.txt \
    nelly:~/baseball-biomechanics/fieldvision/.fv_token.txt
```

### 6. Start the daemon on Nelly

```bash
ssh nelly
cd ~/baseball-biomechanics/fieldvision
tmux new -s fv
source ../.venv/bin/activate
python -u scripts/fv_daemon.py 2>&1 | tee scheduler.log
# Detach with Ctrl-B, d
```

Verify it's polling:

```bash
ssh nelly
tail -f ~/baseball-biomechanics/fieldvision/scheduler.log
```

Should see `Schedule: N games, M live` lines every 10 minutes.

### 7. Verify locally vs remotely

From your Mac, set up VS Code Remote-SSH to Nelly and open the
`baseball-biomechanics` folder. Run a smoke test:

```bash
cd ~/baseball-biomechanics/fieldvision
python scripts/outcome_phase_analysis.py --method jpca
```

Should produce the same outputs as on the Mac.

### 8. Reclaim disk on your Mac (after verification)

```bash
# only after verifying Nelly is scraping cleanly and analyses run
rm -rf ~/Documents/GitHub/baseball-biomechanics/fieldvision/{data,samples,state}
# the local-only ~/fieldvision/ symlink target is now broken too — clean up:
rm -rf ~/fieldvision
```

~62 GB back on your laptop. The repo at
`~/Documents/GitHub/baseball-biomechanics/` keeps working with empty
`fieldvision/data,samples,state/` placeholders (or just no dirs there at
all — the daemon and analyses run on Nelly).

## Open questions for the owner

- Hostname / IP for Nelly itself (after VPN connect)?
- What kind of share is the NAS — NFS, SMB, sshfs? Need to know to mount it
  correctly on Nelly with persistent fstab.
- Is there a per-user quota on the NAS we should be aware of?
- Long-lived processes OK (e.g., the daemon running for weeks)? Confirm
  there's no aggressive idle-killer.
- GPU access — do we have direct CUDA access for future model training
  (RTMPose, etc.) or is the GPU shared via a queueing system?
- What does the `(death...)` prefix on the password mean? Is it a host
  qualifier, a literal password character sequence, or something else?

## Future: systemd unit for the daemon

Once tmux-managed running is verified, replace it with a real systemd unit
so the daemon survives reboots:

```ini
# /etc/systemd/system/fv-daemon.service
[Unit]
Description=FieldVision MLB capture daemon
After=network-online.target

[Service]
Type=simple
User=spencer
WorkingDirectory=/home/spencer/baseball-biomechanics/fieldvision
ExecStart=/home/spencer/baseball-biomechanics/.venv/bin/python -u scripts/fv_daemon.py
StandardOutput=append:/home/spencer/baseball-biomechanics/fieldvision/scheduler.log
StandardError=append:/home/spencer/baseball-biomechanics/fieldvision/scheduler.log
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Install:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now fv-daemon
sudo systemctl status fv-daemon
journalctl -u fv-daemon -f             # live log tail
```
