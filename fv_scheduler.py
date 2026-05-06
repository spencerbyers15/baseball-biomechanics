"""
MLB FieldVision Auto-Scheduler
===============================
Long-running daemon that watches the MLB schedule and automatically spawns
a capture process for each game when it starts. Runs forever (or until you
kill it). This is the "set and forget" wrapper you actually want.

Behavior:
  - Every 10 minutes, fetch today's MLB schedule from statsapi.mlb.com
  - For each scheduled game that has started (or is about to start in ≤5 min),
    spawn a background `fv_game_capture.py` process IF we haven't already
    captured that game today
  - Cap concurrent captures at MAX_CONCURRENT (default 4) to protect RAM
  - At midnight, roll over to the new day's schedule
  - Persist state in ./state/ so the daemon can restart without re-capturing

Usage:
  # Run forever, capturing every live game automatically
  python fv_scheduler.py

  # Run with a concurrency cap  
  python fv_scheduler.py --max-concurrent 2

  # Only capture specific teams
  python fv_scheduler.py --teams "Red Sox,Yankees,Guardians"

  # Dry run (prints what it would do, doesn't actually spawn)
  python fv_scheduler.py --dry-run

To run it in the background on Windows (so it survives you logging out):
  start /B python fv_scheduler.py > scheduler.log 2>&1

Or use PM2 / Task Scheduler / NSSM for a proper Windows service.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


STATE_DIR = Path('./state')
OUTPUT_DIR = Path('./fv_captures')
POLL_INTERVAL = 600  # seconds between schedule checks (10 minutes)
PRE_GAME_LEAD = 300  # start capturing 5 min before first pitch


def fetch_schedule(date_str: str = None) -> list[dict]:
    """Fetch today's MLB games from statsapi.mlb.com."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f'[scheduler] schedule fetch failed: {e}', flush=True)
        return []

    games = []
    for date_entry in data.get('dates', []):
        for g in date_entry.get('games', []):
            status = g.get('status', {})
            abstract = status.get('abstractGameState', '')
            detailed = status.get('detailedState', '')
            game_pk = g.get('gamePk')
            if not game_pk:
                continue
            teams = g.get('teams', {})
            away = teams.get('away', {}).get('team', {}).get('name', '?')
            home = teams.get('home', {}).get('team', {}).get('name', '?')
            game_date = g.get('gameDate', '')  # ISO UTC timestamp
            games.append({
                'gamePk': game_pk,
                'matchup': f'{away} @ {home}',
                'away': away,
                'home': home,
                'abstract': abstract,
                'detailed': detailed,
                'gameDate': game_date,
                'url': f'https://www.mlb.com/gameday/{game_pk}/live',
            })
    return games


def game_start_time(game: dict) -> datetime:
    """Parse game start time as timezone-aware UTC datetime."""
    return datetime.fromisoformat(game['gameDate'].replace('Z', '+00:00'))


def should_capture_now(game: dict) -> bool:
    """True if we should start a capture for this game right now."""
    if game['abstract'] == 'Final':
        return False
    if game['abstract'] == 'Live':
        return True
    if game['abstract'] == 'Preview':
        try:
            start = game_start_time(game)
            now = datetime.now(timezone.utc)
            seconds_until = (start - now).total_seconds()
            return 0 <= seconds_until <= PRE_GAME_LEAD
        except Exception:
            return False
    return False


class Scheduler:
    def __init__(self, max_concurrent: int, teams_filter: list[str] | None,
                 dry_run: bool, python_exe: str, capture_script: Path):
        self.max_concurrent = max_concurrent
        self.teams_filter = teams_filter
        self.dry_run = dry_run
        self.python_exe = python_exe
        self.capture_script = capture_script

        STATE_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Maps gamePk -> subprocess.Popen
        self.active_captures: dict[int, subprocess.Popen] = {}
        # Tracks gamePks we've already launched today (even if finished)
        self.launched_today: set[int] = set()
        self._load_state()

    def _state_path(self) -> Path:
        today = datetime.now().strftime('%Y-%m-%d')
        return STATE_DIR / f'launched_{today}.json'

    def _load_state(self):
        p = self._state_path()
        if p.exists():
            try:
                self.launched_today = set(json.loads(p.read_text()))
                self.log(f'Loaded state: {len(self.launched_today)} games already launched today')
            except Exception:
                self.launched_today = set()

    def _save_state(self):
        self._state_path().write_text(json.dumps(list(self.launched_today)))

    def log(self, msg: str):
        print(f'[{datetime.now().strftime("%H:%M:%S")}] [scheduler] {msg}', flush=True)

    def reap_finished(self):
        """Remove subprocesses that have exited."""
        finished = [pk for pk, proc in self.active_captures.items() if proc.poll() is not None]
        for pk in finished:
            proc = self.active_captures.pop(pk)
            self.log(f'Capture for game {pk} exited with code {proc.returncode}')

    def team_matches(self, game: dict) -> bool:
        if not self.teams_filter:
            return True
        haystack = f'{game["away"]} {game["home"]}'.lower()
        return any(t.lower() in haystack for t in self.teams_filter)

    def spawn_capture(self, game: dict):
        pk = game['gamePk']
        if pk in self.active_captures or pk in self.launched_today:
            return
        if len(self.active_captures) >= self.max_concurrent:
            self.log(f'At concurrent cap ({self.max_concurrent}), deferring {pk} ({game["matchup"]})')
            return
        if not self.team_matches(game):
            return

        self.log(f'Spawning capture: {game["matchup"]} (gamePk={pk})')
        if self.dry_run:
            self.launched_today.add(pk)
            self._save_state()
            return

        cmd = [
            self.python_exe, str(self.capture_script),
            '--game', str(pk),
            '--max-hours', '4.5',
            '--output', str(OUTPUT_DIR),
        ]
        log_path = OUTPUT_DIR / f'game_{pk}_stdout.log'
        logfile = open(log_path, 'w')
        proc = subprocess.Popen(
            cmd, stdout=logfile, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        )
        self.active_captures[pk] = proc
        self.launched_today.add(pk)
        self._save_state()
        self.log(f'  PID {proc.pid}, log -> {log_path}')

    async def run(self):
        self.log(f'Scheduler starting. max_concurrent={self.max_concurrent} '
                 f'teams_filter={self.teams_filter} dry_run={self.dry_run}')
        self.log(f'Output dir: {OUTPUT_DIR.resolve()}')
        self.log(f'Poll interval: {POLL_INTERVAL}s')

        last_date = datetime.now().date()

        while True:
            try:
                # Day rollover
                today = datetime.now().date()
                if today != last_date:
                    self.log(f'Day rolled over: {last_date} -> {today}')
                    self.launched_today.clear()
                    self._save_state()
                    last_date = today

                self.reap_finished()

                games = fetch_schedule()
                self.log(f'Fetched schedule: {len(games)} games today')

                # Log state summary
                live = [g for g in games if g['abstract'] == 'Live']
                preview = [g for g in games if g['abstract'] == 'Preview']
                final = [g for g in games if g['abstract'] == 'Final']
                self.log(f'  Live: {len(live)}, Preview: {len(preview)}, Final: {len(final)}, '
                         f'Active captures: {len(self.active_captures)}')

                # Spawn captures for games that should be captured now
                for game in games:
                    if should_capture_now(game):
                        self.spawn_capture(game)

                # Log upcoming games that will be captured
                for game in preview:
                    if self.team_matches(game) and game['gamePk'] not in self.launched_today:
                        try:
                            start = game_start_time(game)
                            local = start.astimezone()
                            self.log(f'  Upcoming: {game["matchup"]} at {local.strftime("%H:%M %Z")}')
                        except Exception:
                            pass

            except KeyboardInterrupt:
                self.log('KeyboardInterrupt — shutting down scheduler.')
                self.log(f'Note: {len(self.active_captures)} capture subprocesses are still running. '
                         f'They will continue independently. To stop them, kill their PIDs.')
                return
            except Exception as e:
                self.log(f'Main loop error: {e}')

            await asyncio.sleep(POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description='MLB FieldVision auto-scheduler (daemon)')
    parser.add_argument('--max-concurrent', type=int, default=4,
                        help='Max simultaneous game captures (each Chrome ~500MB RAM). Default 4.')
    parser.add_argument('--teams', type=str, default='',
                        help='Comma-separated team name substrings to filter (e.g., "Red Sox,Yankees"). '
                             'Default: all teams.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Log what would happen without spawning processes')
    parser.add_argument('--capture-script', type=str, default='fv_game_capture.py',
                        help='Path to fv_game_capture.py (default: ./fv_game_capture.py)')
    args = parser.parse_args()

    teams_filter = [t.strip() for t in args.teams.split(',') if t.strip()] or None

    capture_script = Path(args.capture_script).resolve()
    if not capture_script.exists():
        print(f'ERROR: capture script not found at {capture_script}')
        print('Make sure fv_game_capture.py is in the same directory, or pass --capture-script')
        sys.exit(1)

    sched = Scheduler(
        max_concurrent=args.max_concurrent,
        teams_filter=teams_filter,
        dry_run=args.dry_run,
        python_exe=sys.executable,
        capture_script=capture_script,
    )
    try:
        asyncio.run(sched.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
