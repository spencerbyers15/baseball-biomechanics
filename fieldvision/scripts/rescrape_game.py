"""Manually re-scrape one game from segment 0. Used to recover games that
were corrupted by the daemon's pre-fix truncation bug.

Usage:
    FV_DATA_DIR=... python3 scripts/rescrape_game.py 824438
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.scrape_team_history import scrape_one_game

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN = (REPO_ROOT / ".fv_token.txt").read_text().strip()

if len(sys.argv) != 2:
    print("usage: rescrape_game.py <gamePk>")
    sys.exit(1)

pk = int(sys.argv[1])
print(f"re-scraping game {pk}...")
result = scrape_one_game(pk, TOKEN, delete_bins=True)
print(f"\nresult: {result}")
