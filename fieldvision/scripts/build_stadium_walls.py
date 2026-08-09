"""Build the per-venue wall-polyline cache from MLB's ballpark GLBs.

Downloads each venue's FieldVision 3D model (~4-5 MB), extracts the
outfield-wall polyline with heights in the tracking frame, and writes
{venue_id}.json files + an index. Cross-checks straight-line fence
distances against statsapi fieldInfo where available.

Needs: numpy, DracoPy (pip install DracoPy), network.

  python scripts/build_stadium_walls.py --out /media/datasets/spencer/fieldvision/stadium_walls
  python scripts/build_stadium_walls.py --out walls_cache --venue 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fieldvision.stadium_walls import (build_venue, resolve_asset_hash,
                                       venue_abbr_map)


def fieldinfo_check(venue_id: int, poly: list[dict]) -> str:
    """Compare our CF-bearing fence radius with statsapi's center distance."""
    try:
        url = (f"https://statsapi.mlb.com/api/v1/venues/{venue_id}"
               f"?hydrate=fieldInfo")
        with urllib.request.urlopen(url, timeout=15) as r:
            fi = json.load(r)["venues"][0].get("fieldInfo", {})
        center = fi.get("center")
        if not center:
            return "no fieldInfo center"
        # statsapi's "center" is the posted deepest-CF marker; compare it to
        # our deepest fence point in the CF sector (fences angle, so dead
        # center bearing is often shallower than the marker).
        cf = [p["r"] for p in poly if abs(p["bearing_deg"]) < 8
              and not p["pole_spike"]]
        if not cf:
            return "no CF bin"
        delta = max(cf) - center
        return f"deepest CF {max(cf):.0f} vs statsapi {center} (Δ{delta:+.0f} ft)"
    except Exception as e:
        return f"check failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--venue", type=int, help="single venueId (default: all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    asset_hash = resolve_asset_hash()
    vmap = venue_abbr_map()
    print(f"asset hash {asset_hash}, {len(vmap)} venues")
    todo = {args.venue: vmap[args.venue]} if args.venue else vmap

    index = {}
    for vid, abbr in sorted(todo.items()):
        try:
            d = build_venue(vid, abbr, asset_hash)
            (out / f"{vid}.json").write_text(json.dumps(d))
            check = fieldinfo_check(vid, d["polyline"])
            index[vid] = {"abbr": abbr, "n_points": d["n_points"],
                          "check": check}
            print(f"  {vid:>5} {abbr:<4} {d['n_points']:>3} pts  | {check}")
        except Exception as e:
            index[vid] = {"abbr": abbr, "error": str(e)[:120]}
            print(f"  {vid:>5} {abbr:<4} FAILED: {e}")
        time.sleep(0.5)
    (out / "index.json").write_text(json.dumps(
        {"asset_hash": asset_hash, "venues": index}, indent=1))
    print(f"cache -> {out}")


if __name__ == "__main__":
    main()
