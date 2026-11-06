"""Staff-facing outputs: per-throw CSV/JSON, per-player summary, one-page HTML."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from .schema import ThrowEvent

COLUMNS = [
    "throw_id", "time_s", "timecode", "clock_time", "thrower_id", "thrower", "arm",
    "target_type", "receiver_id", "receiver", "receiver_conf", "distance_ft", "distance_sd_ft",
    "ball_type", "ball_type_conf", "velo_mph", "velo_lo_mph", "velo_hi_mph", "velo_source",
    "intensity_pct", "throw_conf", "needs_review",
]
DISTANCE_BUCKETS_FT = [(0, 60), (60, 90), (90, 120), (120, 180), (180, 1e9)]


def _timecode(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def to_rows(throws: list[ThrowEvent], roster: Optional[dict] = None,
            session_start: Optional[datetime] = None) -> list[dict]:
    roster = roster or {}

    def name(tid):
        return roster.get(tid, {}).get("name", f"player_{tid}") if tid is not None else None

    rows = []
    for i, t in enumerate(sorted(throws, key=lambda e: e.time_s), start=1):
        rows.append({
            "throw_id": i,
            "time_s": round(t.time_s, 2),
            "timecode": _timecode(t.time_s),
            "clock_time": (session_start + timedelta(seconds=t.time_s)).isoformat(timespec="seconds")
            if session_start else None,
            "thrower_id": t.track_id,
            "thrower": name(t.track_id),
            "arm": t.arm,
            "target_type": t.target_type,
            "receiver_id": t.receiver_id,
            "receiver": name(t.receiver_id),
            "receiver_conf": _r(t.receiver_confidence, 2),
            "distance_ft": _r(t.distance_ft),
            "distance_sd_ft": _r(t.distance_sd_ft),
            "ball_type": t.ball_type,
            "ball_type_conf": _r(t.ball_type_confidence, 2),
            "velo_mph": _r(t.velo_mph),
            "velo_lo_mph": _r(t.velo_lo_mph),
            "velo_hi_mph": _r(t.velo_hi_mph),
            "velo_source": t.velo_source,
            "intensity_pct": _r(t.intensity_pct, 0),
            "throw_conf": _r(t.confidence, 2),
            "needs_review": bool(t.needs_review),
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = {}
    for r in rows:
        s = out.setdefault(r["thrower_id"], {
            "thrower_id": r["thrower_id"], "thrower": r["thrower"], "throws": 0,
            "partner": 0, "wall": 0, "unknown": 0, "ball_types": {},
            **{f"dist_{a}_{b if b < 1e9 else 'plus'}": 0 for a, b in DISTANCE_BUCKETS_FT},
            "velos": [], "high_intensity": 0, "needs_review": 0,
        })
        s["throws"] += 1
        s[r["target_type"]] += 1
        bt = r["ball_type"] or "unclassified"
        s["ball_types"][bt] = s["ball_types"].get(bt, 0) + 1
        if r["distance_ft"] is not None:
            for a, b in DISTANCE_BUCKETS_FT:
                if a <= r["distance_ft"] < b:
                    s[f"dist_{a}_{b if b < 1e9 else 'plus'}"] += 1
        if r["velo_mph"] is not None:
            s["velos"].append(r["velo_mph"])
        if r["intensity_pct"] is not None and r["intensity_pct"] >= 85:
            s["high_intensity"] += 1
        s["needs_review"] += int(r["needs_review"])
    result = []
    for s in out.values():
        v = s.pop("velos")
        s["mean_velo_mph"] = _r(np.mean(v)) if v else None
        s["max_velo_mph"] = _r(np.max(v)) if v else None
        s["ball_types"] = "; ".join(f"{k}: {n}" for k, n in sorted(s["ball_types"].items()))
        result.append(s)
    return sorted(result, key=lambda s: s["thrower"])


def write_csv(rows: list[dict], path: Path, columns: Optional[list] = None) -> None:
    columns = columns or (list(rows[0].keys()) if rows else COLUMNS)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def write_html(rows: list[dict], summary: list[dict], path: Path, title: str = "Throwing session") -> None:
    def table(rs, cols):
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = "".join(
            "<tr" + (' class="review"' if r.get("needs_review") is True else "") + ">"
            + "".join(f"<td>{html.escape('' if r.get(c) is None else str(r.get(c)))}</td>" for c in cols)
            + "</tr>" for r in rs)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    s_cols = list(summary[0].keys()) if summary else []
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --flag:#fff4d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111; --fg:#eee; --muted:#999; --line:#333; --flag:#3a2f10; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; margin:16px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin-top:24px; }}
p {{ color:var(--muted); }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; white-space:nowrap; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 8px; text-align:left; }}
tr.review td {{ background:var(--flag); }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>{len(rows)} throws detected; {sum(r['needs_review'] for r in rows)} flagged for review (highlighted).</p>
<h2>Per player</h2><div class="wrap">{table(summary, s_cols)}</div>
<h2>Every throw</h2><div class="wrap">{table(rows, COLUMNS)}</div>
</body></html>"""
    Path(path).write_text(doc)


def write_all(throws: list[ThrowEvent], out_dir: Path, roster: Optional[dict] = None,
              session_start: Optional[datetime] = None, title: str = "Throwing session") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = to_rows(throws, roster, session_start)
    summary = summarize(rows)
    write_csv(rows, out_dir / "throws.csv", COLUMNS)
    write_csv(summary, out_dir / "players.csv")
    (out_dir / "throws.json").write_text(json.dumps(rows, indent=1))
    write_html(rows, summary, out_dir / "report.html", title)
    return {"rows": rows, "summary": summary}
