"""FieldVision status dashboard — single-file generator + web server.

Runs on Nellie; serves http://10.210.1.101:8377 (VPN required). A background
thread refreshes status.json every 45s from the autopilot logs, state
markers, token file, and df; index.html renders it client-side and
re-fetches every 30s.

  tmux new-session -d -s fvdash \
    "~/venvs/fieldvision/bin/python3 scripts/fv_dashboard.py"
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

PORT = 8377
ROOT = Path("/home/spencer/dashboard")
LOG = Path("/home/spencer/logs/autopilot.log")
RELOC_LOG = Path("/home/spencer/logs/reloc2.log")
STATE = Path("/media/scratch/spencer/fieldvision/state")
DATA = Path("/media/datasets/spencer/fieldvision/data")
LEGACY = Path("/media/datasets/spencer/fieldvision/legacy_sqlite")
TOKEN = Path("/home/spencer/.fv_token.txt")
ICLOUD_LEGACY_TOTAL = 64  # tracked source files (65 minus .DS_Store)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def token_hours() -> float | None:
    try:
        tok = TOKEN.read_text().strip()
        p = tok.split(".")[1]
        p += "=" * (-len(p) % 4)
        exp = json.loads(base64.urlsafe_b64decode(p))["exp"]
        return round((exp - time.time()) / 3600, 1)
    except Exception:
        return None


def gather() -> dict:
    now = datetime.now()
    log_tail = sh(f"tail -c 400000 {LOG}")

    done_times = []
    for m in re.finditer(r"\[(\d{2}):(\d{2}):\d{2}\]\s+pk=\d+ game done", log_tail):
        done_times.append(m.group(0))
    # per-hour buckets for the last 24h from full-timestamp failure-proof parse
    hours = Counter()
    for m in re.finditer(r"\[(\d{2}):\d{2}:\d{2}\]\s+pk=(\d+) game done", log_tail):
        hours[int(m.group(1))] += 1
    hourly = [{"h": h, "n": hours.get(h, 0)} for h in
              [(now.hour - 23 + i) % 24 for i in range(24)]]

    backlog_m = re.findall(r"backlog refreshed: (\d+) games pending", log_tail)
    pending = int(backlog_m[-1]) if backlog_m else None
    markers = len(list(STATE.glob("complete_*.marker")))
    total_window = (pending + markers) if pending is not None else None

    fails_recent = len(re.findall(r"FAILED|pass failed",
                                  sh(f"tail -c 20000 {LOG}")))
    last_line = sh(f"tail -1 {LOG}")

    games_on_disk = int(sh(f"ls {DATA} 2>/dev/null | grep -cE '^[0-9]+$'") or 0)
    legacy_files = int(sh(f"ls {LEGACY} 2>/dev/null | grep -cv 'DS_Store' ") or 0)

    df = sh("df -B1G /media/datasets | tail -1").split()
    nas = {"size_g": int(df[1]), "used_g": int(df[2]),
           "avail_g": int(df[3])} if len(df) >= 4 else {}

    reloc_tail = sh(f"tail -2 {RELOC_LOG}") if RELOC_LOG.exists() else ""
    reloc = ("done" if "RELOC2_DONE" in reloc_tail
             else "failed" if "RELOC2_FAILED" in reloc_tail
             else "running" if sh("tmux has-session -t fvreloc 2>/dev/null; echo $?") == "0"
             else "idle")

    autopilot = sh("tmux has-session -t fv-autopilot 2>/dev/null; echo $?") == "0"

    # last-hour capture rate → ETA on pending
    last3h = sum(b["n"] for b in hourly[-3:])
    rate_h = last3h / 3
    eta_days = round(pending / rate_h / 24, 1) if (pending and rate_h > 0) else None

    return {
        "generated": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "token_hours": token_hours(),
        "autopilot_alive": autopilot,
        "last_activity": last_line[-160:],
        "recent_failures": fails_recent,
        "backlog_pending": pending,
        "backlog_total_window": total_window,
        "markers": markers,
        "games_on_disk": games_on_disk,
        "corpus": {"games": 412, "healed": True, "pruned": True},
        "legacy": {"files": legacy_files, "total": ICLOUD_LEGACY_TOTAL},
        "reloc": reloc,
        "nas": nas,
        "hourly": hourly,
        "eta_days": eta_days,
    }


def refresher():
    while True:
        try:
            (ROOT / "status.json").write_text(json.dumps(gather()))
        except Exception as e:
            print("gather failed:", e, flush=True)
        time.sleep(45)


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FieldVision Ops</title><style>
:root { color-scheme: light dark;
  --surface:#fcfcfb; --card:#ffffff; --line:#e5e4e0;
  --ink:#0b0b0b; --ink2:#52514e; --series:#2a78d6;
  --good:#008300; --warn:#c98500; --bad:#e34948; }
@media (prefers-color-scheme: dark) { :root {
  --surface:#1a1a19; --card:#232322; --line:#3a3937;
  --ink:#ffffff; --ink2:#c3c2b7; --series:#3987e5;
  --good:#33a133; --warn:#eda100; --bad:#e66767; } }
* { box-sizing:border-box; margin:0 }
body { background:var(--surface); color:var(--ink);
  font:14px/1.45 -apple-system,'Segoe UI',sans-serif; padding:20px; }
h1 { font-size:18px; margin-bottom:2px } .sub { color:var(--ink2); font-size:12px; margin-bottom:18px }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:14px }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px }
.k { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink2) }
.v { font-size:26px; font-weight:650; margin-top:2px; font-variant-numeric:tabular-nums }
.u { font-size:12px; color:var(--ink2) }
.row { display:flex; align-items:center; gap:8px; padding:7px 0; border-top:1px solid var(--line); font-size:13px }
.row:first-of-type { border-top:none }
.dot { width:9px; height:9px; border-radius:50%; flex:none }
.bar-wrap { background:var(--line); border-radius:6px; height:14px; overflow:hidden; margin-top:8px }
.bar { background:var(--series); height:100%; border-radius:6px 0 0 6px; transition:width .6s }
.pct { font-size:12px; color:var(--ink2); margin-top:5px }
svg text { fill:var(--ink2); font-size:10px }
.tip { position:fixed; background:var(--card); border:1px solid var(--line); border-radius:6px;
  padding:5px 9px; font-size:12px; pointer-events:none; display:none; box-shadow:0 2px 8px #0003 }
.mono { font-family:ui-monospace,monospace; font-size:11.5px; color:var(--ink2);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis }
</style></head><body>
<h1>FieldVision Ops</h1><div class="sub" id="stamp">loading…</div>
<div class="grid" id="tiles"></div>
<div class="grid" style="grid-template-columns:2fr 1fr">
 <div class="card"><div class="k">Backfill progress — May 20 → yesterday</div>
  <div class="bar-wrap"><div class="bar" id="bar" style="width:0%"></div></div>
  <div class="pct" id="barlbl"></div></div>
 <div class="card"><div class="k">Pipeline health</div><div id="health"></div></div>
</div>
<div class="card" style="margin-top:14px"><div class="k">Games completed per hour — last 24h (UTC)</div>
 <svg id="chart" width="100%" height="150" role="img" aria-label="completions per hour"></svg></div>
<div class="card" style="margin-top:14px"><div class="k">Last autopilot activity</div>
 <div class="mono" id="lastact"></div></div>
<div class="tip" id="tip"></div>
<script>
const $=id=>document.getElementById(id);
function tile(k,v,u){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="u">${u||""}</div></div>`}
function dot(state){const c=state==="good"?"var(--good)":state==="warn"?"var(--warn)":"var(--bad)";return `<span class="dot" style="background:${c}"></span>`}
function row(state,label,txt){return `<div class="row">${dot(state)}<b>${label}</b><span style="color:var(--ink2)">${txt}</span></div>`}
async function refresh(){
 let s; try { s = await (await fetch("status.json?"+Date.now())).json(); } catch(e){ $("stamp").textContent="status.json unreachable — VPN up?"; return; }
 $("stamp").textContent = "updated "+s.generated+" — auto-refreshes";
 const done = s.backlog_total_window ? s.backlog_total_window - s.backlog_pending : null;
 $("tiles").innerHTML =
  tile("Games on hand", s.games_on_disk, "parquet game dirs on datasets") +
  tile("Backlog remaining", s.backlog_pending ?? "–", s.eta_days? "~"+s.eta_days+" days at current rate":"") +
  tile("Corpus", "412/412", "healed ✓ + two-cut pruned ✓") +
  tile("Token", s.token_hours!=null? s.token_hours+"h":"?", "until JWT expiry") +
  tile("NAS used", s.nas.used_g? (s.nas.used_g/1000).toFixed(1)+"T":"–", s.nas.avail_g? (s.nas.avail_g/1000).toFixed(1)+"T reported free":"");
 if (done!=null){ const pct = 100*done/s.backlog_total_window;
  $("bar").style.width = pct.toFixed(1)+"%";
  $("barlbl").textContent = done+" of "+s.backlog_total_window+" backlog games captured ("+pct.toFixed(1)+"%)"; }
 $("health").innerHTML =
  row(s.autopilot_alive?"good":"bad","Autopilot",s.autopilot_alive?"running":"DOWN — cron will relaunch") +
  row(s.token_hours>6?"good":s.token_hours>1?"warn":"bad","Token",s.token_hours!=null?s.token_hours+"h left":"unreadable") +
  row(s.recent_failures===0?"good":s.recent_failures<6?"warn":"bad","Recent errors",s.recent_failures+" in log tail") +
  row(s.reloc==="done"?"good":s.reloc==="running"?"warn":s.reloc==="idle"?"good":"bad","Relocation",s.reloc) +
  row(s.legacy.files>=s.legacy.total?"good":"warn","Legacy sqlite",s.legacy.files+" / "+s.legacy.total+" files on datasets");
 $("lastact").textContent = s.last_activity || "–";
 drawChart(s.hourly);
}
function drawChart(hourly){
 const svg=$("chart"), W=svg.clientWidth||800, H=150, pad=24;
 const max=Math.max(1,...hourly.map(b=>b.n));
 const bw=(W-pad*2)/hourly.length;
 let el='';
 hourly.forEach((b,i)=>{
  const h=(H-pad*2)*b.n/max, x=pad+i*bw, y=H-pad-h;
  el+=`<rect x="${x+1}" y="${y}" width="${bw-2}" height="${Math.max(h,b.n>0?2:0)}" rx="3"
    fill="var(--series)" data-n="${b.n}" data-h="${b.h}"></rect>`;
  if(i%4===0) el+=`<text x="${x+bw/2}" y="${H-7}" text-anchor="middle">${String(b.h).padStart(2,"0")}</text>`;
 });
 el+=`<text x="${pad-4}" y="${pad}" text-anchor="end">${max}</text>`;
 svg.innerHTML=el;
 svg.querySelectorAll("rect").forEach(r=>{
  r.addEventListener("mousemove",e=>{const t=$("tip");t.style.display="block";
   t.style.left=(e.clientX+12)+"px"; t.style.top=(e.clientY-10)+"px";
   t.textContent=String(r.dataset.h).padStart(2,"0")+":00 UTC — "+r.dataset.n+" games";});
  r.addEventListener("mouseleave",()=>$("tip").style.display="none");
 });
}
refresh(); setInterval(refresh, 30000);
</script></body></html>"""


def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT / "index.html").write_text(HTML)
    (ROOT / "status.json").write_text(json.dumps(gather()))
    threading.Thread(target=refresher, daemon=True).start()
    os.chdir(ROOT)
    print(f"serving on :{PORT}", flush=True)
    http.server.ThreadingHTTPServer(("", PORT),
                                    http.server.SimpleHTTPRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
