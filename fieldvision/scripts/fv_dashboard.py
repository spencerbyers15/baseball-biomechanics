"""FieldVision status dashboard — single-file generator + web server.

Runs on Nellie; serves http://10.210.1.101:8377 (VPN required).
Threads: fast stats every 45s; capture-calendar + captured-set every 10 min
(CIFS footer reads are slow); dataset du every 30 min. index.html renders
client-side and re-fetches every 30s.

  tmux new-session -d -s fvdash \
    "~/venvs/fieldvision/bin/python3 scripts/fv_dashboard.py"
"""

from __future__ import annotations

import base64
import http.server
import json
import re
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, date, timedelta
from pathlib import Path

PORT = 8377
ROOT = Path("/home/spencer/dashboard")
LOG = Path("/home/spencer/logs/autopilot.log")
RELOC_LOG = Path("/home/spencer/logs/reloc2.log")
STATE = Path("/media/scratch/spencer/fieldvision/state")
DATASET_ROOT = Path("/media/datasets/spencer/fieldvision")
DATA_TREES = [DATASET_ROOT / "data",
              Path("/media/scratch/spencer/fieldvision/data")]  # scratch until migration done
LEGACY = DATASET_ROOT / "legacy_sqlite"
TOKEN = Path("/home/spencer/.fv_token.txt")
ICLOUD_LEGACY_TOTAL = 64
SEASON_START = "2026-04-14"  # first corpus game

_slow = {"du_bytes": None, "calendar": [], "captured": set(),
         "games_captured": 0, "sched_map": {}}


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception:
        return ""


def token_hours() -> float | None:
    try:
        tok = TOKEN.read_text().strip()
        p = tok.split(".")[1]
        p += "=" * (-len(p) % 4)
        return round((json.loads(base64.urlsafe_b64decode(p))["exp"]
                      - time.time()) / 3600, 1)
    except Exception:
        return None


def finalized(p: Path) -> bool:
    try:
        if p.stat().st_size < 8:
            return False
        with p.open("rb") as f:
            f.seek(-4, 2)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def captured_set() -> set[int]:
    have = set()
    for tree in DATA_TREES:
        if not tree.is_dir():
            continue
        for gdir in tree.iterdir():
            if not gdir.name.isdigit() or int(gdir.name) in have:
                continue
            if any(finalized(f) for f in gdir.glob("actor_frames*.parquet")):
                have.add(int(gdir.name))
    return have


def fetch_schedule() -> list[dict]:
    """Full-season schedule, cached for an hour."""
    cache = ROOT / "sched_cache.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 3600:
        return json.loads(cache.read_text())
    end = date.today().isoformat()
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&startDate={SEASON_START}&endDate={end}")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    out = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            t = g.get("teams", {})
            out.append({"date": day["date"], "pk": g["gamePk"],
                        "away": t.get("away", {}).get("team", {}).get("name", "?"),
                        "home": t.get("home", {}).get("team", {}).get("name", "?")})
    cache.write_text(json.dumps(out))
    return out


def slow_refresher():
    while True:
        try:
            have = captured_set()
            sched = fetch_schedule()
            by_day: dict[str, dict] = {}
            for g in sched:
                e = by_day.setdefault(g["date"], {"total": 0, "captured": 0,
                                                  "missing": []})
                e["total"] += 1
                if g["pk"] in have:
                    e["captured"] += 1
                else:
                    e["missing"].append(f'{g["away"]} @ {g["home"]}')
            _slow["calendar"] = [{"date": d, **v}
                                 for d, v in sorted(by_day.items())]
            _slow["captured"] = have
            _slow["games_captured"] = len(have)
            _slow["sched_map"] = {g["pk"]: g for g in sched}
        except Exception as e:
            print("slow refresh failed:", e, flush=True)
        # dataset size (du is minutes-slow on CIFS; every 3rd cycle ≈ 30 min)
        try:
            du = sh(f"du -sb {DATASET_ROOT} 2>/dev/null | cut -f1")
            if du.isdigit():
                _slow["du_bytes"] = int(du)
        except Exception:
            pass
        time.sleep(600)


def live_slate() -> list[dict]:
    """Today's schedule with statuses, cached 5 min (matchups for live rows)."""
    cache = ROOT / "live_cache.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 300:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    try:
        # MLB slates straddle UTC midnight — fetch yesterday+today.
        d0 = (date.today() - timedelta(days=1)).isoformat()
        d1 = date.today().isoformat()
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
               f"&startDate={d0}&endDate={d1}")
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.load(r)
        out = [{"pk": g["gamePk"],
                "away": g["teams"]["away"]["team"]["name"],
                "home": g["teams"]["home"]["team"]["name"],
                "state": g["status"]["abstractGameState"]}
               for day in d.get("dates", []) for g in day.get("games", [])]
        cache.write_text(json.dumps(out))
        return out
    except Exception:
        return []


def gather() -> dict:
    now = datetime.now()
    log_tail = sh(f"tail -c 400000 {LOG}")
    hours = Counter()
    for m in re.finditer(r"\[(\d{2}):\d{2}:\d{2}\]\s+pk=\d+ game done", log_tail):
        hours[int(m.group(1))] += 1
    hourly = [{"h": h, "n": hours.get(h, 0)} for h in
              [(now.hour - 23 + i) % 24 for i in range(24)]]

    backlog_m = re.findall(r"backlog refreshed: (\d+) games pending", log_tail)
    pending = int(backlog_m[-1]) if backlog_m else None
    markers = len(list(STATE.glob("complete_*.marker")))

    last3h = sum(b["n"] for b in hourly[-3:])
    eta_days = (round(pending / (last3h / 3) / 24, 1)
                if pending and last3h else None)

    reloc_tail = sh(f"tail -2 {RELOC_LOG}") if RELOC_LOG.exists() else ""
    reloc = ("done" if "RELOC2_DONE" in reloc_tail
             else "failed" if "RELOC2_FAILED" in reloc_tail
             else "running" if sh("tmux has-session -t fvreloc 2>/dev/null; echo $?") == "0"
             else "idle")

    recent = sh(f"tail -c 60000 {LOG}")
    working = {}
    for m in re.finditer(r"pk=(\d+) to fetch: (\d+) of (\d+) "
                         r"\((\d+) known gaps, (\d+) already ingested\)", recent):
        pk = int(m.group(1))
        working[pk] = {"pk": pk, "need": int(m.group(2)),
                       "have": int(m.group(5)), "fetched": 0,
                       "rate": None, "pos": m.start()}
    for m in re.finditer(r"pk=(\d+) game done", recent):
        pk = int(m.group(1))
        if pk in working and m.start() > working[pk]["pos"]:
            del working[pk]
    for m in re.finditer(r"pk=(\d+) (\d+)/\d+\s+\(([\d.]+) seg/s", recent):
        pk = int(m.group(1))
        if pk in working and m.start() > working[pk]["pos"]:
            working[pk]["fetched"] = int(m.group(2))
            working[pk]["rate"] = float(m.group(3))
    # Persistent LIVE rows: live games are tracked with quick passes every
    # ~5 min that finish in seconds, so 'currently in-flight' misses them.
    live_rows = []
    for g in live_slate():
        if g["state"] != "Live":
            continue
        pk = g["pk"]
        cov = None
        last = None
        for m in re.finditer(rf"pk={pk} to fetch: (\d+) of (\d+) "
                             rf"\((\d+) known gaps, (\d+) already ingested\)",
                             recent):
            need, tot, gaps, have = map(int, m.groups())
            non_gap = max(tot - gaps, 1)
            cov = round(100 * have / non_gap, 1)
            last = m.start()
        in_pass = pk in working
        live_rows.append({"pk": pk,
                          "matchup": g["away"] + " @ " + g["home"],
                          "cov": cov,
                          "status": "in pass now" if in_pass else "tracking"})
        working.pop(pk, None)  # avoid double-listing below

    cutoff = (date.today() - timedelta(days=1)).isoformat()
    work = []
    for w in working.values():
        g = _slow["sched_map"].get(w["pk"], {})
        goal = w["have"] + w["need"]
        pct = round(100 * (w["have"] + w["fetched"]) / goal, 1) if goal else 0
        work.append({"pk": w["pk"],
                     "matchup": (g.get("away", "?") + " @ " + g.get("home", "?"))
                                if g else "?",
                     "date": g.get("date", "?"),
                     "kind": "live/recent" if g.get("date", "") >= cutoff else "backlog",
                     "pct": pct, "need": w["need"], "fetched": w["fetched"],
                     "rate": w["rate"]})
    work.sort(key=lambda x: x["date"])

    return {
        "generated": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "token_hours": token_hours(),
        "autopilot_alive": sh("tmux has-session -t fv-autopilot 2>/dev/null; echo $?") == "0",
        "last_activity": sh(f"tail -1 {LOG}")[-160:],
        "recent_failures": len(re.findall(r"FAILED|pass failed",
                                          sh(f"tail -c 20000 {LOG}"))),
        "backlog_pending": pending,
        "backlog_total_window": (pending + markers) if pending is not None else None,
        "markers": markers,
        "games_captured": _slow["games_captured"],
        "dataset_bytes": _slow["du_bytes"],
        "legacy": {"files": int(sh(f"ls {LEGACY} 2>/dev/null | grep -cv DS_Store") or 0),
                   "total": ICLOUD_LEGACY_TOTAL},
        "reloc": reloc,
        "hourly": hourly,
        "eta_days": eta_days,
        "calendar": _slow["calendar"],
        "working": work,
        "live_now": live_rows,
    }


def fast_refresher():
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
  --good:#008300; --goodbg:#e2f2e2; --warn:#c98500; --warnbg:#fdf1d8;
  --bad:#e34948; --badbg:#fbe3e3; }
@media (prefers-color-scheme: dark) { :root {
  --surface:#1a1a19; --card:#232322; --line:#3a3937;
  --ink:#ffffff; --ink2:#c3c2b7; --series:#3987e5;
  --good:#33a133; --goodbg:#1d2e1d; --warn:#eda100; --warnbg:#33290f;
  --bad:#e66767; --badbg:#392020; } }
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
.bar { background:var(--series); height:100%; transition:width .6s }
.pct { font-size:12px; color:var(--ink2); margin-top:5px }
svg text { fill:var(--ink2); font-size:10px }
.tip { position:fixed; background:var(--card); border:1px solid var(--line); border-radius:6px;
  padding:6px 10px; font-size:12px; pointer-events:none; display:none;
  box-shadow:0 2px 8px #0003; max-width:320px; z-index:9 }
.mono { font-family:ui-monospace,monospace; font-size:11.5px; color:var(--ink2);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis }
.months { display:flex; flex-wrap:wrap; gap:16px; margin-top:8px }
.month { }
.mname { font-size:12px; font-weight:600; margin-bottom:5px; color:var(--ink2) }
.mgrid { display:grid; grid-template-columns:repeat(7,30px); gap:3px }
.day { height:30px; border-radius:5px; font-size:9.5px; display:flex; flex-direction:column;
  align-items:center; justify-content:center; line-height:1.1; cursor:default;
  border:1px solid var(--line); color:var(--ink2) }
.day b { font-size:10px; color:var(--ink) }
.day.g { background:var(--goodbg); border-color:var(--good) }
.day.w { background:var(--warnbg); border-color:var(--warn) }
.day.b { background:var(--badbg); border-color:var(--bad) }
.legend { font-size:11px; color:var(--ink2); margin-top:8px; display:flex; gap:14px }
.sw { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:4px; vertical-align:-1px }
</style></head><body>
<h1>FieldVision Ops</h1><div class="sub" id="stamp">loading…</div>
<div class="grid" id="tiles"></div>
<div class="card" style="margin-bottom:14px"><div class="k">Currently working on</div>
 <div id="working"></div></div>
<div class="grid" style="grid-template-columns:2fr 1fr">
 <div class="card"><div class="k">Backfill progress — May 20 → yesterday</div>
  <div class="bar-wrap"><div class="bar" id="bar" style="width:0%"></div></div>
  <div class="pct" id="barlbl"></div></div>
 <div class="card"><div class="k">Pipeline health</div><div id="health"></div></div>
</div>
<div class="card" style="margin-top:14px"><div class="k">Capture calendar — every scheduled game, all season</div>
 <div class="months" id="cal"></div>
 <div class="legend">
  <span><span class="sw" style="background:var(--goodbg);border:1px solid var(--good)"></span>all captured</span>
  <span><span class="sw" style="background:var(--warnbg);border:1px solid var(--warn)"></span>partial</span>
  <span><span class="sw" style="background:var(--badbg);border:1px solid var(--bad)"></span>none</span>
  <span>cell = captured/total — hover for missing matchups</span></div></div>
<div class="card" style="margin-top:14px"><div class="k">Games completed per hour — last 24h (UTC)</div>
 <svg id="chart" width="100%" height="150" role="img" aria-label="completions per hour"></svg></div>
<div class="card" style="margin-top:14px"><div class="k">Last autopilot activity</div>
 <div class="mono" id="lastact"></div></div>
<div class="tip" id="tip"></div>
<script>
const $=id=>document.getElementById(id);
const tipShow=(e,t)=>{const el=$("tip");el.style.display="block";el.innerHTML=t;
 el.style.left=Math.min(e.clientX+12,innerWidth-330)+"px";el.style.top=(e.clientY+14)+"px"};
const tipHide=()=>$("tip").style.display="none";
function tile(k,v,u){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="u">${u||""}</div></div>`}
function dot(s){const c=s==="good"?"var(--good)":s==="warn"?"var(--warn)":"var(--bad)";return `<span class="dot" style="background:${c}"></span>`}
function row(s,l,t){return `<div class="row">${dot(s)}<b>${l}</b><span style="color:var(--ink2)">${t}</span></div>`}
function fmtT(b){return b==null?"measuring…":(b/1e12).toFixed(2)+" TB"}
async function refresh(){
 let s; try { s = await (await fetch("status.json?"+Date.now())).json(); } catch(e){ $("stamp").textContent="status.json unreachable — VPN up?"; return; }
 $("stamp").textContent = "updated "+s.generated+" — auto-refreshes";
 const done = s.backlog_total_window ? s.backlog_total_window - s.backlog_pending : null;
 $("tiles").innerHTML =
  tile("Games captured", s.games_captured, "unique games with skeletal data") +
  tile("Backlog remaining", s.backlog_pending ?? "–", s.eta_days? "~"+s.eta_days+" days at current rate":"") +
  tile("Dataset size on NAS", fmtT(s.dataset_bytes), "everything under datasets/spencer/fieldvision") +
  tile("Token", s.token_hours!=null? s.token_hours+"h":"?", "until JWT expiry");
 if (done!=null){ const pct=100*done/s.backlog_total_window;
  $("bar").style.width=pct.toFixed(1)+"%";
  $("barlbl").textContent=done+" of "+s.backlog_total_window+" backlog games captured ("+pct.toFixed(1)+"%)"; }
 $("health").innerHTML =
  row(s.autopilot_alive?"good":"bad","Autopilot",s.autopilot_alive?"running":"DOWN — cron relaunches within 10 min") +
  row("good","Apr–May archive","412 of the games captured — healed + pruned (fixed subset)") +
  row(s.token_hours>6?"good":s.token_hours>1?"warn":"bad","Token",s.token_hours!=null?s.token_hours+"h left":"unreadable") +
  row(s.recent_failures===0?"good":s.recent_failures<6?"warn":"bad","Recent errors",s.recent_failures+" in log tail") +
  row(s.reloc==="done"||s.reloc==="idle"?"good":s.reloc==="running"?"warn":"bad","Relocation",s.reloc) +
  row(s.legacy.files>=s.legacy.total?"good":"warn","Legacy sqlite",s.legacy.files+" / "+s.legacy.total+" files on datasets");
 const liveRows = (s.live_now||[]).map(w=>`<div class="row"><span class="dot" style="background:var(--warn)"></span>
      <b>${w.matchup}</b><span style="color:var(--ink2)">LIVE · ${w.status}${w.cov!=null?" · "+w.cov+"% of game so far":""}</span></div>`).join("");
 $("working").innerHTML = liveRows + ((s.working && s.working.length)
   ? s.working.map(w=>`<div class="row"><span class="dot" style="background:${w.kind==="backlog"?"var(--series)":"var(--warn)"}"></span>
      <b>${w.matchup}</b><span style="color:var(--ink2)">${w.date} · ${w.kind} · pass ${w.fetched}/${w.need} segs${w.rate?" @ "+w.rate+" seg/s":""}</span>
      <span style="margin-left:auto;font-variant-numeric:tabular-nums"><b>${w.pct}%</b> of game</span></div>`).join("")
   : (liveRows ? "" : `<div class="u">idle — between passes (autopilot ticks every ~60s)</div>`));
 $("lastact").textContent = s.last_activity || "–";
 drawCal(s.calendar); drawChart(s.hourly);
}
function drawCal(cal){
 if(!cal||!cal.length){$("cal").innerHTML="<span class='u'>building… (first pass takes ~2 min)</span>";return}
 const byM={};
 cal.forEach(d=>{const m=d.date.slice(0,7);(byM[m]=byM[m]||[]).push(d)});
 const MN=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
 let html="";
 Object.keys(byM).sort().forEach(m=>{
  const [y,mo]=m.split("-").map(Number);
  const first=new Date(Date.UTC(y,mo-1,1)).getUTCDay(); // 0=Sun
  const days={}; byM[m].forEach(d=>days[+d.date.slice(8)]=d);
  const nd=new Date(Date.UTC(y,mo,0)).getUTCDate();
  let cells="";
  for(let i=0;i<first;i++) cells+="<div></div>";
  for(let day=1;day<=nd;day++){
   const d=days[day];
   if(!d){cells+=`<div class="day"><b>${day}</b></div>`;continue}
   const cls=d.captured>=d.total?"g":d.captured>0?"w":"b";
   const miss=d.missing.length?("<b>missing:</b><br>"+d.missing.join("<br>")):"all captured ✓";
   cells+=`<div class="day ${cls}" data-tip="<b>${d.date}</b> — ${d.captured}/${d.total}<br>${miss}">
     <b>${day}</b><span>${d.captured}/${d.total}</span></div>`;
  }
  html+=`<div class="month"><div class="mname">${MN[mo-1]} ${y}</div><div class="mgrid">${cells}</div></div>`;
 });
 $("cal").innerHTML=html;
 document.querySelectorAll(".day[data-tip]").forEach(el=>{
  el.addEventListener("mousemove",e=>tipShow(e,el.dataset.tip));
  el.addEventListener("mouseleave",tipHide);});
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
  r.addEventListener("mousemove",e=>tipShow(e,String(r.dataset.h).padStart(2,"0")+":00 UTC — "+r.dataset.n+" games"));
  r.addEventListener("mouseleave",tipHide);});
}
refresh(); setInterval(refresh, 30000);
</script></body></html>"""


def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT / "index.html").write_text(HTML)
    threading.Thread(target=slow_refresher, daemon=True).start()
    threading.Thread(target=fast_refresher, daemon=True).start()
    print(f"serving on :{PORT}", flush=True)
    import os
    os.chdir(ROOT)
    http.server.ThreadingHTTPServer(("", PORT),
                                    http.server.SimpleHTTPRequestHandler).serve_forever()


if __name__ == "__main__":
    main()
