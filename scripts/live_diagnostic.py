"""
FieldVision Live Capture Diagnostic
====================================
One-shot verification against a live MLB Gameday 3D session. Answers four
questions before we commit to a capture-loop redesign:

  1. JOINT MAP — what are the real bone names at each skeleton index?
     (the existing fv_game_capture.py SKEL_JOINTS map disagrees with the
     canonical map in mannequin_metadata.json — this resolves which is right)

  2. PLAYER IDS — does world.actors.actors carry MLB player IDs natively?
     (if yes, problem 3 collapses to validate-and-persist; if no, we need
     Hungarian-style assignment from lineup + position)

  3. NATIVE FRAME RATE — how often does Hawk-Eye actually update bone data?
     (decide whether to capture at 30 Hz, 60 Hz, or whatever the page emits)

  4. CDP LATENCY — what does each page.evaluate() round trip cost?
     (informs whether to keep one-eval-per-frame or move buffering into the page)

Usage:
  python scripts/live_diagnostic.py --game 824198 --duration 30
  python scripts/live_diagnostic.py --url https://www.mlb.com/gameday/.../824198/live

Outputs:
  diagnostics/raw/diag_{gamePk}_{ts}.json   raw machine-readable dump
  console                                    human-readable summary
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright


# ────────────────────────────────────────────────────────────
# JS payloads
# ────────────────────────────────────────────────────────────

FIND_POSER_JS = """
() => {
    const fvDiv = document.querySelector('[class*="FieldVisionPlayerContainer"]')
                || document.querySelector('[class*="FieldVisionApp"]');
    if (!fvDiv) return { ready: false, reason: 'no_fv_container' };
    const fk = Object.keys(fvDiv).find(k => k.startsWith('__reactFiber'));
    if (!fk) return { ready: false, reason: 'no_fiber' };
    let fiber = fvDiv[fk], found = false;
    for (let d = 0; d < 30 && fiber; d++) {
        if (fiber.memoizedState) {
            let h = fiber.memoizedState;
            for (let i = 0; i < 25 && h; i++) {
                try {
                    if (h.memoizedState?.current?.poserContext) {
                        window.__pc = h.memoizedState.current.poserContext;
                        found = true; break;
                    }
                } catch(e) {}
                h = h.next;
            }
        }
        if (found) break;
        fiber = fiber.return;
    }
    if (!found) return { ready: false, reason: 'no_poser_context' };
    const world = window.__pc.world;
    let playerCount = 0;
    const armSet = new Set();
    world.traverse(obj => {
        if (obj.skeleton?.bones?.length === 103) {
            if (!armSet.has(obj.parent)) { armSet.add(obj.parent); playerCount++; }
        }
    });
    if (playerCount === 0) return { ready: false, reason: 'no_players_loaded' };
    return { ready: true, playerCount };
}
"""

# Returns the actual `.name` for each bone of the FIRST 103-bone skeleton
# we find. This is ground truth — we compare against canonical map afterward.
GET_BONE_NAMES_JS = """
() => {
    const armSet = new Set();
    const skels = [];
    window.__pc.world.traverse(obj => {
        if (obj.skeleton?.bones?.length === 103 && !armSet.has(obj.parent)) {
            armSet.add(obj.parent);
            skels.push(obj.skeleton);
        }
    });
    if (!skels.length) return { error: 'no_103_bone_skeletons' };
    const first = skels[0];
    const bones = first.bones.map((b, i) => ({
        idx: i,
        name: b?.name || null,
        parent: b?.parent?.name || null,
    }));
    return {
        skeletonCount: skels.length,
        boneCount: first.bones.length,
        bones,
    };
}
"""

# Probe what world.actors.actors actually contains. Without knowing if this
# is a Map, an Array, or a plain object, we try several access patterns.
DUMP_ACTORS_JS = """
() => {
    const w = window.__pc?.world;
    if (!w) return { error: 'no_world' };
    const probe = {};
    probe.worldKeys = Object.keys(w).filter(k => !k.startsWith('_')).slice(0, 50);
    const a = w.actors;
    probe.hasActors = !!a;
    if (!a) return probe;
    probe.actorsType = a.constructor?.name || typeof a;
    probe.actorsKeys = Object.keys(a).filter(k => !k.startsWith('_')).slice(0, 30);

    const inner = a.actors;
    probe.hasInnerActors = inner !== undefined;
    if (inner === undefined) return probe;
    probe.innerActorsType = inner?.constructor?.name || typeof inner;

    const dump = [];
    const captureActor = (actor, key) => {
        const entry = { mapKey: null, fields: {} };
        try {
            entry.mapKey = (typeof key === 'object' && key !== null)
                ? ('<' + (key.constructor?.name || 'obj') + '>')
                : String(key);
        } catch(e) { entry.mapKey = '<unrepresentable>'; }
        if (actor && typeof actor === 'object') {
            for (const k of Object.keys(actor)) {
                if (dump.length >= 30) break;
                let v;
                try { v = actor[k]; } catch(e) { entry.fields[k] = '<getter_threw>'; continue; }
                if (v === null || v === undefined) {
                    entry.fields[k] = v;
                } else if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
                    entry.fields[k] = v;
                } else if (typeof v === 'object') {
                    entry.fields[k] = '<' + (v.constructor?.name || 'obj') + '>';
                } else {
                    entry.fields[k] = '<' + typeof v + '>';
                }
            }
        }
        dump.push(entry);
    };

    try {
        if (typeof inner.forEach === 'function') {
            inner.forEach((actor, key) => { if (dump.length < 30) captureActor(actor, key); });
        } else if (Array.isArray(inner)) {
            inner.slice(0, 30).forEach((actor, i) => captureActor(actor, i));
        } else if (typeof inner === 'object') {
            Object.entries(inner).slice(0, 30).forEach(([k, v]) => captureActor(v, k));
        }
    } catch(e) {
        probe.iterationError = String(e);
    }
    probe.actorDump = dump;
    return probe;
}
"""

# Read pelvis (bone 0) world position of every 103-bone skeleton + a frame
# counter. We sample this at high frequency from Python to detect the native
# update rate (a frame counts as "new" when any pelvis position changes).
SAMPLE_PELVIS_JS = """
() => {
    if (!window.__fvDiagSkels) {
        const armSet = new Set();
        const skels = [];
        window.__pc.world.traverse(obj => {
            if (obj.skeleton?.bones?.length === 103 && !armSet.has(obj.parent)) {
                armSet.add(obj.parent);
                skels.push(obj.skeleton);
            }
        });
        window.__fvDiagSkels = skels;
    }
    const skels = window.__fvDiagSkels;
    const pelvises = [];
    for (const s of skels) {
        const e = s.bones[0]?.matrixWorld?.elements;
        if (e) pelvises.push([e[12], e[13], e[14]]);
    }
    return { tPage: performance.now(), pelvises };
}
"""

# Set up an in-page RAF counter so we can compare browser render rate vs
# data update rate.
INSTALL_RAF_COUNTER_JS = """
() => {
    if (window.__fvRaf) return { alreadyInstalled: true };
    window.__fvRaf = { count: 0, t0: performance.now() };
    const tick = () => {
        window.__fvRaf.count++;
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return { installed: true };
}
"""

READ_RAF_COUNTER_JS = """
() => ({
    count: window.__fvRaf.count,
    elapsedMs: performance.now() - window.__fvRaf.t0,
})
"""

# Canonical bone map (from mannequin_metadata.json) — used for comparison
CANONICAL_BONE_MAP = {
    0: "joint_Pelvis", 1: "joint_HipMaster", 2: "joint_HipRT", 3: "joint_KneeRT",
    4: "joint_FootRT", 5: "joint_BallRT", 6: "joint_ToeRT", 7: "joint_ToeRT_end",
    8: "joint_ThighRollRT", 9: "joint_ThighRollRT_end", 10: "joint_HipLT",
    11: "joint_KneeLT", 12: "joint_FootLT", 13: "joint_BallLT", 14: "joint_ToeLT",
    15: "joint_ToeLT_end", 16: "joint_ThighRollLT", 17: "joint_ThighRollLT_end",
    18: "joint_TorsoA", 19: "joint_TorsoB", 20: "joint_Neck", 21: "joint_Neck2",
    22: "joint_Head", 23: "joint_EyeRT", 24: "joint_EyeLT",
    25: "joint_ClavicleRT", 26: "joint_ShoulderRT", 27: "joint_ElbowRT",
    28: "joint_HandRT", 29: "joint_WeaponRT", 30: "joint_WeaponRT_end",
    64: "joint_ClavicleLT", 65: "joint_ShoulderLT", 66: "joint_ElbowLT",
    67: "joint_HandLT", 68: "joint_WeaponLT", 69: "joint_WeaponLT_end",
}

# Indices the existing fv_game_capture.py reads, with the names it claims
# they map to. We compare these against ground truth from GET_BONE_NAMES_JS.
EXISTING_SKEL_JOINTS_CLAIM = {
    0: "Pelvis", 1: "TorsoA", 2: "TorsoB", 3: "ClavicleLT", 6: "ShoulderLT",
    9: "ElbowLT", 14: "HandLT", 42: "ClavicleRT", 45: "ShoulderRT",
    48: "ElbowRT", 53: "HandRT", 81: "Neck", 83: "Head", 86: "HipMaster",
    87: "HipLT", 90: "KneeLT", 91: "FootLT", 95: "HipRT", 98: "KneeRT", 99: "FootRT",
}


# ────────────────────────────────────────────────────────────
# Diagnostic runner
# ────────────────────────────────────────────────────────────

class Diagnostic:
    def __init__(self, url: str, duration_s: float, headless: bool, output_dir: Path):
        self.url = url
        self.duration_s = duration_s
        self.headless = headless
        self.output_dir = output_dir
        parts = url.rstrip("/").split("/")
        self.game_pk = next((p for p in reversed(parts) if p.isdigit()), "unknown")
        self.results: dict = {
            "url": url,
            "gamePk": self.game_pk,
            "startedAt": datetime.now().isoformat(),
        }

    def log(self, msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    async def run(self) -> dict:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                page = await context.new_page()

                self.log(f"Loading {self.url}")
                # mlb.com keeps long-poll/websocket connections open, so
                # `networkidle` would never resolve. Use `domcontentloaded`
                # and rely on FIND_POSER_JS polling for actual readiness.
                await page.goto(self.url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(5000)

                # Click the 3D button if present
                try:
                    btn = page.locator('a:has-text("3D"), button:has-text("3D")')
                    if await btn.count() > 0:
                        self.log("Clicking 3D button")
                        await btn.first.click()
                        await page.wait_for_timeout(5000)
                except Exception as e:
                    self.log(f"3D button click skipped: {e}")

                # Wait for FieldVision to load
                self.log("Waiting for FieldVision (up to 90s)...")
                ready = None
                for attempt in range(45):
                    state = await page.evaluate(FIND_POSER_JS)
                    if state.get("ready"):
                        ready = state
                        self.log(f"FieldVision ready: {state['playerCount']} players")
                        break
                    if attempt % 5 == 0:
                        self.log(f"  waiting... reason={state.get('reason')}")
                    await page.wait_for_timeout(2000)

                self.results["fvReady"] = bool(ready)
                if not ready:
                    self.results["fvReadyReason"] = state
                    self.log("FieldVision did not become ready. Aborting.")
                    return self.results
                self.results["initialPlayerCount"] = ready["playerCount"]

                # ── Q1: bone names ──
                self.log("Probe 1/4: bone names")
                bone_data = await page.evaluate(GET_BONE_NAMES_JS)
                self.results["boneNames"] = bone_data
                self._compare_bone_map(bone_data)

                # ── Q2: actor dump ──
                self.log("Probe 2/4: actors / player IDs")
                actor_data = await page.evaluate(DUMP_ACTORS_JS)
                self.results["actors"] = actor_data
                self._summarize_actors(actor_data)

                # ── Q3: native frame rate via high-frequency pelvis sampling ──
                self.log(f"Probe 3/4: native frame rate ({self.duration_s:.0f}s sampling)")
                await page.evaluate(INSTALL_RAF_COUNTER_JS)
                samples = await self._sample_pelvis(page, self.duration_s)
                raf = await page.evaluate(READ_RAF_COUNTER_JS)
                self.results["nativeFps"] = self._analyse_samples(samples, raf)

                # ── Q4: CDP round-trip latency ──
                self.log("Probe 4/4: CDP round-trip latency")
                self.results["cdpLatency"] = await self._measure_cdp(page)

                self.results["finishedAt"] = datetime.now().isoformat()

            finally:
                await browser.close()

        return self.results

    async def _sample_pelvis(self, page, duration_s: float) -> list[dict]:
        samples: list[dict] = []
        end_time = time.monotonic() + duration_s
        while time.monotonic() < end_time:
            t0 = time.monotonic()
            data = await page.evaluate(SAMPLE_PELVIS_JS)
            t1 = time.monotonic()
            samples.append({
                "tHostMs": (t0 + t1) / 2 * 1000,
                "rttMs": (t1 - t0) * 1000,
                "tPage": data.get("tPage"),
                "pelvises": data.get("pelvises", []),
            })
        return samples

    def _analyse_samples(self, samples: list[dict], raf: dict) -> dict:
        if not samples:
            return {"error": "no_samples"}

        # Detect unique frames: a frame is "new" when ANY pelvis position differs
        # from the previous sample (rounded to 6 decimal places to ignore noise).
        def fingerprint(s):
            return tuple(tuple(round(v, 6) for v in p) for p in s["pelvises"])

        unique_frames: list[dict] = []
        last_fp = None
        for s in samples:
            fp = fingerprint(s)
            if fp != last_fp:
                unique_frames.append(s)
                last_fp = fp

        elapsed_s = (samples[-1]["tHostMs"] - samples[0]["tHostMs"]) / 1000
        if elapsed_s <= 0:
            elapsed_s = self.duration_s

        # Inter-arrival times of unique frames
        inter_arrivals_ms = [
            unique_frames[i]["tHostMs"] - unique_frames[i - 1]["tHostMs"]
            for i in range(1, len(unique_frames))
        ]

        rtts = [s["rttMs"] for s in samples]

        return {
            "totalSamples": len(samples),
            "uniqueFrames": len(unique_frames),
            "elapsedSeconds": round(elapsed_s, 3),
            "uniqueFps": round(len(unique_frames) / elapsed_s, 2) if elapsed_s else None,
            "samplePollFps": round(len(samples) / elapsed_s, 2) if elapsed_s else None,
            "interArrivalMs": {
                "p50": round(statistics.median(inter_arrivals_ms), 2) if inter_arrivals_ms else None,
                "p95": round(_p(inter_arrivals_ms, 0.95), 2) if inter_arrivals_ms else None,
                "p99": round(_p(inter_arrivals_ms, 0.99), 2) if inter_arrivals_ms else None,
                "max": round(max(inter_arrivals_ms), 2) if inter_arrivals_ms else None,
                "min": round(min(inter_arrivals_ms), 2) if inter_arrivals_ms else None,
                "mean": round(statistics.fmean(inter_arrivals_ms), 2) if inter_arrivals_ms else None,
                "stdev": round(statistics.stdev(inter_arrivals_ms), 2) if len(inter_arrivals_ms) > 1 else None,
            },
            "rttMs": {
                "p50": round(statistics.median(rtts), 2),
                "p95": round(_p(rtts, 0.95), 2),
                "p99": round(_p(rtts, 0.99), 2),
                "max": round(max(rtts), 2),
                "mean": round(statistics.fmean(rtts), 2),
            },
            "rafFps": round(raf["count"] / (raf["elapsedMs"] / 1000), 2) if raf.get("elapsedMs") else None,
        }

    async def _measure_cdp(self, page) -> dict:
        rtts = []
        for _ in range(50):
            t0 = time.monotonic()
            await page.evaluate("() => Date.now()")
            rtts.append((time.monotonic() - t0) * 1000)
        return {
            "samples": len(rtts),
            "p50Ms": round(statistics.median(rtts), 2),
            "p95Ms": round(_p(rtts, 0.95), 2),
            "p99Ms": round(_p(rtts, 0.99), 2),
            "maxMs": round(max(rtts), 2),
            "meanMs": round(statistics.fmean(rtts), 2),
        }

    def _compare_bone_map(self, bone_data: dict) -> None:
        if "error" in bone_data:
            self.log(f"  bone names: ERROR — {bone_data['error']}")
            return

        bones = bone_data.get("bones", [])
        index_to_name = {b["idx"]: b["name"] for b in bones}

        # Compare against canonical map
        canonical_matches = 0
        canonical_mismatches = []
        for idx, expected in CANONICAL_BONE_MAP.items():
            actual = index_to_name.get(idx)
            if actual == expected:
                canonical_matches += 1
            else:
                canonical_mismatches.append({"idx": idx, "expected": expected, "actual": actual})

        # Compare against existing fv_game_capture.py SKEL_JOINTS claim
        existing_matches = 0
        existing_mismatches = []
        for idx, claimed in EXISTING_SKEL_JOINTS_CLAIM.items():
            actual = index_to_name.get(idx)
            if actual and claimed.lower() in actual.lower():
                existing_matches += 1
            else:
                existing_mismatches.append({"idx": idx, "claimed": claimed, "actual": actual})

        self.results["boneMapVerdict"] = {
            "canonicalMatches": canonical_matches,
            "canonicalChecked": len(CANONICAL_BONE_MAP),
            "canonicalMismatches": canonical_mismatches[:10],
            "existingMatches": existing_matches,
            "existingChecked": len(EXISTING_SKEL_JOINTS_CLAIM),
            "existingMismatches": existing_mismatches[:10],
        }
        self.log(
            f"  canonical map: {canonical_matches}/{len(CANONICAL_BONE_MAP)} match  |  "
            f"existing fv_game_capture claim: {existing_matches}/{len(EXISTING_SKEL_JOINTS_CLAIM)} match"
        )

    def _summarize_actors(self, actor_data: dict) -> None:
        if actor_data.get("error"):
            self.log(f"  actors: ERROR — {actor_data['error']}")
            return
        dump = actor_data.get("actorDump", [])
        self.log(f"  actors: {len(dump)} entries  type={actor_data.get('innerActorsType')}")
        if dump:
            sample_keys = sorted(set(k for d in dump for k in d.get("fields", {}).keys()))
            id_like = [k for k in sample_keys if any(t in k.lower() for t in ("id", "actor", "mlb", "player"))]
            self.results["actorIdLikeFields"] = id_like
            self.log(f"  id-like fields seen: {id_like}")


def _p(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def print_report(results: dict) -> None:
    print()
    print("=" * 72)
    print(f"FieldVision Live Diagnostic — game {results.get('gamePk')}")
    print("=" * 72)
    print(f"URL:        {results.get('url')}")
    print(f"Started:    {results.get('startedAt')}")
    print(f"Finished:   {results.get('finishedAt', '(aborted)')}")
    print(f"FV ready:   {results.get('fvReady')}  players={results.get('initialPlayerCount', '?')}")

    verdict = results.get("boneMapVerdict")
    if verdict:
        print()
        print("─ Bone map ──────────────────────────────────────────────────────────")
        cm, ce = verdict["canonicalMatches"], verdict["canonicalChecked"]
        em, ee = verdict["existingMatches"], verdict["existingChecked"]
        print(f"  canonical map (mannequin_metadata.json): {cm}/{ce} indices match")
        print(f"  existing capture script claim:           {em}/{ee} indices match")
        if verdict["canonicalMismatches"]:
            print("  first canonical mismatches:")
            for m in verdict["canonicalMismatches"][:5]:
                print(f"    idx {m['idx']}: expected {m['expected']!r}  actual {m['actual']!r}")

    actors = results.get("actors")
    if actors and not actors.get("error"):
        print()
        print("─ Actors / player IDs ───────────────────────────────────────────────")
        print(f"  inner type: {actors.get('innerActorsType')}  entries: {len(actors.get('actorDump', []))}")
        id_like = results.get("actorIdLikeFields", [])
        print(f"  id-like fields: {id_like or '(none — Hungarian matching will be needed)'}")
        for entry in actors.get("actorDump", [])[:3]:
            print(f"    sample: key={entry.get('mapKey')}  fields={list(entry.get('fields', {}).keys())[:8]}")

    fps = results.get("nativeFps")
    if fps and not fps.get("error"):
        print()
        print("─ Frame rate ────────────────────────────────────────────────────────")
        print(f"  poll rate (host):       {fps['samplePollFps']} Hz")
        print(f"  unique data frames:     {fps['uniqueFps']} Hz  ({fps['uniqueFrames']} over {fps['elapsedSeconds']}s)")
        print(f"  in-page RAF:            {fps['rafFps']} Hz")
        ia = fps["interArrivalMs"]
        print(f"  inter-arrival ms:       p50={ia['p50']}  p95={ia['p95']}  p99={ia['p99']}  max={ia['max']}")
        rtt = fps["rttMs"]
        print(f"  page.evaluate RTT ms:   p50={rtt['p50']}  p95={rtt['p95']}  p99={rtt['p99']}  max={rtt['max']}")

    cdp = results.get("cdpLatency")
    if cdp:
        print()
        print("─ CDP round-trip ────────────────────────────────────────────────────")
        print(f"  page.evaluate('() => Date.now()'):  p50={cdp['p50Ms']}ms  p95={cdp['p95Ms']}ms  max={cdp['maxMs']}ms")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="FieldVision live diagnostic")
    parser.add_argument("--url", type=str, help="Full Gameday URL")
    parser.add_argument("--game", type=int, help="Game PK (constructs the URL)")
    parser.add_argument("--duration", type=float, default=20.0, help="Sampling duration in seconds")
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"),
        help="Directory for raw JSON output",
    )
    args = parser.parse_args()

    if args.url:
        url = args.url
    elif args.game:
        url = f"https://www.mlb.com/gameday/{args.game}/live"
    else:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    diag = Diagnostic(url=url, duration_s=args.duration, headless=not args.headed, output_dir=output_dir)
    results = asyncio.run(diag.run())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"diag_{diag.game_pk}_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print_report(results)
    print(f"Raw output: {out_path}")


if __name__ == "__main__":
    main()
