"""
MLB FieldVision Whole-Game Capture
===================================
Captures Hawk-Eye skeletal data for an ENTIRE game, streaming frames to disk
as they arrive. Handles idle periods between pitches, commercial breaks, 
mid-inning delays, and reconnects if FieldVision stalls.

This is designed to be fire-and-forget: launch it when a game starts and 
it will run until the game ends or a max duration is hit, writing
.jsonl (one frame per line) so nothing is lost on crash.

Usage (called directly):
  python fv_game_capture.py --game 824855 --max-hours 4 --output ./captures/

Usage (called by the scheduler):
  The fv_scheduler.py script spawns one of these per game automatically.

Output:
  captures/fv_{gamePk}_{timestamp}/
      ├── metadata.json       (game info, player IDs, connections, joint names)
      └── frames.jsonl        (one JSON frame per line, streamed as captured)

To convert JSONL → single JSON (for the viewer):
  python fv_game_capture.py --convert ./captures/fv_XXX_YYY/
"""

import asyncio
import argparse
import json
import time
import sys
import signal
from pathlib import Path
from datetime import datetime, timedelta

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# JS injection — same as fv_headless.py (proven correct)
# ═══════════════════════════════════════════════════════════

FIND_POSER_JS = """
() => {
    // mlb.com Gameday redesign (~2026-05) renamed the FieldVision wrapper
    // to "DynamicFieldVision..." and removed the 2D→3D toggle. Use a
    // case-insensitive partial match so we catch all variants.
    // (Fix 2026-05-06.)
    const fvDiv = document.querySelector('[class*="FieldVisionPlayerContainer"]')
                || document.querySelector('[class*="FieldVisionApp"]')
                || document.querySelector('[class*="FieldVision" i]');
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

SETUP_CAPTURE_JS = """
() => {
    const SKEL_JOINTS = [0,1,2,3,6,9,14,42,45,48,53,81,83,86,87,90,91,95,98,99];
    const SKEL_CONNS = [
        [0,1],[1,2],[2,81],[81,83],[0,86],[86,95],[95,98],[98,99],
        [86,87],[87,90],[90,91],[2,42],[42,45],[45,48],[48,53],
        [2,3],[3,6],[6,9],[9,14]
    ];
    const BAT_CONNS = [[200,201],[201,202],[202,203],[203,204]];
    const JOINT_NAMES = {
        0:'Pelvis',1:'TorsoA',2:'TorsoB',3:'ClavicleLT',6:'ShoulderLT',
        9:'ElbowLT',14:'HandLT',42:'ClavicleRT',45:'ShoulderRT',
        48:'ElbowRT',53:'HandRT',81:'Neck',83:'Head',86:'HipMaster',
        87:'HipLT',90:'KneeLT',91:'FootLT',95:'HipRT',98:'KneeRT',99:'FootRT',
        200:'BatBottom',201:'BatHandle',202:'BatBody',203:'BatSpot',204:'BatTop'
    };
    const world = window.__pc.world;
    const skels = [];
    const armSet = new Set();
    let batMesh = null;
    world.traverse(obj => {
        if (obj.name === 'Bat' && obj.skeleton?.bones?.length === 5) batMesh = obj;
        if (obj.skeleton?.bones?.length === 103) {
            const arm = obj.parent;
            if (!armSet.has(arm)) {
                armSet.add(arm);
                skels.push({ arm, bones: obj.skeleton.bones });
            }
        }
    });
    const playerIds = [];
    try {
        const actorsObj = window.__pc.world.actors;
        if (actorsObj?.actors?.forEach) {
            actorsObj.actors.forEach((actor, key) => {
                const id = actor?.actorId || actor?.id || actor?.mlbId || key;
                const type = actor?.actorType || actor?.type || 'unknown';
                playerIds.push({ mlbId: id, type });
            });
        }
    } catch(e) {}
    window.__fvCache = { skels, batMesh, SKEL_JOINTS, playerIds };
    return {
        playerCount: skels.length,
        hasBat: !!batMesh,
        playerIds: playerIds.slice(0, 25),
        connections: [...SKEL_CONNS, ...BAT_CONNS],
        jointIndices: [...SKEL_JOINTS, 200, 201, 202, 203, 204],
        jointNames: JOINT_NAMES
    };
}
"""

CAPTURE_FRAME_JS = """
() => {
    const cache = window.__fvCache;
    if (!cache) return null;
    const frame = { t: Date.now(), p: [] };
    for (let si = 0; si < cache.skels.length; si++) {
        const skel = cache.skels[si];
        const j = {};
        for (const idx of cache.SKEL_JOINTS) {
            const e = skel.bones[idx]?.matrixWorld?.elements;
            if (e) j[idx] = [+(e[12].toFixed(2)), +(e[13].toFixed(2)), +(e[14].toFixed(2))];
        }
        let hasBat = false;
        if (cache.batMesh) {
            const hand = j[53];
            if (hand) {
                const bh = cache.batMesh.skeleton.bones[1];
                if (bh?.matrixWorld) {
                    const he = bh.matrixWorld.elements;
                    const dist = Math.sqrt((hand[0]-he[12])**2 + (hand[1]-he[13])**2 + (hand[2]-he[14])**2);
                    if (dist < 3) {
                        hasBat = true;
                        cache.batMesh.skeleton.bones.forEach((bb, bi) => {
                            const e = bb?.matrixWorld?.elements;
                            if (e) j[200+bi] = [+(e[12].toFixed(2)), +(e[13].toFixed(2)), +(e[14].toFixed(2))];
                        });
                    }
                }
            }
        }
        const pid = cache.playerIds[si];
        frame.p.push({
            v: skel.arm.visible,
            j,
            bat: hasBat,
            mlbId: pid?.mlbId || null,
            type: pid?.type || null
        });
    }
    return frame;
}
"""

CHECK_GAME_STATE_JS = """
() => {
    try {
        const gs = window.__pc?.gameState;
        const abstract = gs?.state?.gameData?.status?.abstractGameState
                      || gs?.boxscore?.status?.abstractGameState
                      || 'unknown';
        const detailed = gs?.state?.gameData?.status?.detailedState
                      || gs?.boxscore?.status?.detailedState
                      || 'unknown';
        return { abstract, detailed };
    } catch(e) { return { abstract: 'unknown', detailed: 'error: ' + e.message }; }
}
"""


class GameCapture:
    """Streams frames to JSONL throughout an entire game."""

    def __init__(self, game_url: str, output_root: Path, max_hours: float = 4.5,
                 headless: bool = True, capture_rate_hz: float = 20.0):
        self.game_url = game_url
        self.output_root = output_root
        self.max_seconds = int(max_hours * 3600)
        self.headless = headless
        self.capture_interval = 1.0 / capture_rate_hz

        # Extract gamePk for filenames
        parts = game_url.rstrip('/').split('/')
        self.game_pk = next((p for p in reversed(parts) if p.isdigit()), 'unknown')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = output_root / f'fv_{self.game_pk}_{timestamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.session_dir / 'metadata.json'
        self.frames_path = self.session_dir / 'frames.jsonl'
        self.log_path = self.session_dir / 'capture.log'

        self.frames_written = 0
        self.last_frame_hash = None  # detect stalled data
        self.stall_count = 0

    def log(self, msg: str):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    async def run(self):
        self.log(f'Starting capture for {self.game_url}')
        self.log(f'Session dir: {self.session_dir}')
        self.log(f'Max duration: {self.max_seconds}s ({self.max_seconds/3600:.1f}h)')

        start_time = time.time()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled',
                      '--disable-dev-shm-usage']
            )
            try:
                context = await browser.new_context(
                    viewport={'width': 1280, 'height': 800},
                    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = await context.new_page()

                # Load page
                self.log('Loading page...')
                # mlb.com keeps long-poll/websocket connections open continuously,
                # so wait_until='networkidle' never resolves and goto times out.
                # Use 'domcontentloaded' and rely on FIND_POSER_JS polling below
                # for readiness. (Fix 2026-05-06.)
                await page.goto(self.game_url, wait_until='domcontentloaded', timeout=60000)
                await page.wait_for_timeout(5000)

                # Accept cookies banner if present — it overlays the bottom of the
                # page and intercepts clicks on UI behind it. Try several common
                # button labels. (Fix 2026-05-06.)
                cookie_labels = ['Accept', 'Accept All', 'Accept all', 'I Accept',
                                 'OK', 'Got it', 'Agree', 'I agree']
                for lbl in cookie_labels:
                    try:
                        cb = page.locator(f'button:has-text("{lbl}")')
                        if await cb.count() > 0:
                            await cb.first.click(timeout=3000)
                            self.log(f'Accepted cookies banner ({lbl!r})')
                            await page.wait_for_timeout(2000)
                            break
                    except Exception:
                        continue

                # The 3D toggle isn't always rendered immediately on hydrate.
                # Poll for it with multiple selector strategies for up to 30s.
                self.log('Looking for 3D toggle...')
                three_d_selectors = [
                    'button[aria-label*="3D"]',
                    'a[aria-label*="3D"]',
                    'button:has-text("3D")',
                    'a:has-text("3D")',
                    '[data-testid*="3d"]',
                    '[data-testid*="3D"]',
                    '[class*="threeDee"]',
                    '[class*="ThreeDee"]',
                ]
                clicked_3d = False
                for attempt in range(15):  # up to ~30s
                    for sel in three_d_selectors:
                        try:
                            btn = page.locator(sel)
                            if await btn.count() > 0:
                                # Scroll into view first; the button may be off-screen
                                await btn.first.scroll_into_view_if_needed(timeout=2000)
                                await btn.first.click(timeout=3000)
                                self.log(f'Clicked 3D toggle via selector {sel!r}')
                                clicked_3d = True
                                await page.wait_for_timeout(5000)
                                break
                        except Exception:
                            continue
                    if clicked_3d:
                        break
                    await page.wait_for_timeout(2000)

                if not clicked_3d:
                    # Save evidence so we can iterate if the toggle has been
                    # renamed/relocated by mlb.com.
                    try:
                        ss = self.session_dir / 'no_3d_button.png'
                        await page.screenshot(path=str(ss))
                        self.log(f'No 3D toggle found. Screenshot: {ss}')
                        button_dump = await page.evaluate(
                            """() => Array.from(document.querySelectorAll('a, button'))
                                .map(b => ({
                                    tag: b.tagName,
                                    text: (b.textContent || '').trim().slice(0, 40),
                                    aria: b.getAttribute('aria-label'),
                                    cls: (b.className?.toString?.() || '').slice(0, 60),
                                }))
                                .filter(b => b.text || b.aria)
                                .slice(0, 80)"""
                        )
                        (self.session_dir / 'buttons_dump.json').write_text(
                            json.dumps(button_dump, indent=2))
                        self.log(f'Dumped {len(button_dump)} buttons to buttons_dump.json')
                    except Exception as e:
                        self.log(f'Diagnostic dump failed: {e}')

                # Wait for FieldVision
                self.log('Waiting for FieldVision 3D to load...')
                ready = False
                for attempt in range(45):  # up to 90 seconds
                    state = await page.evaluate(FIND_POSER_JS)
                    if state.get('ready'):
                        self.log(f'FieldVision ready — {state["playerCount"]} players')
                        ready = True
                        break
                    if attempt % 5 == 0:
                        self.log(f'Waiting... ({state.get("reason")})')
                    await page.wait_for_timeout(2000)

                if not ready:
                    self.log('ERROR: FieldVision did not load. Game may not be live yet.')
                    return False

                # Setup capture + save metadata
                setup = await page.evaluate(SETUP_CAPTURE_JS)
                metadata = {
                    'captureTime': datetime.now().isoformat(),
                    'game': await page.title(),
                    'gameUrl': self.game_url,
                    'gamePk': self.game_pk,
                    'connections': setup['connections'],
                    'jointIndices': setup['jointIndices'],
                    'jointNames': setup['jointNames'],
                    'players': setup['playerIds'],
                    'playerCount': setup['playerCount'],
                    'hasBat': setup['hasBat'],
                }
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                self.log(f'Metadata written. {setup["playerCount"]} players, bat: {setup["hasBat"]}')

                # Main capture loop
                self.log('Starting capture loop...')
                last_status_time = time.time()
                game_over_detected = False

                with open(self.frames_path, 'w') as frames_file:
                    while time.time() - start_time < self.max_seconds:
                        loop_start = time.time()

                        try:
                            frame = await page.evaluate(CAPTURE_FRAME_JS)
                            if frame:
                                # Detect stall (no movement = idle)
                                simple_hash = hash(json.dumps(
                                    [p.get('j', {}) for p in frame.get('p', [])],
                                    sort_keys=True
                                ))
                                if simple_hash == self.last_frame_hash:
                                    self.stall_count += 1
                                else:
                                    self.stall_count = 0
                                self.last_frame_hash = simple_hash

                                frames_file.write(json.dumps(frame) + '\n')
                                self.frames_written += 1

                                # Flush every 100 frames so data is on disk
                                if self.frames_written % 100 == 0:
                                    frames_file.flush()
                        except Exception as e:
                            self.log(f'Frame capture error: {e}')

                        # Status log every 60 seconds
                        if time.time() - last_status_time > 60:
                            elapsed = time.time() - start_time
                            self.log(
                                f'[{elapsed/60:.1f}m] {self.frames_written} frames | '
                                f'stall_count={self.stall_count}'
                            )
                            last_status_time = time.time()

                        # Check game state every ~5 minutes
                        if self.frames_written > 0 and self.frames_written % 5000 == 0:
                            try:
                                state = await page.evaluate(CHECK_GAME_STATE_JS)
                                abstract = state.get('abstract', '').lower()
                                if abstract in ('final', 'completed', 'completed early', 'game over'):
                                    self.log(f'Game ended: {state}. Stopping capture.')
                                    game_over_detected = True
                                    break
                            except Exception:
                                pass

                        # If we've stalled for ~2 minutes straight (no data changes), check if page is alive
                        if self.stall_count > 2400:  # ~2min at 20Hz
                            self.log('Long stall detected — re-checking FieldVision state...')
                            try:
                                state = await page.evaluate(FIND_POSER_JS)
                                if not state.get('ready'):
                                    self.log(f'FieldVision not ready: {state}. Reloading page.')
                                    await page.reload(wait_until='domcontentloaded', timeout=60000)
                                    await page.wait_for_timeout(5000)
                                    # Re-click 3D
                                    try:
                                        btn = page.locator('a:has-text("3D"), button:has-text("3D")')
                                        if await btn.count() > 0:
                                            await btn.first.click()
                                            await page.wait_for_timeout(5000)
                                    except Exception:
                                        pass
                                    # Re-setup
                                    for _ in range(30):
                                        s = await page.evaluate(FIND_POSER_JS)
                                        if s.get('ready'): break
                                        await page.wait_for_timeout(2000)
                                    await page.evaluate(SETUP_CAPTURE_JS)
                                    self.log('Recovered.')
                                self.stall_count = 0
                            except Exception as e:
                                self.log(f'Recovery failed: {e}')

                        # Pace the loop
                        elapsed_in_loop = time.time() - loop_start
                        remaining = self.capture_interval - elapsed_in_loop
                        if remaining > 0:
                            await asyncio.sleep(remaining)

                    frames_file.flush()

                elapsed = time.time() - start_time
                self.log(
                    f'Capture done. {self.frames_written} frames over {elapsed/60:.1f} minutes. '
                    f'Game over: {game_over_detected}'
                )

            finally:
                await browser.close()

        # Auto-convert to viewer format if capture yielded data
        if self.frames_written > 0:
            try:
                convert_session(self.session_dir)
            except Exception as e:
                self.log(f'Conversion failed: {e}')

        return self.frames_written > 0


def convert_session(session_dir: Path):
    """Merge metadata.json + frames.jsonl into a single viewer-compatible JSON."""
    metadata_path = session_dir / 'metadata.json'
    frames_path = session_dir / 'frames.jsonl'

    if not metadata_path.exists() or not frames_path.exists():
        raise FileNotFoundError(f'Missing files in {session_dir}')

    with open(metadata_path) as f:
        metadata = json.load(f)

    frames = []
    with open(frames_path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))

    output = {
        **metadata,
        'frameCount': len(frames),
        'frames': frames,
    }

    out_path = session_dir / 'viewer.json'
    with open(out_path, 'w') as f:
        json.dump(output, f)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'Converted {len(frames)} frames → {out_path} ({size_mb:.1f} MB)')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='MLB FieldVision whole-game capture')
    parser.add_argument('--game', type=int, help='Game PK number')
    parser.add_argument('--url', type=str, help='Full Gameday URL')
    parser.add_argument('--max-hours', type=float, default=4.5, help='Max capture duration in hours')
    parser.add_argument('--output', type=str, default='./fv_captures', help='Output root directory')
    parser.add_argument('--rate', type=float, default=20.0, help='Capture rate in Hz (default 20)')
    parser.add_argument('--headed', action='store_true', help='Visible browser (for debugging)')
    parser.add_argument('--convert', type=str, help='Convert an existing session dir to viewer JSON')

    args = parser.parse_args()

    if args.convert:
        convert_session(Path(args.convert))
        return

    if args.game:
        url = f'https://www.mlb.com/gameday/{args.game}/live'
    elif args.url:
        url = args.url
    else:
        parser.print_help()
        return

    cap = GameCapture(
        game_url=url,
        output_root=Path(args.output),
        max_hours=args.max_hours,
        headless=not args.headed,
        capture_rate_hz=args.rate,
    )

    # Graceful shutdown
    def handle_sigint(sig, frame):
        cap.log('SIGINT received — shutting down (frames already written to disk).')
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)

    asyncio.run(cap.run())


if __name__ == '__main__':
    main()
