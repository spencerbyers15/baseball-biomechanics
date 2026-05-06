"""
MLB FieldVision Headless Skeletal Capture
==========================================
Automatically captures Hawk-Eye skeletal tracking data from any MLB game
using a headless browser. No manual DevTools interaction needed.

Usage:
  # Capture 30 seconds from a specific game
  python fv_headless.py --game 824855 --duration 30

  # Capture from a URL directly  
  python fv_headless.py --url "https://www.mlb.com/gameday/red-sox-vs-twins/2026/04/15/823722/live" --duration 60

  # Capture all currently live games (10 seconds each)
  python fv_headless.py --all-live --duration 10

  # Output to specific directory
  python fv_headless.py --game 824855 --duration 30 --output ./captures/

Requirements:
  pip install playwright
  playwright install chromium
"""

import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Install playwright: pip install playwright && playwright install chromium")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# JavaScript to inject into the Gameday page
# ═══════════════════════════════════════════════════════════

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
                        found = true;
                        break;
                    }
                } catch(e) {}
                h = h.next;
            }
        }
        if (found) break;
        fiber = fiber.return;
    }
    
    if (!found) return { ready: false, reason: 'no_poser_context' };
    
    // Check if scene has loaded with players
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
    
    // Cache skeleton references for fast capture
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
    
    // Try to extract player IDs from world.actors.actors
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
    
    // Store for frame capture
    window.__fvCache = { skels, batMesh, SKEL_JOINTS, playerIds };
    
    return {
        playerCount: skels.length,
        hasBat: !!batMesh,
        playerIds: playerIds.slice(0, 20),
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
        
        // Bat
        let hasBat = false;
        if (cache.batMesh) {
            // Proximity match: HandRT close to bat handle
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


async def capture_game(game_url: str, duration: int, output_dir: Path, headless: bool = True):
    """Capture skeletal data from a single game."""
    
    print(f"\n{'='*60}")
    print(f"Capturing: {game_url}")
    print(f"Duration:  {duration}s")
    print(f"{'='*60}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        page = await context.new_page()
        
        # ── Navigate and wait for page load ──
        print("Loading page...")
        await page.goto(game_url, wait_until='networkidle', timeout=30000)
        await page.wait_for_timeout(3000)
        
        # ── Click 3D button if needed ──
        try:
            btn_3d = page.locator('a:has-text("3D"), button:has-text("3D")')
            if await btn_3d.count() > 0:
                print("Clicking 3D button...")
                await btn_3d.first.click()
                await page.wait_for_timeout(5000)
        except Exception as e:
            print(f"  3D button click: {e}")
        
        # ── Wait for FieldVision to load ──
        print("Waiting for FieldVision 3D to load...")
        max_retries = 30
        for attempt in range(max_retries):
            result = await page.evaluate(FIND_POSER_JS)
            if result['ready']:
                print(f"  FieldVision ready! {result['playerCount']} players detected.")
                break
            else:
                if attempt % 5 == 0:
                    print(f"  Waiting... ({result['reason']})")
                await page.wait_for_timeout(2000)
        else:
            print("ERROR: FieldVision failed to load after 60s. Game might not be live or 3D not available.")
            await browser.close()
            return None
        
        # ── Setup capture (cache skeletons, extract player IDs) ──
        print("Setting up capture...")
        setup = await page.evaluate(SETUP_CAPTURE_JS)
        print(f"  Players: {setup['playerCount']}")
        print(f"  Bat: {'yes' if setup['hasBat'] else 'no'}")
        print(f"  Player IDs: {len(setup['playerIds'])}")
        for pid in setup['playerIds'][:5]:
            print(f"    mlbId={pid.get('mlbId', '?')} type={pid.get('type', '?')}")
        
        # ── Capture frames ──
        frames = []
        target_frames = duration * 20  # ~20fps
        print(f"\nCapturing {duration}s ({target_frames} frames)...")
        
        start_time = time.time()
        frame_count = 0
        
        while frame_count < target_frames:
            frame = await page.evaluate(CAPTURE_FRAME_JS)
            if frame:
                frames.append(frame)
                frame_count += 1
                
                if frame_count % 20 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  {frame_count}/{target_frames} frames ({fps:.1f} fps)")
            
            await page.wait_for_timeout(50)  # ~20fps
        
        elapsed = time.time() - start_time
        actual_fps = len(frames) / elapsed if elapsed > 0 else 0
        print(f"\nCaptured {len(frames)} frames in {elapsed:.1f}s ({actual_fps:.1f} fps)")
        
        # ── Build output ──
        game_title = await page.title()
        
        output_data = {
            'captureTime': datetime.now().isoformat(),
            'game': game_title,
            'gameUrl': game_url,
            'connections': setup['connections'],
            'jointIndices': setup['jointIndices'],
            'jointNames': setup['jointNames'],
            'players': setup['playerIds'],
            'frameCount': len(frames),
            'actualFps': round(actual_fps, 1),
            'durationSeconds': round(elapsed, 1),
            'frames': frames
        }
        
        # ── Save to file ──
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract gamePk from URL for filename
        game_pk = game_url.split('/')[-2] if '/live' in game_url else 'unknown'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fv_{game_pk}_{timestamp}.json'
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f)
        
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"\nSaved: {filepath} ({size_mb:.1f} MB)")
        
        await browser.close()
        return filepath


async def find_live_games():
    """Find currently live MLB games via the schedule API."""
    import urllib.request
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}'
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"Failed to fetch schedule: {e}")
        return []
    
    live_games = []
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            status = game.get('status', {}).get('abstractGameState', '')
            if status == 'Live':
                game_pk = game['gamePk']
                away = game['teams']['away']['team']['name']
                home = game['teams']['home']['team']['name']
                game_url = f'https://www.mlb.com/gameday/{game_pk}/live'
                live_games.append({
                    'gamePk': game_pk,
                    'url': game_url,
                    'matchup': f'{away} @ {home}',
                    'status': game.get('status', {}).get('detailedState', '')
                })
    
    return live_games


async def capture_all_live(duration: int, output_dir: Path, headless: bool = True):
    """Capture from all currently live games."""
    games = await find_live_games()
    
    if not games:
        print("No live games found right now.")
        print("Check https://www.mlb.com/scores for game times.")
        return
    
    print(f"\nFound {len(games)} live games:")
    for g in games:
        print(f"  {g['matchup']} ({g['status']}) - {g['url']}")
    
    results = []
    for game in games:
        try:
            filepath = await capture_game(game['url'], duration, output_dir, headless)
            if filepath:
                results.append(filepath)
        except Exception as e:
            print(f"  ERROR capturing {game['matchup']}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Captured {len(results)}/{len(games)} games")
    for r in results:
        print(f"  {r}")


def main():
    parser = argparse.ArgumentParser(description='MLB FieldVision Headless Skeletal Capture')
    parser.add_argument('--game', type=int, help='Game PK number (e.g., 824855)')
    parser.add_argument('--url', type=str, help='Full Gameday URL')
    parser.add_argument('--all-live', action='store_true', help='Capture all currently live games')
    parser.add_argument('--duration', type=int, default=30, help='Capture duration in seconds (default: 30)')
    parser.add_argument('--output', type=str, default='./fv_captures', help='Output directory')
    parser.add_argument('--headed', action='store_true', help='Run with visible browser (for debugging)')
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    headless = not args.headed
    
    if args.all_live:
        asyncio.run(capture_all_live(args.duration, output_dir, headless))
    elif args.game:
        url = f'https://www.mlb.com/gameday/{args.game}/live'
        asyncio.run(capture_game(url, args.duration, output_dir, headless))
    elif args.url:
        asyncio.run(capture_game(args.url, args.duration, output_dir, headless))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python fv_headless.py --game 824855 --duration 30")
        print("  python fv_headless.py --all-live --duration 10")
        print("  python fv_headless.py --url 'https://www.mlb.com/gameday/...' --duration 60")


if __name__ == '__main__':
    main()
