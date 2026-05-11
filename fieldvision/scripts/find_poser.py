"""Locate the FieldVision/DynamicFieldView wrapper that holds poserContext.

The 3D scene IS rendering in Playwright (we have a screenshot showing it),
but the existing FIND_POSER_JS selector returns no_fv_container because mlb.com
renamed the wrapper from FieldVision* to DynamicFieldView*. This finds the
actual element AND walks its React fiber to confirm poserContext is reachable.
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright


PROBE_JS = """
() => {
    const result = { candidateClasses: [], poserSearch: [] };

    // Find every element class containing "field" or "vision" or "dynamic" + "view"
    const allEls = document.querySelectorAll('[class]');
    const seenClasses = new Set();
    for (const el of allEls) {
        const cls = el.className?.toString?.() || '';
        if (!cls) continue;
        if (/(fieldvision|fieldview|dynamicfield|poser|hawk|skel)/i.test(cls)) {
            for (const c of cls.split(/\\s+/)) {
                if (c.length > 3 && !seenClasses.has(c)) {
                    seenClasses.add(c);
                    result.candidateClasses.push(c);
                }
            }
        }
    }

    // Find the canvas (where WebGL renders)
    const canvases = Array.from(document.querySelectorAll('canvas'));
    result.canvasCount = canvases.length;
    result.canvases = canvases.map(c => ({
        w: c.width, h: c.height,
        clientW: c.clientWidth, clientH: c.clientHeight,
        cls: (c.className?.toString?.() || '').slice(0, 100),
        ctx: c.getContext('webgl2') ? 'webgl2' : (c.getContext('webgl') ? 'webgl' : null),
        // Walk up parents to find one with a className
        parents: (() => {
            const parents = [];
            let p = c.parentElement;
            for (let i = 0; i < 15 && p; i++) {
                parents.push({
                    tag: p.tagName,
                    cls: (p.className?.toString?.() || '').slice(0, 100),
                });
                p = p.parentElement;
            }
            return parents;
        })(),
    }));

    // Build candidate roots: by class name selectors AND each canvas's ancestors
    const candidateRoots = [];
    for (const cls of ['DynamicFieldView', 'DynamicFieldVision', 'FieldVisionPlayerContainer',
                       'FieldVisionApp', 'FieldVision']) {
        const el = document.querySelector(`[class*="${cls}" i]`);
        if (el) candidateRoots.push({ selector: `[class*="${cls}" i]`, cls: el.className?.toString?.() || '', el });
    }
    for (const cnv of canvases) {
        let p = cnv.parentElement;
        for (let i = 0; i < 15 && p; i++) {
            if (p.className) {
                candidateRoots.push({ selector: `canvas-parent[${i}]`, cls: p.className?.toString?.() || '', el: p });
            }
            p = p.parentElement;
        }
    }

    for (const { selector, cls, el } of candidateRoots) {
        if (!el) continue;
        const fkKey = Object.keys(el).find(k => k.startsWith('__reactFiber'));
        const probe = { selector, cls: cls.slice(0, 120), hasFiber: !!fkKey };
        if (fkKey) {
            let fiber = el[fkKey];
            let foundPoser = false;
            for (let depth = 0; depth < 30 && fiber && !foundPoser; depth++) {
                if (fiber.memoizedState) {
                    let h = fiber.memoizedState;
                    for (let i = 0; i < 30 && h && !foundPoser; i++) {
                        try {
                            if (h.memoizedState?.current?.poserContext) {
                                window.__pc = h.memoizedState.current.poserContext;
                                foundPoser = true;
                                probe.poserDepth = depth;
                                probe.poserHookIdx = i;
                            }
                        } catch (e) {}
                        h = h.next;
                    }
                }
                if (!foundPoser) fiber = fiber.return;
            }
            probe.foundPoser = foundPoser;
        }
        result.poserSearch.push(probe);
        if (probe.foundPoser) break;  // stop at first success
    }

    // If we set __pc, dump some info about it
    if (window.__pc) {
        const pc = window.__pc;
        result.poserContextKeys = Object.keys(pc).slice(0, 30);
        try {
            const w = pc.world;
            result.world = {
                exists: !!w,
                type: w?.constructor?.name,
                childCount: w?.children?.length,
                hasActors: !!w?.actors,
                actorsKeys: w?.actors ? Object.keys(w.actors).slice(0, 20) : null,
            };
            // Count 103-bone skeletons
            let count = 0;
            const armSet = new Set();
            w?.traverse?.(obj => {
                if (obj.skeleton?.bones?.length === 103 && !armSet.has(obj.parent)) {
                    armSet.add(obj.parent);
                    count++;
                }
            });
            result.skeleton103Count = count;
        } catch (e) {
            result.worldError = String(e);
        }
    }

    return result;
}
"""


async def probe(url: str, headless: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(10000)  # wait for SPA + WebGL to come up

        result = await page.evaluate(PROBE_JS)
        out_path = out_dir / "find_poser.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"out: {out_path}")
        print()
        print(f"candidate FieldVision-y class names: {result['candidateClasses']}")
        print(f"canvas count:                       {result['canvasCount']}")
        for i, c in enumerate(result.get('canvases', [])):
            print(f"  canvas[{i}]  {c['w']}x{c['h']} ({c['ctx']})  cls={c['cls']!r}")
        print()
        print("poser search:")
        for p in result.get('poserSearch', []):
            print(f"  selector={p['selector']!r}")
            print(f"    cls={p['cls']!r}")
            print(f"    hasFiber={p.get('hasFiber')}  foundPoser={p.get('foundPoser')}  depth={p.get('poserDepth')}")
        print()
        if result.get('world'):
            w = result['world']
            print(f"world: exists={w['exists']}  type={w.get('type')}  childCount={w.get('childCount')}")
            print(f"       hasActors={w.get('hasActors')}  actorsKeys={w.get('actorsKeys')}")
            print(f"103-bone skeletons: {result.get('skeleton103Count')}")
        else:
            print("WORLD NOT FOUND (poserContext lookup failed)")
        await browser.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"))
    args = parser.parse_args()
    url = args.url or f"https://www.mlb.com/gameday/{args.game}/live"
    asyncio.run(probe(url, not args.headed, Path(args.out)))


if __name__ == "__main__":
    main()
