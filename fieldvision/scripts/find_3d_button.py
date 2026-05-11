"""Find the FieldVision 3D toggle button.

The user's screenshot shows a small "(3D)" overlay icon in the top-right of
the central video/FieldVision pane. It's not a top-level <button> with text
"3D" — it's likely an SVG icon button with aria-label or a class hint.

This probe:
  1. Loads a live game with a desktop viewport (1920x1080)
  2. Dismisses any cookie banners
  3. Dumps every visible clickable element with bounding rect + attributes
  4. Highlights candidates in the upper-right of the FieldVision pane
  5. Saves a full-page screenshot
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright


PROBE_JS = """
() => {
    // Find the FieldVision wrapper to anchor coordinate searches
    const fv = document.querySelector('[class*="FieldVision" i]');
    const fvRect = fv ? fv.getBoundingClientRect() : null;

    const els = document.querySelectorAll(
        'button, a, [role="button"], [onclick], [tabindex]:not([tabindex="-1"])'
    );
    const all = [];
    for (const el of els) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        if (el.offsetParent === null && el.tagName !== 'BODY') continue;
        const cls = (el.className?.toString?.() || '');
        const aria = el.getAttribute('aria-label') || '';
        const title = el.getAttribute('title') || '';
        const text = (el.textContent || '').trim();
        const innerHTML = el.innerHTML || '';

        // Prioritise: anything mentioning "3d" anywhere
        const haystack = `${cls} ${aria} ${title} ${text} ${innerHTML}`.toLowerCase();
        const has3d = /3\\s*d|three\\s*d/i.test(haystack);

        all.push({
            tag: el.tagName,
            text: text.slice(0, 60),
            aria: aria.slice(0, 60),
            title: title.slice(0, 60),
            cls: cls.slice(0, 120),
            x: Math.round(r.x), y: Math.round(r.y),
            w: Math.round(r.width), h: Math.round(r.height),
            innerHTML: innerHTML.slice(0, 250),
            has3d,
        });
    }

    return {
        viewport: { w: window.innerWidth, h: window.innerHeight },
        fvRect: fvRect ? {
            x: Math.round(fvRect.x), y: Math.round(fvRect.y),
            w: Math.round(fvRect.width), h: Math.round(fvRect.height),
            cls: (fv.className?.toString?.() || '').slice(0, 120),
        } : null,
        clickables: all,
    };
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
        await page.wait_for_timeout(8000)

        # Cookie banner
        for lbl in ("Accept", "Accept All", "Accept all", "I Accept", "OK", "Got it", "Agree"):
            try:
                cb = page.locator(f'button:has-text("{lbl}")')
                if await cb.count() > 0:
                    await cb.first.click(timeout=2000)
                    print(f"accepted cookies: {lbl!r}")
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                pass

        await page.wait_for_timeout(3000)

        ss = out_dir / "find_3d_full.png"
        await page.screenshot(path=str(ss), full_page=False)
        print(f"screenshot: {ss}")

        data = await page.evaluate(PROBE_JS)
        out_path = out_dir / "find_3d.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"data:       {out_path}")
        print()
        print(f"viewport:   {data['viewport']}")
        print(f"fvRect:     {data['fvRect']}")
        print(f"clickables: {len(data['clickables'])}")
        print()

        # Filter for "3d"-mentioning candidates
        hot = [c for c in data["clickables"] if c["has3d"]]
        print(f"=== candidates mentioning '3d' ({len(hot)}) ===")
        for c in hot[:30]:
            print(f"  pos=({c['x']:4},{c['y']:4}) size=({c['w']:3}x{c['h']:3}) "
                  f"tag={c['tag']:6} text={c['text']!r:30} aria={c['aria']!r:30} cls={c['cls'][:60]!r}")
            if c['innerHTML']:
                print(f"    innerHTML: {c['innerHTML'][:160]}")

        # Also: anything in top-right of FV pane
        fr = data["fvRect"]
        if fr:
            x_min = fr["x"] + fr["w"] * 0.6   # right 40% of FV pane
            y_min = fr["y"]
            y_max = fr["y"] + fr["h"] * 0.4   # top 40% of FV pane
            in_corner = [
                c for c in data["clickables"]
                if c["x"] >= x_min and y_min <= c["y"] <= y_max
                and c["w"] < 100 and c["h"] < 100  # small icon-sized
            ]
            print()
            print(f"=== small clickables in top-right of FV pane ({len(in_corner)}) ===")
            for c in in_corner[:20]:
                print(f"  pos=({c['x']:4},{c['y']:4}) size=({c['w']:3}x{c['h']:3}) "
                      f"tag={c['tag']:6} text={c['text']!r:20} aria={c['aria']!r:30} cls={c['cls'][:60]!r}")

        await browser.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Full Gameday URL")
    parser.add_argument("--game", type=int, default=823141, help="Default = ATL @ SEA from screenshot")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"))
    args = parser.parse_args()
    url = args.url or f"https://www.mlb.com/gameday/{args.game}/live"
    asyncio.run(probe(url, not args.headed, Path(args.out)))


if __name__ == "__main__":
    main()
