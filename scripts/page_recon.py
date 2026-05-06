"""Quick reconnaissance — load the gameday page, take a screenshot, dump
the top-level DOM structure so we can see why FieldVision isn't loading."""

import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright


async def recon(url: str, headless: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)

        title = await page.title()
        cur_url = page.url
        print(f"title:      {title}")
        print(f"current url: {cur_url}")

        ss_path = out_dir / "recon_initial.png"
        await page.screenshot(path=str(ss_path), full_page=False)
        print(f"screenshot: {ss_path}")

        info = await page.evaluate(
            """() => {
                const fvSelectors = [
                    '[class*="FieldVision"]',
                    '[class*="fieldvision"]',
                    '[class*="3D"]',
                    '[class*="gameday3d"]',
                    '[data-testid*="3d"]',
                    '[data-testid*="3D"]',
                ];
                const found = {};
                for (const sel of fvSelectors) {
                    const els = document.querySelectorAll(sel);
                    found[sel] = Array.from(els).slice(0, 3).map(e => ({
                        tag: e.tagName,
                        cls: e.className?.toString?.().slice(0, 120) || '',
                        text: (e.textContent || '').trim().slice(0, 80),
                    }));
                }
                const allButtons = Array.from(document.querySelectorAll('a, button'))
                    .map(b => (b.textContent || '').trim())
                    .filter(t => t && t.length < 40);
                const buttonsLooking3D = allButtons.filter(t => /3d|view|gameday/i.test(t));
                const url = location.href;
                const hasGameVid = !!document.querySelector('video');
                return {
                    found,
                    buttonsLooking3D: buttonsLooking3D.slice(0, 30),
                    totalButtons: allButtons.length,
                    url,
                    hasVideo: hasGameVid,
                    bodyClass: document.body.className?.toString?.() || '',
                    bodyChildCount: document.body.children.length,
                    documentReady: document.readyState,
                };
            }"""
        )

        import json
        recon_path = out_dir / "recon.json"
        recon_path.write_text(json.dumps(info, indent=2))
        print(f"recon json: {recon_path}")
        print()
        print("buttons containing '3d|view|gameday':")
        for b in info["buttonsLooking3D"]:
            print(f"  {b!r}")
        print()
        print("FieldVision-class probes:")
        for sel, results in info["found"].items():
            print(f"  {sel}: {len(results)} matches")
            for r in results:
                print(f"    {r['tag']} cls={r['cls'][:80]!r} text={r['text']!r}")

        await browser.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.mlb.com/gameday/824198/live")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"))
    args = parser.parse_args()
    asyncio.run(recon(args.url, not args.headed, Path(args.out)))


if __name__ == "__main__":
    main()
