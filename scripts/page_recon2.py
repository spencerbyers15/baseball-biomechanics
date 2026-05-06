"""Deeper recon — accept the cookies banner, click Retry if present,
walk every Gameday tab and report which (if any) reveals FieldVision.

Logs the network traffic to spot the FieldVision endpoint
(fieldvision-hls.mlbinfra.com) loading or 401-ing."""

import asyncio
import json
import sys
from pathlib import Path
from playwright.async_api import async_playwright


async def recon(url: str, headless: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fv_requests: list[dict] = []

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

        # Capture any FieldVision-bound traffic
        def on_request(req):
            if "fieldvision" in req.url or "mannequin" in req.url:
                fv_requests.append({"phase": "request", "url": req.url, "method": req.method})

        async def on_response(resp):
            if "fieldvision" in resp.url or "mannequin" in resp.url:
                fv_requests.append({"phase": "response", "url": resp.url, "status": resp.status})

        page.on("request", on_request)
        page.on("response", lambda r: asyncio.create_task(on_response(r)))

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(8000)

        # Accept any cookies banner
        for txt in ["Accept", "Accept all", "I Accept", "OK", "Got it"]:
            try:
                btn = page.locator(f'button:has-text("{txt}")')
                if await btn.count() > 0:
                    await btn.first.click(timeout=2000)
                    print(f"accepted banner: {txt!r}")
                    await page.wait_for_timeout(1500)
                    break
            except Exception:
                continue

        ss_after_cookies = out_dir / "recon2_after_cookies.png"
        await page.screenshot(path=str(ss_after_cookies))
        print(f"screenshot (post-cookies): {ss_after_cookies}")

        # Click "Retry" if present
        try:
            retry = page.locator('button:has-text("Retry")')
            if await retry.count() > 0:
                print(f"retry buttons found: {await retry.count()}")
                await retry.first.click(timeout=2000)
                await page.wait_for_timeout(5000)
                ss_after_retry = out_dir / "recon2_after_retry.png"
                await page.screenshot(path=str(ss_after_retry))
                print(f"screenshot (post-retry): {ss_after_retry}")
        except Exception as e:
            print(f"retry click failed: {e}")

        # Enumerate all tabs/buttons by text
        tabs_info = await page.evaluate(
            """() => {
                const out = {};
                const elements = document.querySelectorAll(
                    'a[role="tab"], button[role="tab"], [class*="tab"], [class*="Tab"], nav a, nav button'
                );
                out.tabs = Array.from(elements).slice(0, 60).map(e => ({
                    tag: e.tagName,
                    role: e.getAttribute('role'),
                    cls: (e.className?.toString?.() || '').slice(0, 80),
                    text: (e.textContent || '').trim().slice(0, 40),
                    href: e.getAttribute?.('href') || null,
                    visible: e.offsetParent !== null,
                }));
                // anything with the literal "3D" anywhere on screen
                const all = document.querySelectorAll('*');
                const has3D = [];
                for (const el of all) {
                    const tx = el.textContent;
                    if (tx && tx.length < 20 && /^\\s*3D\\s*$/i.test(tx)) {
                        has3D.push({
                            tag: el.tagName,
                            cls: (el.className?.toString?.() || '').slice(0, 80),
                            visible: el.offsetParent !== null,
                        });
                    }
                }
                out.has3DLiteral = has3D.slice(0, 20);
                return out;
            }"""
        )

        print()
        print(f"tabs found: {len(tabs_info['tabs'])}")
        for t in tabs_info["tabs"]:
            if t["text"]:
                print(f"  {t['tag']} role={t['role']!r} text={t['text']!r} visible={t['visible']} href={t['href']}")
        print()
        print(f"elements with text == '3D': {len(tabs_info['has3DLiteral'])}")
        for h in tabs_info["has3DLiteral"]:
            print(f"  {h}")

        # Try clicking each candidate tab to see if any reveals FV
        # First, look for "Insights" since that's where pitch tracking lives
        for tab_text in ("Insights", "Field View", "Field", "View", "3D"):
            try:
                btn = page.locator(f'a:has-text("{tab_text}"), button:has-text("{tab_text}")')
                if await btn.count() > 0:
                    print(f"\nclicking tab: {tab_text!r}")
                    await btn.first.click(timeout=2000)
                    await page.wait_for_timeout(4000)
                    ss = out_dir / f"recon2_after_{tab_text.replace(' ', '_').lower()}.png"
                    await page.screenshot(path=str(ss))
                    print(f"  screenshot: {ss}")
                    fv_check = await page.evaluate(
                        """() => {
                            const sels = [
                                '[class*="FieldVisionPlayerContainer"]',
                                '[class*="FieldVisionApp"]',
                                '[class*="FieldVision"]',
                            ];
                            for (const s of sels) {
                                const el = document.querySelector(s);
                                if (el) return { sel: s, cls: (el.className?.toString?.() || '').slice(0, 100) };
                            }
                            return null;
                        }"""
                    )
                    if fv_check:
                        print(f"  FOUND FieldVision: {fv_check}")
                        break
                    else:
                        print(f"  no FieldVision element after click")
            except Exception as e:
                print(f"  click {tab_text!r} failed: {e}")

        # Final screenshot
        ss_final = out_dir / "recon2_final.png"
        await page.screenshot(path=str(ss_final))

        print()
        print(f"FieldVision/mannequin network requests: {len(fv_requests)}")
        for r in fv_requests[:10]:
            print(f"  {r}")

        (out_dir / "recon2.json").write_text(json.dumps({
            "tabs": tabs_info,
            "fvRequests": fv_requests,
        }, indent=2))

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
