"""Hunt for JWTs anywhere in a Gameday session — cookies, localStorage,
sessionStorage, request headers, request bodies, response bodies.

A JWT looks like  eyJ...eyJ...sig  (base64url-encoded JSON x2 + signature).
Anything matching that pattern is harvested, deduped, then tested against
fieldvision-hls.mlbinfra.com to see which (if any) the legacy endpoint
accepts.
"""

import asyncio
import base64
import json
import re
import urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

JWT_RX = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")


def find_jwts(blob: str) -> list[str]:
    if not blob:
        return []
    return list(set(JWT_RX.findall(blob)))


def decode_jwt_claims(jwt: str) -> dict | None:
    try:
        parts = jwt.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        # base64url decode, padding-tolerant
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return None


async def hunt(url: str, headless: bool, out_dir: Path, duration_s: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    found_jwts: set[str] = set()
    sources: dict[str, list[str]] = {}

    def add_jwts(source: str, blob: str):
        for j in find_jwts(blob):
            found_jwts.add(j)
            sources.setdefault(j, []).append(source)

    request_log = []
    response_log = []

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

        async def on_request(req):
            try:
                headers = await req.all_headers()
                hdr_blob = "\n".join(f"{k}: {v}" for k, v in headers.items())
                add_jwts(f"req-header  {req.method} {req.url[:80]}", hdr_blob)
                body = req.post_data or ""
                if body:
                    add_jwts(f"req-body    {req.method} {req.url[:80]}", body)
                request_log.append({"url": req.url, "method": req.method, "rt": req.resource_type})
            except Exception:
                pass

        async def on_response(resp):
            try:
                # Only capture text-ish responses, not binary
                ct = (resp.headers.get("content-type") or "").lower()
                if any(t in ct for t in ("json", "text", "javascript", "html", "xml")):
                    body = await resp.text()
                    add_jwts(f"resp-body   {resp.status} {resp.url[:80]}", body)
                add_jwts(f"resp-header {resp.status} {resp.url[:80]}",
                         "\n".join(f"{k}: {v}" for k, v in resp.headers.items()))
                response_log.append({"url": resp.url, "status": resp.status, "ct": ct[:60]})
            except Exception:
                pass

        page.on("request", lambda r: asyncio.create_task(on_request(r)))
        page.on("response", lambda r: asyncio.create_task(on_response(r)))

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(duration_s * 1000)

        # Sweep cookies, localStorage, sessionStorage
        cookies = await ctx.cookies()
        cookie_blob = "\n".join(f"{c['name']}={c['value']}" for c in cookies)
        add_jwts("cookie", cookie_blob)

        try:
            ls = await page.evaluate(
                """() => {
                    const ls = {}, ss = {};
                    try { for (let i = 0; i < localStorage.length; i++) {
                        const k = localStorage.key(i); ls[k] = localStorage.getItem(k);
                    } } catch(e) {}
                    try { for (let i = 0; i < sessionStorage.length; i++) {
                        const k = sessionStorage.key(i); ss[k] = sessionStorage.getItem(k);
                    } } catch(e) {}
                    return { ls, ss };
                }"""
            )
            for k, v in (ls.get("ls") or {}).items():
                add_jwts(f"localStorage[{k}]", v or "")
            for k, v in (ls.get("ss") or {}).items():
                add_jwts(f"sessionStorage[{k}]", v or "")
        except Exception as e:
            print(f"storage scrape failed: {e}")

        await browser.close()

    # Decode every JWT and try it against fieldvision-hls
    test_url = "https://fieldvision-hls.mlbinfra.com/mannequin/823141/1.6.2/manifest.json"
    results = []
    for j in sorted(found_jwts):
        claims = decode_jwt_claims(j) or {}
        # Test against fieldvision-hls
        try:
            req = urllib.request.Request(
                test_url,
                headers={
                    "Authorization": f"Bearer {j}",
                    "x-mannequin-client": "gameday",
                    "Origin": "https://www.mlb.com",
                    "Referer": "https://www.mlb.com/",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
                body = resp.read()[:500].decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                body = e.read()[:500].decode("utf-8", errors="replace")
            except Exception:
                body = ""
        except Exception as e:
            status = -1
            body = f"<{e}>"

        results.append({
            "jwtPrefix": j[:50],
            "jwtSuffix": "..." + j[-30:],
            "claims": {k: v for k, v in claims.items() if k in (
                "iss", "aud", "sub", "scope", "scopes", "userId", "mlbuserId", "mlbUserId",
                "uid", "exp", "iat", "kid", "typ", "client_id", "azp", "permissions"
            )},
            "claimKeys": list(claims.keys()),
            "sources": sources.get(j, [])[:3],
            "fieldvisionHls": {"status": status, "body": body[:200]},
        })

    out = {
        "url": url,
        "totalRequests": len(request_log),
        "totalResponses": len(response_log),
        "jwtCount": len(found_jwts),
        "tested": results,
    }
    out_path = out_dir / "jwt_hunt.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"out:    {out_path}")
    print(f"jwts:   {len(found_jwts)} unique")
    print()
    success = [r for r in results if r["fieldvisionHls"]["status"] == 200]
    if success:
        print("=== JWT(s) THAT GOT 200 ON fieldvision-hls ===")
        for r in success:
            print(f"  {r['jwtPrefix']}{r['jwtSuffix']}")
            print(f"    claims: {r['claims']}")
            print(f"    sources: {r['sources']}")
    else:
        print("=== no JWT got 200; status distribution: ===")
        for r in results:
            print(f"  status={r['fieldvisionHls']['status']:4}  prefix={r['jwtPrefix'][:40]}  "
                  f"sub={r['claims'].get('sub','-')[:30]:30}  scope={(r['claims'].get('scope') or r['claims'].get('scopes') or '-')[:50]}")
            print(f"    keys: {r['claimKeys']}")
            print(f"    sources: {r['sources']}")
            print(f"    body: {r['fieldvisionHls']['body'][:120]}")
            print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://www.mlb.com/gameday/braves-vs-mariners/2026/05/06/823141/live")
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--duration", type=int, default=25)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"))
    args = parser.parse_args()
    asyncio.run(hunt(args.url, not args.headed, Path(args.out), args.duration))


if __name__ == "__main__":
    main()
