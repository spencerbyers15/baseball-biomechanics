"""Sniff every outgoing request from a Gameday session and dump:
  - all unique Authorization header values
  - all unique x-mannequin-client (or similar identity) header values
  - all unique hosts contacted
  - any URL that looks like a token-issuance / login / identity endpoint

Output:
  diagnostics/raw/auth_sniff_{gamePk}.json
"""

import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from playwright.async_api import async_playwright


INTERESTING_HEADER_PATTERNS = re.compile(
    r"^(authorization|x-mannequin-client|x-bamsdk|x-api-key|cookie|x-user-token"
    r"|x-mlb-|x-tap-|x-stitch-|x-fetch-|api-key|bearer)",
    re.I,
)
TOKEN_ENDPOINT_PATTERNS = re.compile(
    r"(token|login|identity|auth|oauth|jwt|session|/v\d/keys|svc-account|bamgrid)",
    re.I,
)


async def sniff(url: str, headless: bool, out_dir: Path, duration_s: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_log: list[dict] = []
    auth_values: dict[str, set] = defaultdict(set)

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
                interesting = {h: v for h, v in headers.items() if INTERESTING_HEADER_PATTERNS.match(h)}
                row = {
                    "url": req.url,
                    "method": req.method,
                    "rt": req.resource_type,
                    "headers": interesting,
                }
                requests_log.append(row)
                for h, v in interesting.items():
                    auth_values[h].add(v[:300])
            except Exception:
                pass

        page.on("request", lambda r: asyncio.create_task(on_request(r)))

        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(duration_s * 1000)

        await browser.close()

    # Distill findings
    hosts = sorted({r["url"].split("/")[2] for r in requests_log if r["url"].startswith("http")})
    auth_endpoints = sorted({
        r["url"].split("?")[0]
        for r in requests_log
        if TOKEN_ENDPOINT_PATTERNS.search(r["url"])
    })
    bearer_tokens = sorted(auth_values.get("authorization", set()))

    out = {
        "url": url,
        "totalRequests": len(requests_log),
        "uniqueHosts": hosts,
        "tokenIssuanceCandidateUrls": auth_endpoints,
        "uniqueAuthHeaderValues": {h: sorted(list(v))[:5] for h, v in auth_values.items()},
        "bearerTokensCount": len(bearer_tokens),
        "bearerTokensSample": [t[:80] + "..." for t in bearer_tokens[:5]],
        "fieldvisionRequests": [r for r in requests_log if "fieldvision" in r["url"] or "mannequin" in r["url"]],
    }

    out_path = out_dir / "auth_sniff.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Also save the bearer tokens raw, separately, for testing
    if bearer_tokens:
        (out_dir / "bearer_tokens.txt").write_text("\n".join(bearer_tokens))

    print(f"out:        {out_path}")
    print(f"hosts:      {len(hosts)}")
    print(f"requests:   {len(requests_log)}")
    print()
    print(f"=== bearer tokens ({len(bearer_tokens)}) ===")
    for t in bearer_tokens[:8]:
        print(f"  {t[:120]}...")
    print()
    print(f"=== unique auth-related header names ===")
    for h, vals in auth_values.items():
        print(f"  {h:30}  -> {len(vals)} unique value(s)")
        for v in sorted(vals)[:2]:
            print(f"    {v[:120]}")
    print()
    print(f"=== token-issuance candidate URLs ({len(auth_endpoints)}) ===")
    for u in auth_endpoints[:30]:
        print(f"  {u}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--game", type=int, default=823141)
    parser.add_argument("--duration", type=int, default=20, help="Seconds to keep sniffing after page load")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "diagnostics" / "raw"))
    args = parser.parse_args()
    url = args.url or f"https://www.mlb.com/gameday/{args.game}/live"
    asyncio.run(sniff(url, not args.headed, Path(args.out), args.duration))


if __name__ == "__main__":
    main()
