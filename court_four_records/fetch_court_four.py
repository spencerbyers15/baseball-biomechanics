#!/usr/bin/env python3
"""
fetch_court_four.py — Retrieve the Master Deed + recorded Condominium Plans for
"Court Four Condominium", 405 Washington St, Brookline, MA, from the Norfolk
County (MA) Registry of Deeds ALIS system (norfolkresearch.org).

WHY THIS SCRIPT EXISTS
----------------------
The Claude Code *web* environment blocks outbound traffic to norfolkdeeds.org /
norfolkresearch.org at its egress proxy (HTTP 403, x-deny-reason: host_not_allowed),
so the retrieval cannot run there. Run this on a machine with normal internet
(Spencer's Mac, or Nellie) and it will drive a real browser through the ALIS UI.

USAGE
-----
    pip install playwright
    python -m playwright install chromium
    python fetch_court_four.py                 # headless, auto
    python fetch_court_four.py --headed         # watch it run / help with hiccups
    python fetch_court_four.py --headed --slow 300

It is deliberately DEFENSIVE: the ALIS DOM could not be inspected from the blocked
environment, so every step uses text-based locators with fallbacks, screenshots at
each step (into ./court_four_records/screenshots/), and verbose logging. If a
selector misses, run with --headed, read the screenshot/log, and adjust the
SELECTOR CANDIDATES near the top.

OUTPUT
------
- ./court_four_records/screenshots/NN_step.png   (one per step)
- ./court_four_records/*.pdf                       (downloaded documents)
- ./court_four_records/summary.json                (book/page, dates, what was found)
- prints a human-readable summary at the end
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.exit("Playwright not installed. Run:\n"
             "  pip install playwright\n"
             "  python -m playwright install chromium")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ENTRY_URL = "https://www.norfolkdeeds.org/research/begin-online-research/"
ALIS_URL = "https://www.norfolkresearch.org/"  # the search app the entry page launches

OUT_DIR = Path(__file__).resolve().parent          # ./court_four_records/
SHOT_DIR = OUT_DIR / "screenshots"

SEARCH_NAME = "Court Four Condominium"
SEARCH_ADDR_NUM = "405"
SEARCH_ADDR_STREET = "Washington"
TOWN = "Brookline"

# Text used to find buttons/links regardless of exact markup. First match wins.
DISCLAIMER_ACCEPT = ["I Agree", "Agree", "Accept", "I Accept", "Continue", "Proceed"]
FREE_ACCESS = ["Free Access", "Free", "Guest", "Public Access", "Continue as Guest"]
RECORDED_LAND = ["Recorded Land", "Recorded"]
REGISTERED_LAND = ["Registered Land", "Land Court", "Registered"]
NAME_SEARCH = ["Name Search", "Name", "Grantor/Grantee", "Party Name"]
PLANS_SEARCH = ["Plans", "Plan Search", "Plan Index"]

_step = 0


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def shot(page, label: str) -> None:
    global _step
    _step += 1
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    p = SHOT_DIR / f"{_step:02d}_{re.sub(r'[^a-zA-Z0-9]+', '_', label).strip('_')}.png"
    try:
        page.screenshot(path=str(p), full_page=True)
        log(f"screenshot -> {p.name}")
    except Exception as e:  # pragma: no cover
        log(f"screenshot failed ({label}): {e}")


def click_first(page, texts, timeout=8000) -> bool:
    """Try clicking the first matching button/link/input by visible text/value."""
    for t in texts:
        for loc in (
            page.get_by_role("button", name=re.compile(re.escape(t), re.I)),
            page.get_by_role("link", name=re.compile(re.escape(t), re.I)),
            page.locator(f"input[type=submit][value*='{t}' i]"),
            page.locator(f"input[type=button][value*='{t}' i]"),
            page.get_by_text(re.compile(re.escape(t), re.I), exact=False),
        ):
            try:
                if loc.count() > 0:
                    loc.first.click(timeout=timeout)
                    log(f"clicked: {t!r}")
                    page.wait_for_load_state("networkidle", timeout=15000)
                    return True
            except PWTimeout:
                continue
            except Exception:
                continue
    log(f"NO clickable match for any of: {texts}")
    return False


def fill_first(page, label_texts, value) -> bool:
    """Fill the first input that matches a label/placeholder/name heuristic."""
    for t in label_texts:
        for loc in (
            page.get_by_label(re.compile(re.escape(t), re.I)),
            page.get_by_placeholder(re.compile(re.escape(t), re.I)),
            page.locator(f"input[name*='{t}' i]"),
            page.locator(f"input[id*='{t}' i]"),
        ):
            try:
                if loc.count() > 0:
                    loc.first.fill(str(value), timeout=5000)
                    log(f"filled {t!r} = {value!r}")
                    return True
            except Exception:
                continue
    log(f"NO input match for any of: {label_texts}")
    return False


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
def accept_disclaimer_and_enter(page) -> None:
    log(f"GET {ENTRY_URL}")
    page.goto(ENTRY_URL, wait_until="networkidle", timeout=45000)
    shot(page, "entry_page")

    # The entry page typically links out to the ALIS app; if a launch link exists,
    # click it, otherwise go straight to the app host.
    if not click_first(page, ["Begin", "Launch", "Online Research", "Search Records",
                              "ALIS", "Start"]):
        log(f"GET {ALIS_URL} (direct)")
        page.goto(ALIS_URL, wait_until="networkidle", timeout=45000)
    shot(page, "alis_landing")

    # Disclaimer / agreement, then Free Access.
    click_first(page, DISCLAIMER_ACCEPT)
    shot(page, "after_disclaimer")
    click_first(page, FREE_ACCESS)
    shot(page, "after_free_access")


def name_search(page, name: str) -> bool:
    log(f"--- Name search: {name!r} ---")
    click_first(page, RECORDED_LAND)
    shot(page, "recorded_land_menu")
    click_first(page, NAME_SEARCH)
    shot(page, "name_search_form")

    # Condo associations are usually indexed as a business/last-name. Try both.
    filled = (fill_first(page, ["Last Name", "Business", "Name", "Grantor", "Grantee", "Party"], name)
              or fill_first(page, ["Last", "Name"], name))
    shot(page, "name_filled")
    click_first(page, ["Search", "Submit", "Find", "Go"])
    page.wait_for_load_state("networkidle", timeout=20000)
    shot(page, "name_results")
    return filled


def parse_book_page(text: str):
    """Pull a 'Book NNNN Page NNN' citation out of arbitrary text."""
    m = re.search(r"book\s*[:#]?\s*(\d{1,6}).{0,12}?page\s*[:#]?\s*(\d{1,5})", text, re.I)
    if m:
        return m.group(1), m.group(2)
    return None, None


def harvest_results(page, want: str):
    """Read the results grid; return list of dicts with any book/page/desc found."""
    rows = []
    try:
        body = page.inner_text("body")
    except Exception:
        body = ""
    bk, pg = parse_book_page(body)
    if bk:
        rows.append({"book": bk, "page": pg, "context": want})
    # Also collect clickable result links for the operator to inspect via screenshots.
    return rows, body


def plans_search(page) -> None:
    log("--- Plans search ---")
    if click_first(page, PLANS_SEARCH):
        shot(page, "plans_form")
        fill_first(page, ["Name", "Subdivision", "Description", "Owner"], SEARCH_NAME)
        click_first(page, ["Search", "Submit", "Find", "Go"])
        page.wait_for_load_state("networkidle", timeout=20000)
        shot(page, "plans_results")


def try_download_current(page, stem: str):
    """Attempt to download the currently-open document/plan as a PDF."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in ["Download", "Save", "PDF", "Print", "Export"]:
        try:
            with page.expect_download(timeout=8000) as dl:
                if not click_first(page, [t], timeout=4000):
                    raise PWTimeout("no button")
            d = dl.value
            target = OUT_DIR / f"{stem}.pdf"
            d.save_as(str(target))
            log(f"downloaded -> {target.name}")
            return str(target)
        except Exception:
            continue
    log(f"could not auto-download {stem}; check screenshots and download manually")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true", help="show the browser window")
    ap.add_argument("--slow", type=int, default=0, help="slow_mo ms between actions")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "target": "Court Four Condominium, 405 Washington St, Brookline, MA",
        "registry": "Norfolk County (MA) Registry of Deeds (ALIS / norfolkresearch.org)",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "master_deed": {"book": None, "page": None, "recording_date": None, "pdf": None},
        "plans": [],
        "unit_deed_fallback": None,
        "downloads": [],
        "notes": [],
    }

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed, slow_mo=args.slow)
        ctx = browser.new_context(
            accept_downloads=True,
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            viewport={"width": 1440, "height": 1000},
        )
        page = ctx.new_page()
        try:
            accept_disclaimer_and_enter(page)

            # Strategy 1: Name search for the condominium master deed.
            name_search(page, SEARCH_NAME)
            rows, body = harvest_results(page, SEARCH_NAME)
            if rows:
                summary["master_deed"].update(book=rows[0]["book"], page=rows[0]["page"])
                summary["notes"].append("Found Book/Page in name-search results page text; "
                                        "open the row in --headed mode to confirm it is the "
                                        "Master Deed, then download.")
                try_download_current(page, f"master_deed_bk{rows[0]['book']}_pg{rows[0]['page']}")

            # Strategy 3: Plans search (separate index).
            plans_search(page)
            prows, _ = harvest_results(page, "Court Four plans")
            for r in prows:
                summary["plans"].append({"book": r["book"], "page": r["page"], "pdf": None})

            # Strategy 2 reminder if nothing found by name.
            if not summary["master_deed"]["book"]:
                summary["notes"].append(
                    "Name search returned no parseable Book/Page. Fallback: in --headed mode, "
                    "do a Recorded-Land address/unit search for '405 Washington St', open a "
                    "recent UNIT deed, and read its 'Master Deed ... Book ___ Page ___' "
                    "citation; then look up that master deed and its plan sheets. Also try "
                    "Registered Land (Land Court) if Recorded Land is empty.")
            shot(page, "final_state")
        except Exception as e:
            summary["notes"].append(f"ERROR: {e!r}")
            log(f"ERROR: {e!r}")
            shot(page, "error_state")
        finally:
            (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
            browser.close()

    # Human-readable summary
    print("\n" + "=" * 64)
    print("COURT FOUR CONDOMINIUM — RETRIEVAL SUMMARY")
    print("=" * 64)
    md = summary["master_deed"]
    print(f"Master Deed : Book {md['book']} Page {md['page']}  pdf={md['pdf']}")
    print(f"Plans       : {summary['plans'] or 'none parsed'}")
    print(f"Downloads   : {summary['downloads'] or 'none auto-downloaded'}")
    print(f"Screenshots : {SHOT_DIR}")
    if summary["notes"]:
        print("Notes:")
        for n in summary["notes"]:
            print(f"  - {n}")
    print("Full machine-readable summary -> court_four_records/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
