# Court Four Condominium — records retrieval

Goal: pull the recorded **Master Deed** and recorded **Condominium Plans** for
"Court Four Condominium", 405 Washington St, Brookline, MA, from the Norfolk
County (MA) Registry of Deeds ALIS system (norfolkresearch.org).

## Status

- `RETRIEVAL_BLOCKED.md` — the Claude Code **web** environment blocks the registry
  hosts at its egress proxy (`HTTP 403 / x-deny-reason: host_not_allowed`), so the
  live retrieval cannot run there. Re-verified blocked on 2026-06-20.
- `fetch_court_four.py` — ready-to-run Playwright driver. Run it from a machine
  with normal internet (Spencer's Mac or Nellie), or from a web env whose network
  policy allows `norfolkdeeds.org` + `norfolkresearch.org`.

## Run it

```bash
pip install playwright
python -m playwright install chromium

# headless auto-run
python court_four_records/fetch_court_four.py

# recommended first time: watch it, so you can fix any selector that misses
python court_four_records/fetch_court_four.py --headed --slow 300
```

Outputs land in this folder:
- `screenshots/NN_step.png` — one per step (disclaimer, free access, search forms, results)
- `*.pdf` — downloaded documents
- `summary.json` — master-deed book/page, plan refs, what downloaded vs not

## Notes on the script

The ALIS DOM could not be inspected from the blocked environment, so the script
uses **text-based locators with fallbacks** and screenshots every step. If a step
misses (e.g. the "Free Access" or "Name Search" button has different wording),
run `--headed`, look at the screenshot, and add the real wording to the
`SELECTOR CANDIDATES` lists near the top of `fetch_court_four.py`.

Search order implemented (matches the task plan):
1. Recorded Land → Name search `Court Four Condominium` → master deed → download.
2. Plans search for the condo plans (separate index).
3. Fallback (documented in `summary.json` notes, do in `--headed`): address/unit
   search for `405 Washington St`, open a recent unit deed, read its
   "Master Deed … Book ___ Page ___" citation, then fetch that deed + plans.
   Try Registered Land (Land Court) if Recorded Land is empty.

Guardrails respected: one polite session, no proxy/allowlist bypass, stop on
CAPTCHA/login/hard block and report.
