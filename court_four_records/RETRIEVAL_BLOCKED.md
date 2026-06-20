# Court Four Condominium — Records Retrieval: BLOCKED (network egress)

**Date:** 2026-06-20
**Target:** Master Deed + recorded Condominium Plans for "Court Four Condominium",
405 Washington St, Brookline, MA — Norfolk County (MA) Registry of Deeds.
**Result:** NOT retrieved. Hard block before any search could run.

## What blocked it

This task runs in Claude Code's **remote (cloud) execution environment**. All
outbound traffic goes through an egress proxy with a host **allowlist**. The
Norfolk Registry hosts are **not on the allowlist**, so every request is rejected
by the proxy *before it ever reaches the registry's servers*.

Exact responses observed (all hosts tried, real browser User-Agent):

| URL | Result |
|-----|--------|
| https://www.norfolkdeeds.org/research/begin-online-research/ | `HTTP/2 403` + header `x-deny-reason: host_not_allowed` |
| https://norfolkdeeds.org/ | `HTTP/2 403` + `x-deny-reason: host_not_allowed` |
| https://www.norfolkresearch.org/ (the ALIS app host) | `HTTP/2 403` + `x-deny-reason: host_not_allowed` |
| (harness WebFetch on the research page) | `HTTP 403 Forbidden` |

For contrast, an allowlisted host (`pypi.org`) returned `HTTP 200`, and a
non-allowlisted host (`google.com`) returned the same `x-deny-reason: host_not_allowed`.
So this is the **environment network policy**, not the registry's robot blocking
and not a CAPTCHA/login wall.

## Why Playwright wouldn't help here

Installing a headless browser was the right plan, but it doesn't change the
outcome in this environment: a headless Chromium's traffic is routed through the
same egress proxy (`CLAUDE_CODE_PROXY_RESOLVES_HOSTS=true`), and the proxy blocks
`norfolkresearch.org` regardless of which client makes the request. I did **not**
attempt to bypass the proxy/allowlist (per the task guardrails).

## How to unblock (pick one)

1. **Recreate this web session's environment with a network policy that allows
   the registry hosts** — at minimum:
   - `norfolkdeeds.org`, `www.norfolkdeeds.org`
   - `norfolkresearch.org`, `www.norfolkresearch.org`
   (The "Trusted"/open or a custom allowlist policy. See
   https://code.claude.com/docs/en/claude-code-on-the-web for network policy
   options.) Then re-run this task and the Playwright plan below will execute.

2. **Run the retrieval locally** (the user's Mac or Nellie, which have normal
   internet). The Playwright script plan in the next section is ready to use there.

## Ready-to-run plan once network is available

Search order on the ALIS app (norfolkresearch.org), Free Access mode:
1. **Recorded Land → Name (Grantor/Grantee) search:** `Court Four Condominium`
   → open the Master Deed, capture Book/Page, download the PDF + any plan sheets.
2. If name search is empty: **find a recent UNIT deed** for 405 Washington St,
   open it, and read the "Master Deed ... Book ___ Page ___" citation; then pull
   that master deed and its plans.
3. **Plans search** for Court Four / 405 Washington St (condo plans are often
   recorded as a separate Plan Book reference).
4. If nothing in Recorded Land, repeat in **Registered Land (Land Court)**.

Take a screenshot at each step; save PDFs to `./court_four_records/` with names
like `master_deed_bkXXXXX_pgYYY.pdf` and `plans_<ref>.pdf`.

## Status of deliverables
- Master deed book/page: **not found** (could not reach the registry).
- Plan reference(s): **not found**.
- PDFs downloaded: **none**.
