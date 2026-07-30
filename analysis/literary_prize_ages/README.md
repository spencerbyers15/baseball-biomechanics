# Author age at Booker / Pulitzer nomination and win, 1976–2025

Question: have the ages of authors at the time of nomination or win been trending
up over the last 50 years?

**Answer: No. Both prizes are remarkably flat over 50 years.** Booker: −0.1
years/decade (p = 0.83). Pulitzer Fiction: +0.3 years/decade (p = 0.68).
Winners-only trends are also statistically indistinguishable from flat. The only
suggestive (not significant) signal is a 2020s uptick in Pulitzer ages: finalists
+ winners averaged 57.6 in 2020–2025 vs 52.0 before (p = 0.08); the seven 2020s
winners averaged 59.9 (Erdrich 67, Kingsolver 68, Phillips 72, Everett 69, ...).

## Data

- **Booker Prize**: winners + full shortlists, 1976–2025 (300 author-year entries).
- **Pulitzer Prize for Fiction**: winners 1976–2025, plus non-winning finalists
  from 1980 (the first year finalists were publicly announced) — 144 entries.
  No award was given in 1977 and 2012.
- Age is approximated as `prize year − birth year` (±1 year of noise; symmetric
  over time, so it doesn't bias the trend).
- Three posthumous honorees excluded from age stats: John Kennedy Toole
  (1981 winner, d. 1969), Raymond Carver (1989 finalist, d. 1988), David Foster
  Wallace (2012 finalist, d. 2008).
- Data was compiled from model knowledge, then verified against web sources
  (thebookerprizes.com, pulitzer.org, Wikipedia, obituaries) by three
  independent verification passes; 7 corrections were applied. Remaining known
  soft spots: birth years for Chetna Maroo, Jonathan Escoffery, Stacey Levine,
  and Margaret Verble are estimates (no public record found); Maaza Mengiste
  sources conflict (1971 vs 1974).

## Key numbers

Decade mean ages (nominees + winners):

| Decade | Booker | Pulitzer |
|--------|--------|----------|
| 1976–79 | 50.7 | 54.3 (winners only) |
| 1980s | 51.1 | 52.1 |
| 1990s | 49.9 | 54.4 |
| 2000s | 47.3 | 52.8 |
| 2010s | 49.9 | 48.6 |
| 2020s | 51.0 | 57.6 |

- Booker extremes: Eleanor Catton, 28 (youngest winner, 2013); Alan Garner, 88
  (oldest shortlistee, 2022).
- Pulitzer extremes: Karen Russell, 31 (2012 finalist); Lore Segal, 80 (2008
  finalist).

## Files

- `dataset.py` — the full dataset (year, role, author, birth year, book)
- `analyze.py` — regression + decade tables + chart
- `prize_ages.csv` — flat export of all 444 entries
- `prize_ages.png` — scatter + rolling mean + trend lines, both prizes

Run: `python analyze.py` (needs pandas, scipy, matplotlib).
