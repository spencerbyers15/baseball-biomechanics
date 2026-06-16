#!/usr/bin/env python3
"""
Premier League player height analysis, last five seasons (2021-22 .. 2025-26).

Builds a player-level dataset (season, club, player, height_cm, minutes), then
computes unweighted (roster) and minutes-weighted mean heights with dispersion
and 95% CIs, and writes a chart.

Data sources (all reachable from this sandbox; see data/SOURCES.md):
  * Heights  : EA Sports FC / FIFA "complete player dataset" (editions FIFA 15
               -> FC 24), height_cm field. Deduplicated to one height per player
               (latest edition). Player height is time-invariant for adults, so a
               name-keyed dictionary built from these editions supplies heights for
               all five seasons. Supplemented by three committed Transfermarkt
               "player-scores" (dcaribou) mirror subsets.
  * Rosters  : Fantasy Premier League (official FPL API snapshots, vaastav repo).
               Per season: every registered PL player, their club, and league
               minutes played. This is the spine: it defines who was in each
               season and supplies the minutes weights.

Transfermarkt and FBref (the originally requested sources) are network-blocked in
this environment (HTTP 403 host_not_allowed); see data/SOURCES.md for the full
reachability log and the substitution rationale.
"""
import csv, glob, os, sys, unicodedata, math
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# ---------------------------------------------------------------- name handling
# Latin letters that do NOT decompose under NFKD and must be transliterated
# explicitly (otherwise e.g. Polish 'Ł' or Nordic 'ø','ð' never match an ASCII form).
_SPECIAL = str.maketrans({
    "ł": "l", "Ł": "l", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d",
    "ð": "d", "Ð": "d", "þ": "th", "Þ": "th", "ı": "i", "ß": "ss",
    "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ŋ": "n",
})

def norm(s):
    """lowercase, transliterate special letters, strip accents, drop punctuation."""
    if not s:
        return ""
    s = s.translate(_SPECIAL)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    out = []
    for c in s:
        out.append(c if c.isalnum() or c.isspace() else " ")
    return " ".join("".join(out).split())

def surname_initial(full):
    """'kevin de bruyne' -> 'k de bruyne' (first initial + remaining tokens)."""
    t = norm(full).split()
    if len(t) < 2:
        return norm(full)
    return t[0][0] + " " + " ".join(t[1:])

# ---------------------------------------------------------------- height sources
def initial_lasttoken(name):
    """'kevin de bruyne' -> 'k bruyne' (first initial + final token)."""
    t = norm(name).split()
    if len(t) < 2:
        return norm(name)
    return t[0][0] + " " + t[-1]

def token_key(name):
    """order-agnostic key from the set of name tokens (>=2 tokens)."""
    t = sorted(set(norm(name).split()))
    return " ".join(t) if len(t) >= 2 else ""

def load_heights():
    """Return lookup dicts. Build several normalized indexes with collision guards.

    Indexes (checked from strict -> loose at match time):
      by_full : full normalized name
      by_si   : first-initial + all-but-first tokens  ('k de bruyne')
      by_il   : first-initial + last token            ('k bruyne')  -- handles
                FPL full legal names vs EA FC common names
      by_tok  : order-agnostic sorted token set       -- handles name-order flips
    For by_il / by_tok we record every height seen and only keep a key if all
    contributing heights agree within 2 cm (else it is an ambiguous collision and
    is dropped, so it can never produce a wrong match).
    """
    by_full = {}
    by_si   = {}
    by_short = {}         # normalized EA FC short_name -> height (for alias targets)
    il_all  = defaultdict(list)
    tok_all = defaultdict(list)
    heights_all = defaultdict(list)

    def add(name, height, tag, short=None):
        try:
            h = float(height)
        except (TypeError, ValueError):
            return
        if not (140 <= h <= 220):
            return
        f = norm(name)
        if f:
            heights_all[f].append(h)
            by_full.setdefault(f, h)
        si = surname_initial(name)
        if si:
            by_si.setdefault(si, h)
        if short:
            s = norm(short)            # 'k. de bruyne' -> 'k de bruyne'
            if s:
                by_si.setdefault(s, h)
                by_short.setdefault(s, h)
            il_all[initial_lasttoken(short)].append(h)
        il_all[initial_lasttoken(name)].append(h)
        tk = token_key(name)
        if tk:
            tok_all[tk].append(h)

    # EA FC / FIFA (primary)
    p = os.path.join(ROOT, "data/heights/eafc_heights.csv")
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            add(row["long_name"], row["height_cm"], "EAFC/FIFA", row["short_name"])

    # Transfermarkt mirrors (supplementary). Different column names per file.
    tm_files = {
        "data/heights/transfermarkt_ekovanda.csv": ("name", "height_in_cm"),
        "data/heights/transfermarkt_mattia.csv":   ("player", "height_in_cm"),
        "data/heights/transfermarkt_agos.csv":     ("name", "height_in_cm"),
    }
    for rel, (ncol, hcol) in tm_files.items():
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            continue
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                add(row.get(ncol, ""), row.get(hcol, ""), "Transfermarkt-mirror")

    # collapse collisions: keep a key only if all heights seen for it agree
    # within 2 cm; otherwise it is ambiguous (different players) and dropped.
    for k, hs in heights_all.items():
        if max(hs) - min(hs) > 2:
            by_full.pop(k, None)
    by_il  = {k: sum(v)/len(v) for k, v in il_all.items()  if max(v)-min(v) <= 2}
    by_tok = {k: sum(v)/len(v) for k, v in tok_all.items() if max(v)-min(v) <= 2}
    return {"full": by_full, "si": by_si, "il": by_il, "tok": by_tok, "short": by_short}

# FPL stores full legal names; EA FC/Transfermarkt list the common playing name.
# These aliases map the FPL legal name to the common name so the join succeeds.
# Heights are still taken from the height dictionary (EA FC) -- nothing is invented.
ALIASES = {
    "Bruno Borges Fernandes": "Bruno Fernandes",
    "Diogo Dalot Teixeira": "Diogo Dalot",
    "José Malheiro de Sá": "José Pedro Malheiro de Sá",
    "Rúben Santos Gato Alves Dias": "Rúben Dias",
    "Rúben Gato Alves Dias": "Rúben Dias",
    "Rúben da Silva Neves": "Rúben Neves",
    "João Palhinha Gonçalves": "João Palhinha",
    "João Maria Lobo Alves Palhares Costa Palhinha Gonçalves": "João Palhinha",
    "Bernardo Veiga de Carvalho e Silva": "Bernardo Silva",
    "Gabriel Martinelli Silva": "Gabriel Martinelli",
    "Andreas Hoelgebaum Pereira": "Andreas Pereira",
    "Jefferson Lerma Solís": "Jefferson Lerma",
    "Emiliano Buendía Stati": "Emiliano Buendía",
    "Vicente Guaita": "Vicente Guaita Panadero",
    "Marc Cucurella": "Marc Cucurella Saseta",
    "Fernando Marçal": "Fernando Marçal de Oliveira",
    "Johann Berg Gudmundsson": "Jóhann Berg Guðmundsson",
    "Jóhann Berg Gudmundsson": "Jóhann Berg Guðmundsson",
    "Hee-Chan Hwang": "Hwang Hee Chan",          # via short-name index (CJK long name)
    "Kepa Arrizabalaga": "Kepa Arrizabalaga Revuelta",
    "Edson Álvarez Velázquez": "Edson Álvarez",
    "Diogo Teixeira da Silva": "Diogo Jota",
    "Emerson Leite de Souza Junior": "Emerson Royal",
    "Ben Brereton": "Benjamin Anthony Brereton Díaz",
    "Junior Firpo Adames": "Junior Firpo",
    "Marcos Senesi Barón": "Marcos Nicolás Senesi Barón",
    "Raúl Jiménez Rodríguez": "Raúl Alonso Jiménez Rodríguez",
    "Yéremy Pino Santos": "Yeremy Jesús Pino Santos",
    "Igor Thiago Nascimento Rodrigues": "Igor Thiago Rodrigues",
    "Junior Kroupi": "Eli Junior Kroupi",
    "Toti Gomes": "Toti Gomes",                  # via short-name index
    "Johann Berg Gudmundsson": "J. Guðmundsson", # via short-name index
    "Jóhann Berg Gudmundsson": "J. Guðmundsson",
    # Genuinely absent from EA FC 24 (PL debutants after the FC24 data freeze):
    # Álex Jiménez, Diego Gómez, Igor Jesus -> left UNMATCHED and reported as gaps.
}

def match_height(full_name, web_name, H):
    """Try progressively looser keys. Returns (height, how) or (None, None)."""
    if full_name in ALIASES:
        a = norm(ALIASES[full_name])
        if a in H["full"]:
            return H["full"][a], "alias"
        if a in H["short"]:
            return H["short"][a], "alias"
    f = norm(full_name)
    if f in H["full"]:
        return H["full"][f], "full"
    tk = token_key(full_name)
    if tk and tk in H["tok"]:
        return H["tok"][tk], "tokens"          # order-agnostic full match
    si = surname_initial(full_name)
    if si in H["si"]:
        return H["si"][si], "surname+initial"
    il = initial_lasttoken(full_name)
    if il in H["il"]:
        return H["il"][il], "initial+lasttoken"
    w = norm(web_name)
    if w and w in H["full"]:
        return H["full"][w], "webname-full"
    return None, None

# ---------------------------------------------------------------- FPL rosters
def load_fpl_season(season):
    base = os.path.join(ROOT, "data/fpl", season)
    teams = {}
    with open(os.path.join(base, "teams.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams[row["id"]] = row["name"]
    players = []
    with open(os.path.join(base, "players_raw.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full = (row.get("first_name", "") + " " + row.get("second_name", "")).strip()
            players.append({
                "club": teams.get(row.get("team", ""), "?"),
                "player": full,
                "web_name": row.get("web_name", ""),
                "minutes": int(row.get("minutes") or 0),
            })
    return players

# ---------------------------------------------------------------- stats
def weighted_stats(values, weights):
    """minutes-weighted mean, sd, se, 95% CI using effective sample size (Kish)."""
    v = [float(x) for x in values]
    w = [float(x) for x in weights]
    sw = sum(w)
    if sw <= 0:
        return None
    mean = sum(wi * vi for wi, vi in zip(w, v)) / sw
    var = sum(wi * (vi - mean) ** 2 for wi, vi in zip(w, v)) / sw
    sd = math.sqrt(var)
    n_eff = (sw ** 2) / sum(wi ** 2 for wi in w)   # Kish effective sample size
    se = sd / math.sqrt(n_eff) if n_eff > 0 else float("nan")
    return mean, sd, se, 1.96 * se, n_eff

def unweighted_stats(values):
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return None
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return mean, sd, se, 1.96 * se, n

# ---------------------------------------------------------------- main
def main():
    H = load_heights()
    print("height dictionary keys: " + ", ".join(f"{k}={len(v)}" for k, v in H.items()))

    rows = []            # tidy player-level output
    per_season = defaultdict(list)
    for season in SEASONS:
        for p in load_fpl_season(season):
            h, how = match_height(p["player"], p["web_name"], H)
            rows.append({
                "season": season, "club": p["club"], "player": p["player"],
                "height": ("" if h is None else int(round(h))),
                "minutes": p["minutes"],
                "height_match": how or "UNMATCHED",
            })
            per_season[season].append((p["club"], p["player"], h, p["minutes"]))

    # ---- tidy player CSV
    out_csv = os.path.join(ROOT, "player_heights.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["season", "club", "player", "height", "minutes", "height_match"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv, f"({len(rows)} rows)")

    # ---- season summary
    summ_rows = []
    clubs_by_season = {}
    wmean_by_season = {}
    for season in SEASONS:
        recs = per_season[season]
        n_roster = len(recs)
        with_h = [(c, pl, h, m) for (c, pl, h, m) in recs if h is not None]
        n_h = len(with_h)
        heights = [h for (_, _, h, _) in with_h]
        # unweighted roster mean (all roster players with a height)
        um = unweighted_stats(heights)
        # minutes-weighted mean (players with height AND minutes>0)
        played = [(h, m) for (_, _, h, m) in with_h if m > 0]
        wm = weighted_stats([h for h, _ in played], [m for _, m in played])
        cov_h = 100.0 * n_h / n_roster if n_roster else 0
        # minutes coverage: share of all league minutes played by height-matched
        # players. This is what governs the reliability of the *weighted* mean --
        # a roster gap only matters if those players actually logged minutes.
        tot_min = sum(m for (_, _, _, m) in recs)
        cov_min = sum(m for (_, _, h, m) in recs if h is not None)
        mins_cov = 100.0 * cov_min / tot_min if tot_min else 0
        n_play_h = len(played)
        clubs_by_season[season] = set(c for (c, _, _, _) in recs)
        wmean_by_season[season] = wm[0] if wm else None
        # FPL snapshot completeness (full season = 38 GW, ever-present ~3420 min)
        max_min = max((m for (_, _, _, m) in recs), default=0)
        complete = "yes" if max_min >= 3300 else f"PARTIAL (~{round(max_min/90)}/38 GW)"
        summ_rows.append({
            "season": season,
            "n_roster": n_roster,
            "n_height": n_h,
            "coverage_height_pct": round(cov_h, 1),
            "coverage_minutes_pct": round(mins_cov, 1),
            "season_complete": complete,
            "n_played_with_height": n_play_h,
            "uw_mean": round(um[0], 2), "uw_sd": round(um[1], 2),
            "uw_se": round(um[2], 3), "uw_ci95": round(um[3], 2),
            "w_mean": round(wm[0], 2), "w_sd": round(wm[1], 2),
            "w_se": round(wm[2], 3), "w_ci95": round(wm[3], 2),
            "w_neff": round(wm[4], 1),
        })

    summ_csv = os.path.join(ROOT, "season_summary.csv")
    with open(summ_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ_rows[0].keys()))
        w.writeheader()
        w.writerows(summ_rows)
    print("wrote", summ_csv)

    # ---- console report
    print("\n=== SEASON SUMMARY ===")
    hdr = ("season", "n_ros", "n_ht", "cov%", "uw_mean±ci", "w_mean±ci", "w_se", "n_eff")
    print("{:<8} {:>5} {:>5} {:>5}  {:>14} {:>14} {:>6} {:>7}".format(*hdr))
    for r in summ_rows:
        print("{:<8} {:>5} {:>5} {:>5}  {:>7.2f}±{:<5.2f} {:>7.2f}±{:<5.2f} {:>6.3f} {:>7.1f}".format(
            r["season"], r["n_roster"], r["n_height"], r["coverage_height_pct"],
            r["uw_mean"], r["uw_ci95"], r["w_mean"], r["w_ci95"], r["w_se"], r["w_neff"]))

    # ---- promotion/relegation turnover decomposition (weighted mean)
    print("\n=== PROMOTION/RELEGATION TURNOVER EFFECT (minutes-weighted mean) ===")
    for i in range(1, len(SEASONS)):
        s0, s1 = SEASONS[i - 1], SEASONS[i]
        c0, c1 = clubs_by_season[s0], clubs_by_season[s1]
        stayed = c0 & c1
        # weighted mean over only the clubs that stayed, each season
        def wmean_clubs(season, clubs):
            recs = [(h, m) for (c, _, h, m) in per_season[season]
                    if c in clubs and h is not None and m > 0]
            ws = weighted_stats([h for h, _ in recs], [m for _, m in recs])
            return ws[0] if ws else None
        full0, full1 = wmean_by_season[s0], wmean_by_season[s1]
        stay0, stay1 = wmean_clubs(s0, stayed), wmean_clubs(s1, stayed)
        total_change = full1 - full0
        continuing_change = stay1 - stay0
        turnover_part = total_change - continuing_change
        print(f"{s0}->{s1}: total dW={total_change:+.2f} cm | "
              f"continuing-{len(stayed)}-clubs d={continuing_change:+.2f} | "
              f"club-turnover share d={turnover_part:+.2f} cm "
              f"(clubs in/out: {len(c1-c0)})")

    # ---- is any movement distinguishable from noise?
    print("\n=== TREND vs NOISE (minutes-weighted mean) ===")
    sr = {r["season"]: r for r in summ_rows}
    print("year-over-year change vs SE of the difference (|z|>1.96 => significant):")
    for i in range(1, len(SEASONS)):
        s0, s1 = SEASONS[i - 1], SEASONS[i]
        d = sr[s1]["w_mean"] - sr[s0]["w_mean"]
        se_d = math.sqrt(sr[s0]["w_se"] ** 2 + sr[s1]["w_se"] ** 2)
        print(f"  {s0}->{s1}: {d:+.2f} cm  SE_diff={se_d:.2f}  z={d/se_d:+.2f}"
              f"  {'SIGNIFICANT' if abs(d/se_d) > 1.96 else 'within noise'}")
    def cumz(a, b):
        d = sr[b]["w_mean"] - sr[a]["w_mean"]
        se_d = math.sqrt(sr[a]["w_se"] ** 2 + sr[b]["w_se"] ** 2)
        return d, se_d, d / se_d
    d, se, z = cumz(SEASONS[0], SEASONS[-1])
    print(f"  cumulative {SEASONS[0]}->{SEASONS[-1]} (incl. PARTIAL final): "
          f"{d:+.2f} cm  z={z:+.2f}  {'significant' if abs(z) > 1.96 else 'within noise'}")
    d, se, z = cumz(SEASONS[0], SEASONS[-2])
    print(f"  cumulative {SEASONS[0]}->{SEASONS[-2]} (4 complete seasons):  "
          f"{d:+.2f} cm  z={z:+.2f}  {'significant' if abs(z) > 1.96 else 'within noise'}")
    print(f"  reference long-run trend ~+1 cm/decade => ~{0.1*4:.1f} cm over these 4 seasons; "
          f"per-season SE ~{sum(r['w_se'] for r in summ_rows)/len(summ_rows):.2f} cm "
          f"=> a 5-season window cannot resolve it.")

    make_chart(summ_rows)
    return summ_rows


def make_chart(summ_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seasons = [r["season"] for r in summ_rows]
    x = list(range(len(seasons)))
    y = [r["w_mean"] for r in summ_rows]
    ci = [r["w_ci95"] for r in summ_rows]
    partial = [not r["season_complete"].startswith("yes") for r in summ_rows]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    # ~1 cm/decade reference line anchored at the first season's value
    ref = [y[0] + 0.1 * i for i in range(len(seasons))]  # 0.1 cm/yr
    ax.plot(x, ref, ls="--", lw=1, color="grey", alpha=0.6,
            label="~1 cm/decade reference")

    # split complete vs provisional markers
    for xi, yi, ei, pi in zip(x, y, ci, partial):
        ax.errorbar(xi, yi, yerr=ei, fmt="o", ms=8, capsize=5, lw=1.6,
                    color="#b08900" if pi else "#1f4e79",
                    mfc="white" if pi else "#1f4e79",
                    label=None)
    ax.plot(x, y, color="#1f4e79", lw=1.4, alpha=0.7, zorder=0)

    # y-axis: real data range + sensible margin (NOT zero-anchored, NOT zoomed)
    lo = min(yi - ei for yi, ei in zip(y, ci))
    hi = max(yi + ei for yi, ei in zip(y, ci))
    pad = max(0.6, 0.25 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xticks(x); ax.set_xticklabels(seasons)
    ax.set_ylabel("Minutes-weighted mean height (cm)")
    ax.set_xlabel("Premier League season")
    ax.set_title("Premier League minutes-weighted mean player height\n(95% CI; error bars use Kish effective n)")
    for xi, yi, pi in zip(x, y, partial):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    # legend: reference line + provisional note
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], ls="--", color="grey", alpha=0.6, label="~1 cm/decade reference"),
        Line2D([0], [0], marker="o", color="#1f4e79", mfc="#1f4e79", ls="", label="complete season"),
        Line2D([0], [0], marker="o", color="#b08900", mfc="white", ls="", label="2025-26 (partial, 29/38 GW)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(ROOT, "height_trend.png")
    fig.savefig(out, dpi=150)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
