"""
Small- vs large-multifamily complaint-TEXT contrast for the overview post
(post0_overview_substack.md, "the broader landscape of DOB complaints").

Mirrors the Asian-owner text supplement (scripts/text_analysis_race.py +
scripts/make_text_figures.py): Monroe, Colaresi & Quinn (2008) informed-
Dirichlet weighted log-odds z-scores of callers' complaint subjects. The two
groups here are building SIZE (small multifamily vs large multifamily
complex) instead of predicted owner race, and ALL complaint categories are
pooled -- the size contrast is about the whole complaint landscape, not one
category, so it surfaces both the different complaint mixes and the different
wording at small vs large buildings.

Groups: residential unit count (PLUTO unitsres), joined to scraped complaint
text on the 10-digit BBL. Small multifamily = 2-9 units; large multifamily =
100+ units; the 10-99 middle is dropped to sharpen the contrast (as the race
plot uses P>0.7 on each side). 311-referenced complaints only (citizen text,
not agency referrals); real caller text only (subject >10 chars, non-internal).

The text utilities (STOP, normalize_token, tokenizer, informed-Dirichlet
engine) are lifted verbatim from scripts/text_analysis_race.py so this figure
stays byte-for-byte comparable to the race version; keep them in sync.

Run `python3 scripts/size_text_figure.py` for the default figure, or with
`--resonly` to hold BUILDING USE constant: restrict the sample to purely
residential buildings, defined IDENTICALLY to regression spec 3 in
data/analysis/owner_commercial_sensitivity.md -- no non-residential units
(unitstotal == unitsres) AND a residential building class (bldgclass first
letter in A/B/C/D). Mixed-use / commercial-class buildings load onto the
"large" side and inject storefront vocabulary (e.g. "sidewalk shed") that reads
as a SIZE effect but is really a USE effect; --resonly removes them and writes a
parallel set of *_resonly artifacts. The default run is unchanged so the two
figures can be compared side by side.

Concurrency: a live scrape may be writing to the DB, so the database is opened
READ-ONLY with a busy timeout and lock-retry backoff. This script never writes
to the database.

Outputs (default / --resonly):
  data/analysis/blog_posts/artifacts/size_complaint_words[_resonly].png
  data/analysis/risk_models/size_word_logodds[_resonly].csv
  data/analysis/risk_models/size_bigram_logodds[_resonly].csv
"""
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl

OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"

# ---- size groups (residential units) ----
SMALL = (2, 9)          # small multifamily: 2-9 units (exclude 1 = single-family)
LARGE_MIN = 100         # large multifamily complex: 100+ units
MIN_UNI = 60            # display floor: pooled unigram count
MIN_BI = 40             # display floor: pooled bigram count
SMALL_LBL = f"{SMALL[0]}-{SMALL[1]} units"
LARGE_LBL = f"{LARGE_MIN}+ units"

# ---- purely-residential restriction (only used with --resonly) ----
# Hold BUILDING USE constant so a size contrast is not contaminated by the
# store/office/mixed-use vocabulary that rides along with large mixed-use
# buildings. Definition is IDENTICAL to regression spec 3 in
# data/analysis/owner_commercial_sensitivity.md: no non-residential units
# (unitstotal == unitsres) AND a residential building class (bldgclass first
# letter in A/B/C/D). This uses ONLY panel columns -- no comarea / Socrata pull
# (spec 3 has no comarea term), so the run stays offline and DB-read-only.
# Writes a parallel set of *_resonly artifacts; the default run is unchanged.
RESIDENTIAL_ONLY = "--resonly" in sys.argv
SUFFIX = "_resonly" if RESIDENTIAL_ONLY else ""
RES_CLASSES = ("A", "B", "C", "D")   # 1-fam / 2-fam / walk-up / elevator apts
# (excludes mixed-use S*, store K*, office O*, condo R*, and all non-res classes)

# ---- text utilities: lifted from scripts/text_analysis_race.py; keep in sync ----
STOP = set("""a an and are as at be been being by for from has have he her his i in is it its
of on or our she that the their them they this to was we were what when where which who will
with you your there here would could should very also am do does did been if so no not nor
they're it's im i'm dont don't cant can't wont won't ny nyc please someone something being
address street ave avenue st blvd road apt apartment number caller customer states reporting
reports state say says said want wants would like know oh th rd nd
into onto out over under through around off up down about after before during between
against within without near behind above below since while because than then now just
also only even still some any all both each more most other another""".split())


def normalize_token(tok: str) -> str:
    """Merge plural and singular forms (families->family, rooms->room)."""
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


# tokens are normalized before the stopword test, so STOP must also contain the
# normalized (stemmed) form of every stopword (see text_analysis_race.py).
STOP |= {normalize_token(w) for w in STOP}


def keep_token(tok: str) -> bool:
    return tok not in STOP


def tokens(t):
    tk = [normalize_token(tok) for tok in re.findall(r"[a-z']{3,}", t)]
    out, i = [], 0
    while i < len(tk):
        if tk[i:i + 3] == ["stop", "work", "order"]:
            out.append("stop work order"); i += 3
        else:
            out.append(tk[i]); i += 1
    return out


def uni_counts(series):
    cnt = Counter()
    for t in series:
        for tok in tokens(t):
            if keep_token(tok):
                cnt[tok] += 1
    return cnt


def bi_counts(series):
    cnt = Counter()
    for t in series:
        tk = tokens(t)
        for x, y in zip(tk, tk[1:]):
            if keep_token(x) and keep_token(y):
                cnt[f"{x} {y}"] += 1
    return cnt


def dirichlet_logodds(c_large, c_small, min_count):
    """Monroe et al. (2008) informed-Dirichlet log-odds z-scores.
    Positive z = more typical at LARGE multifamily; negative = small."""
    vocab = {t for t in set(c_large) | set(c_small)
             if c_large.get(t, 0) + c_small.get(t, 0) >= min_count}
    nL, nS = sum(c_large.values()), sum(c_small.values())
    prior = {t: c_large.get(t, 0) + c_small.get(t, 0) for t in vocab}
    a0 = sum(prior.values())
    k = 500 / a0  # prior strength
    rows = []
    for t in vocab:
        pa = prior[t] * k
        lL = np.log((c_large.get(t, 0) + pa) / (nL + a0 * k - c_large.get(t, 0) - pa))
        lS = np.log((c_small.get(t, 0) + pa) / (nS + a0 * k - c_small.get(t, 0) - pa))
        var = 1 / (c_large.get(t, 0) + pa) + 1 / (c_small.get(t, 0) + pa)
        rows.append({"word": t, "z": (lL - lS) / np.sqrt(var),
                     "n_large": c_large.get(t, 0), "n_small": c_small.get(t, 0)})
    return pd.DataFrame(rows).sort_values("z")


def _connect_ro(retries=6):
    """Open the DB read-only with a busy timeout; retry with backoff on lock.
    A live scrape may hold a write lock, so this figure never opens the DB for
    writing (see scripts/owner_text_figure.py)."""
    uri = f"file:{config.DB_PATH}?mode=ro"
    last = None
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=60)
            conn.execute("PRAGMA busy_timeout=60000;")
            return conn
        except sqlite3.OperationalError as e:  # e.g. database is locked
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise last


def load():
    cols = ["bbl_key", "unitsres"]
    if RESIDENTIAL_ONLY:
        cols += ["unitstotal", "bldgclass"]   # needed for the spec-3 filter
    panel = pd.read_csv(PANEL, dtype={"bbl_key": str}, low_memory=False,
                        usecols=cols)
    panel = panel.dropna(subset=["unitsres"])

    conn = _connect_ro()
    try:
        for attempt in range(6):
            try:
                c = pd.read_sql_query("""
                    SELECT CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
                                WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
                                WHEN 'STATEN ISLAND' THEN '5' END boro, b.block, b.lot,
                           b.subject, b.ref_311
                    FROM open_data o JOIN bis_scrape b USING(complaint_number)
                    WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
                      AND substr(o.date_entered,7,4) >= '2020'""", conn)
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() or attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 30))
    finally:
        conn.close()

    c["bbl_key"] = [make_bbl(b, bl, l) for b, bl, l in zip(c["boro"], c["block"], c["lot"])]
    c = c.merge(panel, on="bbl_key")
    c["subject"] = c["subject"].fillna("").astype(str)
    # citizen complaints only (a 311 reference number => caller-originated)
    c = c[c["ref_311"].fillna("").astype(str).str.strip() != ""]
    # real caller text only: >10 chars and not a DOB-internal sweep code
    internal = c["subject"].str.contains(r"\*\*\*|^\d{6,}\s*\(D\d+\)?$", regex=True, na=False)
    c = c[(c["subject"].str.len() > 10) & ~internal]
    return c


def apply_resonly(c):
    """Restrict to PURELY RESIDENTIAL buildings so building USE is held constant,
    matching regression spec 3 in data/analysis/owner_commercial_sensitivity.md:
    no non-residential units (unitstotal == unitsres) AND a residential building
    class (bldgclass first letter in A/B/C/D). Prints buildings / subjects
    removed overall and for the small (2-9) and large (100+) groups that feed the
    figure. Conservative on missing values: an unknown unitstotal or bldgclass
    fails the test and is dropped."""
    small_m, large_m = c["unitsres"].between(*SMALL), c["unitsres"] >= LARGE_MIN
    n0_bld, n0_sub = c["bbl_key"].nunique(), len(c)
    n0_small, n0_large = int(small_m.sum()), int(large_m.sum())

    cls0 = c["bldgclass"].astype(str).str[0]
    pure = (c["unitstotal"] == c["unitsres"]) & cls0.isin(RES_CLASSES)
    c = c[pure].copy()

    sm, lg = c["unitsres"].between(*SMALL), c["unitsres"] >= LARGE_MIN
    n1_bld, n1_sub = c["bbl_key"].nunique(), len(c)
    n1_small, n1_large = int(sm.sum()), int(lg.sum())
    print("purely-residential filter (spec 3): unitstotal==unitsres AND bldgclass in A/B/C/D")
    print(f"  buildings (distinct BBL): {n0_bld:,} -> {n1_bld:,} (removed {n0_bld - n1_bld:,}, "
          f"{100 * (n0_bld - n1_bld) / n0_bld:.1f}%)")
    print(f"  all caller subjects:      {n0_sub:,} -> {n1_sub:,} (removed {n0_sub - n1_sub:,}, "
          f"{100 * (n0_sub - n1_sub) / n0_sub:.1f}%)")
    print(f"  small ({SMALL_LBL}) subjects: {n0_small:,} -> {n1_small:,} (removed {n0_small - n1_small:,}, "
          f"{100 * (n0_small - n1_small) / max(n0_small, 1):.1f}%)")
    print(f"  large ({LARGE_LBL}) subjects: {n0_large:,} -> {n1_large:,} (removed {n0_large - n1_large:,}, "
          f"{100 * (n0_large - n1_large) / max(n0_large, 1):.1f}%)")
    return c


# ---- figure styling: lifted from scripts/make_text_figures.py ----
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"
AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

DISPLAY_EXCLUDE = {"http", "https", "www", "com",
                   "http www", "new york", "york city", "staten island",
                   "york", "staten", "island", "brooklyn",
                   "queens", "bronx", "manhattan"}
URL_TOKENS = {"http", "https", "www", "com"}


TOP_N = 10          # per side, per kind: 10 single words + 10 word pairs each side of zero
# kind -> (colour, marker, marker size). Kind is carried by BOTH colour and marker
# shape, so it survives colour-blind and greyscale reading. Matches the style set in
# scripts/make_text_figures.py (BLUE circles / AQUA diamonds); keep in sync.
KINDS = [("Single words", BLUE, "o", 58), ("Word pairs", AQUA, "D", 44)]


def _merged_panel(ax, z, zb, side_note):
    """Single words AND word pairs on ONE cartesian plane (see origin/owner/
    strictness/race versions; keep in sync).

    They can share an x-axis because they share a metric: the informed-Dirichlet
    log-odds z-score. Single words sit farther from zero -- they are counted far
    more often, so their z carries less sampling noise -- while word pairs land
    nearer the middle. Points are sorted by z, so the plane reads bottom-left to
    top-right, with the pairs occupying the centre band."""
    def top(df, kind):
        d = df[~df["word"].isin(DISPLAY_EXCLUDE)]
        d = d[~d["word"].apply(lambda w: any(t in URL_TOKENS for t in w.split()))]
        return pd.concat([d.nsmallest(TOP_N, "z"),
                          d.nlargest(TOP_N, "z")]).assign(kind=kind)

    d = (pd.concat([top(z, "Single words"), top(zb, "Word pairs")])
           .sort_values("z").reset_index(drop=True))
    d["y"] = np.arange(len(d))

    ax.axvline(0, color=ZERO, lw=1.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for kind, colour, marker, size in KINDS:
        s = d[d["kind"] == kind]
        ax.scatter(s["z"], s["y"], s=size, color=colour, marker=marker, zorder=3,
                   edgecolors=SURFACE, linewidths=1.4, label=kind)
    for _, row in d.iterrows():
        ha = "right" if row["z"] < 0 else "left"
        off = -7 if row["z"] < 0 else 7
        ax.annotate(row["word"], (row["z"], row["y"]), textcoords="offset points",
                    xytext=(off, 0), va="center", ha=ha, fontsize=9, color=INK)
    ax.set_yticks([])
    zmin, zmax = float(d["z"].min()), float(d["z"].max())
    span = zmax - zmin
    ax.set_xlim(zmin - 0.36 * span, zmax + 0.36 * span)
    ax.set_ylim(-0.9, len(d) - 0.1)
    ax.set_xlabel("informed-Dirichlet log-odds z-score (unitless)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9,
                    handletextpad=0.4, borderaxespad=0.6, scatterpoints=1)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.text(0.985, 0.02, side_note, transform=ax.transAxes, fontsize=8.8,
            color=MUTED, ha="right", va="bottom", style="italic")


def make_figure(z, zb, n_large, n_small):
    fig, ax = plt.subplots(figsize=(7.6, 10.2),
                           gridspec_kw={"left": 0.04, "right": 0.97,
                                        "top": 0.868, "bottom": 0.052})
    _merged_panel(ax, z, zb,
                  f"right of zero = more typical at large multifamily ({LARGE_LBL})\n"
                  f"left of zero = more typical at small multifamily ({SMALL_LBL})")
    fig.suptitle("Words and word pairs that distinguish complaints at small vs. large\nmultifamily buildings",
                 x=0.02, y=0.987, ha="left", fontsize=12.5, color=INK, weight="semibold")
    scale_note = ("top 10 each side for words and for word pairs, ranked together on one\n"
                  "scale · URL fragments and place names omitted for display")
    if RESIDENTIAL_ONLY:
        subtitle = (
            f"Callers' own complaint text (not inspector reports) · 311-referenced complaints, 2020-present · purely\n"
            f"residential buildings only (no non-residential units; building class A-D) · {n_large:,} subjects at large\n"
            f"multifamily ({LARGE_LBL}) vs. {n_small:,} at small multifamily ({SMALL_LBL}) · singular and plural forms combined ·\n"
            f"adjacent words where neither is a stopword · {scale_note}")
    else:
        subtitle = (
            f"Callers' own complaint text (not inspector reports) · 311-referenced complaints, 2020-present · "
            f"{n_large:,} subjects\nat large multifamily ({LARGE_LBL}) vs. {n_small:,} at small multifamily ({SMALL_LBL}) · "
            f"singular and plural forms combined ·\nadjacent words where neither is a stopword · {scale_note}")
    fig.text(0.02, 0.938, subtitle, fontsize=9, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"size_complaint_words{SUFFIX}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


def main():
    print("MODE:", "purely-residential (--resonly)" if RESIDENTIAL_ONLY else "default")
    c = load()
    if RESIDENTIAL_ONLY:
        c = apply_resonly(c)
    small = c[c["unitsres"].between(*SMALL)]["subject"].str.lower()
    large = c[c["unitsres"] >= LARGE_MIN]["subject"].str.lower()
    print(f"small multifamily ({SMALL_LBL}): {len(small):,} caller subjects")
    print(f"large multifamily ({LARGE_LBL}): {len(large):,} caller subjects")

    z = dirichlet_logodds(uni_counts(large), uni_counts(small), MIN_UNI)
    z.to_csv(OUT / f"size_word_logodds{SUFFIX}.csv", index=False)
    print(f"\nunigram log-odds saved ({len(z)} words)")
    print("most small-typical:", ", ".join(z.head(12)["word"]))
    print("most large-typical:", ", ".join(z.tail(12)["word"][::-1]))

    zb = dirichlet_logodds(bi_counts(large), bi_counts(small), MIN_BI)
    zb.to_csv(OUT / f"size_bigram_logodds{SUFFIX}.csv", index=False)
    print(f"\nbigram log-odds saved ({len(zb)} bigrams)")
    print("most small-typical:", ", ".join(zb.head(10)["word"]))
    print("most large-typical:", ", ".join(zb.tail(10)["word"][::-1]))

    make_figure(z, zb, len(large), len(small))


if __name__ == "__main__":
    main()
