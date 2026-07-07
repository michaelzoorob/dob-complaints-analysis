"""
Owner-occupied vs. LLC / out-of-town-owned complaint-TEXT contrast.

Sibling of scripts/size_text_figure.py (the small- vs. large-multifamily
figure): identical Monroe, Colaresi & Quinn (2008) informed-Dirichlet weighted
log-odds z-scores of callers' complaint subjects, same tokenizer, same house
style. Here the two groups are OWNER TYPE among small residential buildings
(<6 residential units, where owner-occupancy is a coherent category), and ALL
complaint categories are pooled -- the contrast is about the whole complaint
landscape at owner-occupied vs. absentee-owned homes, not one category.

Groups (mutually exclusive; a building cannot be both owner-occupied and
absentee, so STAR owner-occupancy takes priority):
  A -- Owner-occupied : owner_occ_star == True (STAR primary-residence
       exemption; the exact project definition in build_risk_dataset.py).
  B -- LLC / outside NYC : NOT owner-occupied AND
       (owner_type == 'llc'  OR  owner_geo == 'outside_nyc').
       owner_type comes from the RE_LLC name regex (build_risk_dataset.py);
       owner_geo is built in the panel from the CLEANED owner_zip_namematch.csv
       crosswalk (add_owner_characteristics.py), so 'outside_nyc' already
       encodes the robust NYC-ZIP + owner_state test -- no re-derivation.

311-referenced complaints only (citizen text, not agency referrals); real
caller text only (subject >10 chars, non-internal).

The text utilities (STOP, normalize_token, tokenizer, informed-Dirichlet
engine) and the plotting machinery are lifted verbatim from
scripts/size_text_figure.py so this figure stays byte-for-byte comparable to
its sibling; keep them in sync.

Concurrency: a live scrape may be writing to the DB, so the database is opened
READ-ONLY with a busy timeout and lock-retry backoff. This script never writes
to the database.

Run `python3 scripts/owner_text_figure.py` for the default figure, or with
`--resonly` to hold building use constant (purely residential small buildings
only -- no non-residential units, residential building class A/B/C/D); that
variant writes the *_resonly artifacts below and isolates the ownership contrast
from the store/restaurant/mixed-use vocabulary of mixed-use buildings.

Outputs (default / --resonly):
  data/analysis/blog_posts/artifacts/owner_occ_vs_llc[_resonly]_complaint_words.png
  data/analysis/blog_posts/artifacts/owner_occ_vs_llc[_resonly]_word_logodds.csv
  data/analysis/blog_posts/artifacts/owner_occ_vs_llc[_resonly]_bigram_logodds.csv
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

ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"

# Run with --resonly to hold BUILDING USE constant: restrict to purely
# residential small buildings (no non-residential units, residential building
# class A/B/C/D -- the canonical spec-3 sample), so the surviving contrast is
# ownership, not the store/restaurant/mixed-use vocabulary that rides along with
# absentee owners. Writes a parallel set of *_resonly artifacts; the default run
# is unchanged so the two figures can be compared side by side.
RESIDENTIAL_ONLY = "--resonly" in sys.argv
SUFFIX = "_resonly" if RESIDENTIAL_ONLY else ""

# ---- owner groups (small residential buildings only) ----
UNITS = (1, 5)          # <6 residential units: single-family + small multifamily
MIN_UNI = 60            # display floor: pooled unigram count
MIN_BI = 40             # display floor: pooled bigram count
OCC_LBL = "owner-occupied"
LLC_LBL = "LLC / out-of-town-owned"

# ---- purely-residential filter (only used with --resonly) ----
# Canonical spec-3 res-only sample (data/analysis/owner_commercial_sensitivity.md):
# unitstotal == unitsres (no non-residential units) AND a residential building
# class. unitstotal, unitsres, and bldgclass all come from the panel -- no
# external comarea pull, so this is byte-for-byte the sibling size figure's filter.
RES_CLASSES = ("A", "B", "C", "D")   # 1-fam / 2-fam / walk-up / elevator apts
# (excludes mixed-use S*, store K*, office O*, and condo R* -- see report)

# ---- text utilities: lifted from scripts/size_text_figure.py; keep in sync ----
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
# normalized (stemmed) form of every stopword (see size_text_figure.py).
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


def dirichlet_logodds(c_llc, c_occ, min_count):
    """Monroe et al. (2008) informed-Dirichlet log-odds z-scores.
    Positive z = more typical at LLC / out-of-town-owned; negative = owner-occupied."""
    vocab = {t for t in set(c_llc) | set(c_occ)
             if c_llc.get(t, 0) + c_occ.get(t, 0) >= min_count}
    nL, nO = sum(c_llc.values()), sum(c_occ.values())
    prior = {t: c_llc.get(t, 0) + c_occ.get(t, 0) for t in vocab}
    a0 = sum(prior.values())
    k = 500 / a0  # prior strength
    rows = []
    for t in vocab:
        pa = prior[t] * k
        lL = np.log((c_llc.get(t, 0) + pa) / (nL + a0 * k - c_llc.get(t, 0) - pa))
        lO = np.log((c_occ.get(t, 0) + pa) / (nO + a0 * k - c_occ.get(t, 0) - pa))
        var = 1 / (c_llc.get(t, 0) + pa) + 1 / (c_occ.get(t, 0) + pa)
        rows.append({"word": t, "z": (lL - lO) / np.sqrt(var),
                     "n_llc_outside": c_llc.get(t, 0), "n_owner_occ": c_occ.get(t, 0)})
    return pd.DataFrame(rows).sort_values("z")


def _connect_ro(retries=6):
    """Open the DB read-only with a busy timeout; retry with backoff on lock."""
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


def apply_resonly(c):
    """Restrict to PURELY RESIDENTIAL small buildings so building use is held
    constant: no non-residential units (unitstotal == unitsres) AND a residential
    building class (bldgclass first letter in A/B/C/D). This is the canonical
    spec-3 res-only sample (data/analysis/owner_commercial_sensitivity.md) and is
    identical to the sibling size figure's filter. Prints the buildings /
    subjects removed."""
    n0_bld = c["bbl_key"].nunique()
    n0_occ = int(c["is_occ"].sum())
    n0_llc = int(c["is_llc_out"].sum())

    cls0 = c["bldgclass"].astype(str).str[0]
    pure = (c["unitstotal"] == c["unitsres"]) & cls0.isin(RES_CLASSES)
    c = c[pure].copy()

    rule = "unitstotal==unitsres AND bldgclass in A/B/C/D"
    print(f"purely-residential filter: {rule}")
    print(f"  buildings (distinct BBL): {n0_bld:,} -> {c['bbl_key'].nunique():,} "
          f"(removed {n0_bld - c['bbl_key'].nunique():,})")
    print(f"  owner-occ subjects: {n0_occ:,} -> {int(c['is_occ'].sum()):,} "
          f"(removed {n0_occ - int(c['is_occ'].sum()):,})")
    print(f"  LLC/outside subjects: {n0_llc:,} -> {int(c['is_llc_out'].sum()):,} "
          f"(removed {n0_llc - int(c['is_llc_out'].sum()):,})")
    return c


def load():
    panel = pd.read_csv(PANEL, dtype={"bbl_key": str}, low_memory=False,
                        usecols=["bbl_key", "unitsres", "unitstotal", "bldgclass",
                                 "owner_type", "owner_occ_star", "owner_geo"])
    panel = panel.dropna(subset=["unitsres"])
    panel = panel[panel["unitsres"].between(*UNITS)]

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

    # mutually-exclusive owner groups (STAR owner-occupancy takes priority)
    c["is_occ"] = c["owner_occ_star"].astype(bool)
    c["is_llc_out"] = (~c["is_occ"]) & (
        (c["owner_type"] == "llc") | (c["owner_geo"] == "outside_nyc"))
    return c


# ---- figure styling: lifted from scripts/size_text_figure.py ----
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"

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
# Normalized single-token place names (boroughs/neighborhoods) that should never
# label a point. Tokens are stemmed (queens -> queen) before display, so the
# exclude set must be stemmed too -- otherwise "queen" slips past a set holding
# "queens". Extends the sibling's DISPLAY_EXCLUDE to catch place names that
# surface once building-use terms are removed (e.g. flushing, queen, and
# place-name bigrams such as "queen block"), honoring the subtitle's promise.
PLACE_TOKENS = {normalize_token(w) for w in
                {"york", "staten", "island", "brooklyn", "queens", "bronx",
                 "manhattan", "flushing", "macon"}}


def _word_panel(ax, z, title):
    z = z[~z["word"].isin(DISPLAY_EXCLUDE)]
    z = z[~z["word"].apply(
        lambda w: any(t in URL_TOKENS or t in PLACE_TOKENS for t in w.split()))]
    d = pd.concat([z.nsmallest(10, "z").sort_values("z"),
                   z.nlargest(10, "z").sort_values("z")])
    ax.axvline(0, color=ZERO, lw=1.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ys = np.arange(len(d))
    ax.scatter(d["z"], ys, s=60, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    for y, (_, row) in zip(ys, d.iterrows()):
        ha = "right" if row["z"] < 0 else "left"
        off = -7 if row["z"] < 0 else 7
        ax.annotate(row["word"], (row["z"], y), textcoords="offset points",
                    xytext=(off, 0), va="center", ha=ha, fontsize=9, color=INK)
    ax.set_yticks([])
    zmin, zmax = float(d["z"].min()), float(d["z"].max())
    span = zmax - zmin
    ax.set_xlim(zmin - 0.34 * span, zmax + 0.34 * span)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _wrap(fig, text, maxfrac=0.985, fontsize=9):
    """Greedily wrap `text` into lines each <= maxfrac of the figure width,
    measured against the real renderer so the subtitle never clips."""
    fw = fig.get_window_extent().width

    def w(s):
        t = fig.text(0.02, 0.5, s, fontsize=fontsize)
        fig.canvas.draw()
        x1 = t.get_window_extent().x1
        t.remove()
        return x1

    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if not cur or w(trial) / fw <= maxfrac:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def make_figure(z, zb, n_llc, n_occ):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0),
                             gridspec_kw={"left": 0.04, "right": 0.97, "top": 0.832,
                                          "bottom": 0.06, "hspace": 0.34})
    _word_panel(axes[0], z, "Single words")
    _word_panel(axes[1], zb, "Word pairs")
    for ax in axes:
        ax.text(0.985, 0.03,
                f"right of zero = more typical at {LLC_LBL}\n"
                f"left of zero = more typical at {OCC_LBL}",
                transform=ax.transAxes, fontsize=8.8, color=MUTED,
                ha="right", va="bottom", style="italic")
        ax.set_xlabel("informed-Dirichlet log-odds z-score (unitless)", fontsize=9)
    fig.suptitle("Words and word pairs that distinguish complaints at owner-occupied\nvs. LLC / out-of-town-owned buildings",
                 x=0.02, y=0.987, ha="left", fontsize=12.5, color=INK, weight="semibold")
    if RESIDENTIAL_ONLY:
        # purely-residential scope note; wrapped dynamically so it never clips
        subtitle = _wrap(fig,
            f"Callers' own complaint text (not inspector reports) · 311-referenced complaints, 2020-present · "
            f"purely residential (residential building class, no non-residential units), <6 units · {n_llc:,} subjects "
            f"at {LLC_LBL} vs. {n_occ:,} at {OCC_LBL} · singular and plural forms combined · adjacent words where "
            f"neither is a stopword · URL fragments and place names omitted for display")
    else:
        subtitle = (
            f"Callers' own complaint text (not inspector reports) · 311-referenced complaints, 2020-present · <6-unit residential\n"
            f"buildings · {n_llc:,} subjects at {LLC_LBL} vs. {n_occ:,} at {OCC_LBL} · "
            f"singular and plural forms combined ·\nadjacent words where neither is a stopword · "
            f"URL fragments and place names omitted for display")
    fig.text(0.02, 0.938, subtitle, fontsize=9, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / f"owner_occ_vs_llc{SUFFIX}_complaint_words.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


def main():
    print("MODE:", "purely-residential (--resonly)" if RESIDENTIAL_ONLY else "default")
    c = load()
    if RESIDENTIAL_ONLY:
        c = apply_resonly(c)
    occ = c[c["is_occ"]]["subject"].str.lower()
    llc = c[c["is_llc_out"]]["subject"].str.lower()
    print(f"owner-occupied ({OCC_LBL}): {len(occ):,} caller subjects")
    print(f"LLC / outside NYC ({LLC_LBL}): {len(llc):,} caller subjects")

    ART.mkdir(parents=True, exist_ok=True)

    z = dirichlet_logodds(uni_counts(llc), uni_counts(occ), MIN_UNI)
    z.to_csv(ART / f"owner_occ_vs_llc{SUFFIX}_word_logodds.csv", index=False)
    print(f"\nunigram log-odds saved ({len(z)} words)")
    print("most owner-occupied:", ", ".join(z.head(12)["word"]))
    print("most LLC/outside:", ", ".join(z.tail(12)["word"][::-1]))

    zb = dirichlet_logodds(bi_counts(llc), bi_counts(occ), MIN_BI)
    zb.to_csv(ART / f"owner_occ_vs_llc{SUFFIX}_bigram_logodds.csv", index=False)
    print(f"\nbigram log-odds saved ({len(zb)} bigrams)")
    print("most owner-occupied:", ", ".join(zb.head(10)["word"]))
    print("most LLC/outside:", ", ".join(zb.tail(10)["word"][::-1]))

    make_figure(z, zb, len(llc), len(occ))


if __name__ == "__main__":
    main()
