"""
311-caller vs agency-initiated complaint-TEXT contrast for the overview post
(post0_overview_substack.md, the "where complaints come from" paragraph).

Mirrors scripts/size_text_figure.py and scripts/owner_text_figure.py: Monroe,
Colaresi & Quinn (2008) informed-Dirichlet weighted log-odds z-scores of the
complaint SUBJECT text. The two groups here are complaint ORIGIN -- whether the
complaint carries a 311 reference number (citizen-initiated) or not (agency-
initiated: sweeps, inter-agency referrals, enforcement work orders). All
categories are pooled: the contrast is about the whole complaint landscape,
surfacing both the different issue mixes and the different wording of the two
streams.

Groups (scraped BIS subject text, complaints entered 2020-present):
  - 311 calls        = a 311 reference number is present (b.ref_311 non-empty)
  - agency-initiated = no 311 reference number (b.ref_311 empty)
Real subject text only: subject >10 chars and not a bare DOB-internal sweep code
(same filter as size_text_figure.py), which drops pure code-only rows on BOTH
sides so the contrast is between descriptive text, not formatting.

The text utilities (STOP, normalize_token, tokenizer, informed-Dirichlet engine)
and figure styling are lifted verbatim from scripts/size_text_figure.py /
scripts/text_analysis_race.py so this figure stays byte-for-byte comparable to
the size and owner versions; keep them in sync.

Concurrency: a scrape or another reader may hold the DB, so it is opened
READ-ONLY with a busy timeout and lock-retry backoff. Never writes the DB.

Outputs:
  data/analysis/blog_posts/artifacts/origin_complaint_words.png
  data/analysis/risk_models/origin_word_logodds.csv
  data/analysis/risk_models/origin_bigram_logodds.csv
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

OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"

MIN_UNI = 80            # display floor: pooled unigram count (large N, so raised)
MIN_BI = 50             # display floor: pooled bigram count
CALLER_LBL = "311 calls"
AGENCY_LBL = "agency-initiated"

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


def dirichlet_logodds(c_agency, c_caller, min_count):
    """Monroe et al. (2008) informed-Dirichlet log-odds z-scores.
    Positive z = more typical of AGENCY-initiated; negative = 311 calls."""
    vocab = {t for t in set(c_agency) | set(c_caller)
             if c_agency.get(t, 0) + c_caller.get(t, 0) >= min_count}
    nA, nC = sum(c_agency.values()), sum(c_caller.values())
    prior = {t: c_agency.get(t, 0) + c_caller.get(t, 0) for t in vocab}
    a0 = sum(prior.values())
    k = 500 / a0  # prior strength
    rows = []
    for t in vocab:
        pa = prior[t] * k
        lA = np.log((c_agency.get(t, 0) + pa) / (nA + a0 * k - c_agency.get(t, 0) - pa))
        lC = np.log((c_caller.get(t, 0) + pa) / (nC + a0 * k - c_caller.get(t, 0) - pa))
        var = 1 / (c_agency.get(t, 0) + pa) + 1 / (c_caller.get(t, 0) + pa)
        rows.append({"word": t, "z": (lA - lC) / np.sqrt(var),
                     "n_agency": c_agency.get(t, 0), "n_caller": c_caller.get(t, 0)})
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
        except sqlite3.OperationalError as e:
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise last


def load():
    conn = _connect_ro()
    try:
        for attempt in range(6):
            try:
                c = pd.read_sql_query("""
                    SELECT b.subject, b.ref_311
                    FROM open_data o JOIN bis_scrape b USING(complaint_number)
                    WHERE b.subject IS NOT NULL
                      AND substr(o.date_entered,7,4) >= '2020'""", conn)
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() or attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 30))
    finally:
        conn.close()

    c["subject"] = c["subject"].fillna("").astype(str)
    c["is_311"] = c["ref_311"].fillna("").astype(str).str.strip() != ""
    # real descriptive text only: >10 chars and not a bare DOB-internal sweep code
    internal = c["subject"].str.contains(r"\*\*\*|^\d{6,}\s*\(D\d+\)?$", regex=True, na=False)
    c = c[(c["subject"].str.len() > 10) & ~internal]
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


def _word_panel(ax, z, title):
    z = z[~z["word"].isin(DISPLAY_EXCLUDE)]
    z = z[~z["word"].apply(lambda w: any(t in URL_TOKENS for t in w.split()))]
    d = pd.concat([z.nsmallest(10, "z").sort_values("z"),
                   z.nlargest(10, "z").sort_values("z")])
    ax.axvline(0, color=ZERO, lw=1.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ys = np.arange(len(d))
    # color by side: caller (negative) blue, agency (positive) red
    colors = [BLUE if zz < 0 else RED for zz in d["z"]]
    ax.scatter(d["z"], ys, s=60, c=colors, zorder=3, edgecolors=SURFACE, linewidths=1.5)
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


def make_figure(z, zb, n_agency, n_caller):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0),
                             gridspec_kw={"left": 0.04, "right": 0.97, "top": 0.832,
                                          "bottom": 0.06, "hspace": 0.34})
    _word_panel(axes[0], z, "Single words")
    _word_panel(axes[1], zb, "Word pairs")
    for ax in axes:
        ax.text(0.985, 0.03,
                f"right of zero = more typical of agency-initiated inspections\n"
                f"left of zero = more typical of 311-caller complaints",
                transform=ax.transAxes, fontsize=8.8, color=MUTED,
                ha="right", va="bottom", style="italic")
        ax.set_xlabel("informed-Dirichlet log-odds z-score (unitless)", fontsize=9)
    fig.suptitle("Words and word pairs that distinguish 311-caller complaints\nfrom agency-initiated inspections",
                 x=0.02, y=0.987, ha="left", fontsize=12.5, color=INK, weight="semibold")
    subtitle = (
        f"Complaint subject text, 2020-present · {n_agency:,} agency-initiated (no 311 reference) vs "
        f"{n_caller:,} 311-referenced\ncomplaints · singular and plural forms combined · adjacent words where "
        f"neither is a stopword ·\nbare internal codes and place names omitted for display")
    fig.text(0.02, 0.938, subtitle, fontsize=9, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "origin_complaint_words.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


def main():
    c = load()
    caller = c[c["is_311"]]["subject"].str.lower()
    agency = c[~c["is_311"]]["subject"].str.lower()
    print(f"311 calls ({CALLER_LBL}): {len(caller):,} subjects")
    print(f"agency-initiated ({AGENCY_LBL}): {len(agency):,} subjects")

    z = dirichlet_logodds(uni_counts(agency), uni_counts(caller), MIN_UNI)
    z.to_csv(OUT / "origin_word_logodds.csv", index=False)
    print(f"\nunigram log-odds saved ({len(z)} words)")
    print("most 311-typical:", ", ".join(z.head(12)["word"]))
    print("most agency-typical:", ", ".join(z.tail(12)["word"][::-1]))

    zb = dirichlet_logodds(bi_counts(agency), bi_counts(caller), MIN_BI)
    zb.to_csv(OUT / "origin_bigram_logodds.csv", index=False)
    print(f"\nbigram log-odds saved ({len(zb)} bigrams)")
    print("most 311-typical:", ", ".join(zb.head(10)["word"]))
    print("most agency-typical:", ", ".join(zb.tail(10)["word"][::-1]))

    make_figure(z, zb, len(agency), len(caller))


if __name__ == "__main__":
    main()
