"""
Strict vs lenient inspector NARRATIVE contrast for the inspector post
(post_inspector_substack.md).

Mirrors scripts/origin_text_figure.py / size_text_figure.py: Monroe, Colaresi
& Quinn (2008) informed-Dirichlet weighted log-odds z-scores, here of the
INSPECTOR'S OWN comments on substantive inspections (violation or no
violation written), comparing inspectors in the top strictness quintile with
inspectors in the bottom quintile.

Inspectors are nested within enforcement units that work different problems
(elevator inspectors vs Quality of Life conversion teams), so raw quintiles
would contrast units, not strictness. Three adjustments isolate strictness:
  1. quintiles are computed WITHIN each inspector's modal assigned unit
     (units with at least 10 qualifying inspectors);
  2. comment volume is balanced within unit x side: each unit contributes the
     same number of comments to the strict pool as to the lenient pool
     (random downsample, fixed seed), so the pooled contrast cannot reflect
     unit composition;
  3. each inspector contributes at most PER_INSPECTOR_CAP comments, so one
     verbose inspector cannot dominate the vocabulary.

Strictness = inspector's violation rate on substantive outcomes (>=30 cases),
the same measure as scripts/inspector_leniency_analysis.py.

Outputs:
  data/analysis/blog_posts/artifacts/strictness_inspector_words.png
  data/analysis/risk_models/strictness_word_logodds.csv
  data/analysis/risk_models/strictness_bigram_logodds.csv
  data/analysis/risk_models/strictness_noviol_word_logodds.csv   (style-only
    variant: no-violation narratives, both sides)

The text utilities and figure styling are lifted from
scripts/origin_text_figure.py / size_text_figure.py; keep them in sync.
Opens the DB read-only; never writes it.
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
from disposition_codes import classify_disposition

OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"

MIN_CASES = 30           # inspector inclusion floor (matches leniency analysis)
MIN_UNIT_INSP = 10       # unit must have this many qualifying inspectors
PER_INSPECTOR_CAP = 500  # max comments one inspector contributes
MIN_UNI = 80             # display floor: pooled unigram count
MIN_BI = 50              # display floor: pooled bigram count
SEED = 20260706

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


def dirichlet_logodds(c_strict, c_lenient, min_count):
    """Monroe et al. (2008) informed-Dirichlet log-odds z-scores.
    Positive z = more typical of STRICT inspectors; negative = lenient."""
    vocab = {t for t in set(c_strict) | set(c_lenient)
             if c_strict.get(t, 0) + c_lenient.get(t, 0) >= min_count}
    nS, nL = sum(c_strict.values()), sum(c_lenient.values())
    prior = {t: c_strict.get(t, 0) + c_lenient.get(t, 0) for t in vocab}
    a0 = sum(prior.values())
    k = 500 / a0  # prior strength
    rows = []
    for t in vocab:
        pa = prior[t] * k
        lS = np.log((c_strict.get(t, 0) + pa) / (nS + a0 * k - c_strict.get(t, 0) - pa))
        lL = np.log((c_lenient.get(t, 0) + pa) / (nL + a0 * k - c_lenient.get(t, 0) - pa))
        var = 1 / (c_strict.get(t, 0) + pa) + 1 / (c_lenient.get(t, 0) + pa)
        rows.append({"word": t, "z": (lS - lL) / np.sqrt(var),
                     "n_strict": c_strict.get(t, 0), "n_lenient": c_lenient.get(t, 0)})
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
        df = pd.read_sql_query("""
            SELECT o.disposition_code, b.inspector_badge, b.assigned_to, b.comments
            FROM open_data o JOIN bis_scrape b USING(complaint_number)
            WHERE b.inspector_badge IS NOT NULL AND b.inspector_badge != ''
              AND o.disposition_code IS NOT NULL AND o.disposition_code != ''
              AND b.comments IS NOT NULL""", conn)
    finally:
        conn.close()

    df["outcome"] = df["disposition_code"].apply(classify_disposition)
    df = df[df["outcome"].isin(["violation", "no_violation"])].copy()
    df["violation_found"] = (df["outcome"] == "violation").astype(int)
    df["comments"] = df["comments"].astype(str)
    df = df[df["comments"].str.len() > 10]
    df["assigned_to"] = df["assigned_to"].fillna("UNKNOWN").astype(str)
    return df


def assign_quintiles(df):
    """Inspector-level strictness, modal unit, and within-unit quintiles."""
    insp = (df.groupby("inspector_badge")
              .agg(cases=("violation_found", "size"),
                   rate=("violation_found", "mean"),
                   unit=("assigned_to", lambda s: s.mode().iat[0]))
              .reset_index())
    insp = insp[insp["cases"] >= MIN_CASES]
    unit_sizes = insp["unit"].value_counts()
    keep_units = unit_sizes[unit_sizes >= MIN_UNIT_INSP].index
    insp = insp[insp["unit"].isin(keep_units)].copy()
    insp["q"] = insp.groupby("unit")["rate"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    print(f"qualifying inspectors: {len(insp)} across {insp['unit'].nunique()} units")
    for u in sorted(insp["unit"].unique()):
        sub = insp[insp["unit"] == u]
        s5, s1 = sub[sub["q"] == 4], sub[sub["q"] == 0]
        print(f"  {u[:38]:<38} n={len(sub):>3}  bottom-Q rate {s1['rate'].mean():.2f} "
              f"top-Q rate {s5['rate'].mean():.2f}")
    return insp


def balanced_comments(df, insp):
    """Strict and lenient comment pools, balanced within unit, capped per inspector."""
    rng = np.random.default_rng(SEED)
    m = df.merge(insp[["inspector_badge", "unit", "q"]], on="inspector_badge", how="inner")
    m = m[m["q"].isin([0, 4])]

    # cap per-inspector contribution (shuffle once, then take head per inspector)
    m = m.sample(frac=1, random_state=SEED)
    m = m.groupby("inspector_badge", group_keys=False).head(PER_INSPECTOR_CAP)

    strict_pool, lenient_pool = [], []
    for u, g in m.groupby("unit"):
        s = g[g["q"] == 4]["comments"]
        l = g[g["q"] == 0]["comments"]
        n = min(len(s), len(l))
        if n == 0:
            continue
        strict_pool.append(s.sample(n, random_state=SEED))
        lenient_pool.append(l.sample(n, random_state=SEED))
        print(f"  unit {u[:38]:<38} contributes {n:,} comments per side")
    strict = pd.concat(strict_pool).str.lower()
    lenient = pd.concat(lenient_pool).str.lower()
    print(f"balanced pools: {len(strict):,} strict vs {len(lenient):,} lenient comments")
    # comment-length contrast for the post
    print(f"mean comment chars: strict {strict.str.len().mean():.0f}, "
          f"lenient {lenient.str.len().mean():.0f}")
    return strict, lenient, m


# ---- figure styling: lifted from scripts/origin_text_figure.py ----
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
    # color by side: lenient (negative) blue, strict (positive) red
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


def make_figure(z, zb, n_strict, n_lenient, n_insp, n_units):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0),
                             gridspec_kw={"left": 0.04, "right": 0.97, "top": 0.832,
                                          "bottom": 0.06, "hspace": 0.34})
    _word_panel(axes[0], z, "Single words")
    _word_panel(axes[1], zb, "Word pairs")
    for ax in axes:
        ax.text(0.985, 0.03,
                f"right of zero = more typical of top-quintile (strict) inspectors\n"
                f"left of zero = more typical of bottom-quintile (lenient) inspectors",
                transform=ax.transAxes, fontsize=8.8, color=MUTED,
                ha="right", va="bottom", style="italic")
        ax.set_xlabel("informed-Dirichlet log-odds z-score (unitless)", fontsize=9)
    fig.suptitle("What strict and lenient inspectors write in their reports",
                 x=0.02, y=0.987, ha="left", fontsize=12.5, color=INK, weight="semibold")
    subtitle = (
        f"Inspector narrative comments on substantive inspections, 2020 through May 2026 · "
        f"{n_strict:,} comments by top-quintile vs\n{n_lenient:,} by bottom-quintile inspectors · "
        f"quintiles of the violation rate computed within each of {n_units} enforcement units\n"
        f"({n_insp} inspectors with 30+ cases) · comment volume balanced within unit · "
        f"singular and plural forms combined")
    fig.text(0.02, 0.938, subtitle, fontsize=9, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "strictness_inspector_words.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


def main():
    df = load()
    print(f"substantive inspections with comments: {len(df):,}")
    insp = assign_quintiles(df)
    strict, lenient, m = balanced_comments(df, insp)

    z = dirichlet_logodds(uni_counts(strict), uni_counts(lenient), MIN_UNI)
    z.to_csv(OUT / "strictness_word_logodds.csv", index=False)
    print(f"\nunigram log-odds saved ({len(z)} words)")
    print("most lenient-typical:", ", ".join(z.head(12)["word"]))
    print("most strict-typical:", ", ".join(z.tail(12)["word"][::-1]))

    zb = dirichlet_logodds(bi_counts(strict), bi_counts(lenient), MIN_BI)
    zb.to_csv(OUT / "strictness_bigram_logodds.csv", index=False)
    print(f"\nbigram log-odds saved ({len(zb)} bigrams)")
    print("most lenient-typical:", ", ".join(zb.head(10)["word"]))
    print("most strict-typical:", ", ".join(zb.tail(10)["word"][::-1]))

    make_figure(z, zb, len(strict), len(lenient),
                len(insp), insp["unit"].nunique())

    # style-only variant: no-violation narratives, balanced within unit like the
    # main figure, so unit case mix cannot drive the contrast
    mnv = m[m["violation_found"] == 0]
    nv_pools = []
    for u, g in mnv.groupby("unit"):
        s = g[g["q"] == 4]["comments"]
        l = g[g["q"] == 0]["comments"]
        n = min(len(s), len(l))
        if n == 0:
            continue
        nv_pools.append((s.sample(n, random_state=SEED), l.sample(n, random_state=SEED)))
    s_nv = pd.concat([a for a, _ in nv_pools]).str.lower()
    l_nv = pd.concat([b for _, b in nv_pools]).str.lower()
    print(f"\nno-violation-only variant (unit-balanced): {len(s_nv):,} strict vs {len(l_nv):,} lenient")
    z_nv = dirichlet_logodds(uni_counts(s_nv), uni_counts(l_nv), 60)
    z_nv.to_csv(OUT / "strictness_noviol_word_logodds.csv", index=False)
    print("no-viol most lenient-typical:", ", ".join(z_nv.head(10)["word"]))
    print("no-viol most strict-typical:", ", ".join(list(z_nv.tail(10)["word"])[::-1]))

    # support stats quoted in the post: pool sizes, comment lengths, and
    # marker-word shares among no-violation reports (inspectors contributing
    # >= 10 such reports to their pool)
    sup = [dict(metric="pool_comments_per_side", value=len(strict)),
           dict(metric="qualifying_inspectors", value=len(insp)),
           dict(metric="qualifying_units", value=insp["unit"].nunique()),
           dict(metric="pool_len_strict", value=round(strict.str.len().mean(), 1)),
           dict(metric="pool_len_lenient", value=round(lenient.str.len().mean(), 1)),
           dict(metric="noviol_pool_per_side", value=len(s_nv))]
    mnv2 = mnv.copy()
    cnt = mnv2.groupby("inspector_badge").size()
    mnv2 = mnv2[mnv2["inspector_badge"].map(cnt) >= 10]
    sq = mnv2[mnv2["q"] == 4]; lq = mnv2[mnv2["q"] == 0]
    sup.append(dict(metric="noviol_floor10_strict_inspectors",
                    value=sq["inspector_badge"].nunique()))
    sup.append(dict(metric="noviol_floor10_lenient_inspectors",
                    value=lq["inspector_badge"].nunique()))
    for pat, name in [("observed|unsafe", "observed_unsafe"),
                      ("swo|rescind|dated", "swo_rescind_dated")]:
        for pool, side in [(sq, "strict"), (lq, "lenient")]:
            share = pool["comments"].fillna("").str.lower().str.contains(pat).mean()
            sup.append(dict(metric=f"marker_{name}_{side}", value=round(share, 3)))
    pd.DataFrame(sup).to_csv(OUT / "strictness_support_stats.csv", index=False)
    print("support stats saved -> strictness_support_stats.csv")


if __name__ == "__main__":
    main()
