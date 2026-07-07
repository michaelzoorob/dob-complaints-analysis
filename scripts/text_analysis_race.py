"""
Text analysis for the Asian-owner article supplement.

Question: conditional on a complaint existing (and its category), does
predicted owner race predict the TEXT of the caller's complaint or of the
inspector's report?

Features were chosen after manual review of ~40 stratified random
complaint/report pairs. Caller-subject features (subjects with >10 chars,
excluding DOB-internal sweep codes): length; hedge/suspicion phrases;
people-surveillance phrases; renting mentions; concrete structural
evidence; digits; vehicle mentions; monitoring-duration phrases;
tenant-insider vs neighbor-observer markers; the word "illegal";
operator-transcribed channel; all-caps share; digits. (BIS text
contains no exclamation marks, so that feature is unmeasurable.)
Inspector features: report present; report length (separately within
substantive and no-access outcomes); and, among no-access reports,
active refusal ("denied/refused") versus nobody-home ("no response").

All analyses use 311-referenced complaints only; records without a 311
reference number are agency-originated (referrals, emergency work
orders), not caller text.

Design: inspection-level OLS/LPM of each feature on race probabilities
with building + owner controls, complaint-category, size-bin, and census
tract fixed effects, SEs clustered by tract (same as the article's
per-inspection models).

Also: Monroe-style informed-Dirichlet log-odds of subject words,
conversion complaints at classified (P>0.7) Asian- vs white-owned homes.

Outputs: data/analysis/risk_models/text_tidy_estimates.csv,
         data/analysis/risk_models/conversion_word_logodds.csv,
         data/analysis/risk_models/conversion_bigram_logodds.csv
"""

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl
from disposition_codes import classify_disposition
from build_risk_dataset import CATEGORY_GROUPS

OUT = config.DATA_DIR / "analysis" / "risk_models"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"

BUILDING_COVARS = [
    "owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "com_class", "log_bldgarea", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
    "any_prior_viol", "geo_nyc_other", "geo_outside_nyc", "geo_unknown",
    "multi_prop_owner",
]

PATTERNS = {
    "hedge": r"suspicious|not sure|i think|seems|appears|possibl|maybe|i believe|don'?t know|feels like|some kind of|something going",
    "people_watch": r"many people|lot of people|lots of people|people coming|coming and going|people living|multiple famil|families living|too many people|people enter",
    "renting": r"\brent|airbnb",
    "evidence_struct": r"permit|egress|gas line|electric|plumb|foundation|ceiling|beam\b|joist|\d+\s*(inch|inches|feet|ft)\b|sq\.? ?ft",
    "vehicles": r"\bcars?\b|parking|driveway|\bvans?\b",
    "duration_watch": r"every ?day|all day|all night|for (weeks|months|years)|past (week|month|year)|constant|24/7",
    "tenant_marker": r"i (live|lived|reside)|my (landlord|apartment|unit|lease|bedroom)|i rent\b|i am a tenant|i'?m a tenant",
    "neighbor_marker": r"neighbou?r|next ?door|across the street|behind (my|our)|adjacent (house|home|property)",
    "illegal_word": r"illegal",
    "operator_channel": r"^\s*(caller|customer|complainant) (states|is reporting|reports)",
}

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


# tokens are normalized before the stopword test, so the STOP set must also
# contain the normalized (stemmed) form of every stopword ('this' -> 'thi',
# 'does' -> 'doe'); otherwise stemmed stopwords leak into the vocabularies
STOP |= {normalize_token(w) for w in STOP}


def keep_token(tok: str) -> bool:
    return tok not in STOP


def load():
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str}, low_memory=False)
    df["bbl_key"] = df["bbl_key"].astype(str)
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])
    # commercial-exposure controls (spec 2b+ from owner_commercial_sensitivity.py):
    # com_class (S/K/O storefront/office/mixed class) + log total floor area, replacing
    # the binary mixed_use flag. comm_bin FE omitted -- near-degenerate on this
    # individually-owned <16-unit subsample (~2% commercial exposure).
    _ba = pd.to_numeric(df["bldgarea"], errors="coerce")
    df["log_bldgarea"] = np.log(_ba.where(_ba > 0))
    df["com_class"] = df["bldgclass"].astype(str).str[0].isin(["S", "K", "O"]).astype(int)
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    df["owner_occ_star"] = df["owner_occ_star"].astype(int)
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])]
    bs = df[(df["owner_type"] == "individual") & (df["unitsres"] < 16)
            & df["p_white"].notna()].copy()

    conn = sqlite3.connect(str(config.DB_PATH))
    c = pd.read_sql_query("""
        SELECT o.complaint_number, o.complaint_category,
               CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
                    WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
                    WHEN 'STATEN ISLAND' THEN '5' END boro, b.block, b.lot,
               b.category_description, b.subject, b.comments, b.ref_311,
               o.disposition_code
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, l) for b, bl, l in zip(c["boro"], c["block"], c["lot"])]
    keep = ["bbl_key", "size_bin", "bct2020", "p_black", "p_hispanic", "p_asian",
            "p_white", "sn_asian", "sn_white"] + BUILDING_COVARS
    insp = c.merge(bs[keep], on="bbl_key")
    insp["subject"] = insp["subject"].fillna("").astype(str)
    insp["ref311"] = insp["ref_311"].fillna("").astype(str).str.strip() != ""
    n0 = len(insp)
    insp = insp[insp["ref311"]].copy()
    print(f"restricted to 311-referenced complaints: {len(insp):,} of {n0:,} "
          f"({n0 - len(insp):,} agency-originated records excluded)")
    insp["comments"] = insp["comments"].fillna("").astype(str)
    insp["outcome"] = insp["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    desc = insp[insp["category_description"].notna()
                & ~insp["category_description"].isin(["Date", ""])]
    cmap = (desc.groupby("complaint_category")["category_description"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    insp["cat_desc"] = insp["complaint_category"].map(cmap).fillna("")
    insp["conv"] = insp["cat_desc"].str.contains(CATEGORY_GROUPS["conversion"],
                                                 regex=True, na=False)
    print(f"inspection-level rows on subsample: {len(insp):,}")
    return insp


def build_features(insp):
    s = insp["subject"].str.lower()
    internal = insp["subject"].str.contains(r"\*\*\*|^\d{6,}\s*\(D\d+\)?$", regex=True, na=False)
    insp["has_subject"] = ((insp["subject"].str.len() > 10) & ~internal).astype(float) * 100
    valid = insp["has_subject"] == 100

    insp["log_words"] = np.log(s.str.split().str.len().clip(lower=1))
    for name, pat in PATTERNS.items():
        insp[name] = s.str.contains(pat, regex=True, na=False).astype(float) * 100
    letters = insp["subject"].str.count(r"[A-Za-z]").clip(lower=1)
    insp["caps_share"] = insp["subject"].str.count(r"[A-Z]") / letters * 100
    insp["has_digit"] = s.str.contains(r"\d", regex=True).astype(float) * 100

    cm = insp["comments"].str.lower()
    insp["has_comments"] = (insp["comments"].str.len() > 10).astype(float) * 100
    insp["log_comment_words"] = np.log(cm.str.split().str.len().clip(lower=1))
    insp["denied"] = cm.str.contains("denied|refused", regex=True, na=False).astype(float) * 100
    return insp, valid


RES = []


def collect(m, name, sample):
    t = m.tidy().reset_index()
    t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct") for c in t.columns]
    t = t.rename(columns={"coefficient": "term", "index": "term"})
    t["model"] = name
    t["sample"] = sample
    t["n"] = m._N
    RES.append(t)
    r = t.set_index("term").loc["p_asian"]
    print(f"  {name:<22} p_asian {r['estimate']:+8.3f} (se {r['std_error']:.3f}, p {r['pr(>|t|)']:.3f}, N {m._N:,})")


def run_models(insp, valid):
    X = "p_black + p_hispanic + p_asian + " + " + ".join(BUILDING_COVARS)
    FE = "complaint_category + size_bin + bct2020"
    vcov = {"CRV1": "bct2020"}

    print("\n== caller-subject features (pp unless noted) ==")
    m = pf.feols(f"has_subject ~ {X} | {FE}", data=insp, vcov=vcov)
    collect(m, "has_subject", "all complaints")
    sub = insp[valid]
    m = pf.feols(f"log_words ~ {X} | {FE}", data=sub, vcov=vcov)
    collect(m, "log_words", "with subject")
    for f in list(PATTERNS) + ["caps_share", "has_digit"]:
        m = pf.feols(f"{f} ~ {X} | {FE}", data=sub, vcov=vcov)
        collect(m, f, "with subject")

    print("\n== inspector-report features ==")
    m = pf.feols(f"has_comments ~ {X} | {FE}", data=insp, vcov=vcov)
    collect(m, "has_comments", "all complaints")
    subst = insp[insp["outcome"].isin(["violation", "no_violation"])
                 & (insp["has_comments"] == 100)]
    m = pf.feols(f"log_comment_words ~ {X} | {FE}", data=subst, vcov=vcov)
    collect(m, "log_comment_words_subst", "substantive w/ report")
    na = insp[(insp["outcome"] == "no_access") & (insp["has_comments"] == 100)]
    m = pf.feols(f"log_comment_words ~ {X} | {FE}", data=na, vcov=vcov)
    collect(m, "log_comment_words_noacc", "no-access w/ report")
    m = pf.feols(f"denied ~ {X} | {FE}", data=na, vcov=vcov)
    collect(m, "denied_vs_noresponse", "no-access w/ report")

    res = pd.concat(RES, ignore_index=True)
    res.to_csv(OUT / "text_tidy_estimates.csv", index=False)
    print(f"saved -> {OUT/'text_tidy_estimates.csv'}")


def _dirichlet_logodds(ca, cw, min_count):
    """Monroe et al. (2008) informed-Dirichlet log-odds z-scores."""
    vocab = {t for t in set(ca) | set(cw) if ca.get(t, 0) + cw.get(t, 0) >= min_count}
    na, nw = sum(ca.values()), sum(cw.values())
    prior = {t: ca.get(t, 0) + cw.get(t, 0) for t in vocab}
    a0 = sum(prior.values())
    k = 500 / a0  # prior strength
    rows = []
    for t in vocab:
        pa = prior[t] * k
        la = np.log((ca.get(t, 0) + pa) / (na + a0 * k - ca.get(t, 0) - pa))
        lw = np.log((cw.get(t, 0) + pa) / (nw + a0 * k - cw.get(t, 0) - pa))
        var = 1 / (ca.get(t, 0) + pa) + 1 / (cw.get(t, 0) + pa)
        rows.append({"word": t, "z": (la - lw) / np.sqrt(var),
                     "n_asian": ca.get(t, 0), "n_white": cw.get(t, 0)})
    return pd.DataFrame(rows).sort_values("z")


def word_logodds(insp, valid):
    """Unigram and bigram log-odds, conversion subjects, classified
    Asian- vs white-owned. Restricted to complaints with a 311 reference
    number, which drops agency-originated records (FDNY referrals,
    emergency work orders) that are not citizen complaints. The token
    sequence "stop work order" is merged into one phrase so its fragments
    do not surface as spurious bigrams. Bigrams are formed on the raw
    token stream (plural-normalized) and kept only when neither token is
    a stopword, so pairs like "living room" and "people living" survive
    while stopword-adjacent artifacts do not."""
    conv = insp[valid & insp["conv"]]
    a = conv[conv["p_asian"] > 0.7]["subject"].str.lower()
    w = conv[conv["p_white"] > 0.7]["subject"].str.lower()

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

    z = _dirichlet_logodds(uni_counts(a), uni_counts(w), 40)
    z.to_csv(OUT / "conversion_word_logodds.csv", index=False)
    print(f"\nunigram log-odds saved ({len(z)} words; {len(a):,} vs {len(w):,} subjects)")
    print("most white-owned-typical:", ", ".join(z.head(12)["word"]))
    print("most asian-owned-typical:", ", ".join(z.tail(12)["word"][::-1]))

    zb = _dirichlet_logodds(bi_counts(a), bi_counts(w), 30)
    zb.to_csv(OUT / "conversion_bigram_logodds.csv", index=False)
    print(f"bigram log-odds saved ({len(zb)} bigrams)")
    print("most white-owned-typical:", ", ".join(zb.head(12)["word"]))
    print("most asian-owned-typical:", ", ".join(zb.tail(12)["word"][::-1]))

    # concordance check: what does "living" collocate with on each side?
    for name, series in [("asian", a), ("white", w)]:
        cnt = Counter()
        for t in series:
            tk = tokens(t)
            for i, tok in enumerate(tk):
                if tok == "living":
                    if i > 0:
                        cnt[f"{tk[i-1]} living"] += 1
                    if i < len(tk) - 1:
                        cnt[f"living {tk[i+1]}"] += 1
        print(f"'living' collocates ({name}):", cnt.most_common(8))


if __name__ == "__main__":
    insp = load()
    insp, valid = build_features(insp)
    run_models(insp, valid)
    word_logodds(insp, valid)
