"""Every number quoted in the locked-door post, written to one CSV.

Closes the post's ad-hoc gaps: the narrative phrase-match shares (the exact
regexes live here, as the methodology promises), the owner-occupancy and
small-house LPMs with confidence intervals, the single-factor R-squared
shares, the tract correlations (with the ACS-complete filter documented),
borough rates, and the raw STAR no-access rates.

Sample construction mirrors locked_door_figure.py: every scraped complaint
2020 through May 2026 whose disposition classifies as violation,
no-violation, or no-access.

Output: data/analysis/risk_models/locked_door_stats.csv
"""
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
from disposition_codes import classify_disposition  # noqa: E402

# The narrative patterns behind "68% posted / 54% silence / 15% refusal".
PAT_POSTED = re.compile(r"LS4|LS-4|LS 4|POSTED", re.I)
PAT_SILENCE = re.compile(r"NO RESPONSE|NO ANSWER|NO ONE|NOBODY|NO TENANT|UNANSWERED", re.I)
PAT_REFUSAL = re.compile(r"REFUS|DENIED|WOULD NOT ALLOW", re.I)

CONV = {"45", "4G", "4W"}
BGRP = [("B", "2-family house"), ("A", "1-family house"), ("C", "Walk-up apartments"),
        ("D", "Elevator building")]


def main() -> None:
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    bc = ("CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2' "
          "WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' END")
    df = pd.read_sql_query(f"""
        SELECT o.disposition_code, o.complaint_category, b.comments, b.borough,
               b.inspector_badge, b.assigned_to, {bc} AS borocode, b.block, b.lot,
               p.bldgclass
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        LEFT JOIN pluto p ON p.borocode = {bc} AND p.block = b.block AND p.lot = b.lot
        WHERE o.disposition_code IS NOT NULL AND o.disposition_code != ''""", conn)
    conn.close()
    df["outcome"] = df["disposition_code"].apply(classify_disposition)
    d = df[df["outcome"].isin(["violation", "no_violation", "no_access"])].copy()
    d["na"] = (d["outcome"] == "no_access").astype(int)
    d["bletter"] = d["bldgclass"].astype(str).str[0].where(d["bldgclass"].notna())

    rows = []

    def add(metric, value, se=None, n=None):
        rows.append(dict(metric=metric, value=value, se=se, n=n))
        print(f"{metric:<48} {value}")

    add("sample_inspections", len(d))
    add("share_no_access", round(d["na"].mean(), 4))
    add("n_no_access", int(d["na"].sum()))
    conv = d[d["complaint_category"] == "45"]
    add("conversion_no_access_share", round(conv["na"].mean(), 4), n=len(conv))

    # narrative phrase shares among no-access reports
    na = d[d["na"] == 1].copy()
    txt = na["comments"].fillna("")
    posted = txt.str.contains(PAT_POSTED)
    silence = txt.str.contains(PAT_SILENCE)
    refusal = txt.str.contains(PAT_REFUSAL)
    add("narr_share_posted", round(posted.mean(), 4), n=len(na))
    add("narr_share_silence", round(silence.mean(), 4), n=len(na))
    add("narr_share_refusal", round(refusal.mean(), 4), n=len(na))
    cause = silence | refusal
    add("narr_share_any_cause", round(cause.mean(), 4), n=len(na))
    add("narr_refusal_share_of_caused", round((refusal & cause).sum() / cause.sum(), 4))
    add("narr_silence_share_of_caused", round((silence & ~refusal & cause).sum() / cause.sum(), 4))

    # borough no-access rates
    for b, r in d.groupby("borough")["na"].mean().items():
        add(f"no_access_share_{b.title().replace(' ', '_')}", round(r, 4))

    # panel join for building class, STAR, tract demographics
    panel = pd.read_csv(
        config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz",
        usecols=["bbl_key", "class_letter", "owner_occ_star", "unitsres", "bct2020",
                 "tract_foreign_born", "tract_renter_share", "tract_pct_black",
                 "tract_poverty", "tract_log_income"],
        dtype={"bbl_key": str})
    d["block_k"] = pd.to_numeric(d["block"], errors="coerce").astype("Int64").astype(str)
    d["lot_k"] = pd.to_numeric(d["lot"], errors="coerce").astype("Int64").astype(str)
    d["bbl_key"] = (d["borocode"].fillna("")
                    + d["block_k"].str.zfill(5) + d["lot_k"].str.zfill(4))
    m = d.merge(panel, on="bbl_key", how="inner")
    add("matched_to_panel", len(m))
    add("panel_match_share", round(len(m) / len(d), 4))

    # building-class no-access gradient (all and excluding conversion categories)
    excl = ~m["complaint_category"].isin(CONV)
    for letter, lab in BGRP:
        s = m[m["class_letter"] == letter]
        add(f"class_{letter}_no_access", round(s["na"].mean(), 4), n=len(s))
        s2 = s[~s["complaint_category"].isin(CONV)]
        add(f"class_{letter}_no_access_exconv", round(s2["na"].mean(), 4), n=len(s2))

    # raw STAR rates and the two LPMs (category FE, tract-clustered)
    add("star_no_access_raw", round(m.loc[m.owner_occ_star == 1, "na"].mean(), 4))
    add("absentee_no_access_raw", round(m.loc[m.owner_occ_star == 0, "na"].mean(), 4))
    m["na100"] = m["na"] * 100
    m["small_house"] = (m["class_letter"].isin(["A", "B"])
                        & (m["unitsres"] <= 3)).astype(int)
    lpm = m.dropna(subset=["bct2020"]).copy()
    for term in ("owner_occ_star", "small_house"):
        fit = pf.feols(f"na100 ~ owner_occ_star + small_house | complaint_category",
                       data=lpm, vcov={"CRV1": "bct2020"})
        b, se = float(fit.coef()[term]), float(fit.se()[term])
        add(f"lpm_{term}_pp", round(b, 3), se=round(se, 3), n=int(fit._N))

    # single-factor R2 shares of the no-access indicator
    def r2(groups):
        mu = d["na"].mean()
        gm = d.groupby(groups)["na"].transform("mean")
        return 1 - ((d["na"] - gm) ** 2).sum() / ((d["na"] - mu) ** 2).sum()

    add("r2_category", round(r2(d["complaint_category"].fillna("UNK")), 4))
    add("r2_assigned_unit", round(r2(d["assigned_to"].fillna("UNK")), 4))
    badge = d["inspector_badge"].fillna("")
    counts = badge.value_counts()
    d["badge30"] = badge.where(badge.map(counts) >= 30, "SMALL")
    add("r2_inspector_30plus", round(r2(d["badge30"]), 4))
    mu = d.loc[d["bletter"].notna(), "na"].mean()
    sub = d[d["bletter"].notna()]
    gm = sub.groupby("bletter")["na"].transform("mean")
    add("r2_building_class",
        round(1 - ((sub.na - gm) ** 2).sum() / ((sub.na - mu) ** 2).sum(), 4), n=len(sub))

    # tract correlations among ACS-complete tracts with >= 10 matched inspections
    tr = (m.dropna(subset=["tract_log_income", "tract_foreign_born",
                           "tract_renter_share", "tract_pct_black", "tract_poverty"])
            .groupby("bct2020")
            .agg(na=("na", "mean"), n=("na", "size"),
                 fb=("tract_foreign_born", "first"),
                 renter=("tract_renter_share", "first"),
                 black=("tract_pct_black", "first"),
                 pov=("tract_poverty", "first")))
    tr = tr[tr.n >= 10]
    add("tracts_10plus_acs_complete", len(tr))
    for col, name in [("fb", "foreign_born"), ("renter", "renter_share"),
                      ("black", "pct_black"), ("pov", "poverty")]:
        add(f"tract_corr_{name}", round(tr["na"].corr(tr[col]), 3))

    # inspector-level no-access spread (v/nv/na denominator, 30+ cases)
    g = d.groupby("inspector_badge").agg(n=("na", "size"), na_rate=("na", "mean"))
    g = g[g.n >= 30]
    add("inspector_median_no_access", round(g["na_rate"].median(), 4), n=len(g))
    add("inspector_p90_no_access", round(g["na_rate"].quantile(0.9), 4))

    out = config.DATA_DIR / "analysis" / "risk_models" / "locked_door_stats.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
