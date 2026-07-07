"""
Complaint origin channel by predicted owner race.

Scraped BIS complaints either carry a 311 service-request number (citizen-
originated via 311) or not (agency-originated: FDNY/HPD referrals,
emergency work orders, OSE sweeps, DOB follow-ups). This script asks
whether the channel mix and the channel-specific volumes differ by
predicted owner race, in the article's standard design.

1. Complaint-level LPM: P(agency-originated) within complaint category,
   size bin, and census tract (composition margin).
2. Property-level PPML: counts of 311-originated and agency-originated
   complaints, 2020 through May 2026 (volume margins), decomposing the
   headline complaint gap into the two channels.

Raw four-group shares and per-100 volumes are printed alongside.
Caveat: agency records are not independent of citizen complaints
(follow-ups and referrals often trace back to earlier 311 activity).

Outputs -> risk_models/origin_tidy_estimates.csv + console report.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"

BUILDING_COVARS = [
    "owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "com_class", "log_bldgarea", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
    "any_prior_viol", "geo_nyc_other", "geo_outside_nyc", "geo_unknown",
    "multi_prop_owner",
]
RACE = ["p_black", "p_hispanic", "p_asian"]


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
    print(f"BISG subsample: {len(bs):,}")

    conn = sqlite3.connect(str(config.DB_PATH))
    c = pd.read_sql_query("""
        SELECT o.complaint_number, o.complaint_category,
               CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
                    WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
                    WHEN 'STATEN ISLAND' THEN '5' END boro, b.block, b.lot,
               b.ref_311, b.category_description
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, l) for b, bl, l in zip(c["boro"], c["block"], c["lot"])]
    c["agency"] = (c["ref_311"].fillna("").astype(str).str.strip() == "").astype(int)
    return bs, c


def main():
    bs, c = load()
    vcov = {"CRV1": "bct2020"}
    X = " + ".join(RACE + BUILDING_COVARS)
    res = []

    def collect(m, name):
        t = m.tidy().reset_index()
        t.columns = [x.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                     for x in t.columns]
        t = t.rename(columns={t.columns[0]: "term"})
        t["model"], t["n"] = name, m._N
        res.append(t)

    keep = ["bbl_key", "size_bin", "bct2020", "p_white"] + RACE + BUILDING_COVARS
    comp = c.merge(bs[keep], on="bbl_key")
    comp["agency100"] = comp["agency"] * 100.0
    conv = comp[comp["complaint_category"] == "45"]

    print("\n== raw agency-originated share of complaints, by classified owner ==")
    for g, p in [("white", "p_white"), ("black", "p_black"),
                 ("hispanic", "p_hispanic"), ("asian", "p_asian")]:
        d = comp[comp[p] > 0.7]
        dc = conv[conv[p] > 0.7]
        print(f"  {g:>9}: all complaints {d['agency'].mean()*100:5.1f}%  "
              f"(n {len(d):,})   conversion {dc['agency'].mean()*100:5.1f}%  (n {len(dc):,})")

    print("\n== complaint-level LPM: P(agency-originated), pp ==")
    m = pf.feols(f"agency100 ~ {X} | complaint_category + size_bin + bct2020",
                 data=comp, vcov=vcov)
    collect(m, "lpm_agency_all")
    t = m.tidy()
    for r in RACE:
        print(f"  all complaints  {r:<11} {t.loc[r,'Estimate']:+6.2f}pp (p {t.loc[r,'Pr(>|t|)']:.4f})")
    m = pf.feols(f"agency100 ~ {X} | size_bin + bct2020", data=conv, vcov=vcov)
    collect(m, "lpm_agency_conv")
    t = m.tidy()
    for r in RACE:
        print(f"  conversion      {r:<11} {t.loc[r,'Estimate']:+6.2f}pp (p {t.loc[r,'Pr(>|t|)']:.4f})")

    # property-level channel volumes
    agg = (comp.groupby("bbl_key")["agency"]
           .agg(n_agency="sum", n_total="count").reset_index())
    agg["n_311"] = agg["n_total"] - agg["n_agency"]
    bs = bs.merge(agg[["bbl_key", "n_agency", "n_311"]], on="bbl_key", how="left")
    bs[["n_agency", "n_311"]] = bs[["n_agency", "n_311"]].fillna(0)

    print("\n== raw channel volumes per 100 properties ==")
    for g, p in [("white", "p_white"), ("black", "p_black"),
                 ("hispanic", "p_hispanic"), ("asian", "p_asian")]:
        d = bs[bs[p] > 0.7]
        print(f"  {g:>9}: 311-originated {d['n_311'].sum()/len(d)*100:5.1f}  "
              f"agency-originated {d['n_agency'].sum()/len(d)*100:5.1f}")

    print("\n== property-level PPML: channel-specific complaint counts ==")
    for outc, label in [("n_311", "311-originated"), ("n_agency", "agency-originated")]:
        m = pf.fepois(f"{outc} ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
        collect(m, f"ppml_{outc}")
        t = m.tidy()
        for r in RACE:
            b, pv = t.loc[r, "Estimate"], t.loc[r, "Pr(>|t|)"]
            print(f"  {label:<18} {r:<11} {(np.exp(b)-1)*100:+7.1f}%  (p {pv:.4f})")

    # cold vs follow-up agency complaints: an agency record is "cold" if the
    # property had no complaint of any kind (open_data, via the BIN crosswalk)
    # in the preceding 730 days. Cold agency contact is the channel that
    # cannot be mechanical follow-through from earlier 311 activity.
    conn = sqlite3.connect(str(config.DB_PATH))
    xw = pd.read_sql_query("SELECT bin, bbl_key FROM bin_bbl_all", conn)
    hist = pd.read_sql_query("""
        SELECT bin, date_entered FROM open_data
        WHERE substr(date_entered,7,4) >= '2018'""", conn)
    conn.close()
    xw = xw.drop_duplicates("bin").set_index("bin")["bbl_key"]
    hist["bbl_key"] = hist["bin"].map(xw)
    hist["date"] = pd.to_datetime(hist["date_entered"], format="%m/%d/%Y", errors="coerce")
    hist = hist.dropna(subset=["bbl_key", "date"])

    ag = comp[comp["agency"] == 1][["bbl_key", "complaint_number"]].copy()
    conn = sqlite3.connect(str(config.DB_PATH))
    dts = pd.read_sql_query("SELECT complaint_number, date_entered FROM open_data", conn)
    conn.close()
    ag = ag.merge(dts, on="complaint_number")
    ag["date"] = pd.to_datetime(ag["date_entered"], format="%m/%d/%Y", errors="coerce")
    ag = ag.dropna(subset=["date"])

    h = hist.groupby("bbl_key")["date"].apply(sorted).to_dict()
    import bisect
    def is_cold(row):
        dates = h.get(row.bbl_key)
        if not dates:
            return 1
        i = bisect.bisect_left(dates, row.date)
        # any strictly earlier complaint within 730 days?
        while i > 0:
            gap = (row.date - dates[i - 1]).days
            if gap <= 0:
                i -= 1
                continue
            return int(gap > 730)
        return 1
    ag["cold"] = [is_cold(r) for r in ag.itertuples()]
    print(f"\nagency complaints: {len(ag):,}; cold share {ag['cold'].mean()*100:.1f}%")

    ncold = ag.groupby("bbl_key")["cold"].sum().rename("n_cold_agency")
    bs = bs.merge(ncold, on="bbl_key", how="left")
    bs["n_cold_agency"] = bs["n_cold_agency"].fillna(0)
    bs["n_warm_agency"] = bs["n_agency"] - bs["n_cold_agency"]

    print("== raw cold/warm agency volumes per 100 properties ==")
    for g, p in [("white", "p_white"), ("black", "p_black"),
                 ("hispanic", "p_hispanic"), ("asian", "p_asian")]:
        d = bs[bs[p] > 0.7]
        print(f"  {g:>9}: cold {d['n_cold_agency'].sum()/len(d)*100:5.1f}  "
              f"warm (follow-up) {d['n_warm_agency'].sum()/len(d)*100:5.1f}")
    print("== property-level PPML: cold vs follow-up agency counts ==")
    for outc, label in [("n_cold_agency", "cold agency"), ("n_warm_agency", "follow-up agency")]:
        m = pf.fepois(f"{outc} ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
        collect(m, f"ppml_{outc}")
        t = m.tidy()
        for r in RACE:
            b, pv = t.loc[r, "Estimate"], t.loc[r, "Pr(>|t|)"]
            print(f"  {label:<18} {r:<11} {(np.exp(b)-1)*100:+7.1f}%  (p {pv:.4f})")

    out = pd.concat(res, ignore_index=True)
    out["pct_change"] = (np.exp(out["estimate"]) - 1) * 100
    out.to_csv(OUT / "origin_tidy_estimates.csv", index=False)
    print(f"\nsaved -> {OUT/'origin_tidy_estimates.csv'}")


if __name__ == "__main__":
    main()
