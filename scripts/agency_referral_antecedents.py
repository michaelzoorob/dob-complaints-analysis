"""
Do other agencies' 311 complaints predict DOB agency-originated contacts?

"Cold" DOB agency contacts (no DOB complaint at the property in the prior
730 days) could still be referral cascades from other citizen channels:
an HPD heat complaint, an NYPD noise call, a DEP report. This script
pulls citywide 311 Service Requests (NYC Open Data erm2-nwe9, all
agencies except DOB) and asks three questions.

1. Reclassification: what share of DOB-cold agency contacts had a
   non-DOB 311 complaint at the same BBL in the prior 730 days ("warm
   via another channel"), and does the predicted-Asian gap survive on
   the fully cold remainder?
2. Antecedents: which non-DOB complaint types precede DOB agency
   contacts within 365 days, and how does the antecedent rate compare
   with a within-property placebo window (the 365 days ending 730 days
   before the event)?
3. Prediction: at the property level, do HPD and noise 311 volumes
   predict cold-agency DOB contacts, and does including them attenuate
   the predicted-Asian coefficient?

Outputs -> risk_models/agency_antecedent_estimates.csv + console report;
           data/analysis/agency_events.csv (event-level, with flags);
           data/analysis/sr311_events_coldbbls.csv.gz (Pull A cache);
           data/analysis/sr311_counts_bybbl.csv.gz (Pull B cache).
"""

import gzip
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
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
CACHE_A = config.DATA_DIR / "analysis" / "sr311_events_coldbbls.csv.gz"
CACHE_B = config.DATA_DIR / "analysis" / "sr311_counts_bybbl.csv.gz"
API = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

BUILDING_COVARS = [
    "owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "mixed_use", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
    "any_prior_viol", "geo_nyc_other", "geo_outside_nyc", "geo_unknown",
    "multi_prop_owner",
]
RACE = ["p_black", "p_hispanic", "p_asian"]


def soda(params, retries=5):
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def load_panel_and_events():
    """Rebuild the agency-event frame exactly as complaint_origin_models
    does, with dates and the DOB-cold flag."""
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str}, low_memory=False)
    df["bbl_key"] = df["bbl_key"].astype(str)
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])
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
        SELECT o.complaint_number, o.date_entered, o.complaint_category,
               CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
                    WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
                    WHEN 'STATEN ISLAND' THEN '5' END boro, b.block, b.lot, b.ref_311
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    c["bbl_key"] = [make_bbl(b, bl, l) for b, bl, l in zip(c["boro"], c["block"], c["lot"])]
    c["agency"] = (c["ref_311"].fillna("").astype(str).str.strip() == "").astype(int)
    c["date"] = pd.to_datetime(c["date_entered"], format="%m/%d/%Y", errors="coerce")
    c = c[c["bbl_key"].isin(set(bs["bbl_key"]))].dropna(subset=["date"])

    xw = pd.read_sql_query("SELECT bin, bbl_key FROM bin_bbl_all", conn)
    hist = pd.read_sql_query("""
        SELECT bin, date_entered FROM open_data
        WHERE substr(date_entered,7,4) >= '2018'""", conn)
    conn.close()
    xw = xw.drop_duplicates("bin").set_index("bin")["bbl_key"]
    hist["bbl_key"] = hist["bin"].map(xw)
    hist["date"] = pd.to_datetime(hist["date_entered"], format="%m/%d/%Y", errors="coerce")
    hist = hist.dropna(subset=["bbl_key", "date"])
    hdict = hist.groupby("bbl_key")["date"].apply(sorted).to_dict()

    ag = c[c["agency"] == 1].copy()
    import bisect
    def dob_cold(row):
        dates = hdict.get(row.bbl_key)
        if not dates:
            return 1
        i = bisect.bisect_left(dates, row.date)
        while i > 0:
            gap = (row.date - dates[i - 1]).days
            if gap <= 0:
                i -= 1
                continue
            return int(gap > 730)
        return 1
    ag["dob_cold"] = [dob_cold(r) for r in ag.itertuples()]
    print(f"agency events: {len(ag):,}; DOB-cold {ag['dob_cold'].mean()*100:.1f}%")
    return bs, c, ag


def pull_a(bbls):
    """All non-DOB 311 SRs since 2016-06-01 at the given BBLs."""
    if CACHE_A.exists():
        print(f"Pull A: using cache {CACHE_A}")
        return pd.read_csv(CACHE_A, dtype={"bbl": str})
    rows = []
    bbls = sorted(bbls)
    CH = 70
    for i in range(0, len(bbls), CH):
        chunk = ",".join(f"'{b}'" for b in bbls[i:i + CH])
        got = soda({
            "$select": "created_date,agency,complaint_type,bbl",
            "$where": (f"bbl in({chunk}) AND agency!='DOB' "
                       "AND created_date>='2016-06-01T00:00:00'"),
            "$limit": 50000})
        rows.extend(got)
        if (i // CH) % 20 == 0:
            print(f"  pull A chunk {i//CH + 1}/{(len(bbls)-1)//CH + 1}, rows {len(rows):,}")
        time.sleep(0.12)
    sr = pd.DataFrame(rows)
    sr.to_csv(CACHE_A, index=False)
    print(f"Pull A: {len(sr):,} SRs cached")
    return sr


def pull_b():
    """Citywide per-BBL non-DOB 311 counts, 2020+, for HPD and noise."""
    if CACHE_B.exists():
        print(f"Pull B: using cache {CACHE_B}")
        return pd.read_csv(CACHE_B, dtype={"bbl": str})
    frames = []
    for fam, where in [
            ("hpd", "agency='HPD'"),
            ("noise", "complaint_type like 'Noise%' AND agency!='DOB'")]:
        off = 0
        while True:
            got = soda({
                "$select": "bbl,count(1) as n",
                "$where": f"{where} AND bbl IS NOT NULL AND created_date>='2020-01-01T00:00:00'",
                "$group": "bbl", "$limit": 400000, "$offset": off})
            if not got:
                break
            d = pd.DataFrame(got)
            d["family"] = fam
            frames.append(d)
            off += 400000
            print(f"  pull B {fam}: +{len(d):,} bbl rows")
            if len(d) < 400000:
                break
            time.sleep(0.2)
    b = pd.concat(frames, ignore_index=True)
    b.to_csv(CACHE_B, index=False)
    print(f"Pull B: {len(b):,} rows cached")
    return b


def main():
    bs, c, ag = load_panel_and_events()
    cold = ag[ag["dob_cold"] == 1].copy()
    bbls = sorted(cold["bbl_key"].unique())
    print(f"DOB-cold events {len(cold):,} on {len(bbls):,} properties")

    sr = pull_a(bbls)
    sr["date"] = pd.to_datetime(sr["created_date"], errors="coerce")
    sr = sr.dropna(subset=["date", "bbl"])
    sdict = {}
    for b, grp in sr.groupby("bbl"):
        g = grp.sort_values("date")
        sdict[b] = (list(g["date"]), list(g["complaint_type"]), list(g["agency"]))

    import bisect
    def window_hit(bbl, d0, d1):
        """Any non-DOB SR in (d0, d1]; return (hit, types in window)."""
        e = sdict.get(bbl)
        if not e:
            return 0, []
        dates, types, _ = e
        lo = bisect.bisect_right(dates, d0)
        hi = bisect.bisect_right(dates, d1)
        return int(hi > lo), types[lo:hi]

    D730, D365, D1095 = (pd.Timedelta(days=x) for x in (730, 365, 1095))
    res_rows, ante_types = [], []
    for r in cold.itertuples():
        h730, _ = window_hit(r.bbl_key, r.date - D730, r.date)
        h365, tt = window_hit(r.bbl_key, r.date - D365, r.date)
        hplac, _ = window_hit(r.bbl_key, r.date - D1095, r.date - D730)
        res_rows.append((r.bbl_key, r.date, h730, h365, hplac))
        ante_types.extend(tt)
    ev = pd.DataFrame(res_rows, columns=["bbl_key", "date", "any730", "any365", "placebo365"])

    print("\n== DOB-cold agency contacts and non-DOB 311 antecedents ==")
    print(f"  non-DOB 311 within prior 730d: {ev['any730'].mean()*100:.1f}%")
    print(f"  within prior 365d: {ev['any365'].mean()*100:.1f}%  "
          f"(placebo window 365d ending 730d earlier: {ev['placebo365'].mean()*100:.1f}%)")
    top = pd.Series(ante_types).value_counts().head(12)
    print("  top antecedent complaint types (365d window):")
    for k, v in top.items():
        print(f"    {k:<38} {v:,}")

    grp = ev.merge(bs[["bbl_key", "p_white", "p_asian"]], on="bbl_key")
    for g, p in [("white", "p_white"), ("asian", "p_asian")]:
        d = grp[grp[p] > 0.7]
        print(f"  antecedent share, {g}-classified: 365d {d['any365'].mean()*100:.1f}%  "
              f"placebo {d['placebo365'].mean()*100:.1f}%  (n {len(d):,})")

    # fully cold counts per property
    ev["fully_cold"] = 1 - ev["any730"]
    nfc = ev.groupby("bbl_key")["fully_cold"].sum().rename("n_fully_cold")
    ncold = cold.groupby("bbl_key").size().rename("n_cold_agency")
    bs = bs.merge(nfc, on="bbl_key", how="left").merge(ncold, on="bbl_key", how="left")
    bs[["n_fully_cold", "n_cold_agency"]] = bs[["n_fully_cold", "n_cold_agency"]].fillna(0)

    b311 = pull_b()
    b311["n"] = pd.to_numeric(b311["n"])
    for fam in ["hpd", "noise"]:
        m = b311[b311["family"] == fam].set_index("bbl")["n"]
        bs[f"n_{fam}311"] = bs["bbl_key"].map(m).fillna(0)
        bs[f"log_{fam}311"] = np.log1p(bs[f"n_{fam}311"])
    print(f"\nproperty-level 311 volumes merged: HPD>0 at "
          f"{(bs['n_hpd311']>0).mean()*100:.1f}% of properties, "
          f"noise>0 at {(bs['n_noise311']>0).mean()*100:.1f}%")

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

    print("\n== PPML: fully cold agency contacts (no DOB or other-311 in 730d) ==")
    m = pf.fepois(f"n_fully_cold ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
    collect(m, "ppml_fully_cold")
    t = m.tidy()
    for r in RACE:
        print(f"  {r:<11} {(np.exp(t.loc[r,'Estimate'])-1)*100:+7.1f}%  (p {t.loc[r,'Pr(>|t|)']:.4f})")

    print("== PPML: cold agency contacts, adding other-channel 311 volumes ==")
    m = pf.fepois(f"n_cold_agency ~ {X} + log_hpd311 + log_noise311 | size_bin + bct2020",
                  data=bs, vcov=vcov)
    collect(m, "ppml_cold_with311")
    t = m.tidy()
    for r in RACE + ["log_hpd311", "log_noise311"]:
        print(f"  {r:<13} {(np.exp(t.loc[r,'Estimate'])-1)*100:+7.1f}%  (p {t.loc[r,'Pr(>|t|)']:.4f})")

    out = pd.concat(res, ignore_index=True)
    out["pct_change"] = (np.exp(out["estimate"]) - 1) * 100
    out.to_csv(OUT / "agency_antecedent_estimates.csv", index=False)
    ev.to_csv(config.DATA_DIR / "analysis" / "agency_events.csv", index=False)
    print(f"\nsaved -> {OUT/'agency_antecedent_estimates.csv'} + agency_events.csv")


if __name__ == "__main__":
    main()
