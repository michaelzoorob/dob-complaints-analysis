"""
Build the property-level risk-factor dataset.

Universe: all PLUTO residential tax lots (unitsres >= 1, numbldgs >= 1).
Outcomes (2020 - May 2026): DOB complaints, disposition-based violations,
no-access outcomes, and independently recorded ECB violations, aggregated
to the tax lot (BBL).
Risk factors: building characteristics (PLUTO), ownership structure
(ownername regex + STAR owner-occupancy), pre-2020 violation history
(ECB + DOB violations 2010-2019), and census tract (bct2020) with ACS
2023 5-year tract covariates.

Output: data/analysis/property_risk_panel.csv.gz
"""

import json
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import dob_ledger
from analysis_config import make_bbl, BOROUGH_NAME_TO_CODE
from disposition_codes import classify_disposition

OUTPUT_DIR = config.DATA_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
ACS_CACHE = OUTPUT_DIR / "acs_tract_2023.csv"

WINDOW_START = "2020-01-01"   # complaint outcome window
WINDOW_END = "2026-05-31"     # analysis cap: through end of May 2026
HISTORY_START, HISTORY_END = 2010, 2019  # pre-period violation history


# ── 1. PLUTO universe ────────────────────────────────────────────────────

def load_pluto(conn) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT p.borocode, p.borough, p.block, p.lot, p.bldgclass, p.landuse,
               p.numfloors, p.yearbuilt, p.unitsres, p.unitstotal, p.bldgarea,
               p.lotarea, p.assesstot, p.zonedist1, p.ownername, p.numbldgs,
               t.bct2020
        FROM pluto p
        LEFT JOIN pluto_tract t
          ON t.borocode = p.borocode AND t.block = p.block AND t.lot = p.lot
    """, conn)
    for c in ["numfloors", "yearbuilt", "unitsres", "unitstotal",
              "bldgarea", "lotarea", "assesstot", "numbldgs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                     in zip(df["borocode"], df["block"], df["lot"])]
    n0 = len(df)
    df = df[(df["unitsres"] >= 1) & (df["numbldgs"] >= 1) & (df["bbl_key"] != "")]
    df = df.drop_duplicates("bbl_key")
    print(f"PLUTO: {n0:,} lots -> {len(df):,} residential (unitsres>=1, numbldgs>=1)")
    print(f"  tract match: {df['bct2020'].notna().mean():.3%}")
    return df


# ── 2. Complaint aggregates 2020+ ────────────────────────────────────────

CATEGORY_GROUPS = {
    "conversion": r"ILLEGAL CONVERSION|ILLEGAL USE|SRO|HOTEL|ROOMING",
    "construction": r"PERMIT|CONSTRUCTION|DEMOLITION|EXCAVATION|SCAFFOLD|SHAKING|CRANE|AFTER HOURS|SITE CONDITIONS|DEBRIS|FENCE|SIDEWALK SHED|RETAINING WALL",
    "elevator": r"ELEVATOR",
    "boiler_mech": r"BOILER|PLUMBING|GAS|ELECTRIC",
}


def load_complaints(conn) -> pd.DataFrame:
    boro_case = """CASE b.borough
        WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
        WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
        WHEN 'STATEN ISLAND' THEN '5' END"""
    df = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.date_entered, o.disposition_code,
               o.complaint_category,
               {boro_case} AS boro_code, b.block, b.lot,
               b.category_description
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
    """, conn)
    df["entered"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce")
    n0 = len(df)
    df = df[(df["entered"] >= WINDOW_START) & (df["entered"] <= WINDOW_END)]
    print(f"Complaints: {n0:,} scraped w/ block-lot -> {len(df):,} in window "
          f"({df['entered'].min():%Y-%m-%d} to {df['entered'].max():%Y-%m-%d})")

    df["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                     in zip(df["boro_code"], df["block"], df["lot"])]
    df = df[df["bbl_key"] != ""]

    # empirical code -> modal description (bis parser leaves 'Date' for some rows)
    desc = df[~df["category_description"].isin(["Date", None, ""])]
    code_map = (desc.groupby("complaint_category")["category_description"]
                .agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    df["cat_desc"] = df["complaint_category"].map(code_map).fillna(
        df["category_description"].where(df["category_description"] != "Date", ""))

    df["outcome"] = df["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    for g, pat in CATEGORY_GROUPS.items():
        df[f"is_{g}"] = df["cat_desc"].str.contains(pat, regex=True, na=False)

    agg = df.groupby("bbl_key").agg(
        n_complaints=("complaint_number", "size"),
        n_viol_disp=("outcome", lambda s: (s == "violation").sum()),
        n_noviol_disp=("outcome", lambda s: (s == "no_violation").sum()),
        n_no_access=("outcome", lambda s: (s == "no_access").sum()),
        n_conv=("is_conversion", "sum"),
        n_constr=("is_construction", "sum"),
        n_elev=("is_elevator", "sum"),
        n_boiler=("is_boiler_mech", "sum"),
    ).reset_index()
    agg["n_substantive"] = agg["n_viol_disp"] + agg["n_noviol_disp"]
    print(f"  aggregated to {len(agg):,} lots; outcome mix: "
          f"{df['outcome'].value_counts(normalize=True).round(3).to_dict()}")
    return agg


# ── 3. STAR owner-occupancy ──────────────────────────────────────────────

def load_star(conn) -> set:
    star = pd.read_sql_query("SELECT boro, block, lot FROM star_exemptions", conn)
    keys = {make_bbl(b, bl, lt) for b, bl, lt
            in zip(star["boro"], star["block"], star["lot"])}
    keys.discard("")
    print(f"STAR: {len(keys):,} unique exempt lots")
    return keys


# ── 4. Violation history (pre-period) + ECB outcome window ──────────────

def load_violation_history(conn) -> pd.DataFrame:
    ecb = pd.read_sql_query("""
        SELECT boro, block, lot, substr(issue_date,1,4) AS yr
        FROM ecb_violations WHERE length(issue_date) >= 8
    """, conn)
    ecb["yr"] = pd.to_numeric(ecb["yr"], errors="coerce")
    ecb["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                      in zip(ecb["boro"], ecb["block"], ecb["lot"])]
    ecb = ecb[ecb["bbl_key"] != ""]

    hist = (ecb[(ecb["yr"] >= HISTORY_START) & (ecb["yr"] <= HISTORY_END)]
            .groupby("bbl_key").size().rename("n_ecb_hist"))
    curr = (ecb[ecb["yr"] >= 2020].groupby("bbl_key").size().rename("n_ecb_2020on"))

    # DOB-ledger history from BOTH systems (BIS + DOB NOW), deduped across
    # systems on (bbl, issue date, type family) — see dob_ledger.py. The
    # DOB NOW dataset adds ~3% of 2010-19 rows beyond BIS and flips
    # any_prior_viol for ~0.13% of residential lots.
    dob_hist = dob_ledger.counts_by_bbl(conn, HISTORY_START, HISTORY_END,
                                        "n_dobviol_hist")

    out = pd.concat([hist, curr, dob_hist], axis=1).fillna(0).reset_index()
    print(f"History: {len(out):,} lots with any ECB/DOB violation record; "
          f"ECB 2010-19 rows {int(hist.sum()):,}, DOB 2010-19 rows {int(dob_hist.sum()):,}")
    return out


# ── 5. ACS tract covariates ──────────────────────────────────────────────

ACS_VARS = {
    "B01003_001E": "pop",
    "B17001_001E": "pov_denom", "B17001_002E": "pov_num",
    "B19013_001E": "med_income",
    "B25003_001E": "tenure_denom", "B25003_003E": "renter_num",
    "B05002_001E": "fb_denom", "B05002_013E": "fb_num",
    "B03002_001E": "race_denom", "B03002_003E": "nh_white",
    "B03002_004E": "nh_black", "B03002_006E": "nh_asian", "B03002_012E": "hispanic",
    "B25014_001E": "crowd_denom",
    "B25014_005E": "crowd_o1", "B25014_006E": "crowd_o2", "B25014_007E": "crowd_o3",
    "B25014_011E": "crowd_r1", "B25014_012E": "crowd_r2", "B25014_013E": "crowd_r3",
}
COUNTY_TO_BORO = {"061": "1", "005": "2", "047": "3", "081": "4", "085": "5"}


def _census_key() -> str:
    import os
    key = os.environ.get("CENSUS_API_KEY", "")
    if not key:
        envf = Path.home() / "Dropbox/nycpol/ami-affordability-map/.env.local"
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("CENSUS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    return key


def fetch_acs() -> pd.DataFrame:
    if ACS_CACHE.exists():
        print(f"ACS: using cache {ACS_CACHE}")
        return pd.read_csv(ACS_CACHE, dtype={"bct2020": str})
    frames = []
    varlist = ",".join(ACS_VARS)
    key = _census_key()
    for county in COUNTY_TO_BORO:
        url = (f"https://api.census.gov/data/2023/acs/acs5?get={varlist}"
               f"&for=tract:*&in=state:36&in=county:{county}&key={key}")
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        f = pd.DataFrame(data[1:], columns=data[0])
        frames.append(f)
        print(f"  ACS county {county}: {len(f):,} tracts")
    acs = pd.concat(frames)
    acs = acs.rename(columns=ACS_VARS)
    for c in ACS_VARS.values():
        acs[c] = pd.to_numeric(acs[c], errors="coerce")
        acs.loc[acs[c] < 0, c] = np.nan  # census sentinel codes
    acs["bct2020"] = acs["county"].map(COUNTY_TO_BORO) + acs["tract"].astype(str).str.zfill(6)
    acs["tract_poverty"] = acs["pov_num"] / acs["pov_denom"].replace(0, np.nan)
    acs["tract_renter_share"] = acs["renter_num"] / acs["tenure_denom"].replace(0, np.nan)
    acs["tract_foreign_born"] = acs["fb_num"] / acs["fb_denom"].replace(0, np.nan)
    acs["tract_pct_black"] = acs["nh_black"] / acs["race_denom"].replace(0, np.nan)
    acs["tract_pct_hispanic"] = acs["hispanic"] / acs["race_denom"].replace(0, np.nan)
    acs["tract_pct_asian"] = acs["nh_asian"] / acs["race_denom"].replace(0, np.nan)
    crowd = acs[["crowd_o1", "crowd_o2", "crowd_o3", "crowd_r1", "crowd_r2", "crowd_r3"]].sum(axis=1)
    acs["tract_overcrowd"] = crowd / acs["crowd_denom"].replace(0, np.nan)
    acs["tract_log_income"] = np.log(acs["med_income"].where(acs["med_income"] > 0))
    keep = ["bct2020", "pop", "tract_poverty", "tract_renter_share",
            "tract_foreign_born", "tract_pct_black", "tract_pct_hispanic",
            "tract_pct_asian", "tract_overcrowd", "tract_log_income", "med_income"]
    acs = acs[keep]
    acs.to_csv(ACS_CACHE, index=False)
    print(f"ACS: {len(acs):,} tracts cached")
    return acs


# ── 6. Ownership classification ──────────────────────────────────────────

RE_NYCHA = re.compile(r"HOUSING AUTH|N\s*Y\s*C\s*H\s*A", re.I)
RE_GOVT = re.compile(r"CITY OF NEW YORK|NYC |N Y C |DEPT|DEPARTMENT|UNITED STATES|"
                     r"STATE OF NEW YORK|HPD|PARKS|TRANSIT|MTA\b|BOARD OF ED|SCHOOL CONSTR|HHC\b", re.I)
RE_LLC = re.compile(r"\bL\.?\s?L\.?\s?C\.?($|\b)", re.I)
RE_CORP = re.compile(r"\bCORP|\bINC\b|\bLLP\b|\bL\.?P\.?$|\bLP\b|ASSOCIATES|ASSOC\b|REALTY|"
                     r"HOLDING|MANAGEMENT|MGMT|PARTNERS|\bGROUP\b|PROPERTIES|VENTURES|"
                     r"DEVELOPMENT|EQUITIES|CHURCH|CONGREGATION|TEMPLE|SYNAGOGUE|HOUSING DEVELOPMENT|\bHDFC\b|\bCO\.$", re.I)
RE_TRUST = re.compile(r"\bTRUST\b|\bESTATE\b|TRUSTEE|IRREVOCABLE|REVOCABLE|LIVING TR", re.I)


def classify_owner(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return "missing"
    if RE_NYCHA.search(name):
        return "nycha"
    if RE_GOVT.search(name):
        return "govt"
    if RE_LLC.search(name):
        return "llc"
    if RE_TRUST.search(name):
        return "trust_estate"
    if RE_CORP.search(name):
        return "corp_other"
    return "individual"


# ── 7. Assemble ──────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(str(config.DB_PATH), timeout=60)  # busy timeout: live scrape may hold write lock
    conn.execute("PRAGMA busy_timeout=60000;")
    pluto = load_pluto(conn)
    comp = load_complaints(conn)
    star = load_star(conn)
    hist = load_violation_history(conn)
    conn.close()
    acs = fetch_acs()

    matched = comp["bbl_key"].isin(set(pluto["bbl_key"]))
    print(f"\nComplaint->universe match: {matched.mean():.1%} of complaint-lots "
          f"({comp.loc[matched, 'n_complaints'].sum():,} of {comp['n_complaints'].sum():,} complaints "
          f"land on residential lots)")

    df = pluto.merge(comp, on="bbl_key", how="left")
    df = df.merge(hist, on="bbl_key", how="left")
    for c in ["n_complaints", "n_viol_disp", "n_noviol_disp", "n_no_access",
              "n_substantive", "n_conv", "n_constr", "n_elev", "n_boiler",
              "n_ecb_hist", "n_ecb_2020on", "n_dobviol_hist"]:
        df[c] = df[c].fillna(0).astype(int)

    # ownership / tenure structure
    df["owner_type"] = df["ownername"].apply(classify_owner)
    df["is_coop"] = df["bldgclass"].isin(["C6", "C8", "D0", "D4"])
    lotnum = pd.to_numeric(df["lot"], errors="coerce")
    df["is_condo"] = ((lotnum >= 7501) & (lotnum <= 7599)) | df["bldgclass"].str.startswith("R", na=False)
    df["owner_occ_star"] = df["bbl_key"].isin(star)

    # building covariates
    df["age"] = np.where(df["yearbuilt"].between(1800, 2026), 2026 - df["yearbuilt"], np.nan)
    df["prewar"] = (df["yearbuilt"].between(1800, 1939)).astype(int)
    df["mixed_use"] = (df["unitstotal"] > df["unitsres"]).astype(int)
    denom_units = df["unitstotal"].where(df["unitstotal"] >= df["unitsres"], df["unitsres"])
    df["area_per_unit"] = df["bldgarea"] / denom_units.replace(0, np.nan)
    df.loc[df["bldgarea"] <= 0, "area_per_unit"] = np.nan
    df["units_per_floor"] = df["unitsres"] / df["numfloors"].replace(0, np.nan)
    df["mzone"] = df["zonedist1"].astype(str).str.startswith("M").astype(int)

    # relative assessed value per unit within borough x class-letter (assessment
    # ratios differ across tax classes, so only within-type ranks are meaningful)
    df["class_letter"] = df["bldgclass"].astype(str).str[0]
    vpu = df["assesstot"] / df["unitsres"]
    vpu = vpu.where((df["assesstot"] > 0) & (df["unitsres"] > 0))
    df["value_rank"] = vpu.groupby(
        [df["borocode"], df["class_letter"]]).rank(pct=True)

    # size bins (exact small counts, coarse above 10)
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 50, 100, 250, 100000]
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
              "11-15", "16-25", "26-50", "51-100", "101-250", "251+"]
    df["size_bin"] = pd.cut(df["unitsres"], bins=bins, labels=labels).astype(str)

    # history flags
    df["any_prior_viol"] = ((df["n_ecb_hist"] > 0) | (df["n_dobviol_hist"] > 0)).astype(int)

    # outcomes
    df["any_complaint"] = (df["n_complaints"] > 0).astype(int)
    df["any_viol_disp"] = (df["n_viol_disp"] > 0).astype(int)
    df["any_ecb_2020on"] = (df["n_ecb_2020on"] > 0).astype(int)

    df = df.merge(acs, on="bct2020", how="left")
    print(f"ACS merge: {df['tract_poverty'].notna().mean():.1%} of lots have tract poverty")

    out = OUTPUT_DIR / "property_risk_panel.csv.gz"
    df.drop(columns=["ownername", "address"], errors="ignore").to_csv(out, index=False)
    print(f"\nSaved {len(df):,} properties x {df.shape[1]} cols -> {out}")
    print("\nOutcome summary:")
    print(f"  any_complaint 2020-26: {df['any_complaint'].mean():.3f}")
    print(f"  any disposition-violation: {df['any_viol_disp'].mean():.4f}")
    print(f"  any ECB violation 2020on: {df['any_ecb_2020on'].mean():.4f}")
    print(f"  owner_type: {df['owner_type'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  owner_occ_star: {df['owner_occ_star'].mean():.3f} | coop {df['is_coop'].mean():.3f} "
          f"| condo {df['is_condo'].mean():.3f}")
    print(f"  size_bin: {df['size_bin'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
