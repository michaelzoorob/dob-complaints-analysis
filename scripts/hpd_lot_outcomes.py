"""Aggregate the raw HPD downloads (fetch_hpd.py) to the committed extracts the neighborhood
analysis and notebook use:

  data/analysis/hpd_lot_outcomes.csv.gz   per lot (bbl_key): n_hpd_complaints (distinct complaint
                                          ids, every tenant call counts), n_hpd_problems,
                                          n_hpd_emergency, n_hpd_heat, n_hpd_violations, n_hpd_viol_c
  data/analysis/hpd_problems.csv.gz       per non-duplicate problem: bbl_key, year_month, emergency,
                                          major category, outcome

Outcomes are read from HPD's own status text on each closed problem:
  violation      "Violations were issued"
  no_violation   "No violations were issued" / "did not violate" / "Heat was not required"
  no_access      "not able to gain access" / "unable to access" / "unable to complete the inspection"
  corrected      resolved by phone or tenant confirmation, no inspection
  prior_violation, open, other   (excluded from the rates)
Complaint and problem counts include duplicate building-wide reports (each is a separate
tenant's call). The problem-level extract used for the hit rate and no-access rate drops
duplicates (problem_duplicate_flag = Y), since they carry the first complaint's outcome.
"""
import gzip
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analysis_config import make_bbl  # noqa: E402

RAW = ROOT / "data" / "hpd"
OUT = ROOT / "data" / "analysis"


def classify(desc: str) -> str:
    d = (desc if isinstance(desc, str) else "").lower()
    if "not able to gain access" in d or "unable to access" in d or "unable to complete the inspection" in d:
        return "no_access"
    if "no violations were issued" in d or "did not violate" in d or "heat was not required" in d:
        return "no_violation"
    if "violations were previously issued" in d:
        return "prior_violation"
    if "violations were issued" in d:
        return "violation"
    if ("verified that the following conditions were corrected" in d or "advised by a tenant" in d
            or "indicated that the condition was corrected" in d or "confirmed heat and hot water had been restored" in d
            or "verified that the conditions were corrected" in d):
        return "corrected"
    if "still open" in d:
        return "open"
    return "other"


def bbl_key(df, bbl_col="bbl", boro=None, block="block", lot="lot"):
    key = df[bbl_col].astype(str).str.replace(r"\.0$", "", regex=True)
    good = key.str.fullmatch(r"\d{10}")
    if boro is not None:
        fallback = pd.Series([make_bbl(b, bl, lt) for b, bl, lt in zip(df[boro], df[block], df[lot])], index=df.index)
        key = key.where(good, fallback)
    return key.where(key.str.fullmatch(r"\d{10}").fillna(False), "")


BORO_CODE = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4", "STATEN ISLAND": "5"}


def main():
    p = pd.read_csv(RAW / "hpd_problems_2020_2026.csv.gz", dtype=str, low_memory=False)
    print(f"problems: {len(p):,}")
    p["boro_code"] = p["borough"].str.upper().map(BORO_CODE)
    p["bbl_key"] = bbl_key(p, "bbl", boro="boro_code")
    p = p[p["bbl_key"] != ""].copy()
    p["dup"] = p["problem_duplicate_flag"].fillna("N").str.upper().eq("Y")
    p["outcome"] = p["status_description"].map(classify)
    p["emergency"] = p["type"].fillna("").str.upper().str.contains("EMERGENCY").astype(int)
    p["heat"] = p["major_category"].fillna("").str.upper().eq("HEAT/HOT WATER").astype(int)
    p["year_month"] = p["received_date"].str[:7]
    print("outcome mix (non-duplicate):"); print(p[~p["dup"]]["outcome"].value_counts(normalize=True).round(3).to_string())
    nd = p[~p["dup"]]
    lots = p.groupby("bbl_key").agg(
        n_hpd_complaints=("complaint_id", "nunique"),
        n_hpd_problems=("problem_id", "size"),
        n_hpd_emergency=("emergency", "sum"),
        n_hpd_heat=("heat", "sum"),
    ).reset_index()
    v = pd.read_csv(RAW / "hpd_violations_2020_2026.csv.gz", dtype=str, low_memory=False)
    print(f"violations: {len(v):,}")
    v["bbl_key"] = bbl_key(v, "bbl", boro="boroid")
    v = v[v["bbl_key"] != ""].copy()
    v["is_c"] = v["class"].fillna("").str.upper().eq("C").astype(int)
    vl = v.groupby("bbl_key").agg(n_hpd_violations=("violationid", "size"), n_hpd_viol_c=("is_c", "sum")).reset_index()
    lots = lots.merge(vl, on="bbl_key", how="outer").fillna(0)
    for c in lots.columns[1:]:
        lots[c] = lots[c].astype(int)
    lots.to_csv(OUT / "hpd_lot_outcomes.csv.gz", index=False)
    print(f"wrote hpd_lot_outcomes.csv.gz: {len(lots):,} lots; complaints {lots.n_hpd_complaints.sum():,}, violations {lots.n_hpd_violations.sum():,}")
    keep = nd[["bbl_key", "year_month", "emergency", "major_category", "outcome"]].copy()
    keep["major_category"] = keep["major_category"].fillna("").str.upper().str.slice(0, 24)
    keep.to_csv(OUT / "hpd_problems.csv.gz", index=False)
    print(f"wrote hpd_problems.csv.gz: {len(keep):,} problems")


if __name__ == "__main__":
    main()
