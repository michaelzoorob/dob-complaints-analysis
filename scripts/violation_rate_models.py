"""
Reduced-form citation outcomes: administrative violations per year.

Outcome = counts of actually-cited violations 2020 - Apr 2026 from the
city's administrative ledgers, independent of the complaint pipeline:
  - n_ecb_2020on  : ECB/OATH violations (penalty-bearing, DOB-issued)
  - n_dobviol_2020on : all DOB violations (incl. periodic/proactive)
Window = 75 months (6.25 years); constant across lots, so PPML IRRs are
unaffected and reported baselines are annualized.

Specs mirror the main study: exact unit-count size bins + census tract FE,
SEs clustered by tract; base covariates and owner-augmented variant.

Output: data/analysis/risk_models/citation_tidy_estimates.csv
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
YEARS = 6.25  # Jan 2020 - Apr 2026

BUILDING_COVARS = [
    "llc", "corp_other", "trust_estate", "nycha", "govt",
    "owner_occ_star", "is_coop", "is_condo",
    "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "mixed_use", "mzone", "multi_bldg",
    "log2_area_per_unit", "value_rank", "any_prior_viol",
]
OWNER_COVARS = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner"]


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str})
    df["bbl_key"] = df["bbl_key"].astype(str)
    for t in ["llc", "corp_other", "trust_estate", "nycha", "govt"]:
        df[t] = (df["owner_type"] == t).astype(int)
    df = df[df["owner_type"] != "missing"]
    for b in ["owner_occ_star", "is_coop", "is_condo"]:
        df[b] = df[b].astype(int)
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

    # fresh join: DOB violations issued 2020+
    conn = sqlite3.connect(str(config.DB_PATH))
    dv = pd.read_sql_query("""
        SELECT boro, block, lot, COUNT(*) AS n_dobviol_2020on
        FROM dob_violations
        WHERE length(issue_date) = 8
          AND substr(issue_date,1,4) BETWEEN '2020' AND '2026'
        GROUP BY boro, block, lot
    """, conn)
    conn.close()
    dv["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                     in zip(dv["boro"], dv["block"], dv["lot"])]
    dv = dv[dv["bbl_key"] != ""].groupby("bbl_key", as_index=False)["n_dobviol_2020on"].sum()
    df = df.merge(dv, on="bbl_key", how="left")
    df["n_dobviol_2020on"] = df["n_dobviol_2020on"].fillna(0).astype(int)
    df["any_dobviol100"] = (df["n_dobviol_2020on"] > 0).astype(int) * 100.0
    df["anyecb100"] = df["any_ecb_2020on"].astype(int) * 100.0

    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])].copy()
    print(f"Sample: {len(df):,}")
    print(f"ECB 2020-26: mean {df['n_ecb_2020on'].mean():.4f}/lot "
          f"({df['n_ecb_2020on'].mean()/YEARS*100:.2f} per 100 lots/yr), "
          f"any: {(df['n_ecb_2020on']>0).mean():.4f}")
    print(f"DOB 2020-26: mean {df['n_dobviol_2020on'].mean():.4f}/lot "
          f"({df['n_dobviol_2020on'].mean()/YEARS*100:.2f} per 100 lots/yr), "
          f"any: {(df['n_dobviol_2020on']>0).mean():.4f}")
    return df


RESULTS = []


def collect(model, name, outcome):
    t = model.tidy().reset_index()
    t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                 for c in t.columns]
    if "coefficient" in t.columns:
        t = t.rename(columns={"coefficient": "term"})
    elif "index" in t.columns:
        t = t.rename(columns={"index": "term"})
    t["model"] = name
    t["outcome"] = outcome
    t["n"] = model._N
    RESULTS.append(t)


def main():
    df = load_frame()
    vcov = {"CRV1": "bct2020"}
    X = " + ".join(BUILDING_COVARS)
    XO = " + ".join(BUILDING_COVARS + OWNER_COVARS)

    print("[1/6] PPML ECB citations (base spec)")
    m = pf.fepois(f"n_ecb_2020on ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "ppml_ecb", "ECB citations 2020-26")

    print("[2/6] PPML DOB violations (base spec)")
    m = pf.fepois(f"n_dobviol_2020on ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "ppml_dobviol", "DOB violations 2020-26")

    print("[3/6] LPM any DOB violation")
    m = pf.feols(f"any_dobviol100 ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "lpm_any_dobviol", "any DOB violation (pp)")

    print("[4/6] PPML ECB citations (owner-augmented)")
    m = pf.fepois(f"n_ecb_2020on ~ {XO} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "ppml_ecb_owner", "ECB citations 2020-26")

    print("[5/6] PPML DOB violations (owner-augmented)")
    m = pf.fepois(f"n_dobviol_2020on ~ {XO} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "ppml_dobviol_owner", "DOB violations 2020-26")

    print("[6/7] BISG subsample: PPML ECB citations")
    bs = df[df["p_white"].notna() & (df["owner_type"] == "individual")
            & (df["unitsres"] < 16)]
    xr = " + ".join([c for c in BUILDING_COVARS if c not in
                     ("llc", "corp_other", "trust_estate", "nycha", "govt",
                      "is_coop", "is_condo")] + OWNER_COVARS)
    m = pf.fepois(f"n_ecb_2020on ~ p_black + p_hispanic + p_asian + {xr}"
                  f" | size_bin + bct2020", data=bs, vcov=vcov)
    collect(m, "bisg_ppml_ecb", "ECB citations 2020-26")

    print("[7/7] BISG subsample: PPML disposition violations")
    m = pf.fepois(f"n_viol_disp ~ p_black + p_hispanic + p_asian + {xr}"
                  f" | size_bin + bct2020", data=bs, vcov=vcov)
    collect(m, "bisg_ppml_viol", "disposition violations 2020-26")

    res = pd.concat(RESULTS, ignore_index=True)
    res["pct_change"] = (np.exp(res["estimate"]) - 1) * 100
    res.to_csv(OUT / "citation_tidy_estimates.csv", index=False)
    print(f"\nSaved -> {OUT/'citation_tidy_estimates.csv'}")

    for name in ["ppml_ecb", "ppml_dobviol", "bisg_ppml_ecb"]:
        g = res[res["model"] == name]
        print(f"\n== {name} (N={g['n'].iloc[0]:,.0f}) ==")
        cols = [c for c in ["term", "estimate", "pct_change", "std_error", "pr(>|t|)"]
                if c in g.columns]
        print(g[cols].round(3).to_string(index=False))

    # annualized descriptives per 100 properties
    rows = []
    def add(name, mask):
        d = df[mask]
        rows.append({"group": name, "n": len(d),
                     "ecb_per100_yr": d["n_ecb_2020on"].mean() / YEARS * 100,
                     "dobviol_per100_yr": d["n_dobviol_2020on"].mean() / YEARS * 100})
    add("All residential", df["unitsres"] >= 0)
    add("LLC", df["llc"] == 1)
    add("Individual", df["owner_type"] == "individual")
    add("Owner-occupied (STAR)", df["owner_occ_star"] == 1)
    add("Pre-1940", df["era_pre1940"] == 1)
    add("Built 2000+", (df[["era_pre1940", "era_4079", "era_8099", "era_unknown"]].sum(axis=1)) == 0)
    add("Prior violation 2010-19", df["any_prior_viol"] == 1)
    add("No prior violation", df["any_prior_viol"] == 0)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "citation_descriptives.csv", index=False)
    print("\n", t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
