"""
Citations issued by complaint type (BISG subsample: individually owned,
<16 units, surname-matched; N ~ 460K properties).

Builds inspection-level records exactly as asian_effect_heterogeneity.
part_b_inspection_level does (open_data JOIN bis_scrape, 2020+, catgroup
via CATEGORY_GROUPS over modal category descriptions, outcome via
classify_disposition), then counts, per property, the complaints whose
outcome is 'violation' within each category group (elevator omitted):
n_viol_conv / n_viol_constr / n_viol_boiler / n_viol_other.

PPML of each count on race probabilities + full covariates, size-bin +
tract FE, tract-clustered SEs — the citation analogue of the category
complaint-volume models (cat_n_*).

Output: data/analysis/risk_models/category_citations.csv + console report.
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
from disposition_codes import classify_disposition
from build_risk_dataset import CATEGORY_GROUPS

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"

BUILDING_COVARS = [
    "owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "mixed_use", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
    "any_prior_viol",
]
OWNER_COVARS = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner"]
RACE = ["p_black", "p_hispanic", "p_asian"]


def load_bisg_frame() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str})
    df["bbl_key"] = df["bbl_key"].astype(str)
    df = df[df["owner_type"] != "missing"]
    df["owner_occ_star"] = df["owner_occ_star"].astype(int)
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
    df["n_other"] = (df["n_complaints"] - df[["n_conv", "n_constr", "n_elev", "n_boiler"]]
                     .sum(axis=1)).clip(lower=0)
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])]
    bs = df[df["p_white"].notna() & (df["owner_type"] == "individual")
            & (df["unitsres"] < 16)].copy()
    print(f"BISG subsample: {len(bs):,} properties; "
          f"complaints {bs['n_complaints'].sum():,}")
    return bs


def load_inspection_level(bs) -> pd.DataFrame:
    """Inspection-level records on the subsample, categorized and classified
    exactly as asian_effect_heterogeneity.part_b_inspection_level."""
    conn = sqlite3.connect(str(config.DB_PATH))
    boro_case = """CASE b.borough
        WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
        WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
        WHEN 'STATEN ISLAND' THEN '5' END"""
    c = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.disposition_code, o.complaint_category,
               {boro_case} AS boro_code, b.block, b.lot, b.category_description
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered, 7, 4) >= '2020'
    """, conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                    in zip(c["boro_code"], c["block"], c["lot"])]
    c = c[c["bbl_key"].isin(set(bs["bbl_key"]))]
    desc = c[~c["category_description"].isin(["Date", None, ""])]
    code_map = (desc.groupby("complaint_category")["category_description"]
                .agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    c["cat_desc"] = c["complaint_category"].map(code_map).fillna("")
    c["catgroup"] = "other"
    for g, pat in CATEGORY_GROUPS.items():
        c.loc[c["cat_desc"].str.contains(pat, regex=True, na=False), "catgroup"] = g
    c["outcome"] = c["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    print(f"  inspection-level records on subsample: {len(c):,} "
          f"({(c['outcome'] == 'violation').sum():,} with a violation issued)")
    return c


RES = []


def collect(model, name):
    t = model.tidy().reset_index()
    t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                 for c in t.columns]
    t = t.rename(columns={"coefficient": "term", "index": "term"})
    t["model"] = name
    t["n"] = model._N
    RES.append(t)
    return t


def main():
    bs = load_bisg_frame()
    c = load_inspection_level(bs)

    # property-level counts of violation-outcome complaints by category group
    v = c[c["outcome"] == "violation"]
    grp = v.groupby(["bbl_key", "catgroup"]).size().unstack(fill_value=0)
    outcomes = [  # (column, catgroup, model name, label)
        ("n_viol_conv", "conversion", "citations_conversion", "conversion"),
        ("n_viol_constr", "construction", "citations_constr", "construction"),
        ("n_viol_boiler", "boiler_mech", "citations_boiler", "boiler/mech"),
        ("n_viol_other", "other", "citations_other", "other"),
    ]
    for col, cg, _, _ in outcomes:
        src = grp[cg] if cg in grp.columns else pd.Series(dtype=float)
        bs[col] = bs["bbl_key"].map(src).fillna(0).astype(int)
        print(f"  {col}: {bs[col].sum():,} citations on "
              f"{(bs[col] > 0).sum():,} properties")

    print("\n=== Citations issued by category (PPML, IRR) ===")
    X = " + ".join(RACE + BUILDING_COVARS + OWNER_COVARS)
    vcov = {"CRV1": "bct2020"}
    for outc, _, name, label in outcomes:
        m = pf.fepois(f"{outc} ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
        t = collect(m, name)
        row = t[t["term"] == "p_asian"].iloc[0]
        b, se = row["estimate"], row["std_error"]
        print(f"  {label:<12} p_asian: {(np.exp(b)-1)*100:+7.1f}%  "
              f"[{(np.exp(b-1.96*se)-1)*100:+.1f}%, {(np.exp(b+1.96*se)-1)*100:+.1f}%]  "
              f"(se {se:.3f}, p {row['pr(>|t|)']:.3f}, N {m._N:,})")

    res = pd.concat(RES, ignore_index=True)
    res.to_csv(OUT / "category_citations.csv", index=False)
    print(f"\nSaved {len(res)} estimates -> {OUT/'category_citations.csv'}")


if __name__ == "__main__":
    main()
