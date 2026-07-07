"""
Heterogeneity in the Asian-owner effect (BISG subsample: individually
owned, <16 units, surname-matched; N ~ 460K).

A. Which complaint types drive the +38% complaint premium?
   PPML of category-specific counts (conversion / construction / elevator
   / boiler-mech / other) on race probabilities + full covariates,
   size-bin + tract FE.
B. Inspection-level margins: within complaint category (category-code FE),
   is the violation rate per substantive inspection lower for Asian-owned
   properties in every category? And is no-access higher?
C. Context interactions: does the complaint premium vary by owner-
   occupancy, era, mixed use, building size, borough, and the tract's
   Asian population share?

Output: data/analysis/risk_models/asian_heterogeneity.csv + console report.
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
    "com_class", "log_bldgarea", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
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
    # commercial-exposure controls (spec 2b+ from owner_commercial_sensitivity.py):
    # com_class (S/K/O storefront/office/mixed class) + log floor area, replacing the
    # binary mixed_use flag. comm_bin FE omitted -- near-degenerate on this
    # individually-owned <16-unit subsample (~2% commercial exposure).
    _ba = pd.to_numeric(df["bldgarea"], errors="coerce")
    df["log_bldgarea"] = np.log(_ba.where(_ba > 0))
    df["com_class"] = df["bldgclass"].astype(str).str[0].isin(["S", "K", "O"]).astype(int)
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


def part_a_categories(bs):
    print("\n=== A. Complaint premium by category (PPML, IRR) ===")
    X = " + ".join(RACE + BUILDING_COVARS + OWNER_COVARS)
    vcov = {"CRV1": "bct2020"}
    for outc, label in [("n_conv", "conversion"), ("n_constr", "construction"),
                        ("n_elev", "elevator"), ("n_boiler", "boiler/mech"),
                        ("n_other", "other"), ("n_complaints", "ALL")]:
        try:
            m = pf.fepois(f"{outc} ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
            t = collect(m, f"cat_{outc}")
            row = t[t["term"] == "p_asian"].iloc[0]
            print(f"  {label:<12} p_asian: {(np.exp(row['estimate'])-1)*100:+7.1f}%  "
                  f"(se {row['std_error']:.3f}, p {row['pr(>|t|)']:.3f}, N {m._N:,})")
        except Exception as e:
            print(f"  {label}: FAILED {e}")

    # raw category mix, classified owners
    hi_a = bs["p_asian"] > 0.7
    hi_w = bs["p_white"] > 0.7
    mix = pd.DataFrame({
        "Asian>0.7": [bs.loc[hi_a, c].sum() / max(bs.loc[hi_a, "n_complaints"].sum(), 1)
                      for c in ["n_conv", "n_constr", "n_elev", "n_boiler", "n_other"]],
        "White>0.7": [bs.loc[hi_w, c].sum() / max(bs.loc[hi_w, "n_complaints"].sum(), 1)
                      for c in ["n_conv", "n_constr", "n_elev", "n_boiler", "n_other"]],
    }, index=["conversion", "construction", "elevator", "boiler/mech", "other"])
    print("\n  Raw complaint-mix shares (owners classified at P>0.7):")
    print(mix.round(3).to_string())
    print(f"  n properties: Asian {hi_a.sum():,}, White {hi_w.sum():,}; "
          f"complaints: {bs.loc[hi_a,'n_complaints'].sum():,} vs {bs.loc[hi_w,'n_complaints'].sum():,}")


def part_b_inspection_level(bs):
    print("\n=== B. Inspection-level margins within complaint category ===")
    conn = sqlite3.connect(str(config.DB_PATH))
    boro_case = """CASE b.borough
        WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
        WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
        WHEN 'STATEN ISLAND' THEN '5' END"""
    c = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.disposition_code, o.complaint_category,
               {boro_case} AS boro_code, b.block, b.lot, b.category_description,
               b.ecb_violation
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
    c["viol100"] = (c["outcome"] == "violation").astype(float) * 100
    c["noacc100"] = (c["outcome"] == "no_access").astype(float) * 100
    c["ecb100"] = (c["ecb_violation"].fillna("").str.strip() != "").astype(float) * 100

    keep = ["bbl_key", "size_bin", "bct2020", "p_white"] + RACE + BUILDING_COVARS + OWNER_COVARS
    insp = c.merge(bs[keep], on="bbl_key")
    print(f"  inspection-level records on subsample: {len(insp):,} "
          f"({(insp['outcome'].isin(['violation','no_violation'])).sum():,} substantive)")

    # per-category p_asian slopes on the substantive margin
    for g in ["conversion", "construction", "elevator", "boiler_mech", "other"]:
        insp[f"pa_{g}"] = insp["p_asian"] * (insp["catgroup"] == g).astype(int)
    pa_terms = " + ".join([f"pa_{g}" for g in
                           ["conversion", "construction", "elevator", "boiler_mech", "other"]])
    Xb = " + ".join(["p_black", "p_hispanic"] + BUILDING_COVARS + OWNER_COVARS)
    vcov = {"CRV1": "bct2020"}

    sub = insp[insp["outcome"].isin(["violation", "no_violation"])]
    m = pf.feols(f"viol100 ~ {pa_terms} + {Xb} | complaint_category + size_bin + bct2020",
                 data=sub, vcov=vcov)
    t = collect(m, "insp_viol_bycat")
    print("\n  Violation per substantive inspection, p_asian slope by category (pp):")
    print(t[t["term"].str.startswith("pa_")][["term", "estimate", "std_error", "pr(>|t|)"]]
          .round(3).to_string(index=False))

    m = pf.feols(f"noacc100 ~ {pa_terms} + {Xb} | complaint_category + size_bin + bct2020",
                 data=insp, vcov=vcov)
    t = collect(m, "insp_noacc_bycat")
    print("\n  No-access per complaint, p_asian slope by category (pp):")
    print(t[t["term"].str.startswith("pa_")][["term", "estimate", "std_error", "pr(>|t|)"]]
          .round(3).to_string(index=False))

    # pooled margins for reference
    m = pf.feols(f"viol100 ~ p_asian + {Xb} | complaint_category + size_bin + bct2020",
                 data=sub, vcov=vcov)
    collect(m, "insp_viol_pooled")
    print(f"\n  pooled p_asian on violation|substantive: "
          f"{m.tidy().loc['p_asian','Estimate']:.2f}pp (p {m.tidy().loc['p_asian','Pr(>|t|)']:.3f})")
    m = pf.feols(f"noacc100 ~ p_asian + {Xb} | complaint_category + size_bin + bct2020",
                 data=insp, vcov=vcov)
    collect(m, "insp_noacc_pooled")
    print(f"  pooled p_asian on no-access: "
          f"{m.tidy().loc['p_asian','Estimate']:.2f}pp (p {m.tidy().loc['p_asian','Pr(>|t|)']:.3f})")

    # ECB citation linked to the complaint (bis_scrape ecb_violation field),
    # same substantive-inspection conditioning as the disposition margin
    m = pf.feols(f"ecb100 ~ {pa_terms} + {Xb} | complaint_category + size_bin + bct2020",
                 data=sub, vcov=vcov)
    collect(m, "insp_ecb_bycat")
    m = pf.feols(f"ecb100 ~ p_asian + {Xb} | complaint_category + size_bin + bct2020",
                 data=sub, vcov=vcov)
    collect(m, "insp_ecb_pooled")
    print(f"  pooled p_asian on ECB|substantive: "
          f"{m.tidy().loc['p_asian','Estimate']:.2f}pp (p {m.tidy().loc['p_asian','Pr(>|t|)']:.3f}) "
          f"[base {sub.loc[sub['p_white']>0.7,'ecb100'].mean():.1f} white-classified]")


def part_c_context(bs):
    print("\n=== C. Context interactions on the complaint premium (PPML) ===")
    vcov = {"CRV1": "bct2020"}
    Xb = " + ".join(["p_black", "p_hispanic"] + BUILDING_COVARS + OWNER_COVARS)

    # tract Asian share interaction (share centered at subsample mean)
    bs = bs.copy()
    bs["asian_share_c"] = bs["tract_pct_asian"] - bs["tract_pct_asian"].mean()
    m = pf.fepois(f"n_complaints ~ p_asian + p_asian:asian_share_c + {Xb} "
                  f"| size_bin + bct2020", data=bs, vcov=vcov)
    t = collect(m, "ctx_tract_share")
    r = t.set_index("term")
    print(f"  p_asian at mean tract Asian share ({bs['tract_pct_asian'].mean():.2f}): "
          f"b={r.loc['p_asian','estimate']:.3f}")
    print(f"  p_asian x tract Asian share (+10pp): "
          f"b={r.loc['p_asian:asian_share_c','estimate']*0.1:+.4f} "
          f"(p {r.loc['p_asian:asian_share_c','pr(>|t|)']:.3f})")

    # binary context interactions
    for var, label in [("owner_occ_star", "owner-occupied"), ("era_pre1940", "pre-1940"),
                       ("com_class", "commercial/mixed building class")]:
        m = pf.fepois(f"n_complaints ~ p_asian + p_asian:{var} + {Xb} "
                      f"| size_bin + bct2020", data=bs, vcov=vcov)
        t = collect(m, f"ctx_{var}")
        r = t.set_index("term")
        b0 = r.loc["p_asian", "estimate"]
        b1 = r.loc[f"p_asian:{var}", "estimate"]
        p1 = r.loc[f"p_asian:{var}", "pr(>|t|)"]
        print(f"  {label:<15} base {(np.exp(b0)-1)*100:+.0f}%  interaction "
              f"{(np.exp(b0+b1)-1)*100:+.0f}% total (int p={p1:.3f})")

    # size strata + boroughs
    for lo, hi, label in [(1, 1, "1 unit"), (2, 4, "2-4 units"), (5, 15, "5-15 units")]:
        s = bs[bs["unitsres"].between(lo, hi)]
        fe = "bct2020" if lo == hi == 1 else "size_bin + bct2020"
        m = pf.fepois(f"n_complaints ~ {' + '.join(RACE)} + " +
                      " + ".join(BUILDING_COVARS + OWNER_COVARS) + f" | {fe}",
                      data=s, vcov=vcov)
        t = collect(m, f"strata_{label.replace(' ','')}")
        row = t[t["term"] == "p_asian"].iloc[0]
        print(f"  {label:<10} p_asian {(np.exp(row['estimate'])-1)*100:+.1f}% "
              f"(p {row['pr(>|t|)']:.3f}, N {m._N:,})")
    for bc, bname in [("1", "Manhattan"), ("2", "Bronx"), ("3", "Brooklyn"),
                      ("4", "Queens"), ("5", "Staten Is")]:
        s = bs[bs["borocode"] == bc]
        if len(s) < 5000:
            continue
        m = pf.fepois(f"n_complaints ~ {' + '.join(RACE)} + " +
                      " + ".join(BUILDING_COVARS + OWNER_COVARS) + " | size_bin + bct2020",
                      data=s, vcov=vcov)
        t = collect(m, f"boro_{bname}")
        row = t[t["term"] == "p_asian"].iloc[0]
        print(f"  {bname:<10} p_asian {(np.exp(row['estimate'])-1)*100:+.1f}% "
              f"(p {row['pr(>|t|)']:.3f}, N {m._N:,})")


def main():
    bs = load_bisg_frame()
    part_a_categories(bs)
    part_b_inspection_level(bs)
    part_c_context(bs)
    res = pd.concat(RES, ignore_index=True)
    res.to_csv(OUT / "asian_heterogeneity.csv", index=False)
    print(f"\nSaved {len(res)} estimates -> {OUT/'asian_heterogeneity.csv'}")


if __name__ == "__main__":
    main()
