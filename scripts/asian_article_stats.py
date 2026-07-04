"""
Reproduces every non-regression number quoted in post6_asian_owner_effect.md.

Outputs (printed):
  1. Raw complaint rates per 100 properties and conversion mix shares for
     classified owners (P>0.7), and the conversion no-access shares.
  2. Excess-complaints sizing: probability-weighted complaint volume x
     excess share implied by the +38% premium; share ending w/o violation.
  3. The concealment bounding exercise: applies the adjusted -2.4pp
     substantiation and +2.4pp no-access gaps to the white baseline of
     violations per 100 complaints; ceiling assumes every excess no-access
     visit concealed a violation cited with certainty.
  4. Robustness: PPML complaint model with fully interacted census-tract x
     size-bin fixed effects (9,400 cells).

Regression coefficients themselves come from asian_effect_heterogeneity.py
/ owner_models.py outputs in data/analysis/risk_models/.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl
from disposition_codes import classify_disposition
from build_risk_dataset import CATEGORY_GROUPS

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
B_COMPLAINTS = 0.3222   # bisg_ppml_ncomp p_asian (owner_tidy_estimates.csv)
GAP_SUBST = 2.4         # insp_viol_pooled p_asian, pp (asian_heterogeneity.csv)
GAP_NOACC = 2.4         # insp_noacc_pooled p_asian, pp
YEARS = 6.36            # Jan 2020 - mid-May 2026


def load_subsample():
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str}, low_memory=False)
    df["bbl_key"] = df["bbl_key"].astype(str)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])]
    bs = df[(df["owner_type"] == "individual") & (df["unitsres"] < 16)
            & df["p_white"].notna()].copy()
    print(f"subsample: {len(bs):,}")
    return bs


def raw_rates(bs):
    print("\n== 1. Raw rates, classified owners (P>0.7) ==")
    out = {}
    for g, p in [("asian", "p_asian"), ("white", "p_white")]:
        d = bs[bs[p] > 0.7]
        n = len(d)
        out[g] = d
        print(f"{g}: n={n:,} | complaints/100 {d['n_complaints'].sum()/n*100:.1f} | "
              f"conversion share {d['n_conv'].sum()/d['n_complaints'].sum():.3f} | "
              f"substantive/complaint {d['n_substantive'].sum()/d['n_complaints'].sum():.3f} | "
              f"viol/substantive {d['n_viol_disp'].sum()/d['n_substantive'].sum():.3f} | "
              f"viol/100 complaints {d['n_viol_disp'].sum()/d['n_complaints'].sum()*100:.1f}")

    conn = sqlite3.connect(str(config.DB_PATH))
    c = pd.read_sql_query("""
        SELECT o.complaint_number, o.disposition_code, o.complaint_category,
               CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
                    WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
                    WHEN 'STATEN ISLAND' THEN '5' END boro, b.block, b.lot,
               b.category_description
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, l) for b, bl, l in zip(c["boro"], c["block"], c["lot"])]
    grp = pd.Series(np.select([bs["p_asian"] > 0.7, bs["p_white"] > 0.7],
                              ["asian", "white"], default=""), index=bs.index)
    key = pd.Series(grp.values, index=bs["bbl_key"].values)
    c["grp"] = c["bbl_key"].map(key)
    c = c[c["grp"].isin(["asian", "white"])]
    desc = c[~c["category_description"].isin(["Date", None, ""])]
    cmap = (desc.groupby("complaint_category")["category_description"]
            .agg(lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    c["cat_desc"] = c["complaint_category"].map(cmap).fillna("")
    conv = c[c["cat_desc"].str.contains(CATEGORY_GROUPS["conversion"], regex=True, na=False)]
    conv = conv.assign(outc=conv["disposition_code"].fillna("").astype(str).apply(classify_disposition))
    for g in ["asian", "white"]:
        d = conv[conv["grp"] == g]
        print(f"{g} conversion complaints: {len(d):,}, no-access share {(d['outc']=='no_access').mean():.3f}")


def sizing(bs):
    print("\n== 2. Excess-complaints sizing ==")
    wtd = (bs["p_asian"] * bs["n_complaints"]).sum()
    excess = wtd * (1 - 1 / np.exp(B_COMPLAINTS))
    cl = bs[bs["p_asian"] > 0.7]
    viol_per_complaint = cl["n_viol_disp"].sum() / cl["n_complaints"].sum()
    print(f"p_asian-weighted complaints: {wtd:,.0f}; excess {excess:,.0f} total "
          f"= {excess/YEARS:,.0f}/yr = {excess/YEARS/365:.1f}/day")
    print(f"share w/o violation: {1-viol_per_complaint:.3f} -> "
          f"{excess/YEARS*(1-viol_per_complaint):,.0f}/yr without a violation")


def bound(bs):
    print("\n== 3. Concealment bound (adjusted gaps on white baseline) ==")
    w = bs[bs["p_white"] > 0.7]
    sub_rate = w["n_substantive"].sum() / w["n_complaints"].sum() * 100
    viol_rate = w["n_viol_disp"].sum() / w["n_substantive"].sum() * 100
    white = sub_rate * viol_rate / 100
    asian = (sub_rate - GAP_NOACC) * (viol_rate - GAP_SUBST) / 100
    print(f"white: {sub_rate:.1f} substantive/100 complaints x {viol_rate:.1f}% = {white:.1f} viol/100 complaints")
    print(f"asian (adjusted gaps): {asian:.1f}; ceiling (+{GAP_NOACC} concealed, cited w.p.1): {asian+GAP_NOACC:.1f}")


def interacted_fe(bs_full):
    import pyfixest as pf
    print("\n== 4. Interacted tract x size FE robustness ==")
    df = bs_full.copy()
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    df["owner_occ_star"] = df["owner_occ_star"].astype(int)
    df["cell"] = df["bct2020"] + "_" + df["size_bin"]
    X = ("p_black + p_hispanic + p_asian + owner_occ_star + era_pre1940 + era_4079"
         " + era_8099 + era_unknown + mixed_use + mzone + multi_bldg"
         " + log2_area_per_unit + value_rank + any_prior_viol"
         " + geo_nyc_other + geo_outside_nyc + geo_unknown + multi_prop_owner")
    m = pf.fepois(f"n_complaints ~ {X} | cell", data=df, vcov={"CRV1": "bct2020"})
    t = m.tidy().loc["p_asian"]
    print(f"cells: {df['cell'].nunique():,}; p_asian b={t['Estimate']:.4f} "
          f"({(np.exp(t['Estimate'])-1)*100:+.1f}%), N={m._N:,}")


def nonwhite_contrasts(bs_full):
    """Asian vs Black / Hispanic contrasts (delta-method SEs) from the
    joint models; quoted in the 'Three margins' and category sections."""
    import pyfixest as pf
    df = bs_full.copy()
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    df["owner_occ_star"] = df["owner_occ_star"].astype(int)
    X = ("p_black + p_hispanic + p_asian + owner_occ_star + era_pre1940 + era_4079"
         " + era_8099 + era_unknown + mixed_use + mzone + multi_bldg"
         " + log2_area_per_unit + value_rank + any_prior_viol"
         " + geo_nyc_other + geo_outside_nyc + geo_unknown + multi_prop_owner")

    def contrast(m, scale):
        names = list(m.coef().index)
        b = m.coef().values
        V = np.asarray(m._vcov)
        ia, ib, ih = (names.index("p_asian"), names.index("p_black"),
                      names.index("p_hispanic"))
        for other, i in [("Black", ib), ("Hispanic", ih)]:
            d = b[ia] - b[i]
            se = np.sqrt(V[ia, ia] + V[i, i] - 2 * V[ia, i])
            eff = (np.exp(d) - 1) * 100 if scale == "pct" else d
            unit = "%" if scale == "pct" else "pp"
            print(f"   Asian vs {other}: {eff:+.1f}{unit} (z={d/se:.1f})")

    print("\n== 5. Asian vs non-white non-Asian contrasts ==")
    print("complaints (PPML)")
    m = pf.fepois(f"n_complaints ~ {X} | size_bin + bct2020", data=df,
                  vcov={"CRV1": "bct2020"})
    contrast(m, "pct")
    print("conversion complaints (PPML)")
    m = pf.fepois(f"n_conv ~ {X} | size_bin + bct2020", data=df,
                  vcov={"CRV1": "bct2020"})
    contrast(m, "pct")

    conn = sqlite3.connect(str(config.DB_PATH))
    c = pd.read_sql_query("""
        SELECT o.complaint_number, o.disposition_code, o.complaint_category,
               CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
                    WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
                    WHEN 'STATEN ISLAND' THEN '5' END boro, b.block, b.lot
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, l) for b, bl, l in zip(c["boro"], c["block"], c["lot"])]
    c["outcome"] = c["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    c["viol100"] = (c["outcome"] == "violation").astype(float) * 100
    c["noacc100"] = (c["outcome"] == "no_access").astype(float) * 100
    keep = (["bbl_key", "size_bin", "bct2020", "p_black", "p_hispanic", "p_asian",
             "owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
             "mixed_use", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
             "any_prior_viol", "geo_nyc_other", "geo_outside_nyc", "geo_unknown",
             "multi_prop_owner"])
    insp = c.merge(df[keep], on="bbl_key")
    sub = insp[insp["outcome"].isin(["violation", "no_violation"])]
    print("violations per substantive inspection (LPM, pp)")
    m = pf.feols(f"viol100 ~ {X} | complaint_category + size_bin + bct2020",
                 data=sub, vcov={"CRV1": "bct2020"})
    contrast(m, "pp")
    print("no-access per complaint (LPM, pp)")
    m = pf.feols(f"noacc100 ~ {X} | complaint_category + size_bin + bct2020",
                 data=insp, vcov={"CRV1": "bct2020"})
    contrast(m, "pp")


def reduced_form_and_marginal(bs):
    """Group-level reduced form and the marginal-complaint substantiation
    arithmetic quoted in the differential-compliance mechanism."""
    print("\n== 6. Reduced form by group and marginal substantiation ==")
    for g, p in [("white", "p_white"), ("black", "p_black"),
                 ("hispanic", "p_hispanic"), ("asian", "p_asian")]:
        d = bs[bs[p] > 0.7]
        n = len(d)
        print(f"  {g}: complaints/100 {d['n_complaints'].sum()/n*100:.1f} | "
              f"viol/100 {d['n_viol_disp'].sum()/n*100:.1f} | "
              f"ECB/100 {d['n_ecb_2020on'].sum()/n*100:.1f}")
    w_sub, w_v = 65.1, 29.3   # white: substantive/100 complaints, viol%/substantive
    base = w_sub * w_v / 100
    a = (w_sub - GAP_NOACC) * (w_v - GAP_SUBST) / 100
    total = 1.38 * a
    marginal = (total - base) / 0.38
    print(f"  violations per 100 complaints: white {base:.1f}, asian adjusted {a:.1f}")
    print(f"  product-implied total at 1.38x volume: {total:.1f} (+{(total/base-1)*100:.0f}%)")
    print(f"  product-implied marginal substantiation: {marginal:.1f} per 100")
    B_VIOL = 0.1109  # bisg_ppml_viol p_asian (citation_tidy_estimates.csv)
    g = np.exp(B_VIOL) - 1
    print(f"  direct count model: +{g*100:.0f}% disposition violations -> "
          f"level {6.3*(1+g):.1f} vs 6.3 per 100; "
          f"marginal substantiation {base*g/0.38:.1f} per 100 (quoted in text)")


if __name__ == "__main__":
    bs = load_subsample()
    raw_rates(bs)
    sizing(bs)
    bound(bs)
    interacted_fe(bs)
    nonwhite_contrasts(bs)
    reduced_form_and_marginal(bs)
