"""
Risk-factor models for 311 building complaints and violations.

Design: property-level regressions over the PLUTO residential universe
(766,939 tax lots), with exact building-size fixed effects (16 unitsres
bins, exact 1-10) and census tract fixed effects, so every estimate is
identified from comparisons among same-size buildings in the same
neighborhood. SEs clustered by census tract.

Margins:
  A. Extensive: any complaint 2020-May 2026 (LPM, pp)
  B. Intensive: complaint count (Poisson PML, IRR)
  C. Citation: any disposition-violation (LPM); violation count (PPML);
     any ECB violation (LPM, independent administrative measure)
  D. Conditional: violations per substantive inspection, weighted by
     inspections (~inspection-level regression); no-access rate.
Tiers:
  1. Borough FE + tract covariates (between-neighborhood gradients)
  2. Tract FE (within-neighborhood; primary)
Robustness: 2-4 unit stratum, 1-unit stratum, excl. condo/coop/govt,
category-specific counts (conversion / construction).

Output: data/analysis/risk_models/*.csv (tidy estimates), summary.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"
OUT.mkdir(exist_ok=True)

BUILDING_COVARS = [
    "llc", "corp_other", "trust_estate", "nycha", "govt",       # owner (ref: individual)
    "owner_occ_star", "is_coop", "is_condo",                    # tenure structure
    "era_pre1940", "era_4079", "era_8099", "era_unknown",       # era (ref: built 2000+)
    "mixed_use", "mzone", "multi_bldg",
    "log2_area_per_unit", "value_rank", "any_prior_viol",
]
TRACT_COVARS = ["tract_poverty10", "tract_renter10", "tract_foreign10",
                "tract_overcrowd10", "tract_log_income_z"]
CAT_SHARES = ["sh_conv", "sh_constr", "sh_elev", "sh_boiler"]

LABELS = {
    "llc": "LLC owner (vs individual)",
    "corp_other": "Corporate/org owner (vs individual)",
    "trust_estate": "Trust/estate owner (vs individual)",
    "nycha": "NYCHA",
    "govt": "Other government owner",
    "owner_occ_star": "Owner-occupied (STAR)",
    "is_coop": "Co-op building",
    "is_condo": "Condominium",
    "era_pre1940": "Built before 1940 (vs 2000+)",
    "era_4079": "Built 1940-1979 (vs 2000+)",
    "era_8099": "Built 1980-1999 (vs 2000+)",
    "era_unknown": "Construction year unknown",
    "mixed_use": "Mixed use (commercial units)",
    "mzone": "Manufacturing zoning",
    "multi_bldg": "Multiple buildings on lot",
    "log2_area_per_unit": "Floor area per unit (per doubling)",
    "value_rank": "Assessed value rank within type (0-1)",
    "any_prior_viol": "Any violation 2010-2019",
    "tract_poverty10": "Tract poverty rate (+10pp)",
    "tract_renter10": "Tract renter share (+10pp)",
    "tract_foreign10": "Tract foreign-born share (+10pp)",
    "tract_overcrowd10": "Tract overcrowding (+10pp)",
    "tract_log_income_z": "Tract log median income (+1 SD)",
}


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str})
    n0 = len(df)

    # owner dummies
    for t in ["llc", "corp_other", "trust_estate", "nycha", "govt"]:
        df[t] = (df["owner_type"] == t).astype(int)
    df = df[df["owner_type"] != "missing"]

    for b in ["owner_occ_star", "is_coop", "is_condo"]:
        df[b] = df[b].astype(int)

    # construction era (ref: 2000+)
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)

    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])

    # outcomes in interpretable units
    df["any100"] = df["any_complaint"] * 100.0
    df["anyviol100"] = df["any_viol_disp"] * 100.0
    df["anyecb100"] = df["any_ecb_2020on"] * 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        df["violrate100"] = np.where(df["n_substantive"] > 0,
                                     df["n_viol_disp"] / df["n_substantive"] * 100.0, np.nan)
        df["noaccrate100"] = np.where(df["n_complaints"] > 0,
                                      df["n_no_access"] / df["n_complaints"] * 100.0, np.nan)
        for g in ["conv", "constr", "elev", "boiler"]:
            df[f"sh_{g}"] = np.where(df["n_complaints"] > 0,
                                     df[f"n_{g}"] / df["n_complaints"], 0.0)

    # tract covariates in +10pp / z units
    df["tract_poverty10"] = df["tract_poverty"] * 10
    df["tract_renter10"] = df["tract_renter_share"] * 10
    df["tract_foreign10"] = df["tract_foreign_born"] * 10
    df["tract_overcrowd10"] = df["tract_overcrowd"] * 10
    li = df["tract_log_income"]
    df["tract_log_income_z"] = (li - li.mean()) / li.std()

    # estimation sample: complete cases on building covariates
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    complete = df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])
    dropped = (~complete).sum()
    df = df[complete].copy()
    print(f"Sample: {n0:,} -> {len(df):,} complete-case "
          f"({dropped:,} dropped for missing area/value/size)")
    return df


RESULTS = []


def collect(model, model_name: str, outcome: str, scale: str, sample: str):
    t = model.tidy().reset_index().rename(columns={"index": "term"})
    t["model"] = model_name
    t["outcome"] = outcome
    t["scale"] = scale
    t["sample"] = sample
    t["N"] = model._N
    RESULTS.append(t)
    return t


def run_models(df: pd.DataFrame):
    vcov = {"CRV1": "bct2020"}
    X = " + ".join(BUILDING_COVARS)
    XT = X + " + " + " + ".join(TRACT_COVARS)

    print("\n[1/10] Tier-1 LPM any complaint (borough FE + tract covariates)")
    m = pf.feols(f"any100 ~ {XT} | size_bin + borocode", data=df, vcov=vcov)
    collect(m, "tier1_lpm_any", "any complaint (pp)", "pp", "universe")

    print("[2/10] Tier-2 LPM any complaint (tract FE)")
    m = pf.feols(f"any100 ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "tract_lpm_any", "any complaint (pp)", "pp", "universe")

    print("[3/10] Tier-2 PPML complaint count (tract FE)")
    m = pf.fepois(f"n_complaints ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "tract_ppml_ncomp", "complaint count", "log", "universe")

    print("[4/10] Tier-2 LPM any disposition-violation")
    m = pf.feols(f"anyviol100 ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "tract_lpm_anyviol", "any violation (pp)", "pp", "universe")

    print("[5/10] Tier-2 PPML violation count")
    m = pf.fepois(f"n_viol_disp ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "tract_ppml_nviol", "violation count", "log", "universe")

    print("[6/10] Tier-2 LPM any ECB violation (administrative)")
    m = pf.feols(f"anyecb100 ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "tract_lpm_anyecb", "any ECB violation (pp)", "pp", "universe")

    print("[7/10] Conditional: violation rate per substantive inspection (wtd)")
    sub = df[df["n_substantive"] > 0]
    m = pf.feols(f"violrate100 ~ {X} | size_bin + bct2020", data=sub,
                 weights="n_substantive", vcov=vcov)
    collect(m, "cond_violrate", "violations per inspection (pp)", "pp", "inspected")
    m = pf.feols(f"violrate100 ~ {X} + {' + '.join(CAT_SHARES)} | size_bin + bct2020",
                 data=sub, weights="n_substantive", vcov=vcov)
    collect(m, "cond_violrate_catadj", "violations per inspection, category-adjusted (pp)",
            "pp", "inspected")

    print("[8/10] Conditional: no-access rate per complaint (wtd)")
    sub = df[df["n_complaints"] > 0]
    m = pf.feols(f"noaccrate100 ~ {X} | size_bin + bct2020", data=sub,
                 weights="n_complaints", vcov=vcov)
    collect(m, "cond_noaccess", "no-access per complaint (pp)", "pp", "complained")

    print("[9/10] Strata: 2-4 units, 1 unit")
    s24 = df[df["unitsres"].between(2, 4)]
    m = pf.feols(f"any100 ~ {X} | size_bin + bct2020", data=s24, vcov=vcov)
    collect(m, "strata24_lpm_any", "any complaint (pp)", "pp", "2-4 units")
    m = pf.fepois(f"n_complaints ~ {X} | size_bin + bct2020", data=s24, vcov=vcov)
    collect(m, "strata24_ppml", "complaint count", "log", "2-4 units")
    s1 = df[df["unitsres"] == 1]
    x1 = " + ".join([c for c in BUILDING_COVARS if c not in
                     ("is_coop", "is_condo", "nycha")])  # not identified in 1-unit
    m = pf.feols(f"any100 ~ {x1} | bct2020", data=s1, vcov=vcov)
    collect(m, "strata1_lpm_any", "any complaint (pp)", "pp", "1 unit")

    print("[10/10] Category-specific PPML + private-only robustness")
    m = pf.fepois(f"n_conv ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "ppml_conversion", "illegal-conversion complaints", "log", "universe")
    m = pf.fepois(f"n_constr ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "ppml_construction", "construction complaints", "log", "universe")
    priv = df[(df["owner_type"].isin(["individual", "llc", "corp_other", "trust_estate"]))
              & (df["is_condo"] == 0) & (df["is_coop"] == 0)]
    xp = " + ".join([c for c in BUILDING_COVARS if c not in
                     ("is_coop", "is_condo", "nycha", "govt")])
    m = pf.feols(f"any100 ~ {xp} | size_bin + bct2020", data=priv, vcov=vcov)
    collect(m, "private_lpm_any", "any complaint (pp)", "pp", "private non-condo")
    m = pf.fepois(f"n_complaints ~ {xp} | size_bin + bct2020", data=priv, vcov=vcov)
    collect(m, "private_ppml", "complaint count", "log", "private non-condo")


def descriptives(df: pd.DataFrame):
    rows = []
    def add(name, mask):
        d = df[mask]
        rows.append({
            "group": name, "n_properties": len(d),
            "any_complaint": d["any_complaint"].mean(),
            "complaints_per_100units": d["n_complaints"].sum() / d["unitsres"].sum() * 100,
            "any_violation": d["any_viol_disp"].mean(),
            "viol_per_inspection": d["n_viol_disp"].sum() / max(d["n_substantive"].sum(), 1),
            "no_access_share": d["n_no_access"].sum() / max(d["n_complaints"].sum(), 1),
        })
    add("All residential", df.index == df.index)
    add("Individual owner", df["owner_type"] == "individual")
    add("LLC owner", df["llc"] == 1)
    add("Corporate/org owner", df["corp_other"] == 1)
    add("Trust/estate", df["trust_estate"] == 1)
    add("Owner-occupied (STAR)", df["owner_occ_star"] == 1)
    add("Not owner-occupied", df["owner_occ_star"] == 0)
    add("Co-op", df["is_coop"] == 1)
    add("Condo", df["is_condo"] == 1)
    add("NYCHA", df["nycha"] == 1)
    add("Pre-1940", df["era_pre1940"] == 1)
    add("Built 2000+", (df["era_pre1940"] + df["era_4079"] + df["era_8099"] + df["era_unknown"]) == 0)
    add("Mixed use", df["mixed_use"] == 1)
    add("Prior violation 2010-19", df["any_prior_viol"] == 1)
    add("No prior violation", df["any_prior_viol"] == 0)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "descriptives.csv", index=False)
    print("\nDescriptives:\n", t.round(3).to_string(index=False))


def main():
    df = load_frame()
    descriptives(df)
    run_models(df)
    res = pd.concat(RESULTS, ignore_index=True)
    # normalize column names across pyfixest versions
    res.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                   for c in res.columns]
    term_col = "coefficient" if "coefficient" in res.columns else "term"
    res = res.rename(columns={term_col: "term"})
    res["label"] = res["term"].map(LABELS).fillna(res["term"])
    res.to_csv(OUT / "tidy_estimates.csv", index=False)
    print(f"\nSaved {len(res)} estimates -> {OUT/'tidy_estimates.csv'}")

    key = res[res["model"].isin(["tract_lpm_any", "tract_ppml_ncomp",
                                 "cond_violrate_catadj", "tract_lpm_anyviol"])]
    for mname, g in key.groupby("model"):
        print(f"\n== {mname} (N={g['n'].iloc[0]:,}) ==" if "n" in g.columns
              else f"\n== {mname} ==")
        cols = [c for c in ["label", "estimate", "std_error", "pr(>|t|)"] if c in g.columns]
        print(g[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
