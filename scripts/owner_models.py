"""
Owner-characteristics tier of the risk-factor study.

Adds to the within-tract, within-size design:
  - absentee geography (name-matched deed mailing zip): same zip as
    property (ref) / elsewhere in NYC / outside NYC / unknown
  - multi-property owner flag (3+ lots, exact PLUTO name)
  - predicted owner race (BISG posterior; surname-only robustness) on the
    individually-owned <16 unit subsample
Also runs the unweighted conditional-violation robustness requested by the
referee for the base specification.

Output: data/analysis/risk_models/owner_tidy_estimates.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"

BUILDING_COVARS = [
    "llc", "corp_other", "trust_estate", "nycha", "govt",
    "owner_occ_star", "is_coop", "is_condo",
    "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "mixed_use", "mzone", "multi_bldg",
    "log2_area_per_unit", "value_rank", "any_prior_viol",
]
OWNER_COVARS = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner"]
CAT_SHARES = ["sh_conv", "sh_constr", "sh_elev", "sh_boiler"]
RACE = ["p_black", "p_hispanic", "p_asian"]
RACE_SN = ["sn_black", "sn_hispanic", "sn_asian"]


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str})
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
    df["any100"] = df["any_complaint"] * 100.0
    df["anyviol100"] = df["any_viol_disp"] * 100.0
    df["anyecb100"] = df["any_ecb_2020on"] * 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        df["violrate100"] = np.where(df["n_substantive"] > 0,
                                     df["n_viol_disp"] / df["n_substantive"] * 100.0, np.nan)
        for g in ["conv", "constr", "elev", "boiler"]:
            df[f"sh_{g}"] = np.where(df["n_complaints"] > 0,
                                     df[f"n_{g}"] / df["n_complaints"], 0.0)
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])].copy()
    print(f"Sample: {len(df):,}")
    print("owner_geo:", df["owner_geo"].value_counts(normalize=True).round(3).to_dict())
    return df


RESULTS = []


def collect(model, name, outcome, sample):
    t = model.tidy().reset_index().rename(columns={"index": "term",
                                                   "Coefficient": "term"})
    t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                 for c in t.columns]
    if "coefficient" in t.columns:
        t = t.rename(columns={"coefficient": "term"})
    t["model"] = name
    t["outcome"] = outcome
    t["sample"] = sample
    t["n"] = model._N
    RESULTS.append(t)


def main():
    df = load_frame()
    vcov = {"CRV1": "bct2020"}
    X = " + ".join(BUILDING_COVARS + OWNER_COVARS)

    print("[1/7] PPML complaint count + owner covars")
    m = pf.fepois(f"n_complaints ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "owner_ppml_ncomp", "complaint count (log)", "universe")

    print("[2/7] LPM any complaint + owner covars")
    m = pf.feols(f"any100 ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "owner_lpm_any", "any complaint (pp)", "universe")

    print("[3/7] conditional violation rate + owner covars (wtd, cat-adjusted)")
    sub = df[df["n_substantive"] > 0]
    m = pf.feols(f"violrate100 ~ {X} + {' + '.join(CAT_SHARES)} | size_bin + bct2020",
                 data=sub, weights="n_substantive", vcov=vcov)
    collect(m, "owner_cond_violrate", "violations per inspection (pp)", "inspected")

    print("[4/7] LPM any ECB violation + owner covars")
    m = pf.feols(f"anyecb100 ~ {X} | size_bin + bct2020", data=df, vcov=vcov)
    collect(m, "owner_lpm_anyecb", "any ECB violation (pp)", "universe")

    print("[5/7] unweighted conditional robustness (base covars)")
    xb = " + ".join(BUILDING_COVARS)
    m = pf.feols(f"violrate100 ~ {xb} + {' + '.join(CAT_SHARES)} | size_bin + bct2020",
                 data=sub, vcov=vcov)
    collect(m, "cond_violrate_catadj_unwtd", "violations per inspection (pp, unwtd)", "inspected")

    # BISG subsample: individually owned, <16 units, surname matched
    bs = df[df["p_white"].notna() & (df["owner_type"] == "individual")
            & (df["unitsres"] < 16)].copy()
    print(f"[6/7] BISG subsample models (N={len(bs):,})")
    xr = " + ".join([c for c in BUILDING_COVARS if c not in
                     ("llc", "corp_other", "trust_estate", "nycha", "govt",
                      "is_coop", "is_condo")] + OWNER_COVARS)
    m = pf.fepois(f"n_complaints ~ {' + '.join(RACE)} + {xr} | size_bin + bct2020",
                  data=bs, vcov=vcov)
    collect(m, "bisg_ppml_ncomp", "complaint count (log)", "individual <16u")
    m = pf.feols(f"any100 ~ {' + '.join(RACE)} + {xr} | size_bin + bct2020",
                 data=bs, vcov=vcov)
    collect(m, "bisg_lpm_any", "any complaint (pp)", "individual <16u")
    bss = bs[bs["n_substantive"] > 0]
    m = pf.feols(f"violrate100 ~ {' + '.join(RACE)} + {xr} + {' + '.join(CAT_SHARES)}"
                 f" | size_bin + bct2020", data=bss, weights="n_substantive", vcov=vcov)
    collect(m, "bisg_cond_violrate", "violations per inspection (pp)", "individual <16u")
    m = pf.feols(f"anyecb100 ~ {' + '.join(RACE)} + {xr} | size_bin + bct2020",
                 data=bs, vcov=vcov)
    collect(m, "bisg_lpm_anyecb", "any ECB violation (pp)", "individual <16u")

    print("[7/7] BISG surname-only robustness")
    m = pf.fepois(f"n_complaints ~ {' + '.join(RACE_SN)} + {xr} | size_bin + bct2020",
                  data=bs, vcov=vcov)
    collect(m, "bisg_sn_ppml_ncomp", "complaint count (log)", "individual <16u")
    m = pf.feols(f"violrate100 ~ {' + '.join(RACE_SN)} + {xr} + {' + '.join(CAT_SHARES)}"
                 f" | size_bin + bct2020", data=bss, weights="n_substantive", vcov=vcov)
    collect(m, "bisg_sn_cond_violrate", "violations per inspection (pp)", "individual <16u")

    res = pd.concat(RESULTS, ignore_index=True)
    res.to_csv(OUT / "owner_tidy_estimates.csv", index=False)
    print(f"\nSaved {len(res)} estimates -> {OUT/'owner_tidy_estimates.csv'}")

    show = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner",
            "p_black", "p_hispanic", "p_asian", "llc", "owner_occ_star"]
    for name in ["owner_ppml_ncomp", "owner_cond_violrate", "bisg_ppml_ncomp",
                 "bisg_cond_violrate", "bisg_lpm_any"]:
        g = res[res["model"] == name]
        g = g[g["term"].isin(show)]
        print(f"\n== {name} (N={g['n'].iloc[0]:,.0f}) ==")
        cols = [c for c in ["term", "estimate", "std_error", "pr(>|t|)"] if c in g.columns]
        print(g[cols].round(4).to_string(index=False))

    # descriptives for the post
    d = df.groupby("owner_geo").agg(n=("bbl_key", "size"),
                                    any_complaint=("any_complaint", "mean"),
                                    any_viol=("any_viol_disp", "mean"))
    print("\n", d.round(3).to_string())


if __name__ == "__main__":
    main()
