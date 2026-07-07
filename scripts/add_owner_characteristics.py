"""
Enrich the property risk panel with owner characteristics:

1. Absentee geography — ACRIS grantee mailing address (owner_mailing_zip
   table) vs. property zip: same zip / elsewhere in NYC / outside NYC.
2. Portfolio scale — exact-normalized PLUTO owner name appearing on 3+
   residential lots (noisy: LLC fragmentation undercounts, common personal
   names overcount; flag, not a count).
3. Predicted owner race (BISG) — for individually owned properties <16
   units: surgeo surname P(race|surname), Bayes-updated with the racial
   composition of *owner-occupant householders* in the owner's best ZCTA
   (ACS 2023 B25003 H/B/D/I), best zip = property zip if STAR else ACRIS
   mailing zip. Both surname-only and updated probabilities are kept.

Output: data/analysis/property_risk_panel_v2.csv.gz
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

AN = config.DATA_DIR / "analysis"
PANEL = AN / "property_risk_panel.csv.gz"
ZIPS = AN / "pluto_zipcodes.csv"
ZCTA_RAW = AN / "zcta_owner_race_raw.json"
OUT = AN / "property_risk_panel_v2.csv.gz"

# NYC zip prefixes -> borough code
NYC_PREFIX = {"100": "1", "101": "1", "102": "1", "103": "5", "104": "2",
              "112": "3", "111": "4", "113": "4", "114": "4", "116": "4"}
QUEENS_EXTRA = {"11004", "11005"}

# national race shares among 2010 surname-list population (4-group renorm)
NATL = {"white": 0.655, "black": 0.128, "asian": 0.049, "hispanic": 0.168}


def zip_to_boro(z: str):
    if not isinstance(z, str) or len(z) < 5 or not z[:5].isdigit():
        return None
    z5 = z[:5]
    if z5 in QUEENS_EXTRA:
        return "4"
    return NYC_PREFIX.get(z5[:3])


def absentee_categories(df: pd.DataFrame, conn) -> pd.DataFrame:
    zips = pd.read_csv(ZIPS, dtype=str)
    zips["bbl_key"] = zips["bbl"].str[:10]
    zips["prop_zip"] = zips["zipcode"].str[:5]
    zips = zips.dropna(subset=["prop_zip"]).drop_duplicates("bbl_key")[["bbl_key", "prop_zip"]]

    # name-matched deed addresses (see rebuild_owner_geo_pandas.py); the raw
    # owner_mailing_zip table contains lender/servicer addresses
    om = pd.read_csv(AN / "owner_zip_namematch.csv", dtype=str)
    om["bbl_key"] = om["bbl_key"].astype(str)
    om = om.drop_duplicates("bbl_key")

    df = df.merge(zips, on="bbl_key", how="left").merge(om, on="bbl_key", how="left")

    ozip_boro = df["owner_zip5"].map(zip_to_boro)
    in_nyc = ozip_boro.notna() & (df["owner_state"].isin(["NY", None, np.nan]) | df["owner_state"].isna())
    has = df["owner_zip5"].notna() & df["owner_zip5"].str.isdigit()

    cat = pd.Series("unknown", index=df.index)
    cat[has & (df["owner_zip5"] == df["prop_zip"])] = "same_zip"
    cat[has & (df["owner_zip5"] != df["prop_zip"]) & in_nyc] = "nyc_other"
    cat[has & ~in_nyc] = "outside_nyc"
    df["owner_geo"] = cat
    print("owner_geo:", df["owner_geo"].value_counts(normalize=True).round(3).to_dict())
    return df


def portfolio_flag(df: pd.DataFrame, conn) -> pd.DataFrame:
    own = pd.read_sql_query(
        "SELECT borocode, block, lot, ownername FROM pluto WHERE ownername IS NOT NULL", conn)
    own["norm"] = own["ownername"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    own = own[own["norm"].str.len() >= 5]
    counts = own["norm"].value_counts()
    df["owner_norm"] = df["ownername_norm"] if "ownername_norm" in df else np.nan
    # re-derive from db by key join (panel dropped ownername)
    from analysis_config import make_bbl
    own["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                      in zip(own["borocode"], own["block"], own["lot"])]
    own = own.drop_duplicates("bbl_key")[["bbl_key", "norm"]]
    df = df.drop(columns=["owner_norm"]).merge(own, on="bbl_key", how="left")
    df["n_owner_props"] = df["norm"].map(counts)
    df["multi_prop_owner"] = ((df["n_owner_props"] >= 3)
                              & df["owner_type"].isin(["individual", "llc", "corp_other", "trust_estate"])
                              ).astype(int)
    print(f"multi_prop_owner (3+ lots, exact name): {df['multi_prop_owner'].mean():.3f}")
    return df


def parse_surname(name) -> str:
    if not isinstance(name, str):
        return ""
    s = name.split(",")[0].strip() if "," in name else name.strip().split(" ")[0]
    s = re.sub(r"[^A-Z\-']", "", s.upper())
    return s if len(s) >= 2 else ""


def bisg(df: pd.DataFrame, conn) -> pd.DataFrame:
    import surgeo
    names = pd.read_sql_query(
        "SELECT borocode, block, lot, ownername FROM pluto WHERE ownername IS NOT NULL", conn)
    from analysis_config import make_bbl
    names["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                        in zip(names["borocode"], names["block"], names["lot"])]
    names = names.drop_duplicates("bbl_key").set_index("bbl_key")["ownername"]

    mask = (df["owner_type"] == "individual") & (df["unitsres"] < 16)
    sub = df.loc[mask, ["bbl_key", "owner_occ_star", "prop_zip", "owner_zip5"]].copy()
    sub["surname"] = sub["bbl_key"].map(names).apply(parse_surname)
    sub = sub[sub["surname"] != ""]

    uniq = pd.Series(sorted(sub["surname"].unique()))
    print(f"BISG: {len(sub):,} properties, {len(uniq):,} unique surnames")
    sm = surgeo.SurnameModel().get_probabilities(uniq)
    sm = sm.set_index("name")
    sm = sm[~sm.index.duplicated()]
    sm["asian"] = sm["api"]
    four = sm[["white", "black", "asian", "hispanic"]]
    four = four.div(four.sum(axis=1), axis=0)  # renormalize to 4 groups
    for c in four.columns:
        sub[f"sn_{c}"] = sub["surname"].map(four[c])
    sub = sub.dropna(subset=["sn_white"])
    print(f"  matched to surname table: {len(sub):,}")

    # ZCTA owner-occupant race prior
    data = json.loads(ZCTA_RAW.read_text())
    z = pd.DataFrame(data[1:], columns=data[0]).rename(columns={
        "B25003H_002E": "w", "B25003B_002E": "b", "B25003D_002E": "a",
        "B25003I_002E": "h", "zip code tabulation area": "zcta"})
    for c in ["w", "b", "a", "h"]:
        z[c] = pd.to_numeric(z[c], errors="coerce").clip(lower=0)
    z["tot"] = z[["w", "b", "a", "h"]].sum(axis=1)
    z = z[z["tot"] >= 30]
    for c, r in zip(["w", "b", "a", "h"], ["white", "black", "asian", "hispanic"]):
        z[f"pr_{r}"] = (z[c] / z["tot"]).clip(0.005, 0.995)
    z = z.set_index("zcta")[[f"pr_{r}" for r in NATL]]

    best = np.where(sub["owner_occ_star"].astype(bool), sub["prop_zip"], sub["owner_zip5"])
    best = pd.Series(best, index=sub.index).fillna(sub["prop_zip"])
    for r in NATL:
        sub[f"prior_{r}"] = best.map(z[f"pr_{r}"])
    have_prior = sub["prior_white"].notna()
    print(f"  ZCTA prior coverage: {have_prior.mean():.1%}")

    # posterior_r ∝ [P(r|surname)/P(r)] * P(r|zip); fallback = surname-only
    post = {}
    for r in NATL:
        post[r] = np.where(have_prior,
                           sub[f"sn_{r}"] / NATL[r] * sub[f"prior_{r}"],
                           sub[f"sn_{r}"])
    tot = sum(post.values())
    for r in NATL:
        sub[f"p_{r}"] = post[r] / tot

    keep = ["bbl_key"] + [f"p_{r}" for r in NATL] + [f"sn_{r}" for r in NATL]
    out = sub[keep]
    print("  mean posterior:", {r: round(out[f'p_{r}'].mean(), 3) for r in NATL})
    print("  mean surname-only:", {r: round(out[f'sn_{r}'].mean(), 3) for r in NATL})
    return df.merge(out, on="bbl_key", how="left")


def main():
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str,
                                   "block": str, "lot": str})
    df["bbl_key"] = df["bbl_key"].astype(str)
    conn = sqlite3.connect(str(config.DB_PATH), timeout=60)  # busy timeout: live scrape may hold write lock
    conn.execute("PRAGMA busy_timeout=60000;")
    df = absentee_categories(df, conn)
    df = portfolio_flag(df, conn)
    df = bisg(df, conn)
    conn.close()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} x {df.shape[1]} -> {OUT}")


if __name__ == "__main__":
    main()
