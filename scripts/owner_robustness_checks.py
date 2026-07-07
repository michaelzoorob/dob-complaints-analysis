"""
Robustness checks quoted in the owner-race and overview posts.

1. PURCHASE RECENCY (Asian-owner post, BISG section). Asian-classified owners
   bought much more recently than white-classified owners, recent purchase is
   a first-order complaint predictor, and adding deed-recency controls to the
   headline specification moves the p_asian complaint coefficient from about
   +37% to about +28%. Quantifies the non-classical BISG error channel
   (McCartan et al. 2024): the cross-sectional gaps bundle ownership tenure
   with race. The ownership-transition design (owner_transition_panel.py)
   compares buyers matched on purchase timing and is the recency-free
   complement.

2. LEDGER SPLIT (overview/risk posts, owner-occupancy). The published
   owner-occupancy effect on DOB-ledger violations uses the deduplicated
   union of the BIS-era and DOB NOW-era datasets (about -10% with commercial
   controls). Split by attributed system, the gap sits entirely in the
   BIS-attributed stream (about -22%); the DOB NOW-attributed stream, which
   is dominated by periodic equipment citations, shows no owner-occupancy
   gap. The union number is the correct headline (it counts every violation
   once); this split documents why it is smaller than the BIS-only
   continuity row in citation_tidy_estimates.csv.

Reads the risk panel via owner_models.load_frame() plus ACRIS deed dates and
the unified violations ledger. Prints results; writes nothing.
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import dob_ledger
import owner_models as om

warnings.filterwarnings("ignore")
VCOV = {"CRV1": "bct2020"}


def add_deed_recency(df: pd.DataFrame) -> pd.DataFrame:
    conn = sqlite3.connect(str(config.DB_PATH))
    deeds = pd.read_sql_query("""
        SELECT l.borough || printf('%05d', CAST(l.block AS INT))
                         || printf('%04d', CAST(l.lot AS INT)) AS bbl_key,
               MAX(substr(m.doc_date, 1, 4)) AS last_deed_yr
        FROM acris_master m JOIN acris_legals l ON m.document_id = l.document_id
        WHERE m.doc_type LIKE 'DEED%' AND m.doc_date IS NOT NULL AND m.doc_date >= '1985'
        GROUP BY 1""", conn)
    conn.close()
    deeds["bbl_key"] = deeds["bbl_key"].astype(str)
    deeds["last_deed_yr"] = pd.to_numeric(deeds["last_deed_yr"], errors="coerce")
    df = df.merge(deeds, on="bbl_key", how="left")
    df["deed_2020p"] = (df["last_deed_yr"] >= 2020).astype(int)
    df["deed_1519"] = df["last_deed_yr"].between(2015, 2019).astype(int)
    df["deed_missing"] = df["last_deed_yr"].isna().astype(int)
    return df


def recency_check(df: pd.DataFrame):
    print("=" * 72)
    print("1. PURCHASE-RECENCY ROBUSTNESS (headline BISG complaint model)")
    print("=" * 72)
    bs = df[df["p_white"].notna() & (df["owner_type"] == "individual")
            & (df["unitsres"] < 16)].copy()
    print(f"BISG subsample: {len(bs):,}; deed matched {bs['last_deed_yr'].notna().mean():.1%}")
    for g, p in [("white", "p_white"), ("asian", "p_asian")]:
        s = bs[bs[p] > 0.7]
        print(f"  {g}-classified: bought 2015+ {(s['last_deed_yr'] >= 2015).mean() * 100:.1f}%, "
              f"2020+ {s['deed_2020p'].mean() * 100:.1f}%")
    bisg_bc = ["owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
               "com_class", "log_bldgarea", "mzone", "multi_bldg", "log2_area_per_unit",
               "value_rank", "any_prior_viol"]
    xr = " + ".join(list(om.RACE) + bisg_bc + list(om.OWNER_COVARS))
    m0 = pf.fepois(f"n_complaints ~ {xr} | size_bin + bct2020", data=bs, vcov=VCOV)
    m1 = pf.fepois(f"n_complaints ~ {xr} + deed_2020p + deed_1519 + deed_missing "
                   f"| size_bin + bct2020", data=bs, vcov=VCOV)
    for tag, m in (("baseline", m0), ("+ recency", m1)):
        b = float(m.coef()["p_asian"])
        print(f"  p_asian {tag:<10}: b={b:.4f} ({(np.exp(b) - 1) * 100:+.1f}%)")
    for t in ("deed_2020p", "deed_1519"):
        print(f"  {t}: {(np.exp(float(m1.coef()[t])) - 1) * 100:+.1f}% complaints")


def ledger_split(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("2. OWNER-OCCUPANCY BY VIOLATION SYSTEM (union components)")
    print("=" * 72)
    conn = sqlite3.connect(str(config.DB_PATH))
    u = dob_ledger.union_frame(conn, verbose=False)
    conn.close()
    u = u[u["year"] >= 2020]
    u["bbl_key"] = u["bbl_key"].astype(str)
    for src, col in (("bis", "n_bis"), ("dobnow", "n_now")):
        c = u[u["source"] == src].groupby("bbl_key").size().rename(col)
        df = df.merge(c, on="bbl_key", how="left")
        df[col] = df[col].fillna(0)
    X = " + ".join(list(om.BUILDING_COVARS) + list(om.OWNER_COVARS))
    for col, lab in (("n_bis", "BIS-attributed"), ("n_now", "DOB NOW-attributed")):
        m = pf.fepois(f"{col} ~ {X} | size_bin + comm_bin + bct2020", data=df, vcov=VCOV)
        b = float(m.coef()["owner_occ_star"])
        p = float(m.pvalue()["owner_occ_star"])
        print(f"  owner_occ_star on {lab:<20}: {(np.exp(b) - 1) * 100:+.1f}% (p={p:.2g})")
    print("  (published union-ledger effect with the same controls: about -10%; "
          "see citation_tidy_estimates.csv ppml_dobviol)")


def main():
    df = om.load_frame()
    df["bbl_key"] = df["bbl_key"].astype(str)
    df = add_deed_recency(df)
    recency_check(df)
    ledger_split(df)


if __name__ == "__main__":
    main()
