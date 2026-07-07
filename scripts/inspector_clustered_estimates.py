"""
Inspector-clustered estimates for the inspector post.

Produces data/analysis/risk_models/inspector_clustered_estimates.csv, the
source for every regression number and figure whisker in the inspector post
(post_inspector_substack.md / inspector_causal_figure.py).

Blocks written:
  firststage  violation_found ~ LOO strictness under six FE ladders
  ladder365   one-year any-permit and any-new-ECB under six FE ladders
  ladder90    90-day any-permit under the same ladders
  windows     both outcomes at 30/60/90/180/365d inside cat x unit x ym x NTA
  tract       first stage + key downstream outcomes inside census-tract cells
              (identification comes from the ~35% of inspections that share a
              cat x unit x ym x tract cell with another inspection)

All standard errors are clustered by inspector (CRV1), the level at which the
leave-one-out strictness treatment varies. Unclustered errors overstate
precision roughly tenfold here; see the pre-publication audit for the
comparison. Spillover outcomes are estimated (also clustered) in
spatial_spillovers.py, which owns the neighbor-count construction.

Inputs: data/analysis/master_panel.csv (build_analysis_dataset.py),
complaint_nta and pluto_tract tables in the SQLite DB.
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

OUT = config.DATA_DIR / "analysis" / "risk_models" / "inspector_clustered_estimates.csv"
BORO = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4", "STATEN ISLAND": "5"}
WINDOWS = [30, 60, 90, 180, 365]


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "analysis" / "master_panel.csv", low_memory=False)
    df["complaint_number"] = df["complaint_number"].astype(str)
    conn = sqlite3.connect(str(config.DB_PATH))
    nta = pd.read_sql_query("SELECT complaint_number, nta FROM complaint_nta", conn)
    tr = pd.read_sql_query("SELECT borocode, block, lot, bct2020 FROM pluto_tract", conn)
    conn.close()
    nta["complaint_number"] = nta["complaint_number"].astype(str)
    df = df.merge(nta, on="complaint_number", how="left")

    df["borocode"] = df["borough"].map(BORO)
    for frame, cols in ((df, ("block", "lot")), (tr, ("block", "lot"))):
        for c in cols:
            frame[c + "_k"] = pd.to_numeric(frame[c], errors="coerce").astype("Int64").astype(str)
    tr = tr.drop_duplicates(subset=["borocode", "block_k", "lot_k"])
    df = df.merge(tr[["borocode", "block_k", "lot_k", "bct2020"]],
                  on=["borocode", "block_k", "lot_k"], how="left")

    df["cat"] = df["category_description"].fillna("UNK")
    df["cat_unit"] = df["cat"] + "|" + df["assigned_to"].fillna("UNK")
    df["cat_unit_ym"] = df["cat_unit"] + "|" + df["year_month"].fillna("UNK")
    df["cell_boro"] = df["cat_unit_ym"] + "|" + df["borough"].fillna("")
    df["cell_nta"] = df["cat_unit_ym"] + "|" + df["nta"].fillna("UNK")
    df["cell_tract"] = df["cat_unit_ym"] + "|" + df["bct2020"].fillna("UNK")
    hr = pd.to_numeric(df["received_time"].astype(str).str.split(":").str[0], errors="coerce")
    df["time_block"] = pd.cut(hr, bins=[-1, 6, 12, 17, 24],
                              labels=["night", "morn", "aft", "eve"]).astype(str)
    df.loc[hr.isna(), "time_block"] = "unk"
    df["dow"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y",
                               errors="coerce").dt.dayofweek.astype(str)
    df["cell_time"] = df["cat_unit_ym"] + "|" + df["time_block"]
    df["kitchen"] = df["cell_time"] + "|" + df["priority"].fillna("UNK").astype(str) + "|" + df["dow"]
    df["none"] = "all"
    for w in WINDOWS:
        df[f"ecb{w}"] = (df[f"future_ecb_{w}d"] > 0).astype(float)
    return df


def run(df, rows, y, fe, block, label):
    d = df[[y, "loo_strictness", fe, "inspector_badge"]].dropna()
    d.columns = ["y", "x", "fe", "insp"]
    m = pf.feols("y ~ x | fe", data=d, vcov={"CRV1": "insp"})
    b, se = float(m.coef()["x"]), float(m.se()["x"])
    rows.append(dict(block=block, label=label, outcome=y, fe=fe, b=b, se=se, t=b / se, n=int(m._N)))
    print(f"[{block}] {label:<28} {y:<16} b={b:+.4f} se={se:.4f} t={b/se:+6.2f} N={m._N:,}")


def main():
    df = load_panel()
    print(f"panel: {len(df):,}; NTA matched {df['nta'].notna().mean():.1%}; "
          f"tract matched {df['bct2020'].notna().mean():.1%}")
    sizes = df.groupby("cell_tract")["complaint_number"].transform("size")
    print(f"share of inspections in singleton tract cells: {(sizes == 1).mean():.1%}")

    LADDER = [("No controls", "none"), ("Complaint category", "cat"),
              ("Category x unit", "cat_unit"), ("Category x unit x month", "cat_unit_ym"),
              ("+ borough", "cell_boro"), ("+ NTA (neighborhood)", "cell_nta")]
    FS = [("No controls", "none"), ("Category", "cat"), ("Category x unit", "cat_unit"),
          ("Cat x unit x month", "cat_unit_ym"), ("+ time of day", "cell_time"),
          ("+ priority + day", "kitchen")]

    rows = []
    for lbl, fe in FS:
        run(df, rows, "violation_found", fe, "firststage", lbl)
    for lbl, fe in LADDER:
        run(df, rows, "any_permit_365d", fe, "ladder365", lbl)
        run(df, rows, "ecb365", fe, "ladder365", lbl)
    for lbl, fe in LADDER:
        run(df, rows, "any_permit_90d", fe, "ladder90", lbl)
    for w in WINDOWS:
        run(df, rows, f"any_permit_{w}d", "cell_nta", "windows", f"{w}d")
        run(df, rows, f"ecb{w}", "cell_nta", "windows", f"{w}d")
    for y, lbl in [("violation_found", "first stage"), ("any_permit_30d", "permit 30d"),
                   ("any_permit_90d", "permit 90d"), ("any_permit_365d", "permit 365d"),
                   ("ecb365", "new ECB 365d")]:
        run(df, rows, y, "cell_tract", "tract", lbl)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
