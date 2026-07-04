"""
Unified DOB violations ledger across two administrative systems.

Sources:
  dob_violations         (Socrata 3h2n-5cm9) — BIS-era ledger, 2.47M rows
  dob_safety_violations  (Socrata 855j-jady) — DOB NOW-era ledger, 1.10M rows

NYC Open Data warns that the DOB NOW dataset overlaps the BIS ledger:
during and after the BIS -> DOB NOW migration the same violation can be
recorded in both systems, so summing raw rows double-counts. The violation
numbering schemes differ (BIS "isn" numbers vs "VIO-..." strings), so exact
ID joins are impossible. Instead both vocabularies are mapped to coarse
type families (boiler, elevator, facade, energy, gas/plumbing,
construction, other) and a DOB NOW row is treated as a duplicate when a
BIS row exists for the same tax lot, same issue date, and same family.
This is deliberately aggressive: several same-day same-family DOB NOW rows
at one lot all count as duplicates of a single BIS row, which biases the
union DOWN rather than double-counting up.

Snapshot note: the local BIS table is a point-in-time download and can end
slightly earlier than the DOB NOW table; duplicates in that trailing gap
(bounded by the BIS rows not yet downloaded, <0.3% of the 2020+ union)
go undetected. Re-running download_auxiliary_data.py refreshes both.

Public API:
  load_ledgers(conn)      -> (bis, now) normalized DataFrames
                             [bbl_key, ymd, year, family, source]
  union_frame(conn)       -> single DataFrame of BIS rows + non-duplicate
                             DOB NOW rows
  counts_by_bbl(conn, y0, y1, col) -> per-lot counts from the deduped union
  overlap_report(conn)    -> printed per-year diagnostics (run as script)
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl

BORO_CODE = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4",
             "STATEN ISLAND": "5"}

ELEV_KEYS = ("AEU", "EVCAT", "HVCAT", "VCAT", "ELEV", "ACC1", "JVIO",
             "VT-", "-VT", "LL1081")
BOILER_KEYS = ("LBLVIO", "HBLVIO", "BLVIO", "BOILER", "LL6291", "BE-EXT")
FACADE_KEYS = ("FISP", "FACADE", "LL11")
ENERGY_KEYS = ("BENCH", "EGRADE", "EGRDEA", "EARCX", "EN-")
GAS_KEYS = ("PL-", "GAS", "GPS", "PLUMB", "LL152")
EXACT = {"E": "elev", "B": "boiler", "P": "gas_plumb", "C": "constr"}

DEVICE_FAMILY = {
    "Boiler": "boiler", "AEUHAZ": "elev", "Elevators": "elev",
    "Facades": "facade", "Benchmarking - LL84": "energy",
    "Energy Grade - LL33": "energy", "Retro-Commissioning - LL87": "energy",
    "Gas Piping - LL152": "gas_plumb",
}


def _family_from_code(code) -> str:
    c = (code if isinstance(code, str) else "").strip().upper()
    if c in EXACT:
        return EXACT[c]
    for keys, fam in ((BOILER_KEYS, "boiler"), (ELEV_KEYS, "elev"),
                      (FACADE_KEYS, "facade"), (ENERGY_KEYS, "energy"),
                      (GAS_KEYS, "gas_plumb")):
        if any(k in c for k in keys):
            return fam
    if "CONSTR" in c:
        return "constr"
    return "other"


def load_ledgers(conn):
    bis = pd.read_sql_query("""
        SELECT boro, block, lot, issue_date, violation_type_code
        FROM dob_violations
        WHERE length(issue_date) = 8
          AND substr(issue_date,1,4) BETWEEN '1990' AND '2026'""", conn)
    bis["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                      in zip(bis["boro"], bis["block"], bis["lot"])]
    bis = bis[bis["bbl_key"] != ""].copy()
    bis["ymd"] = bis["issue_date"]
    bis["family"] = bis["violation_type_code"].map(_family_from_code)
    bis["year"] = bis["ymd"].str[:4].astype(int)
    bis["source"] = "bis"

    now = pd.read_sql_query("""
        SELECT bbl, borough, block, lot, violation_issue_date,
               violation_type, device_type
        FROM dob_safety_violations""", conn)
    good_bbl = now["bbl"].fillna("").str.fullmatch(r"\d{10}")
    fallback = [make_bbl(BORO_CODE.get((b if isinstance(b, str) else "").upper(), ""), bl, lt)
                for b, bl, lt in zip(now["borough"], now["block"], now["lot"])]
    now["bbl_key"] = now["bbl"].where(good_bbl, pd.Series(fallback, index=now.index))
    now = now[now["bbl_key"].fillna("") != ""].copy()
    now["ymd"] = (now["violation_issue_date"].str[:10]
                  .str.replace("-", "", regex=False))
    now = now[now["ymd"].str.fullmatch(r"\d{8}").fillna(False)]
    now["family"] = now["device_type"].map(DEVICE_FAMILY)
    miss = now["family"].isna()
    now.loc[miss, "family"] = now.loc[miss, "violation_type"].map(_family_from_code)
    now["year"] = now["ymd"].str[:4].astype(int)
    now = now[(now["year"] >= 1990) & (now["year"] <= 2026)]
    now["source"] = "dobnow"

    cols = ["bbl_key", "ymd", "year", "family", "source"]
    return bis[cols], now[cols]


def union_frame(conn, verbose=False):
    bis, now = load_ledgers(conn)
    bis_keys = set(zip(bis["bbl_key"], bis["ymd"], bis["family"]))
    dup = [(b, y, f) in bis_keys
           for b, y, f in zip(now["bbl_key"], now["ymd"], now["family"])]
    now = now[~pd.Series(dup, index=now.index)]
    if verbose:
        print(f"BIS rows {len(bis):,}; DOB NOW rows kept {len(now):,} "
              f"(dropped {sum(dup):,} as cross-system duplicates)")
    return pd.concat([bis, now], ignore_index=True)


def counts_by_bbl(conn, y0: int, y1: int, col: str) -> pd.Series:
    u = union_frame(conn)
    u = u[(u["year"] >= y0) & (u["year"] <= y1)]
    return u.groupby("bbl_key").size().rename(col)


def overlap_report(conn):
    bis, now = load_ledgers(conn)
    bis_keys = set(zip(bis["bbl_key"], bis["ymd"], bis["family"]))
    bis_day = set(zip(bis["bbl_key"], bis["ymd"]))
    now = now.copy()
    now["dup_fam"] = [(b, y, f) in bis_keys
                      for b, y, f in zip(now["bbl_key"], now["ymd"], now["family"])]
    now["dup_day"] = [(b, y) in bis_day
                      for b, y in zip(now["bbl_key"], now["ymd"])]
    rep = (now.groupby("year")
           .agg(dobnow=("dup_fam", "size"), dup_fam=("dup_fam", "sum"),
                dup_day=("dup_day", "sum")))
    rep["bis"] = bis.groupby("year").size()
    rep["pct_dup_fam"] = (rep.dup_fam / rep.dobnow * 100).round(1)
    rep["pct_dup_day"] = (rep.dup_day / rep.dobnow * 100).round(1)
    rep["union"] = rep["bis"].fillna(0).astype(int) + rep.dobnow - rep.dup_fam
    print(rep[["bis", "dobnow", "dup_fam", "pct_dup_fam", "pct_dup_day",
               "union"]].fillna(0).astype(int, errors="ignore").to_string())
    print("\n2010-2019 totals: BIS {:,}, DOB NOW unique {:,}".format(
        int(rep.loc[2010:2019, "bis"].sum()),
        int((rep.loc[2010:2019, "dobnow"] - rep.loc[2010:2019, "dup_fam"]).sum())))
    print("2020-2026 totals: BIS {:,}, DOB NOW unique {:,}, union {:,}".format(
        int(rep.loc[2020:2026, "bis"].sum()),
        int((rep.loc[2020:2026, "dobnow"] - rep.loc[2020:2026, "dup_fam"]).sum()),
        int(rep.loc[2020:2026, "union"].sum())))


if __name__ == "__main__":
    conn = sqlite3.connect(str(config.DB_PATH))
    overlap_report(conn)
    conn.close()
