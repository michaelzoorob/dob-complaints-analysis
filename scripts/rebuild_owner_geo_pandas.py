"""Pandas fallback for rebuild_owner_geo.py (same logic, vectorized).
Writes data/analysis/owner_zip_namematch.csv instead of a DB table."""

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl

OUT = config.DATA_DIR / "analysis" / "owner_zip_namematch.csv"


def norm(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True))


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    print("loading parties (valid zip)...")
    p = pd.read_sql_query("""
        SELECT document_id, name, substr(zip,1,5) AS owner_zip5, state AS owner_state
        FROM acris_parties
        WHERE zip IS NOT NULL AND substr(zip,1,5) != '00000'
    """, conn)
    p["pname"] = norm(p["name"])
    p = p[p["pname"].str.len() >= 5].drop(columns=["name"])
    print(f"  {len(p):,} party rows")

    print("loading legals...")
    l = pd.read_sql_query("SELECT document_id, borough, block, lot FROM acris_legals", conn)
    l["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                    in zip(l["borough"], l["block"], l["lot"])]
    l = l[l["bbl_key"] != ""][["document_id", "bbl_key"]].drop_duplicates()
    print(f"  {len(l):,} legal rows")

    print("loading pluto owner names...")
    pl = pd.read_sql_query(
        "SELECT borocode, block, lot, ownername FROM pluto WHERE ownername IS NOT NULL", conn)
    conn.close()
    pl["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                     in zip(pl["borocode"], pl["block"], pl["lot"])]
    pl["oname"] = norm(pl["ownername"])
    pl = pl[pl["oname"].str.len() >= 5].drop_duplicates("bbl_key")[["bbl_key", "oname"]]

    print("joining...")
    m = p.merge(l, on="document_id")
    print(f"  party x lot: {len(m):,}")
    m = m.merge(pl, on="bbl_key")
    m = m[m["pname"] == m["oname"]]
    print(f"  name-matched: {len(m):,} rows, {m['bbl_key'].nunique():,} lots")

    m = (m.groupby(["bbl_key", "owner_zip5", "owner_state"]).size()
          .rename("n_docs").reset_index()
          .sort_values("n_docs", ascending=False)
          .drop_duplicates("bbl_key"))
    m[["bbl_key", "owner_zip5", "owner_state"]].to_csv(OUT, index=False)
    print(f"saved {len(m):,} lots -> {OUT}")
    print(m["owner_state"].value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
