"""
Rebuild owner mailing addresses by matching ACRIS party names to the
current PLUTO owner.

The prebuilt owner_mailing_zip table joined party_type=2 rows to lots
without a document-type filter (acris_master is empty), so it contains
mortgage-lender and servicer addresses. Here we accept a party-2 address
only when the party's normalized name equals the lot's current PLUTO
ownername — i.e., the address the current owner themselves listed on
their own recorded document. Output: table owner_zip_namematch
(bbl_key, owner_zip5, owner_state, n_docs).
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

NORM = ("UPPER(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE({c},' ',''),"
        "',',''),'.',''),'-',''),'''',''),'&',''))")


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute("DROP TABLE IF EXISTS owner_zip_namematch")
    q = f"""
    CREATE TABLE owner_zip_namematch AS
    SELECT pl.borocode || printf('%05d', CAST(pl.block AS INTEGER))
           || printf('%04d', CAST(pl.lot AS INTEGER)) AS bbl_key,
           substr(p.zip, 1, 5) AS owner_zip5,
           p.state AS owner_state,
           COUNT(*) AS n_docs
    FROM acris_parties p
    JOIN acris_legals l ON l.document_id = p.document_id
    JOIN pluto pl ON pl.borocode = l.borough
                 AND CAST(pl.block AS INTEGER) = CAST(l.block AS INTEGER)
                 AND CAST(pl.lot AS INTEGER) = CAST(l.lot AS INTEGER)
    WHERE p.zip IS NOT NULL
      AND substr(p.zip,1,5) != '00000'
      AND pl.ownername IS NOT NULL
      AND {NORM.format(c='p.name')} = {NORM.format(c='pl.ownername')}
    GROUP BY 1, 2, 3
    """
    print("Building name-matched owner zip table (scans 11.8M parties)...")
    conn.execute(q)
    conn.commit()
    n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT bbl_key) FROM owner_zip_namematch").fetchone()
    print(f"rows: {n[0]:,}, distinct lots: {n[1]:,}")

    # collapse to one row per lot: the (zip,state) with the most documents
    df = pd.read_sql_query("SELECT * FROM owner_zip_namematch", conn)
    df = (df.sort_values("n_docs", ascending=False)
            .drop_duplicates("bbl_key")[["bbl_key", "owner_zip5", "owner_state"]])
    conn.execute("DROP TABLE owner_zip_namematch")
    df.to_sql("owner_zip_namematch", conn, index=False)
    conn.execute("CREATE INDEX idx_ozn_bbl ON owner_zip_namematch(bbl_key)")
    conn.commit()
    conn.close()
    print(f"final: {len(df):,} lots with name-matched owner zip")
    print(df["owner_state"].value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
