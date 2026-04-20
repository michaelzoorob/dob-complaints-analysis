"""
Download ACRIS data to get owner mailing addresses for BISG.

Strategy:
1. Download ACRIS Real Property Legals (8h5j-fqxa) — maps document_id → BBL
2. Download ACRIS Real Property Master (bnx9-e6tj) — filter for deeds
3. Download ACRIS Real Property Parties (636b-3b5g) — has owner mailing addresses
4. For each BBL: find most recent deed → get grantee (buyer) → extract mailing zip
"""

import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

PAGE_SIZE = 50_000


def download_paginated(api_url, fields, table_name, conn, where_clause="", start_offset=0):
    """Download from Socrata API with pagination."""
    existing = 0
    try:
        existing = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        if existing > 0 and start_offset == 0:
            print(f"  {table_name} already has {existing:,} rows. Skipping.")
            return existing
    except:
        pass

    select = ",".join(fields)
    offset = start_offset or existing
    total = offset

    while True:
        url = (f"{api_url}?$select={select}"
               f"{'&$where=' + where_clause if where_clause else ''}"
               f"&$limit={PAGE_SIZE}&$offset={offset}&$order=:id")
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Error at offset {offset}: {e}. Retrying...", flush=True)
            time.sleep(10)
            try:
                resp = requests.get(url, timeout=180)
                resp.raise_for_status()
            except Exception as e2:
                print(f"  Retry failed: {e2}. Stopping.", flush=True)
                break

        rows = resp.json()
        if not rows:
            break

        df = pd.DataFrame(rows)
        for col in fields:
            if col not in df.columns:
                df[col] = None
        df = df[fields]
        df.to_sql(table_name, conn, if_exists="append", index=False)

        total += len(df)
        offset += PAGE_SIZE
        print(f"  {table_name}: {total:,} rows...", flush=True)

        if len(rows) < PAGE_SIZE:
            break

    conn.commit()
    return total


def main():
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")

    # ── 1. ACRIS Real Property Master — get deed document IDs ──────────
    print("=" * 60)
    print("1. ACRIS Real Property Master (deed documents)")
    print("=" * 60)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS acris_master (
            document_id TEXT,
            doc_type TEXT,
            doc_date TEXT,
            doc_amount TEXT,
            recorded_datetime TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_acris_master_doc ON acris_master(document_id);
    """)

    download_paginated(
        "https://data.cityofnewyork.us/resource/bnx9-e6tj.json",
        ["document_id", "doc_type", "doc_date", "doc_amount", "recorded_datetime"],
        "acris_master", conn,
        where_clause="doc_type IN ('DEED','DEEDO','DEEDP','DEED, TS')"
    )

    # ── 2. ACRIS Real Property Legals — map document_id → BBL ──────────
    print("\n" + "=" * 60)
    print("2. ACRIS Real Property Legals (document → BBL)")
    print("=" * 60)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS acris_legals (
            document_id TEXT,
            borough TEXT,
            block TEXT,
            lot TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_acris_legals_doc ON acris_legals(document_id);
        CREATE INDEX IF NOT EXISTS idx_acris_legals_bbl ON acris_legals(borough, block, lot);
    """)

    download_paginated(
        "https://data.cityofnewyork.us/resource/8h5j-fqxa.json",
        ["document_id", "borough", "block", "lot"],
        "acris_legals", conn
    )

    # ── 3. ACRIS Parties — owner mailing addresses ─────────────────────
    print("\n" + "=" * 60)
    print("3. ACRIS Parties (owner mailing addresses)")
    print("=" * 60)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS acris_parties (
            document_id TEXT,
            party_type TEXT,
            name TEXT,
            address_1 TEXT,
            address_2 TEXT,
            city TEXT,
            state TEXT,
            zip TEXT,
            country TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_acris_parties_doc ON acris_parties(document_id);
    """)

    # Only download grantees (party_type=2 = buyer/current owner) with addresses
    download_paginated(
        "https://data.cityofnewyork.us/resource/636b-3b5g.json",
        ["document_id", "party_type", "name", "address_1", "address_2",
         "city", "state", "zip", "country"],
        "acris_parties", conn,
        where_clause="party_type='2' AND address_1 IS NOT NULL AND zip IS NOT NULL"
    )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for t in ["acris_master", "acris_legals", "acris_parties"]:
        try:
            c = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:20s}: {c:>12,} rows")
        except:
            print(f"  {t:20s}: not created")

    conn.close()


if __name__ == "__main__":
    main()
