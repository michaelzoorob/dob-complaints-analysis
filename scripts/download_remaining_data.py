"""Resume downloading remaining auxiliary datasets:
1. ECB violations (from offset 1,250,000)
2. DOB violations (full)
3. DOB Safety Violations (new dataset, 855j-jady)
"""

import sqlite3
import sys
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

PAGE_SIZE = 50_000


def download_paginated(api_url, fields, table_name, conn, start_offset=0):
    """Download from Socrata API with pagination."""
    select = ",".join(fields)
    offset = start_offset
    total = start_offset

    while True:
        url = f"{api_url}?$select={select}&$limit={PAGE_SIZE}&$offset={offset}&$order=:id"
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Error at offset {offset}: {e}. Retrying in 10s...")
            import time; time.sleep(10)
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
            except Exception as e2:
                print(f"  Retry failed: {e2}. Stopping.")
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
        print(f"  {table_name}: {total:,} rows (offset {offset:,})...", flush=True)

        if len(rows) < PAGE_SIZE:
            break

    conn.commit()
    return total


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    # 1. Resume ECB violations
    ecb_count = conn.execute("SELECT COUNT(*) FROM ecb_violations").fetchone()[0]
    if ecb_count < 1_800_000:
        print(f"\n=== Resuming ECB violations from offset {ecb_count:,} ===")
        ecb_fields = [
            "ecb_violation_number", "ecb_violation_status",
            "dob_violation_number", "bin", "boro", "block", "lot",
            "issue_date", "served_date", "hearing_date", "severity",
            "violation_type", "violation_description", "respondent_name",
            "penality_imposed", "amount_paid", "balance_due",
            "hearing_status", "certification_status",
        ]
        download_paginated(
            "https://data.cityofnewyork.us/resource/6bgk-3dad.json",
            ecb_fields, "ecb_violations", conn, start_offset=ecb_count
        )
    else:
        print(f"ECB violations already complete: {ecb_count:,}")

    # 2. DOB violations (full download)
    try:
        dob_count = conn.execute("SELECT COUNT(*) FROM dob_violations").fetchone()[0]
    except:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dob_violations (
                isn_dob_bis_viol TEXT, boro TEXT, block TEXT, lot TEXT,
                issue_date TEXT, violation_type_code TEXT, violation_number TEXT,
                house_number TEXT, street TEXT, disposition_comments TEXT,
                description TEXT, violation_category TEXT, violation_type TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_dobv_bbl ON dob_violations(boro, block, lot);
            CREATE INDEX IF NOT EXISTS idx_dobv_issue ON dob_violations(issue_date);
        """)
        dob_count = 0

    if dob_count < 2_400_000:
        print(f"\n=== Downloading DOB violations (from offset {dob_count:,}) ===")
        dob_fields = [
            "isn_dob_bis_viol", "boro", "block", "lot", "issue_date",
            "violation_type_code", "violation_number", "house_number",
            "street", "disposition_comments", "description",
            "violation_category", "violation_type",
        ]
        download_paginated(
            "https://data.cityofnewyork.us/resource/3h2n-5cm9.json",
            dob_fields, "dob_violations", conn, start_offset=dob_count
        )
    else:
        print(f"DOB violations already complete: {dob_count:,}")

    # 3. DOB Safety Violations (new dataset)
    try:
        safety_count = conn.execute("SELECT COUNT(*) FROM dob_safety_violations").fetchone()[0]
    except:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dob_safety_violations (
                violation_number TEXT,
                violation_type TEXT,
                violation_status TEXT,
                violation_issue_date TEXT,
                violation_remarks TEXT,
                device_type TEXT,
                bin TEXT,
                bbl TEXT,
                borough TEXT,
                block TEXT,
                lot TEXT,
                house_number TEXT,
                street TEXT,
                zip TEXT,
                latitude TEXT,
                longitude TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_safety_bin ON dob_safety_violations(bin);
            CREATE INDEX IF NOT EXISTS idx_safety_bbl ON dob_safety_violations(bbl);
            CREATE INDEX IF NOT EXISTS idx_safety_date ON dob_safety_violations(violation_issue_date);
        """)
        safety_count = 0

    if safety_count < 1_000_000:
        print(f"\n=== Downloading DOB Safety Violations (from offset {safety_count:,}) ===")
        safety_fields = [
            "violation_number", "violation_type", "violation_status",
            "violation_issue_date", "violation_remarks", "device_type",
            "bin", "bbl", "borough", "block", "lot",
            "house_number", "street", "zip", "latitude", "longitude",
        ]
        download_paginated(
            "https://data.cityofnewyork.us/resource/855j-jady.json",
            safety_fields, "dob_safety_violations", conn, start_offset=safety_count
        )
    else:
        print(f"Safety violations already complete: {safety_count:,}")

    # Summary
    print(f"\n{'='*60}")
    print("FINAL COUNTS")
    print(f"{'='*60}")
    for table in ["permits", "permits_now", "ecb_violations", "dob_violations", "dob_safety_violations"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:25s}: {count:>12,} rows")
        except:
            print(f"  {table:25s}: not created")

    conn.close()


if __name__ == "__main__":
    main()
