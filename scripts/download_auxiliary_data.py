"""
Download auxiliary NYC Open Data datasets for the inspector leniency analysis:
1. DOB Permits (rbx6-tga4) — 919K rows
2. DOB NOW Permits (w9ak-ipjd) — 886K rows
3. DOB ECB Violations (6bgk-3dad) — 1.8M rows
4. DOB Violations (3h2n-5cm9) — 2.5M rows (BIS-era ledger)
5. DOB Safety Violations (855j-jady) — 1.1M rows (DOB NOW-era ledger; overlaps
   the BIS ledger, so counting code must dedupe across systems — see
   scripts/dob_ledger.py)

All loaded into SQLite for joining with complaints on BBL (boro+block+lot) or BIN.
"""

import sqlite3
import sys
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

DATASETS = {
    "permits": {
        "api": "https://data.cityofnewyork.us/resource/rbx6-tga4.json",
        "table": "permits",
        "schema": """
            CREATE TABLE IF NOT EXISTS permits (
                job_filing_number TEXT,
                work_permit TEXT,
                filing_reason TEXT,
                house_no TEXT,
                street_name TEXT,
                borough TEXT,
                block TEXT,
                lot TEXT,
                bin TEXT,
                bbl TEXT,
                work_type TEXT,
                permit_status TEXT,
                approved_date TEXT,
                issued_date TEXT,
                expired_date TEXT,
                job_description TEXT,
                estimated_job_costs TEXT,
                owner_business_name TEXT,
                owner_name TEXT,
                zip_code TEXT,
                latitude TEXT,
                longitude TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_permits_bin ON permits(bin);
            CREATE INDEX IF NOT EXISTS idx_permits_bbl ON permits(bbl);
            CREATE INDEX IF NOT EXISTS idx_permits_issued ON permits(issued_date);
        """,
        "fields": [
            "job_filing_number", "work_permit", "filing_reason",
            "house_no", "street_name", "borough", "block", "lot",
            "bin", "bbl", "work_type", "permit_status",
            "approved_date", "issued_date", "expired_date",
            "job_description", "estimated_job_costs",
            "owner_business_name", "owner_name", "zip_code",
            "latitude", "longitude",
        ],
    },
    "permits_now": {
        "api": "https://data.cityofnewyork.us/resource/w9ak-ipjd.json",
        "table": "permits_now",
        "schema": """
            CREATE TABLE IF NOT EXISTS permits_now (
                job_filing_number TEXT,
                filing_status TEXT,
                house_no TEXT,
                street_name TEXT,
                borough TEXT,
                block TEXT,
                lot TEXT,
                bin TEXT,
                bbl TEXT,
                job_type TEXT,
                building_type TEXT,
                initial_cost TEXT,
                total_construction_floor_area TEXT,
                existing_dwelling_units TEXT,
                proposed_dwelling_units TEXT,
                filing_date TEXT,
                approved_date TEXT,
                first_permit_date TEXT,
                signoff_date TEXT,
                current_status_date TEXT,
                owner_s_business_name TEXT,
                latitude TEXT,
                longitude TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pnow_bin ON permits_now(bin);
            CREATE INDEX IF NOT EXISTS idx_pnow_bbl ON permits_now(bbl);
            CREATE INDEX IF NOT EXISTS idx_pnow_filing ON permits_now(filing_date);
        """,
        "fields": [
            "job_filing_number", "filing_status", "house_no", "street_name",
            "borough", "block", "lot", "bin", "bbl", "job_type",
            "building_type", "initial_cost", "total_construction_floor_area",
            "existing_dwelling_units", "proposed_dwelling_units",
            "filing_date", "approved_date", "first_permit_date",
            "signoff_date", "current_status_date",
            "owner_s_business_name", "latitude", "longitude",
        ],
    },
    "ecb_violations": {
        "api": "https://data.cityofnewyork.us/resource/6bgk-3dad.json",
        "table": "ecb_violations",
        "schema": """
            CREATE TABLE IF NOT EXISTS ecb_violations (
                ecb_violation_number TEXT,
                ecb_violation_status TEXT,
                dob_violation_number TEXT,
                bin TEXT,
                boro TEXT,
                block TEXT,
                lot TEXT,
                issue_date TEXT,
                served_date TEXT,
                hearing_date TEXT,
                severity TEXT,
                violation_type TEXT,
                violation_description TEXT,
                respondent_name TEXT,
                penality_imposed TEXT,
                amount_paid TEXT,
                balance_due TEXT,
                hearing_status TEXT,
                certification_status TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ecb_bin ON ecb_violations(bin);
            CREATE INDEX IF NOT EXISTS idx_ecb_bbl ON ecb_violations(boro, block, lot);
            CREATE INDEX IF NOT EXISTS idx_ecb_issue ON ecb_violations(issue_date);
        """,
        "fields": [
            "ecb_violation_number", "ecb_violation_status",
            "dob_violation_number", "bin", "boro", "block", "lot",
            "issue_date", "served_date", "hearing_date", "severity",
            "violation_type", "violation_description", "respondent_name",
            "penality_imposed", "amount_paid", "balance_due",
            "hearing_status", "certification_status",
        ],
    },
    "dob_violations": {
        "api": "https://data.cityofnewyork.us/resource/3h2n-5cm9.json",
        "table": "dob_violations",
        "schema": """
            CREATE TABLE IF NOT EXISTS dob_violations (
                isn_dob_bis_viol TEXT,
                boro TEXT,
                block TEXT,
                lot TEXT,
                issue_date TEXT,
                violation_type_code TEXT,
                violation_number TEXT,
                house_number TEXT,
                street TEXT,
                disposition_comments TEXT,
                description TEXT,
                violation_category TEXT,
                violation_type TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_dobv_bbl ON dob_violations(boro, block, lot);
            CREATE INDEX IF NOT EXISTS idx_dobv_issue ON dob_violations(issue_date);
        """,
        "fields": [
            "isn_dob_bis_viol", "boro", "block", "lot", "issue_date",
            "violation_type_code", "violation_number", "house_number",
            "street", "disposition_comments", "description",
            "violation_category", "violation_type",
        ],
    },
    "dob_safety_violations": {
        "api": "https://data.cityofnewyork.us/resource/855j-jady.json",
        "table": "dob_safety_violations",
        "schema": """
            CREATE TABLE IF NOT EXISTS dob_safety_violations (
                violation_number TEXT,
                violation_type TEXT,
                violation_status TEXT,
                device_type TEXT,
                bin TEXT,
                borough TEXT,
                block TEXT,
                lot TEXT,
                bbl TEXT,
                violation_issue_date TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_dsv_bbl ON dob_safety_violations(bbl);
            CREATE INDEX IF NOT EXISTS idx_dsv_issue ON dob_safety_violations(violation_issue_date);
        """,
        "fields": [
            "violation_number", "violation_type", "violation_status",
            "device_type", "bin", "borough", "block", "lot", "bbl",
            "violation_issue_date",
        ],
    },
}

PAGE_SIZE = 50_000


def download_dataset(name: str, spec: dict, conn: sqlite3.Connection):
    """Download a dataset from Socrata API into SQLite."""
    existing = conn.execute(f"SELECT COUNT(*) FROM {spec['table']}").fetchone()[0]
    if existing > 0:
        print(f"  {spec['table']} already has {existing:,} rows. Skipping.")
        return existing

    select = ",".join(spec["fields"])
    offset = 0
    total = 0

    while True:
        url = f"{spec['api']}?$select={select}&$limit={PAGE_SIZE}&$offset={offset}&$order=:id"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        df = pd.DataFrame(rows)
        # Ensure all expected columns exist
        for col in spec["fields"]:
            if col not in df.columns:
                df[col] = None
        df = df[spec["fields"]]

        df.to_sql(spec["table"], conn, if_exists="append", index=False)
        total += len(df)
        offset += PAGE_SIZE
        print(f"  {name}: {total:,} rows (offset {offset:,})...")

        if len(rows) < PAGE_SIZE:
            break

    conn.commit()
    print(f"  {name}: Done — {total:,} rows loaded.")
    return total


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    for name, spec in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Downloading {name}...")
        print(f"{'='*60}")

        conn.executescript(spec["schema"])

        try:
            download_dataset(name, spec, conn)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, spec in DATASETS.items():
        count = conn.execute(f"SELECT COUNT(*) FROM {spec['table']}").fetchone()[0]
        print(f"  {spec['table']:20s}: {count:>12,} rows")

    conn.close()


if __name__ == "__main__":
    main()
