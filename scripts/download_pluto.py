"""Download NYC PLUTO data from Socrata API and load into SQLite.

PLUTO uses BBL (borough-block-lot) as its key, not BIN.
We join to our complaints data using borocode + block + lot.
"""

import sqlite3
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

PLUTO_API = "https://data.cityofnewyork.us/resource/64uk-42ks.json"
PAGE_SIZE = 50_000

PLUTO_FIELDS = [
    "borocode", "borough", "block", "lot", "bbl",
    "latitude", "longitude",
    "bldgclass", "landuse", "numfloors", "yearbuilt",
    "unitsres", "unitstotal", "bldgarea", "lotarea",
    "assesstot", "zonedist1", "address", "ownername", "numbldgs",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS pluto (
    borocode TEXT,
    borough TEXT,
    block TEXT,
    lot TEXT,
    bbl TEXT,
    latitude REAL,
    longitude REAL,
    bldgclass TEXT,
    landuse TEXT,
    numfloors TEXT,
    yearbuilt TEXT,
    unitsres TEXT,
    unitstotal TEXT,
    bldgarea TEXT,
    lotarea TEXT,
    assesstot TEXT,
    zonedist1 TEXT,
    address TEXT,
    ownername TEXT,
    numbldgs TEXT,
    PRIMARY KEY (borocode, block, lot)
);
CREATE INDEX IF NOT EXISTS idx_pluto_latlon ON pluto(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_pluto_bbl ON pluto(bbl);
"""

# Map our borough names to PLUTO borocodes
BOROUGH_TO_CODE = {
    "MANHATTAN": "1",
    "BRONX": "2",
    "BROOKLYN": "3",
    "QUEENS": "4",
    "STATEN ISLAND": "5",
}


def download_pluto():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.executescript(SCHEMA)

    existing = conn.execute("SELECT COUNT(*) FROM pluto").fetchone()[0]
    if existing > 0:
        print(f"pluto table already has {existing:,} rows. Skipping.")
        print("  DROP TABLE pluto to re-download.")
        conn.close()
        return

    select = ",".join(PLUTO_FIELDS)
    offset = 0
    total = 0

    print("Downloading PLUTO data from Socrata API...")
    while True:
        url = (
            f"{PLUTO_API}?$select={select}"
            f"&$where=latitude IS NOT NULL"
            f"&$limit={PAGE_SIZE}&$offset={offset}"
            f"&$order=bbl"
        )
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        insert_rows = []
        for row in rows:
            values = []
            for f in PLUTO_FIELDS:
                v = row.get(f)
                if f in ("latitude", "longitude") and v is not None:
                    v = float(v)
                values.append(v)
            insert_rows.append(values)

        if insert_rows:
            placeholders = ",".join(["?"] * len(PLUTO_FIELDS))
            conn.executemany(
                f"INSERT OR IGNORE INTO pluto ({','.join(PLUTO_FIELDS)}) VALUES ({placeholders})",
                insert_rows,
            )
            conn.commit()

        total += len(insert_rows)
        offset += PAGE_SIZE
        print(f"  Downloaded {total:,} lots (offset {offset:,})...")

        if len(rows) < PAGE_SIZE:
            break

    print(f"\nDone: {total:,} tax lots loaded into pluto table.")

    # Check coverage against our complaints using borocode + block + lot
    coverage = conn.execute("""
        SELECT COUNT(DISTINCT b.complaint_number)
        FROM bis_scrape b
        JOIN pluto p ON p.borocode = (
            CASE b.borough
                WHEN 'MANHATTAN' THEN '1'
                WHEN 'BRONX' THEN '2'
                WHEN 'BROOKLYN' THEN '3'
                WHEN 'QUEENS' THEN '4'
                WHEN 'STATEN ISLAND' THEN '5'
            END
        ) AND p.block = b.block AND p.lot = b.lot
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
    """).fetchone()[0]
    total_complaints = conn.execute(
        "SELECT COUNT(*) FROM bis_scrape WHERE block IS NOT NULL"
    ).fetchone()[0]
    print(f"Coverage: {coverage:,} / {total_complaints:,} complaints matched to PLUTO ({100*coverage/total_complaints:.1f}%)")

    # Sample a matched record
    sample = conn.execute("""
        SELECT b.complaint_number, b.address, b.borough, b.block, b.lot,
               p.latitude, p.longitude, p.bldgclass, p.numfloors, p.yearbuilt
        FROM bis_scrape b
        JOIN pluto p ON p.borocode = (
            CASE b.borough
                WHEN 'MANHATTAN' THEN '1'
                WHEN 'BRONX' THEN '2'
                WHEN 'BROOKLYN' THEN '3'
                WHEN 'QUEENS' THEN '4'
                WHEN 'STATEN ISLAND' THEN '5'
            END
        ) AND p.block = b.block AND p.lot = b.lot
        LIMIT 1
    """).fetchone()
    if sample:
        print(f"\nSample match:")
        print(f"  Complaint {sample[0]}: {sample[1]}, {sample[2]}")
        print(f"  Block/Lot: {sample[3]}/{sample[4]}")
        print(f"  Coords: ({sample[5]}, {sample[6]})")
        print(f"  Building: class={sample[7]}, floors={sample[8]}, built={sample[9]}")

    conn.close()


if __name__ == "__main__":
    download_pluto()
