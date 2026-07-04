"""Download the BIN -> BBL crosswalk from the Building Footprints dataset
(Socrata 5zhs-2jue: bin, base_bbl). Enables joining pre-2020 open_data
complaints (which carry only BIN) to tax lots for the owner-transition
panel."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from download_acris_owner_addresses import download_paginated


def main():
    conn = sqlite3.connect(str(config.DB_PATH), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS bin_bbl (
            bin TEXT,
            base_bbl TEXT
        );
    """)
    download_paginated(
        "https://data.cityofnewyork.us/resource/5zhs-2jue.json",
        ["bin", "base_bbl"], "bin_bbl", conn,
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bin_bbl ON bin_bbl(bin)")
    conn.commit()
    n = conn.execute("SELECT COUNT(*), COUNT(DISTINCT bin) FROM bin_bbl").fetchone()
    print(f"bin_bbl rows: {n[0]:,}, distinct bins: {n[1]:,}")
    conn.close()


if __name__ == "__main__":
    main()
