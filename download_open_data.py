"""Download DOB Complaints Received from NYC Open Data and load into SQLite."""

import sys
from pathlib import Path

import requests

import config
import db


def download_csv(output_path: Path) -> Path:
    """Stream-download the full Socrata CSV export."""
    print(f"Downloading from Socrata to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(config.OPEN_DATA_CSV_URL, stream=True, timeout=300)
    resp.raise_for_status()

    total_bytes = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            f.write(chunk)
            total_bytes += len(chunk)
            print(f"\r  Downloaded {total_bytes / 1e6:.0f} MB", end="", flush=True)

    print(f"\n  Done: {total_bytes / 1e6:.1f} MB")
    return output_path


def run():
    """Download Open Data CSV and load into SQLite."""
    csv_path = config.DATA_DIR / "open_data_raw.csv"

    # Download if not already present
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / 1e6
        print(f"CSV already exists ({size_mb:.0f} MB). Skipping download.")
        print("  Delete data/open_data_raw.csv to re-download.")
    else:
        download_csv(csv_path)

    # Load into SQLite
    print(f"\nLoading CSV into SQLite ({config.DB_PATH}) ...")
    conn = db.get_connection()
    db.init_db(conn)

    existing = conn.execute("SELECT COUNT(*) FROM open_data").fetchone()[0]
    if existing > 0:
        print(f"  open_data table already has {existing:,} rows. Skipping load.")
        print("  To reload, drop the table first.")
    else:
        total = db.load_open_data(csv_path, conn)
        print(f"  Loaded {total:,} rows into open_data table.")

    # Populate scrape queue
    print(f"\nPopulating scrape queue (min_year={config.SCRAPE_MIN_YEAR}) ...")
    pending = db.populate_scrape_queue(conn, config.SCRAPE_MIN_YEAR)
    print(f"  {pending:,} complaints queued for scraping.")

    conn.close()


if __name__ == "__main__":
    run()
