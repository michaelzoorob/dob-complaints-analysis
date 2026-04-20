"""Re-parse HTML archive to extract received_time into bis_scrape."""

import gzip
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def main():
    conn = sqlite3.connect(str(config.DB_PATH))

    # Add column if not exists
    try:
        conn.execute("ALTER TABLE bis_scrape ADD COLUMN received_time TEXT")
        conn.commit()
        print("Added received_time column")
    except sqlite3.OperationalError:
        print("received_time column already exists")

    archive = config.HTML_ARCHIVE_DIR
    files = list(archive.glob("*.html.gz"))
    print(f"Re-parsing {len(files):,} HTML files for time-of-day...")

    updated = 0
    batch = []
    for i, f in enumerate(files):
        cnum = f.stem.replace(".html", "")
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                html = fh.read()
        except Exception:
            continue

        m = re.search(
            r'<b>Received:</b>\s*</td>\s*<td[^>]*>&nbsp;&nbsp;[\d/]+(?:&nbsp;)*\s*([\d:]+)',
            html
        )
        if m:
            batch.append((m.group(1), cnum))

        if len(batch) >= 1000:
            conn.executemany(
                "UPDATE bis_scrape SET received_time = ? WHERE complaint_number = ?",
                batch
            )
            conn.commit()
            updated += len(batch)
            batch = []

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1:,} files, {updated:,} with time...")

    if batch:
        conn.executemany(
            "UPDATE bis_scrape SET received_time = ? WHERE complaint_number = ?",
            batch
        )
        conn.commit()
        updated += len(batch)

    # Stats
    total = conn.execute("SELECT COUNT(*) FROM bis_scrape").fetchone()[0]
    with_time = conn.execute(
        "SELECT COUNT(*) FROM bis_scrape WHERE received_time IS NOT NULL"
    ).fetchone()[0]

    print(f"\nDone: {updated:,} complaints updated with received_time")
    print(f"Coverage: {with_time:,} / {total:,} ({100*with_time/total:.1f}%)")

    # Hour distribution
    rows = conn.execute("""
        SELECT CAST(SUBSTR(received_time, 1, INSTR(received_time, ':') - 1) AS INTEGER) as hour,
               COUNT(*) as cnt
        FROM bis_scrape
        WHERE received_time IS NOT NULL
        GROUP BY hour ORDER BY hour
    """).fetchall()
    print("\nHour distribution:")
    for r in rows:
        bar = "#" * (r[1] // 200)
        print(f"  {r[0]:>2}:00  {r[1]:>6,}  {bar}")

    conn.close()


if __name__ == "__main__":
    main()
