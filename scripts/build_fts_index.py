"""Build FTS5 full-text search index and add performance indexes."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_indexes():
    conn = sqlite3.connect(str(config.DB_PATH))

    # FTS5 index on subject + comments
    print("Creating FTS5 virtual table...")
    conn.execute("DROP TABLE IF EXISTS complaints_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE complaints_fts USING fts5(
            complaint_number,
            subject,
            comments,
            tokenize='porter unicode61'
        )
    """)

    print("Populating FTS5 index...")
    conn.execute("""
        INSERT INTO complaints_fts (complaint_number, subject, comments)
        SELECT complaint_number, COALESCE(subject, ''), COALESCE(comments, '')
        FROM bis_scrape
    """)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM complaints_fts").fetchone()[0]
    print(f"  Indexed {count:,} complaints")

    # Test search
    print("\nTest search for 'scaffolding collapsed':")
    rows = conn.execute("""
        SELECT complaint_number, snippet(complaints_fts, 2, '>>>', '<<<', '...', 30)
        FROM complaints_fts
        WHERE complaints_fts MATCH 'scaffolding collapsed'
        ORDER BY rank
        LIMIT 3
    """).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]}")

    # Performance indexes
    print("\nCreating performance indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_bis_borough ON bis_scrape(borough)",
        "CREATE INDEX IF NOT EXISTS idx_bis_priority ON bis_scrape(priority)",
        "CREATE INDEX IF NOT EXISTS idx_bis_category ON bis_scrape(category_description)",
        "CREATE INDEX IF NOT EXISTS idx_bis_bin ON bis_scrape(bin)",
        "CREATE INDEX IF NOT EXISTS idx_od_status ON open_data(status)",
    ]
    for idx in indexes:
        conn.execute(idx)
        print(f"  {idx.split('idx_')[1].split(' ')[0]}")
    conn.commit()

    print("\nDone!")
    conn.close()


if __name__ == "__main__":
    build_indexes()
