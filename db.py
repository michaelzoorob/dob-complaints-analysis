"""SQLite database layer: schema, progress tracking, queue, and export."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

# ── Schema ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS open_data (
    complaint_number TEXT PRIMARY KEY,
    status TEXT,
    date_entered TEXT,
    house_number TEXT,
    house_street TEXT,
    zip_code TEXT,
    bin TEXT,
    community_board TEXT,
    special_district TEXT,
    complaint_category TEXT,
    unit TEXT,
    disposition_date TEXT,
    disposition_code TEXT,
    inspection_date TEXT,
    dobrundate TEXT
);

CREATE TABLE IF NOT EXISTS bis_scrape (
    complaint_number TEXT PRIMARY KEY,
    bis_status TEXT,
    subject TEXT,
    category_code TEXT,
    category_description TEXT,
    assigned_to TEXT,
    priority TEXT,
    ref_311 TEXT,
    received_date TEXT,
    block TEXT,
    lot TEXT,
    owner TEXT,
    last_inspection TEXT,
    inspector_badge TEXT,
    disposition TEXT,
    ecb_violation TEXT,
    comments TEXT,
    bin TEXT,
    address TEXT,
    borough TEXT,
    scraped_at TEXT
);

CREATE TABLE IF NOT EXISTS scrape_log (
    complaint_number TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    last_attempt_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_scrape_log_status
    ON scrape_log(status);
CREATE INDEX IF NOT EXISTS idx_open_data_date_entered
    ON open_data(date_entered);
CREATE INDEX IF NOT EXISTS idx_open_data_bin
    ON open_data(bin);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a new SQLite connection with WAL mode for concurrency."""
    path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection | None = None):
    """Create all tables and indices."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    conn.executescript(_SCHEMA)
    conn.commit()
    if own_conn:
        conn.close()


# ── Open Data loading ───────────────────────────────────────────────────────

OPEN_DATA_COLUMNS = [
    "complaint_number", "status", "date_entered", "house_number",
    "house_street", "zip_code", "bin", "community_board",
    "special_district", "complaint_category", "unit",
    "disposition_date", "disposition_code", "inspection_date", "dobrundate",
]


def load_open_data(csv_path: Path, conn: sqlite3.Connection | None = None):
    """Bulk-load the Socrata CSV into the open_data table."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    chunk_size = 50_000
    total = 0
    for chunk in pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        # Normalize column names to match our schema
        chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
        # Keep only expected columns (CSV may have extras or different casing)
        cols_present = [c for c in OPEN_DATA_COLUMNS if c in chunk.columns]
        chunk = chunk[cols_present]
        # Deduplicate within chunk, keep last (most recent) entry
        chunk = chunk.drop_duplicates(subset=["complaint_number"], keep="last")
        # Use a temp table to handle cross-chunk duplicates
        chunk.to_sql("_open_data_tmp", conn, if_exists="replace", index=False)
        conn.execute(f"""
            INSERT OR IGNORE INTO open_data ({', '.join(cols_present)})
            SELECT {', '.join(cols_present)} FROM _open_data_tmp
        """)
        conn.execute("DROP TABLE IF EXISTS _open_data_tmp")
        total += len(chunk)
        print(f"  Loaded {total:,} rows...")

    conn.commit()
    if own_conn:
        conn.close()
    return total


# ── Scrape queue ────────────────────────────────────────────────────────────

def populate_scrape_queue(conn: sqlite3.Connection, min_year: int | None = None):
    """Insert pending entries into scrape_log for complaints matching scope.

    Only adds complaints not already in scrape_log.
    """
    min_year = min_year or config.SCRAPE_MIN_YEAR

    # date_entered is MM/DD/YYYY — extract year from last 4 chars
    conn.execute("""
        INSERT OR IGNORE INTO scrape_log (complaint_number, status)
        SELECT o.complaint_number, 'pending'
        FROM open_data o
        WHERE CAST(SUBSTR(o.date_entered, -4) AS INTEGER) >= ?
          AND o.complaint_number NOT IN (SELECT complaint_number FROM scrape_log)
    """, (min_year,))
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM scrape_log WHERE status = 'pending'"
    ).fetchone()[0]
    return count


def get_pending_batch(conn: sqlite3.Connection, limit: int | None = None):
    """Fetch the next batch of complaint numbers to scrape.

    Returns complaints that are 'pending' or 'error' with attempts < max.
    """
    limit = limit or config.BATCH_COMMIT_SIZE
    rows = conn.execute("""
        SELECT complaint_number FROM scrape_log
        WHERE status = 'pending'
           OR (status = 'error' AND attempts < ?)
        ORDER BY ROWID
        LIMIT ?
    """, (config.RETRY_ATTEMPTS, limit)).fetchall()
    return [r[0] for r in rows]


def mark_done(conn: sqlite3.Connection, complaint_number: str, parsed: dict):
    """Record a successfully scraped complaint."""
    now = datetime.now(timezone.utc).isoformat()
    parsed["scraped_at"] = now

    # Insert into bis_scrape
    bis_cols = [
        "complaint_number", "bis_status", "subject", "category_code",
        "category_description", "assigned_to", "priority", "ref_311",
        "received_date", "block", "lot", "owner", "last_inspection",
        "inspector_badge", "disposition", "ecb_violation", "comments",
        "bin", "address", "borough", "scraped_at",
    ]
    values = [parsed.get(c) if c != "complaint_number" else complaint_number
              for c in bis_cols]
    placeholders = ", ".join(["?"] * len(bis_cols))
    col_names = ", ".join(bis_cols)
    conn.execute(
        f"INSERT OR REPLACE INTO bis_scrape ({col_names}) VALUES ({placeholders})",
        values,
    )

    # Update scrape_log
    conn.execute("""
        UPDATE scrape_log
        SET status = 'done', completed_at = ?
        WHERE complaint_number = ?
    """, (now, complaint_number))


def mark_error(conn: sqlite3.Connection, complaint_number: str, error_msg: str):
    """Record a failed scrape attempt."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE scrape_log
        SET status = 'error',
            attempts = attempts + 1,
            last_error = ?,
            last_attempt_at = ?
        WHERE complaint_number = ?
    """, (error_msg, now, complaint_number))


def get_progress(conn: sqlite3.Connection) -> dict:
    """Return counts by scrape_log status."""
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM scrape_log GROUP BY status"
    ).fetchall()
    counts = {r[0]: r[1] for r in rows}
    counts["total"] = sum(counts.values())
    return counts


# ── Export ──────────────────────────────────────────────────────────────────

def export_merged(conn: sqlite3.Connection, output_path: Path | None = None):
    """JOIN open_data + bis_scrape and write to CSV."""
    output_path = output_path or config.EXPORTS_DIR / "merged_complaints.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT
            o.*,
            b.subject,
            b.category_description,
            b.assigned_to,
            b.priority,
            b.ref_311,
            b.block,
            b.lot,
            b.owner,
            b.last_inspection,
            b.inspector_badge,
            b.disposition AS bis_disposition,
            b.ecb_violation,
            b.comments,
            b.address,
            b.borough,
            b.scraped_at
        FROM open_data o
        LEFT JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        WHERE o.complaint_number IN (SELECT complaint_number FROM scrape_log WHERE status = 'done')
    """

    chunk_size = 50_000
    first = True
    total = 0
    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        chunk.to_csv(output_path, mode="w" if first else "a",
                     header=first, index=False)
        first = False
        total += len(chunk)
        print(f"  Exported {total:,} rows...")

    return total, output_path
