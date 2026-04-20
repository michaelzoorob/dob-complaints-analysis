"""CLI orchestrator for the DOB complaints scraping pipeline.

Usage:
    python3 run_pipeline.py download              # Download Open Data + populate queue
    python3 run_pipeline.py scrape                # Scrape BIS Web (resume-safe)
    python3 run_pipeline.py scrape --workers 1    # Override concurrency
    python3 run_pipeline.py export                # Export merged CSV
    python3 run_pipeline.py status                # Show progress
"""

import argparse
import sys

import config
import db


def cmd_download(args):
    import download_open_data
    download_open_data.run()


def cmd_scrape(args):
    import scraper
    # Ensure DB and queue exist
    conn = db.get_connection()
    db.init_db(conn)
    pending = db.get_progress(conn).get("pending", 0)
    if pending == 0:
        print("No complaints in scrape queue. Run 'download' first.")
        conn.close()
        sys.exit(1)
    conn.close()

    scraper.run(max_workers=args.workers)


def cmd_export(args):
    conn = db.get_connection()
    progress = db.get_progress(conn)
    done = progress.get("done", 0)
    if done == 0:
        print("No scraped data to export. Run 'scrape' first.")
        conn.close()
        sys.exit(1)

    print(f"Exporting {done:,} scraped complaints...")
    total, path = db.export_merged(conn)
    print(f"Wrote {total:,} rows to {path}")
    conn.close()


def cmd_status(args):
    conn = db.get_connection()
    db.init_db(conn)

    # Open data count
    od_count = conn.execute("SELECT COUNT(*) FROM open_data").fetchone()[0]
    print(f"Open Data rows: {od_count:,}")

    # Scrape progress
    progress = db.get_progress(conn)
    total = progress.get("total", 0)
    if total == 0:
        print("Scrape queue: empty (run 'download' to populate)")
    else:
        done = progress.get("done", 0)
        errors = progress.get("error", 0)
        pending = progress.get("pending", 0)
        pct = 100 * done / total if total else 0
        print(f"\nScrape queue: {total:,} total")
        print(f"  done:    {done:>10,} ({pct:.1f}%)")
        print(f"  pending: {pending:>10,}")
        print(f"  errors:  {errors:>10,}")

        # Sample recent errors
        if errors > 0:
            recent_errs = conn.execute("""
                SELECT complaint_number, last_error, attempts
                FROM scrape_log WHERE status = 'error'
                ORDER BY last_attempt_at DESC LIMIT 5
            """).fetchall()
            print(f"\n  Recent errors:")
            for r in recent_errs:
                print(f"    {r[0]}: {r[1]} (attempts: {r[2]})")

    # BIS scrape stats
    bis_count = conn.execute("SELECT COUNT(*) FROM bis_scrape").fetchone()[0]
    if bis_count > 0:
        comments_count = conn.execute(
            "SELECT COUNT(*) FROM bis_scrape WHERE comments IS NOT NULL AND comments != ''"
        ).fetchone()[0]
        subject_count = conn.execute(
            "SELECT COUNT(*) FROM bis_scrape WHERE subject IS NOT NULL AND subject != ''"
        ).fetchone()[0]
        print(f"\nBIS scrape field fill rates ({bis_count:,} records):")
        print(f"  comments: {comments_count:,} ({100*comments_count/bis_count:.1f}%)")
        print(f"  subject:  {subject_count:,} ({100*subject_count/bis_count:.1f}%)")

    # HTML archive
    archive = config.HTML_ARCHIVE_DIR
    if archive.exists():
        file_count = sum(1 for _ in archive.glob("*.html.gz"))
        print(f"\nHTML archive: {file_count:,} files")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="DOB Complaints Scraping Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download Open Data and populate scrape queue")

    p_scrape = sub.add_parser("scrape", help="Scrape BIS Web detail pages")
    p_scrape.add_argument("--workers", type=int, default=None,
                          help=f"Number of concurrent workers (default: {config.MAX_WORKERS})")

    sub.add_parser("export", help="Export merged dataset to CSV")
    sub.add_parser("status", help="Show pipeline progress")

    args = parser.parse_args()

    dispatch = {
        "download": cmd_download,
        "scrape": cmd_scrape,
        "export": cmd_export,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
