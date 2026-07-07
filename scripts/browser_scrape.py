"""Browser-driven BIS scrape ingest + work-list.

The native scraper is hard-blocked by Akamai (403 on all programmatic TLS).
Only in-browser fetch() from the BIS origin passes. This module owns the
work-list (which complaints still need scraping) and ingests the parsed
records that the browser returns (saved to a JSON file), writing them into
bis_scrape / scrape_log via the existing db helpers.

The JS parser used in the browser is a direct port of parser.parse_bis_detail;
validate_parse() proves the port matches the Python parser field-for-field.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db          # noqa: E402
import parser      # noqa: E402

WINDOW_START = "20200101"
WINDOW_END = "20260531"   # analysis window cap (end of May 2026)


def _ymd(col="o.date_entered"):
    return f"substr({col},7,4)||substr({col},1,2)||substr({col},4,2)"


def build_worklist(conn, y0="20240101", y1=WINDOW_END, nonfinal_from="20260101",
                   limit=None):
    """Complaint numbers still needing a scrape, newest entered first.

    Two disjoint populations, both capped at the analysis window (y1):
      * no bis_scrape row at all  -> genuinely missing data (whole y0..y1 range)
      * non-final row (ACTIVE / no disposition) entered >= nonfinal_from
        -> was scraped before its inspection completed and has likely gained an
        outcome, inspector, and comments since. Older non-final complaints are
        mostly genuinely stalled and rarely change, so they are excluded.
    """
    q = f"""
        SELECT o.complaint_number, {_ymd()} AS ymd
        FROM open_data o
        LEFT JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        WHERE length(o.date_entered) = 10
          AND {_ymd()} BETWEEN ? AND ?
          AND (b.complaint_number IS NULL
               OR ((b.bis_status = 'ACTIVE'
                    OR COALESCE(b.disposition, '-') IN ('-', ''))
                   AND {_ymd()} >= ?))
        ORDER BY ymd DESC, o.complaint_number DESC
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q, (y0, y1, nonfinal_from)).fetchall()
    return [r[0] for r in rows]


def ingest_file(path, conn=None):
    """Ingest a browser batch result file (JSON array of parsed records).

    Each record has the parser.parse_bis_detail keys plus 'complaintno' and
    '_parse_status'. Returns a summary dict. 'blocked' records are left
    untouched (the browser requeues them after a cookie refresh).
    """
    own = conn is None
    if own:
        conn = db.get_connection()
    data = json.loads(Path(path).read_text())
    batch_deferred = batch_session_block = batch_deferred_total = None
    if isinstance(data, dict):
        recs = data.get("results", [])
        remaining = data.get("remaining")
        batch_deferred = data.get("deferred")
        batch_session_block = data.get("sessionBlock")
        batch_deferred_total = data.get("deferred_total")
    else:
        recs = data
        remaining = None
    n_ok = n_notfound = n_error = n_blocked = 0
    for rec in recs:
        cn = str(rec.get("complaintno") or rec.get("bis_complaint_number") or "").strip()
        status = rec.get("_parse_status")
        if not cn:
            continue
        if status == "blocked":
            n_blocked += 1
            continue
        if status == "ok":
            db.mark_done(conn, cn, rec)
            n_ok += 1
        elif status == "not_found":
            db.mark_done(conn, cn, {"bis_status": "NOT_FOUND"})
            n_notfound += 1
        else:
            db.mark_error(conn, cn, str(rec.get("_parse_error", "parse error")))
            n_error += 1
    conn.commit()
    if own:
        conn.close()
    return {"ok": n_ok, "not_found": n_notfound, "error": n_error,
            "blocked": n_blocked, "total": len(recs), "remaining": remaining,
            "deferred": batch_deferred, "session_block": batch_session_block,
            "deferred_total": batch_deferred_total}


def validate_parse(path):
    """Compare browser-JS parsed fields against parser.parse_bis_detail on the
    same raw HTML. Input file: JSON array of {complaintno, html, js}. Prints a
    per-field diff report and returns True if all fields match.
    """
    recs = json.loads(Path(path).read_text())
    fields = ["bis_status", "subject", "category_code", "category_description",
              "assigned_to", "priority", "ref_311", "received_date", "block",
              "lot", "owner", "last_inspection", "inspector_badge",
              "disposition", "ecb_violation", "comments", "bin", "address",
              "borough"]
    all_ok = True
    for rec in recs:
        cn = rec["complaintno"]
        py = parser.parse_bis_detail(rec["html"])
        js = rec["js"]
        diffs = []
        for f in fields:
            pv = (py.get(f) or "").strip() if isinstance(py.get(f), str) else py.get(f)
            jv = (js.get(f) or "").strip() if isinstance(js.get(f), str) else js.get(f)
            if (pv or None) != (jv or None):
                diffs.append(f"{f}: PY={pv!r} JS={jv!r}")
        status_match = py.get("_parse_status") == js.get("_parse_status")
        if diffs or not status_match:
            all_ok = False
            print(f"\n[{cn}] status py={py.get('_parse_status')} js={js.get('_parse_status')}")
            for d in diffs:
                print("   ", d)
        else:
            print(f"[{cn}] OK  ({len(rec['html'])} bytes, {py.get('_parse_status')})")
    print("\nALL FIELDS MATCH" if all_ok else "\nMISMATCHES FOUND")
    return all_ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "worklist"
    if cmd == "worklist":
        conn = db.get_connection()
        wl = build_worklist(conn)
        print(f"worklist size: {len(wl):,}")
        print("first 5 (newest):", wl[:5])
        print("last 5 (oldest):", wl[-5:])
        conn.close()
    elif cmd == "ingest":
        print(json.dumps(ingest_file(sys.argv[2]), indent=2))
    elif cmd == "validate":
        validate_parse(sys.argv[2])
