"""Fast native BIS scrape — the path that scraped the original 770K, revived.

Native HTTP works once the request carries a COMPLETE, fresh Akamai cookie set
(the HttpOnly ak_bmsc included). We harvest that from the browser into
bis_cookies.json, then blast the worklist in parallel here. When the cookie
decays (a run of blocked responses), the run stops cleanly with NEEDS_REHARVEST
so a fresh cookie can be grabbed and the run resumed — nothing is lost.

Usage:  python3 scripts/native_scrape.py [workers] [delay] [limit]
        workers default 4, delay default 1.0s/worker, limit 0 = unlimited.
Resume-safe: remaining worklist is kept in native_worklist.json; blocked and
un-reached complaints stay in it for the next run.
"""
import sys, time, json, threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import db          # noqa: E402
import parser      # noqa: E402
from browser_scrape import build_worklist  # noqa: E402

TMP = Path("/Users/mzoorob/.claude/jobs/995b2587/tmp")
COOKIE_FILE = TMP / "bis_cookies.json"
WL_FILE = TMP / "native_worklist.json"

BASE = "https://a810-bisweb.nyc.gov/bisweb/OverviewForComplaintServlet"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "?requestid=0&complaintno=1718179",
    "Upgrade-Insecure-Requests": "1",
}

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0

COOKIES = json.loads(COOKIE_FILE.read_text())

_stop = threading.Event()
_tl = threading.local()
_lock = threading.Lock()
_win = deque(maxlen=25)   # recent responses: 1=block, 0=ok


def _sess():
    if not hasattr(_tl, "s"):
        s = requests.Session()
        s.headers.update(HEADERS)
        s.cookies.update(COOKIES)
        _tl.s = s
    return _tl.s


def _note(is_block):
    with _lock:
        _win.append(1 if is_block else 0)
        if len(_win) >= 15 and sum(_win) > len(_win) * 0.5:
            _stop.set()   # sustained blocking => cookie decayed


def fetch_one(cn):
    if _stop.is_set():
        return (cn, None)
    time.sleep(DELAY)
    try:
        r = _sess().get(BASE, params={"requestid": "0", "complaintno": cn}, timeout=30)
        blocked = (r.status_code == 403) or (
            "Overview for Complaint" not in r.text and len(r.text) < 3000)
        if blocked:
            _note(True)
            return (cn, {"_parse_status": "blocked"})
        _note(False)
        return (cn, parser.parse_bis_detail(r.text))
    except Exception as e:
        return (cn, {"_parse_status": "error", "_parse_error": str(e)[:60]})


def main():
    conn = db.get_connection()
    if WL_FILE.exists():
        wl = json.loads(WL_FILE.read_text())
        print(f"resumed worklist from file: {len(wl):,}", flush=True)
    else:
        print("building worklist from DB (one-time ~scan)...", flush=True)
        wl = build_worklist(conn)
        WL_FILE.write_text(json.dumps(wl))
        print(f"worklist built: {len(wl):,}", flush=True)
    if LIMIT:
        wl = wl[:LIMIT]
    print(f"start: {len(wl):,} to do | workers={WORKERS} delay={DELAY}s "
          f"(~{WORKERS/(DELAY+0.3):.1f}/s)", flush=True)

    completed = set()   # ok / not_found / error (removed from worklist)
    n_ok = n_nf = n_blocked = n_err = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_one, cn): cn for cn in wl}
        for fut in as_completed(futs):
            cn, parsed = fut.result()
            if parsed is None:
                continue
            st = parsed.get("_parse_status")
            if st == "ok":
                db.mark_done(conn, cn, parsed); completed.add(cn); n_ok += 1
            elif st == "not_found":
                db.mark_done(conn, cn, {"bis_status": "NOT_FOUND"}); completed.add(cn); n_nf += 1
            elif st == "blocked":
                n_blocked += 1   # leave pending -> stays in worklist for retry
            else:
                db.mark_error(conn, cn, parsed.get("_parse_error", "err")); completed.add(cn); n_err += 1
            tot = n_ok + n_nf + n_err
            if tot and tot % 100 == 0:
                conn.commit()
                rate = (n_ok + n_nf) / (time.time() - t0 + 1e-9)
                print(f"  ok={n_ok:,} nf={n_nf} err={n_err} blocked={n_blocked} "
                      f"rate={rate:.1f}/s remaining~{len(wl)-len(completed):,}", flush=True)

    conn.commit()
    remaining = [cn for cn in wl if cn not in completed]
    # merge back any part of the full worklist we didn't load this run (LIMIT case)
    if LIMIT and WL_FILE.exists():
        full = json.loads(WL_FILE.read_text())
        tail = [cn for cn in full if cn not in set(wl)]
        remaining = remaining + tail
    WL_FILE.write_text(json.dumps(remaining))
    conn.close()
    dt = time.time() - t0
    print(f"\nEND ok={n_ok:,} nf={n_nf} err={n_err} blocked={n_blocked} "
          f"elapsed={dt:.0f}s rate={(n_ok+n_nf)/(dt+1e-9):.1f}/s remaining={len(remaining):,}",
          flush=True)
    print("NEEDS_REHARVEST" if _stop.is_set() else "QUEUE_DONE", flush=True)


if __name__ == "__main__":
    main()
