"""Core BIS Web scraper: concurrent fetching, retry, rate limiting, archival."""

import gzip
import signal
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
import db
from parser import parse_bis_detail


class BISScraper:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or config.MAX_WORKERS
        self._stop = threading.Event()
        self._error_times = deque(maxlen=200)
        self._thread_local = threading.local()
        self._lock = threading.Lock()

        # Stats
        self._done_count = 0
        self._error_count = 0
        self._start_time = None

    # ── Thread-local session ────────────────────────────────────────────

    def _get_session(self) -> requests.Session:
        """Return a thread-local requests.Session with connection pooling."""
        if not hasattr(self._thread_local, "session"):
            s = requests.Session()
            s.headers.update({"User-Agent": config.USER_AGENT})
            adapter = HTTPAdapter(
                pool_connections=1,
                pool_maxsize=1,
                max_retries=Retry(total=0),  # we handle retries ourselves
            )
            s.mount("https://", adapter)
            self._thread_local.session = s
        return self._thread_local.session

    # ── Single complaint fetch ──────────────────────────────────────────

    def _scrape_one(self, complaint_number: str) -> dict:
        """Fetch + parse a single complaint. Runs in worker thread."""
        session = self._get_session()
        last_error = None

        for attempt in range(config.RETRY_ATTEMPTS):
            try:
                time.sleep(config.DELAY_BETWEEN_REQUESTS)

                resp = session.get(
                    config.BIS_WEB_URL,
                    params={"requestid": "0", "complaintno": complaint_number},
                    timeout=config.REQUEST_TIMEOUT,
                )
                resp.raise_for_status()

                # Archive raw HTML
                if config.SAVE_RAW_HTML:
                    self._save_html(complaint_number, resp.text)

                # Parse
                parsed = parse_bis_detail(resp.text)
                return parsed

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = str(e)
                if attempt < config.RETRY_ATTEMPTS - 1:
                    backoff = config.RETRY_BACKOFF_BASE ** (attempt + 1)
                    time.sleep(backoff)
                continue
            except requests.exceptions.HTTPError as e:
                # Don't retry 4xx errors
                if resp.status_code < 500:
                    return {"_parse_status": "error", "_parse_error": f"HTTP {resp.status_code}"}
                last_error = str(e)
                if attempt < config.RETRY_ATTEMPTS - 1:
                    backoff = config.RETRY_BACKOFF_BASE ** (attempt + 1)
                    time.sleep(backoff)
                continue

        return {"_parse_status": "error", "_parse_error": f"max retries: {last_error}"}

    def _save_html(self, complaint_number: str, html: str):
        """Save gzip'd HTML to archive."""
        path = config.HTML_ARCHIVE_DIR / f"{complaint_number}.html.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(html)

    # ── Adaptive rate limiting ──────────────────────────────────────────

    def _record_error(self):
        with self._lock:
            self._error_times.append(time.time())

    def _should_pause(self) -> bool:
        """Check if too many recent errors warrant a pause."""
        with self._lock:
            cutoff = time.time() - 60
            recent = sum(1 for t in self._error_times if t > cutoff)
        return recent > 5

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        """Scrape all pending complaints from the queue."""
        conn = db.get_connection()
        self._start_time = time.time()

        # Graceful shutdown on Ctrl+C
        original_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(sig, frame):
            print("\n\nCtrl+C received — finishing current batch...")
            self._stop.set()

        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            self._run_loop(conn)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            conn.close()

    def _run_loop(self, conn):
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while not self._stop.is_set():
                # Check for adaptive pause
                if self._should_pause():
                    print("  Too many errors in last 60s. Pausing 5 minutes...")
                    for _ in range(300):
                        if self._stop.is_set():
                            return
                        time.sleep(1)

                batch = db.get_pending_batch(conn, config.BATCH_COMMIT_SIZE)
                if not batch:
                    print("\nAll complaints scraped!")
                    break

                # Submit batch to thread pool
                futures = {
                    pool.submit(self._scrape_one, cnum): cnum
                    for cnum in batch
                }

                # Collect results
                results = []
                for future in as_completed(futures):
                    if self._stop.is_set():
                        break
                    cnum = futures[future]
                    try:
                        parsed = future.result()
                        results.append((cnum, parsed))
                    except Exception as e:
                        results.append((cnum, {
                            "_parse_status": "error",
                            "_parse_error": str(e),
                        }))

                # Commit batch to SQLite
                self._commit_batch(conn, results)
                self._log_progress(conn)

    def _commit_batch(self, conn, results: list[tuple[str, dict]]):
        """Write a batch of results to SQLite."""
        for cnum, parsed in results:
            status = parsed.get("_parse_status", "error")
            if status == "ok":
                db.mark_done(conn, cnum, parsed)
                self._done_count += 1
            elif status == "not_found":
                # Complaint doesn't exist on BIS Web — skip permanently
                db.mark_error(conn, cnum, parsed.get("_parse_error", "not found"))
                # Set attempts high so it won't be retried
                conn.execute(
                    "UPDATE scrape_log SET attempts = ? WHERE complaint_number = ?",
                    (config.RETRY_ATTEMPTS, cnum),
                )
                self._error_count += 1
                self._record_error()
            else:
                db.mark_error(conn, cnum, parsed.get("_parse_error", "unknown"))
                self._error_count += 1
                self._record_error()
        conn.commit()

    def _log_progress(self, conn):
        """Print progress stats."""
        elapsed = time.time() - self._start_time
        rate = self._done_count / elapsed if elapsed > 0 else 0
        progress = db.get_progress(conn)
        done = progress.get("done", 0)
        total = progress.get("total", 0)
        errors = progress.get("error", 0)
        pending = progress.get("pending", 0)
        pct = 100 * done / total if total > 0 else 0

        # ETA
        if rate > 0 and pending > 0:
            eta_sec = pending / rate
            eta_h = eta_sec / 3600
            eta_str = f"{eta_h:.1f}h"
        else:
            eta_str = "?"

        print(
            f"  [{pct:5.1f}%] done={done:,} errors={errors:,} "
            f"pending={pending:,} rate={rate:.1f}/s ETA={eta_str}"
        )


def run(max_workers: int | None = None):
    """Entry point for the scraper."""
    scraper = BISScraper(max_workers=max_workers)
    print(f"Starting BIS Web scraper ({scraper.max_workers} workers, "
          f"{config.DELAY_BETWEEN_REQUESTS}s delay)")
    print(f"HTML archive: {'ON' if config.SAVE_RAW_HTML else 'OFF'}")
    print()
    scraper.run()
