"""Central configuration for the DOB complaints scraping pipeline."""

from pathlib import Path

# ── Directories ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "dob_complaints.db"
HTML_ARCHIVE_DIR = DATA_DIR / "html_archive"
EXPORTS_DIR = DATA_DIR / "exports"

# ── Open Data (Socrata) ────────────────────────────────────────────────────

OPEN_DATA_CSV_URL = (
    "https://data.cityofnewyork.us/api/views/eabe-havv/rows.csv"
    "?accessType=DOWNLOAD"
)
OPEN_DATA_API_URL = "https://data.cityofnewyork.us/resource/eabe-havv.json"

# ── BIS Web ─────────────────────────────────────────────────────────────────

BIS_WEB_URL = (
    "https://a810-bisweb.nyc.gov/bisweb/OverviewForComplaintServlet"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# ── Scraper settings ───────────────────────────────────────────────────────

MAX_WORKERS = 3
DELAY_BETWEEN_REQUESTS = 1.5  # seconds, per thread (1.0 triggered 403s after days of scraping)
REQUEST_TIMEOUT = 30  # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 4  # exponential: 4^1=4s, 4^2=16s, 4^3=64s
BATCH_COMMIT_SIZE = 100
SAVE_RAW_HTML = True

# ── Scope ───────────────────────────────────────────────────────────────────
# Phase 1: 2024-2025. Change min_year to expand.

SCRAPE_MIN_YEAR = 2021
