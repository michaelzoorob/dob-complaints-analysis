# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project links NYC Department of Buildings (DOB) complaint metadata from NYC Open Data with the detailed complaint comments and outcomes available only on the BIS Web portal. The result is a novel joined dataset powering a web application for text analysis and map-based exploration of building complaints.

## Data Sources

### DOB Complaints Received (NYC Open Data)
- **Bulk CSV:** `https://data.cityofnewyork.us/api/views/eabe-havv/rows.csv?accessType=DOWNLOAD`
- **API:** `https://data.cityofnewyork.us/resource/eabe-havv.json` (Socrata, SoQL)
- **Size:** ~3M rows, 15 columns
- **Key fields:** `complaint_number`, `status`, `date_entered`, `house_number`, `house_street`, `zip_code`, `bin`, `community_board`, `complaint_category`, `disposition_code`, `disposition_date`, `inspection_date`

### BIS Web Complaint Detail Pages (scraped)
- **Direct lookup:** `GET /bisweb/OverviewForComplaintServlet?requestid=0&complaintno={complaint_number}`
- **Join key:** `complaint_number` from Open Data matches directly — no `vlcompdetlkey` needed
- **Fields only here:** inspector comments/narrative, subject (original complaint text), owner name, inspector badge, ECB violation numbers, priority, assigned unit, block/lot, 311 reference number, category description
- Requires browser-like User-Agent header; ~0.3s response time at moderate load

## Pipeline Architecture

SQLite database (`data/dob_complaints.db`) with three tables:
- `open_data` — full Socrata dataset
- `bis_scrape` — parsed BIS Web fields per complaint
- `scrape_log` — progress tracking (pending/done/error), enables resume after interruption

### Commands
```bash
python3 run_pipeline.py download              # Download Open Data CSV → SQLite + populate queue
python3 run_pipeline.py scrape                # Scrape BIS Web (resume-safe, Ctrl+C safe)
python3 run_pipeline.py scrape --workers 1    # Override concurrency (default: 3)
python3 run_pipeline.py export                # Export merged CSV → data/exports/merged_complaints.csv
python3 run_pipeline.py status                # Show progress + field fill rates
```

### Module Layout
- `config.py` — all constants (URLs, timeouts, concurrency, scope)
- `db.py` — SQLite schema, queue management, export
- `parser.py` — BIS Web HTML parsing (regex-based, tested against real pages)
- `download_open_data.py` — Socrata bulk CSV download + SQLite load
- `scraper.py` — core engine: ThreadPoolExecutor, retry with backoff, rate limiting, HTML archival
- `run_pipeline.py` — CLI orchestrator

### Key Design Details
- Scrape scope controlled by `SCRAPE_MIN_YEAR` in `config.py` (currently 2024)
- Raw HTML saved as gzip in `data/html_archive/` for re-parsing
- 3 concurrent workers, 0.5s delay per thread, exponential retry backoff
- Adaptive rate limiting: pauses 5min if 5+ errors in 60s window

## Workflow

- After making changes, always restart the dev server to pick up the latest code.
- Before marking a task as complete, verify outputs (e.g. run the app, check logs, inspect results) to confirm they behave as expected.

## Analysis Conventions (learned the hard way; do not regress)

- **Cluster standard errors at the level the treatment varies.** Inspector-level treatments (LOO strictness) cluster by inspector; unclustered errors overstated those t-stats roughly tenfold (t=333 became t=41). Property-level owner traits cluster by census tract, which is coarser and conservative.
- **Violation counts use only the deduplicated BIS + DOB NOW union** (`scripts/dob_ledger.py`). The BIS-only row in `citation_tidy_estimates.csv` is retained for continuity, never as a headline; the owner-occupancy gap sits entirely in the BIS-attributed stream (see `scripts/owner_robustness_checks.py`).
- **Commercial-exposure controls** on full-universe models: `com_class` (PLUTO class S/K/O) + `log_bldgarea` + `comm_bin` fixed effects. On the BISG individually-owned <16-unit subsample, drop `comm_bin` (degenerate) and keep `com_class` + `log_bldgarea`.
- **BISG race gaps bundle ownership tenure.** Asian-classified owners bought far more recently; deed-recency controls move the headline complaint gap from about +37% to +28% (`scripts/owner_robustness_checks.py`). Quote the caveat; the ownership-transition design is the recency-matched complement.
- **No `iterrows` or row-wise `.apply` on large tables.** Vectorize (per-group `searchsorted` over sorted event arrays turned a 30-minute loop into 80 seconds; see `scripts/spatial_spillovers.py`).
- **Blog prose:** every quoted number must trace to a CSV in `data/analysis/risk_models/`; never reference earlier or partial versions of an analysis; adversarial replication checks live in `scripts/audit/`.
