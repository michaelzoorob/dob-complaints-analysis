"""Download HPD Housing Maintenance Code complaints/problems (ygpa-z7cr) and violations
(wvxf-dwi5) for the study window from NYC Open Data, paged and resumable: each 50,000-row
page is written to data/hpd/pages/ as it arrives, and a rerun resumes from the last page.
Pages use keyset pagination on a unique id column (WHERE id > last ORDER BY id), which stays
fast at any depth; OFFSET paging slowed to over a minute a page past 3 million rows.
Final outputs: data/hpd/hpd_problems_2020_2026.csv.gz, data/hpd/hpd_violations_2020_2026.csv.gz
(raw downloads, git-ignored). Aggregation to lots happens in hpd_lot_outcomes.py.
"""
import gzip
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hpd"
PAGES = OUT / "pages"
PAGES.mkdir(parents=True, exist_ok=True)
WINDOW = ("2020-01-01T00:00:00", "2026-05-31T23:59:59")
LIMIT = 50000
KEYS = {"problems": "problem_id", "violations": "violationid"}
DATASETS = {
    "problems": ("ygpa-z7cr", "received_date",
                 "problem_id, complaint_id, bbl, block, lot, borough, received_date, type, major_category, "
                 "minor_category, unit_type, problem_status, status_description, problem_duplicate_flag, "
                 "complaint_anonymous_flag"),
    "violations": ("wvxf-dwi5", "novissueddate",
                   "violationid, bbl, boroid, block, lot, class, inspectiondate, novissueddate, novtype, "
                   "rentimpairing, violationstatus, currentstatus"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_page(ds_id, query):
    url = f"https://data.cityofnewyork.us/resource/{ds_id}.csv?" + urlencode({"$query": query})
    for attempt in range(6):
        try:
            with urlopen(Request(url, headers={"User-Agent": "dob-complaints-analysis/1.0"}), timeout=600) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001
            wait = 15 * (attempt + 1)
            log(f"  retry {attempt + 1} after error {type(e).__name__}: {str(e)[:80]} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError("gave up on " + url[:120])


def download(name):
    ds_id, datecol, cols = DATASETS[name]
    key = KEYS[name]
    where = f"{datecol} >= '{WINDOW[0]}' AND {datecol} <= '{WINDOW[1]}'"
    out = OUT / f"hpd_{name}_2020_2026.csv.gz"
    if out.exists():
        log(f"{out.name} exists; skipping"); return
    state = PAGES / f"{name}_state.txt"      # last key fetched, page number
    last, pageno = None, 0
    if state.exists():
        last, pageno = state.read_text().split("\t"); pageno = int(pageno)
        log(f"  resuming {name} after key {last} (page {pageno})")
    header = None
    while True:
        cond = where + (f" AND {key} > '{last}'" if last is not None else "")
        q = f"SELECT {cols} WHERE {cond} ORDER BY {key} LIMIT {LIMIT}"
        t0 = time.time(); text = fetch_page(ds_id, q)
        lines = text.splitlines(); n = len(lines) - 1
        if header is None:
            header = lines[0]
        if n <= 0:
            break
        pageno += 1
        (PAGES / f"{name}_{pageno:05d}.csv").write_text(text)
        # last key = value of the key column in the final row (csv: key column position from header)
        import csv, io
        hdr = next(csv.reader(io.StringIO(lines[0]))); ki = hdr.index(key)
        last = next(csv.reader(io.StringIO(lines[-1])))[ki]
        state.write_text(f"{last}\t{pageno}")
        log(f"  {name} page {pageno}: {n:,} rows in {time.time() - t0:.0f}s (last {key} {last})")
        if n < LIMIT:
            break
    with gzip.open(out, "wt") as f:
        f.write(header + "\n")
        for page in sorted(PAGES.glob(f"{name}_[0-9]*.csv")):
            body = page.read_text().splitlines()[1:]
            if body:
                f.write("\n".join(body) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["problems", "violations"]):
        download(name)
