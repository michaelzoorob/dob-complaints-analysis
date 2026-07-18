"""Composition of complaint outcomes that end in a violation, by violation type.

When a DOB complaint inspection ends in a violation, the BIS disposition code
records which enforcement instrument was issued. This tabulates that mix over
the post's window so the appendix figures are reproducible.

Buckets follow the official BIS Complaint Disposition Codes (Rev. 09/21,
https://www.nyc.gov/assets/buildings/pdf/bis_complaint_disposition_codes.pdf):

  oath_ecb          A8       "OATH Violation Served" -- the penalty summons
                             adjudicated at OATH and published as DOB/ECB Violations
  buildings_dob     A1, A6   "Buildings Violation(s) Served" -- the corrective
                             notice published as DOB Violations
  both              A9       "OATH and DOB Violations Served"
  other_enforcement rest     stop-work orders, vacate orders, criminal summonses,
                             letters of deficiency, and smaller OATH categories

A8 rows carry an ECB violation number on the scraped BIS page 99.9% of the time
while A1/A6 rows almost never do, which is the direct evidence that A8 is the
OATH/ECB instrument and A1/A6 are the non-ECB Buildings instrument.

Output: data/analysis/risk_models/violation_type_composition.csv
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
import disposition_codes as dc  # noqa: E402

WINDOW = ("2020-01-01", "2026-05-31")
YMD = ("substr(date_entered,7,4) || '-' || substr(date_entered,1,2) "
       "|| '-' || substr(date_entered,4,2)")

# (category, reader-facing label, disposition codes) -- "other_enforcement" is
# the residual of VIOLATION_CODES not named here.
BUCKETS = [
    ("oath_ecb", "OATH / ECB violation (penalty summons)", ["A8"]),
    ("buildings_dob", "DOB Buildings violation (corrective notice)", ["A1", "A6"]),
    ("both", "Both an OATH and a DOB violation", ["A9"]),
]


def main() -> None:
    con = sqlite3.connect(str(config.DB_PATH))
    codes = ",".join(f"'{k}'" for k in dc.VIOLATION_CODES)
    counts = dict(con.execute(
        f"SELECT disposition_code, COUNT(*) FROM open_data "
        f"WHERE disposition_code IN ({codes}) "
        f"AND {YMD} BETWEEN ? AND ? GROUP BY disposition_code", WINDOW))
    con.close()

    total = sum(counts.values())
    rows = []
    for category, label, cds in BUCKETS:
        n = sum(counts.get(c, 0) for c in cds)
        rows.append(dict(category=category, label=label, codes="+".join(cds),
                         n=n, share=round(n / total, 4)))
    other = total - sum(r["n"] for r in rows)
    rows.append(dict(
        category="other_enforcement",
        label="Other enforcement (stop-work, vacate, criminal summons, deficiency)",
        codes="rest", n=other, share=round(other / total, 4)))
    rows.append(dict(category="total", label="All violation dispositions",
                     codes="all", n=total, share=1.0))

    df = pd.DataFrame(rows)
    out_path = (config.DATA_DIR / "analysis" / "risk_models"
                / "violation_type_composition.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
