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

Also tabulates the other side of the appendix argument: the deduplicated
BIS + DOB NOW violation union (dob_ledger.py) over the same window, split into
periodic inspection categories (elevator, boiler, gas piping, energy, facade)
versus construction and other, plus the count of complaints that served a
Buildings violation (A1, A6, A9). The union is overwhelmingly periodic, which
is why the disposition and ECB measures move together while the DOB violations
dataset moves separately.

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
import dob_ledger  # noqa: E402

PERIODIC_FAMILIES = ["elev", "boiler", "gas_plumb", "energy", "facade"]

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

    # the union side of the appendix argument (dob_ledger.py; ymd is YYYYMMDD)
    u = dob_ledger.union_frame(con)
    w = u[(u.ymd >= WINDOW[0].replace("-", "")) & (u.ymd <= WINDOW[1].replace("-", ""))]
    union_n = len(w)
    periodic_n = int(w.family.isin(PERIODIC_FAMILIES).sum())

    # the ECB/OATH docket itself: how much of it is escalation-style (failing to
    # certify correction of an earlier violation, or to comply with an order)
    # versus direct condition citations served at the inspection
    ecb_window = ("substr(issue_date,1,4) BETWEEN '2020' AND '2026' "
                  "AND substr(issue_date,1,6) <= '202605'")
    docket_n = con.execute(
        f"SELECT COUNT(*) FROM ecb_violations WHERE {ecb_window}").fetchone()[0]
    escalation_n = con.execute(f"""
        SELECT COUNT(*) FROM ecb_violations WHERE {ecb_window}
        AND (UPPER(violation_description) LIKE '%FAIL%CERTIF%CORRECT%'
          OR UPPER(violation_description) LIKE '%FAIL%COMPLY%COMMISSIONER%'
          OR UPPER(violation_description) LIKE '%FAIL%COMPLY%ORDER%')""").fetchone()[0]
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

    # union rows: shares are relative to the union entry count, not the
    # disposition total above
    buildings_events = sum(counts.get(c, 0) for c in ("A1", "A6", "A9"))
    rows.append(dict(
        category="dob_union_entries",
        label="Deduplicated BIS + DOB NOW violation union entries",
        codes="union", n=union_n, share=1.0))
    rows.append(dict(
        category="dob_union_periodic",
        label="Union entries in periodic categories (elevator, boiler, gas piping, energy, facade)",
        codes="families", n=periodic_n, share=round(periodic_n / union_n, 4)))
    rows.append(dict(
        category="complaint_buildings_events",
        label="Complaints that served a Buildings violation",
        codes="A1+A6+A9", n=buildings_events,
        share=round(buildings_events / union_n, 4)))

    # docket rows: shares are relative to the docket entry count
    rows.append(dict(
        category="ecb_docket_entries",
        label="ECB/OATH summonses issued (penalty docket)",
        codes="docket", n=docket_n, share=1.0))
    rows.append(dict(
        category="ecb_docket_escalation",
        label="Docket charges for failure to certify correction or comply with an order",
        codes="pattern", n=escalation_n, share=round(escalation_n / docket_n, 4)))

    df = pd.DataFrame(rows)
    out_path = (config.DATA_DIR / "analysis" / "risk_models"
                / "violation_type_composition.csv")
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
