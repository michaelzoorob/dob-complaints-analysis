"""
Named case-study event log: the former Pfizer headquarters office-to-residential
conversion at 219 E 42nd St (BIN 1037551, BBL 1013160012) and 235 E 42nd St
(BIN 1037552, BBL 1013160023), Manhattan block 1316. Both buildings sit under
one Metro Loft-style conversion project ("219 GC LLC" / "235 GC LLC"); the
DOB NOW conversion jobs are M01075131-I1 (219: ALT-CO, 481 proposed units,
$96.4M) and M01075133-I1 (235: Alteration CO, 927 proposed units, $98.7M),
both filed 2024-07-03.

Emits one typed row per event to data/analysis/risk_models/pfizer_case_study.csv:
  deed               ACRIS deed transfers (block 1316, lots 12/23), 2020+
  permit_job_filed   permits_now job filings at the two BINs (DOB NOW era)
  work_permit_issued issued work permits from `permits` at the two BINs
  complaint          DOB complaints (open_data joined to bis_scrape), 2020+
  ecb_violation      ECB/OATH violations at the two BINs, 2020+ (plus two
                     rows synthesized from bis_scrape ECB references that
                     post-date the ecb_violations table's 2026-04-02 cutoff)
  safety_violation   DOB NOW safety violations at the two BINs, 2020+

dob_violations (BIS stream) is intentionally excluded: its only 2020+ rows at
this site are the EGRADE entries that duplicate the dob_safety_violations
EGRADE rows (cross-system dedup convention; see scripts/dob_ledger.py).

The scrape of BIS comments covers complaints received through 2026-05-31;
open_data/permits/safety tables extend to early July 2026. Everything here
pre-dates the 2026-07-07 structural incident.

Run:  /private/tmp/pyfix_venv/bin/python scripts/pfizer_case_study.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disposition_codes import (NO_ACCESS_CODES, NO_VIOLATION_CODES,
                               PENDING_CODES, REFERRAL_CODES, VIOLATION_CODES)

DISPOSITION_TEXT = {**VIOLATION_CODES, **NO_VIOLATION_CODES, **NO_ACCESS_CODES,
                    **REFERRAL_CODES, **PENDING_CODES}

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "dob_complaints.db"
OUT = ROOT / "data" / "analysis" / "risk_models" / "pfizer_case_study.csv"

BINS = ("1037551", "1037552")
ADDRESS = {"1037551": "219 E 42 ST", "1037552": "235 E 42 ST"}
BBL = {"1037551": "1013160012", "1037552": "1013160023"}
LOT_TO_BIN = {"12": "1037551", "23": "1037552"}

COLUMNS = [
    "event_date", "event_type", "bin", "address", "bbl", "record_id",
    "category_or_type", "origin", "ref_311", "severity",
    "status_or_disposition", "amount", "penalty_imposed", "penalty_paid",
    "balance_due", "respondent_or_owner", "description",
    "inspector_comments", "source_table",
]

# Inspector-comment quotes used in the blog post; the script fails loudly if
# any of them stops matching the database verbatim.
VERBATIM_QUOTES = {
    "1701635": "I2 - CSE ANSWERED INCIDENT 44880 DUE DOOR FALLING FROM CONTROL "
               "ACCESS ZONE. FULL STOP WORK ORDER ISSUED. PLEASE SEE COMPLAINT "
               "# 1701631.",
    "1711921": "ATOI of an 30-story NB approx. 29,328 Sq Ft. footprint, No "
               "ongoing construction activity. I observed watchpersons on "
               "site, upon request for proof of FDNY watchperson certificate "
               "and at least 10 hours OSHA card as required by code 3303.3 to "
               "verify compliance, none was provided. Creating a fire and "
               "safety hazard.",
    "1723097": "During my inspection of a 30 story new building with existing "
               "elements to remain. I observed multiple workers on multiple "
               "floors performing work. The superintendent of record was not "
               "present and the superintendent log book had not been signed by "
               "the super for more than 3 weeks. Provide and maintain an "
               "alternate superintendent. Comply with code 3301.13.5",
    "1724937": "Debris found along flooropening edges at Exp.1 on the 8th 9th "
               "floors and on the PScaffold platform. Materials pose windborne "
               "risk and create a tripping hazard for workers using the "
               "PScaffold. VI-1",
}
# 311 caller text (subject field), quoted in the post as the caller's words.
VERBATIM_SUBJECT_SNIPPET = (
    "1706633",
    "Today a large item fell and broke through 5 floors and almost hit someone",
)


def iso_mdy(s):
    """MM/DD/YYYY -> YYYY-MM-DD."""
    if not s or pd.isna(s):
        return None
    s = str(s).strip()
    if len(s) == 10 and s[2] == "/" and s[5] == "/":
        return f"{s[6:10]}-{s[0:2]}-{s[3:5]}"
    return None


def iso_compact(s):
    """YYYYMMDD -> YYYY-MM-DD."""
    if not s or pd.isna(s):
        return None
    s = str(s).strip()
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else None


def iso_ts(s):
    """ISO timestamp -> YYYY-MM-DD."""
    if not s or pd.isna(s):
        return None
    return str(s)[:10]


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def deeds(con):
    df = pd.read_sql(
        """SELECT m.document_id, m.doc_type, m.doc_date, m.doc_amount, l.lot,
                  (SELECT group_concat(p.name, ' | ') FROM acris_parties p
                   WHERE p.document_id = m.document_id AND p.party_type = '2')
                      AS grantee
           FROM acris_legals l
           JOIN acris_master m ON m.document_id = l.document_id
           WHERE l.borough = '1' AND l.block = '1316' AND l.lot IN ('12','23')
             AND m.doc_type = 'DEED' AND m.doc_date >= '2020-01-01'""",
        con,
    )
    rows = []
    for r in df.itertuples():
        b = LOT_TO_BIN[r.lot]
        rows.append({
            "event_date": iso_ts(r.doc_date), "event_type": "deed",
            "bin": b, "address": ADDRESS[b], "bbl": BBL[b],
            "record_id": r.document_id, "category_or_type": "DEED",
            "origin": "acris", "amount": num(r.doc_amount),
            "respondent_or_owner": r.grantee,
            "description": f"Property sale recorded; grantee {r.grantee}",
            "source_table": "acris_master/acris_legals",
        })
    return rows


def permit_jobs(con):
    df = pd.read_sql(
        f"""SELECT job_filing_number, filing_status, job_type, initial_cost,
                   existing_dwelling_units, proposed_dwelling_units,
                   filing_date, approved_date, first_permit_date, signoff_date,
                   current_status_date, owner_s_business_name, bin
            FROM permits_now WHERE bin IN {BINS}""",
        con,
    )
    rows = []
    for r in df.itertuples():
        date = (iso_ts(r.filing_date) or iso_ts(r.approved_date)
                or iso_ts(r.first_permit_date) or iso_ts(r.current_status_date))
        units = ""
        if r.existing_dwelling_units is not None or r.proposed_dwelling_units is not None:
            units = (f"; dwelling units {r.existing_dwelling_units or 0}"
                     f" -> {r.proposed_dwelling_units or 0}")
        desc = (f"{r.job_type} filing; initial cost ${num(r.initial_cost) or 0:,.0f}"
                f"{units}; approved {iso_ts(r.approved_date) or '-'}"
                f"; first permit {iso_ts(r.first_permit_date) or '-'}"
                f"; signoff {iso_ts(r.signoff_date) or '-'}")
        rows.append({
            "event_date": date, "event_type": "permit_job_filed",
            "bin": r.bin, "address": ADDRESS[r.bin], "bbl": BBL[r.bin],
            "record_id": r.job_filing_number, "category_or_type": r.job_type,
            "origin": "applicant", "status_or_disposition": r.filing_status,
            "amount": num(r.initial_cost),
            "respondent_or_owner": r.owner_s_business_name,
            "description": desc, "source_table": "permits_now",
        })
    return rows


def work_permits(con):
    df = pd.read_sql(
        f"""SELECT DISTINCT work_permit, filing_reason, work_type,
                   permit_status, issued_date, estimated_job_costs,
                   owner_business_name, job_description, bin
            FROM permits WHERE bin IN {BINS} AND issued_date IS NOT NULL""",
        con,
    )
    rows = []
    for r in df.itertuples():
        rows.append({
            "event_date": iso_ts(r.issued_date), "event_type": "work_permit_issued",
            "bin": r.bin, "address": ADDRESS[r.bin], "bbl": BBL[r.bin],
            "record_id": r.work_permit,
            "category_or_type": f"{r.work_type} ({r.filing_reason})",
            "origin": "applicant", "status_or_disposition": r.permit_status,
            "amount": num(r.estimated_job_costs),
            "respondent_or_owner": r.owner_business_name,
            "description": r.job_description, "source_table": "permits",
        })
    return rows


def complaints(con):
    df = pd.read_sql(
        f"""SELECT o.complaint_number, o.date_entered, o.complaint_category,
                   o.status AS od_status, o.disposition_code,
                   o.disposition_date, o.inspection_date, o.bin,
                   b.category_code, b.category_description, b.ref_311,
                   b.priority, b.disposition, b.subject, b.comments,
                   b.ecb_violation, b.bis_status
            FROM open_data o
            LEFT JOIN bis_scrape b ON b.complaint_number = o.complaint_number
            WHERE o.bin IN {BINS}""",
        con,
    )
    df["event_date"] = df["date_entered"].map(iso_mdy)
    df = df[df["event_date"] >= "2020-01-01"].copy()
    rows = []
    for r in df.itertuples():
        scraped = isinstance(r.category_code, str) and r.category_code
        if isinstance(r.ref_311, str) and r.ref_311.strip():
            origin = "311"
        elif scraped:
            origin = "agency"
        else:
            origin = "unknown (BIS not yet scraped)"
        disp = r.disposition if isinstance(r.disposition, str) else None
        if not disp:
            code_txt = None
            if isinstance(r.disposition_code, str):
                code_txt = " - ".join(
                    x for x in (r.disposition_code,
                                DISPOSITION_TEXT.get(r.disposition_code)) if x)
            disp = " - ".join(
                x for x in (iso_mdy(r.disposition_date), code_txt)
                if x) or r.od_status
        cat = r.category_code if scraped else r.complaint_category
        rows.append({
            "event_date": r.event_date, "event_type": "complaint",
            "bin": r.bin, "address": ADDRESS[r.bin], "bbl": BBL[r.bin],
            "record_id": r.complaint_number, "category_or_type": cat,
            "origin": origin, "ref_311": r.ref_311, "severity": r.priority,
            "status_or_disposition": disp,
            "respondent_or_owner": None,
            "description": r.subject,
            "inspector_comments": r.comments,
            "source_table": "open_data+bis_scrape",
        })
        # carry ECB references for the completeness check below
        rows[-1]["_ecb_refs"] = r.ecb_violation if isinstance(r.ecb_violation, str) else ""
    return rows


def ecb(con, complaint_rows):
    df = pd.read_sql(
        f"""SELECT ecb_violation_number, issue_date, severity, violation_type,
                   violation_description, penality_imposed, amount_paid,
                   balance_due, ecb_violation_status, hearing_status,
                   respondent_name, bin
            FROM ecb_violations
            WHERE bin IN {BINS} AND issue_date >= '20200101'""",
        con,
    )
    rows = []
    for r in df.itertuples():
        rows.append({
            "event_date": iso_compact(r.issue_date), "event_type": "ecb_violation",
            "bin": r.bin, "address": ADDRESS[r.bin], "bbl": BBL[r.bin],
            "record_id": r.ecb_violation_number,
            "category_or_type": r.violation_type, "origin": "dob_enforcement",
            "severity": r.severity,
            "status_or_disposition": " / ".join(
                x for x in (r.ecb_violation_status, r.hearing_status)
                if isinstance(x, str) and x),
            "penalty_imposed": num(r.penality_imposed),
            "penalty_paid": num(r.amount_paid),
            "balance_due": num(r.balance_due),
            "respondent_or_owner": r.respondent_name,
            "description": r.violation_description,
            "source_table": "ecb_violations",
        })
    have = {r["record_id"] for r in rows}
    # ECB numbers cited on scraped complaints but past the table's cutoff
    for c in complaint_rows:
        for ref in c.pop("_ecb_refs", "").split():
            if ref and ref not in have:
                rows.append({
                    "event_date": c["event_date"], "event_type": "ecb_violation",
                    "bin": c["bin"], "address": c["address"], "bbl": c["bbl"],
                    "record_id": ref, "category_or_type": "Construction",
                    "origin": "dob_enforcement",
                    "status_or_disposition": "SERVED (per BIS complaint "
                        f"{c['record_id']}; past ecb_violations cutoff)",
                    "description": c["description"],
                    "source_table": "bis_scrape (ecb_violation reference)",
                })
                have.add(ref)
    for c in complaint_rows:  # strip helper key from any remaining rows
        c.pop("_ecb_refs", None)
    return rows


def safety(con):
    df = pd.read_sql(
        f"""SELECT violation_number, violation_type, violation_status,
                   device_type, violation_issue_date, bin
            FROM dob_safety_violations
            WHERE bin IN {BINS} AND violation_issue_date >= '2020-01-01'""",
        con,
    )
    rows = []
    for r in df.itertuples():
        rows.append({
            "event_date": iso_ts(r.violation_issue_date),
            "event_type": "safety_violation",
            "bin": r.bin, "address": ADDRESS[r.bin], "bbl": BBL[r.bin],
            "record_id": r.violation_number,
            "category_or_type": f"{r.device_type} ({r.violation_type})",
            "origin": "dob_program",
            "status_or_disposition": r.violation_status,
            "source_table": "dob_safety_violations",
        })
    return rows


def verify(con, df):
    cur = con.cursor()
    print("\n== verification against external reporting ==")
    e25 = df[(df.event_type == "ecb_violation") & (df.bin == "1037552")
             & (df.event_date >= "2025-01-01") & (df.event_date <= "2025-12-31")]
    n1 = (e25.severity == "CLASS - 1").sum()
    print(f"235 E 42 St ECB violations in 2025: n={len(e25)} "
          f"(news: 7), CLASS-1={n1} of {len(e25)} (news: all but one), "
          f"imposed=${e25.penalty_imposed.sum():,.0f} (news: ~$32.5k), "
          f"paid=${e25.penalty_paid.sum():,.0f}, "
          f"balance=${e25.balance_due.sum():,.0f}")
    both25 = df[(df.event_type == "ecb_violation")
                & (df.event_date.between("2025-01-01", "2025-12-31"))]
    print(f"Whole site (both BINs) ECB violations in 2025: n={len(both25)}, "
          f"imposed=${both25.penalty_imposed.sum():,.0f}, "
          f"paid=${both25.penalty_paid.sum():,.0f}")
    panel = df[df.record_id == "39543343X"]
    assert len(panel) == 1 and panel.iloc[0].penalty_imposed == 10000, \
        "Aug 2025 falling-panel $10k ECB violation missing"
    print("Aug 2025 falling panel: ECB 39543343X CLASS-1 $10,000 on "
          f"{panel.iloc[0].event_date} — present")

    for cno, quote in VERBATIM_QUOTES.items():
        db = cur.execute("SELECT comments FROM bis_scrape WHERE complaint_number=?",
                         (cno,)).fetchone()[0]
        assert db == quote, f"comments for {cno} no longer verbatim"
    cno, snippet = VERBATIM_SUBJECT_SNIPPET
    db = cur.execute("SELECT subject FROM bis_scrape WHERE complaint_number=?",
                     (cno,)).fetchone()[0]
    assert snippet in db, f"subject snippet for {cno} not found"
    print(f"Verbatim checks: {len(VERBATIM_QUOTES)} inspector comments + "
          f"1 caller subject match the database exactly")


def main():
    con = sqlite3.connect(DB)
    comp = complaints(con)
    rows = deeds(con) + permit_jobs(con) + work_permits(con) + ecb(con, comp) \
        + safety(con) + comp
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df = df.sort_values(["event_date", "event_type", "record_id"],
                        kind="stable").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} events)")
    print("\n== events by type ==")
    print(df.event_type.value_counts().to_string())
    print("\n== events by year x type ==")
    print(df.assign(year=df.event_date.str[:4])
            .pivot_table(index="year", columns="event_type", values="record_id",
                         aggfunc="count", fill_value=0).to_string())
    verify(con, df)
    con.close()


if __name__ == "__main__":
    main()
