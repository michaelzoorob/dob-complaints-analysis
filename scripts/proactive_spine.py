#!/usr/bin/env python3
"""
Shared data spine for the proactive-enforcement analyses (Wave 2 prep).

Builds three cached frames plus a verification summary from the SQLite
database. Cached frames land in data/analysis/proactive/ (gitignored);
the summary lands in data/analysis/risk_models/.

  proactive_events.parquet|csv.gz
      every scraped complaint received 2020-01..2026-05: category prefix,
      family (analysis_config.PROACTIVE_FAMILIES), agency flag (empty
      ref_311), BIN, zero-padded BBL, bct2020 tract + NTA, priority,
      assigned unit, disposition outcome (disposition_codes.
      classify_disposition), linked ECB violation (first token of
      bis_scrape.ecb_violation -> ecb_violations severity + penalty), and
      the permit active at the received date (largest estimated cost when
      several are active at the BIN).
  jobs.parquet|csv.gz
      DOB NOW jobs first permitted 2020+, joined to permit spans, with
      dwelling-unit deltas, conversion flags (strict/relaxed/office),
      tract, and active-month span columns.
  tract_month.csv.gz
      bct2020 x month panel 2020-01..2026-05: caller complaint counts,
      proactive counts by family, active-job counts, residential units —
      ready for the complaint-indexing PPML.
  risk_models/proactive_spine_summary.csv
      row counts and verification shares checked against the Wave-1
      inventory (proactive_enforcement_plan.md).

Run:  /private/tmp/pyfix_venv/bin/python scripts/proactive_spine.py
"""

import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_config import (BOROUGH_NAME_TO_CODE, CATEGORY_DESC_OVERRIDES,
                             PROACTIVE_FAMILIES)
from disposition_codes import classify_disposition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "dob_complaints.db")
OUT_DIR = os.path.join(ROOT, "data", "analysis", "proactive")
RISK_DIR = os.path.join(ROOT, "data", "analysis", "risk_models")

WINDOW_START = pd.Timestamp("2020-01-01")
WINDOW_END = pd.Timestamp("2026-05-31")
N_MONTHS = (2026 - 2020) * 12 + 5  # 2020-01 .. 2026-05 = 77 panel months

FAMILY_ORDER = ["statutory_periodic", "discretionary_field", "followup",
                "mixed_incident", "other"]

CONVERSION_JOB_TYPES = (
    "Alteration CO",
    "ALT-CO - New Building with Existing Elements to Remain",
)

# Wave-1 inventory expectations (proactive_enforcement_plan.md, output B)
EXPECT = {
    "n_events_agency": 249_707,
    "active_share_agency": 0.543,
    "active_share_caller": 0.262,
    "active_share_prefix_8A": 0.836,
    "active_share_prefix_7G": 0.696,
    "ecb_match_share": 0.95,
    "n_conversion_ge10": 630,
    "signoff_fill_share": 0.64,
}


def month_index(ts: pd.Series) -> pd.Series:
    """Vectorized datetime -> panel month ordinal (2020-01 = 0)."""
    return (ts.dt.year - 2020) * 12 + (ts.dt.month - 1)


def blocklot_key(boro: pd.Series, block: pd.Series, lot: pd.Series) -> pd.Series:
    """Normalized boro_block_lot join key ('' when any part is invalid).

    Block/lot arrive zero-padded in some tables and bare in others, so
    normalize through integers on both sides of every join.
    """
    b = pd.to_numeric(block, errors="coerce")
    lo = pd.to_numeric(lot, errors="coerce")
    ok = boro.notna() & (boro != "") & b.notna() & lo.notna()
    out = pd.Series("", index=boro.index, dtype="str")
    out[ok] = (boro[ok].astype("str") + "_" + b[ok].astype("int64").astype("str")
               + "_" + lo[ok].astype("int64").astype("str"))
    return out


def bbl_key(boro: pd.Series, block: pd.Series, lot: pd.Series) -> pd.Series:
    """Zero-padded 10-digit BBL ('' when any part is invalid)."""
    b = pd.to_numeric(block, errors="coerce")
    lo = pd.to_numeric(lot, errors="coerce")
    ok = boro.notna() & (boro != "") & b.notna() & lo.notna()
    out = pd.Series("", index=boro.index, dtype="str")
    out[ok] = (boro[ok].astype("str")
               + b[ok].astype("int64").astype("str").str.zfill(5)
               + lo[ok].astype("int64").astype("str").str.zfill(4))
    return out


def save_frame(df: pd.DataFrame, stem: str) -> str:
    """Write parquet when an engine is available, else csv.gz."""
    path = os.path.join(OUT_DIR, stem + ".parquet")
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        path = os.path.join(OUT_DIR, stem + ".csv.gz")
        df.to_csv(path, index=False, compression="gzip")
        return path


# ── PLUTO lookups ────────────────────────────────────────────────────────

def load_pluto_lookups(conn):
    """Tract lookup (blocklot key -> bct2020), building class, tract units."""
    pt = pd.read_sql_query(
        "SELECT borocode, block, lot, bct2020 FROM pluto_tract", conn)
    pt["key"] = blocklot_key(pt["borocode"], pt["block"], pt["lot"])
    pt = pt[(pt["key"] != "") & pt["bct2020"].notna()]
    pt = pt.drop_duplicates("key")[["key", "bct2020"]]

    pl = pd.read_sql_query(
        "SELECT borocode, block, lot, bldgclass, unitsres FROM pluto", conn)
    pl["key"] = blocklot_key(pl["borocode"], pl["block"], pl["lot"])
    pl["unitsres"] = pd.to_numeric(pl["unitsres"], errors="coerce").fillna(0)
    pl = pl[pl["key"] != ""].drop_duplicates("key")

    units = (pl.merge(pt, on="key", how="inner")
             .groupby("bct2020", as_index=False)["unitsres"].sum()
             .rename(columns={"unitsres": "units_res"}))
    print(f"PLUTO lookups: {len(pt):,} tract keys, {len(pl):,} class keys, "
          f"{len(units):,} tracts with units")
    return pt, pl[["key", "bldgclass"]], units


# ── 1. Complaint events ──────────────────────────────────────────────────

def build_events(conn, pt: pd.DataFrame) -> pd.DataFrame:
    ev = pd.read_sql_query("""
        SELECT b.complaint_number, b.received_date, b.category_code,
               b.priority, b.assigned_to, b.ref_311, b.bin, b.borough,
               b.block, b.lot, b.ecb_violation, o.disposition_code
        FROM bis_scrape b
        LEFT JOIN open_data o ON b.complaint_number = o.complaint_number
        WHERE b.received_date LIKE '__/__/____'
    """, conn)
    ev["received"] = pd.to_datetime(ev["received_date"], format="%m/%d/%Y",
                                    errors="coerce")
    n0 = len(ev)
    ev = ev[ev["received"].between(WINDOW_START, WINDOW_END)].reset_index(drop=True)
    print(f"Events: {n0:,} scraped -> {len(ev):,} received "
          f"{WINDOW_START:%Y-%m}..{WINDOW_END:%Y-%m}")

    # category prefix, family, canonical name (category_code embeds the
    # DOB category name after the 2-char code; hand overrides for the
    # 73/7R "Date" parser artifact)
    code = ev["category_code"].fillna("").astype("str")
    ev["category_prefix"] = code.str[:2]
    ev["family"] = ev["category_prefix"].map(PROACTIVE_FAMILIES).fillna("other")
    ev["category_name"] = code.str[2:].str.strip()
    for pfx, desc in CATEGORY_DESC_OVERRIDES.items():
        ev.loc[ev["category_prefix"] == pfx, "category_name"] = desc

    # agency flag: no 311 reference
    ev["agency"] = (ev["ref_311"].fillna("").astype("str").str.strip() == "")
    ev["agency"] = ev["agency"].astype("int8")

    # geography
    boro = ev["borough"].map(BOROUGH_NAME_TO_CODE)
    ev["bbl"] = bbl_key(boro, ev["block"], ev["lot"])
    ev["key"] = blocklot_key(boro, ev["block"], ev["lot"])
    ev = ev.merge(pt, on="key", how="left")
    nta = pd.read_sql_query(
        "SELECT complaint_number, nta FROM complaint_nta", conn)
    nta = nta.drop_duplicates("complaint_number")
    ev = ev.merge(nta, on="complaint_number", how="left")

    # disposition outcome (classify unique codes once, then map)
    codes = ev["disposition_code"].fillna("").astype("str")
    cmap = {c: classify_disposition(c) for c in codes.unique()}
    ev["outcome"] = codes.map(cmap)

    ev["month"] = ev["received"].dt.strftime("%Y-%m")
    ev["received_date"] = ev["received"].dt.strftime("%Y-%m-%d")
    print(f"  agency share {ev['agency'].mean():.3f}; "
          f"tract match {(ev['bct2020'].notna()).mean():.3f}; "
          f"nta match {(ev['nta'].notna()).mean():.3f}")
    return ev


def add_ecb(ev: pd.DataFrame, conn) -> pd.DataFrame:
    """First token of bis_scrape.ecb_violation -> ecb_violations link."""
    ref = ev["ecb_violation"].fillna("").astype("str").str.strip()
    ev["ecb_number"] = ref.str.split().str[0].fillna("")

    ecb = pd.read_sql_query("""
        SELECT ecb_violation_number AS ecb_number, severity AS ecb_severity,
               penality_imposed
        FROM ecb_violations WHERE ecb_violation_number IS NOT NULL
    """, conn).drop_duplicates("ecb_number")
    ecb["ecb_penalty"] = pd.to_numeric(ecb["penality_imposed"], errors="coerce")
    ev = ev.merge(ecb[["ecb_number", "ecb_severity", "ecb_penalty"]],
                  on="ecb_number", how="left")
    has_ref = ev["ecb_number"] != ""
    matched = ev["ecb_severity"].notna() & has_ref
    print(f"  ECB refs {has_ref.sum():,}; matched {matched.sum():,} "
          f"({matched.sum() / max(has_ref.sum(), 1):.3f})")
    return ev


def add_active_permit(ev: pd.DataFrame, conn) -> pd.DataFrame:
    """Permit active at received date (BIN join, interval filter,
    resolve multiples to the largest estimated_job_costs)."""
    pm = pd.read_sql_query("""
        SELECT bin, job_filing_number, issued_date, expired_date,
               estimated_job_costs
        FROM permits
        WHERE bin IS NOT NULL AND bin != ''
          AND issued_date LIKE '____-__-__%'
          AND expired_date LIKE '____-__-__%'
    """, conn)
    pm["issued"] = pd.to_datetime(pm["issued_date"].str[:10],
                                  format="%Y-%m-%d", errors="coerce")
    pm["expired"] = pd.to_datetime(pm["expired_date"].str[:10],
                                   format="%Y-%m-%d", errors="coerce")
    pm["cost"] = pd.to_numeric(pm["estimated_job_costs"], errors="coerce").fillna(0)
    pm = pm.dropna(subset=["issued", "expired"])
    pm = pm[["bin", "job_filing_number", "issued", "expired", "cost"]]

    left = ev[["bin", "received"]].reset_index().rename(columns={"index": "eid"})
    left = left[left["bin"].fillna("") != ""]
    m = left.merge(pm, on="bin", how="inner")
    n_pairs = len(m)
    m = m[(m["issued"] <= m["received"]) & (m["expired"] >= m["received"])]
    m = m.sort_values("cost", ascending=False, kind="stable").drop_duplicates("eid")

    ev["active_permit"] = np.int8(0)
    ev["active_job_filing_number"] = ""
    ev["active_permit_cost"] = np.nan
    ev.loc[m["eid"], "active_permit"] = np.int8(1)
    ev.loc[m["eid"], "active_job_filing_number"] = m["job_filing_number"].to_numpy()
    ev.loc[m["eid"], "active_permit_cost"] = m["cost"].to_numpy()

    by_agency = ev.groupby("agency")["active_permit"].mean()
    print(f"  permit interval join: {n_pairs:,} BIN pairs -> "
          f"{len(m):,} events at active permits "
          f"(agency {by_agency.get(1, float('nan')):.3f} vs "
          f"caller {by_agency.get(0, float('nan')):.3f})")
    return ev


# ── 2. DOB NOW jobs ──────────────────────────────────────────────────────

def build_jobs(conn, pt: pd.DataFrame, pl_class: pd.DataFrame) -> pd.DataFrame:
    jobs = pd.read_sql_query("""
        SELECT job_filing_number, job_type, building_type, initial_cost,
               total_construction_floor_area, existing_dwelling_units,
               proposed_dwelling_units, filing_date, first_permit_date,
               signoff_date, bin, bbl, borough, block, lot
        FROM permits_now
        WHERE first_permit_date >= '2020-01-01'
    """, conn)
    jobs = jobs.drop_duplicates("job_filing_number").reset_index(drop=True)
    for c in ("initial_cost", "total_construction_floor_area",
              "existing_dwelling_units", "proposed_dwelling_units"):
        jobs[c] = pd.to_numeric(jobs[c], errors="coerce")
    for c in ("filing_date", "first_permit_date", "signoff_date"):
        jobs[c] = pd.to_datetime(jobs[c].str[:10], format="%Y-%m-%d",
                                 errors="coerce")

    # permit spans per job filing (ISO strings sort chronologically)
    sp = pd.read_sql_query("""
        SELECT job_filing_number,
               MIN(substr(issued_date, 1, 10)) AS first_issued,
               MAX(substr(expired_date, 1, 10)) AS last_expired,
               COUNT(*) AS n_permits
        FROM permits
        WHERE issued_date LIKE '____-__-__%'
          AND expired_date LIKE '____-__-__%'
        GROUP BY job_filing_number
    """, conn)
    for c in ("first_issued", "last_expired"):
        sp[c] = pd.to_datetime(sp[c], format="%Y-%m-%d", errors="coerce")
    jobs = jobs.merge(sp, on="job_filing_number", how="left")
    jobs["n_permits"] = jobs["n_permits"].fillna(0).astype(int)

    # BBL: keep the DOB NOW value when well-formed, else rebuild from
    # borough/block/lot; tract + class joins use the normalized key
    boro = jobs["borough"].fillna("").astype("str").str.upper().map(BOROUGH_NAME_TO_CODE)
    built = bbl_key(boro, jobs["block"], jobs["lot"])
    raw = jobs["bbl"].fillna("").astype("str").str.strip()
    jobs["bbl"] = raw.where(raw.str.fullmatch(r"\d{10}"), built)
    jobs["key"] = blocklot_key(boro, jobs["block"], jobs["lot"])
    jobs = jobs.merge(pt, on="key", how="left")
    jobs = jobs.merge(pl_class, on="key", how="left")

    # conversion flags
    is_altco = jobs["job_type"].isin(CONVERSION_JOB_TYPES)
    ex0 = jobs["existing_dwelling_units"].eq(0)
    jobs["conversion"] = (is_altco & ex0
                          & jobs["proposed_dwelling_units"].gt(0)).astype("int8")
    jobs["conversion_ge10"] = (is_altco & ex0
                               & jobs["proposed_dwelling_units"].ge(10)).astype("int8")
    office = jobs["bldgclass"].fillna("").astype("str").str[:1].isin(["O", "K"])
    jobs["conversion_office"] = (jobs["conversion"].astype(bool) & office).astype("int8")

    # active span: first permit until signoff, else last permit expiry
    jobs["active_start"] = jobs["first_permit_date"].fillna(jobs["first_issued"])
    jobs["active_end"] = jobs["signoff_date"].fillna(jobs["last_expired"])
    jobs["active_start_month"] = jobs["active_start"].dt.strftime("%Y-%m")
    jobs["active_end_month"] = jobs["active_end"].dt.strftime("%Y-%m")

    print(f"Jobs: {len(jobs):,} first-permitted 2020+; "
          f"permit-span match {(jobs['n_permits'] > 0).mean():.3f}; "
          f"signoff fill {jobs['signoff_date'].notna().mean():.3f}; "
          f"tract match {jobs['bct2020'].notna().mean():.3f}")
    print(f"  conversion relaxed {int(jobs['conversion'].sum()):,} / "
          f"strict>=10u {int(jobs['conversion_ge10'].sum()):,} / "
          f"office-class {int(jobs['conversion_office'].sum()):,}; "
          f"no active_end {jobs['active_end'].isna().sum():,}")
    return jobs


# ── 3. Tract-month panel ─────────────────────────────────────────────────

def build_tract_month(ev: pd.DataFrame, jobs: pd.DataFrame,
                      pt: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    tracts = np.sort(pt["bct2020"].unique())
    t_id = pd.Series(np.arange(len(tracts)), index=tracts)
    n_t, n_m = len(tracts), N_MONTHS
    months = pd.period_range("2020-01", "2026-05", freq="M").astype(str)

    e = ev[ev["bct2020"].notna()]
    cell = (e["bct2020"].map(t_id) * n_m + month_index(e["received"])).to_numpy()

    cols = {"caller_complaints": np.bincount(cell[(e["agency"] == 0).to_numpy()],
                                             minlength=n_t * n_m)}
    fam = e["family"].to_numpy()
    is_agency = (e["agency"] == 1).to_numpy()
    for f in FAMILY_ORDER:
        cols[f"proactive_{f}"] = np.bincount(cell[is_agency & (fam == f)],
                                             minlength=n_t * n_m)

    # active jobs: expand each job's month span (vectorized repeat/arange)
    j = jobs[jobs["bct2020"].notna() & jobs["active_start"].notna()
             & jobs["active_end"].notna()]
    s = month_index(j["active_start"]).clip(lower=0).to_numpy()
    t = month_index(j["active_end"]).clip(upper=n_m - 1).to_numpy()
    keep = t >= s
    tid = j["bct2020"].map(t_id).to_numpy()[keep]
    s, t = s[keep], t[keep]
    lens = t - s + 1
    base = np.repeat(tid * n_m + s, lens)
    offsets = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    cols["active_jobs"] = np.bincount(base + offsets, minlength=n_t * n_m)

    panel = pd.DataFrame({"bct2020": np.repeat(tracts, n_m),
                          "month": np.tile(months, n_t), **cols})
    panel["proactive_total"] = sum(panel[f"proactive_{f}"] for f in FAMILY_ORDER)
    panel = panel.merge(units, on="bct2020", how="left")
    panel["units_res"] = panel["units_res"].fillna(0).astype(int)

    print(f"Panel: {len(panel):,} rows = {n_t:,} tracts x {n_m} months; "
          f"caller {panel['caller_complaints'].sum():,}, "
          f"proactive {panel['proactive_total'].sum():,}, "
          f"job-months {panel['active_jobs'].sum():,}")
    return panel


# ── 4. Verification summary ──────────────────────────────────────────────

def build_summary(ev: pd.DataFrame, jobs: pd.DataFrame,
                  panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric, value, expected=None, note=""):
        check = ""
        if expected is not None:
            check = ("OK" if abs(value - expected) <= 0.02 * abs(expected)
                     else "DIVERGES")
        rows.append({"metric": metric, "value": round(float(value), 4),
                     "expected_wave1": expected, "check": check, "note": note})

    n = len(ev)
    n_agency = int(ev["agency"].sum())
    add("events_rows", n, note="scraped complaints 2020-01..2026-05")
    add("events_agency_rows", n_agency, EXPECT["n_events_agency"],
        "empty ref_311")
    add("agency_share_overall", n_agency / n)
    add("events_tract_match_share", ev["bct2020"].notna().mean())

    for f in FAMILY_ORDER:
        sub = ev[ev["family"] == f]
        add(f"events_rows__{f}", len(sub))
        add(f"agency_share__{f}", sub["agency"].mean())
        add(f"active_permit_share__{f}", sub["active_permit"].mean())

    by_agency = ev.groupby("agency")["active_permit"].mean()
    add("active_permit_share_agency", by_agency[1],
        EXPECT["active_share_agency"], "Wave-1: 54.3% proactive")
    add("active_permit_share_caller", by_agency[0],
        EXPECT["active_share_caller"], "Wave-1: 26.2% caller")
    for pfx in ("8A", "7G"):
        add(f"active_permit_share_prefix_{pfx}",
            ev.loc[ev["category_prefix"] == pfx, "active_permit"].mean(),
            EXPECT[f"active_share_prefix_{pfx}"], "spot check")

    has_ref = ev["ecb_number"] != ""
    add("ecb_ref_rows", int(has_ref.sum()))
    add("ecb_match_share",
        (ev["ecb_severity"].notna() & has_ref).sum() / has_ref.sum(),
        EXPECT["ecb_match_share"], "Wave-1: ~95% of refs")

    add("jobs_rows", len(jobs), note="permits_now first permit 2020+")
    add("jobs_permit_span_share", (jobs["n_permits"] > 0).mean())
    add("jobs_signoff_fill_share", jobs["signoff_date"].notna().mean(),
        EXPECT["signoff_fill_share"], "Wave-1: 64% filled")
    add("jobs_conversion_relaxed", int(jobs["conversion"].sum()),
        note="Alt-CO family, 0 existing units, >0 proposed")
    add("jobs_conversion_strict_ge10", int(jobs["conversion_ge10"].sum()),
        EXPECT["n_conversion_ge10"], "Wave-1: 630 (proposed>=10)")
    add("jobs_conversion_office_class", int(jobs["conversion_office"].sum()),
        note="relaxed + PLUTO bldgclass O/K")

    add("panel_rows", len(panel))
    add("panel_tracts", panel["bct2020"].nunique())
    add("panel_months", panel["month"].nunique())
    add("panel_caller_total", panel["caller_complaints"].sum())
    add("panel_proactive_total", panel["proactive_total"].sum())
    add("panel_active_job_months", panel["active_jobs"].sum())

    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RISK_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    timings = {}

    def stage(name, fn, *args):
        t = time.time()
        out = fn(*args)
        timings[name] = time.time() - t
        return out

    pt, pl_class, units = stage("pluto_lookups", load_pluto_lookups, conn)
    ev = stage("events", build_events, conn, pt)
    ev = stage("ecb_link", add_ecb, ev, conn)
    ev = stage("active_permit", add_active_permit, ev, conn)
    jobs = stage("jobs", build_jobs, conn, pt, pl_class)
    panel = stage("tract_month", build_tract_month, ev, jobs, pt, units)
    conn.close()

    event_cols = ["complaint_number", "received_date", "month",
                  "category_prefix", "category_name", "family", "agency",
                  "bin", "bbl", "bct2020", "nta", "priority", "assigned_to",
                  "disposition_code", "outcome", "ecb_number", "ecb_severity",
                  "ecb_penalty", "active_permit", "active_job_filing_number",
                  "active_permit_cost"]
    job_cols = ["job_filing_number", "job_type", "building_type",
                "initial_cost", "total_construction_floor_area",
                "existing_dwelling_units", "proposed_dwelling_units",
                "bin", "bbl", "bct2020", "bldgclass", "filing_date",
                "first_permit_date", "signoff_date", "first_issued",
                "last_expired", "n_permits", "active_start", "active_end",
                "active_start_month", "active_end_month", "conversion",
                "conversion_ge10", "conversion_office"]

    t = time.time()
    p1 = save_frame(ev[event_cols], "proactive_events")
    p2 = save_frame(jobs[job_cols], "jobs")
    p3 = os.path.join(OUT_DIR, "tract_month.csv.gz")
    panel.to_csv(p3, index=False, compression="gzip")
    timings["write_outputs"] = time.time() - t

    summary = build_summary(ev, jobs, panel)
    p4 = os.path.join(RISK_DIR, "proactive_spine_summary.csv")
    summary.to_csv(p4, index=False)

    print("\n== Verification summary ==")
    print(summary.to_string(index=False))
    diverges = summary[summary["check"] == "DIVERGES"]
    if len(diverges):
        print(f"\nWARNING: {len(diverges)} metric(s) diverge >2% from Wave-1:")
        print(diverges.to_string(index=False))
    else:
        print("\nAll Wave-1 checks within 2%.")

    print("\n== Outputs ==")
    for p in (p1, p2, p3, p4):
        print(f"  {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
    print("\n== Timing ==")
    for k, v in timings.items():
        print(f"  {k:>15}: {v:6.1f}s")
    print(f"  {'TOTAL':>15}: {time.time() - t0:6.1f}s")


if __name__ == "__main__":
    main()
