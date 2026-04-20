"""
Build the analysis dataset: link complaints → inspections → permits → violations.

This creates the master panel needed for the inspector leniency analysis:
1. Load complaint-inspection data (with inspector strictness)
2. Construct BBL for each complaint
3. Link to subsequent permits at the same BBL
4. Link to subsequent violations at the same BBL
5. Identify neighbors for spatial spillover analysis

Output: data/analysis/master_panel.parquet (or .csv)
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from analysis_config import make_bbl, BOROUGH_NAME_TO_CODE, COMPLIANCE_WINDOWS_DAYS
from disposition_codes import classify_disposition, classify_severity, VIOLATION_CODES

OUTPUT_DIR = config.DATA_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_complaints() -> pd.DataFrame:
    """Load complaint-inspection data with inspector info."""
    conn = sqlite3.connect(str(config.DB_PATH))

    BORO_CASE = """
        CASE b.borough
            WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
            WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
            WHEN 'STATEN ISLAND' THEN '5'
        END
    """

    df = pd.read_sql_query(f"""
        SELECT
            o.complaint_number,
            o.disposition_code,
            o.date_entered,
            o.inspection_date,
            o.community_board,
            o.status,
            b.borough,
            b.priority,
            b.category_description,
            b.assigned_to,
            b.inspector_badge,
            b.block,
            b.lot,
            b.received_time,
            p.latitude,
            p.longitude,
            p.bldgclass,
            p.numfloors,
            p.yearbuilt,
            p.landuse,
            p.unitsres,
            p.assesstot,
            p.bldgarea
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        LEFT JOIN pluto p ON p.borocode = {BORO_CASE}
            AND p.block = b.block AND p.lot = b.lot
        WHERE b.inspector_badge IS NOT NULL
          AND o.disposition_code IS NOT NULL
          AND o.disposition_code != ''
          AND b.block IS NOT NULL
          AND b.lot IS NOT NULL
    """, conn)
    conn.close()

    # Construct BBL
    df["boro_code"] = df["borough"].map(BOROUGH_NAME_TO_CODE)
    df["bbl"] = df.apply(lambda r: make_bbl(r["boro_code"], r["block"], r["lot"]), axis=1)

    # Parse dates
    df["inspection_dt"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y", errors="coerce")
    df["entered_dt"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce")
    df["year_month"] = df["entered_dt"].dt.to_period("M").astype(str)

    # Outcomes
    df["outcome"] = df["disposition_code"].apply(classify_disposition)
    df["severity"] = df["disposition_code"].apply(classify_severity)
    df["violation_found"] = (df["outcome"] == "violation").astype(int)

    # Numeric building vars
    for col in ["numfloors", "yearbuilt", "unitsres", "assesstot", "bldgarea"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df):,} complaints with inspector + BBL")
    print(f"  Unique BBLs: {df['bbl'].nunique():,}")
    print(f"  Unique inspectors: {df['inspector_badge'].nunique()}")
    return df


def compute_strictness(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leave-one-out inspector strictness on the substantive sample."""
    # Restrict to substantive outcomes for strictness computation
    sub = df[df["outcome"].isin(["violation", "no_violation"])].copy()

    # Require min cases
    counts = sub["inspector_badge"].value_counts()
    valid = counts[counts >= 30].index
    sub = sub[sub["inspector_badge"].isin(valid)]

    # Inspector-level stats
    stats = sub.groupby("inspector_badge")["violation_found"].agg(["sum", "count"])
    stats.columns = ["total_viol", "total_n"]

    sub = sub.merge(stats, on="inspector_badge")
    sub["loo_strictness"] = (sub["total_viol"] - sub["violation_found"]) / (sub["total_n"] - 1)
    sub = sub.drop(columns=["total_viol", "total_n"])

    # Also compute inspector-level mean for use in other samples
    inspector_means = sub.groupby("inspector_badge")["violation_found"].mean()
    inspector_means.name = "inspector_strictness"

    print(f"Strictness computed for {sub['inspector_badge'].nunique()} inspectors")
    print(f"  Mean: {sub['loo_strictness'].mean():.3f}")
    print(f"  SD:   {sub['loo_strictness'].std():.3f}")

    return sub, inspector_means


def link_permits(complaints: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """For each complaint, find permits filed at the same BBL within compliance windows."""
    print("\nLinking complaints → permits...")

    # Load permits with dates
    permits = pd.read_sql_query("""
        SELECT bbl, bin, issued_date, approved_date, work_type, filing_reason,
               estimated_job_costs, permit_status, job_description
        FROM permits
        WHERE bbl IS NOT NULL AND bbl != ''
          AND issued_date IS NOT NULL
    """, conn)

    # Also load DOB NOW permits
    permits_now = pd.read_sql_query("""
        SELECT bbl, bin, filing_date, approved_date, first_permit_date,
               job_type, initial_cost, building_type, filing_status
        FROM permits_now
        WHERE bbl IS NOT NULL AND bbl != ''
          AND filing_date IS NOT NULL
    """, conn)

    # Parse dates
    permits["issued_dt"] = pd.to_datetime(permits["issued_date"], errors="coerce")
    permits_now["filing_dt"] = pd.to_datetime(permits_now["filing_date"], errors="coerce")

    # Parse costs
    permits["cost"] = pd.to_numeric(
        permits["estimated_job_costs"].str.replace(r"[,$]", "", regex=True),
        errors="coerce"
    )
    permits_now["cost"] = pd.to_numeric(permits_now["initial_cost"], errors="coerce")

    print(f"  BIS permits: {len(permits):,} with issued date")
    print(f"  DOB NOW permits: {len(permits_now):,} with filing date")

    # For each complaint, count permits within each window
    results = []
    bbls = complaints[["complaint_number", "bbl", "inspection_dt"]].dropna(subset=["inspection_dt"])

    # Pre-index permits by BBL for speed
    permit_by_bbl = permits.groupby("bbl")
    pnow_by_bbl = permits_now.groupby("bbl")

    total = len(bbls)
    for i, (_, row) in enumerate(bbls.iterrows()):
        if (i + 1) % 50000 == 0:
            print(f"  Processing {i+1:,}/{total:,}...")

        bbl = row["bbl"]
        insp_dt = row["inspection_dt"]
        rec = {"complaint_number": row["complaint_number"]}

        # BIS permits after inspection
        if bbl in permit_by_bbl.groups:
            bbl_permits = permit_by_bbl.get_group(bbl)
            after = bbl_permits[bbl_permits["issued_dt"] > insp_dt]

            for window in COMPLIANCE_WINDOWS_DAYS:
                within = after[after["issued_dt"] <= insp_dt + pd.Timedelta(days=window)]
                rec[f"permits_bis_{window}d"] = len(within)
                rec[f"permit_cost_bis_{window}d"] = within["cost"].sum() if len(within) > 0 else 0
        else:
            for window in COMPLIANCE_WINDOWS_DAYS:
                rec[f"permits_bis_{window}d"] = 0
                rec[f"permit_cost_bis_{window}d"] = 0

        # DOB NOW permits after inspection
        if bbl in pnow_by_bbl.groups:
            bbl_pnow = pnow_by_bbl.get_group(bbl)
            after = bbl_pnow[bbl_pnow["filing_dt"] > insp_dt]

            for window in COMPLIANCE_WINDOWS_DAYS:
                within = after[after["filing_dt"] <= insp_dt + pd.Timedelta(days=window)]
                rec[f"permits_now_{window}d"] = len(within)
                rec[f"permit_cost_now_{window}d"] = within["cost"].sum() if len(within) > 0 else 0
        else:
            for window in COMPLIANCE_WINDOWS_DAYS:
                rec[f"permits_now_{window}d"] = 0
                rec[f"permit_cost_now_{window}d"] = 0

        # Combined
        for window in COMPLIANCE_WINDOWS_DAYS:
            rec[f"any_permit_{window}d"] = int(
                rec[f"permits_bis_{window}d"] + rec[f"permits_now_{window}d"] > 0
            )
            rec[f"total_permits_{window}d"] = (
                rec[f"permits_bis_{window}d"] + rec[f"permits_now_{window}d"]
            )
            rec[f"total_permit_cost_{window}d"] = (
                rec[f"permit_cost_bis_{window}d"] + rec[f"permit_cost_now_{window}d"]
            )

        results.append(rec)

    permit_df = pd.DataFrame(results)
    print(f"  Linked {len(permit_df):,} complaints to permit outcomes")

    # Summary
    for window in [90, 180, 365]:
        rate = permit_df[f"any_permit_{window}d"].mean()
        avg_n = permit_df[f"total_permits_{window}d"].mean()
        print(f"  {window}d: {rate:.1%} have any permit, avg {avg_n:.2f} permits")

    return permit_df


def link_future_violations(complaints: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """For each complaint, find future violations at the same BBL."""
    print("\nLinking complaints → future violations...")

    ecb = pd.read_sql_query("""
        SELECT boro, block, lot, issue_date, severity, violation_type,
               penality_imposed
        FROM ecb_violations
        WHERE issue_date IS NOT NULL AND boro IS NOT NULL
    """, conn)

    ecb["bbl"] = ecb.apply(lambda r: make_bbl(r["boro"], r["block"], r["lot"]), axis=1)
    ecb["issue_dt"] = pd.to_datetime(ecb["issue_date"], format="%Y%m%d", errors="coerce")
    ecb["penalty"] = pd.to_numeric(
        ecb["penality_imposed"].str.replace(r"[,$]", "", regex=True),
        errors="coerce"
    )

    print(f"  ECB violations: {len(ecb):,}")

    bbls = complaints[["complaint_number", "bbl", "inspection_dt"]].dropna(subset=["inspection_dt"])
    ecb_by_bbl = ecb.groupby("bbl")

    results = []
    total = len(bbls)
    for i, (_, row) in enumerate(bbls.iterrows()):
        if (i + 1) % 50000 == 0:
            print(f"  Processing {i+1:,}/{total:,}...")

        bbl = row["bbl"]
        insp_dt = row["inspection_dt"]
        rec = {"complaint_number": row["complaint_number"]}

        if bbl in ecb_by_bbl.groups:
            bbl_ecb = ecb_by_bbl.get_group(bbl)
            after = bbl_ecb[bbl_ecb["issue_dt"] > insp_dt]

            for window in COMPLIANCE_WINDOWS_DAYS:
                within = after[after["issue_dt"] <= insp_dt + pd.Timedelta(days=window)]
                rec[f"future_ecb_{window}d"] = len(within)
                rec[f"future_penalty_{window}d"] = within["penalty"].sum() if len(within) > 0 else 0
        else:
            for window in COMPLIANCE_WINDOWS_DAYS:
                rec[f"future_ecb_{window}d"] = 0
                rec[f"future_penalty_{window}d"] = 0

        results.append(rec)

    viol_df = pd.DataFrame(results)
    print(f"  Linked {len(viol_df):,} complaints to future violation outcomes")

    for window in [90, 180, 365]:
        rate = (viol_df[f"future_ecb_{window}d"] > 0).mean()
        print(f"  {window}d: {rate:.1%} have future ECB violations")

    return viol_df


def build_neighbor_index(complaints: pd.DataFrame) -> pd.DataFrame:
    """Build same-block neighbor pairs for spatial spillover analysis."""
    print("\nBuilding neighbor index (same block)...")

    # Group by boro + block
    complaints["boro_block"] = complaints["boro_code"].astype(str) + "_" + complaints["block"].astype(str)
    block_groups = complaints.groupby("boro_block")["bbl"].apply(set).to_dict()

    # For each complaint, count distinct neighbor BBLs on same block
    def count_neighbors(row):
        bb = row["boro_block"]
        if bb not in block_groups:
            return 0
        return len(block_groups[bb]) - 1  # Exclude self

    complaints["n_block_neighbors"] = complaints.apply(count_neighbors, axis=1)
    print(f"  Mean neighbors per complaint: {complaints['n_block_neighbors'].mean():.1f}")
    print(f"  Median: {complaints['n_block_neighbors'].median():.0f}")

    return complaints


def main():
    print("=" * 60)
    print("BUILDING MASTER ANALYSIS DATASET")
    print("=" * 60)

    # Step 1: Load complaints
    complaints = load_complaints()

    # Step 2: Compute strictness
    analysis_sample, inspector_means = compute_strictness(complaints)

    # Step 3: Link permits (if available)
    conn = sqlite3.connect(str(config.DB_PATH))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "permits" in tables:
        pcount = conn.execute("SELECT COUNT(*) FROM permits").fetchone()[0]
        if pcount > 0:
            permit_outcomes = link_permits(analysis_sample, conn)
            analysis_sample = analysis_sample.merge(
                permit_outcomes, on="complaint_number", how="left"
            )
        else:
            print("\nPermits table is empty — skipping permit linkage")
    else:
        print("\nPermits table not found — skipping permit linkage")

    # Step 4: Link future violations (if available)
    if "ecb_violations" in tables:
        ecount = conn.execute("SELECT COUNT(*) FROM ecb_violations").fetchone()[0]
        if ecount > 0:
            viol_outcomes = link_future_violations(analysis_sample, conn)
            analysis_sample = analysis_sample.merge(
                viol_outcomes, on="complaint_number", how="left"
            )
        else:
            print("\nECB violations table is empty — skipping violation linkage")
    else:
        print("\nECB violations table not found — skipping violation linkage")

    conn.close()

    # Step 5: Build neighbor index
    analysis_sample = build_neighbor_index(analysis_sample)

    # Save
    output_path = OUTPUT_DIR / "master_panel.csv"
    analysis_sample.to_csv(output_path, index=False)
    print(f"\nSaved master panel: {len(analysis_sample):,} rows × {analysis_sample.shape[1]} cols")
    print(f"  → {output_path}")

    # Quick summary stats
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Observations: {len(analysis_sample):,}")
    print(f"Inspectors: {analysis_sample['inspector_badge'].nunique()}")
    print(f"Unique BBLs: {analysis_sample['bbl'].nunique():,}")
    print(f"Violation rate: {analysis_sample['violation_found'].mean():.1%}")

    for window in [90, 180, 365]:
        col = f"any_permit_{window}d"
        if col in analysis_sample.columns:
            print(f"Any permit within {window}d: {analysis_sample[col].mean():.1%}")
        col = f"future_ecb_{window}d"
        if col in analysis_sample.columns:
            print(f"Future ECB within {window}d: {(analysis_sample[col] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
