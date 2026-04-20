"""
Inspector Leniency Analysis: Examiner Design
=============================================

Research question: Conditional on a complaint, does the probability that
a violation is found vary depending on which inspector handles the case?

Design: Leave-one-out examiner leniency (Kling 2006, Dobbie et al. 2018)
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from disposition_codes import (
    classify_disposition,
    classify_severity,
    VIOLATION_CODES,
    NO_VIOLATION_CODES,
    NO_ACCESS_CODES,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

OUTPUT_DIR = config.DATA_DIR / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_analysis_data() -> pd.DataFrame:
    """Load and prepare the analysis dataset."""
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
            b.borough,
            b.priority,
            b.category_description,
            b.assigned_to,
            b.inspector_badge,
            b.comments,
            b.subject,
            b.block,
            b.lot,
            p.bldgclass,
            p.numfloors,
            p.yearbuilt,
            p.landuse,
            p.unitsres,
            p.unitstotal,
            p.bldgarea,
            p.lotarea,
            p.assesstot,
            b.received_time
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        LEFT JOIN pluto p ON p.borocode = {BORO_CASE}
            AND p.block = b.block AND p.lot = b.lot
        WHERE b.inspector_badge IS NOT NULL
          AND o.disposition_code IS NOT NULL
          AND o.disposition_code != ''
    """, conn)
    conn.close()

    # Classify outcomes
    df["outcome"] = df["disposition_code"].apply(classify_disposition)
    df["severity"] = df["disposition_code"].apply(classify_severity)
    df["violation_found"] = (df["outcome"] == "violation").astype(int)
    df["no_violation"] = (df["outcome"] == "no_violation").astype(int)
    df["no_access"] = (df["outcome"] == "no_access").astype(int)

    # Parse dates
    df["date_entered_parsed"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce")
    df["year_month"] = df["date_entered_parsed"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["date_entered_parsed"].dt.dayofweek

    # Numeric building characteristics
    for col in ["numfloors", "yearbuilt", "unitsres", "unitstotal", "bldgarea", "lotarea", "assesstot"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Comment length as proxy for inspector thoroughness
    df["comment_length"] = df["comments"].fillna("").str.len()

    # Time of day from received_time
    def parse_hour(t):
        if pd.isna(t) or not isinstance(t, str) or t in ("0:00", "00:00"):
            return np.nan
        try:
            return int(t.split(":")[0])
        except (ValueError, IndexError):
            return np.nan

    df["received_hour"] = df["received_time"].apply(parse_hour)
    df["time_block"] = pd.cut(
        df["received_hour"],
        bins=[-1, 6, 12, 17, 24],
        labels=["night_early", "morning", "afternoon", "evening"],
    ).astype(str)
    # Fill missing with "unknown"
    df.loc[df["received_hour"].isna(), "time_block"] = "unknown"

    print(f"Loaded {len(df):,} complaints with inspector + disposition")
    print(f"Outcome distribution:")
    print(df["outcome"].value_counts().to_string())
    print(f"\nTime block distribution:")
    print(df["time_block"].value_counts().to_string())
    print()

    return df


def build_analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the analysis sample: substantive outcomes only."""
    # Keep only violation or no_violation (inspector exercised discretion)
    sample = df[df["outcome"].isin(["violation", "no_violation"])].copy()

    # Require inspector has at least 30 cases in the sample
    inspector_counts = sample["inspector_badge"].value_counts()
    valid_inspectors = inspector_counts[inspector_counts >= 30].index
    sample = sample[sample["inspector_badge"].isin(valid_inspectors)]

    print(f"Analysis sample: {len(sample):,} complaints")
    print(f"  Inspectors: {sample['inspector_badge'].nunique()}")
    print(f"  Violation rate: {sample['violation_found'].mean():.1%}")
    print()

    return sample


def compute_leave_one_out_leniency(sample: pd.DataFrame) -> pd.DataFrame:
    """Compute leave-one-out inspector violation rate for each observation."""
    # For each inspector: total violations and total cases
    inspector_stats = sample.groupby("inspector_badge").agg(
        total_violations=("violation_found", "sum"),
        total_cases=("violation_found", "count"),
    )

    # Merge back
    sample = sample.merge(inspector_stats, on="inspector_badge", how="left")

    # Leave-one-out: (total_violations - own_violation) / (total_cases - 1)
    sample["loo_strictness"] = (
        (sample["total_violations"] - sample["violation_found"])
        / (sample["total_cases"] - 1)
    )

    sample = sample.drop(columns=["total_violations", "total_cases"])

    print(f"LOO strictness distribution:")
    print(f"  Mean: {sample['loo_strictness'].mean():.3f}")
    print(f"  Std:  {sample['loo_strictness'].std():.3f}")
    print(f"  P10:  {sample['loo_strictness'].quantile(0.1):.3f}")
    print(f"  P50:  {sample['loo_strictness'].quantile(0.5):.3f}")
    print(f"  P90:  {sample['loo_strictness'].quantile(0.9):.3f}")
    print()

    return sample


def demean_by_group(arr: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract group means (within transformation for absorbing FEs)."""
    result = arr.copy().astype(float)
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() > 1:
            result[mask] -= np.nanmean(result[mask])
    return result


def run_balance_tests(sample: pd.DataFrame):
    """Test whether inspector strictness is quasi-randomly assigned.

    If assignment is as-good-as-random conditional on unit + time,
    then LOO strictness should be uncorrelated with case observables.
    """
    from scipy import stats as scipy_stats

    print("=" * 60)
    print("BALANCE TESTS: Is inspector assignment quasi-random?")
    print("=" * 60)
    print()

    # Covariates to test
    covariates = {
        "numfloors": "Number of floors",
        "yearbuilt": "Year built",
        "bldgarea": "Building area (sqft)",
        "assesstot": "Assessed value",
        "unitsres": "Residential units",
        "received_hour": "Hour of day (received)",
        "comment_length": "Comment length",
    }

    # Simple correlations (unconditional)
    print("Unconditional correlations with LOO strictness:")
    for col, label in covariates.items():
        valid = sample[[col, "loo_strictness"]].dropna()
        if len(valid) < 100:
            continue
        corr, pval = scipy_stats.pearsonr(valid[col], valid["loo_strictness"])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {label:30s}  r = {corr:+.4f}  p = {pval:.4f} {sig}")
    print()

    # Conditional: residualize LOO strictness on unit × time FEs using demean
    print("Conditional balance (residualizing on assigned_to × year_month):")
    units = sample["assigned_to"].fillna("UNK").values
    ym = sample["year_month"].fillna("UNK").values
    groups = np.array([f"{u}|{t}" for u, t in zip(units, ym)])
    resid_strict = demean_by_group(sample["loo_strictness"].values, groups)

    for col, label in covariates.items():
        vals = sample[col].values
        valid = ~np.isnan(vals) & ~np.isnan(resid_strict)
        if valid.sum() < 100:
            continue
        corr, pval = scipy_stats.pearsonr(vals[valid], resid_strict[valid])
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {label:30s}  r = {corr:+.4f}  p = {pval:.4f} {sig}")
    print()


def run_main_regression(sample: pd.DataFrame):
    """Run the main leniency regression using within-transformation.

    Uses Frisch-Waugh / demean approach to absorb high-dimensional FEs
    without creating huge dummy matrices.
    """
    print("=" * 60)
    print("MAIN REGRESSION: Does inspector strictness predict outcomes?")
    print("=" * 60)
    print()

    from numpy.linalg import lstsq

    y = sample["violation_found"].values.astype(float)
    strictness = sample["loo_strictness"].values.astype(float)

    valid = ~np.isnan(y) & ~np.isnan(strictness)
    y, strictness = y[valid], strictness[valid]

    # ── Specification 1: No controls ────────────────────────────────────
    X1 = np.column_stack([np.ones(len(y)), strictness])
    beta1, _, _, _ = lstsq(X1, y, rcond=None)
    resid1 = y - X1 @ beta1
    se1 = np.sqrt(np.sum(resid1**2) / (len(y) - 2) * np.linalg.inv(X1.T @ X1)[1, 1])

    print(f"Spec 1: No controls")
    print(f"  β(strictness) = {beta1[1]:.4f}  (SE = {se1:.4f})")
    print(f"  N = {len(y):,}")
    print()

    # ── Specification 2: Category FEs (within transformation) ───────────
    cats = sample["category_description"].fillna("UNKNOWN").values[valid]
    y2 = demean_by_group(y, cats)
    s2 = demean_by_group(strictness, cats)
    X2 = np.column_stack([np.ones(len(y2)), s2])
    beta2, _, _, _ = lstsq(X2, y2, rcond=None)
    resid2 = y2 - X2 @ beta2
    se2 = np.sqrt(np.sum(resid2**2) / (len(y2) - 2) * np.linalg.inv(X2.T @ X2)[1, 1])

    print(f"Spec 2: Category FEs (within-transformation)")
    print(f"  β(strictness) = {beta2[1]:.4f}  (SE = {se2:.4f})")
    print(f"  N = {len(y2):,}, categories = {len(np.unique(cats))}")
    print()

    # ── Specification 3: Category + Unit FEs ────────────────────────────
    units = sample["assigned_to"].fillna("UNKNOWN").values[valid]
    cat_unit = np.array([f"{c}|{u}" for c, u in zip(cats, units)])
    y3 = demean_by_group(y, cat_unit)
    s3 = demean_by_group(strictness, cat_unit)
    X3 = np.column_stack([np.ones(len(y3)), s3])
    beta3, _, _, _ = lstsq(X3, y3, rcond=None)
    resid3 = y3 - X3 @ beta3
    se3 = np.sqrt(np.sum(resid3**2) / (len(y3) - 2) * np.linalg.inv(X3.T @ X3)[1, 1])

    print(f"Spec 3: Category × Unit FEs")
    print(f"  β(strictness) = {beta3[1]:.4f}  (SE = {se3:.4f})")
    print(f"  N = {len(y3):,}, FE groups = {len(np.unique(cat_unit))}")
    print()

    # ── Specification 4: Category × Unit × Year-Month FEs ──────────────
    ym = sample["year_month"].fillna("UNKNOWN").values[valid]
    cat_unit_ym = np.array([f"{c}|{u}|{t}" for c, u, t in zip(cats, units, ym)])
    y4 = demean_by_group(y, cat_unit_ym)
    s4 = demean_by_group(strictness, cat_unit_ym)
    X4 = np.column_stack([np.ones(len(y4)), s4])
    beta4, _, _, _ = lstsq(X4, y4, rcond=None)
    resid4 = y4 - X4 @ beta4
    se4 = np.sqrt(np.sum(resid4**2) / (len(y4) - 2) * np.linalg.inv(X4.T @ X4)[1, 1])

    print(f"Spec 4: Category × Unit × Year-Month FEs")
    print(f"  β(strictness) = {beta4[1]:.4f}  (SE = {se4:.4f})")
    print(f"  N = {len(y4):,}, FE groups = {len(np.unique(cat_unit_ym))}")
    print()

    # ── Specification 5: + Time-of-day FEs ─────────────────────────────
    time_blocks = sample["time_block"].fillna("unknown").values[valid]
    cat_unit_ym_time = np.array([
        f"{c}|{u}|{t}|{tb}" for c, u, t, tb in zip(cats, units, ym, time_blocks)
    ])
    y5 = demean_by_group(y, cat_unit_ym_time)
    s5 = demean_by_group(strictness, cat_unit_ym_time)
    X5 = np.column_stack([np.ones(len(y5)), s5])
    beta5, _, _, _ = lstsq(X5, y5, rcond=None)
    resid5 = y5 - X5 @ beta5
    se5 = np.sqrt(np.sum(resid5**2) / (len(y5) - 2) * np.linalg.inv(X5.T @ X5)[1, 1])

    print(f"Spec 5: Category × Unit × Year-Month × Time-of-Day FEs")
    print(f"  β(strictness) = {beta5[1]:.4f}  (SE = {se5:.4f})")
    print(f"  N = {len(y5):,}, FE groups = {len(np.unique(cat_unit_ym_time))}")
    print()

    # ── Specification 6: + Priority + Day-of-Week ──────────────────────
    pri = sample["priority"].fillna("UNK").values[valid]
    dow = sample["day_of_week"].fillna(-1).astype(str).values[valid]
    full_fe = np.array([
        f"{c}|{u}|{t}|{tb}|{p}|{d}"
        for c, u, t, tb, p, d in zip(cats, units, ym, time_blocks, pri, dow)
    ])
    y6 = demean_by_group(y, full_fe)
    s6 = demean_by_group(strictness, full_fe)
    X6 = np.column_stack([np.ones(len(y6)), s6])
    beta6, _, _, _ = lstsq(X6, y6, rcond=None)
    resid6 = y6 - X6 @ beta6
    se6 = np.sqrt(np.sum(resid6**2) / (len(y6) - 2) * np.linalg.inv(X6.T @ X6)[1, 1])

    print(f"Spec 6: + Priority + Day-of-Week FEs (kitchen sink)")
    print(f"  β(strictness) = {beta6[1]:.4f}  (SE = {se6:.4f})")
    print(f"  N = {len(y6):,}, FE groups = {len(np.unique(full_fe))}")
    print()

    # ── Summary table ───────────────────────────────────────────────────
    print("Summary:")
    print(f"  {'Specification':<55} {'β':>8} {'SE':>8} {'t':>8}")
    print(f"  {'-'*55} {'-'*8} {'-'*8} {'-'*8}")
    for label, b, s in [
        ("(1) No controls", beta1[1], se1),
        ("(2) Category FEs", beta2[1], se2),
        ("(3) Category × Unit FEs", beta3[1], se3),
        ("(4) Cat × Unit × Year-Month", beta4[1], se4),
        ("(5) + Time-of-Day", beta5[1], se5),
        ("(6) + Priority + Day-of-Week (kitchen sink)", beta6[1], se6),
    ]:
        t = b / s if s > 0 else 0
        print(f"  {label:<55} {b:>8.4f} {s:>8.4f} {t:>8.2f}")
    print()

    return beta6[1], se6


def inspector_fixed_effects_decomposition(sample: pd.DataFrame):
    """How much outcome variance is explained by inspector identity?"""
    print("=" * 60)
    print("VARIANCE DECOMPOSITION: How much does inspector identity matter?")
    print("=" * 60)
    print()

    y = sample["violation_found"].values.astype(float)
    total_var = np.var(y)

    # R² from inspector FEs alone
    inspector_means = sample.groupby("inspector_badge")["violation_found"].transform("mean")
    r2_inspector = 1 - np.var(y - inspector_means.values) / total_var

    # R² from category FEs alone
    cat_means = sample.groupby("category_description")["violation_found"].transform("mean")
    r2_category = 1 - np.var(y - cat_means.values) / total_var

    # R² from unit FEs alone
    unit_means = sample.groupby("assigned_to")["violation_found"].transform("mean")
    r2_unit = 1 - np.var(y - unit_means.values) / total_var

    print(f"Total outcome variance: {total_var:.4f}")
    print(f"  R² from inspector FEs alone:  {r2_inspector:.4f} ({r2_inspector*100:.1f}%)")
    print(f"  R² from category FEs alone:   {r2_category:.4f} ({r2_category*100:.1f}%)")
    print(f"  R² from unit FEs alone:       {r2_unit:.4f} ({r2_unit*100:.1f}%)")
    print()


def no_access_analysis(df: pd.DataFrame):
    """Analyze no-access rates as a measure of inspector effort."""
    print("=" * 60)
    print("NO-ACCESS ANALYSIS: Inspector effort in gaining access")
    print("=" * 60)
    print()

    # Sample: all complaints with an inspector (including no-access)
    sub = df[df["outcome"].isin(["violation", "no_violation", "no_access"])].copy()

    inspector_counts = sub["inspector_badge"].value_counts()
    valid = inspector_counts[inspector_counts >= 30].index
    sub = sub[sub["inspector_badge"].isin(valid)]

    sub["no_access_flag"] = (sub["outcome"] == "no_access").astype(int)

    rates = sub.groupby("inspector_badge").agg(
        cases=("no_access_flag", "count"),
        no_access_rate=("no_access_flag", "mean"),
    )

    print(f"No-access rate distribution across {len(rates)} inspectors:")
    print(f"  Mean:   {rates['no_access_rate'].mean():.1%}")
    print(f"  Median: {rates['no_access_rate'].median():.1%}")
    print(f"  P10:    {rates['no_access_rate'].quantile(0.1):.1%}")
    print(f"  P90:    {rates['no_access_rate'].quantile(0.9):.1%}")
    print()

    # Correlation between strictness and no-access rate
    strict_rates = sub.groupby("inspector_badge").agg(
        violation_rate=("violation_found", lambda x: x[x.index.isin(
            sub[sub["outcome"].isin(["violation", "no_violation"])].index
        )].mean()),
        no_access_rate=("no_access_flag", "mean"),
    ).dropna()

    if len(strict_rates) > 10:
        from scipy import stats as scipy_stats
        corr, pval = scipy_stats.pearsonr(
            strict_rates["violation_rate"],
            strict_rates["no_access_rate"]
        )
        print(f"Correlation(violation rate, no-access rate) across inspectors:")
        print(f"  r = {corr:+.3f}, p = {pval:.4f}")
        print(f"  {'Strict inspectors report MORE no-access' if corr > 0 else 'Strict inspectors report LESS no-access'}")
    print()


def save_inspector_profiles(sample: pd.DataFrame, df: pd.DataFrame):
    """Save inspector-level summary for further analysis."""
    profiles = sample.groupby("inspector_badge").agg(
        total_cases=("violation_found", "count"),
        violation_rate=("violation_found", "mean"),
        mean_severity=("severity", "mean"),
        mean_comment_length=("comment_length", "mean"),
        n_categories=("category_description", "nunique"),
        top_category=("category_description", lambda x: x.value_counts().index[0]),
        n_boroughs=("borough", "nunique"),
        primary_unit=("assigned_to", lambda x: x.value_counts().index[0]),
        loo_strictness=("loo_strictness", "mean"),
    ).reset_index()

    # Add no-access rate from full data
    no_access = df[df["inspector_badge"].isin(profiles["inspector_badge"])]
    na_rates = no_access.groupby("inspector_badge")["no_access"].mean()
    profiles = profiles.merge(na_rates.rename("no_access_rate"), on="inspector_badge", how="left")

    output_path = OUTPUT_DIR / "inspector_profiles.csv"
    profiles.to_csv(output_path, index=False)
    print(f"Saved {len(profiles)} inspector profiles to {output_path}")

    return profiles


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("INSPECTOR LENIENCY ANALYSIS")
    print("Examiner Design: Leave-One-Out Strictness Measure")
    print("=" * 60)
    print()

    # Load data
    df = load_analysis_data()

    # Build analysis sample
    sample = build_analysis_sample(df)

    # Compute LOO leniency
    sample = compute_leave_one_out_leniency(sample)

    # Variance decomposition
    inspector_fixed_effects_decomposition(sample)

    # Balance tests
    try:
        run_balance_tests(sample)
    except ImportError:
        print("(scipy not available — skipping balance tests)")
        print()

    # Main regression
    try:
        beta, se = run_main_regression(sample)
    except Exception as e:
        print(f"Regression failed: {e}")
        print()

    # No-access analysis
    no_access_analysis(df)

    # Save profiles
    profiles = save_inspector_profiles(sample, df)

    print()
    print("=" * 60)
    print("DONE. Output saved to data/analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
