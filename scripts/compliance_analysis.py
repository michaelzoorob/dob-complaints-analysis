"""
Causal Analysis: Inspector Strictness → Compliance Behavior
============================================================

Key questions:
1. Do stricter inspectors induce more permit filings (real compliance)?
2. Do stricter inspectors reduce future violations (deterrence)?
3. Are these effects concentrated in specific building/complaint types?

Uses the master panel built by build_analysis_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

OUTPUT_DIR = config.DATA_DIR / "analysis"


def demean_by_group(arr: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Within transformation: subtract group means. Vectorized via pandas."""
    s = pd.Series(arr, dtype=float)
    g = pd.Series(groups)
    return (s - s.groupby(g).transform("mean")).values


def within_regression(y, x, groups, label=""):
    """Run within-group regression (FE absorbed via demeaning)."""
    valid = ~np.isnan(y) & ~np.isnan(x)
    y, x, g = y[valid], x[valid], groups[valid]

    y_dm = demean_by_group(y, g)
    x_dm = demean_by_group(x, g)

    X = np.column_stack([np.ones(len(y_dm)), x_dm])
    beta, _, _, _ = np.linalg.lstsq(X, y_dm, rcond=None)
    resid = y_dm - X @ beta
    n = len(y_dm)
    n_groups = len(np.unique(g))

    # SE
    se = np.sqrt(np.sum(resid**2) / (n - 2) * np.linalg.inv(X.T @ X)[1, 1])
    t = beta[1] / se if se > 0 else 0

    return {
        "label": label, "beta": beta[1], "se": se, "t": t,
        "n": n, "n_groups": n_groups, "y_mean": np.mean(y),
    }


def main():
    # Load master panel
    print("Loading master panel...")
    df = pd.read_csv(OUTPUT_DIR / "master_panel.csv")
    print(f"  {len(df):,} observations, {df.shape[1]} columns")
    print()

    # Attach NTA (neighborhood tabulation area) from the spatial join table
    import sqlite3
    conn = sqlite3.connect(str(config.DB_PATH))
    nta = pd.read_sql_query("SELECT complaint_number, nta FROM complaint_nta", conn)
    conn.close()
    df["complaint_number"] = df["complaint_number"].astype(str)
    nta["complaint_number"] = nta["complaint_number"].astype(str)
    df = df.merge(nta, on="complaint_number", how="left")
    print(f"  NTA matched: {df['nta'].notna().mean():.1%} of rows, {df['nta'].nunique()} NTAs")
    print()

    # Build FE groups
    df["cat_unit"] = df["category_description"].fillna("UNK") + "|" + df["assigned_to"].fillna("UNK")
    df["cat_unit_ym"] = df["cat_unit"] + "|" + df["year_month"].fillna("UNK")

    strictness = df["loo_strictness"].values

    # ================================================================
    # PART 1: COMPLIANCE (PERMITS)
    # ================================================================
    print("=" * 70)
    print("PART 1: INSPECTOR STRICTNESS → PERMIT FILING (COMPLIANCE)")
    print("=" * 70)
    print()

    # Main results table
    print(f"{'Outcome':<35} {'β':>8} {'SE':>8} {'t':>7} {'N':>10} {'Y mean':>8}")
    print("-" * 80)

    groups = df["cat_unit_ym"].values

    for window in [30, 60, 90, 180, 365]:
        # Any permit (extensive margin)
        col = f"any_permit_{window}d"
        if col in df.columns:
            r = within_regression(df[col].values.astype(float), strictness, groups,
                                  f"Any permit ({window}d)")
            print(f"  {r['label']:<33} {r['beta']:>8.4f} {r['se']:>8.4f} {r['t']:>7.2f} {r['n']:>10,} {r['y_mean']:>8.3f}")

    print()
    for window in [90, 180, 365]:
        # Number of permits (intensive margin)
        col = f"total_permits_{window}d"
        if col in df.columns:
            r = within_regression(df[col].values.astype(float), strictness, groups,
                                  f"N permits ({window}d)")
            print(f"  {r['label']:<33} {r['beta']:>8.4f} {r['se']:>8.4f} {r['t']:>7.2f} {r['n']:>10,} {r['y_mean']:>8.3f}")

    print()
    for window in [90, 180, 365]:
        # Permit cost (investment magnitude)
        col = f"total_permit_cost_{window}d"
        if col in df.columns:
            # Log cost + 1 for interpretability
            y = np.log1p(df[col].values.astype(float))
            r = within_regression(y, strictness, groups,
                                  f"Log permit cost ({window}d)")
            print(f"  {r['label']:<33} {r['beta']:>8.4f} {r['se']:>8.4f} {r['t']:>7.2f} {r['n']:>10,} {r['y_mean']:>8.3f}")

    # ================================================================
    # PART 2: DETERRENCE (FUTURE VIOLATIONS)
    # ================================================================
    print()
    print("=" * 70)
    print("PART 2: INSPECTOR STRICTNESS → FUTURE VIOLATIONS (DETERRENCE)")
    print("=" * 70)
    print()

    print(f"{'Outcome':<35} {'β':>8} {'SE':>8} {'t':>7} {'N':>10} {'Y mean':>8}")
    print("-" * 80)

    for window in [30, 60, 90, 180, 365]:
        col = f"future_ecb_{window}d"
        if col in df.columns:
            # Any future ECB violation
            y = (df[col].values > 0).astype(float)
            r = within_regression(y, strictness, groups,
                                  f"Any future ECB ({window}d)")
            print(f"  {r['label']:<33} {r['beta']:>8.4f} {r['se']:>8.4f} {r['t']:>7.2f} {r['n']:>10,} {r['y_mean']:>8.3f}")

    print()
    for window in [90, 180, 365]:
        col = f"future_penalty_{window}d"
        if col in df.columns:
            y = np.log1p(df[col].values.astype(float))
            r = within_regression(y, strictness, groups,
                                  f"Log future penalties ({window}d)")
            print(f"  {r['label']:<33} {r['beta']:>8.4f} {r['se']:>8.4f} {r['t']:>7.2f} {r['n']:>10,} {r['y_mean']:>8.3f}")

    # ================================================================
    # PART 3: HETEROGENEITY
    # ================================================================
    print()
    print("=" * 70)
    print("PART 3: HETEROGENEITY BY BUILDING TYPE")
    print("=" * 70)
    print()

    # Split by land use
    df["landuse_cat"] = df["landuse"].map({
        "1": "1-2 Family", "2": "Multi-Family", "3": "Mixed Use",
        "4": "Commercial", "5": "Industrial",
    }).fillna("Other")

    print(f"{'Subgroup':<25} {'Any permit 90d':>16} {'Future ECB 90d':>16}")
    print(f"{'':25} {'β (SE)':>16} {'β (SE)':>16}")
    print("-" * 60)

    for landuse_cat in ["1-2 Family", "Multi-Family", "Mixed Use", "Commercial"]:
        sub = df[df["landuse_cat"] == landuse_cat]
        if len(sub) < 500:
            continue
        g = sub["cat_unit_ym"].values
        s = sub["loo_strictness"].values

        r1 = within_regression(sub["any_permit_90d"].values.astype(float), s, g)
        r2 = within_regression((sub["future_ecb_90d"].values > 0).astype(float), s, g)

        print(f"  {landuse_cat:<23} {r1['beta']:>7.3f} ({r1['se']:.3f})   {r2['beta']:>7.3f} ({r2['se']:.3f})")

    # Split by borough
    print()
    print(f"{'Borough':<25} {'Any permit 90d':>16} {'Future ECB 90d':>16}")
    print(f"{'':25} {'β (SE)':>16} {'β (SE)':>16}")
    print("-" * 60)

    for boro in ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]:
        sub = df[df["borough"] == boro]
        if len(sub) < 500:
            continue
        g = sub["cat_unit_ym"].values
        s = sub["loo_strictness"].values

        r1 = within_regression(sub["any_permit_90d"].values.astype(float), s, g)
        r2 = within_regression((sub["future_ecb_90d"].values > 0).astype(float), s, g)

        print(f"  {boro:<23} {r1['beta']:>7.3f} ({r1['se']:.3f})   {r2['beta']:>7.3f} ({r2['se']:.3f})")

    # ================================================================
    # PART 4: ROBUSTNESS — DIFFERENT FE STRUCTURES
    # ================================================================
    print()
    print("=" * 70)
    print("PART 4: ROBUSTNESS — PERMIT FILING (90d) ACROSS SPECIFICATIONS")
    print("=" * 70)
    print()

    y = df["any_permit_90d"].values.astype(float)

    specs = [
        ("No controls", np.ones(len(df), dtype=str)),
        ("Category FEs", df["category_description"].fillna("UNK").values),
        ("Category × Unit", df["cat_unit"].values),
        ("Category × Unit × YM", df["cat_unit_ym"].values),
        ("+ Borough", (df["cat_unit_ym"].astype(str) + "|" + df["borough"].fillna("").astype(str)).values),
        ("+ NTA (neighborhood)",
         (df["cat_unit_ym"].astype(str) + "|" + df["nta"].fillna("UNK").astype(str)).values),
    ]

    print(f"  {'Specification':<40} {'β':>8} {'SE':>8} {'t':>8} {'FE groups':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    for label, groups in specs:
        r = within_regression(y, strictness, groups, label)
        print(f"  {label:<40} {r['beta']:>8.4f} {r['se']:>8.4f} {r['t']:>8.2f} {r['n_groups']:>12,}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
