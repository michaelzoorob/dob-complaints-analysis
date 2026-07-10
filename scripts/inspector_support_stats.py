"""Supporting statistics quoted in the inspector post that previously lived
only on script stdout: the variance decomposition, closure-code share,
strictness percentiles, the strictness/no-access correlation, base rates for
the downstream outcomes under both conventions, NTA/tract coverage, follow-up
censoring, and the IV division behind "about +3 pp". Report lengths and
marker-word shares live in strictness_text_figure.py (strictness_support_stats.csv),
which owns the balanced-pool construction.

LOO-variant first stages (fresh-verdict, leave-out-building margins) remain in
scripts/audit/critic_A_firststage.py and critic_C_selection.py, which own that
machinery.

Output: data/analysis/risk_models/inspector_support_stats.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from inspector_clustered_estimates import load_panel  # noqa: E402

MARKERS_STRICT = ["OBSERVED", "UNSAFE"]
MARKERS_LENIENT = ["SWO", "RESCIND", "DATED"]
MARKER_MIN_REPORTS = 10  # inspectors with fewer no-violation reports are excluded


def r2_of(y, groups):
    mu = y.mean()
    gm = y.groupby(groups).transform("mean")
    return 1 - ((y - gm) ** 2).sum() / ((y - mu) ** 2).sum()


def main() -> None:
    df = load_panel()
    y = df["violation_found"].astype(float)
    rows = []

    def add(metric, value, se=None, n=None):
        rows.append(dict(metric=metric, value=value, se=se, n=n))
        print(f"{metric:<46} {value}")

    add("panel_rows", len(df))
    add("nta_coverage", round(df["nta"].notna().mean(), 4))
    add("tract_coverage", round(df["bct2020"].notna().mean(), 4))
    sizes = df.groupby("cell_tract")["complaint_number"].transform("size")
    add("tract_singleton_share", round((sizes == 1).mean(), 4))

    # variance decomposition
    r2_cat = r2_of(y, df["cat"])
    r2_unit = r2_of(y, df["assigned_to"].fillna("UNK"))
    r2_insp = r2_of(y, df["inspector_badge"])
    r2_cell4 = r2_of(y, df["cat_unit_ym"])
    add("r2_category", round(r2_cat, 4))
    add("r2_unit", round(r2_unit, 4))
    add("r2_inspector", round(r2_insp, 4))
    add("r2_case_mix_cat_unit_ym", round(r2_cell4, 4))
    d = df[["violation_found", "cat_unit_ym", "inspector_badge", "cat"]].dropna()
    m = pf.feols("violation_found ~ 1 | cat_unit_ym + inspector_badge", data=d)
    r2_joint = 1 - float((m.resid() ** 2).sum()) / ((d.violation_found - d.violation_found.mean()) ** 2).sum()
    add("r2_case_mix_plus_inspector", round(r2_joint, 4))
    add("r2_inspector_increment", round(r2_joint - r2_cell4, 4))
    m2 = pf.feols("violation_found ~ 1 | cat + inspector_badge", data=d)
    r2_ci = 1 - float((m2.resid() ** 2).sum()) / ((d.violation_found - d.violation_found.mean()) ** 2).sum()
    add("r2_inspector_unique_beyond_category", round(r2_ci - r2_cat, 4))
    add("r2_category_unique_beyond_inspector", round(r2_ci - r2_insp, 4))

    # closure codes inside no-violation, strictness percentiles
    prof = pd.read_csv(config.DATA_DIR / "analysis" / "inspector_profiles.csv")
    add("inspectors_30plus", len(prof))
    for q, name in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        add(f"strictness_{name}", round(prof["violation_rate"].quantile(q), 4))

    # DB-side blocks: no-access correlation, report lengths, marker shares
    import sqlite3
    from disposition_codes import classify_disposition
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    db = pd.read_sql_query(
        """SELECT b.inspector_badge, b.comments, o.disposition_code
           FROM bis_scrape b JOIN open_data o USING(complaint_number)
           WHERE o.disposition_code IS NOT NULL AND o.disposition_code != ''""", conn)
    conn.close()
    db["outc"] = db["disposition_code"].map(classify_disposition)
    db = db[db["outc"].isin(["violation", "no_violation", "no_access"])]
    g = db.groupby("inspector_badge").agg(
        n=("outc", "size"),
        n_subst=("outc", lambda s: s.isin(["violation", "no_violation"]).sum()),
        n_viol=("outc", lambda s: (s == "violation").sum()),
        na_rate=("outc", lambda s: (s == "no_access").mean()))
    g = g[g.n >= 30]
    g["viol"] = g.n_viol / g.n_subst  # strictness on the substantive denominator
    add("corr_strictness_no_access", round(g["viol"].corr(g["na_rate"]), 3), n=len(g))

    # downstream base rates, both conventions
    for out, col in [("permit30", "any_permit_30d"), ("permit90", "any_permit_90d"),
                     ("permit365", "any_permit_365d"), ("ecb365", "ecb365")]:
        add(f"base_{out}_all_rows", round(df[col].mean(), 4))
        nta_sizes = df.groupby("cell_nta")["complaint_number"].transform("size")
        add(f"base_{out}_nta_estimation", round(df.loc[nta_sizes >= 2, col].mean(), 4))
        tr_sizes = df.groupby("cell_tract")["complaint_number"].transform("size")
        add(f"base_{out}_tract_estimation", round(df.loc[tr_sizes >= 2, col].mean(), 4))

    # follow-up censoring: share of 2025-26 inspections with <1yr of ECB data
    dates = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce")
    add("share_censored_ecb365",
        round((dates > pd.Timestamp("2025-04-02")).mean(), 4))

    # IV division: reduced form / first stage, delta-method CI (NTA cells)
    est = pd.read_csv(config.DATA_DIR / "analysis" / "risk_models"
                      / "inspector_clustered_estimates.csv")
    fs = est[(est.block == "firststage") & (est.fe == "cell_nta")].iloc[0]
    rf = est[(est.block == "windows") & (est.outcome == "ecb365")
             & (est.label == "365d")].iloc[0]
    iv = rf.b / fs.b
    se_iv = abs(iv) * np.sqrt((rf.se / rf.b) ** 2 + (fs.se / fs.b) ** 2)
    add("iv_viol_to_ecb365_pp", round(iv * 100, 2), se=round(se_iv * 100, 2))

    out_path = config.DATA_DIR / "analysis" / "risk_models" / "inspector_support_stats.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
