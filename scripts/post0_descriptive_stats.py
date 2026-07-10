"""Descriptive statistics quoted in the overview post that previously had no
script writing them to a CSV: the window complaint count, the ECB-linked share,
the caller/agency origin split, borough per-lot rates, tract percentiles,
after-hours outcome shares and timing, inspector and building counts, and the
database snapshot sizes.

Output: data/analysis/risk_models/post0_descriptive_stats.csv
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
import disposition_codes as dc  # noqa: E402

WINDOW = ("2020-01-01", "2026-05-31")


def main() -> None:
    con = sqlite3.connect(str(config.DB_PATH))
    rows = []

    def add(metric, value):
        rows.append(dict(metric=metric, value=value))
        print(f"{metric:<44} {value}")

    # scrape universe and window-filed count (open_data dates are MM/DD/YYYY)
    add("scraped_pages", con.execute("SELECT COUNT(*) FROM bis_scrape").fetchone()[0])
    q = """
    SELECT COUNT(*) FROM open_data
    WHERE substr(date_entered,7,4) || '-' || substr(date_entered,1,2) || '-' || substr(date_entered,4,2)
          BETWEEN ? AND ?
    """
    add("window_filed_complaints", con.execute(q, WINDOW).fetchone()[0])

    # scraped-page fill rates, ECB link, badges, buildings
    b = pd.read_sql_query(
        "SELECT comments, subject, ecb_violation, ref_311, inspector_badge, bin"
        " FROM bis_scrape", con)
    add("share_with_inspection_comments",
        round((b["comments"].fillna("").str.strip() != "").mean(), 4))
    add("share_with_complaint_text",
        round((b["subject"].fillna("").str.strip() != "").mean(), 4))
    add("n_with_complaint_text", int((b["subject"].fillna("").str.strip() != "").sum()))
    add("share_ecb_linked", round((b["ecb_violation"].fillna("").str.strip() != "").mean(), 4))
    add("distinct_inspector_badges",
        int(b["inspector_badge"].fillna("").astype(str).str.strip().replace("", np.nan).nunique()))
    add("distinct_bins", con.execute(
        """SELECT COUNT(DISTINCT bin) FROM open_data
           WHERE bin IS NOT NULL AND TRIM(bin) != ''
             AND substr(date_entered,7,4) || '-' || substr(date_entered,1,2)
                 || '-' || substr(date_entered,4,2) BETWEEN ? AND ?""",
        WINDOW).fetchone()[0])
    add("origin_share_311", round((b["ref_311"].fillna("").str.strip() != "").mean(), 4))
    add("origin_share_agency", round((b["ref_311"].fillna("").str.strip() == "").mean(), 4))

    # monthly volume over the window
    dm = pd.read_sql_query(
        """SELECT substr(date_entered,7,4) || '-' || substr(date_entered,1,2) AS ym,
                  COUNT(*) n
           FROM open_data
           WHERE substr(date_entered,7,4) >= '2020'
           GROUP BY ym ORDER BY ym""", con)
    dm = dm[dm.ym <= "2026-05"]
    add("months_in_window", len(dm))
    add("monthly_average", round(dm.n.sum() / len(dm), 1))
    add("daily_average", round(dm.n.sum() / (len(dm) * 30.437), 0))

    # after-hours work (category 04): outcomes and inspection timing
    ah = pd.read_sql_query(
        """SELECT disposition_code, inspection_date, date_entered
           FROM open_data
           WHERE complaint_category = '04'
             AND substr(date_entered,7,4) || '-' || substr(date_entered,1,2)
                 || '-' || substr(date_entered,4,2) BETWEEN ? AND ?""",
        con, params=WINDOW)
    out = ah["disposition_code"].fillna("").map(dc.classify_disposition)
    add("afterhours_n", len(ah))
    add("afterhours_share_noviol", round((out == "no_violation").mean(), 4))
    add("afterhours_share_viol", round((out == "violation").mean(), 4))
    d0 = pd.to_datetime(ah["date_entered"], format="%m/%d/%Y", errors="coerce")
    d1 = pd.to_datetime(ah["inspection_date"], format="%m/%d/%Y", errors="coerce")
    lag = (d1 - d0).dt.days.dropna()
    lag = lag[lag >= 0]
    add("afterhours_share_same_day", round((lag == 0).mean(), 4))
    add("afterhours_median_days_to_inspection", float(lag.median()))

    # borough per-lot rates and tract percentiles from the risk panel
    panel = pd.read_csv(config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz",
                        usecols=["borough", "n_complaints", "unitsres", "bct2020"])
    per_lot = panel.groupby("borough").agg(compl=("n_complaints", "sum"),
                                           lots=("n_complaints", "size"))
    per_lot["per_lot"] = per_lot.compl / per_lot.lots
    for boro, r in per_lot.iterrows():
        add(f"complaints_per_lot_{boro}", round(r.per_lot, 2))
    add("manhattan_vs_queens_per_lot_ratio",
        round(per_lot.loc["MN", "per_lot"] / per_lot.loc["QN", "per_lot"], 2))

    tr = panel.groupby("bct2020").agg(compl=("n_complaints", "sum"),
                                      units=("unitsres", "sum"))
    tr = tr[tr.units >= 200]
    tr["per100"] = tr.compl / tr.units * 100
    add("tracts_with_200plus_units", len(tr))
    add("tract_per100_p90", round(np.percentile(tr.per100, 90), 2))
    add("tract_per100_p10", round(np.percentile(tr.per100, 10), 2))
    add("tract_p90_p10_ratio",
        round(np.percentile(tr.per100, 90) / np.percentile(tr.per100, 10), 2))

    # database snapshot sizes
    gz = config.DATA_DIR / "dob_complaints.db.gz"
    if gz.exists():
        add("db_gz_gib", round(gz.stat().st_size / 2**30, 2))
    add("db_gib", round(Path(config.DB_PATH).stat().st_size / 2**30, 2))

    con.close()
    out_path = config.DATA_DIR / "analysis" / "risk_models" / "post0_descriptive_stats.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
