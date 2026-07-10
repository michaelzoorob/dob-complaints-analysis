"""ECB vs DOB violation datasets: where the fine dollars live.

Supports the overview post's claim that ECB/OATH is the penalty docket.
The DOB violations dataset (3h2n-5cm9) carries no penalty column at all;
ECB violations (6bgk-3dad) carry penalty_imposed / amount_paid. This script
totals ECB penalties over the analysis window and writes the summary CSV.

Output: data/analysis/risk_models/ecb_penalty_stats.csv
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

WINDOW_START = "20200101"
WINDOW_END = "20260531"  # through May 2026, matching the post window


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)

    dob_cols = [r[1] for r in con.execute("PRAGMA table_info(dob_violations)")]
    assert not any("penal" in c or "fine" in c or "amount" in c for c in dob_cols), (
        "dob_violations unexpectedly gained a penalty column; revisit the claim"
    )

    row = con.execute(
        """
        SELECT COUNT(*)                                                   AS n_citations,
               SUM(CASE WHEN CAST(penality_imposed AS REAL) > 0
                        THEN 1 ELSE 0 END)                                AS n_with_penalty,
               SUM(CAST(penality_imposed AS REAL))                        AS penalties_imposed,
               SUM(CAST(amount_paid AS REAL))                             AS penalties_paid
        FROM ecb_violations
        WHERE issue_date >= ? AND issue_date <= ?
        """,
        (WINDOW_START, WINDOW_END),
    ).fetchone()
    n, n_pos, imposed, paid = row

    median_pos = con.execute(
        """
        SELECT AVG(p) FROM (
            SELECT CAST(penality_imposed AS REAL) AS p
            FROM ecb_violations
            WHERE issue_date >= ? AND issue_date <= ?
              AND CAST(penality_imposed AS REAL) > 0
            ORDER BY p
            LIMIT 2 - (SELECT COUNT(*) FROM ecb_violations
                       WHERE issue_date >= ? AND issue_date <= ?
                         AND CAST(penality_imposed AS REAL) > 0) % 2
            OFFSET (SELECT (COUNT(*) - 1) / 2 FROM ecb_violations
                    WHERE issue_date >= ? AND issue_date <= ?
                      AND CAST(penality_imposed AS REAL) > 0)
        )
        """,
        (WINDOW_START, WINDOW_END) * 3,
    ).fetchone()[0]

    out = pd.DataFrame(
        [
            dict(
                window=f"{WINDOW_START}-{WINDOW_END}",
                n_ecb_citations=n,
                n_with_positive_penalty=n_pos,
                share_with_positive_penalty=n_pos / n,
                penalties_imposed_usd=imposed,
                penalties_paid_usd=paid,
                median_positive_penalty_usd=median_pos,
                dob_violations_penalty_columns=0,
            )
        ]
    )
    dest = config.DATA_DIR / "analysis" / "risk_models" / "ecb_penalty_stats.csv"
    out.to_csv(dest, index=False)
    print(out.T.to_string())
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
