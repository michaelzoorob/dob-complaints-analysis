"""Headline panel counts and violation-measure correlations for the overview post.

Writes the handful of numbers the posts quote about the panel itself, which
previously existed only as build-time prints: the lot universe, the number of
scraped complaints matching a residential lot, deed-address owner-geography
coverage, and the pairwise correlations among the three violation measures
(disposition violations, ECB citations, DOB violation records 2020 through
May 2026, the deduplicated BIS + DOB NOW union).

Output: data/analysis/risk_models/panel_headline_counts.csv
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
import dob_ledger  # noqa: E402

def main() -> None:
    panel = pd.read_csv(
        config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz",
        usecols=["bbl_key", "n_complaints", "n_viol_disp", "n_ecb_2020on", "owner_geo"],
    )
    panel["bbl_key"] = panel["bbl_key"].astype(str)

    conn = sqlite3.connect(str(config.DB_PATH))
    u = dob_ledger.union_frame(conn, verbose=False)
    conn.close()
    u = u[(u["ymd"] >= "20200101") & (u["ymd"] <= "20260531")]
    u["bbl_key"] = u["bbl_key"].astype(str)
    union_counts = u.groupby("bbl_key").size().rename("n_dobviol_union")
    panel = panel.merge(union_counts, on="bbl_key", how="left")
    panel["n_dobviol_union"] = panel["n_dobviol_union"].fillna(0)

    rows = [
        dict(metric="universe_lots", value=len(panel)),
        dict(metric="matched_complaints_2020_may2026", value=panel["n_complaints"].sum()),
        dict(metric="owner_geo_coverage_share",
             value=(panel["owner_geo"] != "unknown").mean()),
    ]
    pairs = [
        ("n_viol_disp", "n_ecb_2020on"),
        ("n_viol_disp", "n_dobviol_union"),
        ("n_ecb_2020on", "n_dobviol_union"),
    ]
    for a, b in pairs:
        rows.append(dict(metric=f"pearson_{a}_vs_{b}", value=panel[a].corr(panel[b])))
        rows.append(dict(metric=f"spearman_{a}_vs_{b}",
                         value=panel[a].corr(panel[b], method="spearman")))

    out = pd.DataFrame(rows)
    dest = config.DATA_DIR / "analysis" / "risk_models" / "panel_headline_counts.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
