"""Discretionary-field inspection volume and violation yield by year.

Backs the post's Local Law 79 passage: volume kept falling after 2022 while
yield per inspection rose sharply in 2025-26, the period the LL79 predictive
program staffed up. Output: data/analysis/risk_models/proactive_yearly_yield.csv
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

ev = pd.read_csv(config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz",
                 usecols=["received_date", "agency", "family", "outcome"])
ev["year"] = ev["received_date"].str[:4]
d = ev[(ev.agency == 1) & (ev.family == "discretionary_field")]
out = d.groupby("year").agg(
    n=("outcome", "size"),
    violation_yield=("outcome", lambda s: round((s == "violation").mean(), 4)))

# agency share of all inspections by year (quoted in the portfolio section)
share = (ev.groupby("year")["agency"].mean().round(4).rename("agency_share"))
out = out.join(share)
dest = config.DATA_DIR / "analysis" / "risk_models" / "proactive_yearly_yield.csv"
out.to_csv(dest)
print(out.to_string(), "\nwrote", dest)
