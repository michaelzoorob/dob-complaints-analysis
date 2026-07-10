#!/usr/bin/env python3
"""Adversarial critic B: permit join, monitoring-per-permit, boom dilution.

Attacks three claims in post_proactive_substack.md:
  (1) "monitoring per 100 active jobs fell 33% 2023->2025; 2026 still 28%
      below 2023" -- is the denominator (active DOB NOW job filings)
      inflated late-window by subsequent-filing (-S) proliferation and
      deflated early by the missing legacy-BIS stock? Are the numerator
      declines program wind-downs rather than capacity? Does the ECB
      "independent" series actually corroborate in the quoted window?
  (2) conversion IRR 1.72 (proactive_monitoring_per_permit.py) -- label
      accuracy of the strict-conversion flag, filing-vs-project unit,
      largest-cost event hoovering, followup/reverse-causation, caller
      placebo.
  (3) the borough-month elasticity spec (n=385).

Everything prints; nothing here edits analysis scripts or CSVs the post
quotes. Run:
  /private/tmp/pyfix_venv/bin/python scripts/audit/critic_proactive_B_permits_dilution.py
"""
import re
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports"
sys.path.insert(0, ROOT + "/scripts")
DB = ROOT + "/data/dob_complaints.db"
SPINE = ROOT + "/data/analysis/proactive"
RM = ROOT + "/data/analysis/risk_models"

WINDOW_START = pd.Timestamp("2020-01-01")
WINDOW_END = pd.Timestamp("2026-05-31")
DAYS_PER_MONTH = 30.4375
MONTHS_IN_YR = {str(y): 12 for y in range(2020, 2026)} | {"2026": 5}

t0 = time.time()


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def sec(s):
    print(f"\n-- {s} --", flush=True)


# ────────────────────────────────────────────────────────────────────────
banner("LOAD SPINE FRAMES")
ev = pd.read_csv(
    SPINE + "/proactive_events.csv.gz",
    usecols=["received_date", "category_prefix", "family", "agency", "bin",
             "bbl", "bct2020", "nta", "active_job_filing_number"],
    dtype={"bbl": str, "bct2020": str, "nta": str, "bin": str,
           "active_job_filing_number": str},
    parse_dates=["received_date"], low_memory=False)
ev["yr"] = ev["received_date"].dt.year.astype(str)

jobs = pd.read_csv(
    SPINE + "/jobs.csv.gz",
    usecols=["job_filing_number", "job_type", "building_type", "initial_cost",
             "total_construction_floor_area", "existing_dwelling_units",
             "proposed_dwelling_units", "bbl", "bct2020", "bldgclass",
             "first_permit_date", "signoff_date", "active_start",
             "active_end", "conversion", "conversion_ge10",
             "conversion_office"],
    dtype={"bbl": str, "bct2020": str},
    parse_dates=["first_permit_date", "signoff_date", "active_start",
                 "active_end"])
jobs["suffix"] = jobs["job_filing_number"].str.extract(r"-([A-Z])\d+$")
jobs["base"] = jobs["job_filing_number"].str.replace(r"-[A-Z]\d+$", "",
                                                     regex=True)
print(f"events {len(ev):,}; jobs {len(jobs):,} "
      f"(suffix mix {jobs['suffix'].value_counts().to_dict()})")

# ────────────────────────────────────────────────────────────────────────
banner("ATTACK 1a: NUMERATOR, DENOMINATOR-FREE (raw discretionary counts)")
disc = ev[(ev["agency"] == 1) & (ev["family"] == "discretionary_field")]
cnt = disc.pivot_table(index="category_prefix", columns="yr",
                       values="family", aggfunc="size").fillna(0).astype(int)
cnt = cnt[[c for c in cnt.columns if "2019" < c < "2027"]]
cnt.loc["TOTAL"] = cnt.sum()
print(cnt.to_string())

CORE = ["8A", "1X", "91"]  # construction-site compliance / EWO / workers
SWEEPY = ["7G", "6X", "1Y", "2Y", "5G"]  # sweeps, watchlist, EWO-lump, padlk
core = cnt.loc[CORE].sum()
sweepy = cnt.loc[SWEEPY].sum()
sec("core (8A+1X+91) vs sweep/watchlist programs, monthly rates")
for label, s in [("core", core), ("sweep/watchlist", sweepy),
                 ("TOTAL", cnt.loc["TOTAL"])]:
    m23, m25 = s["2023"] / 12, s["2025"] / 12
    m26 = s["2026"] / 5
    print(f"  {label:>16}: 2023 {m23:7.1f}/mo  2025 {m25:7.1f}/mo "
          f"({(m25 / m23 - 1) * 100:+5.1f}%)  2026 {m26:7.1f}/mo "
          f"({(m26 / m23 - 1) * 100:+5.1f}% vs 2023)")
print("  8A alone: 2023 %.0f -> 2025 %.0f (%+.1f%%)" % (
    cnt.loc["8A", "2023"], cnt.loc["8A", "2025"],
    (cnt.loc["8A", "2025"] / cnt.loc["8A", "2023"] - 1) * 100))

sec("tract-match drift (panel numerator drops unmatched events)")
g = disc.groupby("yr").agg(n=("family", "size"),
                           tract=("bct2020", lambda s: s.notna().mean()))
g["in_panel"] = (g["n"] * g["tract"]).round(0).astype(int)
print(g.round(4).to_string())

sec("per-borough raw discretionary change 2023->2025 (bbl digit)")
disc_b = disc[disc["bbl"].fillna("").str[0].isin(list("12345"))]
bb = disc_b.pivot_table(index=disc_b["bbl"].str[0], columns="yr",
                        values="family", aggfunc="size").fillna(0)
bb["chg_23_25"] = (bb["2025"] / bb["2023"] - 1) * 100
print(bb[["2023", "2024", "2025", "chg_23_25"]].round(1).to_string())

sec("scrape completeness: bis_scrape vs open_data by received year")
conn = sqlite3.connect(DB)
cov = pd.read_sql_query("""
    SELECT substr(o.date_entered, 7, 4) AS yr, COUNT(*) n_open,
           SUM(CASE WHEN b.complaint_number IS NOT NULL THEN 1 ELSE 0 END)
               AS n_scraped
    FROM open_data o LEFT JOIN bis_scrape b
      ON o.complaint_number = b.complaint_number
    WHERE substr(o.date_entered, 7, 4) BETWEEN '2020' AND '2026'
    GROUP BY 1""", conn)
cov["share"] = (cov["n_scraped"] / cov["n_open"]).round(4)
print(cov.to_string(index=False))
cov26 = pd.read_sql_query("""
    SELECT substr(o.date_entered, 1, 2) AS mo, COUNT(*) n_open,
           SUM(CASE WHEN b.complaint_number IS NOT NULL THEN 1 ELSE 0 END)
               AS n_scraped
    FROM open_data o LEFT JOIN bis_scrape b
      ON o.complaint_number = b.complaint_number
    WHERE substr(o.date_entered, 7, 4) = '2026'
    GROUP BY 1""", conn)
cov26["share"] = (cov26["n_scraped"] / cov26["n_open"]).round(3)
print("2026 by month (window ends May; unscraped rows are June-July,"
      " OUTSIDE the window -> numerator NOT undercounted):")
print(cov26.to_string(index=False))

sec("agency share inside discretionary prefixes by year (ref_311 drift?)")
allf = ev[ev["family"] == "discretionary_field"]
ash = allf.pivot_table(index="yr", values="agency", aggfunc="mean").round(4)
print(ash.to_string())

# ────────────────────────────────────────────────────────────────────────
banner("ATTACK 1b: DENOMINATOR — filings vs projects vs legacy overhang")

months = pd.period_range("2020-01", "2026-05", freq="M")


def monthly_stock(df):
    s = df["active_start"].dt.to_period("M")
    e = df["active_end"].dt.to_period("M")
    ok = s.notna() & e.notna() & (e >= s)
    s, e = s[ok], e[ok]
    lo = np.maximum(s.astype(int).to_numpy(), months[0].ordinal)
    hi = np.minimum(e.astype(int).to_numpy(), months[-1].ordinal)
    keep = hi >= lo
    lo, hi = lo[keep], hi[keep]
    lens = hi - lo + 1
    idx = np.repeat(lo - months[0].ordinal, lens) + (
        np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    out = pd.Series(np.bincount(idx, minlength=len(months)),
                    index=months.astype(str))
    return out


# base-level collapse: one row per project, union span
base = jobs.groupby("base").agg(
    active_start=("active_start", "min"), active_end=("active_end", "max"),
    initial_cost=("initial_cost", "max"), n_filings=("base", "size"))
base = base.reset_index()

stock_all = monthly_stock(jobs)
stock_i = monthly_stock(jobs[jobs["suffix"] == "I"])
stock_base = monthly_stock(base)

sec("avg monthly active stock by year under three definitions")
tab = pd.DataFrame({"all_filings(published)": stock_all,
                    "I_filings_only": stock_i,
                    "base_projects(union span)": stock_base})
tab["yr"] = tab.index.str[:4]
ytab = tab.groupby("yr").mean().round(0).astype(int)
for c in ytab.columns:
    ytab[c + "_chg23"] = ((ytab[c] / ytab.loc["2023", c] - 1) * 100).round(1)
print(ytab.to_string())

sec("monitoring per 100 active jobs under each denominator (raw numerator)")
num = disc.groupby("yr").size()
for label, st in [("published_all_filings", stock_all),
                  ("I_filings_only", stock_i),
                  ("base_projects", stock_base)]:
    r = {}
    for y in ["2023", "2024", "2025", "2026"]:
        den = st[st.index.str[:4] == y].mean()
        r[y] = num[y] / MONTHS_IN_YR[y] / den * 100
    print(f"  {label:>24}: " + "  ".join(f"{y} {v:.2f}" for y, v in r.items())
          + f"  | 2025 vs 2023 {(r['2025'] / r['2023'] - 1) * 100:+.1f}%"
          + f"  2026 vs 2023 {(r['2026'] / r['2023'] - 1) * 100:+.1f}%")

sec("what the extra filings are: initial permits on -S filings by work type")
wt = pd.read_sql_query("""
    SELECT substr(issued_date,1,4) yr, work_type, COUNT(*) n
    FROM permits
    WHERE issued_date >= '2020-01-01' AND filing_reason = 'Initial Permit'
      AND job_filing_number NOT LIKE '%-I%'
    GROUP BY 1, 2""", conn)
wtp = wt.pivot_table(index="work_type", columns="yr", values="n",
                     aggfunc="sum").fillna(0).astype(int)
wtp["tot"] = wtp.sum(axis=1)
print(wtp.sort_values("tot", ascending=False).head(8).to_string())

sec("distinct NEW projects per year (flow, I filings first permitted)")
fl = jobs[jobs["suffix"] == "I"].groupby(
    jobs["first_permit_date"].dt.year).size()
print(fl.loc[2020:2026].to_string())

sec("expired-date tail check (typo-driven zombie jobs?)")
far = (jobs["active_end"] >= "2027-06-01").sum()
print(f"  jobs with active_end >= 2027-06: {far:,} of {len(jobs):,} "
      f"(>=2028: {(jobs['active_end'] >= '2028-01-01').sum()}) -> "
      "far-future expiry typos negligible")

sec("compounded correction: raw numerator + base-project denominator "
    "+ legacy overhang")
for L23 in [0, 10_000, 15_000]:
    L25, L26 = 0.25 * L23, 0.12 * L23
    b23 = stock_base[stock_base.index.str[:4] == "2023"].mean() + L23
    b25 = stock_base[stock_base.index.str[:4] == "2025"].mean() + L25
    b26 = stock_base[stock_base.index.str[:4] == "2026"].mean() + L26
    r23, r25, r26 = (num["2023"] / 12 / b23 * 100,
                     num["2025"] / 12 / b25 * 100,
                     num["2026"] / 5 / b26 * 100)
    print(f"  L23={L23:>6,}: 2023 {r23:.2f}  2025 {r25:.2f} "
          f"({(r25 / r23 - 1) * 100:+.1f}%)  2026 {r26:.2f} "
          f"({(r26 / r23 - 1) * 100:+.1f}%)   [published: -33.2% / -28.8%]")

sec("legacy-BIS overhang sensitivity (jobs filed in BIS, invisible here)")
# survival of DOB NOW cohorts as proxy for BIS cohort survival
j20 = jobs[(jobs["suffix"] == "I")
           & (jobs["first_permit_date"] < "2021-03-01")]
for probe in ["2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"]:
    alive = (j20["active_end"] >= probe).mean()
    print(f"  DOB NOW pre-2021-03 cohort ({len(j20):,} jobs) still active "
          f"{probe[:7]}: {alive:.3f}")
print("  BIS accepted NB/Alt filings until 2021-03; its 2019-2021 cohorts"
      " are missing from every denominator above.")
for L23 in [5_000, 10_000, 15_000, 20_000]:
    L25, L26 = 0.25 * L23, 0.12 * L23
    r23 = num["2023"] / 12 / (stock_all[stock_all.index.str[:4] == "2023"].mean() + L23) * 100
    r25 = num["2025"] / 12 / (stock_all[stock_all.index.str[:4] == "2025"].mean() + L25) * 100
    r26 = num["2026"] / 5 / (stock_all[stock_all.index.str[:4] == "2026"].mean() + L26) * 100
    print(f"  L23={L23:,}: per100 2023 {r23:.2f} 2025 {r25:.2f} "
          f"({(r25 / r23 - 1) * 100:+.1f}%) 2026 {r26:.2f} "
          f"({(r26 / r23 - 1) * 100:+.1f}%)")

# ────────────────────────────────────────────────────────────────────────
banner("ATTACK 1c: ECB 'independent corroboration'")
ecb = pd.read_sql_query("""
    SELECT substr(issue_date,1,4) AS yr, violation_type, COUNT(*) n
    FROM ecb_violations
    WHERE violation_type IN ('Site Safety','Cranes and Derricks',
                             'Construction')
      AND issue_date >= '20150101'
    GROUP BY 1, 2""", conn)
ep = ecb.pivot_table(index="yr", columns="violation_type", values="n",
                     aggfunc="sum").fillna(0).astype(int)
print(ep.to_string())
for c in ep.columns:
    ch = (ep.loc["2025", c] / ep.loc["2023", c] - 1) * 100
    print(f"  {c}: 2023->2025 {ch:+.1f}%  (2019 level {ep.loc['2019', c]:,}; "
          f"post quotes 2020->2024 for Site Safety = "
          f"{(ep.loc['2024', c] / ep.loc['2020', c] - 1) * 100:+.1f}%)")
conn.close()

# ────────────────────────────────────────────────────────────────────────
banner("ATTACK 6: are 'the right construction sites' (D10 cost) less "
       "inspected over time? (title check)")
# assign discretionary+followup events to base projects, largest cost at BBL
jobs_v = jobs[jobs["active_start"].notna() & jobs["active_end"].notna()].copy()
jobs_v["span_start"] = jobs_v["active_start"].clip(lower=WINDOW_START)
jobs_v["span_end"] = jobs_v["active_end"].clip(upper=WINDOW_END)
jobs_v = jobs_v[jobs_v["span_end"] >= jobs_v["span_start"]]

base_v = jobs_v.groupby("base").agg(
    bbl=("bbl", "first"), bct2020=("bct2020", "first"),
    span_start=("span_start", "min"), span_end=("span_end", "max"),
    initial_cost=("initial_cost", "max"),
    cost_sum=("initial_cost", "sum"),
    floor=("total_construction_floor_area", "max"),
    conversion=("conversion", "max"),
    conversion_ge10=("conversion_ge10", "max"),
    conversion_office=("conversion_office", "max"),
    n_filings=("base", "size"),
    first_permit_date=("first_permit_date", "min")).reset_index()
# job type of the largest-cost filing
jt = (jobs_v.sort_values("initial_cost", ascending=False, kind="stable")
      .drop_duplicates("base")[["base", "job_type"]])
base_v = base_v.merge(jt, on="base")
base_v["months"] = ((base_v["span_end"] - base_v["span_start"]).dt.days
                    + 1) / DAYS_PER_MONTH

dset = ev[(ev["agency"] == 1)
          & ev["family"].isin(["discretionary_field", "followup"])].copy()
dset = dset.reset_index(drop=True)
dset["eid"] = dset.index
dset["link_base"] = dset["active_job_filing_number"].str.replace(
    r"-[A-Z]\d+$", "", regex=True)

spanb = base_v[["base", "bbl", "span_start", "span_end", "initial_cost"]]
d1 = dset.merge(spanb[["base", "span_start", "span_end"]],
                left_on="link_base", right_on="base", how="inner")
d1 = d1[d1["received_date"].between(d1["span_start"], d1["span_end"])]
d1 = d1[["eid", "base", "received_date", "category_prefix"]]
rest = dset[~dset["eid"].isin(d1["eid"])]
f1 = rest.merge(spanb, on="bbl", how="inner")
f1 = f1[f1["received_date"].between(f1["span_start"], f1["span_end"])]
f1 = (f1.sort_values("initial_cost", ascending=False, kind="stable")
      .drop_duplicates("eid"))[["eid", "base", "received_date",
                                "category_prefix"]]
asg = pd.concat([d1, f1], ignore_index=True)
print(f"base-level assignment: {len(asg):,} of {len(dset):,} events "
      f"({len(asg) / len(dset):.1%}; direct {len(d1):,} fallback {len(f1):,})")

base_v["decile"] = pd.qcut(base_v["initial_cost"].rank(method="first"), 10,
                           labels=False) + 1
asg = asg.merge(base_v[["base", "decile"]], on="base")
asg["yr"] = asg["received_date"].dt.year.astype(str)

sec("events per 100 project-months, by cost decile x year")
rows = []
for y in ["2022", "2023", "2024", "2025", "2026"]:
    y0 = pd.Timestamp(f"{y}-01-01")
    y1 = min(pd.Timestamp(f"{y}-12-31"), WINDOW_END)
    ov_lo = base_v["span_start"].clip(lower=y0)
    ov_hi = base_v["span_end"].clip(upper=y1)
    mo = ((ov_hi - ov_lo).dt.days + 1).clip(lower=0) / DAYS_PER_MONTH
    for dec, lab in [(10, "D10"), (None, "all")]:
        mask = (base_v["decile"] == dec) if dec else pd.Series(
            True, index=base_v.index)
        em = asg["yr"] == y
        if dec:
            em = em & (asg["decile"] == dec)
        n_ev = em.sum()
        n_ev8a = (em & (asg["category_prefix"] == "8A")).sum()
        jm = mo[mask].sum()
        rows.append({"yr": y, "slice": lab,
                     "rate_per100": 100 * n_ev / jm,
                     "rate8A_per100": 100 * n_ev8a / jm,
                     "job_months": int(jm)})
d10 = pd.DataFrame(rows)
print(d10.pivot(index="yr", columns="slice",
                values="rate_per100").round(2).to_string())
print("\n8A compliance only:")
print(d10.pivot(index="yr", columns="slice",
                values="rate8A_per100").round(2).to_string())

# ────────────────────────────────────────────────────────────────────────
banner("ATTACK 2: monitoring-per-permit IRR 1.72")

sec("2a conversion_ge10 composition (label accuracy)")
c630 = jobs[jobs["conversion_ge10"] == 1]
cls = c630["bldgclass"].fillna("?").astype(str).str[0].value_counts()
print("PLUTO class (CURRENT vintage = post-conversion for finished jobs):")
print("  " + ", ".join(f"{k}:{v}" for k, v in cls.items()))
print(f"  office O: {cls.get('O', 0)} ({cls.get('O', 0) / len(c630):.1%}); "
      f"K stores: {cls.get('K', 0)}; D/C/R residential (post-hoc?): "
      f"{cls.get('D', 0) + cls.get('C', 0) + cls.get('R', 0)}; "
      f"E/F warehouse+factory: {cls.get('E', 0) + cls.get('F', 0)}; "
      f"M church: {cls.get('M', 0)}; G garage: {cls.get('G', 0)}; "
      f"H hotel: {cls.get('H', 0)}")
conn = sqlite3.connect(DB)
desc = pd.read_sql_query("""
    SELECT job_filing_number, job_description FROM permits
    WHERE job_description IS NOT NULL AND job_description != ''""", conn)
conn.close()
desc = desc.drop_duplicates("job_filing_number")
c630 = c630.merge(desc, on="job_filing_number", how="left")
# also try sibling filings of the same base for a description
have = c630["job_description"].notna()
sib = desc.copy()
sib["base"] = sib["job_filing_number"].str.replace(r"-[A-Z]\d+$", "",
                                                   regex=True)
sib = sib.drop_duplicates("base")[["base", "job_description"]]
c630 = c630.merge(sib, on="base", how="left", suffixes=("", "_sib"))
c630["desc"] = c630["job_description"].fillna(c630["job_description_sib"])
dd = c630["desc"].fillna("").str.upper()
print(f"description coverage: {(dd != '').mean():.1%}")
for kw in ["OFFICE", "HOTEL", "DORMITOR", "WAREHOUSE", "FACTORY", "CHURCH",
           "SCHOOL", "GARAGE", "COMMERCIAL", "CONVER"]:
    print(f"  mentions {kw}: {dd.str.contains(kw).sum()}")
print("sample non-office-class descriptions:")
for s in c630.loc[~c630["bldgclass"].fillna("").str[0].isin(["O"]),
                  "desc"].dropna().head(8):
    print("   -", str(s)[:140])
print(f"\nstrict conversions that are -S/-Z filings (project already "
      f"counted under its -I1 sibling): "
      f"{(c630['suffix'] != 'I').sum()} of {len(c630)}")

sec("2b: PPML refits (filing-level published spec, then base-level + "
    "assignment variants)")
try:
    import pyfixest as pf

    def fit(frame, outcome, flag, label, offset="log_months"):
        x = ("log_cost + log_floor + floor_missing + jt_new_building + "
             "jt_alteration_co + jt_altco_existing + jt_full_demolition")
        m = pf.fepois(f"{outcome} ~ {flag} + {x} | nta^permit_q",
                      data=frame, offset=offset, vcov={"CRV1": "bbl"})
        td = m.tidy().reset_index()
        r = td[td["Coefficient"] == flag].iloc[0]
        print(f"  {label:>44}: IRR {np.exp(r['Estimate']):.2f} "
              f"[{np.exp(r['2.5%']):.2f}, {np.exp(r['97.5%']):.2f}] "
              f"n={m._N:,}")
        return np.exp(r["Estimate"])

    xw = (ev.dropna(subset=["bct2020", "nta"])
          .groupby("bct2020")["nta"].agg(lambda s: s.mode().iat[0]))

    def prep(frame, cost_col="initial_cost"):
        f = frame.copy()
        f["months"] = ((f["span_end"] - f["span_start"]).dt.days
                       + 1) / DAYS_PER_MONTH
        f["nta"] = f["bct2020"].map(xw)
        f["permit_q"] = (f["first_permit_date"].dt.to_period("Q").astype(str))
        f["log_cost"] = np.log1p(f[cost_col].clip(lower=0))
        fl = f["floor"] if "floor" in f else f["total_construction_floor_area"]
        f["floor_missing"] = fl.isna().astype("int8")
        f["log_floor"] = np.log1p(fl.fillna(0).clip(lower=0))
        f["log_months"] = np.log(f["months"])
        for col, val in [
                ("jt_new_building", "New Building"),
                ("jt_alteration_co", "Alteration CO"),
                ("jt_altco_existing",
                 "ALT-CO - New Building with Existing Elements to Remain"),
                ("jt_full_demolition", "Full Demolition")]:
            f[col] = (f["job_type"] == val).astype("int8")
        return f[f["nta"].notna()].reset_index(drop=True)

    # replicate the published filing-level frame + assignment
    span_f = jobs_v[["job_filing_number", "bbl", "span_start", "span_end",
                     "initial_cost"]]
    df_ = dset.merge(span_f[["job_filing_number", "span_start", "span_end"]],
                     left_on="active_job_filing_number",
                     right_on="job_filing_number", how="inner")
    df_ = df_[df_["received_date"].between(df_["span_start"],
                                           df_["span_end"])]
    df_ = df_[["eid", "job_filing_number"]]
    rest_f = dset[~dset["eid"].isin(df_["eid"])]
    ff = rest_f.merge(span_f, on="bbl", how="inner")
    ff = ff[ff["received_date"].between(ff["span_start"], ff["span_end"])]
    ff_max = (ff.sort_values("initial_cost", ascending=False, kind="stable")
              .drop_duplicates("eid"))[["eid", "job_filing_number"]]
    ff_min = (ff.sort_values("initial_cost", ascending=True, kind="stable")
              .drop_duplicates("eid"))[["eid", "job_filing_number"]]

    filing = prep(jobs_v.rename(columns={
        "total_construction_floor_area": "floor"}))
    cmax = (pd.concat([df_, ff_max]).groupby("job_filing_number").size())
    filing["n_disc"] = filing["job_filing_number"].map(cmax).fillna(0)
    filing["n_disc"] = filing["n_disc"].astype(int)
    irr_pub = fit(filing, "n_disc", "conversion_ge10",
                  "published replication (filing level, max-cost)")

    filing["n_direct"] = (filing["job_filing_number"]
                          .map(df_.groupby("job_filing_number").size())
                          .fillna(0).astype(int))
    fit(filing, "n_direct", "conversion_ge10",
        "direct-link events only (no BBL fallback)")

    cmin = (pd.concat([df_, ff_min]).groupby("job_filing_number").size())
    filing["n_min"] = (filing["job_filing_number"].map(cmin)
                       .fillna(0).astype(int))
    fit(filing, "n_min", "conversion_ge10",
        "fallback ties to SMALLEST-cost filing")

    # base-project level: events from the base assignment above
    basef = prep(base_v, cost_col="initial_cost")
    nb = asg.groupby("base").size()
    basef["n_disc"] = basef["base"].map(nb).fillna(0).astype(int)
    fit(basef, "n_disc", "conversion_ge10",
        "BASE-PROJECT level (union span, any-filing flag)")

    # single-active-project BBLs only (no hoovering possible)
    multi = base_v.groupby("bbl")["base"].size()
    solo = basef[basef["bbl"].map(multi) == 1]
    print(f"    [solo subsample: {len(solo):,} projects, "
          f"{int(solo['conversion_ge10'].sum())} strict conversions, "
          f"{int(solo.loc[solo['conversion_ge10'] == 1, 'n_disc'].sum())} "
          f"events on them; multi-project BBL share of conversions "
          f"{1 - basef.loc[basef['conversion_ge10'] == 1, 'bbl'].map(multi).eq(1).mean():.1%}"
          f" -> solo estimate is UNINFORMATIVE, do not quote]")
    fit(solo, "n_disc", "conversion_ge10",
        "single-project BBLs only (base level)")

    # exclude followup family (reverse-causation channel)
    asg_fam = asg.merge(dset[["eid", "family"]], on="eid")
    ndf = asg_fam[asg_fam["family"] == "discretionary_field"].groupby(
        "base").size()
    basef["n_disc_only"] = basef["base"].map(ndf).fillna(0).astype(int)
    fit(basef, "n_disc_only", "conversion_ge10",
        "discretionary_field only (followup excluded)")

    # caller placebo: caller-originated construction-incident complaints
    caller = ev[(ev["agency"] == 0)
                & (ev["family"] == "mixed_incident")].reset_index(drop=True)
    caller["eid"] = caller.index
    caller["link_base"] = caller["active_job_filing_number"].str.replace(
        r"-[A-Z]\d+$", "", regex=True)
    c1 = caller.merge(spanb[["base", "span_start", "span_end"]],
                      left_on="link_base", right_on="base", how="inner")
    c1 = c1[c1["received_date"].between(c1["span_start"], c1["span_end"])]
    c1 = c1[["eid", "base"]]
    crest = caller[~caller["eid"].isin(c1["eid"])]
    c2 = crest.merge(spanb, on="bbl", how="inner")
    c2 = c2[c2["received_date"].between(c2["span_start"], c2["span_end"])]
    c2 = (c2.sort_values("initial_cost", ascending=False, kind="stable")
          .drop_duplicates("eid"))[["eid", "base"]]
    ncall = pd.concat([c1, c2]).groupby("base").size()
    basef["n_caller"] = basef["base"].map(ncall).fillna(0).astype(int)
    fit(basef, "n_caller", "conversion_ge10",
        "PLACEBO: caller incident complaints (same assignment)")

    # caller placebo at the published filing level too
    cf1 = caller.merge(span_f[["job_filing_number", "span_start",
                               "span_end"]],
                       left_on="active_job_filing_number",
                       right_on="job_filing_number", how="inner")
    cf1 = cf1[cf1["received_date"].between(cf1["span_start"],
                                           cf1["span_end"])]
    cf1 = cf1[["eid", "job_filing_number"]]
    cfr = caller[~caller["eid"].isin(cf1["eid"])]
    cf2 = cfr.merge(span_f, on="bbl", how="inner")
    cf2 = cf2[cf2["received_date"].between(cf2["span_start"],
                                           cf2["span_end"])]
    cf2 = (cf2.sort_values("initial_cost", ascending=False, kind="stable")
           .drop_duplicates("eid"))[["eid", "job_filing_number"]]
    ncf = pd.concat([cf1, cf2]).groupby("job_filing_number").size()
    filing["n_caller"] = (filing["job_filing_number"].map(ncf)
                          .fillna(0).astype(int))
    fit(filing, "n_caller", "conversion_ge10",
        "PLACEBO at published filing level")

    # all-caller placebo (any family, not just incidents)
    caller_all = ev[ev["agency"] == 0].reset_index(drop=True)
    caller_all["eid"] = caller_all.index
    ca = caller_all.merge(span_f, on="bbl", how="inner")
    ca = ca[ca["received_date"].between(ca["span_start"], ca["span_end"])]
    ca = (ca.sort_values("initial_cost", ascending=False, kind="stable")
          .drop_duplicates("eid"))[["eid", "job_filing_number"]]
    nca = ca.groupby("job_filing_number").size()
    filing["n_caller_all"] = (filing["job_filing_number"].map(nca)
                              .fillna(0).astype(int))
    fit(filing, "n_caller_all", "conversion_ge10",
        "PLACEBO: ALL caller complaints (filing level)")

except Exception as e:  # pragma: no cover
    print(f"PPML section failed: {e}")

# ────────────────────────────────────────────────────────────────────────
banner("ATTACK 3: elasticity spec (borough-month, n=385)")
est = pd.read_csv(RM + "/proactive_dilution_estimates.csv")
el = est[est["term"] == "log_jobs"][
    ["model", "estimate", "ci_low", "ci_high", "p_value", "n"]]
print("published specs (log_jobs coefficient):")
print(el.round(3).to_string(index=False))

try:
    import pyfixest as pf
    tm = pd.read_csv(SPINE + "/tract_month.csv.gz", dtype={"bct2020": str})
    tm["boro"] = tm["bct2020"].str[0]
    bm = (tm.groupby(["boro", "month"], as_index=False)
          .agg(monitoring=("proactive_discretionary_field", "sum")))
    # borough denominators under I-only and base definitions
    for lab, jset in [("I_only", jobs_v[jobs_v["suffix"] == "I"]),
                      ("base_projects", base_v.assign(
                          active_start=base_v["span_start"],
                          active_end=base_v["span_end"]))]:
        jj = jset[jset["bbl"].fillna("").str[0].isin(list("12345"))].copy()
        rowsb = []
        for b, gdf in jj.groupby(jj["bbl"].str[0]):
            st = monthly_stock(gdf)
            rowsb.append(pd.DataFrame({"boro": b, "month": st.index,
                                       "jobs": st.values}))
        stx = pd.concat(rowsb)
        d = bm.merge(stx, on=["boro", "month"])
        d = d[d["jobs"] > 0]
        d["log_jobs"] = np.log(d["jobs"])
        mi = (d["month"].str[:4].astype(int) * 12
              + d["month"].str[5:].astype(int))
        d["month_t"] = mi - mi.min()
        for wlab, dd_ in [("full", d), ("2023on", d[d["month"] >= "2023-01"])]:
            m = pf.fepois("monitoring ~ log_jobs + month_t | boro",
                          data=dd_, vcov={"CRV1": "month"})
            td = m.tidy().reset_index()
            r = td[td["Coefficient"] == "log_jobs"].iloc[0]
            print(f"  {lab:>14} {wlab:>7}: elasticity {r['Estimate']:+.3f} "
                  f"[{r['2.5%']:+.3f}, {r['97.5%']:+.3f}] (n={m._N})")
except Exception as e:  # pragma: no cover
    print(f"elasticity refit failed: {e}")

print(f"\nTOTAL {time.time() - t0:.0f}s")
