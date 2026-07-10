#!/usr/bin/env python3
"""
Boom dilution (proactive plan hypothesis #6): did discretionary field
monitoring keep pace with the construction pipeline?

Spine: data/analysis/proactive/tract_month.csv.gz (borough = first digit
of bct2020) aggregated to borough x month, 2020-01..2026-05. Monitoring =
proactive_discretionary_field (agency-originated sweeps / compliance /
work-order categories per analysis_config.PROACTIVE_FAMILIES). Exposure =
active_jobs, which since the Critic B fix counts BASE PROJECTS (DOB NOW
filings collapsed on the -I1/-S2/... suffix with union active spans, per
proactive_spine.py) — filing-level denominators drifted upward as -S/-P/-Z
subsequent filings proliferated with DOB NOW adoption, which overstated
the per-100 decline (published -33.2% 2023->2025 became ~-24% corrected).

Program decomposition (Critic B #2): the per-project decline is a sweep
wind-down, not a construction-oversight pullback. The estimates CSV gets a
program_decomposition block: sweep/watchlist prefixes (7G/6X/1Y/2Y/5G) vs
the construction core (8A compliance + 1X emergency work orders) by year,
citywide raw counts; plus 8A events per 100 project-months at
top-cost-decile projects by year (event -> base-project assignment via
active_job_key then largest-cost BBL fallback).

Elasticity models are retained but SUPERSEDED as point estimates (Critic
B #6: spec-fragile, sign-flips in the mature window on corrected
denominators). Do not quote the 0.28; the unit-elasticity rejection
(p_vs_unit_elasticity) is the only surviving use. ECB Site Safety change
rows carry in-window (2023->2025) values alongside the superseded
2020->2024 span, which crosses the era_post break (Critic B #3).

Outputs
  data/analysis/risk_models/proactive_dilution_estimates.csv
  data/analysis/risk_models/proactive_dilution_series.csv
  data/analysis/blog_posts/artifacts/proactive_dilution.png

Run:  /private/tmp/pyfix_venv/bin/python scripts/proactive_boom_dilution.py
"""

import os
import sqlite3
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_config import BOROUGH_CODE_TO_NAME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "dob_complaints.db")
SPINE_DIR = os.path.join(ROOT, "data", "analysis", "proactive")
RISK_DIR = os.path.join(ROOT, "data", "analysis", "risk_models")
ART_DIR = os.path.join(ROOT, "data", "analysis", "blog_posts", "artifacts")

WINDOW = ("2020-01", "2026-05")
WINDOW_START_TS = pd.Timestamp("2020-01-01")
WINDOW_END_TS = pd.Timestamp("2026-05-31")
MATURE_START = "2023-01"  # DOB NOW job stock mature; pre-2023 ramp is
                          # partly system adoption, not real exposure
DAYS_PER_MONTH = 30.4375
MONTHS_IN_YEAR = {str(y): 12 for y in range(2020, 2026)} | {"2026": 5}

# program decomposition prefixes (discretionary_field family)
SWEEP_PREFIXES = ("7G", "6X", "1Y", "2Y", "5G")  # area sweeps, watchlists,
                                                 # EWO-lump, padlock
CORE_PREFIXES = ("8A", "1X")                     # construction compliance +
                                                 # emergency work orders

BORO_NAME = {k: v.title().replace("Staten Island", "Staten Island")
             for k, v in BOROUGH_CODE_TO_NAME.items()}
GEO_ORDER = ["Citywide", "Manhattan", "Bronx", "Brooklyn", "Queens",
             "Staten Island"]

# house style (constants from scripts/make_descriptive_figures.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
TINT = "#f2f1ec"
ZERO_C = "#b9b7ac"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# ── data ─────────────────────────────────────────────────────────────────

def load_borough_month() -> pd.DataFrame:
    """Tract-month spine -> borough x month monitoring + active projects."""
    tm = pd.read_csv(os.path.join(SPINE_DIR, "tract_month.csv.gz"),
                     dtype={"bct2020": str})
    tm["boro"] = tm["bct2020"].str[0]
    bm = (tm.groupby(["boro", "month"], as_index=False)
            .agg(monitoring=("proactive_discretionary_field", "sum"),
                 active_jobs=("active_jobs", "sum")))
    bm = bm[bm["month"].between(*WINDOW)].reset_index(drop=True)
    n_boro, n_month = bm["boro"].nunique(), bm["month"].nunique()
    print(f"Borough panel: {len(bm)} rows = {n_boro} boroughs x "
          f"{n_month} months; monitoring {bm['monitoring'].sum():,}, "
          f"project-months {bm['active_jobs'].sum():,}; "
          f"zero cells monitoring={int((bm['monitoring'] == 0).sum())}, "
          f"projects={int((bm['active_jobs'] == 0).sum())}")
    return bm


def crosscheck_jobs(bm: pd.DataFrame) -> None:
    """Active projects re-derived from jobs.csv.gz (BBL-digit borough)
    should track the tract-panel aggregate; the panel drops ~3%
    tract-unmatched."""
    j = pd.read_csv(os.path.join(SPINE_DIR, "jobs.csv.gz"),
                    dtype={"bbl": str},
                    usecols=["bbl", "active_start_month", "active_end_month"])
    j["boro"] = j["bbl"].str[0]
    j = j.dropna(subset=["active_start_month", "active_end_month"])
    j = j[j["boro"].isin(list("12345"))
          & (j["active_end_month"] >= j["active_start_month"])
          & (j["active_start_month"] <= WINDOW[1])   # clamp span to the
          & (j["active_end_month"] >= WINDOW[0])]    # panel window

    months = pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)
    m_id = pd.Series(np.arange(len(months)), index=months)
    s = j["active_start_month"].clip(lower=WINDOW[0]).map(m_id).astype(int)
    e = j["active_end_month"].clip(upper=WINDOW[1]).map(m_id).astype(int)
    keep = (e >= s).to_numpy()
    b_id = (j["boro"].astype(int) - 1).to_numpy()[keep]
    s, e = s.to_numpy()[keep], e.to_numpy()[keep]
    lens = e - s + 1
    base = np.repeat(b_id * len(months) + s, lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    counts = np.bincount(base + offs, minlength=5 * len(months))
    chk = pd.DataFrame({"boro": np.repeat(list("12345"), len(months)),
                        "month": np.tile(months, 5), "jobs_bbl": counts})
    m = bm.merge(chk, on=["boro", "month"], how="left")
    corr = np.corrcoef(m["active_jobs"], m["jobs_bbl"])[0, 1]
    ratio = m["active_jobs"].sum() / m["jobs_bbl"].sum()
    print(f"  cross-check vs jobs.csv.gz BBL boroughs: corr {corr:.4f}, "
          f"panel/BBL level ratio {ratio:.3f}")


# ── program decomposition (Critic B #2) ──────────────────────────────────

def load_agency_events() -> pd.DataFrame:
    """Agency-originated events with prefix, family, BBL, base-job link."""
    ev = pd.read_csv(os.path.join(SPINE_DIR, "proactive_events.csv.gz"),
                     usecols=["received_date", "category_prefix", "family",
                              "agency", "bbl", "active_job_key"],
                     dtype={"category_prefix": str, "bbl": str,
                            "active_job_key": str},
                     parse_dates=["received_date"], low_memory=False)
    ev = ev[ev["agency"] == 1].reset_index(drop=True)
    ev["yr"] = ev["received_date"].dt.year.astype(str)
    print(f"Agency events for decomposition: {len(ev):,} "
          f"(discretionary_field "
          f"{(ev['family'] == 'discretionary_field').sum():,})")
    return ev


def load_projects() -> pd.DataFrame:
    """Base projects with window-clipped spans and cost deciles."""
    j = pd.read_csv(os.path.join(SPINE_DIR, "jobs.csv.gz"),
                    usecols=["job_key", "bbl", "initial_cost",
                             "active_start", "active_end"],
                    dtype={"bbl": str},
                    parse_dates=["active_start", "active_end"])
    j = j[j["active_start"].notna() & j["active_end"].notna()]
    j["span_start"] = j["active_start"].clip(lower=WINDOW_START_TS)
    j["span_end"] = j["active_end"].clip(upper=WINDOW_END_TS)
    j = j[j["span_end"] >= j["span_start"]].reset_index(drop=True)
    j["decile"] = pd.qcut(j["initial_cost"].rank(method="first"), 10,
                          labels=False) + 1
    print(f"Projects with valid clipped spans: {len(j):,} "
          f"(D10 cost floor "
          f"{j.loc[j['decile'] == 10, 'initial_cost'].min():,.0f})")
    return j


def program_mix_rows(ev: pd.DataFrame) -> pd.DataFrame:
    """Sweep/watchlist vs construction-core volumes by year, citywide raw
    agency counts (tract-unmatched included, unlike the panel series)."""
    disc = ev[ev["family"] == "discretionary_field"]
    pfx = (disc.pivot_table(index="category_prefix", columns="yr",
                            values="family", aggfunc="size")
           .fillna(0).astype(int))
    pfx = pfx[[c for c in pfx.columns if "2019" < c < "2027"]]

    sweep = pfx.loc[[p for p in SWEEP_PREFIXES if p in pfx.index]].sum()
    core = pfx.loc[[p for p in CORE_PREFIXES if p in pfx.index]].sum()
    core91 = core + pfx.loc["91"]
    total = pfx.sum()
    groups = {
        "sweep_watchlist": (sweep, "prefixes " + "/".join(SWEEP_PREFIXES)
                            + " (area sweeps, watchlists, padlock)"),
        "core_8A1X": (core, "prefixes 8A+1X (construction compliance + "
                      "emergency work orders)"),
        "core_8A1X91": (core91, "critic variant incl 91 worker "
                        "endangerment (plan quotes +1.8%/+6.2% from this)"),
        "prefix_8A": (pfx.loc["8A"], "8A construction compliance alone"),
        "other_disc": (total - sweep - core,
                       "remaining discretionary_field prefixes"),
        "total_disc": (total, "all discretionary_field"),
    }

    rows = []

    def add(term, value, note=""):
        rows.append({"term": term, "estimate": float(value),
                     "model": "program_decomposition", "note": note})

    for grp, (s, note) in groups.items():
        for yr in s.index:
            suf = " (Jan-May)" if yr == "2026" else ""
            add(f"{grp}_events_per_month_{yr}", s[yr] / MONTHS_IN_YEAR[yr],
                f"citywide raw agency counts{suf}; {note}")
        m23 = s["2023"] / 12
        add(f"{grp}_change_pct_2023_to_2025",
            (s["2025"] / 12 / m23 - 1) * 100, note)
        add(f"{grp}_change_pct_2023_to_2026",
            (s["2026"] / 5 / m23 - 1) * 100,
            note + "; 2026 = Jan-May, right-censored")

    decline_total = total["2023"] / 12 - total["2025"] / 12
    decline_sweep = sweep["2023"] / 12 - sweep["2025"] / 12
    add("sweep_share_of_total_disc_decline_2023_2025_pct",
        100 * decline_sweep / decline_total,
        "share of the 2023->2025 monthly-rate decline in discretionary_field "
        "accounted for by sweep/watchlist programs (>100%: the core rose)")

    print("\nProgram decomposition (citywide raw counts, events/month):")
    show = pd.DataFrame({g: s for g, (s, _) in groups.items()}).T
    show = show / pd.Series(MONTHS_IN_YEAR)
    print(show.round(1).to_string())
    print(f"  sweep/watchlist 2023->2025 "
          f"{(sweep['2025'] / 12 / (sweep['2023'] / 12) - 1) * 100:+.1f}%, "
          f"core 8A+1X {(core['2025'] / 12 / (core['2023'] / 12) - 1) * 100:+.1f}% "
          f"(incl 91: {(core91['2025'] / 12 / (core91['2023'] / 12) - 1) * 100:+.1f}%); "
          f"sweeps account for {100 * decline_sweep / decline_total:.0f}% of "
          f"the total discretionary decline")
    return pd.DataFrame(rows)


def topdecile_8a_rows(ev: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """8A per 100 project-months at top-cost-decile projects by year.
    Event -> base-project assignment: direct active_job_key link inside
    the span, then largest-cost BBL fallback (spine convention)."""
    dset = ev[ev["family"].isin(["discretionary_field", "followup"])].copy()
    dset = dset.reset_index(drop=True)
    dset["eid"] = dset.index

    span = proj[["job_key", "bbl", "span_start", "span_end", "initial_cost"]]
    d1 = dset.merge(span[["job_key", "span_start", "span_end"]],
                    left_on="active_job_key", right_on="job_key",
                    how="inner")
    d1 = d1[d1["received_date"].between(d1["span_start"], d1["span_end"])]
    d1 = d1[["eid", "job_key", "received_date", "category_prefix"]]

    rest = dset[~dset["eid"].isin(d1["eid"]) & dset["bbl"].notna()]
    f1 = rest.merge(span[span["bbl"].notna()], on="bbl", how="inner")
    f1 = f1[f1["received_date"].between(f1["span_start"], f1["span_end"])]
    f1 = (f1.sort_values("initial_cost", ascending=False, kind="stable")
          .drop_duplicates("eid"))[["eid", "job_key", "received_date",
                                    "category_prefix"]]
    asg = pd.concat([d1, f1], ignore_index=True)
    asg = asg.merge(proj[["job_key", "decile"]], on="job_key")
    asg["yr"] = asg["received_date"].dt.year.astype(str)
    print(f"\nEvent -> project assignment for 8A decile trend: "
          f"{len(asg):,} of {len(dset):,} disc+followup events "
          f"({len(asg) / len(dset):.1%}; direct {len(d1):,}, "
          f"BBL fallback {len(f1):,})")

    rows = []

    def add(term, value, note=""):
        rows.append({"term": term, "estimate": float(value),
                     "model": "program_decomposition", "note": note})

    rates = {}
    for yr in ("2022", "2023", "2024", "2025", "2026"):
        y0 = pd.Timestamp(f"{yr}-01-01")
        y1 = min(pd.Timestamp(f"{yr}-12-31"), WINDOW_END_TS)
        mo = ((proj["span_end"].clip(upper=y1)
               - proj["span_start"].clip(lower=y0)).dt.days + 1)
        mo = mo.clip(lower=0) / DAYS_PER_MONTH
        for dec, lab in ((10, "D10"), (None, "all")):
            mask = (proj["decile"] == 10) if dec else pd.Series(
                True, index=proj.index)
            em = asg["yr"] == yr
            if dec:
                em = em & (asg["decile"] == 10)
            jm = mo[mask].sum()
            r8a = 100 * (em & (asg["category_prefix"] == "8A")).sum() / jm
            rall = 100 * em.sum() / jm
            rates[(lab, yr)] = r8a
            suf = "; 2026 = Jan-May" if yr == "2026" else ""
            add(f"rate_8A_per100_projmonths_{lab}_{yr}", r8a,
                f"8A events per 100 base-project months, {lab}{suf}")
            add(f"rate_disc_followup_per100_projmonths_{lab}_{yr}", rall,
                f"disc+followup events per 100 base-project months, {lab}{suf}")
    for lab in ("D10", "all"):
        add(f"rate_8A_per100_projmonths_{lab}_change_pct_2023_to_2025",
            (rates[(lab, "2025")] / rates[(lab, "2023")] - 1) * 100)
        add(f"rate_8A_per100_projmonths_{lab}_change_pct_2023_to_2026",
            (rates[(lab, "2026")] / rates[(lab, "2023")] - 1) * 100,
            "2026 = Jan-May, right-censored")

    print("8A per 100 project-months (top cost decile vs all):")
    for lab in ("D10", "all"):
        ser = "  ".join(f"{yr} {rates[(lab, yr)]:.2f}"
                        for yr in ("2022", "2023", "2024", "2025", "2026"))
        print(f"  {lab:>4}: {ser}  | 2023->2026 "
              f"{(rates[(lab, '2026')] / rates[(lab, '2023')] - 1) * 100:+.1f}%")
    return pd.DataFrame(rows)


def load_ecb() -> pd.DataFrame:
    """Monthly ECB Site Safety + Cranes and Derricks citations by borough."""
    conn = sqlite3.connect(DB_PATH)
    ecb = pd.read_sql_query("""
        SELECT boro, substr(issue_date, 1, 6) AS yyyymm, violation_type,
               COUNT(*) AS n
        FROM ecb_violations
        WHERE violation_type IN ('Site Safety', 'Cranes and Derricks')
          AND issue_date >= '20200101' AND issue_date <= '20260531'
          AND boro IN ('1', '2', '3', '4', '5')
        GROUP BY 1, 2, 3
    """, conn)
    conn.close()
    ecb["month"] = ecb["yyyymm"].str[:4] + "-" + ecb["yyyymm"].str[4:]
    wide = (ecb.pivot_table(index=["boro", "month"], columns="violation_type",
                            values="n", aggfunc="sum", fill_value=0)
               .rename(columns={"Site Safety": "ecb_site_safety",
                                "Cranes and Derricks": "ecb_cranes"})
               .reset_index())
    print(f"ECB 2020-01..2026-05: site safety "
          f"{wide['ecb_site_safety'].sum():,}, cranes "
          f"{wide['ecb_cranes'].sum():,} (Wave-1: 3,552 / 1,688)")
    return wide[["boro", "month", "ecb_site_safety", "ecb_cranes"]]


def build_series(bm: pd.DataFrame, ecb: pd.DataFrame) -> pd.DataFrame:
    """Citywide + borough monthly series with per-100 rate and 2020=100
    indices (raw rate for every month, early denominators and all)."""
    s = bm.merge(ecb, on=["boro", "month"], how="left")
    s[["ecb_site_safety", "ecb_cranes"]] = (
        s[["ecb_site_safety", "ecb_cranes"]].fillna(0).astype(int))
    s["geo"] = s["boro"].map(BORO_NAME)

    city = (s.groupby("month", as_index=False)
              [["monitoring", "active_jobs", "ecb_site_safety", "ecb_cranes"]]
              .sum())
    city["geo"] = "Citywide"
    out = pd.concat([city, s.drop(columns="boro")], ignore_index=True)

    out["monitoring_per_100_jobs"] = (
        out["monitoring"] / out["active_jobs"] * 100)
    base = (out[out["month"].str[:4] == "2020"]
            .groupby("geo")[["active_jobs", "monitoring"]].mean())
    out["idx_active_jobs_2020avg100"] = (
        out["active_jobs"] / out["geo"].map(base["active_jobs"]) * 100)
    out["idx_monitoring_2020avg100"] = (
        out["monitoring"] / out["geo"].map(base["monitoring"]) * 100)

    out["geo"] = pd.Categorical(out["geo"], GEO_ORDER, ordered=True)
    out = (out.sort_values(["geo", "month"]).reset_index(drop=True)
              [["geo", "month", "active_jobs", "monitoring",
                "monitoring_per_100_jobs", "ecb_site_safety", "ecb_cranes",
                "idx_active_jobs_2020avg100", "idx_monitoring_2020avg100"]]
              .rename(columns={"monitoring": "monitoring_events"}))
    return out


# ── models ───────────────────────────────────────────────────────────────

def run_models(bm: pd.DataFrame) -> pd.DataFrame:
    d = bm.copy()
    d["log_jobs"] = np.log(d["active_jobs"])
    mi = (d["month"].str[:4].astype(int) * 12
          + d["month"].str[5:].astype(int))
    d["month_t"] = mi - mi.min()
    mature = d[d["month"] >= MATURE_START]

    superseded = ("SUPERSEDED (Critic B #6): point elasticity is "
                  "spec-fragile and sign-flips in the mature window on "
                  "corrected project-level denominators; do not quote. "
                  "Kept only for the unit-elasticity rejection "
                  "(p_vs_unit_elasticity). ")
    specs = [
        ("ppml_trend_cl_month", "monitoring ~ log_jobs + month_t | boro",
         d, {"CRV1": "month"},
         superseded + "was headline: borough FE + linear month trend, "
         "cluster by month"),
        ("ppml_trend_cl_borough", "monitoring ~ log_jobs + month_t | boro",
         d, {"CRV1": "boro"},
         superseded + "cluster by borough (5 clusters, small-sample caveat)"),
        ("ppml_trend_2023on", "monitoring ~ log_jobs + month_t | boro",
         mature, {"CRV1": "month"},
         superseded + "2023-01..2026-05, DOB NOW job stock mature"),
        ("ppml_no_trend", "monitoring ~ log_jobs | boro",
         d, {"CRV1": "month"},
         superseded + "no trend (raw co-movement, adoption ramp included)"),
        ("ppml_month_fe", "monitoring ~ log_jobs | boro + month",
         d, {"CRV1": "month"},
         superseded + "borough + month FE, cross-borough identification"),
    ]

    rows = []
    for name, fml, data, vcov, note in specs:
        m = pf.fepois(fml, data=data, vcov=vcov)
        t = m.tidy().reset_index().rename(columns={
            "Coefficient": "term", "Estimate": "estimate",
            "Std. Error": "std_error", "t value": "t_value",
            "Pr(>|t|)": "p_value", "2.5%": "ci_low", "97.5%": "ci_high"})
        t["model"] = name
        t["n"] = pd.array([m._N] * len(t), dtype="Int64")
        t["vcov"] = "CRV1 " + list(vcov.values())[0]
        t["note"] = note
        is_el = t["term"] == "log_jobs"
        z1 = (t["estimate"] - 1.0) / t["std_error"]
        t["p_vs_unit_elasticity"] = np.where(
            is_el, 2 * stats.norm.sf(np.abs(z1)), np.nan)
        rows.append(t)
        el = t.loc[is_el].iloc[0]
        print(f"  {name:>22}: elasticity {el['estimate']:.3f} "
              f"[{el['ci_low']:.3f}, {el['ci_high']:.3f}] "
              f"p_vs_1 {el['p_vs_unit_elasticity']:.2e} (n={m._N})")
    return pd.concat(rows, ignore_index=True)


def descriptive_rows(series: pd.DataFrame) -> pd.DataFrame:
    """Citywide quotables: yearly averages/totals plus the deltas the post
    will cite. 2026 covers January-May only."""
    c = series[series["geo"] == "Citywide"].copy()
    c["year"] = c["month"].str[:4]
    y = c.groupby("year").agg(
        jobs=("active_jobs", "mean"), monit=("monitoring_events", "mean"),
        per100=("monitoring_per_100_jobs", "mean"),
        ss=("ecb_site_safety", "sum"), cr=("ecb_cranes", "sum"))

    rows = []

    def add(term, value, note=""):
        rows.append({"term": term, "estimate": float(value),
                     "model": "descriptive_citywide", "note": note})

    for yr, r in y.iterrows():
        suf = " (Jan-May)" if yr == "2026" else ""
        add(f"active_jobs_monthly_avg_{yr}", r["jobs"],
            f"active BASE PROJECTS mean/month{suf} (Critic B collapse)")
        add(f"monitoring_monthly_avg_{yr}", r["monit"], f"mean/month{suf}")
        add(f"per100_avg_{yr}", r["per100"],
            f"mean of monthly rates per 100 active projects{suf}; "
            "pre-2023 denominators understated (DOB NOW adoption ramp)")
        add(f"ecb_site_safety_total_{yr}", r["ss"], f"citations{suf}")
        add(f"ecb_cranes_total_{yr}", r["cr"], f"citations{suf}")

    def pct(a, b):
        return (b / a - 1) * 100

    add("per100_change_pct_2023_to_2025", pct(y.loc["2023", "per100"],
                                              y.loc["2025", "per100"]),
        "mature-window decline, project-level denominator "
        "(published filing-level figure was -33.2%)")
    add("per100_change_pct_2023_to_2026", pct(y.loc["2023", "per100"],
                                              y.loc["2026", "per100"]),
        "2026 = Jan-May avg, right-censored")
    add("per100_change_pct_2024_to_2026", pct(y.loc["2024", "per100"],
                                              y.loc["2026", "per100"]),
        "2026 = Jan-May avg")
    add("monitoring_change_pct_2022_to_2025", pct(y.loc["2022", "monit"],
                                                  y.loc["2025", "monit"]),
        "absolute monitoring decline")
    add("monitoring_change_pct_2023_to_2025", pct(y.loc["2023", "monit"],
                                                  y.loc["2025", "monit"]),
        "absolute monitoring decline, mature window")
    add("active_jobs_change_pct_2022_to_2025", pct(y.loc["2022", "jobs"],
                                                   y.loc["2025", "jobs"]),
        "base projects")
    add("active_jobs_change_pct_2023_to_2025", pct(y.loc["2023", "jobs"],
                                                   y.loc["2025", "jobs"]),
        "base projects; Critic B: project stock FELL while filing "
        "stock rose (+7.7%)")
    add("ecb_site_safety_change_pct_2020_to_2024",
        pct(y.loc["2020", "ss"], y.loc["2024", "ss"]),
        "SUPERSEDED (Critic B #3): spans the era_post break; do not "
        "quote as corroboration")
    add("ecb_site_safety_change_pct_2023_to_2025",
        pct(y.loc["2023", "ss"], y.loc["2025", "ss"]),
        "in-window check: Site Safety RISES 2023->2025, does not "
        "corroborate a monitoring decline")
    add("ecb_site_safety_change_pct_2024_to_2025",
        pct(y.loc["2024", "ss"], y.loc["2025", "ss"]), "partial rebound")
    add("ecb_cranes_change_pct_2020_to_2025",
        pct(y.loc["2020", "cr"], y.loc["2025", "cr"]),
        "SUPERSEDED (Critic B #3): spans the era_post break")
    add("ecb_cranes_change_pct_2023_to_2025",
        pct(y.loc["2023", "cr"], y.loc["2025", "cr"]), "in-window check")
    return pd.DataFrame(rows)


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(series: pd.DataFrame) -> str:
    c = series[series["geo"] == "Citywide"].reset_index(drop=True)
    x = pd.to_datetime(c["month"] + "-01")
    mature = c["month"] >= MATURE_START

    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=160)
    ax.axvspan(x.iloc[0], pd.Timestamp(MATURE_START + "-01"),
               color=TINT, zorder=0)
    ax.axhline(100, color=ZERO_C, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot(x, c["idx_active_jobs_2020avg100"], color=BLUE, linewidth=2,
            zorder=3)
    ax.plot(x, c["idx_monitoring_2020avg100"], color=RED, linewidth=2,
            zorder=3)

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ymax = max(c["idx_active_jobs_2020avg100"].max(),
               c["idx_monitoring_2020avg100"].max()) * 1.14
    ax.set_ylim(0, ymax)
    ax.set_ylabel("index, 2020 monthly average = 100", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # per-100 rate on the right axis, drawn once the project stock is mature
    p2max = c.loc[mature, "monitoring_per_100_jobs"].max() * 1.35
    ax2 = ax.twinx()
    ax2.plot(x[mature], c.loc[mature, "monitoring_per_100_jobs"],
             color=AQUA, linewidth=2, linestyle=(0, (5, 2)), zorder=3)
    ax2.set_ylim(0, p2max)
    ax2.set_ylabel("monitoring events per 100 active projects",
                   fontsize=10.5, color=AQUA)
    ax2.tick_params(axis="y", colors=AQUA)
    ax2.spines["right"].set_color(BASE)
    for s in ["top", "left", "bottom"]:
        ax2.spines[s].set_visible(False)

    # inline labels
    ax.text(pd.Timestamp("2024-06-01"), 0.94 * ymax,
            "Active construction projects", fontsize=11.5, color=BLUE,
            fontweight="bold")
    ax.text(pd.Timestamp("2021-11-01"), 0.36 * ymax,
            "Discretionary field monitoring\n(sweeps, compliance, work orders)",
            fontsize=11.5, color=RED, fontweight="bold")
    # label the right-axis series at its start so the connector stays short
    # and never crosses the red annotation or either data series
    y_start = c.loc[mature, "monitoring_per_100_jobs"].iloc[0] / p2max * ymax
    ax.annotate("per 100 active projects (right axis)",
                xy=(pd.Timestamp(MATURE_START + "-01"), y_start + 4),
                xytext=(pd.Timestamp("2022-12-01"), y_start + 16),
                fontsize=10.5, color=AQUA, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=0.9))
    yy = c["month"].str[:4]
    r23 = c.loc[yy == "2023", "monitoring_per_100_jobs"].mean()
    r25 = c.loc[yy == "2025", "monitoring_per_100_jobs"].mean()
    r26 = c.loc[yy == "2026", "monitoring_per_100_jobs"].mean()
    ax.text(pd.Timestamp("2025-11-15"), 0.06 * ymax,
            f"{r23:.1f} → {r26:.1f}\nper 100 projects", fontsize=10,
            color=AQUA, ha="right")
    ax.text(pd.Timestamp("2020-02-15"), 0.93 * ymax,
            "DOB NOW adoption ramp:\nproject counts incomplete before 2023",
            fontsize=9.5, color=MUTED, style="italic")

    ax.set_title(f"Monitoring per active project fell "
                 f"{abs(r25 / r23 - 1) * 100:.0f}% from 2023 to 2025",
                 loc="left", fontsize=15, fontweight="bold", color=INK,
                 pad=26)
    ax.text(0, 1.015, "citywide monthly series, January 2020 – May 2026 "
            "· active DOB NOW base projects (filing suffixes collapsed) vs "
            "agency-initiated field monitoring",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)

    fig.tight_layout()
    path = os.path.join(ART_DIR, "proactive_dilution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RISK_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    bm = load_borough_month()
    crosscheck_jobs(bm)
    ecb = load_ecb()
    series = build_series(bm, ecb)

    print("\nPPML elasticities (monitoring ~ log active projects) — "
          "SUPERSEDED point estimates, unit-elasticity rejection only:")
    est = run_models(bm)

    ev_ag = load_agency_events()
    proj = load_projects()
    est = pd.concat([est, descriptive_rows(series), program_mix_rows(ev_ag),
                     topdecile_8a_rows(ev_ag, proj)], ignore_index=True)

    p_est = os.path.join(RISK_DIR, "proactive_dilution_estimates.csv")
    p_ser = os.path.join(RISK_DIR, "proactive_dilution_series.csv")
    est.to_csv(p_est, index=False)
    series.to_csv(p_ser, index=False)
    p_fig = make_figure(series)

    city = series[series["geo"] == "Citywide"]
    yr = city.assign(year=city["month"].str[:4]).groupby("year")
    print("\nCitywide by year (2026 = Jan-May):")
    print(yr.agg(jobs_avg=("active_jobs", "mean"),
                 monit_avg=("monitoring_events", "mean"),
                 per100_avg=("monitoring_per_100_jobs", "mean"),
                 site_safety=("ecb_site_safety", "sum"),
                 cranes=("ecb_cranes", "sum")).round(2).to_string())

    print("\n== Outputs ==")
    for p in (p_est, p_ser, p_fig):
        print(f"  {p} ({os.path.getsize(p) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
