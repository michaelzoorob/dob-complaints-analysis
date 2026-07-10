#!/usr/bin/env python3
"""
Boom dilution (proactive plan hypothesis #6): did discretionary field
monitoring keep pace with the construction boom?

Spine: data/analysis/proactive/tract_month.csv.gz (borough = first digit
of bct2020) aggregated to borough x month, 2020-01..2026-05. Monitoring =
proactive_discretionary_field (agency-originated sweeps / compliance /
work-order categories per analysis_config.PROACTIVE_FAMILIES). Exposure =
active_jobs (DOB NOW jobs first permitted 2020+, active span per
proactive_spine.py). ECB Site Safety and Cranes and Derricks citations
come straight from ecb_violations (issue_date YYYYMMDD, boro 1-5).

Headline model: borough x month PPML of monitoring counts on
log(active jobs) with borough FE and a linear month trend. Elasticity
below one means oversight dilutes as the job stock grows. SEs cluster by
month (cross-borough correlation; Driscoll-Kraay is unavailable for
fepois in pyfixest 0.60) with a borough-cluster robustness (5 clusters,
small-sample caveat). Because the job universe only accumulates DOB NOW
filings from 2020 (legacy BIS permits not ingested; plan flags the gap),
the 2020-22 exposure ramp is partly mechanical adoption; a 2023-onward
window and a borough+month-FE variant probe that.

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
MATURE_START = "2023-01"  # DOB NOW job stock mature; pre-2023 ramp is
                          # partly system adoption, not real exposure

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
    """Tract-month spine -> borough x month monitoring + active jobs."""
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
          f"job-months {bm['active_jobs'].sum():,}; "
          f"zero cells monitoring={int((bm['monitoring'] == 0).sum())}, "
          f"jobs={int((bm['active_jobs'] == 0).sum())}")
    return bm


def crosscheck_jobs(bm: pd.DataFrame) -> None:
    """Active jobs re-derived from jobs.csv.gz (BBL-digit borough) should
    track the tract-panel aggregate; the panel drops ~3% tract-unmatched."""
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

    specs = [
        ("ppml_trend_cl_month", "monitoring ~ log_jobs + month_t | boro",
         d, {"CRV1": "month"},
         "headline: borough FE + linear month trend, cluster by month"),
        ("ppml_trend_cl_borough", "monitoring ~ log_jobs + month_t | boro",
         d, {"CRV1": "boro"},
         "robustness: cluster by borough (5 clusters, small-sample caveat)"),
        ("ppml_trend_2023on", "monitoring ~ log_jobs + month_t | boro",
         mature, {"CRV1": "month"},
         "robustness: 2023-01..2026-05, DOB NOW job stock mature"),
        ("ppml_no_trend", "monitoring ~ log_jobs | boro",
         d, {"CRV1": "month"},
         "robustness: no trend (raw co-movement, adoption ramp included)"),
        ("ppml_month_fe", "monitoring ~ log_jobs | boro + month",
         d, {"CRV1": "month"},
         "robustness: borough + month FE, cross-borough identification"),
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
        add(f"active_jobs_monthly_avg_{yr}", r["jobs"], f"mean/month{suf}")
        add(f"monitoring_monthly_avg_{yr}", r["monit"], f"mean/month{suf}")
        add(f"per100_avg_{yr}", r["per100"],
            f"mean of monthly rates{suf}; pre-2023 denominators "
            "understated (DOB NOW adoption ramp)")
        add(f"ecb_site_safety_total_{yr}", r["ss"], f"citations{suf}")
        add(f"ecb_cranes_total_{yr}", r["cr"], f"citations{suf}")

    def pct(a, b):
        return (b / a - 1) * 100

    add("per100_change_pct_2023_to_2025", pct(y.loc["2023", "per100"],
                                              y.loc["2025", "per100"]),
        "mature-window dilution")
    add("per100_change_pct_2024_to_2026", pct(y.loc["2024", "per100"],
                                              y.loc["2026", "per100"]),
        "2026 = Jan-May avg")
    add("monitoring_change_pct_2022_to_2025", pct(y.loc["2022", "monit"],
                                                  y.loc["2025", "monit"]),
        "absolute monitoring decline during boom")
    add("active_jobs_change_pct_2022_to_2025", pct(y.loc["2022", "jobs"],
                                                   y.loc["2025", "jobs"]))
    add("active_jobs_change_pct_2023_to_2025", pct(y.loc["2023", "jobs"],
                                                   y.loc["2025", "jobs"]))
    add("ecb_site_safety_change_pct_2020_to_2024",
        pct(y.loc["2020", "ss"], y.loc["2024", "ss"]))
    add("ecb_site_safety_change_pct_2024_to_2025",
        pct(y.loc["2024", "ss"], y.loc["2025", "ss"]), "partial rebound")
    add("ecb_cranes_change_pct_2020_to_2025",
        pct(y.loc["2020", "cr"], y.loc["2025", "cr"]))
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
    ax.set_ylim(0, 660)
    ax.set_ylabel("index, 2020 monthly average = 100", fontsize=10.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # per-100 rate on the right axis, drawn once the job stock is mature
    ax2 = ax.twinx()
    ax2.plot(x[mature], c.loc[mature, "monitoring_per_100_jobs"],
             color=AQUA, linewidth=2, linestyle=(0, (5, 2)), zorder=3)
    ax2.set_ylim(0, 2.1)
    ax2.set_ylabel("monitoring events per 100 active jobs", fontsize=10.5,
                   color=AQUA)
    ax2.tick_params(axis="y", colors=AQUA)
    ax2.spines["right"].set_color(BASE)
    for s in ["top", "left", "bottom"]:
        ax2.spines[s].set_visible(False)

    # inline labels
    ax.text(pd.Timestamp("2024-06-01"), 570, "Active construction jobs",
            fontsize=11.5, color=BLUE, fontweight="bold")
    ax.text(pd.Timestamp("2021-11-01"), 237,
            "Discretionary field monitoring\n(sweeps, compliance, work orders)",
            fontsize=11.5, color=RED, fontweight="bold")
    ax.annotate("per 100 active jobs (right axis)",
                xy=(pd.Timestamp("2023-08-01"),
                    c.loc[c["month"] == "2023-08",
                          "monitoring_per_100_jobs"].iat[0] / 2.1 * 660),
                xytext=(pd.Timestamp("2022-01-15"), 105),
                fontsize=10.5, color=AQUA,
                arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=0.9))
    r23 = c.loc[c["month"].str[:4] == "2023", "monitoring_per_100_jobs"].mean()
    r26 = c.loc[c["month"].str[:4] == "2026", "monitoring_per_100_jobs"].mean()
    ax.text(pd.Timestamp("2025-11-15"), 40,
            f"{r23:.1f} → {r26:.1f}\nper 100 jobs", fontsize=10,
            color=AQUA, ha="right")
    ax.text(pd.Timestamp("2020-02-15"), 615,
            "DOB NOW adoption ramp:\njob counts incomplete before 2023",
            fontsize=9.5, color=MUTED, style="italic")

    ax.set_title("Construction boomed. Proactive monitoring did not.",
                 loc="left", fontsize=15, fontweight="bold", color=INK,
                 pad=14)
    ax.text(0, 1.015, "citywide monthly series, January 2020 – May 2026 "
            "· active DOB NOW jobs vs agency-initiated field monitoring",
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

    print("\nPPML elasticities (monitoring ~ log active jobs):")
    est = run_models(bm)
    est = pd.concat([est, descriptive_rows(series)], ignore_index=True)

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
