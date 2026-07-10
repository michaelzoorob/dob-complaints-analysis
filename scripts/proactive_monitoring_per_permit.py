#!/usr/bin/env python3
"""
Hypothesis #1 (proactive_enforcement_plan.md): monitoring per permit.

Does DOB's discretionary construction oversight scale with project size,
and do office-to-residential conversions draw more (or less) of it than
otherwise-similar alterations?

Unit: DOB NOW job first permitted 2020+ with a valid active span
(data/analysis/proactive/jobs.csv.gz, built by proactive_spine.py), span
clipped to the event window 2020-01..2026-05.

Outcome: count of agency-originated complaint events (empty ref_311) in
the discretionary construction families — discretionary_field (8A
compliance, 7G sweeps, 1X EWO, 91 worker endangerment, ...) plus followup
(7R, 4G, ...) — assigned to the job. statutory_periodic (7K/7F/6V boiler,
facade, tenant-protection cycles, ...) is run separately as a contrast:
cyclical volume should not respond to job traits the way discretionary
targeting does.

Event -> job assignment (one job per event, no double counting):
  1. direct: the event's active-permit job_filing_number (largest-cost
     permit active at the received date, from the spine) matches a spine
     job and the received date falls inside that job's clipped span;
  2. fallback: BBL match with received date inside the clipped span,
     ties resolved to the largest initial_cost (spine convention).

Model: PPML with a log(active months) offset,
  y ~ conversion_flag + log(initial_cost+1) + log(floor_area+1)
      + floor_missing + job-type dummies | nta^first_permit_quarter,
  SEs clustered on BBL (treatment varies at the lot). Conversion flags
  reported for all three spine definitions: relaxed (Alt-CO family,
  0 existing units, >0 proposed; n=1,337), strict (proposed>=10; n=630),
  office-class (relaxed + PLUTO class O/K; n=412). NTA comes from the
  modal events-file tract->NTA crosswalk.

Outputs
  data/analysis/risk_models/proactive_monitoring_estimates.csv
  data/analysis/risk_models/proactive_per_permit_figure.csv
  data/analysis/blog_posts/artifacts/proactive_per_permit.png

Run:  /private/tmp/pyfix_venv/bin/python scripts/proactive_monitoring_per_permit.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "data" / "analysis" / "proactive"
RM = ROOT / "data" / "analysis" / "risk_models"
ART = ROOT / "data" / "analysis" / "blog_posts" / "artifacts"

WINDOW_START = pd.Timestamp("2020-01-01")
WINDOW_END = pd.Timestamp("2026-05-31")  # event coverage ends here
DAYS_PER_MONTH = 30.4375

DISC_FAMILIES = ("discretionary_field", "followup")
STAT_FAMILY = "statutory_periodic"

CONV_DEFS = {
    "conversion": "relaxed (Alt-CO, 0 existing units, >0 proposed)",
    "conversion_ge10": "strict (relaxed + proposed>=10 units)",
    "conversion_office": "office-class (relaxed + PLUTO class O/K)",
}

JOB_TYPE_DUMMIES = {
    "jt_new_building": "New Building",
    "jt_alteration_co": "Alteration CO",
    "jt_altco_existing": "ALT-CO - New Building with Existing Elements to Remain",
    "jt_full_demolition": "Full Demolition",
}  # base category: Alteration

X_TERMS = ("log_cost + log_floor + floor_missing + "
           + " + ".join(JOB_TYPE_DUMMIES))

# ── house style (constants from scripts/make_descriptive_figures.py) ────
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

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.1f}M" if v < 10e6 else f"${v / 1e6:.0f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


# ── data assembly ────────────────────────────────────────────────────────

def load_jobs() -> pd.DataFrame:
    jobs = pd.read_csv(
        SPINE / "jobs.csv.gz",
        usecols=["job_filing_number", "job_type", "initial_cost",
                 "total_construction_floor_area", "bbl", "bct2020",
                 "first_permit_date", "active_start", "active_end",
                 "conversion", "conversion_ge10", "conversion_office"],
        dtype={"bbl": "str", "bct2020": "str"},
        parse_dates=["first_permit_date", "active_start", "active_end"])
    n0 = len(jobs)

    jobs = jobs[jobs["active_start"].notna() & jobs["active_end"].notna()]
    jobs["span_start"] = jobs["active_start"].clip(lower=WINDOW_START)
    jobs["span_end"] = jobs["active_end"].clip(upper=WINDOW_END)
    jobs = jobs[jobs["span_end"] >= jobs["span_start"]].reset_index(drop=True)
    jobs["months"] = ((jobs["span_end"] - jobs["span_start"]).dt.days + 1) / DAYS_PER_MONTH

    print(f"Jobs: {n0:,} in spine -> {len(jobs):,} with a valid active span "
          f"inside {WINDOW_START:%Y-%m}..{WINDOW_END:%Y-%m} "
          f"({jobs['months'].sum():,.0f} job-months, median span "
          f"{jobs['months'].median():.1f} mo)")
    return jobs


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(
        SPINE / "proactive_events.csv.gz",
        usecols=["received_date", "family", "agency", "bbl", "bct2020",
                 "nta", "active_job_filing_number"],
        dtype={"bbl": "str", "bct2020": "str", "nta": "str",
               "active_job_filing_number": "str"},
        parse_dates=["received_date"], low_memory=False)
    ev = ev[ev["agency"] == 1]
    print(f"Events: {len(ev):,} agency-originated 2020-01..2026-05; "
          f"disc {ev['family'].isin(DISC_FAMILIES).sum():,}, "
          f"stat {(ev['family'] == STAT_FAMILY).sum():,}")
    return ev


def tract_nta_crosswalk(ev: pd.DataFrame) -> pd.Series:
    xw = (ev.dropna(subset=["bct2020", "nta"])
          .groupby("bct2020")["nta"]
          .agg(lambda s: s.mode().iat[0]))
    print(f"Tract->NTA crosswalk from events: {len(xw):,} tracts, "
          f"{xw.nunique()} NTAs")
    return xw


def assign_events(ev: pd.DataFrame, jobs: pd.DataFrame,
                  label: str) -> pd.Series:
    """Assign each event to at most one job; return counts per job."""
    e = ev.reset_index(drop=True)
    e["eid"] = e.index

    span = jobs[["job_filing_number", "bbl", "span_start", "span_end",
                 "initial_cost"]]

    # 1. direct link on the spine's active-permit job filing number
    d = e.merge(span[["job_filing_number", "span_start", "span_end"]],
                left_on="active_job_filing_number",
                right_on="job_filing_number", how="inner")
    n_link = len(d)
    d = d[d["received_date"].between(d["span_start"], d["span_end"])]
    d = d[["eid", "job_filing_number"]]

    # 2. fallback: BBL + received date inside the clipped span,
    #    largest initial_cost wins when several jobs are active
    rest = e[~e["eid"].isin(d["eid"])].drop(columns=["job_filing_number"],
                                            errors="ignore")
    f = rest.merge(span, on="bbl", how="inner")
    f = f[f["received_date"].between(f["span_start"], f["span_end"])]
    f = (f.sort_values("initial_cost", ascending=False, kind="stable")
         .drop_duplicates("eid"))[["eid", "job_filing_number"]]

    assigned = pd.concat([d, f], ignore_index=True)
    print(f"  {label}: {len(e):,} events -> {len(assigned):,} assigned "
          f"({len(assigned) / len(e):.1%}); direct {len(d):,} "
          f"(of {n_link:,} in-spine links, {n_link - len(d):,} outside span), "
          f"bbl-span fallback {len(f):,}")
    return assigned.groupby("job_filing_number").size()


def build_model_frame() -> pd.DataFrame:
    jobs = load_jobs()
    ev = load_events()
    xw = tract_nta_crosswalk(ev)

    print("Event -> job assignment:")
    disc = ev[ev["family"].isin(DISC_FAMILIES)]
    stat = ev[ev["family"] == STAT_FAMILY]
    jobs["n_disc"] = (jobs["job_filing_number"]
                      .map(assign_events(disc, jobs, "discretionary+followup"))
                      .fillna(0).astype(int))
    jobs["n_stat"] = (jobs["job_filing_number"]
                      .map(assign_events(stat, jobs, "statutory contrast"))
                      .fillna(0).astype(int))

    # covariates
    jobs["nta"] = jobs["bct2020"].map(xw)
    jobs["permit_q"] = jobs["first_permit_date"].dt.to_period("Q").astype(str)
    jobs["log_cost"] = np.log1p(jobs["initial_cost"].clip(lower=0))
    floor = jobs["total_construction_floor_area"]
    jobs["floor_missing"] = floor.isna().astype("int8")
    jobs["log_floor"] = np.log1p(floor.fillna(0).clip(lower=0))
    jobs["log_months"] = np.log(jobs["months"])
    for col, val in JOB_TYPE_DUMMIES.items():
        jobs[col] = (jobs["job_type"] == val).astype("int8")

    mf = jobs[jobs["nta"].notna()].reset_index(drop=True)
    print(f"\nModel frame: {len(mf):,} jobs with NTA "
          f"({len(mf) / len(jobs):.1%} of span-valid); "
          f"disc events kept {mf['n_disc'].sum():,}, "
          f"stat {mf['n_stat'].sum():,}")
    print("  conversion flags in sample: "
          + ", ".join(f"{k} {int(mf[k].sum()):,}" for k in CONV_DEFS))
    print(f"  disc rate per 100 job-months: all "
          f"{100 * mf['n_disc'].sum() / mf['months'].sum():.2f}; conversions "
          f"{100 * mf.loc[mf['conversion'] == 1, 'n_disc'].sum() / mf.loc[mf['conversion'] == 1, 'months'].sum():.2f}")
    return mf


# ── models ───────────────────────────────────────────────────────────────

def run_models(mf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, out_label in [("n_disc", "discretionary_field+followup"),
                               ("n_stat", "statutory_periodic (contrast)")]:
        for flag, flag_label in CONV_DEFS.items():
            fml = f"{outcome} ~ {flag} + {X_TERMS} | nta^permit_q"
            t0 = time.time()
            m = pf.fepois(fml, data=mf, offset="log_months",
                          vcov={"CRV1": "bbl"})
            td = m.tidy().reset_index()
            n_obs = int(m._N)
            for _, r in td.iterrows():
                b, se = r["Estimate"], r["Std. Error"]
                lo, hi = r["2.5%"], r["97.5%"]
                rows.append({
                    "outcome": outcome, "outcome_label": out_label,
                    "conv_def": flag, "conv_def_label": flag_label,
                    "term": r["Coefficient"],
                    "b": b, "se": se, "t": r["t value"],
                    "p": r["Pr(>|t|)"], "ci_lo": lo, "ci_hi": hi,
                    "irr": np.exp(b), "irr_lo": np.exp(lo),
                    "irr_hi": np.exp(hi),
                    "n_obs": n_obs, "n_jobs_sample": len(mf),
                    "n_events": int(mf[outcome].sum()),
                    "fe": "nta^first_permit_quarter",
                    "offset": "log(active months)",
                    "cluster": "bbl",
                    "window": "2020-01..2026-05",
                })
            key = td[td["Coefficient"] == flag].iloc[0]
            print(f"  {outcome} ~ {flag}: b={key['Estimate']:+.3f} "
                  f"(se {key['Std. Error']:.3f}), IRR "
                  f"{np.exp(key['Estimate']):.2f} "
                  f"[{np.exp(key['2.5%']):.2f}, {np.exp(key['97.5%']):.2f}], "
                  f"n={n_obs:,} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(mf: pd.DataFrame) -> pd.DataFrame:
    mf = mf.copy()
    mf["decile"] = (pd.qcut(mf["initial_cost"].rank(method="first"),
                            10, labels=False) + 1)
    g = mf.groupby("decile").agg(
        n_jobs=("job_filing_number", "size"),
        job_months=("months", "sum"),
        events=("n_disc", "sum"),
        median_cost=("initial_cost", "median"))
    g["rate_per_100"] = 100 * g["events"] / g["job_months"]

    conv = mf[mf["conversion"] == 1]
    gc = conv.groupby("decile").agg(
        conv_n_jobs=("job_filing_number", "size"),
        conv_job_months=("months", "sum"),
        conv_events=("n_disc", "sum"))
    gc["conv_rate_per_100"] = 100 * gc["conv_events"] / gc["conv_job_months"]
    g = g.join(gc).reset_index()

    fig_csv = RM / "proactive_per_permit_figure.csv"
    g.to_csv(fig_csv, index=False)

    fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=160)
    bars = ax.bar(g["decile"], g["rate_per_100"], width=0.72, color=BLUE,
                  alpha=0.85, zorder=2, label="all jobs")

    show = g["conv_n_jobs"].fillna(0) >= 5
    (marks,) = ax.plot(
        g.loc[show, "decile"], g.loc[show, "conv_rate_per_100"],
        linestyle="none", marker="D", markersize=9, color=RED,
        markeredgecolor="white", markeredgewidth=1.2, zorder=4,
        label="office-to-resi conversions (relaxed definition)")
    halo = [pe.withStroke(linewidth=2.5, foreground=SURFACE)]
    for _, r in g.loc[show].iterrows():
        ax.annotate(f"n={int(r['conv_n_jobs'])}",
                    (r["decile"], r["conv_rate_per_100"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=RED,
                    path_effects=halo, zorder=5)

    style_ax(ax)
    ax.set_xticks(g["decile"])
    ax.set_xticklabels([f"D{int(d)}\n{money(c)}" for d, c in
                        zip(g["decile"], g["median_cost"])], fontsize=9)
    ax.set_xlabel("initial-cost decile (median estimated cost)",
                  fontsize=10.5)
    ax.set_ylabel("agency inspections per 100 job-months", fontsize=10.5)
    ax.set_title("DOB discretionary oversight concentrates on the costliest projects",
                 loc="left", fontsize=15, fontweight="bold", color=INK,
                 pad=30)
    ax.text(0, 1.03,
            "Agency-initiated discretionary + follow-up complaint events per 100 active job-months, "
            "DOB NOW jobs first permitted 2020–2026",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.legend(handles=[bars, marks], loc="upper left", frameon=False,
              fontsize=10)
    fig.tight_layout()
    out = ART / "proactive_per_permit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out}")
    print(f"Figure data -> {fig_csv}")
    return g


# ── verdict ──────────────────────────────────────────────────────────────

def print_verdict(est: pd.DataFrame) -> None:
    print("\n== Conversion verdict (PPML, offset log active months, "
          "FE nta x first-permit quarter, SE clustered on BBL) ==")
    for flag, flag_label in CONV_DEFS.items():
        d = est[(est["outcome"] == "n_disc") & (est["conv_def"] == flag)
                & (est["term"] == flag)].iloc[0]
        s = est[(est["outcome"] == "n_stat") & (est["conv_def"] == flag)
                & (est["term"] == flag)].iloc[0]
        print(f"  {flag_label}:")
        print(f"    discretionary: IRR {d['irr']:.2f} "
              f"[{d['irr_lo']:.2f}, {d['irr_hi']:.2f}] "
              f"(b={d['b']:+.3f}, se {d['se']:.3f}, n={d['n_obs']:,})")
        print(f"    statutory     : IRR {s['irr']:.2f} "
              f"[{s['irr_lo']:.2f}, {s['irr_hi']:.2f}] "
              f"(b={s['b']:+.3f}, se {s['se']:.3f}, n={s['n_obs']:,})")


def main() -> None:
    t0 = time.time()
    RM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    mf = build_model_frame()
    print("\nPPML models:")
    est = run_models(mf)
    out = RM / "proactive_monitoring_estimates.csv"
    est.to_csv(out, index=False)
    print(f"\nEstimates -> {out} ({len(est)} rows)")

    make_figure(mf)
    print_verdict(est)
    print(f"\nTotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
