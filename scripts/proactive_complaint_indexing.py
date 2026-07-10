#!/usr/bin/env python3
"""
Hypothesis 4 (complaint indexing): is DOB proactive enforcement indexed on
local caller complaint volume, or does it just follow the permit pipeline?

Tract x month PPML of agency-initiated complaint counts on log1p caller
complaints in prior months, with tract and calendar-month fixed effects.
Spine: data/analysis/proactive/tract_month.csv.gz (2,322 tracts x 77
months, 2020-01..2026-05; lags consume the first three months, so the
estimation window is 74 outcome months, 2020-04..2026-05, common to all
specs).

Outcomes: proactive_total (headline), proactive_discretionary_field
(sweeps/compliance/enforcement where DOB chooses where to go), and
proactive_statutory_periodic as a built-in placebo. Statutory cycles
(7K/7F/6V local-law tracking) are set by statute, so their elasticity to
last month's caller volume should be ~0; the script prints an explicit
pass/fail on that.

Specs per outcome (SEs clustered by tract; headline re-estimated with
two-way tract + month clustering):
  lag1_nojobs   log1p(caller t-1)                          | tract + month
  lag1_jobs     ... + log1p(active jobs t)   <- HEADLINE
  sum3_nojobs   log1p(caller t-1 + t-2 + t-3)
  sum3_jobs     ... + log1p(active jobs t)
  dist3_jobs    log1p caller at t-1, t-2, t-3 separately + jobs control

The horse race is lag1_nojobs vs lag1_jobs: if "indexing on complaints"
is really co-location with construction, the caller elasticity should
collapse once the active-jobs control enters.

Panel note (state wherever these numbers are quoted): agency-initiated
events geo-match to tracts at 94.1% (caller complaints 97.9%), so
proactive counts are slightly undercounted. Tract fixed effects absorb
level undercounting; the elasticity is biased only if match rates trend
differentially within tracts.

Writes  data/analysis/risk_models/proactive_indexing_estimates.csv
        data/analysis/blog_posts/artifacts/proactive_indexing.png

Run:  /private/tmp/pyfix_venv/bin/python scripts/proactive_complaint_indexing.py
"""

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, "data", "analysis", "proactive", "tract_month.csv.gz")
OUT_CSV = os.path.join(ROOT, "data", "analysis", "risk_models",
                       "proactive_indexing_estimates.csv")
ART_DIR = os.path.join(ROOT, "data", "analysis", "blog_posts", "artifacts")
OUT_PNG = os.path.join(ART_DIR, "proactive_indexing.png")

GEO_NOTE = ("agency events geo-match to tracts at 94.1% (callers 97.9%); "
            "proactive counts slightly undercounted")

OUTCOMES = {
    "proactive_total": "all agency-initiated complaints",
    "proactive_discretionary_field": "discretionary field programs",
    "proactive_statutory_periodic": "statutory periodic cycles (placebo)",
}

# model_name -> (RHS terms, short description)
SPECS = {
    "lag1_nojobs": (["log_caller_l1"],
                    "log1p caller complaints t-1, no jobs control"),
    "lag1_jobs": (["log_caller_l1", "log_jobs"],
                  "HEADLINE: log1p caller t-1 + log1p active jobs t"),
    "sum3_nojobs": (["log_caller_sum3"],
                    "log1p caller complaints summed t-1..t-3, no jobs control"),
    "sum3_jobs": (["log_caller_sum3", "log_jobs"],
                  "log1p caller sum t-1..t-3 + log1p active jobs t"),
    "dist3_jobs": (["log_caller_l1", "log_caller_l2", "log_caller_l3",
                    "log_jobs"],
                   "distributed lags t-1..t-3 + log1p active jobs t"),
}

# ── house figure style (constants from scripts/make_descriptive_figures.py) ──
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
ZERO_C = "#b9b7ac"

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


# ── panel construction ───────────────────────────────────────────────────

def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL)
    df = df.sort_values(["bct2020", "month"]).reset_index(drop=True)
    g = df.groupby("bct2020")
    for k in (1, 2, 3):
        df[f"caller_l{k}"] = g["caller_complaints"].shift(k)
    df["caller_sum3"] = df["caller_l1"] + df["caller_l2"] + df["caller_l3"]

    df = df.dropna(subset=["caller_l3"]).reset_index(drop=True)
    n_tracts, n_months = df["bct2020"].nunique(), df["month"].nunique()
    assert len(df) == n_tracts * n_months, "panel lost balance after lagging"
    assert (df["month"].min(), df["month"].max()) == ("2020-04", "2026-05")

    for c in ("caller_l1", "caller_l2", "caller_l3", "caller_sum3"):
        df[f"log_{c}"] = np.log1p(df[c])
    df["log_jobs"] = np.log1p(df["active_jobs"])
    print(f"Estimation panel: {n_tracts:,} tracts x {n_months} months "
          f"({len(df):,} rows), outcomes 2020-04..2026-05")
    return df


# ── estimation ───────────────────────────────────────────────────────────

def run_model(df, outcome, model_name, rhs, vcov, vcov_name):
    fml = f"{outcome} ~ {' + '.join(rhs)} | bct2020 + month"
    m = pf.fepois(fml, data=df, vcov=vcov)
    ci = m.confint()
    rows = []
    for term in m.coef().index:
        b = float(m.coef()[term])
        rows.append({
            "outcome": outcome,
            "model": model_name,
            "vcov": vcov_name,
            "term": term,
            "estimate": b,
            "std_error": float(m.se()[term]),
            "t_value": float(m.tstat()[term]),
            "pr(>|t|)": float(m.pvalue()[term]),
            "25pct": float(ci.loc[term].iloc[0]),
            "975pct": float(ci.loc[term].iloc[1]),
            "n": int(m._N),
            "pct_per_doubling": 100 * (2 ** b - 1),
            "spec": SPECS[model_name][1],
            "note": GEO_NOTE,
        })
    return rows


def estimate_all(df) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for model_name, (rhs, _) in SPECS.items():
            t0 = time.time()
            rows += run_model(df, outcome, model_name, rhs,
                              {"CRV1": "bct2020"}, "CRV1 tract")
            print(f"  {outcome:>30} {model_name:<12} "
                  f"({time.time() - t0:4.1f}s)")
        # two-way clustering robustness on the headline spec
        rows += run_model(df, outcome, "lag1_jobs", SPECS["lag1_jobs"][0],
                          {"CRV1": "bct2020+month"}, "CRV1 tract+month")
        print(f"  {outcome:>30} lag1_jobs    (two-way tract+month)")
    return pd.DataFrame(rows)


def print_horse_race(est: pd.DataFrame) -> None:
    def get(outcome, model, term, vcov="CRV1 tract"):
        r = est[(est.outcome == outcome) & (est.model == model)
                & (est.term == term) & (est.vcov == vcov)].iloc[0]
        return r

    print("\n== Horse race: caller elasticity with vs without jobs control ==")
    print(f"{'outcome':>30} {'spec':<12} {'elasticity':>10} {'95% CI':>20}")
    for outcome in OUTCOMES:
        for model, term in [("lag1_nojobs", "log_caller_l1"),
                            ("lag1_jobs", "log_caller_l1"),
                            ("sum3_nojobs", "log_caller_sum3"),
                            ("sum3_jobs", "log_caller_sum3")]:
            r = get(outcome, model, term)
            print(f"{outcome:>30} {model:<12} {r.estimate:10.4f} "
                  f"[{r['25pct']:7.4f}, {r['975pct']:7.4f}]")

    print("\n== Two-way (tract+month) clustering, headline spec ==")
    for outcome in OUTCOMES:
        r = get(outcome, "lag1_jobs", "log_caller_l1", "CRV1 tract+month")
        print(f"{outcome:>30} lag1_jobs    {r.estimate:10.4f} "
              f"[{r['25pct']:7.4f}, {r['975pct']:7.4f}]")

    print("\n== Statutory placebo check ==")
    r = get("proactive_statutory_periodic", "lag1_jobs", "log_caller_l1")
    verdict = ("PASSES (CI covers 0)"
               if r["25pct"] <= 0 <= r["975pct"] else
               "NOT CLEAN: CI excludes 0 -- statutory cycles co-move with "
               "caller volume; say so wherever the headline is quoted")
    print(f"  statutory elasticity {r.estimate:.4f} "
          f"[{r['25pct']:.4f}, {r['975pct']:.4f}] -> {verdict}")
    d = get("proactive_discretionary_field", "lag1_jobs", "log_caller_l1")
    print(f"  discretionary {d.estimate:.4f} [{d['25pct']:.4f}, "
          f"{d['975pct']:.4f}] vs statutory {r.estimate:.4f}: "
          f"gap {d.estimate - r.estimate:+.4f} (CIs "
          f"{'overlap' if d['25pct'] <= r['975pct'] and r['25pct'] <= d['975pct'] else 'disjoint'})"
          " -- if the placebo is not ~0, the caller elasticity reads as "
          "shared local-activity shocks, not discretionary indexing")
    print(f"\nPanel note: {GEO_NOTE}.")


# ── figure ───────────────────────────────────────────────────────────────

def residualize(df, var, controls):
    """FE residuals via pyfixest (Frisch-Waugh with tract + month FEs)."""
    rhs = " + ".join(controls) if controls else "1"
    return pf.feols(f"{var} ~ {rhs} | bct2020 + month", data=df).resid()


def binned(rx, ry, n_bins=20):
    q = pd.qcut(rx, n_bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"q": q, "rx": rx, "ry": ry}).groupby("q").mean()
    return d["rx"].to_numpy(), d["ry"].to_numpy()


def make_figure(df, est: pd.DataFrame) -> None:
    panels = [
        ("lag1_nojobs", [], "Without permit control",
         "tract + month fixed effects only"),
        ("lag1_jobs", ["log_jobs"], "With permit control",
         "also partialling out log(1 + active jobs)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=160, sharey=True)

    for ax, (model, controls, ptitle, psub) in zip(axes, panels):
        rx = residualize(df, "log_caller_l1", controls)
        ry = residualize(df, "proactive_total", controls)
        bx, by = binned(rx, ry)
        slope, icpt = np.polyfit(rx, ry, 1)
        xs = np.array([bx.min(), bx.max()])
        ax.axhline(0, color=ZERO_C, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axvline(0, color=ZERO_C, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.plot(xs, icpt + slope * xs, color=RED, linewidth=1.6, zorder=2)
        ax.scatter(bx, by, s=42, color=BLUE, zorder=3)
        style_ax(ax)

        r = est[(est.outcome == "proactive_total") & (est.model == model)
                & (est.term == "log_caller_l1")
                & (est.vcov == "CRV1 tract")].iloc[0]
        ax.text(0.03, 0.955,
                f"PPML elasticity {r.estimate:.3f}\n"
                f"95% CI [{r['25pct']:.3f}, {r['975pct']:.3f}]",
                transform=ax.transAxes, fontsize=11.5, color=INK,
                va="top", linespacing=1.5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
        ax.set_title(ptitle, loc="left", fontsize=12.5, fontweight="bold",
                     color=INK, pad=20)
        ax.text(0, 1.03, psub, transform=ax.transAxes, fontsize=10,
                color=MUTED)
        ax.set_xlabel("log(1 + caller complaints, t−1), residualized",
                      fontsize=10.5)
    axes[0].set_ylabel("proactive complaints per tract-month, residualized",
                       fontsize=10.5)

    fig.suptitle("Does proactive enforcement follow last month's callers?",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.925,
             "agency-initiated DOB complaints per tract-month vs caller "
             "complaints the month before · 2,322 tracts × 74 months, "
             "Apr 2020–May 2026", fontsize=10.5, color=MUTED)
    fig.text(0.01, 0.015,
             "Dots: means of 20 equal-count bins of tract-month residuals "
             "(both axes residualized as labeled). Line: OLS fit through the "
             "underlying observations.\nAnnotated elasticities: PPML with "
             "the same controls, SEs clustered by tract. Agency events "
             "geo-match to tracts at 94.1% (callers 97.9%), so proactive "
             "counts are slightly undercounted.",
             fontsize=8.8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=[0, 0.055, 1, 0.90])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {OUT_PNG}")


def main() -> None:
    t0 = time.time()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)

    df = load_panel()
    est = estimate_all(df)
    est.to_csv(OUT_CSV, index=False)
    print(f"\nEstimates ({len(est)} rows) -> {OUT_CSV}")

    print_horse_race(est)
    make_figure(df, est)
    print(f"\nTOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
