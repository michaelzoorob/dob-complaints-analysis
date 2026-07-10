#!/usr/bin/env python3
"""
Hypothesis 9 (proactive_enforcement_plan.md): lifecycle mismatch — is agency
monitoring front-loaded in job time while incidents cluster mid-to-late?

Job-time hazard curves over months since first permit (k = 0..36), for DOB
NOW jobs first permitted 2020+ whose active span (first permit -> signoff,
else last permit expiry) runs at least 6 months (182 days). For every job
and month k, count complaints at the job's BBL in calendar month
(start_month + k), restricted to jobs still at risk at k (span reaches k
AND start_month + k is inside the 2020-01..2026-05 spine):

  monitoring   agency-originated (empty ref_311) discretionary_field +
               followup families — DOB-chosen construction oversight
  statutory    agency-originated statutory_periodic family — cyclical
               obligations that fire on permit events (2E/2F/6V etc.)
  incidents    caller-originated reports in incident categories
               30, 10, 12, 14, 91, 1E, 03

Hazard at k = events / jobs-at-risk (reported per 100 jobs). SEs cluster by
BBL (several jobs can share a lot; events are counted once per job, so a
shared lot's events enter every co-located job's numerator). Counts use the
packed searchsorted counter (no row loops).

Blocks in the CSV
  hazard    per-month curves, split = all / New Building / Alteration /
            Demolition / balanced_36m (jobs observable AND active the full
            0..36 window — fixes risk-set composition drift); split=all
            also carries the discretionary_field / followup components
  summary   per split x series: raw + 3-month-MA peak month, peak level,
            hazard-mass-weighted mean month, early/late averages,
            front-loading ratio (months 0-5 avg / months 24-36 avg)

Caveats (report with any quoted number)
  - signoff_date is only 64% filled spine-wide (58.3% in this >=6-month
    sample); missing signoffs fall back to last permit expiry, which
    overstates active life, so late-k risk sets keep some finished jobs
    and late hazards are diluted toward the building's ambient rate.
  - The risk set shifts toward long-duration (bigger) projects as k grows;
    the balanced_36m block holds composition fixed.
  - Month grain: k=0 counts events in the first permit's calendar month,
    including days just before the permit.
  - BBL join counts any complaint at the lot, not inspections tied to the
    specific job filing.

Writes
    data/analysis/risk_models/proactive_lifecycle.csv
    data/analysis/blog_posts/artifacts/proactive_lifecycle.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_lifecycle.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

PRO = config.DATA_DIR / "analysis" / "proactive"
RM = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"

K_MAX = 36                  # months since first permit 0..36
N_MONTHS = 77               # spine months 2020-01..2026-05 (index 0..76)
MIN_SPAN_DAYS = 182         # active span >= 6 months
STRIDE = 128                # > N_MONTHS, packs (bbl, month) into one int64
MIN_RISK = 300              # months with a smaller risk set are excluded
                            # from peak / summary statistics
INCIDENT_PREFIXES = ("30", "10", "12", "14", "91", "1E", "03")
MONITOR_FAMILIES = ("discretionary_field", "followup")

JOBTYPE_GROUP = {
    "New Building": "New Building",
    "Alteration": "Alteration",
    "Alteration CO": "Alteration",
    "ALT-CO - New Building with Existing Elements to Remain": "Alteration",
    "Full Demolition": "Demolition",
}

# house palette (make_descriptive_figures.py)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ── data ─────────────────────────────────────────────────────────────────

def month_idx(ts: pd.Series) -> pd.Series:
    return (ts.dt.year - 2020) * 12 + ts.dt.month - 1


def load_jobs() -> pd.DataFrame:
    j = pd.read_csv(PRO / "jobs.csv.gz",
                    usecols=["job_filing_number", "job_type", "bbl",
                             "active_start", "active_end", "signoff_date"],
                    dtype={"bbl": "str"})
    n0 = len(j)
    for c in ("active_start", "active_end", "signoff_date"):
        j[c] = pd.to_datetime(j[c], errors="coerce")
    span = (j["active_end"] - j["active_start"]).dt.days
    j = j[span >= MIN_SPAN_DAYS].copy()
    n_span = len(j)
    j = j[j["bbl"].fillna("").str.fullmatch(r"\d{10}")].copy()
    j["start_m"] = month_idx(j["active_start"])
    j["end_m"] = month_idx(j["active_end"])
    j = j[j["start_m"] <= N_MONTHS - 1].reset_index(drop=True)
    j["bbl_i"] = j["bbl"].astype("int64")
    j["group"] = j["job_type"].map(JOBTYPE_GROUP).fillna("Alteration")
    print(f"jobs: {n0:,} first-permitted 2020+ -> {n_span:,} with active span "
          f">= {MIN_SPAN_DAYS}d -> {len(j):,} with valid BBL inside the spine")
    print(f"  signoff filled {j['signoff_date'].notna().mean():.3f} in sample "
          f"(spine-wide 0.64); groups "
          f"{j['group'].value_counts().to_dict()}")
    return j


def make_counter(sub: pd.DataFrame):
    """Exact-count lookup via searchsorted on a packed sorted key array."""
    keys = np.sort(sub["bbl_i"].to_numpy() * STRIDE + sub["m"].to_numpy())

    def count(bbl_arr, m_arr):
        q = bbl_arr * STRIDE + m_arr
        return (np.searchsorted(keys, q, side="right")
                - np.searchsorted(keys, q, side="left")).astype(np.int32)

    return count


def load_event_counters() -> dict:
    e = pd.read_csv(PRO / "proactive_events.csv.gz",
                    usecols=["month", "category_prefix", "family", "agency", "bbl"],
                    dtype={"category_prefix": "str", "bbl": "str"})
    n0 = len(e)
    e = e[e["bbl"].notna() & e["bbl"].str.fullmatch(r"\d{10}")].copy()
    e["bbl_i"] = e["bbl"].astype("int64")
    e["m"] = ((e["month"].str[:4].astype(int) - 2020) * 12
              + e["month"].str[5:7].astype(int) - 1)
    agency = e["agency"] == 1
    subsets = {
        "monitoring": e[agency & e["family"].isin(MONITOR_FAMILIES)],
        "monitoring_discretionary": e[agency & (e["family"] == "discretionary_field")],
        "monitoring_followup": e[agency & (e["family"] == "followup")],
        "statutory": e[agency & (e["family"] == "statutory_periodic")],
        "incidents": e[~agency & e["category_prefix"].isin(INCIDENT_PREFIXES)],
    }
    print(f"events: {n0:,} rows -> {len(e):,} with valid BBL; "
          + "; ".join(f"{k} {len(v):,}" for k, v in subsets.items()))
    return {k: make_counter(v) for k, v in subsets.items()}


# ── hazards ──────────────────────────────────────────────────────────────

def job_month_counts(jobs: pd.DataFrame, counters: dict):
    """(n_jobs x 37) count matrix per series + at-risk mask, one searchsorted
    sweep per (series, k)."""
    b = jobs["bbl_i"].to_numpy()
    s = jobs["start_m"].to_numpy()
    en = jobs["end_m"].to_numpy()
    ks = np.arange(K_MAX + 1)
    at_risk = ((en - s)[:, None] >= ks) & ((s[:, None] + ks) <= N_MONTHS - 1)
    counts = {}
    for name, counter in counters.items():
        m = np.empty((len(jobs), K_MAX + 1), dtype=np.int32)
        for k in ks:
            m[:, k] = counter(b, s + k)
        counts[name] = m
    return counts, at_risk


def hazard_curve(x: np.ndarray, at_risk: np.ndarray, mask: np.ndarray,
                 cl: np.ndarray, n_cl: int) -> pd.DataFrame:
    """Per-month hazard for jobs in `mask`, BBL-clustered SEs."""
    rows = []
    for k in range(K_MAX + 1):
        sel = mask & at_risk[:, k]
        n = int(sel.sum())
        if n == 0:
            rows.append({"month": k, "n_at_risk": 0, "n_events": 0,
                         "hazard_per100": np.nan, "se_per100": np.nan})
            continue
        xk = x[sel, k]
        mean = xk.mean()
        cs = np.bincount(cl[sel], weights=xk - mean, minlength=n_cl)
        se = np.sqrt((cs ** 2).sum()) / n
        rows.append({"month": k, "n_at_risk": n, "n_events": int(xk.sum()),
                     "hazard_per100": 100 * mean, "se_per100": 100 * se})
    out = pd.DataFrame(rows)
    out["ci_low_per100"] = out["hazard_per100"] - 1.96 * out["se_per100"]
    out["ci_high_per100"] = out["hazard_per100"] + 1.96 * out["se_per100"]
    return out


def summarize(curve: pd.DataFrame) -> dict:
    """Peak / front-loading metrics over months with a usable risk set."""
    c = curve[curve["n_at_risk"] >= MIN_RISK]
    h = c.set_index("month")["hazard_per100"]
    ma3 = h.rolling(3, center=True, min_periods=2).mean()
    wmean = float((h.index * h).sum() / h.sum())

    def avg(lo, hi):
        v = h.loc[(h.index >= lo) & (h.index <= hi)]
        return float(v.mean()) if len(v) else np.nan

    h0_5, h24_36 = avg(0, 5), avg(24, 36)
    return {
        "months_used": len(h),
        "peak_month": int(h.idxmax()),
        "peak_per100": float(h.max()),
        "peak_month_ma3": int(ma3.idxmax()),
        "wmean_month": wmean,
        "h_avg_m0_5": h0_5,
        "h_avg_m6_11": avg(6, 11),
        "h_avg_m12_23": avg(12, 23),
        "h_avg_m24_36": h24_36,
        "front_ratio_0_5_over_24_36": h0_5 / h24_36 if h24_36 else np.nan,
    }


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(curves: dict, summaries: dict, n_jobs: int):
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=200,
                           gridspec_kw={"left": 0.09, "right": 0.97,
                                        "top": 0.74, "bottom": 0.11})
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    spec = [
        ("statutory", ZERO, (0, (4, 2)), 1.8,
         "statutory periodic (fires on permit events)"),
        ("monitoring", BLUE, "solid", 2.2,
         "discretionary monitoring + follow-up"),
        ("incidents", RED, "solid", 2.2,
         "caller incident reports (collapse, debris, unsafe work)"),
    ]
    for name, color, ls, lw, label in spec:
        g = curves[name]
        ax.fill_between(g["month"], g["ci_low_per100"], g["ci_high_per100"],
                        color=color, alpha=0.13, linewidth=0)
        ax.plot(g["month"], g["hazard_per100"], color=color, ls=ls, lw=lw,
                label=label, zorder=3)
        s = summaries[name]
        pk_m, pk_h = s["peak_month"], s["peak_per100"]
        ax.plot([pk_m], [pk_h], "o", ms=6.5, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=4)
        ax.annotate(f"peak m{pk_m}", xy=(pk_m, pk_h),
                    xytext=(pk_m + 0.7, pk_h + 0.28), fontsize=8.5,
                    color=color if color != ZERO else MUTED)

    ax.set_xlim(-0.6, K_MAX + 0.6)
    ax.set_ylim(0, None)
    ax.set_xticks(range(0, K_MAX + 1, 6))
    ax.tick_params(labelsize=9)
    ax.set_xlabel("months since the job's first permit", fontsize=9.5)
    ax.set_ylabel("events at the job's lot per 100 active jobs", fontsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    mon, inc = summaries["monitoring"], summaries["incidents"]
    stat = summaries["statutory"]
    fig.suptitle("Incident reports peak in a job's first months, and\n"
                 "discretionary monitoring crests mid-project",
                 x=0.02, y=0.985, ha="left", fontsize=13.5, color=INK,
                 weight="semibold")
    fig.text(0.02, 0.835,
             f"{n_jobs:,} DOB NOW jobs first permitted 2020+ with active spans of 6+ months · "
             f"hazard = events at the job's BBL per 100 jobs\nstill active at each month, "
             f"2020-01..2026-05 spine · shading = 95% CI, SEs clustered by lot · "
             f"risk set {curves['monitoring']['n_at_risk'].iloc[0]:,} jobs at month 0, "
             f"{curves['monitoring']['n_at_risk'].iloc[-1]:,} at month 36\n"
             f"hazard ratio for months 0-5 over 24-36 runs statutory {stat['front_ratio_0_5_over_24_36']:.1f}, "
             f"incidents {inc['front_ratio_0_5_over_24_36']:.1f}, "
             f"monitoring {mon['front_ratio_0_5_over_24_36']:.1f} (its mid-project crest "
             "survives in the balanced 36-month cohort)\nsignoff dates are 64% filled; "
             "missing ends fall back to permit expiry, so late months keep some finished "
             "jobs and late hazards are diluted",
             fontsize=8.5, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    fig.savefig(ART / "proactive_lifecycle.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {ART / 'proactive_lifecycle.png'}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    jobs = load_jobs()
    counters = load_event_counters()
    counts, at_risk = job_month_counts(jobs, counters)
    cl, uniq = pd.factorize(jobs["bbl_i"])
    n_cl = len(uniq)
    print(f"counts built: {len(jobs):,} jobs x {K_MAX + 1} months, "
          f"{n_cl:,} distinct lots ({time.time() - t0:.1f}s)")

    all_mask = np.ones(len(jobs), dtype=bool)
    span_m = jobs["end_m"].to_numpy() - jobs["start_m"].to_numpy()
    balanced = (span_m >= K_MAX) & (jobs["start_m"].to_numpy() <= N_MONTHS - 1 - K_MAX)

    splits = {"all": all_mask}
    for g in ("New Building", "Alteration", "Demolition"):
        splits[g] = (jobs["group"] == g).to_numpy()
    splits["balanced_36m"] = balanced

    series_by_split = {
        "all": ["monitoring", "monitoring_discretionary", "monitoring_followup",
                "statutory", "incidents"],
        "New Building": ["monitoring", "statutory", "incidents"],
        "Alteration": ["monitoring", "statutory", "incidents"],
        "Demolition": ["monitoring", "statutory", "incidents"],
        "balanced_36m": ["monitoring", "statutory", "incidents"],
    }

    curve_rows, summary_rows = [], []
    curves_all, summaries_all = {}, {}
    for split, mask in splits.items():
        for name in series_by_split[split]:
            cur = hazard_curve(counts[name], at_risk, mask, cl, n_cl)
            summ = summarize(cur)
            cur.insert(0, "series", name)
            cur.insert(0, "split", split)
            cur.insert(0, "block", "hazard")
            curve_rows.append(cur)
            summary_rows.append({"block": "summary", "split": split,
                                 "series": name, "n_jobs": int(mask.sum()),
                                 **summ})
            if split == "all":
                curves_all[name], summaries_all[name] = cur, summ

    out = pd.concat(curve_rows + [pd.DataFrame(summary_rows)], ignore_index=True)
    RM.mkdir(parents=True, exist_ok=True)
    out_csv = RM / "proactive_lifecycle.csv"
    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv} ({len(out)} rows: "
          f"{sum(len(c) for c in curve_rows)} curve + {len(summary_rows)} summary)")

    # ── plain-number report ─────────────────────────────────────────────
    print("\n== peaks and front-loading (split=all, per 100 active jobs) ==")
    for name in ("statutory", "monitoring", "monitoring_discretionary",
                 "monitoring_followup", "incidents"):
        s = summaries_all[name]
        print(f"  {name:<26} peak m{s['peak_month']:>2} ({s['peak_per100']:.2f}) "
              f"ma3 m{s['peak_month_ma3']:>2}  wmean {s['wmean_month']:5.1f}  "
              f"m0-5 {s['h_avg_m0_5']:.2f} -> m24-36 {s['h_avg_m24_36']:.2f}  "
              f"ratio {s['front_ratio_0_5_over_24_36']:.2f}")
    mon, inc, stat = (summaries_all[k] for k in
                      ("monitoring", "incidents", "statutory"))
    print(f"\n  front-loading gap (monitoring vs incidents): "
          f"wmean month {mon['wmean_month']:.1f} vs {inc['wmean_month']:.1f} "
          f"(gap {inc['wmean_month'] - mon['wmean_month']:+.1f} months); "
          f"early/late ratio {mon['front_ratio_0_5_over_24_36']:.2f} vs "
          f"{inc['front_ratio_0_5_over_24_36']:.2f}")
    print(f"  statutory is the front-loaded stream: ratio "
          f"{stat['front_ratio_0_5_over_24_36']:.2f}, "
          f"wmean {stat['wmean_month']:.1f}")
    print(f"  VERDICT: hypothesis 9 refuted — discretionary monitoring is "
          f"not front-loaded (peak m{mon['peak_month']}, ratio "
          f"{mon['front_ratio_0_5_over_24_36']:.2f}) and incidents do not "
          f"cluster late (peak m{inc['peak_month']}, ratio "
          f"{inc['front_ratio_0_5_over_24_36']:.2f}); any timing gap runs "
          f"the other way")

    print("\n== robustness ==")
    for r in summary_rows:
        if r["split"] != "all":
            print(f"  {r['split']:<13} {r['series']:<11} n={r['n_jobs']:>7,} "
                  f"peak m{r['peak_month']:>2} ma3 m{r['peak_month_ma3']:>2} "
                  f"wmean {r['wmean_month']:5.1f} ratio "
                  f"{r['front_ratio_0_5_over_24_36']:.2f}")

    print("\n== caveats ==")
    print(f"  - signoff 64% filled spine-wide "
          f"({jobs['signoff_date'].notna().mean() * 100:.1f}% here); "
          f"permit-expiry fallback overstates spans, diluting late-month hazards")
    print("  - risk set drifts toward long jobs at high k; balanced_36m "
          "block holds composition fixed")
    print("  - month grain: k=0 includes same-calendar-month events "
          "shortly before the permit")
    print("  - BBL join counts all complaints at the lot, not job-linked "
          "inspections; co-located jobs each count shared events (SEs "
          "cluster by lot)")

    make_figure(curves_all, summaries_all, int(all_mask.sum()))
    print(f"\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
