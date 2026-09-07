"""Figures for the neighborhood post (data/analysis/blog_posts/post_neighborhoods_substack.md).

Fig 1  neighborhood_forest.png: per-demographic gradients on four measures (caller
       complaints, ECB citations, DOB violation records, hit rate among accessed caller
       inspections), building size held fixed (filled) vs building-stock controls added
       (hollow). Whiskers are 95% CIs clustered by tract.
Fig 2  neighborhood_decomposition.png: Gelbach decomposition of the any-caller-complaint
       gradient into the part running through each building-stock control group and the
       part left over.
Inputs: risk_models/neighborhood_gradients.csv, neighborhood_hitrate.csv,
        neighborhood_decomposition.csv (neighborhood_gradients.py).
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

matplotlib.use("Agg")
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
RM = config.DATA_DIR / "analysis" / "risk_models"
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, RED, INK2 = "#2a78d6", "#e34948", "#3a3a37"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})
ORDER = ["tract_poverty10", "tract_log_income_z", "tract_renter10", "tract_foreign10",
         "tract_overcrowd10", "tract_black10", "tract_hispanic10", "tract_asian10"]
LABEL = {"tract_poverty10": "Poverty rate, +10 pp", "tract_log_income_z": "Median income, +1 SD (log)",
         "tract_renter10": "Renter share, +10 pp", "tract_foreign10": "Foreign-born share, +10 pp",
         "tract_overcrowd10": "Overcrowded households, +10 pp", "tract_black10": "Black share, +10 pp",
         "tract_hispanic10": "Hispanic share, +10 pp", "tract_asian10": "Asian share, +10 pp"}
GROUP_LABEL = {"via_era": "building age", "via_value": "value within tract", "via_ownership": "ownership",
               "via_use_size": "use and size detail", "via_history": "prior violations", "direct": "left over"}
GROUP_COLOR = {"via_era": "#7a5195", "via_value": "#ef5675", "via_ownership": "#ffa600",
               "via_use_size": "#58a4b0", "via_history": "#003f5c", "direct": BASE}


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK, lw=0.9)


def forest():
    g = pd.read_csv(RM / "neighborhood_gradients.csv")
    g = g[g["entry"] == "bivariate"]
    h = pd.read_csv(RM / "neighborhood_hitrate.csv")
    panels = [
        ("n_caller", "Caller complaints\n(% difference)", "pct"),
        ("n_ecb_2020on", "ECB citations\n(% difference)", "pct"),
        ("n_dobviol_2020on", "DOB violation records\n(% difference)", "pct"),
        ("hit", "Violation found per accessed\ncaller inspection (pp)", "pp"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 7.0), sharey=True,
                             gridspec_kw=dict(left=0.19, right=0.985, top=0.74, bottom=0.11, wspace=0.16))
    y = np.arange(len(ORDER))[::-1]
    for ax, (key, title, scale) in zip(axes, panels):
        _style(ax)
        if key == "hit":
            tot = h[h.spec == "hit_inspector_fe"].set_index("term")
            dir_ = h[h.spec == "hit_inspector_fe_controls"].set_index("term")
            est_t, lo_t, hi_t = tot.loc[ORDER, "estimate"], tot.loc[ORDER, "ci_lo"], tot.loc[ORDER, "ci_hi"]
            est_d, lo_d, hi_d = dir_.loc[ORDER, "estimate"], dir_.loc[ORDER, "ci_lo"], dir_.loc[ORDER, "ci_hi"]
        else:
            sub = g[g.outcome == key]
            tot = sub[sub.spec == "total"].set_index("term"); dir_ = sub[sub.spec == "direct"].set_index("term")
            est_t, lo_t, hi_t = tot.loc[ORDER, "pct_change"], tot.loc[ORDER, "pct_lo"], tot.loc[ORDER, "pct_hi"]
            est_d, lo_d, hi_d = dir_.loc[ORDER, "pct_change"], dir_.loc[ORDER, "pct_lo"], dir_.loc[ORDER, "pct_hi"]
        off = 0.17
        ax.hlines(y + off, lo_t, hi_t, color=BLUE, lw=1.6)
        ax.plot(est_t, y + off, "o", ms=7, color=BLUE, mec=BLUE, zorder=3)
        ax.hlines(y - off, lo_d, hi_d, color=INK2, lw=1.6)
        ax.plot(est_d, y - off, "o", ms=7, mfc=SURFACE, mec=INK2, mew=1.6, zorder=3)
        ax.set_title(title, fontsize=11, loc="left", color=INK2, pad=10)
        lo, hi = min(lo_t.min(), lo_d.min()), max(hi_t.max(), hi_d.max())
        pad = (hi - lo) * 0.12
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yticks(y)
    axes[0].set_yticklabels([LABEL[t] for t in ORDER], fontsize=10.5)
    axes[0].tick_params(axis="y", length=0)
    fig.text(0.02, 0.945, "How complaints, citations, and violations vary with tract demographics",
             fontsize=15, weight="bold", color=INK)
    fig.text(0.02, 0.895, "Filled: building size held fixed only.   Hollow: building age, value within tract, ownership, use, "
             "and prior violations also held fixed.   Whiskers: 95% CIs clustered by tract.",
             fontsize=10, color=MUTED)
    fig.text(0.02, 0.86, "Residential lots in PLUTO, 2020 to May 2026; hit-rate panel is complaint level with inspector fixed effects.",
             fontsize=10, color=MUTED)
    out = ART / "neighborhood_forest.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


def decomposition():
    d = pd.read_csv(RM / "neighborhood_decomposition.csv")
    comps = ["via_era", "via_value", "via_ownership", "via_use_size", "via_history", "direct"]
    fig, ax = plt.subplots(figsize=(11.5, 7.0), gridspec_kw=dict(left=0.27, right=0.98, top=0.80, bottom=0.21))
    _style(ax)
    y = np.arange(len(ORDER))[::-1]
    for i, term in enumerate(ORDER):
        row = d[d.term == term].set_index("component")["points"]
        pos = neg = 0.0
        for c in comps:
            v = float(row[c])
            if v >= 0:
                ax.barh(y[i], v, left=pos, color=GROUP_COLOR[c], edgecolor=SURFACE, lw=1.2, height=0.62); pos += v
            else:
                ax.barh(y[i], v, left=neg, color=GROUP_COLOR[c], edgecolor=SURFACE, lw=1.2, height=0.62); neg += v
        tot = float(row["total"])
        ax.plot([tot], [y[i]], marker="|", ms=22, mew=2.2, color=INK, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[t] for t in ORDER], fontsize=10.5); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change in the probability of any caller complaint, 2020 to May 2026 (percentage points)", fontsize=10)
    handles = [matplotlib.patches.Patch(color=GROUP_COLOR[c], label=GROUP_LABEL[c]) for c in comps]
    handles.append(matplotlib.lines.Line2D([], [], marker="|", ms=14, mew=2, color=INK, ls="", label="total difference"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False, ncol=4)
    fig.text(0.02, 0.945, "How much of each neighborhood difference in complaints is associated with building characteristics", fontsize=15, weight="bold", color=INK)
    fig.text(0.02, 0.895, "Exact decomposition of the gap between the size-only and the fully controlled estimate (Gelbach 2016); "
             "colored segments sum to that gap,", fontsize=10, color=MUTED)
    fig.text(0.02, 0.86, "grey is what remains with every building characteristic held fixed. Linear probability model, size, commercial-unit, and borough fixed effects.",
             fontsize=10, color=MUTED)
    out = ART / "neighborhood_decomposition.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


def hpd_forest():
    g = pd.read_csv(RM / "neighborhood_hpd_gradients.csv")
    g = g[(g["entry"] == "bivariate") & (g["sample"] == "all lots")]
    h = pd.read_csv(RM / "neighborhood_hpd_hitrate.csv")
    panels = [
        ("n_hpd_complaints", "HPD complaints\n(% difference)", "pct"),
        ("n_hpd_violations", "HPD violations\n(% difference)", "pct"),
        ("hit", "HPD violation issued per\ninspected problem (pp)", "pp"),
        ("noaccess", "HPD inspection ended\nwith no access (pp)", "pp"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 7.0), sharey=True,
                             gridspec_kw=dict(left=0.19, right=0.985, top=0.74, bottom=0.11, wspace=0.16))
    y = np.arange(len(ORDER))[::-1]
    for ax, (key, title, scale) in zip(axes, panels):
        _style(ax)
        if key in ("hit", "noaccess"):
            b, c = (("hpd_hit_base", "hpd_hit_controls") if key == "hit" else ("hpd_noaccess_base", "hpd_noaccess_controls"))
            tot = h[h.spec == b].set_index("term"); dir_ = h[h.spec == c].set_index("term")
            est_t, lo_t, hi_t = tot.loc[ORDER, "estimate"], tot.loc[ORDER, "ci_lo"], tot.loc[ORDER, "ci_hi"]
            est_d, lo_d, hi_d = dir_.loc[ORDER, "estimate"], dir_.loc[ORDER, "ci_lo"], dir_.loc[ORDER, "ci_hi"]
        else:
            sub = g[g.outcome == key]
            tot = sub[sub.spec == "total"].set_index("term"); dir_ = sub[sub.spec == "direct"].set_index("term")
            est_t, lo_t, hi_t = tot.loc[ORDER, "pct_change"], tot.loc[ORDER, "pct_lo"], tot.loc[ORDER, "pct_hi"]
            est_d, lo_d, hi_d = dir_.loc[ORDER, "pct_change"], dir_.loc[ORDER, "pct_lo"], dir_.loc[ORDER, "pct_hi"]
        off = 0.17
        ax.hlines(y + off, lo_t, hi_t, color=BLUE, lw=1.6)
        ax.plot(est_t, y + off, "o", ms=7, color=BLUE, mec=BLUE, zorder=3)
        ax.hlines(y - off, lo_d, hi_d, color=INK2, lw=1.6)
        ax.plot(est_d, y - off, "o", ms=7, mfc=SURFACE, mec=INK2, mew=1.6, zorder=3)
        ax.set_title(title, fontsize=11, loc="left", color=INK2, pad=10)
        lo, hi = min(lo_t.min(), lo_d.min()), max(hi_t.max(), hi_d.max())
        pad = (hi - lo) * 0.12
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yticks(y)
    axes[0].set_yticklabels([LABEL[t] for t in ORDER], fontsize=10.5)
    axes[0].tick_params(axis="y", length=0)
    fig.text(0.02, 0.945, "The same comparisons in HPD's housing-maintenance system", fontsize=15, weight="bold", color=INK)
    fig.text(0.02, 0.895, "Filled: building size held fixed only.   Hollow: building age, value within tract, ownership, use, and prior violations "
             "also held fixed.   Whiskers: 95% CIs clustered by tract.", fontsize=10, color=MUTED)
    fig.text(0.02, 0.86, "HPD complaints and violations per lot, 2020 to May 2026; the two right panels are problem level with major-category fixed effects.",
             fontsize=10, color=MUTED)
    out = ART / "neighborhood_hpd_forest.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    ART.mkdir(parents=True, exist_ok=True)
    forest()
    decomposition()
    if (RM / "neighborhood_hpd_gradients.csv").exists():
        hpd_forest()
