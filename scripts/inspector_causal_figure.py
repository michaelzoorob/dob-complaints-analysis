"""
Figures for the inspector Substack post (post_inspector_substack.md).

Produces:
  artifacts/inspector_strictness_dist.png  -- cross-inspector variation in the
    violation rate (644 inspectors with 30+ substantive cases), the raw
    material of the examiner design.
  artifacts/inspector_causal.png -- two panels:
    (A) the 90-day permit-filing effect of inspector strictness across
        fixed-effect specifications (the selection story and where the
        positive within-neighborhood response appears);
    (B) downstream effects by horizon with case-mix FEs
        (category x unit x year-month): any permit filing and any future
        ECB violation, 30 to 365 days after inspection.

Estimates are transcribed from the July 2026 run of
scripts/compliance_analysis.py on the rebuilt master panel (N = 550,087);
re-run that script to reproduce them. Betas are per 1.0 change in
leave-one-out strictness; the figure rescales to percentage points per
10pp-stricter inspector (beta * 0.10 * 100).

Styling matches the other blog figures (size/origin/strictness text figures).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"; GREEN = "#2e9e62"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# ---- estimates: read from the clustered-SE grid produced alongside
# compliance_analysis.py (see risk_models/inspector_clustered_estimates.csv).
# All whiskers use standard errors clustered by inspector.
EST = pd.read_csv(config.DATA_DIR / "analysis" / "risk_models" / "inspector_clustered_estimates.csv")
SCALE = 10.0  # beta -> pp per 10pp-stricter inspector


def strictness_distribution():
    prof = pd.read_csv(config.DATA_DIR / "analysis" / "inspector_profiles.csv")
    rates = prof["violation_rate"].dropna() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.9),
                           gridspec_kw={"left": 0.085, "right": 0.97,
                                        "top": 0.80, "bottom": 0.155})
    ax.hist(rates, bins=32, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    p10, p50, p90 = np.percentile(rates, [10, 50, 90])
    for p, lbl in [(p10, f"10th pctile {p10:.0f}%"), (p50, f"median {p50:.0f}%"),
                   (p90, f"90th pctile {p90:.0f}%")]:
        ax.axvline(p, color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.text(p, ax.get_ylim()[1] * 0.97, " " + lbl, rotation=90,
                va="top", ha="right", fontsize=8.2, color=INK2)
    ax.set_xlabel("inspector's violation rate on substantive inspections (%)", fontsize=9)
    ax.set_ylabel("inspectors", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The same complaint, a different inspector, a different verdict",
                 x=0.02, y=0.975, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.885,
             f"Violation rate across {len(rates):,} DOB inspectors with 30+ substantive "
             f"inspections, 2020 through May 2026",
             fontsize=9, color=MUTED, va="top")
    out = ART / "inspector_strictness_dist.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


def causal_panels():
    lad = EST[EST["block"] == "ladder365"]
    order = ["No controls", "Complaint category", "Category x unit",
             "Category x unit x month", "+ borough", "+ NTA (neighborhood)"]
    rows = []
    for lbl in order:
        p = lad[(lad["label"] == lbl) & (lad["outcome"] == "any_permit_365d")].iloc[0]
        e = lad[(lad["label"] == lbl) & (lad["outcome"] == "ecb365")].iloc[0]
        rows.append((lbl, p["b"], p["se"], e["b"], e["se"]))
    tr = EST[EST["block"] == "tract"]
    if len(tr):
        p = tr[tr["outcome"] == "any_permit_365d"]
        e = tr[tr["outcome"] == "ecb365"]
        if len(p) and len(e):
            rows.append(("+ census tract", p.iloc[0]["b"], p.iloc[0]["se"],
                         e.iloc[0]["b"], e.iloc[0]["se"]))
    win = EST[EST["block"] == "windows"]
    wrows = []
    for w in [30, 60, 90, 180, 365]:
        p = win[win["outcome"] == f"any_permit_{w}d"].iloc[0]
        e = win[win["outcome"] == f"ecb{w}"].iloc[0]
        wrows.append((w, p["b"], p["se"], e["b"], e["se"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.8),
                                   gridspec_kw={"left": 0.30, "right": 0.955,
                                                "top": 0.85, "bottom": 0.07,
                                                "hspace": 0.44})
    GREEN_L, RED_L = "any permit filed", "any new ECB violation"
    ys = np.arange(len(rows))[::-1]
    for dy, idx_b, idx_s, c, l in [(+0.16, 1, 2, GREEN, GREEN_L),
                                   (-0.16, 3, 4, RED, RED_L)]:
        for y, row in zip(ys, rows):
            eff, ci = row[idx_b] * SCALE, 1.96 * row[idx_s] * SCALE
            ax1.errorbar(eff, y + dy, xerr=ci, fmt="o", ms=6, color=c,
                         ecolor=c, elinewidth=1.5, capsize=2.6, zorder=3,
                         markeredgecolor=SURFACE, markeredgewidth=1.1,
                         label=l if y == ys[0] else None)
    ax1.axvline(0, color=ZERO, lw=1.1)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax1.set_xlabel("pp change in one-year outcome per 10pp-stricter inspector", fontsize=9)
    ax1.set_title("A. One-year effects, by fixed-effect specification", loc="left",
                  fontsize=10, color=INK2, pad=8)
    ax1.legend(frameon=False, fontsize=8.8, loc="lower left")
    ax1.tick_params(labelsize=8.5)
    ax1.spines[["top", "right", "left"]].set_visible(False)

    x = np.arange(len(wrows))
    for idx_b, idx_s, color, lbl, dx in [(1, 2, GREEN, GREEN_L, -0.09),
                                          (3, 4, RED, RED_L, +0.09)]:
        eff = np.array([r[idx_b] for r in wrows]) * SCALE
        ci = np.array([1.96 * r[idx_s] for r in wrows]) * SCALE
        ax2.errorbar(x + dx, eff, yerr=ci, fmt="o-", ms=7, lw=1.4, color=color,
                     ecolor=color, elinewidth=1.6, capsize=3, zorder=3,
                     markeredgecolor=SURFACE, markeredgewidth=1.2, label=lbl)
    ax2.axhline(0, color=ZERO, lw=1.1)
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r[0]}d" for r in wrows], fontsize=9)
    ax2.set_xlabel("window after the inspection", fontsize=9)
    ax2.set_ylabel("pp per 10pp-stricter inspector", fontsize=9)
    ax2.set_title("B. Effects by window, inside category x unit x month x NTA cells", loc="left",
                  fontsize=10, color=INK2, pad=8)
    ax2.legend(frameon=False, fontsize=8.8, loc="upper left")
    ax2.tick_params(labelsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Downstream effects of drawing a stricter inspector",
                 x=0.02, y=0.982, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.938,
             "Permit filing and new ECB violations at the inspected property, per 10pp of leave-one-out inspector\n"
             "strictness. Panel A varies the fixed effects with the outcome window held at one year; panel B varies\n"
             "the window inside category x unit x month x NTA cells. Whiskers are 95% CIs, clustered by inspector.",
             fontsize=9, color=MUTED, va="top")
    out = ART / "inspector_causal.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    strictness_distribution()
    causal_panels()
