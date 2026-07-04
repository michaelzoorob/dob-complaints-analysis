"""
Figures for the risk-factor study (post 5).

Fig 1: two-panel forest plot — complaint margin (PPML, % change) and
       conditional violation margin (pp per inspection, category-adjusted),
       both within census tract x building-size bin.
Fig 2: two-margin scatter — "scrutiny vs substance" typology.

Reads data/analysis/risk_models/tidy_estimates.csv.
Writes data/analysis/blog_posts/artifacts/risk_forest.png, risk_two_margin.png
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
RES = config.DATA_DIR / "analysis" / "risk_models" / "tidy_estimates.csv"

# palette (dataviz reference, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

FACTORS = [  # display order (grouped), term -> short label
    ("any_prior_viol", "Cited for a violation, 2010–19"),
    ("llc", "LLC-owned (vs. individual)"),
    ("corp_other", "Corporate/org owner (vs. individual)"),
    ("trust_estate", "Trust or estate owner (vs. individual)"),
    ("owner_occ_star", "Owner-occupied (STAR)"),
    ("is_coop", "Co-op building"),
    ("is_condo", "Condominium"),
    ("nycha", "NYCHA"),
    ("era_pre1940", "Built before 1940 (vs. 2000+)"),
    ("era_4079", "Built 1940–79 (vs. 2000+)"),
    ("era_8099", "Built 1980–99 (vs. 2000+)"),
    ("mixed_use", "Mixed use (commercial units)"),
    ("log2_area_per_unit", "Floor area per unit (per doubling)"),
    ("value_rank", "Assessed value, bottom→top of type"),
]
# owner tier: estimated in the owner-augmented specification
OWNER_FACTORS = [
    ("geo_nyc_other", "Owner's address elsewhere in NYC (vs. same zip)"),
    ("geo_outside_nyc", "Owner's address outside NYC (vs. same zip)"),
    ("multi_prop_owner", "Multi-property owner (3+ lots)"),
]
OWNER_RES = config.DATA_DIR / "analysis" / "risk_models" / "owner_tidy_estimates.csv"
CITE_RES = config.DATA_DIR / "analysis" / "risk_models" / "citation_tidy_estimates.csv"


def _rows(comp, cite, viol, factors, group):
    rows = []
    for term, label in factors:
        c, e, v = comp.loc[term], cite.loc[term], viol.loc[term]
        rows.append({
            "term": term, "label": label, "group": group,
            "c_est": (np.exp(c["estimate"]) - 1) * 100,
            "c_lo": (np.exp(c["estimate"] - 1.96 * c["std_error"]) - 1) * 100,
            "c_hi": (np.exp(c["estimate"] + 1.96 * c["std_error"]) - 1) * 100,
            "c_p": c["pr(>|t|)"],
            "e_est": (np.exp(e["estimate"]) - 1) * 100,
            "e_lo": (np.exp(e["estimate"] - 1.96 * e["std_error"]) - 1) * 100,
            "e_hi": (np.exp(e["estimate"] + 1.96 * e["std_error"]) - 1) * 100,
            "e_p": e["pr(>|t|)"],
            "v_est": v["estimate"], "v_lo": v["estimate"] - 1.96 * v["std_error"],
            "v_hi": v["estimate"] + 1.96 * v["std_error"], "v_p": v["pr(>|t|)"],
        })
    return rows


def load():
    r = pd.read_csv(RES)
    rc = pd.read_csv(CITE_RES)
    comp = r[r["model"] == "tract_ppml_ncomp"].set_index("term")
    cite = rc[rc["model"] == "ppml_ecb"].set_index("term")
    viol = r[r["model"] == "cond_violrate_catadj"].set_index("term")
    rows = _rows(comp, cite, viol, FACTORS, "base")
    ro = pd.read_csv(OWNER_RES)
    ocomp = ro[ro["model"] == "owner_ppml_ncomp"].set_index("term")
    ocite = rc[rc["model"] == "ppml_ecb_owner"].set_index("term")
    oviol = ro[ro["model"] == "owner_cond_violrate"].set_index("term")
    rows += _rows(ocomp, ocite, oviol, OWNER_FACTORS, "owner")
    return pd.DataFrame(rows)


def forest(df: pd.DataFrame):
    n = len(df)
    # top-to-bottom order with a gap before the owner tier
    y = np.arange(n, dtype=float)[::-1]
    gap = 0.9
    y[df["group"].values == "owner"] -= gap
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 7.4), sharey=True,
                             gridspec_kw={"wspace": 0.06, "left": 0.27, "right": 0.988,
                                          "top": 0.82, "bottom": 0.08})
    panels = [
        (axes[0], "c", "Complaints received, 2020–26\n(% difference)", "%"),
        (axes[1], "e", "ECB citations issued, 2020–26\n(% difference)", "%"),
        (axes[2], "v", "Violations per inspection\n(percentage-point difference)", "pp"),
    ]
    for ax, pre, title, unit in panels:
        ax.axvline(0, color=BASE, lw=1.2, zorder=1)
        ax.grid(axis="x", color=GRID, lw=1.0, zorder=0)
        for yi, (_, row) in zip(y, df.iterrows()):
            sig = row[f"{pre}_p"] < 0.05
            ax.plot([row[f"{pre}_lo"], row[f"{pre}_hi"]], [yi, yi],
                    color=BLUE, lw=2, solid_capstyle="round", zorder=2, alpha=0.85)
            ax.plot(row[f"{pre}_est"], yi, "o", ms=8.5, zorder=3,
                    markerfacecolor=BLUE if sig else SURFACE,
                    markeredgecolor=SURFACE if sig else BLUE,
                    markeredgewidth=2)
            xoff = (row[f"{pre}_hi"] - row[f"{pre}_lo"]) * 0.0
            val = f"{row[f'{pre}_est']:+.0f}%" if unit == "%" else f"{row[f'{pre}_est']:+.1f}"
            ax.annotate(val, (row[f"{pre}_hi"] + xoff, yi), textcoords="offset points",
                        xytext=(6, -0.5), va="center", ha="left", fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_ylim(-0.6 - gap, n - 0.4)
        ydiv = y[df["group"].values == "owner"].max() + 0.5 + gap / 2
        ax.axhline(ydiv, color=GRID, lw=1.0)
    axes[0].set_xlim(-60, 135)
    axes[1].set_xlim(-75, 205)
    axes[2].set_xlim(-8, 8.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["label"], fontsize=9.5, color=INK)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Risk factors, comparing same-size buildings in the same census tract",
                 x=0.02, y=0.975, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.925, "766,382 residential properties · census-tract and unit-count fixed effects · "
                          "filled = p<0.05 · whiskers = 95% CI, SEs clustered by tract ·\n"
                          "bottom panel: owner tier, from the owner-augmented specification "
                          "(deed mailing address name-matched to current owner)",
             fontsize=9, color=MUTED, va="top")
    out = ART / "risk_forest.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def two_margin(df: pd.DataFrame):
    d = df[~df["term"].isin(["era_4079", "log2_area_per_unit",
                             "geo_nyc_other", "multi_prop_owner"])].copy()
    fig, ax = plt.subplots(figsize=(9.4, 7.0),
                           gridspec_kw={"left": 0.10, "right": 0.97, "top": 0.86, "bottom": 0.10})
    ax.grid(color=GRID, lw=1.0, zorder=0)
    ax.axhline(0, color=BASE, lw=1.2, zorder=1)
    ax.axvline(0, color=BASE, lw=1.2, zorder=1)

    ax.scatter(d["c_est"], d["v_est"], s=110, color=BLUE, zorder=3,
               edgecolors=SURFACE, linewidths=2)

    offsets = {  # hand-placed label offsets (points): (dx, dy, ha)
        "any_prior_viol": (0, 12, "center"),
        "llc": (10, -3, "left"),
        "corp_other": (0, -17, "center"),
        "trust_estate": (0, -17, "center"),
        "owner_occ_star": (-10, -4, "right"),
        "is_coop": (10, 4, "left"),
        "is_condo": (-10, 2, "right"),
        "nycha": (10, -4, "left"),
        "era_pre1940": (10, 0, "left"),
        "era_8099": (0, 12, "center"),
        "mixed_use": (10, -3, "left"),
        "value_rank": (10, -1, "left"),
        "geo_outside_nyc": (-10, -14, "right"),
        "multi_prop_owner": (-4, -16, "center"),
        "geo_nyc_other": (-10, 6, "right"),
    }
    short = {"geo_outside_nyc": "Owner outside NYC",
             "corp_other": "Corporate/org owner"}
    for _, row in d.iterrows():
        dx, dy, ha = offsets[row["term"]]
        ax.annotate(short.get(row["term"], row["label"]), (row["c_est"], row["v_est"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9.3, color=INK)

    # quadrant glosses
    glosses = [
        (128, 7.9, "more complaints,\nmore violations when inspected", "right", "top"),
        (-52, 7.9, "fewer complaints,\nmore violations when inspected", "left", "top"),
        (128, -7.3, "more complaints,\nfewer violations when inspected", "right", "bottom"),
        (-52, -7.3, "fewer complaints,\nfewer violations when inspected", "left", "bottom"),
    ]
    for x, yy, s, ha, va in glosses:
        ax.text(x, yy, s, ha=ha, va=va, fontsize=8.8, color=MUTED, style="italic")

    ax.set_xlim(-60, 135)
    ax.set_ylim(-8.4, 8.4)
    ax.set_xlabel("Effect on complaints received (%), same size & tract", fontsize=10)
    ax.set_ylabel("Effect on violations per inspection (pp)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Scrutiny vs. substance: the two margins of building enforcement",
                 loc="left", fontsize=12.5, color=INK, weight="semibold", y=1.10)
    ax.text(0, 1.045, "x: % difference in 311/DOB complaints · y: pp difference in violation rate per substantive inspection,\n"
                      "category-adjusted · both within census tract × building-size bin · baseline violation rate: 31 per 100 inspections",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    out = ART / "risk_two_margin.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    df = load()
    print(df.round(2).to_string(index=False))
    forest(df)
    two_margin(df)
