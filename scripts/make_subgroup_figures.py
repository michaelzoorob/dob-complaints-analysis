"""
Figures for the within-Asian subgroup analysis (research note).

Fig A  volume margins by subgroup: all complaints and illegal-conversion
       complaints, percent vs classified-white, rows ordered by sample size.
Fig B  violations issued (ECB and disposition ledgers, two series) and
       per-inspection margins (violation per substantive inspection and
       no-access per complaint, two series), same rows.

Reads risk_models/asian_subgroup_estimates.csv + _descriptives.csv.
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
RM = config.DATA_DIR / "analysis" / "risk_models"

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"
AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

LABELS = {
    "chinese": "Chinese", "korean": "Korean", "vietnamese": "Vietnamese",
    "indian": "Indian / other South Asian",
    "muslim_sa": "Bangladeshi / Pakistani",
    "sikh_punjabi": "Sikh / Punjabi", "indo_caribbean": "Indo-Caribbean",
    "singh": "Singh (shared surname)", "nepali_himalayan": "Nepali / Himalayan",
    "filipino": "Filipino", "japanese": "Japanese",
    "asian_small": "Small subgroups pooled",
}


def load():
    est = pd.read_csv(RM / "asian_subgroup_estimates.csv")
    desc = pd.read_csv(RM / "asian_subgroup_descriptives.csv").set_index("subgroup")
    terms = [t for t in est["term"].unique()
             if t.startswith("sg_") and t != "sg_asian_small"]
    order = sorted(terms, key=lambda t: -desc.loc[t[3:], "n_props"]
                   if t[3:] in desc.index else 0)
    return est, desc, order


def get(est, model, term, kind):
    r = est[(est["model"] == model) & (est["term"] == term)]
    if r.empty:
        return None
    b, se, p = r.iloc[0]["estimate"], r.iloc[0]["std_error"], r.iloc[0]["pr(>|t|)"]
    if kind == "pct":
        return ((np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100,
                (np.exp(b + 1.96 * se) - 1) * 100, p)
    return (b, b - 1.96 * se, b + 1.96 * se, p)


def dot(ax, x, y, lo, hi, p, color=None, unit="%"):
    col = color if color else (BLUE if x >= 0 else RED)
    sig = p < 0.05
    ax.plot([lo, hi], [y, y], color=col, lw=1.5, solid_capstyle="round",
            zorder=2, alpha=0.75)
    ax.plot(x, y, "o", ms=6.5, zorder=3,
            markerfacecolor=col if sig else SURFACE,
            markeredgecolor=SURFACE if sig else col, markeredgewidth=1.6)
    lab = (f"{x:+.0f}%" if unit == "%" else f"{x:+.1f}") + ("" if sig else " (n.s.)")
    ax.annotate(lab, (hi, y), textcoords="offset points", xytext=(5, 0),
                va="center", ha="left", fontsize=8, color=INK2)


def style(ax, ys, ticklabels, xlim, xlabel):
    ax.axvline(0, color=ZERO, lw=1.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(ticklabels, fontsize=9, color=INK)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)


def fig_volume(est, desc, order):
    ys = np.arange(len(order))[::-1]
    labels = [f"{LABELS.get(t[3:], t[3:])}\n(n={desc.loc[t[3:], 'n_props']:,})"
              if t[3:] in desc.index else LABELS.get(t[3:], t[3:]) for t in order]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0),
                             gridspec_kw={"left": 0.315, "right": 0.94, "top": 0.82,
                                          "bottom": 0.06, "hspace": 0.35})
    for ax, model, title in [
            (axes[0], "ppml_n_complaints", "All complaints, percent difference"),
            (axes[1], "ppml_n_conv", "Illegal-conversion complaints, percent difference")]:
        vals = [get(est, model, t, "pct") for t in order]
        hi = max(v[2] for v in vals if v)
        lo = min(v[1] for v in vals if v)
        for y, v in zip(ys, vals):
            if v:
                dot(ax, v[0], y, v[1], v[2], v[3])
        style(ax, ys, labels, (min(lo * 1.1, -12), hi * 1.32),
              "percent difference vs. classified-white owners")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)
    fig.suptitle("Complaint volume by Asian-origin subgroup",
                 x=0.02, y=0.978, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.925, "Owner subgroups classified from distinctive surnames · same census tract & building size,\n"
                          "same covariates as the main analysis · reference = classified-white owners · whiskers = 95% CI ·\n"
                          "hollow = not significant at 5% · SINGH kept separate (shared Punjabi and Indo-Caribbean surname) ·\n"
                          "Filipino and Japanese owners (n=168 pooled) estimated but omitted here for legibility",
             fontsize=9, color=MUTED, va="top")
    fig.savefig(ART / "asian_subgroup_volume.png", dpi=200)
    plt.close(fig)
    print("saved asian_subgroup_volume.png")


def fig_outcomes(est, desc, order):
    ys = np.arange(len(order))[::-1]
    labels = [f"{LABELS.get(t[3:], t[3:])}" for t in order]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0),
                             gridspec_kw={"left": 0.315, "right": 0.94, "top": 0.845,
                                          "bottom": 0.06, "hspace": 0.35})

    ax = axes[0]
    vals_e = [get(est, "ppml_n_ecb_2020on", t, "pct") for t in order]
    vals_v = [get(est, "ppml_n_viol_disp", t, "pct") for t in order]
    for y, ve, vv in zip(ys, vals_e, vals_v):
        if ve:
            dot(ax, ve[0], y + 0.17, ve[1], ve[2], ve[3], color=BLUE)
        if vv:
            dot(ax, vv[0], y - 0.17, vv[1], vv[2], vv[3], color=AQUA)
    hi = max(v[2] for v in vals_e + vals_v if v)
    lo = min(v[1] for v in vals_e + vals_v if v)
    style(ax, ys, labels, (min(lo * 1.1, -12), hi * 1.35),
          "percent difference vs. classified-white owners")
    ax.set_title("Violations issued, two ledgers, percent difference",
                 loc="left", fontsize=9.5, color=INK2, pad=8)
    ax.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", ls="", label="ECB/OATH penalty citations"),
                       plt.Line2D([], [], color=AQUA, marker="o", ls="", label="disposition violations")],
              loc="upper right", frameon=False, fontsize=8)

    ax = axes[1]
    vals_s = [get(est, "lpm_viol100", t, "pp") for t in order]
    vals_n = [get(est, "lpm_noacc100", t, "pp") for t in order]
    for y, vs, vn in zip(ys, vals_s, vals_n):
        if vs:
            dot(ax, vs[0], y + 0.17, vs[1], vs[2], vs[3], color=RED, unit="pp")
        if vn:
            dot(ax, vn[0], y - 0.17, vn[1], vn[2], vn[3], color=MUTED, unit="pp")
    hi = max(v[2] for v in vals_s + vals_n if v)
    lo = min(v[1] for v in vals_s + vals_n if v)
    style(ax, ys, labels, (lo * 1.25, hi * 1.4), "percentage-point difference")
    ax.set_title("Per-inspection margins, percentage points",
                 loc="left", fontsize=9.5, color=INK2, pad=8)
    ax.legend(handles=[plt.Line2D([], [], color=RED, marker="o", ls="", label="violation per substantive inspection"),
                       plt.Line2D([], [], color=MUTED, marker="o", ls="", label="no access per complaint")],
              loc="lower right", bbox_to_anchor=(1.0, 1.06), borderaxespad=0,
              frameon=False, fontsize=8)

    fig.suptitle("Violation and inspection outcomes by Asian-origin subgroup",
                 x=0.02, y=0.978, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.928, "Same design as the volume figure · per-inspection margins estimated within complaint code ·\n"
                          "whiskers = 95% CI · hollow = not significant at 5%",
             fontsize=9, color=MUTED, va="top")
    fig.savefig(ART / "asian_subgroup_outcomes.png", dpi=200)
    plt.close(fig)
    print("saved asian_subgroup_outcomes.png")


def main():
    est, desc, order = load()
    fig_volume(est, desc, order)
    fig_outcomes(est, desc, order)


if __name__ == "__main__":
    main()
