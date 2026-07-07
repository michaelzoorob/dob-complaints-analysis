"""
Figures for the text-analysis supplement of the Asian-owner article.

Fig T1  caller-subject and inspector-report feature gaps (pp), dot-CI
Fig T2  word-level log-odds, conversion-complaint subjects, classified
        Asian- vs white-owned homes

Reads risk_models/text_tidy_estimates.csv and conversion_word_logodds.csv.
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

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def dot(ax, x, y, lo, hi, p):
    col = BLUE if x >= 0 else RED
    sig = p < 0.05
    ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2, alpha=0.8)
    ax.plot(x, y, "o", ms=8, zorder=3,
            markerfacecolor=col if sig else SURFACE,
            markeredgecolor=SURFACE if sig else col, markeredgewidth=1.8)
    lab = f"{x:+.1f}" + ("" if sig else " (n.s.)")
    ax.annotate(lab, (hi, y), textcoords="offset points", xytext=(6, 0),
                va="center", ha="left", fontsize=9, color=INK2)


def fig_features():
    r = pd.read_csv(RM / "text_tidy_estimates.csv")
    r = r[r["term"] == "p_asian"].set_index("model")

    caller = [  # (model, label); bases = rates at classified-white homes
        ("people_watch", 'People-surveillance phrases (3% base)'),
        ("illegal_word", 'The word "illegal" (22% base)'),
        ("evidence_struct", "Structural-evidence terms (40% base)"),
        ("hedge", "Hedge or suspicion phrases (6% base)"),
        ("renting", "Renting mentions (5% base)"),
        ("tenant_marker", "Tenant-insider markers (2% base)"),
        ("neighbor_marker", "Neighbor markers (19% base)"),
        ("has_subject", "Complaint has caller text (61% base)"),
    ]
    inspect = [
        ("denied_vs_noresponse", "No-access report records refusal (18% base)"),
        ("has_comments", "Inspector report present (97% base)"),
        ("log_comment_words_subst", "Report length, resolved inspections (%)"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.8),
                             gridspec_kw={"left": 0.455, "right": 0.95, "top": 0.79,
                                          "bottom": 0.06, "hspace": 0.50,
                                          "height_ratios": [8, 3]})
    for ax, rows, title, ylab in [
            (axes[0], caller, "Caller's complaint text\n(pp gap unless noted)",
             "caller-text feature"),
            (axes[1], inspect, "Inspector's report\n(pp gap unless noted)",
             "report feature")]:
        ax.axvline(0, color=ZERO, lw=1.1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ys = np.arange(len(rows))[::-1]
        for y, (m, lab) in zip(ys, rows):
            row = r.loc[m]
            est, se, p = row["estimate"], row["std_error"], row["pr(>|t|)"]
            if m.startswith("log_"):
                est, se = est * 100, se * 100  # log points -> approx %
            dot(ax, est, y, est - 1.96 * se, est + 1.96 * se, p)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab in rows], fontsize=9, color=INK)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK2, labelpad=12)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_xlim(-3.4, 3.4)
    axes[1].set_xlim(-4.4, 3.2)
    axes[0].set_xlabel("percentage-point difference", fontsize=8.5)
    axes[1].set_xlabel("percentage-point difference (report-length row in percent)", fontsize=8.5)
    fig.suptitle("Text features of complaints and reports, predicted-Asian vs. white-owned",
                 x=0.02, y=0.975, ha="left", fontsize=12, color=INK, weight="semibold")
    fig.text(0.02, 0.938, "Effect of predicted-Asian ownership, within complaint code, census tract, and size class ·\n"
                          "311-referenced complaints only · N = 84,652 with caller text; presence, length, and refusal\n"
                          "rows use their own samples (139,302 / 71,294 / 51,370) · whiskers = 95% CI · hollow = n.s. ·\n"
                          "bases are rates at white-owned homes",
             fontsize=9, color=MUTED, va="top")
    fig.savefig(ART / "asian_text_features.png", dpi=200)
    plt.close(fig)
    print("saved asian_text_features.png")


DISPLAY_EXCLUDE = {"http", "https", "www", "com",           # URL fragments
                   "http www", "new york", "york city",
                   "staten island",
                   "york", "staten", "island", "brooklyn",
                   "queens", "bronx", "manhattan"}           # place/URL artifacts


URL_TOKENS = {"http", "https", "www", "com"}


def _word_panel(ax, csv, title, xlim):
    z = pd.read_csv(RM / csv)
    z = z[~z["word"].isin(DISPLAY_EXCLUDE)]
    z = z[~z["word"].apply(lambda w: any(t in URL_TOKENS for t in w.split()))]
    d = pd.concat([z.nsmallest(10, "z").sort_values("z"),
                   z.nlargest(10, "z").sort_values("z")])
    ax.axvline(0, color=ZERO, lw=1.1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ys = np.arange(len(d))
    ax.scatter(d["z"], ys, s=60, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    for y, (_, row) in zip(ys, d.iterrows()):
        ha = "right" if row["z"] < 0 else "left"
        off = -7 if row["z"] < 0 else 7
        ax.annotate(row["word"], (row["z"], y), textcoords="offset points",
                    xytext=(off, 0), va="center", ha=ha, fontsize=9, color=INK)
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)


def fig_words():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0),
                             gridspec_kw={"left": 0.04, "right": 0.97, "top": 0.832,
                                          "bottom": 0.06, "hspace": 0.34})
    _word_panel(axes[0], "conversion_word_logodds.csv", "Single words", (-17, 25))
    _word_panel(axes[1], "conversion_bigram_logodds.csv", "Word pairs", (-15, 21))
    for ax in axes:
        ax.text(0.985, 0.03, "right of zero = more typical at Asian-owned homes\n"
                             "left of zero = more typical at white-owned homes",
                transform=ax.transAxes, fontsize=8.8, color=MUTED,
                ha="right", va="bottom", style="italic")
    for ax in axes:
        ax.set_xlabel("informed-Dirichlet log-odds z-score (unitless)", fontsize=9)
    fig.suptitle("Words and word pairs that distinguish illegal-conversion complaints,\nby predicted owner race",
                 x=0.02, y=0.987, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.938, "Callers' own complaint text (not inspector reports) · 311-referenced complaints only · 13,041 conversion\n"
                          "complaints at Asian-owned (P>0.7) homes vs. 3,849 at white-owned · singular and plural forms combined ·\n"
                          "adjacent words where neither is a stopword · URL fragments and place names omitted for display",
             fontsize=9, color=MUTED, va="top")
    fig.savefig(ART / "asian_conversion_words.png", dpi=200)
    plt.close(fig)
    print("saved asian_conversion_words.png")


if __name__ == "__main__":
    fig_features()
    fig_words()
