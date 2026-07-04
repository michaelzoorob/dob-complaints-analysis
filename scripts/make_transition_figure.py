"""
Event-study figure for the ownership-transition robustness check.

Two stacked panels, arm's-length sales only, W->A vs W->W benchmark:
conversion complaints (headline) and all complaints, coefficients by year
relative to the deed (ref = year before), transition FE + calendar-year FE,
95% CIs clustered by lot. Reads risk_models/transition_eventstudy.csv.
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
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def series(es, outcome):
    term = es.columns[0]
    g = es[(es["contrast"] == "W->A vs W->W") & (es["outcome"] == outcome)
           & es[term].astype(str).str.contains("event_t::", na=False)].copy()
    g["t"] = g[term].str.extract(r"event_t::(-?\d+)").astype(int)
    g = g.sort_values("t")
    ref = pd.DataFrame({"t": [-1], "Estimate": [0.0], "Std. Error": [0.0]})
    g = pd.concat([g[["t", "Estimate", "Std. Error"]], ref]).sort_values("t")
    g["lo"] = g["Estimate"] - 1.96 * g["Std. Error"]
    g["hi"] = g["Estimate"] + 1.96 * g["Std. Error"]
    return g


def main():
    es = pd.read_csv(RM / "transition_eventstudy.csv")
    es = es[es["arms_length"] == True]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.1), sharex=True,
                             gridspec_kw={"left": 0.14, "right": 0.96, "top": 0.80,
                                          "bottom": 0.075, "hspace": 0.46})
    def draw(ax, outc, color, xoff=0.0, label=None):
        g = series(es, outc)
        ax.plot(g["t"] + xoff, g["Estimate"], color=color, lw=2, zorder=3, label=label)
        for _, r in g.iterrows():
            ax.plot([r["t"] + xoff, r["t"] + xoff], [r["lo"], r["hi"]], color=color,
                    lw=1.4, alpha=0.8, zorder=2)
        ax.plot(g["t"] + xoff, g["Estimate"], "o", ms=7, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)

    panels = [
        (axes[0], "Illegal-conversion complaints per year"),
        (axes[1], "All complaints per year"),
        (axes[2], "Violations issued per year, two ledgers"),
    ]
    for ax, title in panels:
        ax.axhline(0, color=ZERO, lw=1.1)
        ax.axvline(-0.5, color=MUTED, lw=1.0, ls=(0, (1, 2)))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)
        ax.set_ylabel("difference, counts per year", fontsize=8.5)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(-4, 5))
    draw(axes[0], "n_conv", BLUE)
    draw(axes[1], "n_comp", BLUE)
    draw(axes[2], "n_ecb", BLUE, xoff=-0.08, label="ECB/OATH penalty citations")
    draw(axes[2], "n_viol", AQUA, xoff=0.08, label="disposition violations")
    axes[2].legend(loc="upper left", frameon=False, fontsize=8.5)
    axes[0].text(-0.45, axes[0].get_ylim()[1] * 0.92, "deed recorded",
                 fontsize=8.5, color=MUTED, ha="left", va="top", style="italic")
    axes[2].set_xlabel("years relative to the sale (reference = year before)",
                       fontsize=9.5)
    fig.suptitle("Complaints at the same property before and after a sale\n"
                 "to a predicted-Asian buyer, relative to a sale to a white buyer",
                 x=0.02, y=0.98, ha="left", fontsize=12, color=INK, weight="semibold")
    fig.text(0.02, 0.91, "Arm's-length sales of white-owned small homes (8,439 to predicted-Asian buyers vs. 13,278 to white\n"
                         "buyers; same-surname family transfers excluded) · transition and calendar-year fixed effects ·\n"
                         "whiskers = 95% CI, SEs clustered by lot",
             fontsize=8.5, color=MUTED, va="top")
    fig.savefig(ART / "asian_transition_eventstudy.png", dpi=200)
    plt.close(fig)
    print("saved asian_transition_eventstudy.png")


if __name__ == "__main__":
    main()
