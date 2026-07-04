"""
Figures for the Asian-owner article (post 6).

Fig 1  two-margin headline panel (volume % | once-inspector-arrives pp)
Fig 2  category-margin matrix: complaint premium / substantiation gap /
       no-access gap by complaint type (the centerpiece)
Fig 3  context dot plot: size, occupancy, borough + reference line

Conventions: blue = more than comparable white-owned, red = less;
hollow = not significant at 5%; whiskers = 95% CI; all estimates within
census tract x building-size bin. Reads risk_models CSVs; occupancy-split
models are re-estimated here for clean per-group CIs.
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
TINT = "#f2f1ec"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def tidy(path, model):
    r = pd.read_csv(path)
    return r[r["model"] == model].set_index("term")


def pct(row):
    b, se = row["estimate"], row["std_error"]
    return ((np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100,
            (np.exp(b + 1.96 * se) - 1) * 100, row["pr(>|t|)"])


def pp(row):
    b, se = row["estimate"], row["std_error"]
    return (b, b - 1.96 * se, b + 1.96 * se, row["pr(>|t|)"])


def dot(ax, x, y, lo, hi, p, label=None, dy=0, unit="%"):
    col = BLUE if x >= 0 else RED
    sig = p < 0.05
    ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round",
            zorder=2, alpha=0.8)
    ax.plot(x, y, "o", ms=8, zorder=3,
            markerfacecolor=col if sig else SURFACE,
            markeredgecolor=SURFACE if sig else col, markeredgewidth=1.8)
    if label is None:
        label = (f"{x:+.0f}%" if unit == "%" else f"{x:+.1f}") + ("" if sig else " (n.s.)")
    ax.annotate(label, (hi, y), textcoords="offset points", xytext=(6, dy),
                va="center", ha="left", fontsize=9, color=INK2)


def fig1():
    het = RM / "asian_heterogeneity.csv"
    cit = RM / "citation_tidy_estimates.csv"
    comp = pct(tidy(RM / "owner_tidy_estimates.csv", "bisg_ppml_ncomp").loc["p_asian"])
    ecb = pct(tidy(cit, "bisg_ppml_ecb").loc["p_asian"])
    violct = pct(tidy(cit, "bisg_ppml_viol").loc["p_asian"])
    viol = pp(tidy(het, "insp_viol_pooled").loc["p_asian"])
    ecbinsp = pp(tidy(het, "insp_ecb_pooled").loc["p_asian"])
    noacc = pp(tidy(het, "insp_noacc_pooled").loc["p_asian"])

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0),
                             gridspec_kw={"left": 0.30, "right": 0.96, "top": 0.76,
                                          "bottom": 0.12, "hspace": 0.70})
    a = axes[0]
    a.axvline(0, color=ZERO, lw=1.1)
    a.grid(axis="x", color=GRID, lw=0.8)
    for y, est in [(2, comp), (1, violct), (0, ecb)]:
        dot(a, est[0], y, est[1], est[2], est[3])
    a.set_yticks([2, 1, 0]); a.set_yticklabels(["Complaints received", "Disposition violations\nissued", "Penalty (ECB)\ncitations issued"], fontsize=9.5, color=INK)
    a.set_xlim(-5, 60); a.set_ylim(-0.6, 2.6)
    a.set_xlabel("percent difference vs. comparable white-owned homes", fontsize=8.5)
    a.set_title("Enforcement volume\n(% vs. comparable white-owned)", loc="left", fontsize=9.5, color=INK2, pad=8)
    a.tick_params(axis="y", length=0); a.tick_params(labelsize=8.5)
    a.spines[["top", "right", "left"]].set_visible(False)

    b = axes[1]
    b.axvline(0, color=ZERO, lw=1.1); b.grid(axis="x", color=GRID, lw=0.8)
    for y, est, note in [(2, viol, None), (1, ecbinsp, None), (0, noacc, None)]:
        dot(b, est[0], y, est[1], est[2], est[3], unit="pp")
    b.set_yticks([2, 1, 0])
    b.set_yticklabels(["Disposition violation\ncited", "ECB citation\nissued", "Inspector unable\nto get in"], fontsize=9.5, color=INK)
    b.set_xlim(-5, 5); b.set_ylim(-0.6, 2.6)
    b.set_xlabel("percentage-point difference", fontsize=8.5)
    b.set_title("Once an inspector arrives\n(percentage points)", loc="left", fontsize=9.5, color=INK2, pad=8)
    b.tick_params(axis="y", length=0); b.tick_params(labelsize=8.5)
    b.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Enforcement volume and per-inspection outcomes",
                 x=0.02, y=0.975, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.90, "Predicted-Asian vs. predicted-white owned properties, same census tract & building size ·\n"
                         "individually owned, <16 units · whiskers = 95% CI",
             fontsize=9, color=MUTED, va="top")
    fig.text(0.30, 0.012, "baselines ≈29 of 100 substantive inspections yield a violation, ≈26 an ECB citation ·\n"
                          "inspection-weighted property-level violation estimate −3.1pp",
             fontsize=8, color=MUTED, va="bottom")
    fig.savefig(ART / "asian_two_margins.png", dpi=200)
    plt.close(fig)


def fig2():
    het = RM / "asian_heterogeneity.csv"
    citcat = RM / "category_citations.csv"
    rows = [  # (label, count-model, citation-model, viol-term, noacc-term)
        ("Illegal conversion", "cat_n_conv", "citations_conversion",
         "pa_conversion", "pa_conversion"),
        ("Boiler / mechanical", "cat_n_boiler", "citations_boiler",
         "pa_boiler_mech", "pa_boiler_mech"),
        ("Construction", "cat_n_constr", "citations_constr",
         "pa_construction", "pa_construction"),
        ("All other", "cat_n_other", "citations_other",
         "pa_other", "pa_other"),
    ]
    viol = tidy(het, "insp_viol_bycat"); noacc = tidy(het, "insp_noacc_bycat")

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.7),
                             gridspec_kw={"left": 0.30, "right": 0.96, "top": 0.858,
                                          "bottom": 0.055, "hspace": 0.78})
    ys = np.arange(len(rows))[::-1]

    for ax in axes:
        ax.axhspan(ys[0] - 0.42, ys[0] + 0.42, color=TINT, zorder=0)
        ax.axvline(0, color=ZERO, lw=1.1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right", "left"]].set_visible(False)

    for y, (label, cm, xm, vt, nt) in zip(ys, rows):
        c = pct(tidy(het, cm).loc["p_asian"])
        dot(axes[0], c[0], y, c[1], c[2], c[3])
        e = pct(tidy(citcat, xm).loc["p_asian"])
        dot(axes[1], e[0], y, e[1], e[2], e[3])
        v = pp(viol.loc[vt]); dot(axes[2], v[0], y, v[1], v[2], v[3], unit="pp")
        n = pp(noacc.loc[nt]); dot(axes[3], n[0], y, n[1], n[2], n[3], unit="pp")

    axes[0].set_xlim(-10, 175)
    axes[1].set_xlim(-10, 175)
    axes[2].set_xlim(-11, 11); axes[3].set_xlim(-11, 11)
    for ax in axes[:2]:
        ax.set_xlabel("percent difference vs. comparable white-owned homes", fontsize=8)
    for ax in axes[2:]:
        ax.set_xlabel("percentage-point difference", fontsize=8)
    axes[0].set_title("Complaint volume\n(% vs. comparable white-owned)", loc="left", fontsize=9.5, color=INK2, pad=8)
    axes[1].set_title("Violations issued\n(% vs. comparable white-owned)", loc="left", fontsize=9.5, color=INK2, pad=8)
    axes[2].set_title("Violations found per inspection\n(pp gap · left = substantiated less often)",
                      loc="left", fontsize=9.5, color=INK2, pad=8)
    axes[3].set_title("Inspector unable to get in\n(pp gap · right = more often)",
                      loc="left", fontsize=9.5, color=INK2, pad=8)

    fig.suptitle("Gaps by complaint type",
                 x=0.02, y=0.985, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.958, "Predicted-Asian vs. predicted-white owners, within census tract, building-size class,\n"
                          "and complaint code · whiskers = 95% CI · hollow = n.s. · elevator complaints omitted\n"
                          "(too few on 1–15 unit homes) · panels 1–2 and panels 3–4 each share one scale",
             fontsize=9, color=MUTED, va="top")
    fig.savefig(ART / "asian_category_matrix.png", dpi=200)
    plt.close(fig)


def fig3():
    import pyfixest as pf
    het = RM / "asian_heterogeneity.csv"
    # occupancy split re-estimated for clean per-group CIs
    df = pd.read_csv(config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz",
                     dtype={"bct2020": str, "size_bin": str, "borocode": str}, low_memory=False)
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])]
    bs = df[(df["owner_type"] == "individual") & (df["unitsres"] < 16) & df["p_white"].notna()]
    X = ("p_black + p_hispanic + p_asian + era_pre1940 + era_4079 + era_8099 + era_unknown"
         " + mixed_use + mzone + multi_bldg + log2_area_per_unit + value_rank + any_prior_viol"
         " + geo_nyc_other + geo_outside_nyc + geo_unknown + multi_prop_owner")
    occ = {}
    for flag, lab in [(True, "Owner-occupied (STAR)"), (False, "Absentee-owned")]:
        s = bs[bs["owner_occ_star"] == flag]
        m = pf.fepois(f"n_complaints ~ {X} | size_bin + bct2020", data=s, vcov={"CRV1": "bct2020"})
        t = m.tidy().loc["p_asian"]
        occ[lab] = pct({"estimate": t["Estimate"], "std_error": t["Std. Error"],
                        "pr(>|t|)": t["Pr(>|t|)"]})

    rows = []
    rows.append(("BUILDING SIZE", None))
    for m, lab in [("strata_1unit", "Single-family"), ("strata_2-4units", "2–4 units"),
                   ("strata_5-15units", "5–15 units")]:
        rows.append((lab, pct(tidy(het, m).loc["p_asian"])))
    rows.append(("OCCUPANCY", None))
    for lab in ["Absentee-owned", "Owner-occupied (STAR)"]:
        rows.append((lab, occ[lab]))
    rows.append(("BOROUGH", None))
    for m, lab in [("boro_Bronx", "Bronx"), ("boro_Queens", "Queens"),
                   ("boro_Staten Is", "Staten Island"), ("boro_Brooklyn", "Brooklyn")]:
        rows.append((lab, pct(tidy(het, m).loc["p_asian"])))

    fig, ax = plt.subplots(figsize=(6.8, 4.8),
                           gridspec_kw={"left": 0.28, "right": 0.95, "top": 0.79, "bottom": 0.09})
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.axvline(0, color=ZERO, lw=1.1)
    ax.axvline(38, color=MUTED, lw=1.0, ls=(0, (1, 2)))
    ax.text(38, len(rows) - 0.35, "all properties +38%", fontsize=8.5, color=MUTED,
            ha="center", va="bottom")

    ylabels, ypos = [], []
    y = len(rows) - 1
    for lab, est in rows:
        if est is None:
            ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(),
                    fontsize=8, color=MUTED, ha="right", va="center")
            y -= 0.7
            continue
        dot(ax, est[0], y, est[1], est[2], est[3])
        ylabels.append(lab); ypos.append(y)
        y -= 1
    ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=9.5, color=INK)
    ax.set_xlim(-35, 115)
    ax.set_xlabel("percent difference in complaints received vs. comparable white-owned homes",
                  fontsize=8.5)
    ax.set_ylim(y + 0.4, len(rows) - 0.1)
    ax.tick_params(axis="y", length=0); ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("The complaint gap by building size, occupancy, and borough",
                 x=0.02, y=0.965, ha="left", fontsize=12, color=INK, weight="semibold")
    fig.text(0.02, 0.90, "Effect of predicted-Asian ownership on complaints received (%),\n"
                         "within tract and size class · whiskers = 95% CI · hollow = n.s. ·\n"
                         "Manhattan omitted (fewer than 5,000 small homes in the sample)",
             fontsize=8.5, color=MUTED, va="top")
    fig.savefig(ART / "asian_context.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 saved")
    fig2(); print("fig2 saved")
    fig3(); print("fig3 saved")
