"""
Two-panel census-tract choropleth for the proactive-enforcement post.

Panel A (left): agency-initiated discretionary inspections (family
discretionary_field, agency == 1 in the proactive events spine) per 1,000
residential risk-panel lots by 2020 census tract, 2020-May 2026. Sequential
blue, matching the raw complaint map.

Panel B (right): the reallocation gap made visible. Never-swept high-risk
buildings per 1,000 scored lots by tract, where high-risk = top decile of
the pre-2020 risk score from proactive_becker_margin.load_panel_scored()
(imported, so the feature build and logit are identical by construction)
and never-swept = no agency 7G event at the BBL in the events spine.
Sequential red (the house violation color).

Both panels mask tracts with fewer than 50 lots in their denominator
(looser than the 200-unit masking in make_complaint_maps.py: the 200-lot
rule grayed out most of Manhattan, whose tracts hold few residential
lots, and Manhattan is the story's core geography).

Inputs : data/analysis/proactive/proactive_events.csv.gz
         data/analysis/property_risk_panel_v2.csv.gz
         data/dob_complaints.db (via proactive_becker_margin -> dob_ledger)
         data/nyct2020.geojson
Outputs: data/analysis/blog_posts/artifacts/proactive_map.png
         data/analysis/risk_models/proactive_map_tracts.csv
         + console report of top-10 NTAs per panel

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_maps.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import proactive_becker_margin as becker

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
GEO = config.DATA_DIR / "nyct2020.geojson"
OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
WINDOW = "2020-01..2026-05"

# house style (constants from scripts/make_complaint_maps.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
NODATA = "#ebe9e3"
BLUE = "#2a78d6"
RED = "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

MIN_LOTS = 50   # mask tracts with fewer panel lots (200 grayed out most of
                # Manhattan, where tracts hold few residential lots)

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "seqblue", ["#eef3fb", "#9cc0ea", BLUE, "#123a6b"])
RED_CMAP = LinearSegmentedColormap.from_list(
    "seqred", ["#fbefee", "#f0a29d", RED, "#6b1414"])


def tract_table() -> tuple[pd.DataFrame, dict]:
    """One row per tract: discretionary intensity + never-swept high-risk rate."""
    diag = {}

    # ── panel A numerator: discretionary agency events per tract ────────
    ev = pd.read_csv(SPINE, usecols=["family", "agency", "bct2020"])
    ev = ev[(ev["family"] == "discretionary_field") & (ev["agency"] == 1)]
    diag["disc_events_total"] = len(ev)
    ev = ev[ev["bct2020"].notna()].copy()
    diag["disc_events_no_tract"] = diag["disc_events_total"] - len(ev)
    ev["bct"] = becker.norm_tract(ev["bct2020"])
    n_disc = ev.groupby("bct").size().rename("n_disc_events")

    # ── panel A denominator: all panel lots per tract ───────────────────
    lots = pd.read_csv(PANEL, usecols=["bct2020"], dtype={"bct2020": str})
    panel_lots = lots.groupby("bct2020").size().rename("panel_lots")
    diag["panel_lots_total"] = len(lots)

    # ── panel B: score the panel with becker's exact feature build+logit,
    #    flag 7G-swept BBLs exactly as becker part (b) does ──────────────
    df, auc, k = becker.load_panel_scored()
    diag["model_auc"] = auc
    diag["model_params"] = k
    diag["scored_lots_total"] = len(df)

    ev7g = pd.read_csv(SPINE, usecols=["category_prefix", "agency", "bbl"],
                       dtype={"bbl": str})
    ev7g = ev7g[(ev7g["category_prefix"] == "7G") & (ev7g["agency"] == 1)]
    bbls = set(ev7g["bbl"].dropna().astype(str))
    df["swept7g"] = df["bbl_key"].isin(bbls).astype(int)

    cut = df["p_hat"].quantile(0.90)
    df["highrisk"] = (df["p_hat"] >= cut).astype(int)
    df["hr_unswept"] = ((df["highrisk"] == 1) & (df["swept7g"] == 0)).astype(int)
    diag["risk_score_p90_cut"] = float(cut)
    diag["n_highrisk"] = int(df["highrisk"].sum())
    diag["n_highrisk_swept"] = int((df["highrisk"] & df["swept7g"]).sum())
    diag["n_highrisk_never_swept"] = int(df["hr_unswept"].sum())

    tb = df.groupby("bct2020").agg(scored_lots=("bbl_key", "size"),
                                   n_highrisk=("highrisk", "sum"),
                                   n_hr_never_swept=("hr_unswept", "sum"))

    # ── assemble ─────────────────────────────────────────────────────────
    t = pd.concat([panel_lots, n_disc, tb], axis=1).reset_index()
    t = t.rename(columns={"index": "bct2020"})
    for c in ["n_disc_events", "scored_lots", "n_highrisk", "n_hr_never_swept"]:
        t[c] = t[c].fillna(0).astype(int)
    diag["disc_events_no_panel_tract"] = (diag["disc_events_total"]
                                          - diag["disc_events_no_tract"]
                                          - int(t["n_disc_events"].sum()))

    t["disc_per_1000"] = t["n_disc_events"] / t["panel_lots"] * 1000.0
    t["hr_never_swept_per_1000"] = np.where(
        t["scored_lots"] > 0, t["n_hr_never_swept"] / t["scored_lots"] * 1000.0,
        np.nan)
    t["ok_a"] = t["panel_lots"].fillna(0) >= MIN_LOTS
    t["ok_b"] = t["scored_lots"] >= MIN_LOTS
    t["window"] = WINDOW
    return t, diag


def draw_panel(fig, ax, gdf, col, okcol, cmap, vmax, title, subtitle, cbar_label):
    ax.set_axis_off()
    okmask = gdf[okcol].fillna(False).astype(bool)
    gdf[~okmask].plot(ax=ax, color=NODATA, edgecolor=SURFACE, linewidth=0.2)
    gdf[okmask].plot(ax=ax, column=col, cmap=cmap, norm=Normalize(0, vmax),
                     edgecolor=SURFACE, linewidth=0.2)
    gdf.dissolve("borocode").boundary.plot(ax=ax, color=INK2, linewidth=0.7)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK,
                 pad=20)
    ax.text(0, 1.008, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, vmax))
    cb = fig.colorbar(sm_, ax=ax, orientation="horizontal", fraction=0.035,
                      pad=0.01, aspect=32, shrink=0.62)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2, labelsize=10)
    cb.set_label(cbar_label, fontsize=11, color=INK2)


def nta_report(gdf, num, den, label, n=10):
    """NTA rates over ALL tracts (masking is a per-tract display choice;
    NTA-level denominators are big enough to be honest), NTA den >= MIN_LOTS."""
    sub = gdf[gdf[den].fillna(0) > 0]
    nta = (sub.groupby(["boroname", "ntaname"])
           .agg(num=(num, "sum"), den=(den, "sum"), tracts=(num, "size"))
           .reset_index())
    nta = nta[nta["den"] >= MIN_LOTS]
    nta["per_1000"] = nta["num"] / nta["den"] * 1000.0
    top = nta.sort_values("per_1000", ascending=False).head(n)
    print(f"\ntop {n} NTAs, {label}:")
    print(top[["boroname", "ntaname", "num", "den", "tracts", "per_1000"]]
          .round(1).to_string(index=False))
    return top


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    print("[1/3] tract table (events + becker-scored panel)")
    t, diag = tract_table()
    for k, v in diag.items():
        print(f"  {k}: {v:,.3f}" if isinstance(v, float) else f"  {k}: {v:,}")

    gdf = gpd.read_file(GEO)
    gdf = gdf.merge(t, left_on="boroct2020", right_on="bct2020", how="left")
    print(f"  tracts in geojson {len(gdf)}, with panel lots "
          f"{gdf['panel_lots'].notna().sum()}, mapped A {(gdf['ok_a'] == True).sum()}, "
          f"mapped B {(gdf['ok_b'] == True).sum()}")

    csv_cols = ["boroct2020", "boroname", "ntaname", "panel_lots", "scored_lots",
                "n_disc_events", "disc_per_1000", "n_highrisk",
                "n_hr_never_swept", "hr_never_swept_per_1000", "ok_a", "ok_b",
                "window"]
    out = gdf[csv_cols].rename(columns={"boroct2020": "bct2020"})
    out.to_csv(OUT / "proactive_map_tracts.csv", index=False)

    print("[2/3] figure")
    lit_a = gdf.loc[gdf["ok_a"] == True, "disc_per_1000"]
    lit_b = gdf.loc[gdf["ok_b"] == True, "hr_never_swept_per_1000"]
    vmax_a = float(lit_a.quantile(0.98))
    vmax_b = float(lit_b.quantile(0.98))

    fig, axes = plt.subplots(1, 2, figsize=(19, 10.5), dpi=160)
    draw_panel(
        fig, axes[0], gdf, "disc_per_1000", "ok_a", BLUE_CMAP, vmax_a,
        "Where DOB's discretionary inspections go",
        "agency-initiated discretionary inspections per 1,000 residential lots "
        f"· 2020–May 2026 · gray = under {MIN_LOTS} lots",
        "inspections per 1,000 residential lots (top 2% capped)")
    draw_panel(
        fig, axes[1], gdf, "hr_never_swept_per_1000", "ok_b", RED_CMAP, vmax_b,
        "High-risk buildings the sweeps never reached",
        "never-swept lots in the top decile of pre-2020 predicted risk, "
        f"per 1,000 scored lots · gray = under {MIN_LOTS} lots",
        "never-swept high-risk lots per 1,000 lots (top 2% capped)")
    fig.tight_layout()
    fig.savefig(ART / "proactive_map.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {ART / 'proactive_map.png'}")
    print(f"  wrote {OUT / 'proactive_map_tracts.csv'}")

    print("[3/3] NTA report")
    nta_report(gdf, "n_disc_events", "panel_lots",
               "discretionary inspections per 1,000 lots (panel A)")
    nta_report(gdf, "n_hr_never_swept", "scored_lots",
               "never-swept high-risk lots per 1,000 scored lots (panel B)")

    withlots = gdf[gdf["panel_lots"].notna()]
    boro = (withlots.groupby("boroname")
            .agg(tracts=("ok_a", "size"), lit=("ok_a", "sum"))
            .assign(masked_share=lambda d: 1 - d.lit / d.tracts))
    print("\nmask shares by borough (tracts with panel lots, panel A rule):")
    print(boro.round(2).to_string())

    both = gdf[(gdf["ok_a"] == True) & (gdf["ok_b"] == True)]
    rho = both["disc_per_1000"].corr(both["hr_never_swept_per_1000"],
                                     method="spearman")
    print(f"\nspearman(disc intensity, never-swept high-risk rate) across "
          f"{len(both)} tracts: {rho:+.3f}")
    print(f"vmax A (p98) {vmax_a:.1f}, max {lit_a.max():.1f}; "
          f"vmax B (p98) {vmax_b:.1f}, max {lit_b.max():.1f}")


if __name__ == "__main__":
    main()
