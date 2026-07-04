"""
Census-tract choropleths for the descriptive overview post.

Map 1 (desc_map_raw.png): complaints per 100 residential units, 2020-May 2026.
Map 2 (desc_map_residual.png): observed / expected complaints, where expected
comes from a property-level Poisson of complaint counts on building
characteristics only (exact size bins, ownership type, owner-occupancy,
co-op/condo, construction era, mixed use, floor area per unit, assessed-value
rank, prior violations, multi-building lots, M-zoning). Geography enters the
model nowhere, so the ratio shows where complaint volume runs above or below
what the housing stock alone predicts.

Unit: 2020 census tract (matches the model FE unit); denominator: PLUTO
residential units; tracts with <200 units drawn in gray.

Inputs: property_risk_panel.csv.gz, data/nyct2020.geojson
Outputs: data/analysis/blog_posts/artifacts/desc_map_{raw,residual}.png
         + console report of extreme tracts (with NTA names)
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import risk_factor_models as rfm

ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
GEO = config.DATA_DIR / "nyct2020.geojson"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
NODATA = "#ebe9e3"
BLUE = "#2a78d6"
RED = "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

MIN_UNITS = 200

RAW_CMAP = LinearSegmentedColormap.from_list("rawblue", ["#eef3fb", "#9cc0ea", BLUE, "#123a6b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", [BLUE, "#8fb8e6", "#f2f1ec", "#f0a29d", RED])


def tract_table() -> pd.DataFrame:
    df = rfm.load_frame()
    covars = [c for c in rfm.BUILDING_COVARS]
    size_d = pd.get_dummies(df["size_bin"], prefix="sz", drop_first=True).astype(float)
    df["log2_units"] = np.log2(df["unitsres"].clip(lower=1))
    X = pd.concat([df[covars].astype(float), df[["log2_units"]], size_d], axis=1)
    X = sm.add_constant(X)
    y = df["n_complaints"].astype(float)
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    df["expected"] = model.predict(X)
    print(f"Poisson fit on {len(df):,} lots; sum obs {y.sum():,.0f} vs sum exp {df['expected'].sum():,.0f}")

    t = df.groupby("bct2020").agg(obs=("n_complaints", "sum"), exp=("expected", "sum"),
                                  units=("unitsres", "sum")).reset_index()
    t["per100"] = t.obs / t.units * 100
    t["ratio"] = t.obs / t.exp
    t["ok"] = t.units >= MIN_UNITS
    return t


def draw(gdf, col, cmap, norm, title, subtitle, cbar_label, fname, cbar_ticks=None,
         cbar_ticklabels=None):
    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=160)
    ax.set_axis_off()
    okmask = gdf["ok"].fillna(False).astype(bool)
    base = gdf[~okmask]
    base.plot(ax=ax, color=NODATA, edgecolor=SURFACE, linewidth=0.2)
    lit = gdf[okmask]
    lit.plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor=SURFACE, linewidth=0.2)
    gdf.dissolve("borocode").boundary.plot(ax=ax, color=INK2, linewidth=0.7)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color=INK, pad=16)
    ax.text(0, 1.005, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm_, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01,
                      aspect=32, shrink=0.62)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2, labelsize=10)
    cb.set_label(cbar_label, fontsize=11, color=INK2)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
        cb.set_ticklabels(cbar_ticklabels)
    fig.tight_layout()
    fig.savefig(ART / fname, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def main():
    t = tract_table()
    gdf = gpd.read_file(GEO)
    gdf = gdf.merge(t, left_on="boroct2020", right_on="bct2020", how="left")
    print(f"tracts: {len(gdf)}, with data: {gdf['obs'].notna().sum()}, "
          f"mapped (>= {MIN_UNITS} units): {(gdf['ok'] == True).sum()}")

    vmax = float(t.loc[t.ok, "per100"].quantile(0.98))
    draw(gdf, "per100", RAW_CMAP, Normalize(vmin=0, vmax=vmax),
         "Building complaints per 100 residential units",
         "2020-May 2026 · 2020 census tracts · gray = fewer than 200 residential units",
         "complaints per 100 residential units (top 2% capped)",
         "desc_map_raw.png")

    gdf["log2ratio"] = np.log2(gdf["ratio"].where(gdf["ratio"] > 0))
    gdf["log2ratio"] = gdf["log2ratio"].clip(-1.322, 1.322)  # x0.4 .. x2.5
    draw(gdf, "log2ratio", DIV_CMAP, TwoSlopeNorm(vcenter=0, vmin=-1.322, vmax=1.322),
         "Complaints relative to what the buildings predict",
         "observed / expected from a Poisson model of building characteristics only "
         "(size, ownership, era, value, prior violations) · no geography in the model",
         "observed / expected complaints",
         "desc_map_residual.png",
         cbar_ticks=[-1.322, -1, 0, 1, 1.322],
         cbar_ticklabels=["×0.4", "×0.5", "×1", "×2", "×2.5"])

    lit = gdf[gdf["ok"].fillna(False).astype(bool)].copy()
    hi = lit.nlargest(12, "ratio")[["boroname", "ntaname", "obs", "exp", "ratio", "per100"]]
    lo = lit.nsmallest(12, "ratio")[["boroname", "ntaname", "obs", "exp", "ratio", "per100"]]
    print("\nhighest observed/expected:\n", hi.to_string(index=False))
    print("\nlowest observed/expected:\n", lo.to_string(index=False))
    nta = (lit.groupby(["boroname", "ntaname"])
           .apply(lambda g: pd.Series({"obs": g.obs.sum(), "exp": g.exp.sum()}), include_groups=False)
           .assign(ratio=lambda d: d.obs / d.exp).sort_values("ratio"))
    print("\nNTA-level extremes (low):\n", nta.head(8).round(2).to_string())
    print("\nNTA-level extremes (high):\n", nta.tail(8).round(2).to_string())


if __name__ == "__main__":
    main()
