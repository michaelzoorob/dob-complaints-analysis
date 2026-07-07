"""
Figure for the locked-door post (no-access outcomes).

artifacts/no_access_building.png, two panels:
  A. No-access share by complaint category (the biggest categories), showing
     the range from elevator (~1%) to illegal conversion (~73%).
  B. No-access share by PLUTO building class, all complaints and with the
     conversion-type categories (45/4G/4W) excluded, showing the gradient is
     architectural rather than case mix.

Sample: inspections ending in a violation, no violation, or no access,
2020 through May 2026. Styling matches the other blog figures.
"""
import sqlite3
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
from disposition_codes import classify_disposition

ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

CATS = [("45", "Illegal conversion"), ("05", "Construction without a permit"),
        ("7J", "After-hours work"), ("58", "Defective boiler"),
        ("83", "Construction safety sweep"), ("30", "Building shaking or unstable"),
        ("23", "Vacant open building"), ("6S", "Elevator")]
CONV = {"45", "4G", "4W"}
BGRP = [("B", "2-family house"), ("A", "1-family house"), ("C", "Walk-up apartments"),
        ("S", "Store with homes above"), ("D", "Elevator building")]


def load():
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    bc = ("CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2' "
          "WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' END")
    df = pd.read_sql_query(f"""
        SELECT o.disposition_code, o.complaint_category, p.bldgclass
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        LEFT JOIN pluto p ON p.borocode = {bc} AND p.block = b.block AND p.lot = b.lot
        WHERE o.disposition_code IS NOT NULL AND o.disposition_code != ''""", conn)
    conn.close()
    df["outcome"] = df["disposition_code"].apply(classify_disposition)
    d = df[df["outcome"].isin(["violation", "no_violation", "no_access"])].copy()
    d["na"] = (d["outcome"] == "no_access").astype(int)
    d["bletter"] = d["bldgclass"].astype(str).str[0]
    return d


def main():
    d = load()
    print(f"sample: {len(d):,}; no-access {d['na'].mean() * 100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 8.6),
                                   gridspec_kw={"left": 0.31, "right": 0.94,
                                                "top": 0.86, "bottom": 0.06,
                                                "hspace": 0.42})
    # Panel A: categories
    rows = []
    for code, lab in CATS:
        s = d[d["complaint_category"] == code]
        rows.append((lab, s["na"].mean() * 100, len(s)))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    ax1.barh(ys, [r[1] for r in rows], height=0.62, color=BLUE, zorder=3)
    for y, (lab, v, n) in zip(ys, rows):
        ax1.text(v + 1.2, y, f"{v:.0f}%", va="center", fontsize=10, color=INK)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax1.set_xlim(0, 84)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_title("A. Share of inspections ending with no access, by complaint type",
                  loc="left", fontsize=10.5, color=INK2, pad=8)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(labelsize=9)

    # Panel B: building classes, all vs conversion excluded
    labs, alls, excl = [], [], []
    for letter, lab in BGRP:
        s = d[d["bletter"] == letter]
        labs.append(lab)
        alls.append(s["na"].mean() * 100)
        excl.append(s[~s["complaint_category"].isin(CONV)]["na"].mean() * 100)
    ys = np.arange(len(labs))[::-1]
    ax2.barh(ys + 0.19, alls, height=0.36, color=BLUE, zorder=3, label="all complaints")
    ax2.barh(ys - 0.19, excl, height=0.36, color=RED, zorder=3,
             label="illegal-conversion complaints excluded")
    for y, a, e in zip(ys, alls, excl):
        ax2.text(a + 1.0, y + 0.19, f"{a:.0f}%", va="center", fontsize=9.5, color=INK)
        ax2.text(e + 1.0, y - 0.19, f"{e:.0f}%", va="center", fontsize=9.5, color=INK)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labs, fontsize=10, color=INK)
    ax2.set_xlim(0, 46)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_title("B. Share ending with no access, by building type",
                  loc="left", fontsize=10.5, color=INK2, pad=8)
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(labelsize=9)

    fig.suptitle("Where inspectors cannot get in", x=0.02, y=0.982, ha="left",
                 fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.94,
             f"{len(d):,} inspections ending in a violation, no violation, or no access, "
             f"2020 through May 2026.\nBuilding types are PLUTO building classes "
             f"(A, B, C, D, S) for the 83% of inspections matched to a tax lot.",
             fontsize=9, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "no_access_building.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved ->", out)


if __name__ == "__main__":
    main()
