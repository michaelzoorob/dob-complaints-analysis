"""
Raw-levels table for the Asian-subgroups post appendix.

Renders data/analysis/risk_models/asian_subgroup_descriptives.csv as a styled
table image: per-100-property raw levels by owner subgroup, White reference
last, Filipino/Japanese omitted (pooled in models). Replaces a hand-made PNG
whose "Properties" and "Complaints" headers collided; column anchors here are
padded and a renderer-measured gap assertion fails the build on any collision.

Writes data/analysis/blog_posts/artifacts/asian_subgroup_rawtable.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
RM = config.DATA_DIR / "analysis" / "risk_models"
OUT = ART / "asian_subgroup_rawtable.png"

# house style (scripts/make_descriptive_figures.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
TINT = "#f2f1ec"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

# display label, CSV key — same row order as the live table
ROWS = [
    ("Chinese", "chinese"),
    ("Bangladeshi / Pakistani", "muslim_sa"),
    ("Indo-Caribbean", "indo_caribbean"),
    ("Singh", "singh"),
    ("Indian / other South Asian", "indian"),
    ("Korean", "korean"),
    ("Vietnamese", "vietnamese"),
    ("Sikh / Punjabi", "sikh_punjabi"),
    ("Nepali / Himalayan", "nepali_himalayan"),
    ("White (reference)", "white_ref"),
]

# header, CSV column, right-anchor x in figure fraction (generously spaced)
COLS = [
    ("Properties", "n_props", 0.400),
    ("Complaints", "compl_100", 0.525),
    ("Conversion\ncomplaints", "conv_100", 0.655),
    ("Disposition\nviolations", "viol_100", 0.785),
    ("ECB\ncitations", "ecb_100", 0.875),
    ("Owner-\noccupied %", "owner_occ", 0.978),
]

T_LEFT, T_RIGHT = 0.045, 0.978
SUB_X = 0.055
TOP_Y, HEAD_Y, BOT_Y = 0.955, 0.855, 0.200
ROW_H = (HEAD_Y - BOT_Y) / len(ROWS)
MIN_GAP_PX = 18


def fmt(col, v):
    return f"{int(v):,}" if col == "n_props" else f"{v:.1f}"


def main():
    df = pd.read_csv(RM / "asian_subgroup_descriptives.csv").set_index("subgroup")
    n_fil = int(df.loc["filipino", "n_props"])
    n_jap = int(df.loc["japanese", "n_props"])

    fig = plt.figure(figsize=(12.2, 5.85), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # texts per column for the collision check: col 0 = subgroup, 1.. = numeric
    col_texts = [[] for _ in range(len(COLS) + 1)]

    # rules
    for y in (TOP_Y, HEAD_Y):
        ax.plot([T_LEFT, T_RIGHT], [y, y], color=BASE, linewidth=1.2)
    ax.plot([T_LEFT, T_RIGHT], [BOT_Y, BOT_Y], color=BASE, linewidth=1.2)

    # header
    head_c = (TOP_Y + HEAD_Y) / 2
    col_texts[0].append(ax.text(SUB_X, head_c, "Subgroup", ha="left", va="center",
                                fontsize=14, color=INK))
    for j, (head, _, xa) in enumerate(COLS):
        col_texts[j + 1].append(ax.text(xa, head_c, head, ha="right", va="center",
                                        fontsize=13, color=MUTED, linespacing=1.15))

    # body
    for i, (label, key) in enumerate(ROWS):
        y_top = HEAD_Y - i * ROW_H
        yc = y_top - ROW_H / 2
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((T_LEFT, y_top - ROW_H), T_RIGHT - T_LEFT,
                                       ROW_H, facecolor=TINT, edgecolor="none",
                                       zorder=0))
        col_texts[0].append(ax.text(SUB_X, yc, label, ha="left", va="center",
                                    fontsize=14.5, color=INK))
        for j, (_, col, xa) in enumerate(COLS):
            col_texts[j + 1].append(ax.text(xa, yc, fmt(col, df.loc[key, col]),
                                            ha="right", va="center",
                                            fontsize=14.5, color=INK))

    ax.text(0.03, 0.075,
            f"Raw levels per 100 properties, 2020 through May 2026. "
            f"Filipino (n={n_fil}) and Japanese (n={n_jap}) owners omitted "
            f"(pooled in models).",
            ha="left", va="center", fontsize=11, color=MUTED)

    # measured collision check: column bounding boxes must not touch
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    spans = []
    for texts in col_texts:
        boxes = [t.get_window_extent(renderer) for t in texts]
        spans.append((min(b.x0 for b in boxes), max(b.x1 for b in boxes)))
    w_px = fig.get_figwidth() * fig.dpi
    names = ["Subgroup"] + [h.replace("\n", " ") for h, _, _ in COLS]
    ok = True
    for k in range(len(spans) - 1):
        gap = spans[k + 1][0] - spans[k][1]
        print(f"gap {names[k]!r} -> {names[k + 1]!r}: {gap:.0f}px")
        if gap < MIN_GAP_PX:
            ok = False
    if spans[0][0] < 0 or spans[-1][1] > w_px:
        ok = False
        print("text overflows figure edge")
    assert ok, "column collision or overflow — widen anchors"

    fig.savefig(OUT)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
