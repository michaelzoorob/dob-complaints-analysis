"""
Figures for the descriptive overview post (post 0).

Fig 1: monthly complaint volume, 2020-01..2026-04 (partial May 2026 dropped)
Fig 2: complaint outcome shares (share of 774,944 scraped complaints)
Fig 3: top-10 complaint categories, 100% stacked outcome composition
Fig 4: complaints and disposition-violations per residential unit by size bin
Fig 5: owner-occupied (STAR) vs absentee, 2/3/4-unit homes, per-unit rates
Fig 6: complaints per residential unit by borough

Inputs: data/dob_complaints.db (fig 1-3), property_risk_panel.csv.gz (fig 4-6).
Writes data/analysis/blog_posts/artifacts/desc_*.png
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from disposition_codes import classify_disposition

ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
RM = config.DATA_DIR / "analysis" / "risk_models"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel.csv.gz"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
TINT = "#f2f1ec"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_timeline(conn):
    tl = pd.read_sql_query("""
        SELECT substr(o.date_entered,7,4)||'-'||substr(o.date_entered,1,2) AS ym, COUNT(*) n
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        GROUP BY 1 ORDER BY 1""", conn)
    tl = tl[tl.ym <= "2026-05"]  # window ends with complete May 2026
    tl.to_csv(RM / "desc_monthly.csv", index=False)
    x = pd.to_datetime(tl.ym + "-01")
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=160)
    ax.fill_between(x, tl.n, color=BLUE, alpha=0.10, linewidth=0)
    ax.plot(x, tl.n, color=BLUE, linewidth=2)
    style_ax(ax)
    ax.set_ylim(0, 16500)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("DOB complaints filed per month, 2020–2026", loc="left",
                 fontsize=15, fontweight="bold", color=INK, pad=14)
    n_total = int(tl.n.sum())
    ax.text(0, 1.015, f"{n_total:,} complaints scraped from BIS Web, January 2020 through May 2026",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    covid = x[tl.ym.tolist().index("2020-04")]
    ax.annotate("COVID-19 pause", xy=(covid, tl.n[tl.ym == "2020-04"].iat[0]),
                xytext=(covid + pd.Timedelta(days=110), 4600), fontsize=10.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    mean = tl.n.mean()
    ax.axhline(mean, color=ZERO_C, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(x.iloc[8], 2500, f"average {mean:,.0f} per month", fontsize=10.5, color=INK2)
    fig.tight_layout()
    fig.savefig(ART / "desc_timeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_outcomes(conn):
    rows = pd.read_sql_query("""
        SELECT o.disposition_code, COUNT(*) n FROM open_data o
        JOIN bis_scrape b USING(complaint_number) GROUP BY 1""", conn)
    rows["outc"] = rows.disposition_code.fillna("").astype(str).map(classify_disposition)
    g = rows.groupby("outc").n.sum()
    total = g.sum()
    (g.rename("n").to_frame().assign(share=lambda d: d.n / total)
       .to_csv(RM / "desc_outcome_shares.csv"))
    order = [
        ("no_violation", "No violation found"),
        ("violation", "Violation issued"),
        ("no_access", "No access — inspector couldn't get in"),
        ("referral", "Referred to another unit or agency"),
        ("other", "Administrative closure / other"),
        ("pending", "Still pending"),
    ]
    labels = [lab for k, lab in order]
    vals = [g.get(k, 0) for k, _ in order]
    colors = [BASE, RED, BLUE, GRID, GRID, GRID]
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=160)
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.62, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 6000, yi, f"{v:,}  ({v/total*100:.1f}%)", va="center",
                fontsize=11, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.set_xlim(0, 470000)
    ax.xaxis.set_visible(False)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_title("What happens to a DOB complaint", loc="left", fontsize=15,
                 fontweight="bold", color=INK, pad=26)
    ax.text(-0.28, 1.045, f"share of all {total:,} scraped complaints, 2020–May 2026",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(ART / "desc_outcomes.png", bbox_inches="tight")
    plt.close(fig)


CAT_LABELS = {
    "45": "Illegal conversion",
    "05": "Construction without a permit",
    "6S": "Elevator out (only one on property)",
    "7J": "Work w/o permit, occupied building",
    "6M": "Elevator out (multiple devices)",
    "04": "After-hours construction",
    "8A": "Construction-safety compliance",
    "7G": "Construction sweep (agency-initiated)",
    "73": "Failure to maintain",
    "1X": "DOB enforcement work order",
    "30": "Building shaking / unstable",
    "58": "Defective boiler",
    "23": "Sidewalk shed / scaffold defect",
}


def fig_categories(conn):
    cat = pd.read_sql_query("""
        SELECT b.category_code cat, o.disposition_code FROM bis_scrape b
        JOIN open_data o USING(complaint_number)""", conn)
    cat["outc"] = cat.disposition_code.fillna("").astype(str).map(classify_disposition)
    cat["code2"] = cat.cat.str.strip().str[:2]
    top = cat.code2.value_counts().head(10)
    QUOTED = ["04", "7G", "8A", "45", "6S", "05"]  # codes the posts cite by name
    codes = list(dict.fromkeys(list(top.index) + QUOTED))
    rows = []
    for code in codes:
        s = cat[cat.code2 == code]
        o = s.outc.value_counts(normalize=True)
        rows.append(dict(code=code, n=len(s), viol=o.get("violation", 0),
                         noviol=o.get("no_violation", 0), noacc=o.get("no_access", 0)))
    df = pd.DataFrame(rows)
    df.to_csv(RM / "desc_category_outcomes.csv", index=False)
    print(df.round(3).to_string(index=False))
    df = df.head(10)
    df["other"] = 1 - df.viol - df.noviol - df.noacc
    fig, ax = plt.subplots(figsize=(12.5, 6.6), dpi=160)
    y = np.arange(len(df))[::-1]
    left = np.zeros(len(df))
    for col, color, lab in [("viol", RED, "Violation"), ("noviol", BASE, "No violation"),
                            ("noacc", BLUE, "No access"), ("other", GRID, "Other")]:
        v = df[col].values * 100
        ax.barh(y, v, left=left, height=0.62, color=color, label=lab,
                edgecolor=SURFACE, linewidth=2)
        left += v
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(101.5, yi, f"n={r.n:,}", va="center", fontsize=10, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([CAT_LABELS.get(c, c) for c in df.code], fontsize=12, color=INK)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.10), ncol=4, frameon=False,
              fontsize=10.5, handlelength=1.1, handleheight=1.1)
    fig.suptitle("The ten biggest complaint categories, and how they end",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.925, "share of complaints by inspection outcome · 2020–May 2026",
             fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(ART / "desc_categories.png", bbox_inches="tight")
    plt.close(fig)


def fig_size(df):
    bins = [(1, 1), (2, 2), (3, 4), (5, 10), (11, 20), (21, 50), (51, 100), (101, 10**9)]
    labels = ["1 unit", "2", "3–4", "5–10", "11–20", "21–50",
              "51–100", "100+"]
    rows = []
    for (lo, hi), lab in zip(bins, labels):
        s = df[(df.unitsres >= lo) & (df.unitsres <= hi)]
        rows.append(dict(bin=lab, props=len(s), units=int(s.unitsres.sum()),
                         cpu=s.n_complaints.sum() / s.unitsres.sum(),
                         vpu=s.n_viol_disp.sum() / s.unitsres.sum()))
    s24 = df[(df.unitsres >= 2) & (df.unitsres <= 4)]
    csv_rows = rows + [dict(bin="2-4 combined", props=len(s24),
                            units=int(s24.unitsres.sum()),
                            cpu=s24.n_complaints.sum() / s24.unitsres.sum(),
                            vpu=s24.n_viol_disp.sum() / s24.unitsres.sum())]
    pd.DataFrame(csv_rows).assign(cpu_per100=lambda d: d.cpu * 100,
                                  vpu_per100=lambda d: d.vpu * 100) \
        .to_csv(RM / "desc_per_unit_size.csv", index=False)
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), dpi=160, sharex=True)
    x = np.arange(len(d))
    d["cpu"] *= 100
    d["vpu"] *= 100
    for ax, col, color, ttl in [(axes[0], "cpu", BLUE, "Complaints per 100 residential units"),
                                (axes[1], "vpu", RED, "Violations (dispositions) per 100 residential units")]:
        ax.bar(x, d[col], width=0.62, color=color)
        for xi, v in zip(x, d[col]):
            ax.text(xi, v + d[col].max() * 0.03, f"{v:.1f}", ha="center", fontsize=10.5,
                    color=INK2)
        style_ax(ax)
        ax.set_ylim(0, d[col].max() * 1.18)
        ax.set_title(ttl, loc="left", fontsize=13, fontweight="bold", color=INK, pad=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r.bin}\n{r.props:,} bldgs\n{r.units:,} units"
                             for _, r in d.iterrows()], fontsize=9.5, color=INK2)
    fig.suptitle("Per apartment, small buildings dominate the complaint system",
                 x=0.045, ha="left", fontsize=15, fontweight="bold", color=INK)
    axes[0].text(0, 1.14, "totals 2020–May 2026 over the PLUTO residential universe "
                 "(766,939 tax lots, 3.7M units)", transform=axes[0].transAxes,
                 fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(ART / "desc_per_unit_size.png", bbox_inches="tight")
    plt.close(fig)


def fig_star(df):
    rows = []
    for u in [2, 3, 4]:
        s = df[df.unitsres == u]
        for occ, lab in [(True, "Owner-occupied (STAR)"), (False, "Absentee-owned")]:
            g = s[s.owner_occ_star == occ]
            rows.append(dict(units=u, occ=lab, n=len(g),
                             cpu=g.n_complaints.sum() / g.unitsres.sum(),
                             vpu=g.n_viol_disp.sum() / g.unitsres.sum()))
    d = pd.DataFrame(rows)
    d["cpu"] *= 100
    d["vpu"] *= 100
    d.to_csv(RM / "desc_star_gap.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), dpi=160)
    for ax, col, ttl in [(axes[0], "cpu", "Complaints per 100 units"),
                         (axes[1], "vpu", "Violations per 100 units")]:
        x = np.arange(3)
        for k, (lab, color) in enumerate([("Owner-occupied (STAR)", AQUA),
                                          ("Absentee-owned", BLUE)]):
            v = d[d.occ == lab][col].values
            ax.bar(x + (k - 0.5) * 0.34, v, width=0.3, color=color, label=lab)
            for xi, vi in zip(x, v):
                ax.text(xi + (k - 0.5) * 0.34, vi + d[col].max() * 0.025, f"{vi:.1f}",
                        ha="center", fontsize=10, color=INK2)
        style_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(["2 units", "3 units", "4 units"], fontsize=11.5, color=INK)
        ax.set_ylim(0, d[col].max() * 1.30)
        ax.set_title(ttl, loc="left", fontsize=13, fontweight="bold", color=INK, pad=8)
    axes[0].legend(loc="upper left", frameon=False, fontsize=10.5)
    fig.suptitle("Same size home, different owner: owner-occupied vs. absentee, 2–4 unit homes",
                 x=0.04, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ART / "desc_star_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig_borough(df):
    b = df.groupby("borough").agg(compl=("n_complaints", "sum"),
                                  units=("unitsres", "sum")).reset_index()
    names = {"BK": "Brooklyn", "QN": "Queens", "SI": "Staten Island",
             "BX": "Bronx", "MN": "Manhattan"}
    b["name"] = b.borough.map(names)
    b["cpu"] = b.compl / b.units * 100
    b = b.sort_values("cpu")
    b.to_csv(RM / "desc_borough.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=160)
    y = np.arange(len(b))
    ax.barh(y, b.cpu, height=0.6, color=BLUE)
    for yi, (_, r) in zip(y, b.iterrows()):
        ax.text(r.cpu + 0.35, yi, f"{r.cpu:.1f}", va="center", fontsize=11.5, color=INK2)
        ax.text(0.35, yi, f"{r['name']}  ·  {r.units/1e6:.2f}M units", va="center",
                fontsize=11.5, color=SURFACE, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 23)
    ax.xaxis.set_visible(False)
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)
    fig.suptitle("Complaints per 100 residential units by borough, 2020–May 2026",
                 x=0.01, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.895, "complaints matched to residential tax lots ÷ residential units",
             fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(ART / "desc_borough.png", bbox_inches="tight")
    plt.close(fig)


ZERO_C = "#b9b7ac"

if __name__ == "__main__":
    conn = sqlite3.connect(str(config.DB_PATH))
    fig_timeline(conn)
    fig_outcomes(conn)
    fig_categories(conn)
    conn.close()
    panel = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str})
    fig_size(panel)
    fig_star(panel)
    fig_borough(panel)
    print("wrote", *(f.name for f in sorted(ART.glob("desc_*.png"))))
