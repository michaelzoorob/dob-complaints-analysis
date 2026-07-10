"""
Ownership churn and complaint exposure (the churn post).

For every residential lot, the year of the most recent deed (ACRIS, deed-type
documents since 1985) is binned into purchase eras. Outcomes are 2020 through
May 2026 complaint counts. Models are the standard risk-panel PPML with
building and owner covariates, size + commercial-bin + census-tract fixed
effects, and tract-clustered SEs, so the gradient compares same-size lots on
the same blocks. The omitted era is pre-1995.

Outputs:
  data/analysis/risk_models/churn_estimates.csv   (era coefficients + raw rates)
  data/analysis/blog_posts/artifacts/churn_gradient.png
"""

import sqlite3
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import owner_models as om

warnings.filterwarnings("ignore")
OUT = config.DATA_DIR / "analysis" / "risk_models" / "churn_estimates.csv"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
ERAS = [("d9504", "1995-2004"), ("d0514", "2005-2014"), ("d1519", "2015-2019"), ("d20", "2020-2026")]

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; RED = "#e34948"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load():
    df = om.load_frame()
    df["bbl_key"] = df["bbl_key"].astype(str)
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    deeds = pd.read_sql_query("""
        SELECT l.borough || printf('%05d', CAST(l.block AS INT))
                         || printf('%04d', CAST(l.lot AS INT)) AS bbl_key,
               MAX(substr(m.doc_date, 1, 4)) AS yr
        FROM acris_master m JOIN acris_legals l ON m.document_id = l.document_id
        WHERE m.doc_type LIKE 'DEED%' AND m.doc_date IS NOT NULL AND m.doc_date >= '1985'
        GROUP BY 1""", conn)
    conn.close()
    deeds["bbl_key"] = deeds["bbl_key"].astype(str)
    deeds["yr"] = pd.to_numeric(deeds["yr"], errors="coerce")
    df = df.merge(deeds, on="bbl_key", how="left")
    df["d9504"] = df["yr"].between(1995, 2004).astype(int)
    df["d0514"] = df["yr"].between(2005, 2014).astype(int)
    df["d1519"] = df["yr"].between(2015, 2019).astype(int)
    df["d20"] = (df["yr"] >= 2020).astype(int)
    df["dmiss"] = df["yr"].isna().astype(int)
    return df


def main():
    df = load()
    print(f"panel {len(df):,}; deed matched {df['yr'].notna().mean():.1%}")
    rows = []
    raw_bins = [("pre-1995", df["yr"] < 1995)] + \
               [(lab, {"1995-2004": df["yr"].between(1995, 2004),
                       "2005-2014": df["yr"].between(2005, 2014),
                       "2015-2019": df["yr"].between(2015, 2019),
                       "2020-2026": df["yr"] >= 2020}[lab]) for _, lab in ERAS] + \
               [("no deed found", df["yr"].isna())]
    for lab, mask in raw_bins:
        s = df[mask]
        rows.append(dict(block="raw", era=lab, outcome="complaints_per_100",
                         b=s["n_complaints"].mean() * 100, se=np.nan, t=np.nan, n=len(s)))
        print(f"  raw {lab:<14} {len(s)/len(df)*100:5.1f}% of lots, "
              f"complaints/100 {s['n_complaints'].mean()*100:6.1f}")

    X = " + ".join(list(om.BUILDING_COVARS) + list(om.OWNER_COVARS)
                   + [e for e, _ in ERAS] + ["dmiss"])
    est = {}
    for y, lab in [("n_complaints", "all complaints"), ("n_conv", "conversion complaints")]:
        m = pf.fepois(f"{y} ~ {X} | size_bin + comm_bin + bct2020", data=df,
                      vcov={"CRV1": "bct2020"})
        est[lab] = []
        for var, era in ERAS:
            b, se = float(m.coef()[var]), float(m.se()[var])
            est[lab].append((era, b, se))
            rows.append(dict(block="ppml", era=era, outcome=lab, b=b, se=se,
                             t=b / se, n=int(m._N)))
            print(f"  {lab:<22} {era:<10} {(np.exp(b)-1)*100:+6.1f}% (t={b/se:+.1f})")
        bd, sed = float(m.coef()["dmiss"]), float(m.se()["dmiss"])
        rows.append(dict(block="ppml", era="no deed found", outcome=lab, b=bd, se=sed,
                         t=bd / sed, n=int(m._N)))
    rows.append(dict(block="meta", era="panel", outcome="residential_lots",
                     b=len(df), se=np.nan, t=np.nan, n=len(df)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved -> {OUT}")

    fig, ax = plt.subplots(figsize=(7.4, 4.6),
                           gridspec_kw={"left": 0.10, "right": 0.965,
                                        "top": 0.78, "bottom": 0.13})
    x = np.arange(len(ERAS))
    for lab, color, dx in [("all complaints", BLUE, -0.07),
                           ("conversion complaints", RED, +0.07)]:
        eff = np.array([(np.exp(b) - 1) * 100 for _, b, _ in est[lab]])
        lo = np.array([(np.exp(b - 1.96 * se) - 1) * 100 for _, b, se in est[lab]])
        hi = np.array([(np.exp(b + 1.96 * se) - 1) * 100 for _, b, se in est[lab]])
        ax.errorbar(x + dx, eff, yerr=[eff - lo, hi - eff], fmt="o-", ms=7, lw=1.4,
                    color=color, ecolor=color, elinewidth=1.6, capsize=3, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=lab)
    ax.axhline(0, color=ZERO, lw=1.1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([era for _, era in ERAS], fontsize=10)
    ax.set_xlabel("year of the current owner's deed", fontsize=9.5)
    ax.set_ylabel("% more complaints than pre-1995 owners", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9.2, loc="upper left")
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Recently purchased homes draw far more complaints",
                 x=0.02, y=0.975, ha="left", fontsize=12.5, color=INK, weight="semibold")
    fig.text(0.02, 0.885,
             "766,382 residential lots. Poisson models of 2020 through May 2026 complaint counts on purchase era,\n"
             "within census tract and building-size class, with the full building and owner covariates. Whiskers are\n"
             "95% confidence intervals, clustered by tract. The omitted group is owners who bought before 1995.",
             fontsize=8.8, color=MUTED, va="top")
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "churn_gradient.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
