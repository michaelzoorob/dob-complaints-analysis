"""
Becker margin for proactive enforcement (plan hypothesis #2).

Question: is DOB's discretionary proactive capacity allocated where the
marginal violation is? Two exercises:

(a) Inspection-level LPM on agency-initiated discretionary_field events
    (8A compliance, 7G sweeps, 1X EWO, 91 worker endangerment, ...).
    Outcome = violation disposition (disposition_codes taxonomy, pending
    dropped). Treatment = tract proactive-intensity decile, where intensity
    = agency discretionary events per 1,000 risk-panel lots over the full
    window (2020-01..2026-05). FE = category_prefix x year; SEs clustered
    by tract (the level the treatment varies). A flat or rising hit-rate
    gradient means heavily-visited tracts have not been pushed down to a
    lower marginal yield, i.e. allocation is not yield-equalizing.

(b) Reallocation exercise on 7G sweeps. Train a logit of any-violation-
    2020on (ECB citations OR deduped BIS+DOB NOW union ledger, per
    dob_ledger.py) on strictly pre-2020 / slow-moving panel observables:
    era, size (unit bins + log floor area), building-class letter,
    2010-19 violation history, tract poverty. NO 2020+ outcomes enter as
    regressors. Score all panel lots; compare mean predicted risk of lots
    that received a 7G sweep vs the equal-count highest-scoring
    never-swept lots; translate the gap into extra violations per 1,000
    inspections. In-sample calibration by predicted-risk decile is
    reported so the translation is honest.

Honest caveats carried in the outputs:
  - the training outcome is realized enforcement, not latent risk; lots
    are only cited when someone shows up, so predicted risk partly
    encodes where DOB already looks (biases AGAINST the never-swept
    top-decile lots, since unvisited high-risk lots can't show 2020+
    violations; the measured gap is if anything understated on that
    axis, but overstated to the extent sweeps themselves generated the
    violations that trained the model's history features);
  - PLUTO class/size and ACS tract poverty are current-vintage but
    slow-moving; 2010-19 history is strictly pre-window;
  - 7G targets construction sites including non-residential lots, so
    only sweeps at residential panel lots enter the comparison (match
    rate reported).

Inputs : data/analysis/proactive/proactive_events.csv.gz
         data/analysis/property_risk_panel_v2.csv.gz
         data/dob_complaints.db (dob_ledger union for the 2020+ outcome)
Outputs: data/analysis/risk_models/proactive_becker_estimates.csv
         data/analysis/risk_models/proactive_reallocation.csv
         data/analysis/blog_posts/artifacts/proactive_becker.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_becker_margin.py
"""

import re
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import dob_ledger

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
WINDOW = "2020-01..2026-05"

# house style (constants from scripts/make_descriptive_figures.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
TINT = "#f2f1ec"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def norm_tract(s: pd.Series) -> pd.Series:
    """bct2020 arrives float in the spine, str in the panel; normalize."""
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def clean_tidy(model, name, outcome, n=None):
    """pyfixest tidy -> house column names (matches violation_rate_models)."""
    t = model.tidy().reset_index()
    t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                 for c in t.columns]
    if "coefficient" in t.columns:
        t = t.rename(columns={"coefficient": "term"})
    elif "index" in t.columns:
        t = t.rename(columns={"index": "term"})
    t["model"] = name
    t["outcome"] = outcome
    t["n"] = model._N if n is None else n
    return t


# ── (a) hit rate by tract-intensity decile ──────────────────────────────

def load_events() -> tuple[pd.DataFrame, dict]:
    ev = pd.read_csv(SPINE, usecols=[
        "received_date", "category_prefix", "family", "agency",
        "bct2020", "outcome"])
    ev = ev[(ev["agency"] == 1) & (ev["family"] == "discretionary_field")]
    acct = {"n_events_total": len(ev)}
    ev = ev[ev["bct2020"].notna()].copy()
    acct["n_dropped_no_tract"] = acct["n_events_total"] - len(ev)
    ev["bct"] = norm_tract(ev["bct2020"])
    n0 = len(ev)
    ev = ev[ev["outcome"] != "pending"]
    acct["n_dropped_pending"] = n0 - len(ev)
    ev["hit100"] = (ev["outcome"] == "violation").astype(float) * 100.0
    ev["year"] = ev["received_date"].str[:4]
    ev["cat_year"] = ev["category_prefix"] + "_" + ev["year"]
    return ev, acct


def build_intensity(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    lots = pd.read_csv(PANEL, usecols=["bct2020"], dtype={"bct2020": str})
    lots_per_tract = lots.groupby("bct2020").size().rename("panel_lots")

    tr = ev.groupby("bct").size().rename("n_events").to_frame()
    tr = tr.join(lots_per_tract, how="left")
    acct = {"n_tracts_with_events": len(tr),
            "n_tracts_no_panel_lots": int(tr["panel_lots"].isna().sum())}
    dropped = ev["bct"].isin(tr.index[tr["panel_lots"].isna()])
    acct["n_events_dropped_no_panel_lots"] = int(dropped.sum())

    tr = tr[tr["panel_lots"].notna()].copy()
    tr["intensity"] = tr["n_events"] / tr["panel_lots"] * 1000.0
    tr["decile"] = (pd.qcut(tr["intensity"].rank(method="first"), 10,
                            labels=False) + 1)
    tr["decile"] = tr["decile"].astype(int).map("{:02d}".format)
    acct["n_tracts_decile"] = len(tr)
    acct["n_tracts_zero_events"] = int(
        (~lots_per_tract.index.isin(tr.index)).sum())

    ev = ev.merge(tr[["intensity", "decile"]], left_on="bct",
                  right_index=True, how="inner")
    ev["decile_num"] = ev["decile"].astype(int)
    return ev, tr, acct


def cluster_mean(y: np.ndarray, cl: np.ndarray):
    """Mean with CRV1-style cluster-robust SE (intercept-only OLS)."""
    ybar = y.mean()
    s = pd.Series(y - ybar).groupby(pd.Series(cl)).sum()
    g, n = len(s), len(y)
    v = g / (g - 1) * (s ** 2).sum() / n ** 2
    return ybar, float(np.sqrt(v)), g


def part_a(ev: pd.DataFrame, tr: pd.DataFrame, acct: dict) -> pd.DataFrame:
    rows = []
    vcov = {"CRV1": "bct"}
    out_name = "violation disposition per inspection (pp)"

    # raw hit rate by decile, cluster-robust CI (tract)
    for d, sub in ev.groupby("decile"):
        m, se, g = cluster_mean(sub["hit100"].to_numpy(), sub["bct"].to_numpy())
        t = tr[tr["decile"] == d]
        rows.append({
            "term": f"decile_{d}", "estimate": m, "std_error": se,
            "t_value": m / se, "pr(>|t|)": np.nan,
            "25pct": m - 1.96 * se, "975pct": m + 1.96 * se,
            "model": "raw_decile_mean", "outcome": out_name, "n": len(sub),
            "n_tracts": len(t), "mean_intensity": t["intensity"].mean(),
            "median_intensity": t["intensity"].median(),
        })

    # FE LPM: decile dummies (ref = decile 01), category_prefix x year FE
    m = pf.feols("hit100 ~ i(decile, ref='01') | cat_year", data=ev, vcov=vcov)
    t = clean_tidy(m, "lpm_fe_decile", out_name + ", vs decile 01")
    t["term"] = ["decile_" + re.search(r"(\d{2})\]?$", x).group(1)
                 for x in t["term"]]
    rows += t.to_dict("records")

    # FE LPM: linear in decile (summary slope), pooled and per category
    m = pf.feols("hit100 ~ decile_num | cat_year", data=ev, vcov=vcov)
    rows += clean_tidy(m, "lpm_fe_linear", out_name + ", per decile").to_dict("records")
    for pfx in ["7G", "8A", "1X"]:
        sub = ev[ev["category_prefix"] == pfx]
        m = pf.feols("hit100 ~ decile_num | year", data=sub, vcov=vcov)
        rows += clean_tidy(m, f"lpm_fe_linear__{pfx}",
                           out_name + ", per decile").to_dict("records")

    # sample accounting so the post can trace every count
    for k, v in acct.items():
        rows.append({"term": k, "estimate": v, "model": "sample",
                     "outcome": "count"})
    res = pd.DataFrame(rows)
    res["window"] = WINDOW
    return res


# ── (b) reallocation exercise on 7G sweeps ──────────────────────────────

PANEL_DTYPES = {"bct2020": str, "size_bin": str, "borocode": str}


def load_panel_scored():
    df = pd.read_csv(PANEL, dtype=PANEL_DTYPES)
    df["bbl_key"] = df["bbl_key"].astype(str)

    # outcome: any violation 2020on = ECB citation OR deduped DOB union
    conn = sqlite3.connect(str(config.DB_PATH), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000;")
    dob = dob_ledger.counts_by_bbl(conn, 2020, 2026, "n_dobviol_2020on")
    conn.close()
    df = df.merge(dob.reset_index().assign(
        bbl_key=lambda d: d["bbl_key"].astype(str)), on="bbl_key", how="left")
    df["n_dobviol_2020on"] = df["n_dobviol_2020on"].fillna(0).astype(int)
    df["any_viol_2020on"] = ((df["n_ecb_2020on"] > 0)
                             | (df["n_dobviol_2020on"] > 0)).astype(int)

    # features: era, size, class, 2010-19 history, tract poverty. Nothing
    # measured 2020+ enters the design matrix.
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    ba = pd.to_numeric(df["bldgarea"], errors="coerce")
    lb = np.log(ba.where(ba > 0))
    df["no_bldgarea"] = lb.isna().astype(int)
    df["log_bldgarea"] = lb.fillna(lb.median())
    cl = df["class_letter"].astype(str)
    keep = cl.value_counts()
    cl = cl.where(cl.map(keep) >= 1000, "rare")
    df["class_grp"] = cl
    df["log1p_ecb_hist"] = np.log1p(df["n_ecb_hist"])
    df["log1p_dobviol_hist"] = np.log1p(df["n_dobviol_hist"])

    n0 = len(df)
    df = df[df["tract_poverty"].notna() & df["size_bin"].notna()].copy()
    print(f"  panel scored sample {len(df):,} of {n0:,} lots")

    X = pd.concat([
        pd.DataFrame({"const": 1.0}, index=df.index),
        df[["era_pre1940", "era_4079", "era_8099", "era_unknown",
            "log_bldgarea", "no_bldgarea",
            "log1p_ecb_hist", "log1p_dobviol_hist", "any_prior_viol",
            "tract_poverty"]].astype(float),
        pd.get_dummies(df["size_bin"], prefix="size", dtype=float).drop(
            columns=["size_1"]),
        pd.get_dummies(df["class_grp"], prefix="cls", dtype=float).drop(
            columns=["cls_A"]),
    ], axis=1)
    y = df["any_viol_2020on"].to_numpy(float)
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    df["p_hat"] = np.asarray(model.predict(X))
    df["p100"] = df["p_hat"] * 100.0

    # rank-based AUC (no sklearn in this venv)
    r = pd.Series(df["p_hat"]).rank().to_numpy()
    n1, n_all = y.sum(), len(y)
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (n_all - n1))
    print(f"  logit params {X.shape[1]}, in-sample AUC {auc:.3f}, "
          f"base rate {y.mean():.4f}")
    return df, auc, X.shape[1]


def part_b(df: pd.DataFrame, auc: float, k_params: int):
    ev7g = pd.read_csv(SPINE, usecols=["category_prefix", "agency", "bbl"],
                       dtype={"bbl": str})
    ev7g = ev7g[(ev7g["category_prefix"] == "7G") & (ev7g["agency"] == 1)]
    bbls = set(ev7g["bbl"].dropna().astype(str))
    df["swept7g"] = df["bbl_key"].isin(bbls).astype(int)
    n_swept = int(df["swept7g"].sum())

    unswept = df[df["swept7g"] == 0]
    top = unswept.nlargest(n_swept, "p_hat")
    swept = df[df["swept7g"] == 1]

    # gap with tract-clustered CI: stack the two disjoint groups
    stack = pd.concat([
        swept.assign(model_pick=0), top.assign(model_pick=1)],
        ignore_index=True)
    m = pf.feols("p100 ~ model_pick", data=stack, vcov={"CRV1": "bct2020"})
    t = clean_tidy(m, "gap", "predicted risk (pp)")
    gap = t.loc[t["term"] == "model_pick"].iloc[0]

    rows = []

    def add(term, value, note=""):
        rows.append({"block": "headline", "term": term,
                     "value": float(value), "note": note})

    add("n_7g_events", len(ev7g), "agency 7G sweep events in spine")
    add("n_7g_bbls", len(bbls), "unique swept BBLs (incl. non-residential)")
    add("n_swept_panel_lots", n_swept,
        "swept lots matched to residential risk panel")
    add("swept_panel_match_share", n_swept / max(len(bbls), 1))
    add("mean_pred_swept_pct", swept["p100"].mean(),
        "mean predicted any-violation-2020on risk, swept lots")
    add("mean_pred_top_unswept_pct", top["p100"].mean(),
        "equal-count highest-scoring never-swept lots")
    add("gap_pp", gap["estimate"], "top-unswept minus swept, tract-clustered")
    add("gap_pp_se", gap["std_error"])
    add("gap_pp_ci_lo", gap["25pct"])
    add("gap_pp_ci_hi", gap["975pct"])
    add("extra_viol_per_1000", gap["estimate"] * 10,
        "gap x 10 = extra violations per 1,000 inspections")
    add("extra_viol_per_1000_ci_lo", gap["25pct"] * 10)
    add("extra_viol_per_1000_ci_hi", gap["975pct"] * 10)
    add("actual_rate_swept_pct", swept["any_viol_2020on"].mean() * 100,
        "realized 2020+ rate; partly caused by the sweeps themselves")
    add("actual_rate_top_unswept_pct", top["any_viol_2020on"].mean() * 100,
        "realized rate despite no sweep (understates latent risk)")
    # group composition: makes the saturation caveat traceable (the model's
    # top lots are big buildings whose 77-month any-violation probability is
    # near 1 under the statutory periodic regimes)
    add("units_median_swept", swept["unitsres"].median())
    add("units_median_top_unswept", top["unitsres"].median())
    add("share_100plus_units_swept", (swept["unitsres"] >= 100).mean())
    add("share_100plus_units_top_unswept", (top["unitsres"] >= 100).mean())
    add("share_prior_viol_swept", swept["any_prior_viol"].mean(),
        "any 2010-19 violation history")
    add("share_prior_viol_top_unswept", top["any_prior_viol"].mean())
    add("model_auc", auc, "in-sample rank AUC")
    add("model_params", k_params)
    add("panel_base_rate_pct", df["any_viol_2020on"].mean() * 100)
    add("n_panel_scored", len(df))

    # calibration by predicted-risk decile (in-sample; params << n)
    df["cal_decile"] = pd.qcut(df["p_hat"].rank(method="first"), 10,
                               labels=False) + 1
    cal = df.groupby("cal_decile").agg(
        mean_pred_pct=("p100", "mean"),
        actual_rate_pct=("any_viol_2020on", lambda s: s.mean() * 100),
        n=("p_hat", "size")).reset_index()
    for _, r in cal.iterrows():
        rows.append({"block": "calibration",
                     "term": f"pred_decile_{int(r['cal_decile']):02d}",
                     "value": r["mean_pred_pct"],
                     "actual_rate_pct": r["actual_rate_pct"],
                     "n": int(r["n"]),
                     "note": "value = mean predicted risk (pct)"})
    res = pd.DataFrame(rows)
    res["window"] = WINDOW
    return res, swept["p100"].mean(), top["p100"].mean(), gap


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(res_a: pd.DataFrame, swept_p: float, top_p: float, gap):
    raw = res_a[res_a["model"] == "raw_decile_mean"].copy()
    raw["d"] = raw["term"].str[-2:].astype(int)
    raw = raw.sort_values("d")
    fe = res_a[res_a["model"] == "lpm_fe_decile"].copy()
    fe["d"] = fe["term"].str[-2:].astype(int)
    fe = fe.sort_values("d")
    anchor = raw.loc[raw["d"] == 1, "estimate"].iat[0]
    slope = res_a[(res_a["model"] == "lpm_fe_linear")
                  & (res_a["term"] == "decile_num")].iloc[0]

    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=160)
    x = raw["d"].to_numpy()

    ax.errorbar(x - 0.13, raw["estimate"], yerr=1.96 * raw["std_error"],
                fmt="o", color=MUTED, ecolor=MUTED, elinewidth=1.6,
                capsize=3, markersize=7, label="raw hit rate")
    fe_lvl = np.r_[anchor, anchor + fe["estimate"].to_numpy()]
    fe_lo = np.r_[np.nan, anchor + fe["25pct"].to_numpy()]
    fe_hi = np.r_[np.nan, anchor + fe["975pct"].to_numpy()]
    xs = np.arange(1, 11)
    ax.plot(xs + 0.13, fe_lvl, color=BLUE, linewidth=2, zorder=3)
    ax.errorbar(xs + 0.13, fe_lvl,
                yerr=[fe_lvl - np.where(np.isnan(fe_lo), fe_lvl, fe_lo),
                      np.where(np.isnan(fe_hi), fe_lvl, fe_hi) - fe_lvl],
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.6, capsize=3,
                markersize=7, zorder=4,
                label="within category × year (anchored at decile 1)")

    style_ax(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}" for d in xs], fontsize=11)
    ax.set_xlabel("tract proactive-intensity decile "
                  "(agency discretionary inspections per 1,000 residential lots, "
                  "2020–May 2026)", fontsize=11)
    ax.set_ylabel("violations per 100 inspections", fontsize=11)
    ymax = max(raw["975pct"].max(), np.nanmax(fe_hi)) * 1.45
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    ax.set_title("Where DOB inspects most, does it find more?",
                 loc="left", fontsize=15, fontweight="bold", color=INK, pad=14)
    ax.text(0, 1.015, "hit rate of agency-initiated discretionary inspections "
            "by how intensively DOB works the tract · 95% CIs, "
            "SEs clustered by tract",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.text(0.02, 0.955,
            f"within-category slope {slope['estimate']:+.2f} pp per decile "
            f"(95% CI {slope['25pct']:+.2f} to {slope['975pct']:+.2f})",
            transform=ax.transAxes, fontsize=10.5, color=INK2, va="top")

    fig.tight_layout()

    # inset: reallocation bar (7G sweeps vs model-picked lots), placed in
    # the empty band below the hit-rate series; RED = the house violation
    # color (BLUE stays reserved for the adjusted series above)
    axi = fig.add_axes([0.15, 0.22, 0.17, 0.24])
    axi.set_facecolor(TINT)
    bars = axi.bar([0, 1], [swept_p, top_p], width=0.62, color=[BASE, RED])
    for b, v in zip(bars, [swept_p, top_p]):
        axi.text(b.get_x() + b.get_width() / 2, v + top_p * 0.04, f"{v:.1f}%",
                 ha="center", fontsize=9.5, color=INK2)
    axi.set_xticks([0, 1])
    axi.set_xticklabels(["lots DOB\nswept (7G)", "highest-risk\nnever-swept"],
                        fontsize=8.5, color=INK)
    axi.set_yticks([])
    for s in ["top", "right", "left"]:
        axi.spines[s].set_visible(False)
    axi.set_ylim(0, top_p * 1.32)
    axi.set_title("predicted any-violation risk,\nequal-count comparison",
                  fontsize=9, color=INK2, pad=5)
    axi.text(0.5, -0.46,
             f"+{gap['estimate'] * 10:,.0f} violations per 1,000 inspections\n"
             f"(95% CI {gap['25pct'] * 10:,.0f}–{gap['975pct'] * 10:,.0f})",
             transform=axi.transAxes, ha="center", fontsize=9,
             color=INK2, fontweight="bold")

    fig.savefig(ART / "proactive_becker.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    print("[1/4] events + tract intensity deciles")
    ev, acct = load_events()
    ev, tr, acct2 = build_intensity(ev)
    acct.update(acct2)
    print(f"  {len(ev):,} inspections across {acct['n_tracts_decile']:,} "
          f"tracts; dropped {acct['n_dropped_no_tract']:,} no-tract, "
          f"{acct['n_events_dropped_no_panel_lots']:,} no-panel-lot, "
          f"{acct['n_dropped_pending']:,} pending")

    print("[2/4] part (a): hit rate by intensity decile")
    res_a = part_a(ev, tr, acct)
    res_a.to_csv(OUT / "proactive_becker_estimates.csv", index=False)
    raw = res_a[res_a["model"] == "raw_decile_mean"]
    print(raw[["term", "estimate", "25pct", "975pct", "n", "n_tracts",
               "mean_intensity"]].round(2).to_string(index=False))
    sl = res_a[res_a["model"].str.startswith("lpm_fe_linear")]
    print(sl[["model", "term", "estimate", "std_error", "25pct", "975pct",
              "n"]].round(3).to_string(index=False))

    print("[3/4] part (b): reallocation exercise (logit + 7G comparison)")
    df, auc, k = load_panel_scored()
    res_b, swept_p, top_p, gap = part_b(df, auc, k)
    res_b.to_csv(OUT / "proactive_reallocation.csv", index=False)
    hd = res_b[res_b["block"] == "headline"]
    print(hd[["term", "value", "note"]].round(3).to_string(index=False))
    cal = res_b[res_b["block"] == "calibration"]
    print(cal[["term", "value", "actual_rate_pct", "n"]].round(2)
          .to_string(index=False))

    print("[4/4] figure")
    make_figure(res_a, swept_p, top_p, gap)
    print(f"wrote {OUT / 'proactive_becker_estimates.csv'}\n"
          f"      {OUT / 'proactive_reallocation.csv'}\n"
          f"      {ART / 'proactive_becker.png'}")

    slope = res_a[(res_a["model"] == "lpm_fe_linear")
                  & (res_a["term"] == "decile_num")].iloc[0]
    verdict = ("RISING" if slope["25pct"] > 0 else
               "FALLING" if slope["975pct"] < 0 else "FLAT (CI spans 0)")
    print(f"\nVerdict: within category x year, hit rate is {verdict} in tract "
          f"intensity ({slope['estimate']:+.2f} pp per decile, "
          f"95% CI {slope['25pct']:+.2f}..{slope['975pct']:+.2f}); "
          f"flat or rising = not yield-equalized.")
    print(f"Reallocation: swapping the {int(hd.loc[hd.term == 'n_swept_panel_lots', 'value'].iat[0]):,} "
          f"swept residential lots for the same count of highest-predicted-risk "
          f"never-swept lots raises predicted yield "
          f"{gap['estimate']:.1f} pp = {gap['estimate'] * 10:,.0f} extra "
          f"violations per 1,000 inspections "
          f"(95% CI {gap['25pct'] * 10:,.0f}..{gap['975pct'] * 10:,.0f}).")
    print("Caveat: treat the reallocation gain as an upper bound. The model's "
          "top lots are large buildings (see units_median_top_unswept) whose "
          "77-month any-violation probability nears 1 partly through the "
          "statutory boiler/elevator streams in the union ledger, and the "
          "outcome measures realized enforcement, not latent risk.")


if __name__ == "__main__":
    main()
