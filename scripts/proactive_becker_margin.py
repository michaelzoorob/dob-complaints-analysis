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
    never-swept lots. The headline contrast is a PREVALENCE gap: the
    difference in predicted 77-month any-violation prevalence per 1,000
    LOTS, not a per-visit find rate. In-sample calibration by
    predicted-risk decile is reported so the translation is honest.

(b') Audit blocks appended to proactive_reallocation.csv (adversarial
    review 2026-07-10, critic D; constructions mirror
    scripts/audit/critic_proactive_D_becker.py):
      realized_basis   — the same swap on realized 2020-26 rates for BOTH
                         groups (the logit underpredicts the swept
                         subgroup, so the predicted-basis gap overstates);
      sweep_findable   — outcomes a sweep could actually write (any ECB or
                         union construction family; strictest field-only
                         variant SIGN FLIPS), original top set and a refit
                         logit with its own AUC;
      per_visit        — what visits actually find per inspection: 7G
                         yield at top-decile-risk lots vs its own lots,
                         8A / all-discretionary yields at the flagged
                         lots, and the implied program-mix cap;
      framing          — never-swept base rates (top-decile over-coverage)
                         and the 2019 7G cohort invisible to the 2020+
                         spine;
      out_of_time      — a 2000-09-history -> 2010-19-outcome logit whose
                         top never-swept picks are checked against
                         realized 2020-26 rates (no 2020s data anywhere).

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
from analysis_config import make_bbl

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


# ── shared model pieces (used by part (b) and the audit blocks) ─────────

ECB_DEVICE_TYPES = {"Elevators", "Boilers"}  # periodic device streams in ECB


def design_matrix(df: pd.DataFrame,
                  hist_cols=("log1p_ecb_hist", "log1p_dobviol_hist",
                             "any_prior_viol")) -> pd.DataFrame:
    """Part-(b) logit design matrix. hist_cols is swappable so the
    out-of-time variant (2000-09 history) reuses the identical layout."""
    return pd.concat([
        pd.DataFrame({"const": 1.0}, index=df.index),
        df[["era_pre1940", "era_4079", "era_8099", "era_unknown",
            "log_bldgarea", "no_bldgarea", *hist_cols,
            "tract_poverty"]].astype(float),
        pd.get_dummies(df["size_bin"], prefix="size", dtype=float).drop(
            columns=["size_1"]),
        pd.get_dummies(df["class_grp"], prefix="cls", dtype=float).drop(
            columns=["cls_A"]),
    ], axis=1)


def fit_logit(X: pd.DataFrame, y) -> np.ndarray:
    m = sm.GLM(np.asarray(y, float), X, family=sm.families.Binomial()).fit()
    return np.asarray(m.predict(X))


def rank_auc(p, y) -> float:
    """Rank-based AUC (no sklearn in this venv)."""
    r = pd.Series(p).rank().to_numpy()
    y = np.asarray(y, float)
    n1, n_all = y.sum(), len(y)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (n_all - n1)))


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

    # 8A-composition note (critic D): the top-decile dip in the pooled
    # slope is 8A mix, not crowding — excluding 8A the gradient is ~flat
    sub = ev[ev["category_prefix"] != "8A"]
    m = pf.feols("hit100 ~ decile_num | cat_year", data=sub, vcov=vcov)
    rows += clean_tidy(
        m, "lpm_fe_linear__excl_8A",
        out_name + ", per decile (8A-composition check: the pooled "
        "top-decile dip is 8A share, not crowding)").to_dict("records")

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

    X = design_matrix(df)
    y = df["any_viol_2020on"].to_numpy(float)
    df["p_hat"] = fit_logit(X, y)
    df["p100"] = df["p_hat"] * 100.0

    auc = rank_auc(df["p_hat"], y)
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
    add("prevalence_gap_per_1000_lots", gap["estimate"] * 10,
        "RELABELED (was extra_viol_per_1000): predicted 77-month "
        "any-violation prevalence gap x 10, per 1,000 LOTS; a "
        "predicted-vs-predicted prevalence contrast, NOT a per-visit find "
        "rate; see realized_basis and per_visit blocks for quotable margins")
    add("prevalence_gap_per_1000_lots_ci_lo", gap["25pct"] * 10)
    add("prevalence_gap_per_1000_lots_ci_hi", gap["975pct"] * 10,
        "CI reflects sampling error of the prevalence contrast only, not "
        "the modeling choices the audit blocks vary")
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

    rows += audit_rows(df, swept, top, n_swept)

    res = pd.DataFrame(rows)
    res["window"] = WINDOW
    return res, swept["p100"].mean(), top["p100"].mean(), gap


def audit_rows(df: pd.DataFrame, swept: pd.DataFrame, top: pd.DataFrame,
               n_swept: int) -> list[dict]:
    """Critic-D audit blocks (adversarial review 2026-07-10). Constructions
    mirror scripts/audit/critic_proactive_D_becker.py; requires df to carry
    p_hat, cal_decile and swept7g already (part (b) sets them)."""
    rows = []

    def add(block, term, value, note=""):
        rows.append({"block": block, "term": term, "value": float(value),
                     "note": note})

    conn = sqlite3.connect(str(config.DB_PATH), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000;")

    # ── realized_basis: the same swap using realized 2020-26 rates for
    #    BOTH groups (the logit underpredicts the swept subgroup, which is
    #    selected on DOB-side unobservables, so the predicted-vs-predicted
    #    prevalence gap overstates the realized one) ────────────────────
    real_swept = swept["any_viol_2020on"].mean() * 100
    real_top = top["any_viol_2020on"].mean() * 100
    add("realized_basis", "realized_gap_per_1000_lots",
        (real_top - real_swept) * 10,
        f"realized-vs-realized: {real_top:.1f} minus {real_swept:.1f} "
        "(actual_rate_* rows) x 10; still a 77-month prevalence contrast "
        "per 1,000 lots, and swept-lot paper partly sweep-caused")
    print(f"  realized-vs-realized gap {(real_top - real_swept) * 10:,.0f} "
          f"per 1,000 lots ({real_top:.1f} vs {real_swept:.1f})")

    # ── sweep_findable: strip the periodic/administrative streams the
    #    union outcome bundles (boiler/elevator/facade/energy/gas fire
    #    without any sweep). v2 = any inspector-written paper (any ECB or
    #    union construction family); v3 = strictly field paper (ECB excl.
    #    elevator/boiler device types, or union construction) ───────────
    ecb = pd.read_sql_query("""
        SELECT ecb_violation_number AS num, boro, block, lot, violation_type
        FROM ecb_violations
        WHERE length(issue_date) >= 8
          AND substr(issue_date,1,4) BETWEEN '2020' AND '2026'""", conn)
    ecb["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                      in zip(ecb["boro"], ecb["block"], ecb["lot"])]
    ecb = ecb[ecb["bbl_key"] != ""]
    ecb_flags = (ecb.assign(field=~ecb["violation_type"].isin(ECB_DEVICE_TYPES))
                 .groupby("bbl_key")
                 .agg(ecb_any=("num", "size"), ecb_field=("field", "max")))
    ecb_flags["ecb_any"] = 1
    ecb_flags.index = ecb_flags.index.astype(str)

    u = dob_ledger.union_frame(conn)
    u = u[(u["year"] >= 2020) & (u["year"] <= 2026)]
    constr = (u["family"] == "constr").groupby(u["bbl_key"].astype(str)).max()

    df["_ecb_any"] = df["bbl_key"].map(ecb_flags["ecb_any"]).fillna(0).astype(int)
    df["_ecb_field"] = df["bbl_key"].map(ecb_flags["ecb_field"]).fillna(0).astype(int)
    df["_constr"] = df["bbl_key"].map(constr).fillna(False).astype(int)
    df["viol_v2"] = ((df["_ecb_any"] == 1) | (df["_constr"] == 1)).astype(int)
    df["viol_v3"] = ((df["_ecb_field"] == 1) | (df["_constr"] == 1)).astype(int)

    sw, tp = df.loc[swept.index], df.loc[top.index]
    v2_sw, v2_tp = sw["viol_v2"].mean() * 100, tp["viol_v2"].mean() * 100
    v3_sw, v3_tp = sw["viol_v3"].mean() * 100, tp["viol_v3"].mean() * 100
    add("sweep_findable", "sweep_findable_rate_swept_pct", v2_sw,
        "any 2020-26 ECB citation or union construction-family violation")
    add("sweep_findable", "sweep_findable_rate_top_unswept_pct", v2_tp,
        "original top set")
    add("sweep_findable", "sweep_findable_gap_per_1000", (v2_tp - v2_sw) * 10,
        "periodic boiler/elevator/facade/energy/gas streams removed from "
        "the outcome; original top set")
    X2 = design_matrix(df)
    p2 = fit_logit(X2, df["viol_v2"])
    auc2 = rank_auc(p2, df["viol_v2"])
    df["_p_hat_v2"] = p2
    top2 = df[df["swept7g"] == 0].nlargest(n_swept, "_p_hat_v2")
    add("sweep_findable", "sweep_findable_auc_refit", auc2,
        "logit refit on the sweep-findable outcome, same design matrix")
    add("sweep_findable", "sweep_findable_gap_refit_per_1000",
        (top2["viol_v2"].mean() * 100 - v2_sw) * 10,
        f"top set re-picked by the refit model; its realized sweep-findable "
        f"rate is {top2['viol_v2'].mean() * 100:.1f}")
    add("sweep_findable", "field_only_rate_swept_pct", v3_sw,
        "ECB excl. elevator/boiler device types, or union construction")
    add("sweep_findable", "field_only_rate_top_unswept_pct", v3_tp,
        "original top set")
    add("sweep_findable", "field_only_gap_per_1000", (v3_tp - v3_sw) * 10,
        "SIGN FLIPS: on the strictest field-only outcome the swept lots "
        "out-realize the model's top never-swept picks")
    print(f"  sweep-findable gap {(v2_tp - v2_sw) * 10:,.0f} (refit "
          f"{(top2['viol_v2'].mean() * 100 - v2_sw) * 10:,.0f}, AUC {auc2:.3f}); "
          f"field-only {(v3_tp - v3_sw) * 10:,.0f}")

    # ── per_visit: what inspections actually find, per visit ────────────
    evd = pd.read_csv(SPINE, usecols=["category_prefix", "family", "agency",
                                      "outcome", "bbl"], dtype={"bbl": str})
    evd = evd[(evd["agency"] == 1) & (evd["family"] == "discretionary_field")
              & (evd["outcome"] != "pending")].copy()
    evd["hit"] = (evd["outcome"] == "violation").astype(float)
    evd = evd.merge(df[["bbl_key", "cal_decile"]], left_on="bbl",
                    right_on="bbl_key", how="left")
    m7g = evd[evd["category_prefix"] == "7G"]
    own = m7g[m7g["cal_decile"].notna()]
    d10 = m7g[m7g["cal_decile"] == 10]
    y_own = own["hit"].mean() * 100
    y_d10 = d10["hit"].mean() * 100
    at_top = evd[evd["bbl"].isin(set(top["bbl_key"]))]
    m8a = at_top[at_top["category_prefix"] == "8A"]
    y_alldisc = at_top["hit"].mean() * 100
    y_8a = m8a["hit"].mean() * 100
    add("per_visit", "sweep_yield_top_decile_pct", y_d10,
        f"7G per-visit violation yield at top-decile-risk lots, "
        f"n={len(d10):,} visits")
    add("per_visit", "sweep_yield_own_lots_pct", y_own,
        f"7G per-visit yield at the panel lots it actually swept, "
        f"n={len(own):,} visits")
    add("per_visit", "sweep_yield_gap_per_1000", (y_d10 - y_own) * 10,
        "re-aiming sweeps at top-decile lots buys ~nothing per visit")
    add("per_visit", "compliance_8a_yield_at_flagged_pct", y_8a,
        f"8A per-visit yield at the {n_swept:,} flagged (top never-swept) "
        f"lots, n={len(m8a):,} visits")
    add("per_visit", "alldisc_yield_at_flagged_pct", y_alldisc,
        f"all discretionary programs at the flagged lots, n={len(at_top):,} "
        f"non-7G visits")
    add("per_visit", "program_mix_cap_per_1000", (y_alldisc - y_own) * 10,
        "all-discretionary yield at flagged lots minus 7G yield at its own "
        "lots, x10: the most a PROGRAM change (8A/LL79-style visits, not a "
        "sweep re-aim) could buy per 1,000 visits")
    print(f"  per-visit: 7G d10 {y_d10:.1f} vs own {y_own:.1f}; at flagged "
          f"lots 8A {y_8a:.1f} / all-disc {y_alldisc:.1f} -> program-mix cap "
          f"{(y_alldisc - y_own) * 10:,.0f} per 1,000")

    # ── framing: base rates and the 2019 sweep cohort ────────────────────
    add("framing", "panel_never_swept_pct", (1 - df["swept7g"].mean()) * 100,
        "share of ALL scored panel lots with no 2020+ 7G sweep; "
        "never-swept is the base condition, not neglect")
    cut = df["p_hat"].quantile(0.90)
    hr = df[df["p_hat"] >= cut]
    add("framing", "top_decile_never_swept_pct",
        (1 - hr["swept7g"].mean()) * 100,
        f"n={len(hr):,} top-decile-risk lots")
    add("framing", "sweep_overcoverage_top_decile",
        hr["swept7g"].mean() / df["swept7g"].mean(),
        "P(swept | top decile) / P(swept): sweeps OVER-cover the top "
        "decile relative to the panel")
    od = pd.read_sql_query("""
        SELECT bin, substr(date_entered,7,4) AS yr FROM open_data
        WHERE complaint_category = '7G'""", conn)
    pre = od[od["yr"] < "2020"]
    bb = pd.read_sql_query("SELECT bin, bbl_key FROM bin_bbl_all", conn)
    pre_bbls = set(pre.merge(bb, on="bin", how="inner")["bbl_key"].astype(str))
    add("framing", "n_7g_events_2019", len(pre),
        "7G category exists only since 2019; these precede the 2020+ "
        "events spine")
    add("framing", "top_unswept_with_2019_sweep",
        int(top["bbl_key"].isin(pre_bbls).sum()),
        f"of the {n_swept:,} top never-swept lots, had a 2019 7G sweep "
        "invisible to the 2020+ spine; say 'in the 2020-26 window', "
        "not 'no sweep ever'")
    print(f"  framing: never-swept {100 * (1 - df['swept7g'].mean()):.1f}% of "
          f"all lots, top decile {100 * (1 - hr['swept7g'].mean()):.1f}%, "
          f"over-coverage {hr['swept7g'].mean() / df['swept7g'].mean():.1f}x; "
          f"2019 sweeps {len(pre):,}, hit {int(top['bbl_key'].isin(pre_bbls).sum())} "
          f"of the top {n_swept:,}")

    # ── out_of_time: 2000-09 history -> 2010-19 outcome; the top picks of
    #    a model that never saw any 2020s data, checked against realized
    #    2020-26 rates ────────────────────────────────────────────────────
    ecb_old = pd.read_sql_query("""
        SELECT boro, block, lot FROM ecb_violations
        WHERE length(issue_date) >= 8
          AND substr(issue_date,1,4) BETWEEN '2000' AND '2009'""", conn)
    ecb_old["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                          in zip(ecb_old["boro"], ecb_old["block"],
                                 ecb_old["lot"])]
    ecb00 = ecb_old[ecb_old["bbl_key"] != ""].groupby("bbl_key").size()
    dob00 = dob_ledger.counts_by_bbl(conn, 2000, 2009, "n00")
    df["_log1p_ecb_hist00"] = np.log1p(df["bbl_key"].map(ecb00).fillna(0))
    df["_log1p_dobviol_hist00"] = np.log1p(df["bbl_key"].map(dob00).fillna(0))
    df["_any_prior00"] = ((df["_log1p_ecb_hist00"] > 0)
                          | (df["_log1p_dobviol_hist00"] > 0)).astype(int)
    Xo = design_matrix(df, hist_cols=("_log1p_ecb_hist00",
                                      "_log1p_dobviol_hist00",
                                      "_any_prior00"))
    po = fit_logit(Xo, df["any_prior_viol"])  # outcome: any 2010-19 violation
    auc_o = rank_auc(po, df["any_prior_viol"])
    df["_p_oot"] = po
    top_o = df[df["swept7g"] == 0].nlargest(n_swept, "_p_oot")
    add("out_of_time", "oot_auc_2000s_to_2010s", auc_o,
        "logit of any 2010-19 violation on 2000-09 history + slow-moving "
        "traits; no 2020s data anywhere in fit or selection")
    add("out_of_time", "oot_top_realized_any_pct",
        top_o["any_viol_2020on"].mean() * 100,
        f"realized 2020-26 any-violation rate of that model's top "
        f"{n_swept:,} never-swept picks")
    print(f"  out-of-time: AUC {auc_o:.3f}; top picks realize "
          f"{top_o['any_viol_2020on'].mean() * 100:.1f}% in 2020-26")

    conn.close()
    return rows


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
    # opaque surface-colored patch so the top gridline does not run through
    # the legend handles/text
    ax.legend(loc="upper right", frameon=True, framealpha=1.0,
              facecolor=SURFACE, edgecolor="none", fontsize=10.5)
    ax.set_title("Where DOB inspects most, does it find more?",
                 loc="left", fontsize=15, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.015, "hit rate of agency-initiated discretionary inspections "
            "by how intensively DOB works the tract · 95% CIs, "
            "SEs clustered by tract",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.text(0.02, 0.985,
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
    aud = res_b[~res_b["block"].isin(["headline", "calibration"])]
    print("\naudit blocks (critic D):")
    print(aud[["block", "term", "value"]].round(3).to_string(index=False))

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
    aval = lambda term: res_b.loc[res_b["term"] == term, "value"].iat[0]
    print(f"Reallocation: swapping the {int(hd.loc[hd.term == 'n_swept_panel_lots', 'value'].iat[0]):,} "
          f"swept residential lots for the same count of highest-predicted-risk "
          f"never-swept lots raises predicted 77-month any-violation "
          f"prevalence {gap['estimate']:.1f} pp = {gap['estimate'] * 10:,.0f} "
          f"per 1,000 LOTS "
          f"(95% CI {gap['25pct'] * 10:,.0f}..{gap['975pct'] * 10:,.0f}) — "
          f"a prevalence gap, NOT a per-visit find rate.")
    print(f"Quotable margins (audit blocks): realized-vs-realized "
          f"{aval('realized_gap_per_1000_lots'):,.0f} per 1,000 lots; "
          f"sweep-findable {aval('sweep_findable_gap_per_1000'):,.0f} "
          f"(refit {aval('sweep_findable_gap_refit_per_1000'):,.0f}); "
          f"field-only {aval('field_only_gap_per_1000'):,.0f} (sign flip); "
          f"per-visit sweep re-aim {aval('sweep_yield_gap_per_1000'):,.0f}; "
          f"program-mix cap {aval('program_mix_cap_per_1000'):,.0f}.")
    print("Caveat: the outcome measures realized enforcement, not latent "
          "risk; the model's top lots are large buildings (see "
          "units_median_top_unswept) whose 77-month any-violation "
          "probability nears 1 partly through the statutory "
          "boiler/elevator streams in the union ledger.")


if __name__ == "__main__":
    main()
