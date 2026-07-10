#!/usr/bin/env python3
"""
Hypothesis 5 (proactive_enforcement_plan.md): do unstable-building incidents
redirect agency-initiated enforcement to the lot, and to its block, afterward?

Stacked event study around each lot's FIRST caller-originated category-30
(structural-stability) complaint. Event months 2021-01..2025-05, so a full
+/-12-month window fits inside the 2020-01..2026-05 spine.

Outcomes (BBL x calendar month), all counting agency-originated complaints
(empty ref_311 in bis_scrape). Critic C (critic_proactive_C_eventstudies.py)
showed the old cat-30-only exclusion missed the mandated incident response:
roughly two thirds of the month-0 excess was 2F structural-monitoring
enrollment, 1X emergency work orders, 7R class-1 follow-ups and same-visit
companion paperwork, and ~61% of month-0 agency events sat within +/-3 days
of the trigger. The series are therefore:

  y_strict  (HEADLINE, series focal_strict): excludes category 30, the
      order/monitoring machinery categories {1X, 2F, 7R}, AND any agency
      complaint logged at a treated lot within +/-3 days of its trigger
      (the first caller cat-30 report in the event month). What remains is
      new agency attention beyond the mandated incident response.
  y_excl30  (series focal_total_response): excludes category 30 only —
      the TOTAL RESPONSE, orders and monitoring enrollment included. This
      is the series the post originally headlined; kept as a labeled
      variant, never as the headline.
  y_incl30  (series focal_incl_cat30): adds agency cat-30 follow-ups back.

Design (stacked, clean controls): for each treated lot, up to 5 control
lots sampled without replacement from the same census tract (same-borough
fallback), restricted to spine lots that are never treated or treated more
than 24 months after the cohort month (their own +/-12 window cannot
overlap the cohort window) and whose block has no caller cat-30 complaint
anywhere in the cohort window (kills block spillover onto controls; also
removes same-block lots). Spec follows owner_transition_panel.py:

    y ~ i(event_t, treat, ref=-1) + C(event_t) | bbl + cal_month

Focal series cluster by BBL (treatment varies at the lot). The same-block
spillover series takes block-FIRST incidents, treats the other spine lots
on the block (up to 10 per event, own-treatment-clean), keeps the
total-response outcome, and clusters by block, the level the spillover
treatment varies (CLAUDE.md convention).

CSV additions from the critic-C fixes:
  - pct rows: month-0 and months-0-3 effects as a percent of BOTH the
    12-month pre mean and the month -1 treated level (the pre period ramps
    upward, so the 12-month mean overstates percent effects);
  - ramp rows: month-0 and months-0-3 minus a linear / quadratic
    extrapolation of the pre-period coefficients (t = -12..-2), with
    delta-method z — the discontinuity-above-the-ramp test;
  - m0_decomposition rows: category shares of the total-response month-0
    excess (2F / 1X / 7R / 73 / 2L / ...), the share of month-0 agency
    events within +/-3 days of the trigger, and the union share;
  - block-aggregate power rows: the largest block-level months-0-3 effect
    the spillover CI permits, as a share of the focal effect.

Universe caveat: control and neighbor lots come from the complaint spine
(lots with at least one scraped complaint 2020-2026, n=205k with tracts);
lots with no complaints ever are all-zero on the outcome and are not
sampled.

Writes
    data/analysis/risk_models/proactive_incident_estimates.csv
    data/analysis/blog_posts/artifacts/proactive_incident.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_incident_eventstudy.py
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
RM = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"

WINDOW = 12                 # event months -12..+12
FIRST_M, LAST_M = 12, 64    # 2021-01 .. 2025-05 (2020-01 = 0)
N_MONTHS = 77               # 2020-01 .. 2026-05
K_CONTROLS = 5              # matched controls per event
MAX_NEIGHBORS = 10          # same-block lots per spillover event
CLEAR = 2 * WINDOW          # controls' own treatment > m0 + 24
STRIDE = 128                # > N_MONTHS, packs (id, month) into one int64
SEED = 20260710

MECH_CATS = ["1X", "2F", "7R"]   # emergency work orders / structural
                                 # monitoring enrollment / class-1 follow-up
TRIG_DAYS = 3                    # +/- days of the trigger scrubbed (strict)
DECOMP_CATS = ["2F", "1X", "7R", "73", "2L"]   # always written to the CSV

SERIES_LABELS = {
    "focal_strict": ("HEADLINE - new attention beyond the mandated response "
                     "(drops cat-30, 1X/2F/7R orders-and-monitoring, and any "
                     "agency event within +/-3 days of the trigger)"),
    "focal_total_response": ("variant - total response (orders and "
                             "monitoring enrollment included; excludes "
                             "cat-30 only)"),
    "focal_incl_cat30": ("variant - total response including agency cat-30 "
                         "follow-ups"),
    "block_spillover": ("same-block neighbors of block-first incidents, "
                        "total-response outcome"),
}

# house palette (make_descriptive_figures.py)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def zp(z):
    """Two-sided normal p-value."""
    return math.erfc(abs(z) / math.sqrt(2.0))


# ── data ─────────────────────────────────────────────────────────────────

def load_spine():
    df = pd.read_csv(SPINE, usecols=["received_date", "month",
                                     "category_prefix", "category_name",
                                     "agency", "bbl", "bct2020"],
                     dtype={"category_prefix": "str", "bbl": "str"})
    n0 = len(df)
    df = df[df["bbl"].notna() & df["bbl"].str.fullmatch(r"\d{10}")].copy()
    df["bbl_i"] = df["bbl"].astype("int64")
    df["block"] = df["bbl_i"] // 10_000          # boro + 5-digit block
    df["m"] = ((df["month"].str[:4].astype(int) - 2020) * 12
               + df["month"].str[5:7].astype(int) - 1)
    print(f"spine: {n0:,} rows -> {len(df):,} with valid BBL; "
          f"{df['bbl_i'].nunique():,} lots, months 0..{df['m'].max()}")
    return df


def make_counter(sub):
    """Exact-count lookup via searchsorted on a packed sorted key array."""
    keys = np.sort(sub["bbl_i"].to_numpy() * STRIDE + sub["m"].to_numpy())

    def count(bbl_arr, m_arr):
        q = bbl_arr * STRIDE + m_arr
        return (np.searchsorted(keys, q, side="right")
                - np.searchsorted(keys, q, side="left")).astype(np.int32)

    return count


def build_inputs(df):
    """Events, universe, outcome counters, block-incident window counter."""
    c30 = df[(df["category_prefix"] == "30") & (df["agency"] == 0)]
    treat_m = c30.groupby("bbl_i")["m"].min()          # first caller cat-30

    # universe: one row per lot with a tract (control + neighbor pool)
    uni = (df[df["bct2020"].notna()]
           .groupby("bbl_i", as_index=False)
           .agg(tract=("bct2020", "first"), block=("block", "first")))
    uni["boro"] = uni["bbl_i"] // 1_000_000_000
    uni["treat_m"] = uni["bbl_i"].map(treat_m)

    ev = treat_m[(treat_m >= FIRST_M) & (treat_m <= LAST_M)].reset_index()
    ev.columns = ["bbl_i", "m0"]
    n_all = len(ev)
    ev = ev.merge(uni[["bbl_i", "tract", "block", "boro"]], on="bbl_i", how="inner")
    ev["event_id"] = np.arange(len(ev))
    print(f"events: {len(treat_m):,} ever-treated lots; {n_all:,} first-treated "
          f"2021-01..2025-05; {len(ev):,} with a census tract (kept)")

    # outcome counters
    count_excl30 = make_counter(df[(df["agency"] == 1) & (df["category_prefix"] != "30")])
    count_incl30 = make_counter(df[df["agency"] == 1])

    # caller cat-30 incidents per block, for the control-contamination filter
    blk_keys = np.sort(c30["block"].to_numpy() * STRIDE + c30["m"].to_numpy())

    def block_incidents(block_arr, m0_arr):
        lo = block_arr * STRIDE + (m0_arr - WINDOW)
        hi = block_arr * STRIDE + (m0_arr + WINDOW)
        return (np.searchsorted(blk_keys, hi, side="right")
                - np.searchsorted(blk_keys, lo, side="left"))

    blk_first = c30.groupby("block")["m"].min()
    return ev, uni, count_excl30, count_incl30, block_incidents, blk_first


def trigger_dates(df, ev):
    """Per treated lot: earliest caller cat-30 received_date in month m0."""
    c30 = df[(df["category_prefix"] == "30") & (df["agency"] == 0)]
    tr = c30.merge(ev[["bbl_i", "m0"]], on="bbl_i")
    tr = tr[tr["m"] == tr["m0"]]
    trig = tr.groupby("bbl_i")["received_date"].min()   # ISO strings sort ok
    print(f"triggers: dates for {len(trig):,}/{len(ev):,} treated lots")
    return trig


def make_strict_counter(df, trig):
    """Agency non-30 events minus {1X,2F,7R} minus the +/-3-day trigger
    window at treated lots (same-visit companion paperwork: 73, 2L, 10 ...).
    Control/neighbor lots have no trigger, so only the symmetric category
    exclusion binds there."""
    ag = df[(df["agency"] == 1) & (df["category_prefix"] != "30")].copy()
    tdate = ag["bbl_i"].map(trig)
    dd = (pd.to_datetime(ag["received_date"], format="%Y-%m-%d")
          - pd.to_datetime(tdate, format="%Y-%m-%d")).dt.days
    keep = (dd.isna() | (dd.abs() > TRIG_DAYS)) & ~ag["category_prefix"].isin(MECH_CATS)
    strict = ag[keep]
    n_win = int((~(dd.isna() | (dd.abs() > TRIG_DAYS))).sum())
    print(f"strict outcome: {len(ag):,} agency non-30 events -> {len(strict):,} "
          f"({int(ag['category_prefix'].isin(MECH_CATS).sum()):,} in 1X/2F/7R; "
          f"{n_win:,} within +/-{TRIG_DAYS}d of a trigger)")
    return make_counter(strict)


# ── month-0 decomposition of the total-response excess ───────────────────

def decompose_m0(df, ev, trig):
    """What the (old, total-response) month-0 spike is made of."""
    n_ev = len(ev)
    ag = df[(df["agency"] == 1) & (df["category_prefix"] != "30")].merge(
        ev[["bbl_i", "m0"]], on="bbl_i")
    ag["et"] = ag["m"] - ag["m0"]
    m0ev = ag[ag["et"] == 0].copy()
    pre = ag[ag["et"].between(-12, -1)]
    pre_rate = pre.groupby("category_prefix").size() / 12.0 / n_ev
    m0_rate = m0ev.groupby("category_prefix").size() / n_ev
    excess = (m0_rate - pre_rate.reindex(m0_rate.index).fillna(0.0))
    excess = excess.sort_values(ascending=False)
    tot = float(excess.sum())

    dd = (pd.to_datetime(m0ev["received_date"], format="%Y-%m-%d")
          - pd.to_datetime(m0ev["bbl_i"].map(trig), format="%Y-%m-%d")).dt.days
    within3 = float((dd.abs() <= TRIG_DAYS).mean())
    union = float(((dd.abs() <= TRIG_DAYS)
                   | m0ev["category_prefix"].isin(MECH_CATS)).mean())
    mech_share = float(excess.reindex(MECH_CATS).fillna(0.0).sum() / tot)
    names = df.groupby("category_prefix")["category_name"].first()

    cats = list(dict.fromkeys(list(excess.head(8).index) + DECOMP_CATS))
    rows = [{"series": "m0_decomposition", "series_label":
             "composition of the total-response month-0 excess at treated lots",
             "term": "excess_total_per_lot", "coef": tot,
             "n_treated_units": n_ev,
             "note": "month-0 agency non-30 rate minus the 12-month pre-mean "
                     "rate, per treated lot"}]
    for cat in cats:
        v = float(excess.get(cat, 0.0))
        rows.append({"series": "m0_decomposition", "series_label":
                     "composition of the total-response month-0 excess at "
                     "treated lots",
                     "term": f"excess_share_{cat}", "coef": v,
                     "share_of_excess": v / tot, "n_treated_units": n_ev,
                     "note": str(names.get(cat))})
    for term, val, note in [
            ("m0_share_within_3d_of_trigger", within3,
             "share of month-0 agency non-30 events logged within +/-3 days "
             "of the trigger complaint"),
            ("m0_share_mech_or_3d_union", union,
             "share of month-0 agency non-30 events that are 1X/2F/7R OR "
             "within +/-3 days of the trigger"),
            ("m0_excess_share_mech_cats", mech_share,
             "1X+2F+7R share of the month-0 excess")]:
        rows.append({"series": "m0_decomposition", "series_label":
                     "composition of the total-response month-0 excess at "
                     "treated lots",
                     "term": term, "coef": val, "n_treated_units": n_ev,
                     "note": note})

    print(f"\nmonth-0 decomposition ({n_ev:,} treated lots): total excess "
          f"{tot:+.4f} agency non-30 complaints per lot")
    for cat in cats:
        v = float(excess.get(cat, 0.0))
        print(f"  {cat:>3} {str(names.get(cat))[:52]:<52} {v:+.4f} "
              f"({v / tot * 100:5.1f}% of excess)")
    print(f"  1X+2F+7R share of excess {mech_share:.1%}; within +/-3d of "
          f"trigger {within3:.1%}; union {union:.1%}")
    return pd.DataFrame(rows)


# ── matching ─────────────────────────────────────────────────────────────

def _clean_candidates(cand, block_incidents):
    """Not-yet-treated with a clear window, on an incident-free block."""
    cand = cand[cand["cbbl"] != cand["bbl_i"]]
    ok = cand["treat_m"].isna() | (cand["treat_m"] > cand["m0"] + CLEAR)
    cand = cand[ok]
    n_inc = block_incidents(cand["cblock"].to_numpy(), cand["m0"].to_numpy())
    return cand[n_inc == 0]


def match_controls(ev, uni, block_incidents, rng):
    """Up to K_CONTROLS per event, without replacement, tract then borough."""
    pool = uni.rename(columns={"bbl_i": "cbbl", "block": "cblock"})

    cand = ev[["event_id", "bbl_i", "m0", "tract"]].merge(
        pool[["cbbl", "cblock", "tract", "treat_m"]], on="tract")
    cand = _clean_candidates(cand, block_incidents)
    cand = cand.sample(frac=1.0, random_state=rng).drop_duplicates("cbbl")
    matched = cand.groupby("event_id").head(K_CONTROLS)

    got = matched.groupby("event_id").size()
    short_ids = ev.loc[~ev["event_id"].isin(got.index), "event_id"]
    print(f"controls: tract stage {len(matched):,} lots for "
          f"{got.index.nunique():,}/{len(ev):,} events "
          f"(mean {got.mean():.2f} per event); {len(short_ids):,} events to "
          f"borough fallback")

    if len(short_ids):
        fb = ev[ev["event_id"].isin(short_ids)][["event_id", "bbl_i", "m0", "boro"]]
        cand_b = fb.merge(pool[["cbbl", "cblock", "boro", "treat_m"]], on="boro")
        cand_b = _clean_candidates(cand_b, block_incidents)
        cand_b = cand_b[~cand_b["cbbl"].isin(matched["cbbl"])]
        cand_b = cand_b.sample(frac=1.0, random_state=rng).drop_duplicates("cbbl")
        fb_matched = cand_b.groupby("event_id").head(K_CONTROLS)
        matched = pd.concat([matched, fb_matched], ignore_index=True)
        print(f"controls: borough fallback added {len(fb_matched):,} lots")

    return matched[["event_id", "cbbl", "cblock", "m0"]].rename(
        columns={"cbbl": "bbl", "cblock": "block"})


def match_neighbors(ev, uni, blk_first, rng):
    """Other spine lots on the block of block-FIRST incidents."""
    ev_bf = ev[ev["m0"] == ev["block"].map(blk_first)]
    nb = ev_bf[["event_id", "bbl_i", "m0", "block"]].merge(
        uni[["bbl_i", "block", "treat_m"]].rename(columns={"bbl_i": "nbbl"}),
        on="block")
    nb = nb[nb["nbbl"] != nb["bbl_i"]]
    nb = nb[nb["treat_m"].isna() | (nb["treat_m"] > nb["m0"] + CLEAR)]
    nb = nb.sample(frac=1.0, random_state=rng)
    nb = nb.groupby("event_id").head(MAX_NEIGHBORS)
    print(f"spillover: {len(ev_bf):,}/{len(ev):,} events are block-first; "
          f"{len(nb):,} neighbor lots "
          f"(mean {len(nb) / max(nb['event_id'].nunique(), 1):.2f} per event)")
    return nb[["event_id", "nbbl", "block", "m0"]].rename(columns={"nbbl": "bbl"})


# ── panel ────────────────────────────────────────────────────────────────

def expand(units, treat, count_excl30, count_incl30, count_strict=None):
    """units (bbl, block, m0) -> 25 rows each with searchsorted counts."""
    w = 2 * WINDOW + 1
    et = np.tile(np.arange(-WINDOW, WINDOW + 1), len(units))
    out = pd.DataFrame({
        "bbl": np.repeat(units["bbl"].to_numpy(), w),
        "block": np.repeat(units["block"].to_numpy(), w),
        "event_t": et,
        "cal_m": np.repeat(units["m0"].to_numpy(), w) + et,
        "treat": np.int8(treat),
    })
    b, c = out["bbl"].to_numpy(), out["cal_m"].to_numpy()
    out["y_excl30"] = count_excl30(b, c)
    out["y_incl30"] = count_incl30(b, c)
    if count_strict is not None:
        out["y_strict"] = count_strict(b, c)
    return out


# ── estimation ───────────────────────────────────────────────────────────

def _ramp_rows(names, beta, V, idx, series, label):
    """Month-0 / months-0-3 net of a polynomial extrapolation of the pre
    coefficients (t=-12..-2, ref month -1 excluded), delta-method z."""
    pre_ts = np.array(sorted(k for k in idx if k < -1), dtype=float)
    pre_ix = np.array([idx[int(t)] for t in pre_ts])
    rows = []
    for deg, dlab in [(1, "linear"), (2, "quadratic")]:
        X = np.vander(pre_ts, deg + 1, increasing=True)
        H = np.linalg.pinv(X.T @ X) @ X.T            # (deg+1, n_pre)

        def w_pred(t_star):
            x = np.array([t_star ** j for j in range(deg + 1)])
            return x @ H                              # weights on pre coefs

        for tgt, ks in [("m0", [0]), ("avg03", [0, 1, 2, 3])]:
            a = np.zeros(len(names))
            preds = []
            for k in ks:
                a[idx[k]] += 1.0 / len(ks)
                w = w_pred(float(k))
                preds.append(float(w @ beta[pre_ix]))
                for wj, ij in zip(w, pre_ix):
                    a[ij] -= wj / len(ks)
            est = float(a @ beta)
            se = float(np.sqrt(a @ V @ a))
            z = est / se
            rows.append({"series": series, "series_label": label,
                         "term": f"ramp_{tgt}_minus_{dlab}",
                         "coef": est, "se": se,
                         "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
                         "pval": zp(z), "z": z,
                         "ramp_pred": float(np.mean(preds)),
                         "note": (f"{'month 0' if tgt == 'm0' else 'months 0-3 avg'} "
                                  f"minus the {dlab} continuation of the "
                                  f"pre-period ramp (fit on t=-12..-2)")})
    return rows


def run_series(panel, ycol, cluster, series, meta, ramp=True):
    import pyfixest as pf
    label = SERIES_LABELS[series]
    m = pf.feols(f"{ycol} ~ i(event_t, treat, ref=-1) + C(event_t) | bbl + cal_m",
                 data=panel, vcov={"CRV1": cluster})
    names = [str(n) for n in m._coefnames]
    tid = m.tidy().reset_index()
    term_col = tid.columns[0]
    g = tid[tid[term_col].str.startswith("event_t::")
            & tid[term_col].str.endswith(":treat")].copy()
    g["event_time"] = g[term_col].str.extract(r"event_t::(-?\d+):").astype(int)
    g = g.sort_values("event_time")

    beta = m.coef().to_numpy()
    V = m._vcov
    idx = {}
    for i, n in enumerate(names):
        if n.startswith("event_t::") and n.endswith(":treat"):
            idx[int(n.split("::")[1].split(":")[0])] = i

    # joint pre-trend Wald (event months -12..-2), cluster-robust chi2
    pre_ix = [i for k, i in idx.items() if k < -1]
    R = np.zeros((len(pre_ix), len(names)))
    for j, i in enumerate(pre_ix):
        R[j, i] = 1.0
    w = m.wald_test(R=R)
    pre_chi2, pre_p = float(w["statistic"]), float(w["pvalue"])

    # month 0 and the months 0-3 average (linear combinations)
    m0 = float(beta[idx[0]])
    m0_se = float(np.sqrt(V[idx[0], idx[0]]))
    a = np.zeros(len(names))
    for k in range(4):
        a[idx[k]] = 0.25
    jump = float(a @ beta)
    jump_se = float(np.sqrt(a @ V @ a))

    tr = panel[panel["treat"] == 1]
    base = float(tr.loc[tr["event_t"] < 0, ycol].mean())          # 12m pre mean
    m1_level = float(tr.loc[tr["event_t"] == -1, ycol].mean())    # month -1

    rows = [{"series": series, "series_label": label, "term": "event_month",
             "event_time": -1, "coef": 0.0, "se": np.nan, "ci_low": np.nan,
             "ci_high": np.nan, "pval": np.nan}]
    for _, r in g.iterrows():
        rows.append({"series": series, "series_label": label,
                     "term": "event_month",
                     "event_time": int(r["event_time"]),
                     "coef": r["Estimate"], "se": r["Std. Error"],
                     "ci_low": r["2.5%"], "ci_high": r["97.5%"],
                     "pval": r["Pr(>|t|)"]})
    rows.append({"series": series, "series_label": label,
                 "term": "avg_months_0_3", "event_time": np.nan,
                 "coef": jump, "se": jump_se,
                 "ci_low": jump - 1.96 * jump_se,
                 "ci_high": jump + 1.96 * jump_se,
                 "pval": zp(jump / jump_se)})

    # percent framings against BOTH baselines (critic C, F2)
    for tgt, est, se in [("m0", m0, m0_se), ("avg03", jump, jump_se)]:
        for dname, denom in [("pre12_mean", base), ("m1_level", m1_level)]:
            rows.append({"series": series, "series_label": label,
                         "term": f"pct_{tgt}_vs_{dname}", "event_time": np.nan,
                         "coef": est / denom * 100.0,
                         "se": se / denom * 100.0,
                         "ci_low": (est - 1.96 * se) / denom * 100.0,
                         "ci_high": (est + 1.96 * se) / denom * 100.0,
                         "pval": zp(est / se),
                         "note": f"denominator {dname} = {denom:.4f}"})

    if ramp:
        rows += _ramp_rows(names, np.asarray(beta), np.asarray(V), idx,
                           series, label)

    out = pd.DataFrame(rows)
    post_rows = out[(out["term"] == "event_month") & (out["event_time"] >= 1)]
    dead = post_rows[(post_rows["ci_low"] <= 0) & (post_rows["ci_high"] >= 0)]
    decay = int(dead["event_time"].min()) if len(dead) else np.nan

    out["cluster"] = cluster
    out["n_treated_units"] = meta["n_treated"]
    out["n_control_units"] = meta["n_control"]
    out["n_obs"] = m._N
    out["baseline_mean_treated"] = base
    out["m1_level_treated"] = m1_level
    out["pretrend_chi2"] = pre_chi2
    out["pretrend_df"] = len(pre_ix)
    out["pretrend_p"] = pre_p
    out["decay_first_ci0_month"] = decay

    print(f"\n[{series}] N={m._N:,} obs, {meta['n_treated']:,} treated / "
          f"{meta['n_control']:,} control units, cluster={cluster}")
    print(f"  month 0: {m0:+.4f} (se {m0_se:.4f}); months 0-3 avg "
          f"{jump:+.4f} (se {jump_se:.4f})")
    print(f"  vs 12m pre mean {base:.4f}: m0 {m0 / base * 100:+.0f}%, "
          f"quarter {jump / base * 100:+.0f}%   |   vs month -1 level "
          f"{m1_level:.4f}: m0 {m0 / m1_level * 100:+.0f}%, quarter "
          f"{jump / m1_level * 100:+.0f}%")
    if ramp:
        rl = out[out["term"] == "ramp_m0_minus_linear"].iloc[0]
        rq = out[out["term"] == "ramp_m0_minus_quadratic"].iloc[0]
        print(f"  ramp test (m0 minus extrapolated pre-trend): linear "
              f"{rl['coef']:+.4f} (z {rl['z']:.1f}), quadratic "
              f"{rq['coef']:+.4f} (z {rq['z']:.1f})")
    print(f"  pre-trend joint chi2({len(pre_ix)}) = {pre_chi2:.1f}, "
          f"p = {pre_p:.3f}; first post month (>=1) with CI covering 0: "
          f"{decay if not (isinstance(decay, float) and np.isnan(decay)) else '>12'}")
    return out


def block_power_rows(est, ev, uni, blk_first):
    """What block-level months-0-3 effect does the spillover CI permit?"""
    def val(series, term, col="coef"):
        r = est[(est["series"] == series) & (est["term"] == term)]
        return float(r[col].iloc[0])

    hi = val("block_spillover", "avg_months_0_3", "ci_high")
    avg_total = val("focal_total_response", "avg_months_0_3")
    avg_strict = val("focal_strict", "avg_months_0_3")

    ev_bf = ev[ev["m0"] == ev["block"].map(blk_first)]
    blk_sizes = uni.groupby("block")["bbl_i"].size()
    nb_mean = float((ev_bf["block"].map(blk_sizes) - 1).clip(lower=0).mean())
    agg_hi = hi * nb_mean

    label = "power of the block-spillover null (months 0-3 CI upper bound)"
    rows = [
        {"term": "block_size_mean_excl_focal", "coef": nb_mean,
         "note": "spine lots per treated block excluding the focal lot, "
                 "block-first events"},
        {"term": "per_neighbor_ci_high_months_0_3", "coef": hi,
         "note": "spillover months 0-3 CI upper bound, agency complaints "
                 "per neighbor lot-month"},
        {"term": "per_neighbor_max_share_of_focal_total",
         "coef": hi / avg_total,
         "note": "CI-permitted per-neighbor effect / focal total-response "
                 "months 0-3 effect"},
        {"term": "per_neighbor_max_share_of_focal_strict",
         "coef": hi / avg_strict,
         "note": "CI-permitted per-neighbor effect / focal strict months 0-3 "
                 "effect (spillover outcome is total-response)"},
        {"term": "block_aggregate_ci_high_months_0_3", "coef": agg_hi,
         "note": "CI upper bound x mean block size: largest block-aggregate "
                 "added agency complaints per block-month the design permits"},
        {"term": "block_aggregate_max_share_of_focal_total",
         "coef": agg_hi / avg_total,
         "note": "block-aggregate CI bound / focal total-response months 0-3 "
                 "effect"},
        {"term": "block_aggregate_max_share_of_focal_strict",
         "coef": agg_hi / avg_strict,
         "note": "block-aggregate CI bound / focal strict months 0-3 effect "
                 "(spillover outcome is total-response)"},
    ]
    for r in rows:
        r.update({"series": "block_spillover", "series_label": label})
    print(f"\nblock-spillover power: mean block size (excl focal) {nb_mean:.1f}; "
          f"per-neighbor CI high {hi:+.4f} ({hi / avg_total * 100:.1f}% of "
          f"focal total, {hi / avg_strict * 100:.1f}% of focal strict); "
          f"block aggregate up to {agg_hi:+.4f} = "
          f"{agg_hi / avg_total * 100:.0f}% of focal total / "
          f"{agg_hi / avg_strict * 100:.0f}% of focal strict")
    return pd.DataFrame(rows)


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(est, meta):
    def series(name):
        g = est[(est["series"] == name) & (est["term"] == "event_month")].copy()
        return g.sort_values("event_time")

    def val(name, term, col="coef"):
        r = est[(est["series"] == name) & (est["term"] == term)]
        return float(r[col].iloc[0])

    strict, total, spill = (series("focal_strict"),
                            series("focal_total_response"),
                            series("block_spillover"))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 6.4), dpi=200,
        gridspec_kw={"left": 0.06, "right": 0.985, "top": 0.685,
                     "bottom": 0.10, "wspace": 0.24})

    def draw(ax, g, color, xoff, label, ms=5.5):
        x = g["event_time"] + xoff
        ax.plot(x, g["coef"], color=color, lw=2, zorder=3, label=label)
        ok = g["se"].notna()
        ax.vlines(x[ok], g.loc[ok, "ci_low"], g.loc[ok, "ci_high"],
                  color=color, lw=1.4, alpha=0.8, zorder=2)
        ax.plot(x, g["coef"], "o", ms=ms, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)

    for ax in (ax1, ax2):
        ax.axhline(0, color=ZERO, lw=1.1)
        ax.axvline(-0.5, color=MUTED, lw=1.0, ls=(0, (1, 2)))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(range(-12, 13, 2))
        ax.tick_params(labelsize=9)
        ax.set_xlabel("months relative to the complaint (reference = month "
                      "before)", fontsize=9.5)

    # panel A: total response vs strict, same lot
    draw(ax1, total, MUTED, -0.12, "total response (orders +\nmonitoring included)")
    draw(ax1, strict, BLUE, +0.12, "new attention (strict outcome)")
    ax1.set_ylabel("difference in agency-initiated complaints\nper lot-month",
                   fontsize=9)
    ylo1, yhi1 = ax1.get_ylim()
    ax1.set_ylim(ylo1, yhi1 + 0.05 * (yhi1 - ylo1))     # apex whisker headroom
    # opaque surface-colored patch so gridlines do not strike the legend text
    ax1.legend(loc="upper left", frameon=True, framealpha=1.0,
               facecolor=SURFACE, edgecolor="none", fontsize=9)
    ymax1 = ax1.get_ylim()[1]
    ax1.text(-0.85, ymax1 * 0.58, "unstable-building\ncomplaint filed",
             fontsize=8.5, color=MUTED, ha="right", va="top", style="italic",
             bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax1.set_title("Most of the spike is the mandated response",
                  loc="left", fontsize=12, color=INK, weight="semibold",
                  pad=10)

    # panel B: strict, zoomed, vs the rest of the block
    draw(ax2, strict, BLUE, -0.12, "the lot itself (strict outcome)")
    draw(ax2, spill, AQUA, +0.12, "other lots on the same block")
    ax2.set_ylabel("difference in agency-initiated complaints\nper lot-month",
                   fontsize=9)
    ax2.legend(loc="upper right", frameon=True, framealpha=1.0,
               facecolor=SURFACE, edgecolor="none", fontsize=9)
    ax2.set_title("New attention, zoomed: brief at the lot, flat on the block",
                  loc="left", fontsize=12, color=INK, weight="semibold",
                  pad=10)

    fig.suptitle("After an unstable-building complaint, the enforcement spike "
                 "is mostly the mandated response;\nnew attention is modest, "
                 "fades within a quarter, and never reaches the block",
                 x=0.02, y=0.985, ha="left", fontsize=13.5, color=INK,
                 weight="semibold")

    s_m0 = float(strict.loc[strict["event_time"] == 0, "coef"].iloc[0])
    t_m0 = float(total.loc[total["event_time"] == 0, "coef"].iloc[0])
    s_a, s_lo, s_hi = (val("focal_strict", "avg_months_0_3"),
                       val("focal_strict", "avg_months_0_3", "ci_low"),
                       val("focal_strict", "avg_months_0_3", "ci_high"))
    p_pre = val("focal_strict", "pct_avg03_vs_pre12_mean")
    p_m1 = val("focal_strict", "pct_avg03_vs_m1_level")
    rz = val("focal_strict", "ramp_m0_minus_linear", "z")
    sh_2f = val("m0_decomposition", "excess_share_2F", "share_of_excess")
    sh_1x = val("m0_decomposition", "excess_share_1X", "share_of_excess")
    sh_7r = val("m0_decomposition", "excess_share_7R", "share_of_excess")
    w3 = val("m0_decomposition", "m0_share_within_3d_of_trigger")
    sp_a, sp_lo, sp_hi = (val("block_spillover", "avg_months_0_3"),
                          val("block_spillover", "avg_months_0_3", "ci_low"),
                          val("block_spillover", "avg_months_0_3", "ci_high"))
    agg_share = val("block_spillover", "block_aggregate_max_share_of_focal_total")

    fig.text(0.02, 0.87,
             f"First caller-reported category-30 complaint per lot, 2021-2025 "
             f"({meta['n_treated_focal']:,} lots; {meta['n_neighbors']:,} same-block "
             f"lots) vs. {meta['n_control']:,} not-yet-treated lots matched within "
             f"census tract · lot and calendar-month fixed effects · whiskers = 95% "
             f"CI, SEs clustered by lot (focal) and by block (spillover)\n"
             f"strict outcome drops the mandated response: category 30, emergency "
             f"work orders (1X), structural-monitoring enrollment (2F), class-1 "
             f"follow-ups (7R), and any agency complaint within ±3 days of the "
             f"trigger — {sh_2f:.0%} of the month-0 excess is 2F, {sh_1x:.0%} 1X, "
             f"{sh_7r * 100:.1f}% 7R; {w3:.0%} of month-0 events sit within ±3 "
             f"days\n"
             f"strict series: month 0 {s_m0:+.3f} (total response {t_m0:+.3f}); "
             f"months 0-3 average {s_a:+.4f} [{s_lo:+.4f}, {s_hi:+.4f}] per "
             f"lot-month = {p_pre:+.0f}% of the 12-month pre mean, {p_m1:+.0f}% of "
             f"the month −1 level (the pre-period ramps up, joint p < 0.001) · "
             f"month-0 discontinuity above a linear ramp: z = {rz:.0f}\n"
             f"block spillover months 0-3: {sp_a:+.4f} [{sp_lo:+.4f}, {sp_hi:+.4f}] "
             f"per neighbor lot-month; the CI still permits a block-aggregate rise "
             f"up to {agg_share:.0%} of the focal total-response effect",
             fontsize=8.3, color=MUTED, va="top", linespacing=1.5)

    fig.savefig(ART / "proactive_incident.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {ART / 'proactive_incident.png'}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_spine()
    ev, uni, count_excl30, count_incl30, block_incidents, blk_first = build_inputs(df)

    # rng call order matches the published run: controls, then neighbors
    controls = match_controls(ev, uni, block_incidents, rng)
    neighbors = match_neighbors(ev, uni, blk_first, rng)

    trig = trigger_dates(df, ev)
    count_strict = make_strict_counter(df, trig)
    decomp = decompose_m0(df, ev, trig)

    treated_units = ev.rename(columns={"bbl_i": "bbl"})[["bbl", "block", "m0"]]
    pan_treated = expand(treated_units, 1, count_excl30, count_incl30, count_strict)
    pan_controls = expand(controls, 0, count_excl30, count_incl30, count_strict)
    pan_neighbors = expand(neighbors, 1, count_excl30, count_incl30, count_strict)

    focal_panel = pd.concat([pan_treated, pan_controls], ignore_index=True)
    # spillover regression: neighbor lots vs the same matched controls,
    # dropping controls that themselves sit on a treated block
    keep = ~pan_controls["bbl"].isin(set(neighbors["bbl"]))
    spill_panel = pd.concat([pan_neighbors, pan_controls[keep]], ignore_index=True)
    print(f"\npanels: focal {len(focal_panel):,} rows, spillover "
          f"{len(spill_panel):,} rows "
          f"({(~keep).sum() // (2 * WINDOW + 1):,} control lots dropped as neighbors)")

    n_ctrl = controls["bbl"].nunique()
    meta_f = {"n_treated": len(treated_units), "n_control": n_ctrl}
    est = pd.concat([
        run_series(focal_panel, "y_strict", "bbl", "focal_strict", meta_f),
        run_series(focal_panel, "y_excl30", "bbl", "focal_total_response", meta_f),
        run_series(focal_panel, "y_incl30", "bbl", "focal_incl_cat30", meta_f),
        run_series(spill_panel, "y_excl30", "block", "block_spillover",
                   {"n_treated": neighbors["bbl"].nunique(),
                    "n_control": int(keep.sum()) // (2 * WINDOW + 1)},
                   ramp=False),
    ], ignore_index=True)

    est = pd.concat([est, block_power_rows(est, ev, uni, blk_first), decomp],
                    ignore_index=True)
    cols = ["series", "series_label", "term", "event_time", "coef", "se",
            "ci_low", "ci_high", "pval", "z", "ramp_pred", "share_of_excess",
            "cluster", "n_treated_units", "n_control_units", "n_obs",
            "baseline_mean_treated", "m1_level_treated", "pretrend_chi2",
            "pretrend_df", "pretrend_p", "decay_first_ci0_month", "note"]
    est = est.reindex(columns=cols)

    RM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    out_csv = RM / "proactive_incident_estimates.csv"
    est.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv} ({len(est)} rows)")

    make_figure(est, {"n_treated_focal": len(treated_units),
                      "n_neighbors": neighbors["bbl"].nunique(),
                      "n_control": n_ctrl})

    # ── headline report ──
    def val(series, term, col="coef"):
        r = est[(est["series"] == series) & (est["term"] == term)]
        return float(r[col].iloc[0])

    s_m0 = float(est[(est["series"] == "focal_strict")
                     & (est["term"] == "event_month")
                     & (est["event_time"] == 0)]["coef"].iloc[0])
    print("\nHEADLINE (focal_strict = new attention beyond the mandated response)")
    print(f"  month 0: {s_m0:+.4f} per lot-month (total response "
          f"{float(est[(est['series'] == 'focal_total_response') & (est['term'] == 'event_month') & (est['event_time'] == 0)]['coef'].iloc[0]):+.4f})")
    print(f"  months 0-3 avg: {val('focal_strict', 'avg_months_0_3'):+.4f} "
          f"[{val('focal_strict', 'avg_months_0_3', 'ci_low'):+.4f}, "
          f"{val('focal_strict', 'avg_months_0_3', 'ci_high'):+.4f}]")
    print(f"  = {val('focal_strict', 'pct_avg03_vs_pre12_mean'):+.0f}% of the "
          f"12-month pre mean; {val('focal_strict', 'pct_avg03_vs_m1_level'):+.0f}% "
          f"of the month -1 level")
    print(f"  ramp-continuation z (m0 minus linear ramp): "
          f"{val('focal_strict', 'ramp_m0_minus_linear', 'z'):.1f} "
          f"(total-response series: "
          f"{val('focal_total_response', 'ramp_m0_minus_linear', 'z'):.1f})")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
