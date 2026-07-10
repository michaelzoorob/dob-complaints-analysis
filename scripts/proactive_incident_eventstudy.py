#!/usr/bin/env python3
"""
Hypothesis 5 (proactive_enforcement_plan.md): do unstable-building incidents
redirect agency-initiated enforcement to the lot, and to its block, afterward?

Stacked event study around each lot's FIRST caller-originated category-30
(structural-stability) complaint. Event months 2021-01..2025-05, so a full
+/-12-month window fits inside the 2020-01..2026-05 spine.

Outcome (BBL x calendar month): count of agency-originated complaints
(empty ref_311 in bis_scrape) EXCLUDING category 30, so the mandated cat-30
response and duplicate reports cannot mechanically create the jump; the
trigger itself is caller-originated and so never enters any outcome. A
variant that adds agency cat-30 follow-ups back in is estimated separately
(series focal_incl_cat30).

Design (stacked, clean controls): for each treated lot, up to 5 control
lots sampled without replacement from the same census tract (same-borough
fallback), restricted to spine lots that are never treated or treated more
than 24 months after the cohort month (their own +/-12 window cannot
overlap the cohort window) and whose block has no caller cat-30 complaint
anywhere in the cohort window (kills block spillover onto controls; also
removes same-block lots). Spec follows owner_transition_panel.py:

    y ~ i(event_t, treat, ref=-1) + C(event_t) | bbl + cal_month

Focal series clusters by BBL (treatment varies at the lot). The same-block
spillover series takes block-FIRST incidents, treats the other spine lots
on the block (up to 10 per event, own-treatment-clean), and clusters by
block, the level the spillover treatment varies (CLAUDE.md convention).

Universe caveat: control and neighbor lots come from the complaint spine
(lots with at least one scraped complaint 2020-2026, n=205k with tracts);
lots with no complaints ever are all-zero on the outcome and are not
sampled.

Writes
    data/analysis/risk_models/proactive_incident_estimates.csv
    data/analysis/blog_posts/artifacts/proactive_incident.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_incident_eventstudy.py
"""

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

# house palette (make_descriptive_figures.py)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"; AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": ZERO, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ── data ─────────────────────────────────────────────────────────────────

def load_spine():
    df = pd.read_csv(SPINE, usecols=["month", "category_prefix", "agency",
                                     "bbl", "bct2020"],
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

def expand(units, treat, count_excl30, count_incl30):
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
    return out


# ── estimation ───────────────────────────────────────────────────────────

def run_series(panel, ycol, cluster, series, meta):
    import pyfixest as pf
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

    def treat_idx(keep):
        out = []
        for i, n in enumerate(names):
            if n.startswith("event_t::") and n.endswith(":treat"):
                k = int(n.split("::")[1].split(":")[0])
                if keep(k):
                    out.append(i)
        return out

    # joint pre-trend Wald (event months -12..-2), cluster-robust chi2
    pre = treat_idx(lambda k: k < -1)
    R = np.zeros((len(pre), len(names)))
    for j, i in enumerate(pre):
        R[j, i] = 1.0
    w = m.wald_test(R=R)
    pre_chi2, pre_p = float(w["statistic"]), float(w["pvalue"])

    # average of months 0..3 (linear combination)
    post = treat_idx(lambda k: 0 <= k <= 3)
    a = np.zeros(len(names))
    a[post] = 1.0 / len(post)
    jump = float(a @ beta)
    jump_se = float(np.sqrt(a @ V @ a))

    base = float(panel.loc[(panel["treat"] == 1) & (panel["event_t"] < 0), ycol].mean())

    rows = [{"series": series, "term": "event_month", "event_time": -1,
             "coef": 0.0, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "pval": np.nan}]
    for _, r in g.iterrows():
        rows.append({"series": series, "term": "event_month",
                     "event_time": int(r["event_time"]),
                     "coef": r["Estimate"], "se": r["Std. Error"],
                     "ci_low": r["2.5%"], "ci_high": r["97.5%"],
                     "pval": r["Pr(>|t|)"]})
    rows.append({"series": series, "term": "avg_months_0_3", "event_time": np.nan,
                 "coef": jump, "se": jump_se,
                 "ci_low": jump - 1.96 * jump_se, "ci_high": jump + 1.96 * jump_se,
                 "pval": np.nan})
    out = pd.DataFrame(rows)

    post_rows = out[(out["term"] == "event_month") & (out["event_time"] >= 1)]
    dead = post_rows[(post_rows["ci_low"] <= 0) & (post_rows["ci_high"] >= 0)]
    decay = int(dead["event_time"].min()) if len(dead) else np.nan

    out["cluster"] = cluster
    out["n_treated_units"] = meta["n_treated"]
    out["n_control_units"] = meta["n_control"]
    out["n_obs"] = m._N
    out["baseline_mean_treated"] = base
    out["pretrend_chi2"] = pre_chi2
    out["pretrend_df"] = len(pre)
    out["pretrend_p"] = pre_p
    out["decay_first_ci0_month"] = decay

    print(f"\n[{series}] N={m._N:,} obs, {meta['n_treated']:,} treated / "
          f"{meta['n_control']:,} control units, cluster={cluster}")
    print(f"  months 0-3 jump: {jump:+.4f} per lot-month (se {jump_se:.4f}), "
          f"baseline {base:.4f} -> {jump / base * 100:+.0f}%")
    print(f"  pre-trend joint chi2({len(pre)}) = {pre_chi2:.1f}, p = {pre_p:.3f}")
    print(f"  first post month (>=1) with CI covering 0: "
          f"{decay if not np.isnan(decay) else '>12'}")
    return out


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(est, meta):
    def series(name):
        g = est[(est["series"] == name) & (est["term"] == "event_month")].copy()
        return g.sort_values("event_time")

    foc, spl = series("focal"), series("block_spillover")
    fig, ax = plt.subplots(figsize=(8.6, 5.5), dpi=200,
                           gridspec_kw={"left": 0.105, "right": 0.97,
                                        "top": 0.755, "bottom": 0.115})

    def draw(g, color, xoff, label):
        x = g["event_time"] + xoff
        ax.plot(x, g["coef"], color=color, lw=2, zorder=3, label=label)
        ok = g["se"].notna()
        ax.vlines(x[ok], g.loc[ok, "ci_low"], g.loc[ok, "ci_high"],
                  color=color, lw=1.4, alpha=0.8, zorder=2)
        ax.plot(x, g["coef"], "o", ms=6, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)

    ax.axhline(0, color=ZERO, lw=1.1)
    ax.axvline(-0.5, color=MUTED, lw=1.0, ls=(0, (1, 2)))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(range(-12, 13, 2))
    ax.tick_params(labelsize=9)
    draw(foc, BLUE, -0.12, "the lot itself")
    draw(spl, AQUA, +0.12, "other lots on the same block")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_xlabel("months relative to the complaint (reference = month before)",
                  fontsize=9.5)
    ax.set_ylabel("difference in agency-initiated complaints\nper lot-month",
                  fontsize=9)
    ymax = ax.get_ylim()[1]
    ax.text(-0.85, ymax * 0.97, "unstable-building\ncomplaint filed",
            fontsize=8.5, color=MUTED, ha="right", va="top", style="italic")

    fig.suptitle("After an unstable-building complaint, agency enforcement\n"
                 "spikes at the lot for three months and stays flat on the block",
                 x=0.02, y=0.985, ha="left", fontsize=13, color=INK,
                 weight="semibold")
    fig.text(0.02, 0.845,
             f"First caller-reported category-30 complaint per lot, 2021-2025 "
             f"({meta['n_treated_focal']:,} lots; {meta['n_neighbors']:,} same-block "
             f"lots) vs. {meta['n_control']:,} not-yet-treated\n"
             "lots matched within census tract · outcome counts agency-initiated "
             "complaints excluding category-30 follow-ups\n"
             "lot and calendar-month fixed effects · whiskers = 95% CI, SEs "
             "clustered by lot (focal) and by block (spillover)\n"
             "focal pre-event coefficients drift upward toward the complaint "
             "(joint pre-trend p < 0.001)",
             fontsize=8.5, color=MUTED, va="top")
    fig.savefig(ART / "proactive_incident.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {ART / 'proactive_incident.png'}")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_spine()
    ev, uni, count_excl30, count_incl30, block_incidents, blk_first = build_inputs(df)

    controls = match_controls(ev, uni, block_incidents, rng)
    neighbors = match_neighbors(ev, uni, blk_first, rng)

    treated_units = ev.rename(columns={"bbl_i": "bbl"})[["bbl", "block", "m0"]]
    pan_treated = expand(treated_units, 1, count_excl30, count_incl30)
    pan_controls = expand(controls, 0, count_excl30, count_incl30)
    pan_neighbors = expand(neighbors, 1, count_excl30, count_incl30)

    focal_panel = pd.concat([pan_treated, pan_controls], ignore_index=True)
    # spillover regression: neighbor lots vs the same matched controls,
    # dropping controls that themselves sit on a treated block
    keep = ~pan_controls["bbl"].isin(set(neighbors["bbl"]))
    spill_panel = pd.concat([pan_neighbors, pan_controls[keep]], ignore_index=True)
    print(f"\npanels: focal {len(focal_panel):,} rows, spillover "
          f"{len(spill_panel):,} rows "
          f"({(~keep).sum() // (2 * WINDOW + 1):,} control lots dropped as neighbors)")

    n_ctrl = controls["bbl"].nunique()
    est = pd.concat([
        run_series(focal_panel, "y_excl30", "bbl", "focal",
                   {"n_treated": len(treated_units), "n_control": n_ctrl}),
        run_series(focal_panel, "y_incl30", "bbl", "focal_incl_cat30",
                   {"n_treated": len(treated_units), "n_control": n_ctrl}),
        run_series(spill_panel, "y_excl30", "block", "block_spillover",
                   {"n_treated": neighbors["bbl"].nunique(),
                    "n_control": int(keep.sum()) // (2 * WINDOW + 1)}),
    ], ignore_index=True)

    RM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    out_csv = RM / "proactive_incident_estimates.csv"
    est.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv} ({len(est)} rows)")

    make_figure(est, {"n_treated_focal": len(treated_units),
                      "n_neighbors": neighbors["bbl"].nunique(),
                      "n_control": n_ctrl})
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
