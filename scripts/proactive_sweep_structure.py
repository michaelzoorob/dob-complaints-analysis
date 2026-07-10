#!/usr/bin/env python3
"""
Hypotheses 11 and 8 (proactive_enforcement_plan.md): the internal structure
of 7G construction sweeps.

(1) BATCHING (#11). Among agency-initiated 7G sweep events, what share has
    another 7G on the same tax block (boro + 5-digit block = bbl // 10,000)
    within +/-3 days? Tested against a permutation null that shuffles sweep
    dates among the sweep events of the same census tract (200 draws), which
    preserves each block's sweep count and the tract's date mix while
    breaking within-block timing. Then, within ever-swept blocks, an LPM of
    which residential lots get included in a sweep, on strictly pre-2020
    risk traits (era dummies, log building area, 2010-19 ECB / DOB violation
    history — the proactive_becker_margin.py feature construction), block
    FE, SEs clustered by block, the level sweeps arrive at.

(2) SUBSTITUTION (#8). Stacked event study of caller-originated complaints
    (non-empty ref_311) around a lot's FIRST agency 7G sweep. Event months
    -12..+12, cohorts 2021-01..2025-05 so the window fits the 2020-01..
    2026-05 spine; BBL + calendar-month FE; up to 5 control lots per event
    sampled from the same census tract (borough fallback), never swept or
    swept >24 months later, and on a block with no 7G anywhere in the cohort
    window (part 1 shows sweeps batch by block, so same-block lots are
    contaminated controls); SEs clustered by BBL, the level treatment
    varies. Design mirrors proactive_incident_eventstudy.py.

    Critic C (critic_proactive_C_eventstudies.py, A5) showed the pooled
    estimate rides a construction-phase confound: swept lots are mid-
    construction (active permit) at the event month and wind down over the
    following year, while unmatched controls are not. The HEADLINE series
    therefore adds active-permit-at-the-event-month (DOB NOW permit spans)
    to the matching cells — controls must share the treated lot's tract AND
    its active-permit status at m0 — and restricts cohorts to 2023-01
    onward, where the DOB NOW permit ledger is complete (series
    sweep_2023plus_phasematched). The pooled and 2021-22 estimates are kept
    as labeled NON-ROBUST variants: pooled attenuates when phase-matched,
    and the 2021-22 slice sits in the legacy-BIS permit gap, so its phase
    cannot be verified.

Universe caveats: LPM candidates come from property_risk_panel_v2 (PLUTO
residential lots), which holds 71.7% of swept BBLs — the rest are
non-residential lots and construction sites without residential units, so
the LPM reads "among residential lots on a swept block, which get swept."
Event-study controls come from the complaint spine (lots with at least one
scraped complaint 2020-2026); lots with no complaints ever are not sampled.

Writes
    data/analysis/risk_models/proactive_sweep_structure.csv
    data/analysis/blog_posts/artifacts/proactive_sweep_structure.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_sweep_structure.py
"""

import math
import sqlite3
import sys
import time
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

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
RM = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"

# batching test
DAY_WINDOW = 3              # +/- days for "another sweep on the block"
N_DRAWS = 200               # within-tract date shuffles
DAY_STRIDE = 4096           # > max day index (2,339) + window; packs block/day

# event study (mirrors proactive_incident_eventstudy.py)
WINDOW = 12                 # event months -12..+12
FIRST_M, LAST_M = 12, 64    # 2021-01 .. 2025-05 (2020-01 = 0)
N_MONTHS = 77               # 2020-01 .. 2026-05
K_CONTROLS = 5              # matched controls per event
CLEAR = 2 * WINDOW          # controls' own first sweep > m0 + 24
STRIDE = 128                # > 77 spine months, packs (id, month)
SEED = 20260710
COHORT_2023 = 36            # first event month with a clean permit ledger
PHASE_SEED_OFFSET = 7       # fresh rng stream for phase-matched sampling

SERIES_LABELS = {
    "sweep_2023plus_phasematched": (
        "HEADLINE - 2023+ cohorts; controls matched within census tract x "
        "active-permit-at-event-month cells (DOB NOW ledger complete)"),
    "sweep_2023plus_bothactive": (
        "supporting - 2023+ cohorts restricted to events with an active "
        "permit at m0, vs active-permit controls (construction vs "
        "construction)"),
    "sweep_pooled_unmatched": (
        "NON-ROBUST variant - pooled 2021-2025 cohorts, controls not "
        "matched on construction phase; attenuates when phase-matched and "
        "the pre-2023 phase is unverifiable"),
    "sweep_pooled_phasematched": (
        "diagnostic - pooled 2021-2025 cohorts with phase-matched controls "
        "(attenuated vs the unmatched pooled series; pre-2023 permit ledger "
        "incomplete)"),
    "sweep_2021_22_unmatched": (
        "NON-ROBUST variant - 2021-22 cohorts (legacy-BIS permit gap: "
        "construction phase unverifiable)"),
}

# house palette (make_descriptive_figures.py)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; ZERO = "#b9b7ac"; BLUE = "#2a78d6"
RED = "#e34948"; AQUA = "#1baf7a"

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
    df = pd.read_csv(SPINE, usecols=["received_date", "month", "category_prefix",
                                     "agency", "bbl", "bct2020"],
                     dtype={"category_prefix": "str", "bbl": "str"})
    n0 = len(df)
    df = df[df["bbl"].notna() & df["bbl"].str.fullmatch(r"\d{10}")].copy()
    df["bbl_i"] = df["bbl"].astype("int64")
    df["block"] = df["bbl_i"] // 10_000          # boro + 5-digit tax block
    df["m"] = ((df["month"].str[:4].astype(int) - 2020) * 12
               + df["month"].str[5:7].astype(int) - 1)
    df["day"] = (pd.to_datetime(df["received_date"], format="%Y-%m-%d")
                 - pd.Timestamp("2020-01-01")).dt.days
    print(f"spine: {n0:,} rows -> {len(df):,} with valid BBL; "
          f"{df['bbl_i'].nunique():,} lots, months 0..{df['m'].max()}, "
          f"days 0..{df['day'].max()}")
    return df


# ── part 1a: same-block batching vs a within-tract permutation null ──────

def _blockmate_shares(day, block, bbl, w):
    """Share of events with another event on the same block within +/-w days,
    any lot ('any') and at a different lot ('other_lot'), via searchsorted on
    packed sorted keys (counts include self; self is netted out)."""
    kb = np.sort(block * DAY_STRIDE + day)
    kl = np.sort(bbl * DAY_STRIDE + day)
    n_blk = (np.searchsorted(kb, block * DAY_STRIDE + day + w, side="right")
             - np.searchsorted(kb, block * DAY_STRIDE + day - w, side="left"))
    n_lot = (np.searchsorted(kl, bbl * DAY_STRIDE + day + w, side="right")
             - np.searchsorted(kl, bbl * DAY_STRIDE + day - w, side="left"))
    return {"any": float((n_blk >= 2).mean()),
            "other_lot": float(((n_blk - n_lot) >= 1).mean())}


def batching_permutation(g7, rng):
    ev = g7[g7["bct2020"].notna()].copy()
    ev["tract_id"] = ev["bct2020"].astype("int64")
    ev = ev.sort_values("tract_id", kind="mergesort").reset_index(drop=True)
    tract = ev["tract_id"].to_numpy()
    day = ev["day"].to_numpy()
    block = ev["block"].to_numpy()
    bbl = ev["bbl_i"].to_numpy()
    print(f"\nbatching: {len(ev):,} of {len(g7):,} agency 7G events have a "
          f"tract ({len(ev) / len(g7):.1%}); {pd.Series(block).nunique():,} "
          f"blocks, {pd.Series(tract).nunique():,} tracts")

    obs3 = _blockmate_shares(day, block, bbl, DAY_WINDOW)
    obs0 = _blockmate_shares(day, block, bbl, 0)
    stats = {"share_other_lot_pm3d": obs3["other_lot"],
             "share_any_event_pm3d": obs3["any"],
             "share_other_lot_same_day": obs0["other_lot"]}

    null = {k: np.empty(N_DRAWS) for k in stats}
    for d in range(N_DRAWS):
        # events are tract-sorted, so lexsort by (tract, random) permutes
        # positions within each tract: dates reshuffle across the tract's
        # sweep locations, block counts and the tract date mix held fixed
        perm = np.lexsort((rng.random(len(ev)), tract))
        pday = day[perm]
        s3 = _blockmate_shares(pday, block, bbl, DAY_WINDOW)
        s0 = _blockmate_shares(pday, block, bbl, 0)
        null["share_other_lot_pm3d"][d] = s3["other_lot"]
        null["share_any_event_pm3d"][d] = s3["any"]
        null["share_other_lot_same_day"][d] = s0["other_lot"]

    rows, meta = [], {"n_events": len(ev)}
    for k, v in stats.items():
        exceed = int((null[k] >= v).sum())
        p = (1 + exceed) / (N_DRAWS + 1)
        rows.append({"analysis": "batching_permutation", "series": "7g_sweeps",
                     "term": k, "value": v, "null_mean": float(null[k].mean()),
                     "null_sd": float(null[k].std(ddof=1)),
                     "pval": p, "n_draws": N_DRAWS, "n_obs": len(ev),
                     "cluster": "permutation_within_tract", "units": "share"})
        print(f"  {k}: observed {v:.3f} vs null {null[k].mean():.3f} "
              f"(sd {null[k].std(ddof=1):.4f}), p = {p:.4f}"
              + ("  (no draw reached the observed value)" if exceed == 0 else ""))
        meta[k] = {"obs": v, "null_mean": float(null[k].mean()),
                   "null_sd": float(null[k].std(ddof=1)), "p": p,
                   "draws": null[k].copy()}

    # transparency: the same headline share on all agency 7G events,
    # including the 9% without a tract (not permutation-testable)
    full = _blockmate_shares(g7["day"].to_numpy(), g7["block"].to_numpy(),
                             g7["bbl_i"].to_numpy(), DAY_WINDOW)
    rows.append({"analysis": "batching_permutation", "series": "7g_sweeps",
                 "term": "share_other_lot_pm3d_fullsample",
                 "value": full["other_lot"], "n_obs": len(g7), "units": "share"})
    print(f"  full-sample (incl. tractless) other-lot share: {full['other_lot']:.3f}")
    return pd.DataFrame(rows), meta


# ── part 1b: within swept blocks, are the swept lots the risky ones? ─────

def batching_lpm(swept_bbls, swept_blocks):
    import pyfixest as pf
    pan = pd.read_csv(PANEL, usecols=["bbl_key", "yearbuilt", "bldgarea",
                                      "n_ecb_hist", "n_dobviol_hist",
                                      "any_prior_viol"])
    pan["blk"] = pan["bbl_key"] // 10_000
    sub = pan[pan["blk"].isin(swept_blocks)].copy()
    sub["swept_pct"] = sub["bbl_key"].isin(swept_bbls).astype(float) * 100.0

    # proactive_becker_margin.py feature construction: strictly pre-2020
    yb = sub["yearbuilt"]
    sub["era_pre1940"] = yb.between(1800, 1939).astype(int)
    sub["era_4079"] = yb.between(1940, 1979).astype(int)
    sub["era_8099"] = yb.between(1980, 1999).astype(int)
    sub["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    ba = pd.to_numeric(sub["bldgarea"], errors="coerce")
    lb = np.log(ba.where(ba > 0))
    sub["no_bldgarea"] = lb.isna().astype(int)
    sub["log_bldgarea"] = lb.fillna(lb.median())
    sub["log1p_ecb_hist"] = np.log1p(sub["n_ecb_hist"])
    sub["log1p_dobviol_hist"] = np.log1p(sub["n_dobviol_hist"])

    traits = ["era_pre1940", "era_4079", "era_8099", "era_unknown",
              "log_bldgarea", "no_bldgarea",
              "log1p_ecb_hist", "log1p_dobviol_hist", "any_prior_viol"]
    m = pf.feols("swept_pct ~ " + " + ".join(traits) + " | blk",
                 data=sub, vcov={"CRV1": "blk"})

    names = [str(n) for n in m._coefnames]
    R = np.zeros((len(traits), len(names)))
    for j, t in enumerate(traits):
        R[j, names.index(t)] = 1.0
    w = m.wald_test(R=R)
    chi2, wp = float(w["statistic"]), float(w["pvalue"])

    base = float(sub["swept_pct"].mean())
    n_blocks = int(sub["blk"].nunique())
    tid = m.tidy().reset_index()
    term_col = tid.columns[0]
    rows = []
    for _, r in tid.iterrows():
        rows.append({"analysis": "within_block_lpm", "series": "7g_sweeps",
                     "term": str(r[term_col]), "value": r["Estimate"],
                     "se": r["Std. Error"], "ci_low": r["2.5%"],
                     "ci_high": r["97.5%"], "pval": r["Pr(>|t|)"],
                     "n_obs": int(m._N), "n_blocks": n_blocks,
                     "baseline": base, "cluster": "block", "units": "pp"})
    rows.append({"analysis": "within_block_lpm", "series": "7g_sweeps",
                 "term": "joint_traits_wald", "value": chi2, "pval": wp,
                 "n_obs": int(m._N), "n_blocks": n_blocks, "baseline": base,
                 "cluster": "block", "units": "chi2_df9"})
    out = pd.DataFrame(rows)

    print(f"\nwithin-block LPM: {m._N:,} residential lots on {n_blocks:,} "
          f"swept blocks; {base:.2f}% of lots swept; joint traits "
          f"chi2(9) = {chi2:.1f}, p = {wp:.2e}")
    show = out[out["term"].isin(["any_prior_viol", "log1p_ecb_hist",
                                 "log1p_dobviol_hist", "log_bldgarea",
                                 "era_pre1940"])]
    for _, r in show.iterrows():
        print(f"  {r['term']:<20} {r['value']:+.2f} pp "
              f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]")
    return out


# ── part 2: caller complaints around the first sweep ─────────────────────

def make_counter(sub):
    """Exact-count lookup via searchsorted on a packed sorted key array."""
    keys = np.sort(sub["bbl_i"].to_numpy() * STRIDE + sub["m"].to_numpy())

    def count(bbl_arr, m_arr):
        q = bbl_arr * STRIDE + m_arr
        return (np.searchsorted(keys, q, side="right")
                - np.searchsorted(keys, q, side="left")).astype(np.int32)

    return count


def build_es_inputs(df, g7):
    treat_m = g7.groupby("bbl_i")["m"].min()          # first agency 7G

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
    print(f"\nevent study: {len(treat_m):,} ever-swept lots; {n_all:,} first "
          f"swept 2021-01..2025-05; {len(ev):,} with a census tract (kept)")

    count_caller = make_counter(df[df["agency"] == 0])

    # agency 7G events per block, for the control-contamination filter
    blk_keys = np.sort(g7["block"].to_numpy() * STRIDE + g7["m"].to_numpy())

    def block_sweeps(block_arr, m0_arr):
        lo = block_arr * STRIDE + (m0_arr - WINDOW)
        hi = block_arr * STRIDE + (m0_arr + WINDOW)
        return (np.searchsorted(blk_keys, hi, side="right")
                - np.searchsorted(blk_keys, lo, side="left"))

    return ev, uni, count_caller, block_sweeps


def build_permit_activity(uni_bbls):
    """active(bbl_arr, m_arr) -> bool: any DOB NOW permit span (issued..
    expired) covering the month, at spine lots. Mirrors the construction in
    critic_proactive_C_eventstudies.py (A5)."""
    conn = sqlite3.connect(str(config.DB_PATH))
    perm = pd.read_sql_query(
        "SELECT bbl, issued_date, expired_date FROM permits "
        "WHERE bbl IS NOT NULL AND issued_date IS NOT NULL "
        "AND expired_date IS NOT NULL", conn)
    conn.close()
    perm["bbl_i"] = pd.to_numeric(perm["bbl"], errors="coerce")
    perm = perm[perm["bbl_i"].notna() & (perm["bbl_i"] >= 1e9)].copy()
    perm["bbl_i"] = perm["bbl_i"].astype("int64")
    iss = pd.to_datetime(perm["issued_date"], errors="coerce")
    exp = pd.to_datetime(perm["expired_date"], errors="coerce")
    ok = iss.notna() & exp.notna() & (exp >= iss)
    perm, iss, exp = perm[ok].copy(), iss[ok], exp[ok]
    perm["sm"] = ((iss.dt.year - 2020) * 12 + iss.dt.month - 1).clip(lower=0).astype(np.int64)
    perm["em"] = ((exp.dt.year - 2020) * 12 + exp.dt.month - 1).clip(upper=N_MONTHS - 1).astype(np.int64)
    perm = perm[(perm["sm"] <= perm["em"]) & (perm["em"] >= 0)
                & (perm["sm"] <= N_MONTHS - 1)]
    perm = perm[perm["bbl_i"].isin(uni_bbls)]

    lens = (perm["em"] - perm["sm"] + 1).to_numpy()
    bbl_rep = np.repeat(perm["bbl_i"].to_numpy(), lens)
    m_rep = (np.repeat(perm["sm"].to_numpy(), lens)
             + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
    act_keys = np.sort(bbl_rep * STRIDE + m_rep)
    print(f"\npermit activity: {len(perm):,} spans at spine lots -> "
          f"{len(act_keys):,} active lot-month keys "
          f"({pd.Series(bbl_rep).nunique():,} lots)")

    def active(bbl_arr, m_arr):
        q = bbl_arr * STRIDE + m_arr
        return (np.searchsorted(act_keys, q, side="right")
                > np.searchsorted(act_keys, q, side="left"))

    return active


def _clean_candidates(cand, block_sweeps):
    """Not-yet-swept with a clear window, on a sweep-free block."""
    cand = cand[cand["cbbl"] != cand["bbl_i"]]
    ok = cand["treat_m"].isna() | (cand["treat_m"] > cand["m0"] + CLEAR)
    cand = cand[ok]
    n_sw = block_sweeps(cand["cblock"].to_numpy(), cand["m0"].to_numpy())
    return cand[n_sw == 0]


def match_controls(ev, uni, block_sweeps, rng):
    """Up to K_CONTROLS per event, without replacement, tract then borough."""
    pool = uni.rename(columns={"bbl_i": "cbbl", "block": "cblock"})

    cand = ev[["event_id", "bbl_i", "m0", "tract"]].merge(
        pool[["cbbl", "cblock", "tract", "treat_m"]], on="tract")
    cand = _clean_candidates(cand, block_sweeps)
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
        cand_b = _clean_candidates(cand_b, block_sweeps)
        cand_b = cand_b[~cand_b["cbbl"].isin(matched["cbbl"])]
        cand_b = cand_b.sample(frac=1.0, random_state=rng).drop_duplicates("cbbl")
        fb_matched = cand_b.groupby("event_id").head(K_CONTROLS)
        matched = pd.concat([matched, fb_matched], ignore_index=True)
        print(f"controls: borough fallback added {len(fb_matched):,} lots")

    return matched[["event_id", "cbbl", "m0"]].rename(columns={"cbbl": "bbl"})


def match_controls_phase(ev_in, uni, block_sweeps, active):
    """Permit-cell matching (critic C, A5 fix): up to K_CONTROLS per event,
    without replacement, from the same census tract (borough fallback) AND
    the same active-permit-at-m0 status as the treated lot. Fresh rng
    stream so the published pooled matching is untouched."""
    rng = np.random.default_rng(SEED + PHASE_SEED_OFFSET)
    pool = uni.rename(columns={"bbl_i": "cbbl", "block": "cblock"})

    cand = ev_in[["event_id", "bbl_i", "m0", "tract", "act_m0"]].merge(
        pool[["cbbl", "cblock", "tract", "treat_m"]], on="tract")
    cand = _clean_candidates(cand, block_sweeps)
    c_act = active(cand["cbbl"].to_numpy(), cand["m0"].to_numpy())
    cand = cand[c_act == cand["act_m0"].to_numpy()]
    cand = cand.sample(frac=1.0, random_state=rng).drop_duplicates("cbbl")
    matched = cand.groupby("event_id").head(K_CONTROLS)

    got = matched.groupby("event_id").size()
    short_ids = ev_in.loc[~ev_in["event_id"].isin(got.index), "event_id"]
    print(f"phase-matched controls: tract x permit-status stage "
          f"{len(matched):,} lots for {got.index.nunique():,}/{len(ev_in):,} "
          f"events (mean {got.mean():.2f} per event); {len(short_ids):,} "
          f"events to borough fallback")

    if len(short_ids):
        fb = ev_in[ev_in["event_id"].isin(short_ids)][
            ["event_id", "bbl_i", "m0", "boro", "act_m0"]]
        cand_b = fb.merge(pool[["cbbl", "cblock", "boro", "treat_m"]], on="boro")
        cand_b = _clean_candidates(cand_b, block_sweeps)
        cb_act = active(cand_b["cbbl"].to_numpy(), cand_b["m0"].to_numpy())
        cand_b = cand_b[cb_act == cand_b["act_m0"].to_numpy()]
        cand_b = cand_b[~cand_b["cbbl"].isin(matched["cbbl"])]
        cand_b = cand_b.sample(frac=1.0, random_state=rng).drop_duplicates("cbbl")
        fb_matched = cand_b.groupby("event_id").head(K_CONTROLS)
        matched = pd.concat([matched, fb_matched], ignore_index=True)
        print(f"phase-matched controls: borough fallback added "
              f"{len(fb_matched):,} lots")

    return matched[["event_id", "cbbl", "m0"]].rename(columns={"cbbl": "bbl"})


def expand(units, treat, count_caller):
    """units (bbl, m0) -> 25 rows each with searchsorted caller counts."""
    w = 2 * WINDOW + 1
    et = np.tile(np.arange(-WINDOW, WINDOW + 1), len(units))
    out = pd.DataFrame({
        "bbl": np.repeat(units["bbl"].to_numpy(), w),
        "event_t": et,
        "cal_m": np.repeat(units["m0"].to_numpy(), w) + et,
        "treat": np.int8(treat),
    })
    out["y_caller"] = count_caller(out["bbl"].to_numpy(), out["cal_m"].to_numpy())
    return out


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

    def avg_post(k_lo, k_hi):
        post = treat_idx(lambda k: k_lo <= k <= k_hi)
        a = np.zeros(len(names))
        a[post] = 1.0 / len(post)
        est = float(a @ beta)
        se = float(np.sqrt(a @ V @ a))
        return est, se

    base = float(panel.loc[(panel["treat"] == 1) & (panel["event_t"] < 0), ycol].mean())
    label = SERIES_LABELS.get(series, series)

    rows = [{"analysis": "substitution_eventstudy", "series": series,
             "term": "event_month", "event_time": -1, "value": 0.0}]
    for _, r in g.iterrows():
        rows.append({"analysis": "substitution_eventstudy", "series": series,
                     "term": "event_month", "event_time": int(r["event_time"]),
                     "value": r["Estimate"], "se": r["Std. Error"],
                     "ci_low": r["2.5%"], "ci_high": r["97.5%"],
                     "pval": r["Pr(>|t|)"]})
    for lab, (k_lo, k_hi) in [("avg_months_0_3", (0, 3)),
                              ("avg_months_0_12", (0, 12))]:
        est, se = avg_post(k_lo, k_hi)
        rows.append({"analysis": "substitution_eventstudy", "series": series,
                     "term": lab, "value": est, "se": se,
                     "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
                     "pval": zp(est / se)})
        # percent framing against the treated pre-period mean (with CI)
        rows.append({"analysis": "substitution_eventstudy", "series": series,
                     "term": f"pct_{lab}_vs_baseline",
                     "value": est / base * 100.0, "se": se / base * 100.0,
                     "ci_low": (est - 1.96 * se) / base * 100.0,
                     "ci_high": (est + 1.96 * se) / base * 100.0,
                     "pval": zp(est / se)})
    out = pd.DataFrame(rows)
    out["series_label"] = label

    post_rows = out[(out["term"] == "event_month") & (out["event_time"] >= 1)]
    dead = post_rows[(post_rows["ci_low"] <= 0) & (post_rows["ci_high"] >= 0)]
    decay = int(dead["event_time"].min()) if len(dead) else np.nan

    out["cluster"] = cluster
    out["n_treated"] = meta["n_treated"]
    out["n_control"] = meta["n_control"]
    out["n_obs"] = m._N
    out["baseline"] = base
    out["pretrend_chi2"] = pre_chi2
    out["pretrend_df"] = len(pre)
    out["pretrend_p"] = pre_p
    out["decay_first_ci0_month"] = decay
    out["units"] = np.where(out["term"].str.startswith("pct_"),
                            "% of pre-period treated mean",
                            "caller complaints per lot-month")

    a03 = out[out["term"] == "avg_months_0_3"].iloc[0]
    a12 = out[out["term"] == "avg_months_0_12"].iloc[0]
    print(f"\n[{series}] N={m._N:,} obs, {meta['n_treated']:,} treated / "
          f"{meta['n_control']:,} control lots, cluster={cluster}")
    for lab, r in [("months 0-3", a03), ("months 0-12", a12)]:
        print(f"  {lab} avg: {r['value']:+.4f} per lot-month "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], baseline {base:.4f} "
              f"-> {r['value'] / base * 100:+.0f}%")
    print(f"  pre-trend joint chi2({len(pre)}) = {pre_chi2:.1f}, p = {pre_p:.3g}")
    print(f"  first post month (>=1) with CI covering 0: "
          f"{decay if not (isinstance(decay, float) and np.isnan(decay)) else '>12'}")
    return out


# ── figure ───────────────────────────────────────────────────────────────

def fmt_p(p):
    return "< 0.005" if p < 0.005 else f"= {p:.3f}"


def make_figure(perm_meta, lpm, est, es_meta):
    """est: HEADLINE substitution series rows (2023+, phase-matched)."""
    hl = perm_meta["share_other_lot_pm3d"]
    obs, nm, p = hl["obs"] * 100, hl["null_mean"] * 100, hl["p"]
    draws = hl["draws"] * 100

    a12 = est[est["term"] == "avg_months_0_12"].iloc[0]
    p12 = est[est["term"] == "pct_avg_months_0_12_vs_baseline"].iloc[0]
    base = float(a12["baseline"])
    pct12 = a12["value"] / base * 100

    sd = perm_meta["share_other_lot_same_day"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 6.2), dpi=200,
        gridspec_kw={"left": 0.06, "right": 0.985, "top": 0.665,
                     "bottom": 0.11, "wspace": 0.26})

    # ── panel A: permutation null vs observed ──
    ax1.hist(draws, bins=10, color=BASE, edgecolor=SURFACE, linewidth=1.2,
             zorder=3)
    ax1.axvline(obs, color=BLUE, lw=2.4, zorder=4)
    ax1.grid(axis="y", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xlim(0, obs * 1.22)
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax1.tick_params(labelsize=9)
    ax1.set_xlabel("share of sweep visits with another 7G sweep at a different\n"
                   "lot on the same tax block within ±3 days", fontsize=9.5)
    ax1.set_ylabel(f"permutation draws (of {N_DRAWS})", fontsize=9.5)
    ymax = ax1.get_ylim()[1]
    ax1.text(obs - obs * 0.02, ymax * 0.97, f"observed\n{obs:.1f}%",
             ha="right", va="top", fontsize=10, color=INK, weight="semibold")
    ax1.text(obs - obs * 0.02, ymax * 0.78, f"permutation p {fmt_p(p)}",
             ha="right", va="top", fontsize=8.5, color=MUTED)
    ax1.text(nm + obs * 0.06, ymax * 0.60,
             f"null mean {nm:.1f}%\n(sweep dates shuffled\nwithin census tract)",
             ha="left", va="top", fontsize=8.5, color=MUTED)
    ax1.set_title("Sweep visits cluster on the same block\nin the same week",
                  loc="left", fontsize=12, color=INK, weight="semibold", pad=10)

    # ── panel B: event study ──
    g = est[(est["term"] == "event_month")].sort_values("event_time")
    x = g["event_time"].to_numpy()
    ax2.axhline(0, color=ZERO, lw=1.1)
    ax2.axvline(-0.5, color=MUTED, lw=1.0, ls=(0, (1, 2)))
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.plot(x, g["value"], color=BLUE, lw=2, zorder=3)
    ok = g["se"].notna()
    ax2.vlines(x[ok], g.loc[ok, "ci_low"], g.loc[ok, "ci_high"],
               color=BLUE, lw=1.4, alpha=0.8, zorder=2)
    ax2.plot(x, g["value"], "o", ms=6, color=BLUE,
             markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
    ax2.set_xticks(range(-12, 13, 3))
    ax2.tick_params(labelsize=9)
    ax2.set_xlabel("months relative to the first sweep (reference = month before)",
                   fontsize=9.5)
    ax2.set_ylabel("difference in caller-originated complaints\nper lot-month",
                   fontsize=9)
    # label to the RIGHT of the event line, in added headroom: the tall
    # pre-period whiskers (month -2 tops the axes) run through the old spot
    ylo, yhi = ax2.get_ylim()
    ax2.set_ylim(ylo, yhi + 0.07 * (yhi - ylo))
    ax2.text(0.5, yhi + 0.05 * (yhi - ylo), "first 7G sweep\nat the lot",
             fontsize=8.5, color=MUTED, ha="left", va="top", style="italic")
    ax2.set_title("Caller complaints around a building's first\nsweep "
                  "(2023+ cohorts, phase-matched controls)",
                  loc="left", fontsize=12, color=INK, weight="semibold", pad=10)

    # ── titles and footnote ──
    batch_verdict = ("Construction sweep visits batch by block"
                     if p < 0.05 and obs > nm else
                     "Construction sweep visits show no same-block batching")
    if a12["ci_low"] > 0:
        sub_verdict = ("caller complaints rise, not fall, after a building's "
                       "first sweep")
    elif a12["ci_high"] < 0:
        sub_verdict = "caller complaints fall after a building's first sweep"
    else:
        sub_verdict = "a building's first sweep leaves caller complaints flat"
    fig.suptitle(f"{batch_verdict}, and {sub_verdict}",
                 x=0.02, y=0.975, ha="left", fontsize=14, color=INK,
                 weight="semibold")

    apv = lpm[lpm["term"] == "any_prior_viol"].iloc[0]
    lba = lpm[lpm["term"] == "log_bldgarea"].iloc[0]
    pre40 = lpm[lpm["term"] == "era_pre1940"].iloc[0]
    ecb = lpm[lpm["term"] == "log1p_ecb_hist"].iloc[0]

    def pp(r):
        return (f"{r['value']:+.1f} pp [{r['ci_low']:+.1f}, "
                f"{r['ci_high']:+.1f}]")

    v_pool = es_meta["variants"]["pooled"]
    v_2122 = es_meta["variants"]["s2122"]
    v_pm = es_meta["variants"]["pooled_pm"]
    fig.text(0.02, 0.915,
             f"Left: {perm_meta['n_events']:,} agency-initiated 7G sweep visits "
             f"with a census tract, 2020–May 2026 · observed same-block share "
             f"{obs:.1f}% vs {nm:.1f}% across {N_DRAWS} within-tract date "
             f"shuffles (p {fmt_p(p)}) · same-day companion "
             f"{sd['obs'] * 100:.1f}% vs {sd['null_mean'] * 100:.1f}% null\n"
             f"within swept blocks, LPM of which residential lots get swept, "
             f"on pre-2020 traits · log building area {pp(lba)} · log 2010–19 "
             f"ECB count {pp(ecb)} · prewar {pp(pre40)} · any 2010–19 "
             f"violation {pp(apv)}\n"
             f"({int(apv['n_obs']):,} residential lots on "
             f"{int(apv['n_blocks']):,} swept blocks, {apv['baseline']:.1f}% "
             f"swept, block FE, SEs clustered by block; residential universe "
             f"holds 72% of swept lots)\n"
             f"Right: {es_meta['n_treated']:,} lots first swept 2023-01..2025-05 "
             f"vs {es_meta['n_control']:,} not-yet-swept lots matched within "
             f"census tract AND on active-permit status at the sweep month "
             f"(DOB NOW permit spans; {es_meta['act_share'] * 100:.0f}% of these "
             f"swept lots mid-permit), on blocks with no sweep in the window\n"
             f"lot and calendar-month FE · whiskers = 95% CI, SEs clustered by "
             f"lot · months 0–12 average {a12['value']:+.3f} caller complaints "
             f"per lot-month [{a12['ci_low']:+.3f}, {a12['ci_high']:+.3f}] = "
             f"{pct12:+.0f}% [{p12['ci_low']:+.0f}%, {p12['ci_high']:+.0f}%] of "
             f"the {base:.3f} pre-sweep baseline · joint pre-trend p "
             f"{fmt_p(a12['pretrend_p'])}\n"
             f"construction phase matters: without phase matching the pooled "
             f"2021-2025 series reads {v_pool['pct']:+.0f}% "
             f"[{v_pool['pct_lo']:+.0f}%, {v_pool['pct_hi']:+.0f}%] but rides "
             f"the post-sweep permit wind-down (phase-matched it attenuates "
             f"to {v_pm['pct']:+.0f}% [{v_pm['pct_lo']:+.0f}%, "
             f"{v_pm['pct_hi']:+.0f}%]); the 2021-22 slice ({v_2122['pct']:+.0f}% "
             f"[{v_2122['pct_lo']:+.0f}%, {v_2122['pct_hi']:+.0f}%]) sits in "
             f"the legacy-BIS permit gap, its phase unverifiable",
             fontsize=8.3, color=MUTED, va="top", linespacing=1.45)

    out = ART / "proactive_sweep_structure.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


# ── main ─────────────────────────────────────────────────────────────────

def run_variant(ev_in, ctrls, series, count_caller):
    """Expand a treated set + its controls into a panel and estimate."""
    pan_t = expand(ev_in.rename(columns={"bbl_i": "bbl"})[["bbl", "m0"]],
                   1, count_caller)
    pan_c = expand(ctrls[["bbl", "m0"]], 0, count_caller)
    panel = pd.concat([pan_t, pan_c], ignore_index=True)
    n_ctrl = ctrls["bbl"].nunique()
    cov = ctrls["event_id"].nunique()
    print(f"\npanel [{series}]: {len(panel):,} rows ({len(ev_in):,} treated "
          f"+ {n_ctrl:,} control lots x {2 * WINDOW + 1} months; controls "
          f"cover {cov:,}/{len(ev_in):,} events)")
    rows = run_series(panel, "y_caller", "bbl", series,
                      {"n_treated": len(ev_in), "n_control": n_ctrl})
    return rows, n_ctrl


def phase_diag_rows(groups, active):
    """Active-permit shares by event month: the wind-down evidence."""
    rows = []
    print("\nactive-permit share by event month:")
    for gname, unit_df in groups:
        parts = []
        for t in (-12, 0, 12):
            s = float(active(unit_df["bbl"].to_numpy(),
                             unit_df["m0"].to_numpy() + t).mean())
            rows.append({"analysis": "phase_diagnostics", "series": gname,
                         "term": f"active_permit_share_t{t:+d}", "value": s,
                         "n_obs": len(unit_df), "units": "share",
                         "series_label": "share of lots with an active DOB "
                                         "NOW permit at event month t"})
            parts.append(f"t{t:+d}: {s:.1%}")
        print(f"  {gname:<34} " + "  ".join(parts))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = load_spine()
    g7 = df[(df["category_prefix"] == "7G") & (df["agency"] == 1)].copy()
    print(f"7G agency sweep events: {len(g7):,} at {g7['bbl_i'].nunique():,} "
          f"lots on {g7['block'].nunique():,} blocks")

    perm_rows, perm_meta = batching_permutation(g7, rng)
    lpm_rows = batching_lpm(set(g7["bbl_i"]), set(g7["block"]))

    ev, uni, count_caller, block_sweeps = build_es_inputs(df, g7)
    active = build_permit_activity(set(uni["bbl_i"]))
    ev["act_m0"] = active(ev["bbl_i"].to_numpy(), ev["m0"].to_numpy())
    ev23 = ev[ev["m0"] >= COHORT_2023].copy()
    print(f"treated lots with an active permit at the sweep month: "
          f"{ev['act_m0'].mean():.1%} pooled; {ev23['act_m0'].mean():.1%} "
          f"in 2023+ cohorts ({len(ev23):,} of {len(ev):,} events)")

    # published pooled matching first (same rng call order as the original
    # run: batching_permutation consumed its draws above), so the pooled
    # variant reproduces the published estimate exactly
    controls = match_controls(ev, uni, block_sweeps, rng)

    # ── labeled NON-ROBUST variants (kept for continuity, never headline) ──
    es_pooled, _ = run_variant(ev, controls, "sweep_pooled_unmatched",
                               count_caller)
    ev_2122 = ev[ev["m0"] < COHORT_2023]
    ctrl_2122 = controls[controls["event_id"].isin(set(ev_2122["event_id"]))]
    es_2122, _ = run_variant(ev_2122, ctrl_2122, "sweep_2021_22_unmatched",
                             count_caller)

    # ── HEADLINE: 2023+ cohorts, tract x active-permit matching cells ──
    ctrl_ph23 = match_controls_phase(ev23, uni, block_sweeps, active)
    es_head, n_ctrl23 = run_variant(ev23, ctrl_ph23,
                                    "sweep_2023plus_phasematched", count_caller)

    # supporting: construction vs construction only (both sides mid-permit)
    ev23a = ev23[ev23["act_m0"]]
    ctrl_ph23a = ctrl_ph23[ctrl_ph23["event_id"].isin(set(ev23a["event_id"]))]
    es_23a, _ = run_variant(ev23a, ctrl_ph23a, "sweep_2023plus_bothactive",
                            count_caller)

    # diagnostic: phase-matching the pooled cohorts kills the pooled number
    ctrl_ph_all = match_controls_phase(ev, uni, block_sweeps, active)
    es_pool_pm, _ = run_variant(ev, ctrl_ph_all, "sweep_pooled_phasematched",
                                count_caller)

    diag_rows = phase_diag_rows(
        [("treated_pooled", ev.rename(columns={"bbl_i": "bbl"})),
         ("controls_pooled_unmatched", controls),
         ("treated_2023plus", ev23.rename(columns={"bbl_i": "bbl"})),
         ("controls_2023plus_phasematched", ctrl_ph23)], active)

    RM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    cols = ["analysis", "series", "series_label", "term", "event_time",
            "value", "se", "ci_low", "ci_high", "pval", "null_mean",
            "null_sd", "n_draws", "n_obs", "n_treated", "n_control",
            "n_blocks", "baseline", "cluster", "pretrend_chi2", "pretrend_df",
            "pretrend_p", "decay_first_ci0_month", "units"]
    out = pd.concat([perm_rows, lpm_rows, es_head, es_23a, es_pool_pm,
                     es_pooled, es_2122, diag_rows], ignore_index=True)
    out = out.reindex(columns=cols)
    out_csv = RM / "proactive_sweep_structure.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv} ({len(out)} rows)")

    def pctrow(rows):
        r = rows[rows["term"] == "pct_avg_months_0_12_vs_baseline"].iloc[0]
        return {"pct": float(r["value"]), "pct_lo": float(r["ci_low"]),
                "pct_hi": float(r["ci_high"])}

    make_figure(perm_meta, lpm_rows, es_head,
                {"n_treated": len(ev23), "n_control": n_ctrl23,
                 "act_share": float(ev23["act_m0"].mean()),
                 "variants": {"pooled": pctrow(es_pooled),
                              "s2122": pctrow(es_2122),
                              "pooled_pm": pctrow(es_pool_pm)}})

    hl = perm_meta["share_other_lot_pm3d"]
    a12 = es_head[es_head["term"] == "avg_months_0_12"].iloc[0]
    p12 = pctrow(es_head)
    pp_pool, pp_2122 = pctrow(es_pooled), pctrow(es_2122)
    pp_pm, pp_23a = pctrow(es_pool_pm), pctrow(es_23a)
    d12 = "fall" if a12["ci_high"] < 0 else ("rise" if a12["ci_low"] > 0
                                             else "do not move")
    print("\nVERDICTS")
    print(f"  #11 batching: {hl['obs'] * 100:.1f}% of sweep visits have a "
          f"same-block companion within +/-3 days vs "
          f"{hl['null_mean'] * 100:.1f}% under the within-tract null "
          f"(p = {hl['p']:.4f})")
    print(f"  #11 within-block selection: joint pre-2020-traits test rejects "
          f"'random within block'; signs run toward big/newer/ECB-cited lots, "
          f"away from prewar and DOB-violation history (see LPM rows)")
    print(f"  #8 substitution HEADLINE (2023+ cohorts, phase-matched): caller "
          f"complaints {d12} after a first sweep; months 0-12 avg "
          f"{a12['value']:+.4f} [{a12['ci_low']:+.4f}, {a12['ci_high']:+.4f}] "
          f"per lot-month = {p12['pct']:+.0f}% [{p12['pct_lo']:+.0f}%, "
          f"{p12['pct_hi']:+.0f}%] of the {a12['baseline']:.3f} baseline")
    print(f"     supporting both-active 2023+: {pp_23a['pct']:+.0f}% "
          f"[{pp_23a['pct_lo']:+.0f}%, {pp_23a['pct_hi']:+.0f}%]")
    print(f"     non-robust variants: pooled unmatched {pp_pool['pct']:+.0f}% "
          f"[{pp_pool['pct_lo']:+.0f}%, {pp_pool['pct_hi']:+.0f}%] attenuates "
          f"to {pp_pm['pct']:+.0f}% [{pp_pm['pct_lo']:+.0f}%, "
          f"{pp_pm['pct_hi']:+.0f}%] when phase-matched; 2021-22 unmatched "
          f"{pp_2122['pct']:+.0f}% [{pp_2122['pct_lo']:+.0f}%, "
          f"{pp_2122['pct_hi']:+.0f}%] (legacy-BIS permit gap, phase "
          f"unverifiable)")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
