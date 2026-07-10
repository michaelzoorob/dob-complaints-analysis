#!/usr/bin/env python3
"""Critic script C (proactive post): adversarial attacks on the two event
studies quoted in post_proactive_substack.md.

Targets
  scripts/proactive_incident_eventstudy.py
      month-0 jump +0.137 (5x baseline), months 0-3 avg +0.0465 (+144%),
      block spillover "does not move at all", focal pre-trend ramp
  scripts/proactive_sweep_structure.py (substitution half)
      post-sweep caller complaints months 0-12 avg -0.0218 (-24%)

Attacks
  A1  Mechanical contamination: does excluding category 30 actually remove
      the mandated response? Decompose the month-0 spike by category /
      family / same-day-as-trigger; re-estimate with 1X (EWO), 2F
      (structural-monitoring enrollment), 7R (class-1 follow-up) removed,
      and with a +/-3-day trigger-window scrub at treated lots.
  A2  Pre-trend ramp: fit linear (and quadratic) counterfactuals from the
      pre coefficients alone, extrapolate to months 0..3, delta-method test
      of "month 0 is just the ramp's continuation"; recompute % effects
      against the month -1 level instead of the 12-month pre mean.
  A3  Stacked-design bugs: control-contamination recount, unit-reuse audit,
      dropping C(event_t) / cal_m FE (collinearity sensitivity), fully
      saturated event x event-time cell FE (Cengiz-style, kills cross-
      cohort contamination), per-cohort-year estimates + event-weighted
      average (Sun-Abraham-style heterogeneity check), not-yet-treated-only
      and never-treated-only control variants.
  A4  Block spillover power: what per-neighbor and block-aggregate effects
      does the months 0-3 CI actually rule out, scaled against the focal
      effect and block size; block-first-only control variant.
  A5  Substitution construction-phase confound: rebuild the sweep event
      study with control lots required to have an ACTIVE PERMIT at the
      match month (and a both-sides-active + 2023+ cohort variant); direct
      evidence table of active-permit shares by event month.

Read-only with respect to analysis outputs: prints everything, writes no
CSVs into risk_models/.

Run: /private/tmp/pyfix_venv/bin/python scripts/audit/critic_proactive_C_eventstudies.py
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import proactive_incident_eventstudy as pie  # noqa: E402
import proactive_sweep_structure as pss  # noqa: E402
import pyfixest as pf  # noqa: E402

RM = config.DATA_DIR / "analysis" / "risk_models"
T0 = time.time()


def log(msg=""):
    print(msg, flush=True)


def hdr(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ── shared estimation helper (mirrors the target scripts' formula) ────────

def fit_es(panel, ycol, cluster, extra_fe=None, drop_evt_main=False,
           drop_calm=False):
    """Return (model, names, beta, V, coef-by-event-time dict, helpers)."""
    rhs = f"i(event_t, treat, ref=-1)"
    if not drop_evt_main:
        rhs += " + C(event_t)"
    fe = []
    if not drop_calm:
        fe.append("cal_m")
    fe = ["bbl"] + fe + (extra_fe or [])
    fml = f"{ycol} ~ {rhs} | " + " + ".join(fe)
    m = pf.feols(fml, data=panel, vcov={"CRV1": cluster})
    names = [str(n) for n in m._coefnames]
    beta = m.coef().to_numpy()
    V = m._vcov

    idx = {}
    for i, n in enumerate(names):
        if n.startswith("event_t::") and n.endswith(":treat"):
            idx[int(n.split("::")[1].split(":")[0])] = i
    coefs = {k: beta[i] for k, i in idx.items()}

    def lincom(weights):  # {event_time: weight} -> (est, se)
        a = np.zeros(len(names))
        for k, w in weights.items():
            a[idx[k]] += w
        return float(a @ beta), float(np.sqrt(a @ V @ a))

    return m, names, beta, V, idx, coefs, lincom


def avg_post(lincom, lo, hi):
    ks = list(range(lo, hi + 1))
    return lincom({k: 1.0 / len(ks) for k in ks})


# ══════════════════════════════════════════════════════════════════════════
# PART I — incident event study (proactive_incident_eventstudy.py)
# ══════════════════════════════════════════════════════════════════════════

hdr("PART I: rebuild the incident event study exactly (same SEED, same order)")

rng = np.random.default_rng(pie.SEED)
df = pie.load_spine()
ev, uni, count_excl30, count_incl30, block_incidents, blk_first = pie.build_inputs(df)
controls = pie.match_controls(ev, uni, block_incidents, rng)
neighbors = pie.match_neighbors(ev, uni, blk_first, rng)

treated_units = ev.rename(columns={"bbl_i": "bbl"})[["bbl", "block", "m0"]]
pan_treated = pie.expand(treated_units, 1, count_excl30, count_incl30)
pan_controls = pie.expand(controls, 0, count_excl30, count_incl30)
pan_neighbors = pie.expand(neighbors, 1, count_excl30, count_incl30)
focal_panel = pd.concat([pan_treated, pan_controls], ignore_index=True)

pub = pd.read_csv(RM / "proactive_incident_estimates.csv")
pub_f = pub[(pub.series == "focal") & (pub.term == "event_month")]
pub_m0 = float(pub_f.loc[pub_f.event_time == 0, "coef"].iloc[0])
pub_avg = float(pub[(pub.series == "focal")
                    & (pub.term == "avg_months_0_3")]["coef"].iloc[0])
pub_base = float(pub_f["baseline_mean_treated"].iloc[0])

m_f, names_f, beta_f, V_f, idx_f, coefs_f, lincom_f = fit_es(
    focal_panel, "y_excl30", "bbl")
my_m0 = coefs_f[0]
my_avg, my_avg_se = avg_post(lincom_f, 0, 3)
base_pre12 = float(focal_panel.loc[(focal_panel.treat == 1)
                                   & (focal_panel.event_t < 0), "y_excl30"].mean())
log(f"\nreproduction: month-0 coef mine {my_m0:.6f} vs published {pub_m0:.6f} "
    f"(diff {abs(my_m0 - pub_m0):.2e})")
log(f"reproduction: months 0-3 avg mine {my_avg:.6f} vs published {pub_avg:.6f}")
log(f"reproduction: pre-12m treated baseline mine {base_pre12:.6f} vs "
    f"published {pub_base:.6f}")
REPRO_OK = abs(my_m0 - pub_m0) < 1e-4 and abs(my_avg - pub_avg) < 1e-4
log(f"reproduction {'EXACT' if REPRO_OK else 'FAILED — comparisons below are '
    'internally consistent but not byte-identical to the CSV'}")


# ── A1: composition of the month-0 spike ─────────────────────────────────

hdr("A1: what is the month-0 spike made of? (mandated-response machinery)")

# auxiliary spine copy with the date / unit / id columns pie drops
aux = pd.read_csv(pie.SPINE, usecols=["complaint_number", "received_date",
                                      "month", "category_prefix",
                                      "category_name", "agency", "bbl",
                                      "assigned_to"],
                  dtype={"category_prefix": "str", "bbl": "str",
                         "complaint_number": "str"})
aux = aux[aux["bbl"].notna() & aux["bbl"].str.fullmatch(r"\d{10}")].copy()
aux["bbl_i"] = aux["bbl"].astype("int64")
aux["m"] = ((aux["month"].str[:4].astype(int) - 2020) * 12
            + aux["month"].str[5:7].astype(int) - 1)

# trigger date per treated lot = first caller cat-30 date in month m0
c30x = aux[(aux["category_prefix"] == "30") & (aux["agency"] == 0)]
tr = c30x.merge(ev[["bbl_i", "m0"]], on="bbl_i")
tr0 = tr[tr["m"] == tr["m0"]]
trig = tr0.groupby("bbl_i")["received_date"].min().rename("trig_date")
trig_no = (tr0.assign(no=pd.to_numeric(tr0["complaint_number"],
                                       errors="coerce"))
           .groupby("bbl_i")["no"].min().rename("trig_no"))

ag = aux[(aux["agency"] == 1) & (aux["category_prefix"] != "30")].merge(
    ev[["bbl_i", "m0"]], on="bbl_i")
ag["et"] = ag["m"] - ag["m0"]
ag = ag.merge(trig, on="bbl_i", how="left")
ag["dd"] = (pd.to_datetime(ag["received_date"])
            - pd.to_datetime(ag["trig_date"])).dt.days

n_ev = len(ev)
win = ag[(ag.et >= -12) & (ag.et <= 12)]
m0ev = ag[ag.et == 0]
pre_rate_cat = ag[(ag.et >= -12) & (ag.et <= -1)].groupby(
    "category_prefix").size() / 12 / n_ev
m0_rate_cat = m0ev.groupby("category_prefix").size() / n_ev
excess = (m0_rate_cat - pre_rate_cat.reindex(m0_rate_cat.index).fillna(0))
excess = excess.sort_values(ascending=False)
tot_excess = float(excess.sum())
log(f"\ntreated lots {n_ev:,}; month-0 agency non-30 rate "
    f"{m0ev.shape[0] / n_ev:.4f}; pre rate {pre_rate_cat.sum():.4f}; "
    f"raw excess {tot_excess:.4f} (regression coef {my_m0:.4f})")
log("\nexcess agency complaints per treated lot at month 0, by category:")
namemap = (aux.groupby("category_prefix")["category_name"].first())
for cat, v in excess.head(12).items():
    log(f"  {cat:>3} {str(namemap.get(cat))[:52]:<52} {v:+.4f} "
        f"({v / tot_excess * 100:4.1f}% of excess)")

MECH_CATS = ["1X", "2F", "7R"]
mech_share = float(excess.reindex(MECH_CATS).fillna(0).sum() / tot_excess)
sameday1 = float((m0ev["dd"].abs() <= 1).mean())
sameday3 = float((m0ev["dd"].abs() <= 3).mean())
ert = float((m0ev["assigned_to"] == "EMERGENCY RESPONSE TEAM").mean())
log(f"\n1X+2F+7R share of month-0 excess: {mech_share:.1%}")
log(f"month-0 agency events within +/-1 day of the trigger: {sameday1:.1%}; "
    f"within +/-3 days: {sameday3:.1%}")
log(f"month-0 agency events assigned to EMERGENCY RESPONSE TEAM: {ert:.1%}")
# same-complaint-number family: numeric complaint numbers adjacent to trigger
m0no = (m0ev.assign(no=pd.to_numeric(m0ev["complaint_number"],
                                     errors="coerce"))
        .merge(trig_no, on="bbl_i", how="left"))
both = m0no["no"].notna() & m0no["trig_no"].notna()
adj = ((m0no["no"] - m0no["trig_no"]).abs() <= 3) & both
log(f"month-0 agency events with a complaint number within 3 of the "
    f"trigger's (same intake batch): {adj.sum() / max(both.sum(), 1):.1%} "
    f"of the {both.sum():,} numeric pairs")

# union definition: mechanical = 1X/2F/7R category OR within +/-3d of trigger
m0ev_mech = m0ev[(m0ev["category_prefix"].isin(MECH_CATS))
                 | (m0ev["dd"].abs() <= 3)]
union_share = len(m0ev_mech) / max(len(m0ev), 1)
log(f"month-0 events that are 1X/2F/7R OR within +/-3d of trigger: "
    f"{union_share:.1%} of all month-0 agency non-30 events")

# ── re-estimate with machinery removed ──
log("\nre-estimating with mandated machinery removed from the outcome ...")
aux_nomech = aux[(aux["agency"] == 1)
                 & (~aux["category_prefix"].isin(MECH_CATS))
                 & (aux["category_prefix"] != "30")]
count_nomech = pie.make_counter(aux_nomech)

# strict: additionally scrub ANY agency event within +/-3 days of the
# treated lot's trigger (same-visit companion paperwork, e.g. 73, 2L, 10)
d2 = aux[(aux["agency"] == 1) & (aux["category_prefix"] != "30")].merge(
    trig, left_on="bbl_i", right_index=True, how="left")
dd_all = (pd.to_datetime(d2["received_date"])
          - pd.to_datetime(d2["trig_date"])).dt.days
d_strict = d2[~(dd_all.abs() <= 3) | dd_all.isna()]
d_strict = d_strict[~d_strict["category_prefix"].isin(MECH_CATS)]
count_strict = pie.make_counter(d_strict)

fp2 = focal_panel.copy()
fp2["y_nomech"] = count_nomech(fp2["bbl"].to_numpy(), fp2["cal_m"].to_numpy())
fp2["y_strict"] = count_strict(fp2["bbl"].to_numpy(), fp2["cal_m"].to_numpy())

res_a1 = {}
for col, lab in [("y_nomech", "drop 1X/2F/7R categories (symmetric)"),
                 ("y_strict", "also scrub +/-3d trigger window at treated")]:
    _, _, _, _, _, cfs, lc = fit_es(fp2, col, "bbl")
    a, se = avg_post(lc, 0, 3)
    b = float(fp2.loc[(fp2.treat == 1) & (fp2.event_t < 0), col].mean())
    res_a1[col] = (cfs[0], a, se, b)
    log(f"  [{lab}]")
    log(f"    month-0 coef {cfs[0]:+.4f} (orig {my_m0:+.4f}, "
        f"{(my_m0 - cfs[0]) / my_m0 * 100:.0f}% of the jump removed)")
    log(f"    months 0-3 avg {a:+.4f} [se {se:.4f}] on pre-mean {b:.4f} "
        f"-> {a / b * 100:+.0f}% (orig +{my_avg / base_pre12 * 100:.0f}%)")


# ── A2: is month 0 / months 0-3 just the ramp's continuation? ────────────

hdr("A2: ramp counterfactual (linear + quadratic extrapolation, exact vcov)")

pre_ts = np.array(sorted(k for k in coefs_f if k < -1))          # -12..-2
pre_ix = np.array([idx_f[k] for k in pre_ts])
pre_c = np.array([coefs_f[k] for k in pre_ts])


def ramp_test(ts_fit, cs_fit, include_ref, deg, targets, label):
    """OLS of pre coefs on event time; delta-method test of post - fit."""
    t_all = np.append(ts_fit, -1.0) if include_ref else ts_fit.astype(float)
    X = np.vander(t_all, deg + 1, increasing=True)               # 1, t, t^2
    H = np.linalg.pinv(X.T @ X) @ X.T                            # (deg+1, n)
    # prediction at target t*: p = x*' H y, where y = (pre coefs, [0])
    rows = []
    for t_star in targets:
        x_star = np.array([t_star ** j for j in range(deg + 1)])
        w_full = x_star @ H                                      # weight per point
        w_pre = w_full[: len(ts_fit)]                            # ref point y=0
        pred = float(w_pre @ cs_fit)
        # contrast: coef(t*) - pred, var via full V
        a = np.zeros(len(names_f))
        a[idx_f[t_star]] = 1.0
        for wj, ij in zip(w_pre, pre_ix):
            a[ij] -= wj
        est = float(a @ beta_f)
        se = float(np.sqrt(a @ V_f @ a))
        rows.append((t_star, pred, est, se))
    # months 0-3 average net of ramp
    a = np.zeros(len(names_f))
    for t_star in targets:
        x_star = np.array([t_star ** j for j in range(deg + 1)])
        w_pre = (x_star @ H)[: len(ts_fit)]
        a[idx_f[t_star]] += 1.0 / len(targets)
        for wj, ij in zip(w_pre, pre_ix):
            a[ij] -= wj / len(targets)
    avg_net = float(a @ beta_f)
    avg_net_se = float(np.sqrt(a @ V_f @ a))
    log(f"\n[{label}]")
    for t_star, pred, est, se in rows:
        log(f"  t={t_star:+d}: ramp continuation {pred:+.4f}; "
            f"coef - ramp = {est:+.4f} (se {se:.4f}, z {est / se:+.1f})")
    log(f"  months 0-3 avg net of ramp: {avg_net:+.4f} (se {avg_net_se:.4f}) "
        f"vs raw {my_avg:+.4f} -> ramp explains "
        f"{(my_avg - avg_net) / my_avg * 100:+.0f}% of the quarter effect")
    return avg_net, avg_net_se


net_a, se_a = ramp_test(pre_ts, pre_c, False, 1, [0, 1, 2, 3],
                        "linear fit on t=-12..-2 (excl. reference)")
net_b, se_b = ramp_test(pre_ts, pre_c, True, 1, [0, 1, 2, 3],
                        "linear fit on t=-12..-1 (incl. reference point 0)")
net_c, se_c = ramp_test(pre_ts, pre_c, False, 2, [0, 1, 2, 3],
                        "quadratic fit on t=-12..-2")

# denominator critique: % effects are quoted against the 12-month pre mean,
# but the coefficients are measured against the month -1 level
tr_rows = focal_panel[focal_panel.treat == 1]
lvl_m1 = float(tr_rows.loc[tr_rows.event_t == -1, "y_excl30"].mean())
lvl_last3 = float(tr_rows.loc[tr_rows.event_t.between(-3, -1), "y_excl30"].mean())
ct_rows = focal_panel[focal_panel.treat == 0]
ct_m1 = float(ct_rows.loc[ct_rows.event_t == -1, "y_excl30"].mean())
ct_03 = float(ct_rows.loc[ct_rows.event_t.between(0, 3), "y_excl30"].mean())
cf_lvl_03 = lvl_m1 + (ct_03 - ct_m1)
log(f"\ntreated level at t=-1: {lvl_m1:.4f}; months -3..-1: {lvl_last3:.4f}; "
    f"12-month pre mean used in the post: {base_pre12:.4f}")
log(f"control drift t=-1 -> 0..3: {ct_03 - ct_m1:+.4f}; counterfactual level "
    f"for months 0-3: {cf_lvl_03:.4f}")
log(f"month-0 multiple: published {pub_m0 / pub_base + 1:.1f}x the pre-12m "
    f"mean; vs {my_m0 / lvl_m1 + 1:.1f}x the month -1 level")
log(f"months 0-3: published +{pub_avg / pub_base * 100:.0f}% of pre-12m mean; "
    f"vs +{my_avg / cf_lvl_03 * 100:.0f}% of the month -1 counterfactual level")
log(f"combined (strict outcome, month -1 baseline): "
    f"{res_a1['y_strict'][1] / (res_a1['y_strict'][3] + lvl_m1 - base_pre12) * 100:+.0f}% "
    f"over months 0-3")


# ── A3: stacked-design bugs ──────────────────────────────────────────────

hdr("A3: stacked-design audit")

# (a) contamination recounts
c30 = df[(df["category_prefix"] == "30") & (df["agency"] == 0)]
ctrl_set = controls[["bbl", "m0"]].copy()
c30_keys = np.sort(c30["bbl_i"].to_numpy() * pie.STRIDE + c30["m"].to_numpy())


def window_count(keys, ids, lo, hi):
    return (np.searchsorted(keys, ids * pie.STRIDE + hi, side="right")
            - np.searchsorted(keys, ids * pie.STRIDE + lo, side="left"))


n_caller30 = window_count(c30_keys, ctrl_set["bbl"].to_numpy(),
                          ctrl_set["m0"].to_numpy() - 12,
                          ctrl_set["m0"].to_numpy() + 12)
a30 = df[(df["category_prefix"] == "30") & (df["agency"] == 1)]
a30_keys = np.sort(a30["bbl_i"].to_numpy() * pie.STRIDE + a30["m"].to_numpy())
n_agency30 = window_count(a30_keys, ctrl_set["bbl"].to_numpy(),
                          ctrl_set["m0"].to_numpy() - 12,
                          ctrl_set["m0"].to_numpy() + 12)
dup_ctrl = controls["bbl"].duplicated().sum()
overlap = len(set(controls["bbl"]) & set(treated_units["bbl"]))
dup_rows = focal_panel.duplicated(["bbl", "cal_m"]).sum()
log(f"controls with a caller cat-30 inside their +/-12m window: "
    f"{(n_caller30 > 0).sum()} of {len(ctrl_set):,} (design says 0)")
log(f"controls with an AGENCY cat-30 inside the window (not filtered by "
    f"design): {(n_agency30 > 0).sum():,} ({(n_agency30 > 0).mean():.2%}) — "
    f"outcome excludes cat-30 itself but not its 1X/2F paperwork; direction "
    f"= attenuation of the focal contrast")
log(f"control lots used in >1 event: {dup_ctrl}; treated lots also serving "
    f"as controls: {overlap}; duplicated (bbl, cal_m) rows in the stacked "
    f"panel: {dup_rows:,}")

# (b) FE-structure sensitivity: drop C(event_t) main / drop cal_m FE
for kwargs, lab in [({"drop_evt_main": True}, "drop C(event_t) mains"),
                    ({"drop_calm": True}, "drop cal_m FE")]:
    _, _, _, _, _, cfs, lc = fit_es(focal_panel, "y_excl30", "bbl", **kwargs)
    a, se = avg_post(lc, 0, 3)
    log(f"  [{lab:<22}] month-0 {cfs[0]:+.4f}; months 0-3 {a:+.4f} "
        f"(se {se:.4f})   [orig {my_m0:+.4f} / {my_avg:+.4f}]")

# (c) fully saturated stacked estimator: event_id x event_t cell FE
# (re-expand with event_id attached per ROLE — a lot can be a control for
# an early event and treated in a later one, so a bbl->event map is unsafe)
W25 = 2 * pie.WINDOW + 1


def expand_eid(units, treat):
    et = np.tile(np.arange(-pie.WINDOW, pie.WINDOW + 1), len(units))
    out = pd.DataFrame({
        "bbl": np.repeat(units["bbl"].to_numpy(), W25),
        "event_id": np.repeat(units["event_id"].to_numpy(), W25),
        "event_t": et,
        "cal_m": np.repeat(units["m0"].to_numpy(), W25) + et,
        "treat": np.int8(treat),
    })
    out["y_excl30"] = count_excl30(out["bbl"].to_numpy(),
                                   out["cal_m"].to_numpy())
    return out


fp3 = pd.concat([
    expand_eid(ev.rename(columns={"bbl_i": "bbl"})[["bbl", "m0", "event_id"]], 1),
    expand_eid(controls, 0)], ignore_index=True)
fp3["evt_cell"] = (fp3["event_id"].astype(np.int64) * 32
                   + (fp3["event_t"] + pie.WINDOW))
_, _, _, _, _, cfs_sat, lc_sat = fit_es(
    fp3, "y_excl30", "bbl", extra_fe=["evt_cell"], drop_evt_main=True,
    drop_calm=True)
a_sat, se_sat = avg_post(lc_sat, 0, 3)
log(f"  [saturated event x event-time cell FE (Cengiz)] month-0 "
    f"{cfs_sat[0]:+.4f}; months 0-3 {a_sat:+.4f} (se {se_sat:.4f})")

# (d) per-cohort-year estimates, event-weighted (Sun-Abraham-style)
fp3["cohort_m0"] = fp3["cal_m"] - fp3["event_t"]
fp3["cohort_yr"] = 2020 + fp3["cohort_m0"] // 12
rows = []
for yr, g in fp3.groupby("cohort_yr"):
    n_tr = g.loc[g.treat == 1, "bbl"].nunique()
    try:
        _, _, _, _, _, cfs_y, lc_y = fit_es(g, "y_excl30", "bbl")
        a_y, se_y = avg_post(lc_y, 0, 3)
        rows.append((yr, n_tr, cfs_y[0], a_y, se_y))
        log(f"  cohort {yr}: n={n_tr:,} treated; month-0 {cfs_y[0]:+.4f}; "
            f"months 0-3 {a_y:+.4f} (se {se_y:.4f})")
    except Exception as e:  # tiny cohorts
        log(f"  cohort {yr}: skipped ({e})")
wts = np.array([r[1] for r in rows], dtype=float)
wts /= wts.sum()
wavg_m0 = float(np.sum(wts * np.array([r[2] for r in rows])))
wavg_03 = float(np.sum(wts * np.array([r[3] for r in rows])))
log(f"  event-weighted cohort average: month-0 {wavg_m0:+.4f}; months 0-3 "
    f"{wavg_03:+.4f}   [pooled {my_m0:+.4f} / {my_avg:+.4f}]")

# (e) control-pool variants: not-yet-treated-only and never-treated-only


def match_controls_variant(mode):
    r2 = np.random.default_rng(pie.SEED + 1)
    pool = uni.rename(columns={"bbl_i": "cbbl", "block": "cblock"})
    cand = ev[["event_id", "bbl_i", "m0", "tract"]].merge(
        pool[["cbbl", "cblock", "tract", "treat_m"]], on="tract")
    cand = cand[cand["cbbl"] != cand["bbl_i"]]
    if mode == "nyt":      # eventually treated, >24m later
        ok = cand["treat_m"].notna() & (cand["treat_m"] > cand["m0"] + pie.CLEAR)
    else:                  # never treated
        ok = cand["treat_m"].isna()
    cand = cand[ok]
    n_inc = block_incidents(cand["cblock"].to_numpy(), cand["m0"].to_numpy())
    cand = cand[n_inc == 0]
    cand = cand.sample(frac=1.0, random_state=r2).drop_duplicates("cbbl")
    matched = cand.groupby("event_id").head(pie.K_CONTROLS)
    return matched[["event_id", "cbbl", "cblock", "m0"]].rename(
        columns={"cbbl": "bbl", "cblock": "block"})


for mode, lab in [("nyt", "not-yet-treated-only controls"),
                  ("never", "never-treated-only controls")]:
    mc = match_controls_variant(mode)
    pc = pie.expand(mc, 0, count_excl30, count_incl30)
    pnl = pd.concat([pan_treated, pc], ignore_index=True)
    _, _, _, _, _, cfs_v, lc_v = fit_es(pnl, "y_excl30", "bbl")
    a_v, se_v = avg_post(lc_v, 0, 3)
    n_ev_cov = mc["event_id"].nunique()
    log(f"  [{lab:<31}] {mc['bbl'].nunique():,} controls covering "
        f"{n_ev_cov:,}/{len(ev):,} events; month-0 {cfs_v[0]:+.4f}; "
        f"months 0-3 {a_v:+.4f} (se {se_v:.4f})")


# ── A4: block-spillover power ────────────────────────────────────────────

hdr("A4: what can the block-spillover CI actually rule out?")

sp = pub[(pub.series == "block_spillover")]
sp_avg = sp[sp.term == "avg_months_0_3"].iloc[0]
sp_base = float(sp["baseline_mean_treated"].iloc[0])
est, se, lo, hi = (float(sp_avg["coef"]), float(sp_avg["se"]),
                   float(sp_avg["ci_low"]), float(sp_avg["ci_high"]))
log(f"published months 0-3 spillover: {est:+.4f} [{lo:+.4f}, {hi:+.4f}] on "
    f"baseline {sp_base:.4f}")
log(f"in % of the NEIGHBOR baseline: {est / sp_base * 100:+.0f}% "
    f"[{lo / sp_base * 100:+.0f}%, {hi / sp_base * 100:+.0f}%] — the CI "
    f"allows up to a {hi / sp_base * 100:+.0f}% rise in neighbor rates")
log(f"per-neighbor effect as share of the focal per-lot effect "
    f"({my_avg:+.4f}): CI [{lo / my_avg * 100:+.1f}%, "
    f"{hi / my_avg * 100:+.1f}%] — rules out per-lot spillover above "
    f"{hi / my_avg * 100:.1f}% of focal")

# block sizes: FULL spine-lot count on block-first treated blocks
ev_bf = ev[ev["m0"] == ev["block"].map(blk_first)]
blk_sizes = uni.groupby("block")["bbl_i"].size()
nb_full = (ev_bf["block"].map(blk_sizes) - 1).clip(lower=0)
log(f"\nblock-first events {len(ev_bf):,}; spine lots per treated block "
    f"(excl. focal): mean {nb_full.mean():.1f}, median {nb_full.median():.0f}"
    f"; sampled neighbors capped at {pie.MAX_NEIGHBORS}")
agg_hi = hi * nb_full.mean()
log(f"block-AGGREGATE spillover the CI allows: up to {agg_hi:+.4f} added "
    f"agency complaints per block-month = {agg_hi / my_avg * 100:.0f}% of "
    f"the focal months 0-3 effect — 'the block in total' could still absorb "
    f"a non-trivial share")
log(f"MDE (1.96 x se) per neighbor: {1.96 * se:.4f} = "
    f"{1.96 * se / sp_base * 100:.0f}% of neighbor baseline")

# variant: spillover controls restricted to block-first events' controls
keep = ~pan_controls["bbl"].isin(set(neighbors["bbl"]))
ctrl_bf = controls.merge(ev_bf[["event_id"]], on="event_id")
pc_bf = pan_controls[keep & pan_controls["bbl"].isin(set(ctrl_bf["bbl"]))]
spill_panel2 = pd.concat([pan_neighbors, pc_bf], ignore_index=True)
_, _, _, _, _, cfs_sp, lc_sp = fit_es(spill_panel2, "y_excl30", "block")
a_sp, se_sp = avg_post(lc_sp, 0, 3)
log(f"variant, controls from block-first events only: months 0-3 "
    f"{a_sp:+.4f} (se {se_sp:.4f})   [published {est:+.4f}]")


# ══════════════════════════════════════════════════════════════════════════
# PART II — sweep substitution (proactive_sweep_structure.py part 2)
# ══════════════════════════════════════════════════════════════════════════

hdr("PART II / A5: sweep substitution — construction-phase confound")

df2 = pss.load_spine()
g7 = df2[(df2["category_prefix"] == "7G") & (df2["agency"] == 1)].copy()

# reproduce the rng state: batching_permutation consumes one rng.random(n)
# per draw before match_controls runs in pss.main
rng2 = np.random.default_rng(pss.SEED)
n_tracted = int(g7["bct2020"].notna().sum())
for _ in range(pss.N_DRAWS):
    rng2.random(n_tracted)

ev2, uni2, count_caller, block_sweeps = pss.build_es_inputs(df2, g7)
controls2 = pss.match_controls(ev2, uni2, block_sweeps, rng2)
pan_tr2 = pss.expand(ev2.rename(columns={"bbl_i": "bbl"})[["bbl", "m0"]],
                     1, count_caller)
pan_ct2 = pss.expand(controls2, 0, count_caller)
panel2 = pd.concat([pan_tr2, pan_ct2], ignore_index=True)

ov2 = len(set(controls2["bbl"]) & set(ev2["bbl_i"]))
dup2 = panel2.duplicated(["bbl", "cal_m"]).sum()
log(f"sweep panel: treated lots also serving as controls: {ov2}; "
    f"duplicated (bbl, cal_m) rows: {dup2:,}")

pub2 = pd.read_csv(RM / "proactive_sweep_structure.csv")
pub2_12 = pub2[(pub2.analysis == "substitution_eventstudy")
               & (pub2.term == "avg_months_0_12")].iloc[0]
_, _, _, _, _, cfs2, lc2 = fit_es(panel2, "y_caller", "bbl")
a12, a12se = avg_post(lc2, 0, 12)
b2 = float(panel2.loc[(panel2.treat == 1) & (panel2.event_t < 0),
                      "y_caller"].mean())
log(f"\nreproduction: months 0-12 avg mine {a12:+.6f} vs published "
    f"{float(pub2_12['value']):+.6f} (diff {abs(a12 - float(pub2_12['value'])):.2e}); "
    f"baseline {b2:.4f} -> {a12 / b2 * 100:+.0f}%")

# ── active-permit lot-months from the permits table ──
log("\nbuilding lot-month active-permit spans from permits (DOB NOW) ...")
conn = sqlite3.connect(str(ROOT / "data" / "dob_complaints.db"))
perm = pd.read_sql_query(
    "SELECT bbl, issued_date, expired_date FROM permits "
    "WHERE bbl IS NOT NULL AND issued_date IS NOT NULL "
    "AND expired_date IS NOT NULL", conn)
conn.close()
perm["bbl_i"] = pd.to_numeric(perm["bbl"], errors="coerce")
perm = perm[perm["bbl_i"].notna() & (perm["bbl_i"] >= 1e9)]
perm["bbl_i"] = perm["bbl_i"].astype("int64")
iss = pd.to_datetime(perm["issued_date"], errors="coerce")
exp = pd.to_datetime(perm["expired_date"], errors="coerce")
perm = perm[iss.notna() & exp.notna() & (exp >= iss)].copy()
perm["sm"] = ((iss.dt.year - 2020) * 12 + iss.dt.month - 1).clip(lower=0)
perm["em"] = ((exp.dt.year - 2020) * 12 + exp.dt.month - 1).clip(upper=76)
perm = perm[(perm["sm"] <= perm["em"]) & (perm["em"] >= 0) & (perm["sm"] <= 76)]
perm = perm[perm["bbl_i"].isin(set(uni2["bbl_i"]))]
lens = (perm["em"] - perm["sm"] + 1).to_numpy()
bbl_rep = np.repeat(perm["bbl_i"].to_numpy(), lens)
m_rep = np.concatenate([np.arange(s, e + 1) for s, e in
                        zip(perm["sm"].to_numpy(), perm["em"].to_numpy())])
act_keys = np.sort(bbl_rep * pie.STRIDE + m_rep)
log(f"permits at spine lots: {len(perm):,} spans -> {len(act_keys):,} "
    f"active lot-month keys ({pd.Series(bbl_rep).nunique():,} lots)")


def active(bbl_arr, m_arr):
    q = bbl_arr * pie.STRIDE + m_arr
    return (np.searchsorted(act_keys, q, side="right")
            > np.searchsorted(act_keys, q, side="left"))


tr_act = active(ev2["bbl_i"].to_numpy(), ev2["m0"].to_numpy())
log(f"treated lots with an active permit at the sweep month: "
    f"{tr_act.mean():.1%} (Wave-1 BIN-based benchmark for 7G: 69.6%)")
ct_act = active(controls2["bbl"].to_numpy(), controls2["m0"].to_numpy())
log(f"ORIGINAL controls with an active permit at m0: {ct_act.mean():.1%} — "
    f"the phase mismatch under attack")

# active-permit share by event month (winding-down evidence)
log("\nactive-permit share by event month (treated vs original controls):")
for lab, unit_df in [("treated", ev2.rename(columns={"bbl_i": "bbl"})),
                     ("orig controls", controls2)]:
    shares = []
    for et in [-12, -6, 0, 6, 12]:
        s = active(unit_df["bbl"].to_numpy(), unit_df["m0"].to_numpy() + et)
        shares.append(f"t{et:+d}: {s.mean():.1%}")
    log(f"  {lab:<14} " + "  ".join(shares))


def match_controls_permit(ev_in, require_active=True):
    r3 = np.random.default_rng(pss.SEED + 7)
    pool = uni2.rename(columns={"bbl_i": "cbbl", "block": "cblock"})
    cand = ev_in[["event_id", "bbl_i", "m0", "tract"]].merge(
        pool[["cbbl", "cblock", "tract", "treat_m"]], on="tract")
    cand = cand[cand["cbbl"] != cand["bbl_i"]]
    ok = cand["treat_m"].isna() | (cand["treat_m"] > cand["m0"] + pss.CLEAR)
    cand = cand[ok]
    n_sw = block_sweeps(cand["cblock"].to_numpy(), cand["m0"].to_numpy())
    cand = cand[n_sw == 0]
    if require_active:
        cand = cand[active(cand["cbbl"].to_numpy(), cand["m0"].to_numpy())]
    cand = cand.sample(frac=1.0, random_state=r3).drop_duplicates("cbbl")
    matched = cand.groupby("event_id").head(pss.K_CONTROLS)
    return matched[["event_id", "cbbl", "m0"]].rename(columns={"cbbl": "bbl"})


def run_sweep_variant(ev_in, ctrls, lab):
    pt = pss.expand(ev_in.rename(columns={"bbl_i": "bbl"})[["bbl", "m0"]],
                    1, count_caller)
    pc = pss.expand(ctrls, 0, count_caller)
    pnl = pd.concat([pt, pc], ignore_index=True)
    m_v, nm, _, _, idx_v, cfs_v, lc_v = fit_es(pnl, "y_caller", "bbl")
    a_v, se_v = avg_post(lc_v, 0, 12)
    a3_v, se3_v = avg_post(lc_v, 0, 3)
    b_v = float(pnl.loc[(pnl.treat == 1) & (pnl.event_t < 0), "y_caller"].mean())
    # pre-trend joint test
    pre_i = [i for k, i in idx_v.items() if k < -1]
    R = np.zeros((len(pre_i), len(nm)))
    for j, i in enumerate(pre_i):
        R[j, i] = 1.0
    w = m_v.wald_test(R=R)
    log(f"  [{lab}]")
    log(f"    {ev_in.shape[0]:,} treated / {ctrls['bbl'].nunique():,} controls "
        f"(mean {len(ctrls) / max(ctrls['event_id'].nunique(), 1):.2f} per "
        f"event; {ctrls['event_id'].nunique():,}/{ev_in.shape[0]:,} events "
        f"covered)")
    log(f"    months 0-12 avg {a_v:+.4f} [se {se_v:.4f}, "
        f"{a_v - 1.96 * se_v:+.4f}, {a_v + 1.96 * se_v:+.4f}] on baseline "
        f"{b_v:.4f} -> {a_v / b_v * 100:+.0f}%  (published -24%)")
    log(f"    months 0-3 avg {a3_v:+.4f} (se {se3_v:.4f}); pre-trend p "
        f"{float(w['pvalue']):.3f}")
    return a_v, se_v, b_v


log("\nre-matched variants:")
ctrl_act_all = match_controls_permit(ev2, require_active=True)
run_sweep_variant(ev2, ctrl_act_all,
                  "controls REQUIRED active-permit at m0 (all treated)")

ev2_act = ev2[tr_act].copy()
ctrl_act_both = match_controls_permit(ev2_act, require_active=True)
res_both = run_sweep_variant(
    ev2_act, ctrl_act_both,
    "both sides active-permit at m0 (construction vs construction)")

ev2_act23 = ev2_act[ev2_act["m0"] >= 36].copy()
ctrl_act_23 = match_controls_permit(ev2_act23, require_active=True)
run_sweep_variant(ev2_act23, ctrl_act_23,
                  "both sides active + cohorts 2023-01 onward (permit "
                  "coverage clean)")

# phase table for the matched variant
log("\nactive-permit share by event month, both-sides-active variant:")
for lab, unit_df in [("treated", ev2_act.rename(columns={"bbl_i": "bbl"})),
                     ("matched controls", ctrl_act_both)]:
    shares = []
    for et in [-12, -6, 0, 6, 12]:
        s = active(unit_df["bbl"].to_numpy(), unit_df["m0"].to_numpy() + et)
        shares.append(f"t{et:+d}: {s.mean():.1%}")
    log(f"  {lab:<17} " + "  ".join(shares))

# ── A5b: how much decline does winding-down mechanically explain? ────────

hdr("A5b: mechanical winding-down arithmetic")

# within-lot gradient of caller complaints on active-permit status, fit on
# CONTROL lot-months only (no sweep in their windows)
pc_act = pan_ct2.copy()
pc_act["act"] = active(pc_act["bbl"].to_numpy(),
                       pc_act["cal_m"].to_numpy()).astype(float)
m_grad = pf.feols("y_caller ~ act | bbl + cal_m", data=pc_act,
                  vcov={"CRV1": "bbl"})
b_act = float(m_grad.coef().iloc[0])
se_act = float(m_grad.se().iloc[0])
log(f"within-lot caller-complaint gradient on active-permit status "
    f"(control lots, bbl+cal_m FE): {b_act:+.4f} per lot-month "
    f"(se {se_act:.4f})")

# active-share paths by event time, treated vs original controls
tr_units = ev2.rename(columns={"bbl_i": "bbl"})[["bbl", "m0"]]
ets = np.arange(-pss.WINDOW, pss.WINDOW + 1)
sT = np.array([active(tr_units["bbl"].to_numpy(),
                      tr_units["m0"].to_numpy() + t).mean() for t in ets])
sC = np.array([active(controls2["bbl"].to_numpy(),
                      controls2["m0"].to_numpy() + t).mean() for t in ets])
i_m1 = list(ets).index(-1)
did_act = (sT - sT[i_m1]) - (sC - sC[i_m1])
post = (ets >= 0) & (ets <= 12)
implied = float(b_act * did_act[post].mean())
log(f"treated active share t-1 -> t+12: {sT[i_m1]:.1%} -> {sT[-1]:.1%}; "
    f"controls {sC[i_m1]:.1%} -> {sC[-1]:.1%}")
log(f"months 0-12 avg DiD in active share: {did_act[post].mean():+.3f}")
log(f"implied mechanical effect on caller complaints: {implied:+.4f} per "
    f"lot-month = {implied / a12 * 100:.0f}% of the estimated {a12:+.4f} "
    f"(published -24%)")

# same-cohort comparisons: unmatched vs phase-matched inside each era —
# isolates the confound share where permit coverage is clean (2023+) and
# shows what drives the full-sample collapse (2021-22, legacy-BIS gap)
ev2_23 = ev2[ev2["m0"] >= 36].copy()
ctrl_23_noact = match_controls_permit(ev2_23, require_active=False)
run_sweep_variant(ev2_23, ctrl_23_noact,
                  "cohorts 2023+ / controls NOT permit-matched (baseline "
                  "for the 2023+ matched variant)")

ev2_pre = ev2[ev2["m0"] < 36].copy()
ctrl_pre_noact = match_controls_permit(ev2_pre, require_active=False)
run_sweep_variant(ev2_pre, ctrl_pre_noact,
                  "cohorts 2021-22 / controls NOT permit-matched")

ev2_act_pre = ev2_act[ev2_act["m0"] < 36].copy()
ctrl_act_pre = match_controls_permit(ev2_act_pre, require_active=True)
run_sweep_variant(ev2_act_pre, ctrl_act_pre,
                  "cohorts 2021-22 / both sides active (permit ledger "
                  "INCOMPLETE here — legacy BIS gap)")

log(f"\ntotal runtime {time.time() - T0:.0f}s")
log("\nDone. Findings are synthesized in the critic report.")
