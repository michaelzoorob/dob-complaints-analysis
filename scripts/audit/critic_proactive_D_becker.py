"""Critic script D (proactive post): attack the Becker-margin reallocation
claim (+459 violations per 1,000 inspections; 98.8 predicted / 98.5 realized
for never-swept top scorers), the flat yield curve (-0.44pp/decile), and the
map (Spearman +0.81; 93.2% of top-decile-risk lots never swept).

Attack surfaces:
  F1  counterfactual accounting: predicted-vs-realized basis, subgroup
      calibration of the logit on the swept subgroup, sweep-caused outcome
      contamination bound, per-LOT vs per-INSPECTION units.
  F2  outcome circularity: the 2020-26 any-violation outcome bundles
      periodic/statutory streams (boiler/elevator/facade/energy/gas) that
      fire without any sweep; recompute with sweep-findable outcomes
      (any ECB; ECB excl. device types; union construction family) and
      refit the logit on the sweep-findable outcome.
  F3  per-visit translation: what do inspections actually find, per visit,
      at high-score lots (7G and all-discretionary yields by risk decile,
      and directly at the top-unswept set).
  F4  framing: 93.2%-never-swept vs the 98.55% base rate (sweep enrichment
      of the top decile), "no sweep ever" vs the 2019 7G cohort invisible
      to the 2020+ spine, group composition.
  F5  leakage: feature vintages (ACS 2023 5yr, current PLUTO), split-half
      AUC/calibration, drop-tract-poverty sensitivity, and an out-of-time
      model (2000-09 history -> 2010-19 outcome) whose top picks are then
      checked against realized 2020-26 rates.
  F6  flat decile curve robustness: alternative intensity denominator
      (all PLUTO lots, not residential panel lots), small-tract exclusion,
      dropping decile 10, building-level controls, program decomposition,
      no-access gradient.
  F7  map: Spearman sensitivity to MIN_LOTS, denominator mechanics,
      risk-stock-vs-unswept-rate near-equivalence.

Read-only against the pipeline outputs; writes one evidence CSV:
  data/analysis/risk_models/critic_proactive_D_becker.csv

Run: /private/tmp/pyfix_venv/bin/python scripts/audit/critic_proactive_D_becker.py
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

ROOT = Path("/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import config
import dob_ledger
import proactive_becker_margin as becker
from analysis_config import make_bbl

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
MAPCSV = config.DATA_DIR / "analysis" / "risk_models" / "proactive_map_tracts.csv"
OUTCSV = config.DATA_DIR / "analysis" / "risk_models" / "critic_proactive_D_becker.csv"

PERIODIC_FAMS = {"boiler", "elev", "facade", "energy", "gas_plumb"}
ECB_DEVICE_TYPES = {"Elevators", "Boilers"}  # periodic device streams inside ECB

T0 = time.time()
ROWS = []  # evidence rows for the CSV


def log(term, value, note=""):
    ROWS.append({"finding": CURRENT, "term": term, "value": value, "note": note})


def stamp(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def rank_auc(p, y):
    r = pd.Series(p).rank().to_numpy()
    y = np.asarray(y, float)
    n1, n = y.sum(), len(y)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (n - n1)))


def build_X(df, hist_cols=("log1p_ecb_hist", "log1p_dobviol_hist", "any_prior_viol")):
    """Replicates becker.load_panel_scored()'s design matrix exactly, with a
    swappable history block (for the out-of-time variant)."""
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


def fit_logit(X, y):
    m = sm.GLM(np.asarray(y, float), X, family=sm.families.Binomial()).fit()
    return np.asarray(m.predict(X)), m


# ══════════════════════════════════════════════════════════════════════════
stamp("[F0] reproduce becker part (b): score panel, flag swept lots")
CURRENT = "F0"

df, auc, k = becker.load_panel_scored()
ev7g = pd.read_csv(SPINE, usecols=["category_prefix", "agency", "bbl",
                                   "outcome", "ecb_number", "active_permit"],
                   dtype={"bbl": str, "ecb_number": str})
ev7g = ev7g[(ev7g["category_prefix"] == "7G") & (ev7g["agency"] == 1)]
swept_bbls = set(ev7g["bbl"].dropna().astype(str)) - {""}
df["swept7g"] = df["bbl_key"].isin(swept_bbls).astype(int)
n_swept = int(df["swept7g"].sum())
unswept = df[df["swept7g"] == 0]
top = unswept.nlargest(n_swept, "p_hat")
swept = df[df["swept7g"] == 1]
top_idx = set(top.index)

log("auc_reproduced", auc)
log("n_swept_panel_lots", n_swept)
log("mean_pred_swept", swept["p100"].mean())
log("mean_pred_top_unswept", top["p100"].mean())
assert abs(auc - 0.880) < 0.005, "failed to reproduce AUC"
assert abs(swept["p100"].mean() - 52.9) < 0.3, "failed to reproduce swept mean"
stamp(f"  reproduced: AUC {auc:.3f}, swept {n_swept:,}, "
      f"pred swept {swept['p100'].mean():.1f} vs top {top['p100'].mean():.1f}")

# unmatched swept bbls: formatting bug or genuinely non-residential?
conn = sqlite3.connect(str(config.DB_PATH), timeout=60)
conn.execute("PRAGMA busy_timeout=60000;")
pt = pd.read_sql_query(
    "SELECT borocode, block, lot, bct2020 FROM pluto_tract", conn)
pt["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                 in zip(pt["borocode"], pt["block"], pt["lot"])]
all_pluto_bbls = set(pt["bbl_key"]) - {""}
unmatched = swept_bbls - set(df["bbl_key"])
in_pluto = sum(b in all_pluto_bbls for b in unmatched)
log("swept_bbls_unmatched_to_panel", len(unmatched))
log("unmatched_found_in_full_pluto", in_pluto,
    "high share = real non-residential lots, not a join bug")
stamp(f"  unmatched swept bbls {len(unmatched):,}; of those in full PLUTO: "
      f"{in_pluto:,} ({in_pluto / max(len(unmatched), 1):.1%})")

# ══════════════════════════════════════════════════════════════════════════
stamp("[F1] counterfactual accounting: predicted vs realized, contamination")
CURRENT = "F1"

pred_gap = top["p100"].mean() - swept["p100"].mean()
real_swept = swept["any_viol_2020on"].mean() * 100
real_top = top["any_viol_2020on"].mean() * 100
real_gap = real_top - real_swept
log("claimed_extra_per_1000_pred_basis", pred_gap * 10)
log("realized_basis_extra_per_1000", real_gap * 10,
    "same exercise using realized 77-month rates for BOTH groups")
log("swept_underprediction_pp", real_swept - swept["p100"].mean(),
    "realized minus predicted on the swept subgroup")
stamp(f"  predicted-basis gap {pred_gap:.1f}pp (x10 = {pred_gap*10:.0f}); "
      f"realized-basis gap {real_gap:.1f}pp (x10 = {real_gap*10:.0f})")
stamp(f"  swept subgroup: predicted {swept['p100'].mean():.1f} vs realized "
      f"{real_swept:.1f} (+{real_swept - swept['p100'].mean():.1f}pp miscalibration)")

# subgroup calibration: panel-wide decile calibration does NOT transfer to
# the swept subgroup (selected on DOB-side unobservables + sweep-caused hits)
df["cal_decile"] = pd.qcut(df["p_hat"].rank(method="first"), 10, labels=False) + 1
sub = (df[df["swept7g"] == 1].groupby("cal_decile")
       .agg(pred=("p100", "mean"),
            real=("any_viol_2020on", lambda s: s.mean() * 100),
            n=("p_hat", "size")))
print("\n  swept-subgroup calibration by panel-wide predicted decile:")
print(sub.round(1).to_string(), flush=True)
for d, r in sub.iterrows():
    log(f"swept_calib_d{int(d):02d}_pred", r["pred"])
    log(f"swept_calib_d{int(d):02d}_real", r["real"], f"n={int(r['n'])}")

# sweep-caused contamination bound: swept lots whose ONLY 2020+ violations
# are ECBs written by the 7G events themselves
ecb = pd.read_sql_query("""
    SELECT ecb_violation_number AS num, boro, block, lot, violation_type,
           substr(issue_date,1,4) AS yr
    FROM ecb_violations
    WHERE length(issue_date) >= 8 AND substr(issue_date,1,4) BETWEEN '2020' AND '2026'
""", conn)
ecb["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                  in zip(ecb["boro"], ecb["block"], ecb["lot"])]
ecb = ecb[ecb["bbl_key"] != ""]
sweep_ecb_nums = set(ev7g["ecb_number"].dropna().astype(str)) - {""}
ecb["from_sweep"] = ecb["num"].astype(str).isin(sweep_ecb_nums)
per_lot = ecb.groupby("bbl_key").agg(n_ecb=("num", "size"),
                                     n_sweep_ecb=("from_sweep", "sum"))
sw2 = swept.merge(per_lot, left_on="bbl_key", right_index=True, how="left")
sw2[["n_ecb", "n_sweep_ecb"]] = sw2[["n_ecb", "n_sweep_ecb"]].fillna(0)
only_sweep_paper = ((sw2["any_viol_2020on"] == 1)
                    & (sw2["n_dobviol_2020on"] == 0)
                    & (sw2["n_ecb"] > 0)
                    & (sw2["n_ecb"] == sw2["n_sweep_ecb"]))
real_swept_x = (sw2["any_viol_2020on"] & ~only_sweep_paper).mean() * 100
log("swept_lots_only_sweep_generated_paper", int(only_sweep_paper.sum()),
    "any_viol=1 solely via ECBs the 7G events themselves wrote")
log("realized_swept_excl_sweep_generated", real_swept_x)
log("realized_basis_extra_per_1000_excl_sweepgen", (real_top - real_swept_x) * 10)
stamp(f"  swept lots whose only paper is sweep-written ECBs: "
      f"{int(only_sweep_paper.sum()):,}; swept realized excl. those "
      f"{real_swept_x:.1f} -> realized gap {(real_top - real_swept_x):.1f}pp "
      f"(x10 = {(real_top - real_swept_x)*10:.0f})")

# lots vs inspections: the swap is per-LOT but quoted per-INSPECTION
n_events_at_swept = len(ev7g[ev7g["bbl"].isin(set(swept["bbl_key"]))])
log("n_7g_events_at_swept_panel_lots", n_events_at_swept,
    "the 11,099 lots absorbed this many sweep inspections")
stamp(f"  7G events at the swept panel lots: {n_events_at_swept:,} "
      f"({n_events_at_swept / n_swept:.2f} per lot)")

# ══════════════════════════════════════════════════════════════════════════
stamp("[F2] outcome circularity: periodic streams vs sweep-findable outcomes")
CURRENT = "F2"

u = dob_ledger.union_frame(conn)
u = u[(u["year"] >= 2020) & (u["year"] <= 2026)]
fam_mix = u["family"].value_counts()
print("  union-ledger 2020+ family mix:")
print((fam_mix / fam_mix.sum()).round(3).to_string(), flush=True)
for f, n in fam_mix.items():
    log(f"union2020_family_{f}", int(n))
uflags = (u.assign(constr=(u["family"] == "constr").astype(int),
                   periodic=u["family"].isin(PERIODIC_FAMS).astype(int),
                   other=(u["family"] == "other").astype(int))
          .groupby("bbl_key")[["constr", "periodic", "other"]].max())
ecb_flags = (ecb.assign(field=~ecb["violation_type"].isin(ECB_DEVICE_TYPES))
             .groupby("bbl_key")
             .agg(ecb_any=("num", "size"), ecb_field=("field", "max")))
ecb_flags["ecb_any"] = 1

for fr, cols in ((uflags, ["constr", "periodic", "other"]),
                 (ecb_flags, ["ecb_any", "ecb_field"])):
    fr.index = fr.index.astype(str)
    for c in cols:
        df[c] = df["bbl_key"].map(fr[c]).fillna(0).astype(int)

# v2: any inspector-written paper (any ECB or union construction family)
# v3: strictly field paper (ECB excl. device types, or union constr)
df["viol_v2"] = ((df["ecb_any"] == 1) | (df["constr"] == 1)).astype(int)
df["viol_v3"] = ((df["ecb_field"] == 1) | (df["constr"] == 1)).astype(int)
df["periodic_only"] = ((df["any_viol_2020on"] == 1) & (df["viol_v2"] == 0)).astype(int)

swept = df[df["swept7g"] == 1]
top = df.loc[list(top_idx)]
for name, g in (("swept", swept), ("top_unswept", top)):
    for oc in ("any_viol_2020on", "viol_v2", "viol_v3", "periodic_only"):
        log(f"{name}_{oc}_rate", g[oc].mean() * 100)
    stamp(f"  {name}: any {g['any_viol_2020on'].mean()*100:.1f} | "
          f"ECB-or-constr {g['viol_v2'].mean()*100:.1f} | "
          f"field-only {g['viol_v3'].mean()*100:.1f} | "
          f"periodic-only {g['periodic_only'].mean()*100:.1f}")
gap_v2 = (top["viol_v2"].mean() - swept["viol_v2"].mean()) * 100
gap_v3 = (top["viol_v3"].mean() - swept["viol_v3"].mean()) * 100
log("realized_gap_v2_per_1000", gap_v2 * 10, "sweep-findable, original top set")
log("realized_gap_v3_per_1000", gap_v3 * 10)

# refit the logit on the sweep-findable outcome and re-pick the top set
stamp("  refitting logit on viol_v2 (sweep-findable) ...")
X = build_X(df)
p2, m2 = fit_logit(X, df["viol_v2"])
df["p_hat_v2"] = p2
auc2 = rank_auc(p2, df["viol_v2"])
top2 = df[df["swept7g"] == 0].nlargest(n_swept, "p_hat_v2")
log("auc_v2", auc2)
log("mean_pred_v2_swept", df.loc[swept.index, "p_hat_v2"].mean() * 100)
log("mean_pred_v2_top2", top2["p_hat_v2"].mean() * 100)
log("pred_gap_v2_per_1000",
    (top2["p_hat_v2"].mean() - df.loc[swept.index, "p_hat_v2"].mean()) * 1000)
log("realized_v2_top2", top2["viol_v2"].mean() * 100)
log("realized_v2_gap_refit_per_1000",
    (top2["viol_v2"].mean() - swept["viol_v2"].mean()) * 1000)
log("realized_any_top2", top2["any_viol_2020on"].mean() * 100)
stamp(f"  v2 refit: AUC {auc2:.3f}; predicted swept "
      f"{df.loc[swept.index, 'p_hat_v2'].mean()*100:.1f} vs top2 "
      f"{top2['p_hat_v2'].mean()*100:.1f}; realized v2 top2 "
      f"{top2['viol_v2'].mean()*100:.1f} vs swept {swept['viol_v2'].mean()*100:.1f} "
      f"-> gap x10 = {(top2['viol_v2'].mean() - swept['viol_v2'].mean())*1000:.0f}")

# ══════════════════════════════════════════════════════════════════════════
stamp("[F3] per-visit translation: what inspections actually find at high-risk lots")
CURRENT = "F3"

evd = pd.read_csv(SPINE, usecols=["category_prefix", "family", "agency",
                                  "outcome", "bbl"], dtype={"bbl": str})
evd = evd[(evd["agency"] == 1) & (evd["family"] == "discretionary_field")
          & (evd["outcome"] != "pending")].copy()
evd["hit"] = (evd["outcome"] == "violation").astype(float)
evd = evd.merge(df[["bbl_key", "p_hat", "cal_decile"]], left_on="bbl",
                right_on="bbl_key", how="left")
log("disc_events_matched_to_panel_share", evd["p_hat"].notna().mean())

m7g = evd[evd["category_prefix"] == "7G"]
yield_7g_swept_lots = m7g.loc[m7g["p_hat"].notna(), "hit"].mean() * 100
by_dec = (evd[evd["p_hat"].notna()].groupby("cal_decile")
          .agg(all_disc=("hit", "mean"), n=("hit", "size")))
by_dec_7g = (m7g[m7g["p_hat"].notna()].groupby("cal_decile")
             .agg(yield_7g=("hit", "mean"), n7g=("hit", "size")))
tab = by_dec.join(by_dec_7g)
tab[["all_disc", "yield_7g"]] = tab[["all_disc", "yield_7g"]] * 100
print("\n  per-visit violation yield by panel risk decile of the visited lot:")
print(tab.round(1).to_string(), flush=True)
for d, r in tab.iterrows():
    log(f"pervisit_d{int(d):02d}_all_disc", r["all_disc"], f"n={int(r['n'])}")
    log(f"pervisit_d{int(d):02d}_7g", r["yield_7g"],
        f"n={int(r['n7g']) if not np.isnan(r['n7g']) else 0}")

# yields measured AT the exact top-unswept lots (other programs visited them)
at_top = evd[evd["bbl"].isin(set(top["bbl_key"]))]
at_top_by = at_top.groupby("category_prefix").agg(y=("hit", "mean"), n=("hit", "size"))
at_top_by["y"] *= 100
print("\n  discretionary visits AT the 11,099 top-unswept lots, by program:")
print(at_top_by.sort_values("n", ascending=False).round(1).to_string(), flush=True)
log("pervisit_at_top_unswept_all", at_top["hit"].mean() * 100,
    f"n={len(at_top)} non-7G discretionary visits at those exact lots")
y7g_top_lots = m7g[m7g["cal_decile"] == 10]["hit"].mean() * 100
log("pervisit_7g_at_decile10_lots", y7g_top_lots,
    "7G yield when sweeps DID visit top-decile-risk lots")
log("pervisit_7g_at_its_own_lots", yield_7g_swept_lots)
log("implied_honest_gain_7g_frame_per_1000",
    (y7g_top_lots - yield_7g_swept_lots) * 10)
log("implied_honest_gain_disc_frame_per_1000",
    (at_top["hit"].mean() * 100 - yield_7g_swept_lots) * 10)
stamp(f"  7G per-visit yield at its own lots {yield_7g_swept_lots:.1f}%; at "
      f"decile-10 lots {y7g_top_lots:.1f}%; all-disc at top-unswept lots "
      f"{at_top['hit'].mean()*100:.1f}% -> honest per-visit gain "
      f"{(y7g_top_lots - yield_7g_swept_lots)*10:.0f} to "
      f"{(at_top['hit'].mean()*100 - yield_7g_swept_lots)*10:.0f} per 1,000, "
      f"vs claimed +459")

# ══════════════════════════════════════════════════════════════════════════
stamp("[F4] framing: base rates, 2019 sweeps, composition")
CURRENT = "F4"

cut = df["p_hat"].quantile(0.90)
hr = df[df["p_hat"] >= cut]
share_swept_hr = hr["swept7g"].mean()
share_swept_all = df["swept7g"].mean()
log("top_decile_n", len(hr))
log("top_decile_never_swept_share", (1 - share_swept_hr) * 100)
log("panel_never_swept_share", (1 - share_swept_all) * 100)
log("sweep_enrichment_top_decile", share_swept_hr / share_swept_all,
    "P(swept | top decile) / P(swept), >1 = sweeps over-cover the top decile")
stamp(f"  top decile {len(hr):,} lots, never-swept {100*(1-share_swept_hr):.1f}% "
      f"vs panel-wide never-swept {100*(1-share_swept_all):.1f}%; "
      f"sweep enrichment {share_swept_hr/share_swept_all:.1f}x")

od = pd.read_sql_query("""
    SELECT bin, substr(date_entered,7,4) AS yr FROM open_data
    WHERE complaint_category = '7G'""", conn)
log("open_data_7g_min_year", int(od["yr"].min()), "7G category exists only since")
pre = od[od["yr"] < "2020"]
bb = pd.read_sql_query("SELECT bin, bbl_key FROM bin_bbl_all", conn)
pre_bbls = set(pre.merge(bb, on="bin", how="inner")["bbl_key"].astype(str))
n_top_pre = top["bbl_key"].isin(pre_bbls).sum()
n_hr_unswept_pre = hr.loc[hr["swept7g"] == 0, "bbl_key"].isin(pre_bbls).sum()
log("n_7g_events_2019", len(pre))
log("top11099_with_2019_sweep", int(n_top_pre),
    "'no sweep ever' lots that in fact had a 2019 7G sweep")
log("hr_unswept_with_2019_sweep", int(n_hr_unswept_pre),
    f"of {int((hr['swept7g'] == 0).sum())} map-version never-swept top-decile lots")
stamp(f"  2019 7G events {len(pre):,}; of the 11,099 'never-swept' top lots, "
      f"{n_top_pre} had a 2019 sweep; of the {int((hr['swept7g']==0).sum()):,} "
      f"map never-swept top-decile lots, {n_hr_unswept_pre}")

log("active_permit_share_7g_events", ev7g["active_permit"].mean(),
    "sweeps are construction-site enforcement")
for name, g in (("swept", swept), ("top_unswept", top)):
    log(f"{name}_median_units", g["unitsres"].median())
    log(f"{name}_prewar_share", g["prewar"].mean())
    log(f"{name}_median_yearbuilt", g["yearbuilt"].median())
log("top_unswept_yearbuilt_ge2020", int((top["yearbuilt"] >= 2020).sum()))

# ══════════════════════════════════════════════════════════════════════════
stamp("[F5] leakage: vintages, split-half, drop-poverty, out-of-time model")
CURRENT = "F5"

log("acs_vintage", 2023, "acs/acs5 2023 = 2019-2023 window; NOT strictly pre-2020")
log("pluto_vintage", "current-socrata",
    "class/size/age/bldgarea are current-vintage, not 2019 PLUTO")
log("value_rank_in_features", 0, "value_rank is NOT a model feature (verified)")

rng = np.random.default_rng(42)
half = rng.random(len(df)) < 0.5
pA, _ = fit_logit(X[half], df.loc[half, "any_viol_2020on"])
mA = sm.GLM(np.asarray(df.loc[half, "any_viol_2020on"], float), X[half],
            family=sm.families.Binomial()).fit()
pB = np.asarray(mA.predict(X[~half]))
aucB = rank_auc(pB, df.loc[~half, "any_viol_2020on"])
log("split_half_holdout_auc", aucB)
hold = df.loc[~half].copy()
hold["pB"] = pB
hold["decB"] = pd.qcut(hold["pB"].rank(method="first"), 10, labels=False) + 1
calB = hold.groupby("decB").agg(pred=("pB", lambda s: s.mean() * 100),
                                real=("any_viol_2020on", lambda s: s.mean() * 100))
log("split_half_top_decile_pred", calB.loc[10, "pred"])
log("split_half_top_decile_real", calB.loc[10, "real"])
topB = hold[hold["swept7g"] == 0].nlargest(int(n_swept / 2), "pB")
log("split_half_topunswept_pred", topB["pB"].mean() * 100)
log("split_half_topunswept_real", topB["any_viol_2020on"].mean() * 100)
stamp(f"  split-half holdout AUC {aucB:.3f}; holdout top-unswept pred "
      f"{topB['pB'].mean()*100:.1f} vs realized {topB['any_viol_2020on'].mean()*100:.1f}")

Xnp = X.drop(columns=["tract_poverty"])
pnp, _ = fit_logit(Xnp, df["any_viol_2020on"])
auc_np = rank_auc(pnp, df["any_viol_2020on"])
top_np = df.assign(pnp=pnp)[df["swept7g"] == 0].nlargest(n_swept, "pnp")
jac = len(top_idx & set(top_np.index)) / len(top_idx | set(top_np.index))
log("auc_drop_tract_poverty", auc_np)
log("top_set_jaccard_drop_poverty", jac)
log("top_realized_drop_poverty", top_np["any_viol_2020on"].mean() * 100)
stamp(f"  drop tract_poverty: AUC {auc_np:.3f}, top-set Jaccard {jac:.2f}, "
      f"realized {top_np['any_viol_2020on'].mean()*100:.1f}")

# out-of-time: 2000-09 history -> 2010-19 outcome; then check its top picks'
# realized 2020-26 rates (a model that never saw any 2020s data)
stamp("  building 2000-09 history features ...")
ecb_old = pd.read_sql_query("""
    SELECT boro, block, lot FROM ecb_violations
    WHERE length(issue_date) >= 8
      AND substr(issue_date,1,4) BETWEEN '2000' AND '2009'""", conn)
ecb_old["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                      in zip(ecb_old["boro"], ecb_old["block"], ecb_old["lot"])]
ecb_hist00 = ecb_old[ecb_old["bbl_key"] != ""].groupby("bbl_key").size()
dob_hist00 = dob_ledger.counts_by_bbl(conn, 2000, 2009, "n00")
df["log1p_ecb_hist00"] = np.log1p(df["bbl_key"].map(ecb_hist00).fillna(0))
df["log1p_dobviol_hist00"] = np.log1p(df["bbl_key"].map(dob_hist00).fillna(0))
df["any_prior00"] = ((df["log1p_ecb_hist00"] > 0)
                     | (df["log1p_dobviol_hist00"] > 0)).astype(int)
df["outcome_1019"] = df["any_prior_viol"]  # any 2010-19 ECB/DOB violation
Xo = build_X(df, hist_cols=("log1p_ecb_hist00", "log1p_dobviol_hist00",
                            "any_prior00"))
po, _ = fit_logit(Xo, df["outcome_1019"])
auc_o = rank_auc(po, df["outcome_1019"])
df["p_oot"] = po
top_o = df[df["swept7g"] == 0].nlargest(n_swept, "p_oot")
log("oot_auc_2000s_predicting_2010s", auc_o)
log("oot_top_realized_any_2020on", top_o["any_viol_2020on"].mean() * 100,
    "realized 2020-26 rate of top picks from a model with no 2020s data")
log("oot_top_realized_v2_2020on", top_o["viol_v2"].mean() * 100)
log("oot_vs_orig_top_jaccard",
    len(set(top_o.index) & top_idx) / len(set(top_o.index) | top_idx))
stamp(f"  OOT model (2000-09 hist -> 2010-19 outcome) AUC {auc_o:.3f}; its "
      f"top never-swept picks realize {top_o['any_viol_2020on'].mean()*100:.1f}% "
      f"any / {top_o['viol_v2'].mean()*100:.1f}% sweep-findable in 2020-26")

# ══════════════════════════════════════════════════════════════════════════
stamp("[F6] flat decile curve: denominator, small tracts, controls, programs")
CURRENT = "F6"

eva = pd.read_csv(SPINE, usecols=["received_date", "category_prefix", "family",
                                  "agency", "bct2020", "outcome", "bbl"],
                  dtype={"bbl": str})
eva = eva[(eva["agency"] == 1) & (eva["family"] == "discretionary_field")]
eva = eva[eva["bct2020"].notna()].copy()
eva["bct"] = becker.norm_tract(eva["bct2020"])
eva = eva[eva["outcome"] != "pending"]
eva["hit100"] = (eva["outcome"] == "violation").astype(float) * 100.0
eva["year"] = eva["received_date"].str[:4]
eva["cat_year"] = eva["category_prefix"] + "_" + eva["year"]

lots = pd.read_csv(PANEL, usecols=["bct2020"], dtype={"bct2020": str})
lots_per_tract = lots.groupby("bct2020").size().rename("panel_lots")
trb = eva.groupby("bct").size().rename("n_events").to_frame().join(
    lots_per_tract, how="left")
trb = trb[trb["panel_lots"].notna()].copy()
trb["intensity"] = trb["n_events"] / trb["panel_lots"] * 1000.0


def decile_slope(evx, tr_frame, label, fe="cat_year"):
    tr2 = tr_frame.copy()
    tr2["decile_num"] = pd.qcut(tr2["intensity"].rank(method="first"), 10,
                                labels=False) + 1
    e = evx.merge(tr2[["decile_num"]], left_on="bct", right_index=True,
                  how="inner").copy()
    m = pf.feols(f"hit100 ~ decile_num | {fe}", data=e, vcov={"CRV1": "bct"})
    b = m.coef().iloc[0]
    se = m.se().iloc[0]
    log(f"slope_{label}", b, f"se={se:.3f}, n={m._N}")
    d10 = e.loc[e["decile_num"] == 10, "hit100"].mean()
    d1 = e.loc[e["decile_num"] == 1, "hit100"].mean()
    stamp(f"  {label:<42} slope {b:+.3f} (se {se:.3f})  raw d1 {d1:.1f} d10 {d10:.1f}")
    return b, se


decile_slope(eva, trb, "baseline_replication")

pt["bct"] = becker.norm_tract(pt["bct2020"])
all_lots = pt.groupby("bct").size().rename("all_lots")
tr_alt = eva.groupby("bct").size().rename("n_events").to_frame().join(
    all_lots, how="left")
tr_alt = tr_alt[tr_alt["all_lots"].notna() & (tr_alt["all_lots"] > 0)].copy()
tr_alt["intensity"] = tr_alt["n_events"] / tr_alt["all_lots"] * 1000.0
decile_slope(eva, tr_alt, "denominator_all_pluto_lots")

decile_slope(eva, trb[trb["panel_lots"] >= 50], "tracts_ge50_panel_lots")

tr9 = trb.copy()
tr9["decile_num"] = pd.qcut(tr9["intensity"].rank(method="first"), 10,
                            labels=False) + 1
e9 = eva.merge(tr9[["decile_num"]], left_on="bct", right_index=True, how="inner")
e9 = e9[e9["decile_num"] <= 9]
m9 = pf.feols("hit100 ~ decile_num | cat_year", data=e9, vcov={"CRV1": "bct"})
log("slope_deciles_1to9", m9.coef().iloc[0], f"se={m9.se().iloc[0]:.3f}")
stamp(f"  deciles 1-9 only slope {m9.coef().iloc[0]:+.3f} (se {m9.se().iloc[0]:.3f})")

# building-level controls (events matched to panel lots)
evb = eva.merge(tr9[["decile_num"]], left_on="bct", right_index=True, how="inner")
evb = evb.merge(df[["bbl_key", "log_bldgarea", "no_bldgarea", "size_bin",
                    "class_grp"]], left_on="bbl", right_on="bbl_key", how="inner")
log("events_matched_to_panel_share_f6", len(evb) / len(e9.index.union(evb.index)))
mb = pf.feols("hit100 ~ decile_num + log_bldgarea + no_bldgarea "
              "| cat_year + size_bin + class_grp", data=evb, vcov={"CRV1": "bct"})
log("slope_building_controls", mb.coef().loc["decile_num"],
    f"se={mb.se().loc['decile_num']:.3f}, n={mb._N} (panel-matched events only)")
mp = pf.feols("hit100 ~ decile_num | cat_year", data=evb, vcov={"CRV1": "bct"})
log("slope_panel_matched_no_controls", mp.coef().iloc[0],
    f"se={mp.se().iloc[0]:.3f}, same sample as controls row")
stamp(f"  panel-matched events {len(evb):,}: slope no-controls "
      f"{mp.coef().iloc[0]:+.3f} -> with building controls "
      f"{mb.coef().loc['decile_num']:+.3f}")

# program decomposition + composition + access by decile
e9all = eva.merge(tr9[["decile_num"]], left_on="bct", right_index=True, how="inner")
mix = (e9all.assign(one=1).pivot_table(index="decile_num",
                                       columns="category_prefix", values="one",
                                       aggfunc="sum", fill_value=0))
mix = mix[mix.sum().sort_values(ascending=False).index[:6]]
mix = mix.div(mix.sum(axis=1), axis=0) * 100
print("\n  category mix by intensity decile (% of decile events, top 6):")
print(mix.round(1).to_string(), flush=True)
no8a = e9all[e9all["category_prefix"] != "8A"]
m_no8a = pf.feols("hit100 ~ decile_num | cat_year", data=no8a, vcov={"CRV1": "bct"})
log("slope_excluding_8A", m_no8a.coef().iloc[0], f"se={m_no8a.se().iloc[0]:.3f}")
noacc = (e9all.assign(na=(e9all["outcome"] == "no_access").astype(float) * 100)
         .groupby("decile_num")["na"].mean())
log("no_access_d01", noacc.loc[1])
log("no_access_d10", noacc.loc[10])
stamp(f"  slope excluding 8A {m_no8a.coef().iloc[0]:+.3f} "
      f"(se {m_no8a.se().iloc[0]:.3f}); no-access d1 {noacc.loc[1]:.1f}% "
      f"d10 {noacc.loc[10]:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
stamp("[F7] map: Spearman sensitivity + denominator mechanics")
CURRENT = "F7"

t = pd.read_csv(MAPCSV)
t = t[t["panel_lots"].notna()].copy()
for min_lots in (20, 50, 100, 200, 400):
    ok = (t["panel_lots"] >= min_lots) & (t["scored_lots"] >= min_lots)
    b = t[ok]
    rho = b["disc_per_1000"].corr(b["hr_never_swept_per_1000"], method="spearman")
    pea = b["disc_per_1000"].corr(b["hr_never_swept_per_1000"])
    log(f"spearman_minlots_{min_lots}", rho, f"n={ok.sum()}, pearson={pea:.3f}")
    stamp(f"  MIN_LOTS {min_lots:>3}: spearman {rho:+.3f} pearson {pea:+.3f} "
          f"(n={ok.sum():,})")

b = t[(t["panel_lots"] >= 50) & (t["scored_lots"] >= 50)].copy()
b["hr_rate_all"] = b["n_highrisk"] / b["scored_lots"] * 1000
rho_stock = b["disc_per_1000"].corr(b["hr_rate_all"], method="spearman")
rho_bb = b["hr_never_swept_per_1000"].corr(b["hr_rate_all"], method="spearman")
rho_size = b["disc_per_1000"].corr(b["panel_lots"], method="spearman")
log("spearman_disc_vs_riskstock", rho_stock,
    "panel A vs top-decile RATE incl. swept (risk-stock map)")
log("spearman_unsweptrate_vs_riskstock", rho_bb,
    "panel B is nearly the risk-stock map")
log("spearman_disc_vs_panel_lots", rho_size)
rk = b[["disc_per_1000", "hr_never_swept_per_1000"]].rank()
rk["lg"] = np.log(b["panel_lots"])
r1 = sm.OLS(rk["disc_per_1000"], sm.add_constant(rk["lg"])).fit().resid
r2 = sm.OLS(rk["hr_never_swept_per_1000"], sm.add_constant(rk["lg"])).fit().resid
log("partial_rank_corr_given_log_lots", float(np.corrcoef(r1, r2)[0, 1]))
stamp(f"  disc vs risk-stock {rho_stock:+.3f}; panelB vs risk-stock {rho_bb:+.3f}; "
      f"disc vs tract size {rho_size:+.3f}; partial rank corr "
      f"{float(np.corrcoef(r1, r2)[0, 1]):+.3f}")

n_hr = int(t["n_highrisk"].sum())
n_hru = int(t["n_hr_never_swept"].sum())
log("map_top_decile_lots", n_hr)
log("map_never_swept_top_decile", n_hru)
log("map_never_swept_share", n_hru / n_hr * 100)

conn.close()
out = pd.DataFrame(ROWS)
out.to_csv(OUTCSV, index=False)
stamp(f"wrote {OUTCSV} ({len(out)} evidence rows)")
