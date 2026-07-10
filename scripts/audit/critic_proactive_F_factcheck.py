#!/usr/bin/env python3
"""Critic F: adversarial number audit of post_proactive_substack.md.

Recomputes every computable quantitative claim in the post against the CSVs
in data/analysis/risk_models/ (proactive_*.csv, pfizer_case_study.csv) and
the SQLite DB (verbatim quotes, Pfizer caller/agency splits). External/news
claims are audited separately (not computable here) and only anchored where
the plan file recorded a source.

Output: PASS/WARN/FAIL lines, then a summary. WARN = rounding or framing
drift; FAIL = number does not match its source of truth.
"""
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RM = ROOT / "data" / "analysis" / "risk_models"
POST = ROOT / "data" / "analysis" / "blog_posts" / "post_proactive_substack.md"
DB = ROOT / "data" / "dob_complaints.db"

RESULTS = []


def check(cid, desc, post_val, computed, ok, warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((status, cid, desc, post_val, computed))
    print(f"[{status}] {cid}: {desc}\n       post={post_val}  computed={computed}")


def close(a, b, tol=0.05):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- load CSVs
deco = pd.read_csv(RM / "proactive_decomposition.csv")
spine = pd.read_csv(RM / "proactive_spine_summary.csv")
origin = pd.read_csv(RM / "proactive_origin_information.csv")
mon = pd.read_csv(RM / "proactive_monitoring_estimates.csv")
dil = pd.read_csv(RM / "proactive_dilution_estimates.csv")
idx = pd.read_csv(RM / "proactive_indexing_estimates.csv")
inc = pd.read_csv(RM / "proactive_incident_estimates.csv")
beck = pd.read_csv(RM / "proactive_becker_estimates.csv")
reall = pd.read_csv(RM / "proactive_reallocation.csv")
yy = pd.read_csv(RM / "proactive_yearly_yield.csv")
yt = pd.read_csv(RM / "proactive_yield_turn.csv")
sweep = pd.read_csv(RM / "proactive_sweep_structure.csv")
eq = pd.read_csv(RM / "proactive_equity.csv")
ppfig = pd.read_csv(RM / "proactive_per_permit_figure.csv")
life = pd.read_csv(RM / "proactive_lifecycle.csv")
mapt = pd.read_csv(RM / "proactive_map_tracts.csv")
pfz = pd.read_csv(RM / "pfizer_case_study.csv")

post_text = POST.read_text()

print("=" * 78)
print("SECTION 1 — spine / decomposition")
print("=" * 78)

sv = dict(zip(spine.metric, spine.value))
check("S1", "782,755 scraped complaint records 2020-2026/05",
      782755, int(sv["events_rows"]), sv["events_rows"] == 782755)
check("S2", "249,707 agency-initiated (no 311 ref)",
      249707, int(sv["events_agency_rows"]), sv["events_agency_rows"] == 249707)
check("S3", "'about one third' / 32% agency share",
      "32%", f"{sv['agency_share_overall']*100:.1f}%",
      31.5 <= sv["agency_share_overall"] * 100 <= 32.4)

d = deco[deco.group == "main"].set_index("slice")
stat, fup = d.loc["statutory_periodic"], d.loc["followup_family"]
warm, cold = d.loc["discretionary_warm"], d.loc["discretionary_cold"]
check("S4", "statutory 27.5% of agency work", 27.5,
      round(stat.share_of_agency * 100, 1), close(stat.share_of_agency * 100, 27.5))
check("S5", "statutory finds violations 13.7%", 13.7,
      round(stat.violation_disposition_rate * 100, 1),
      close(stat.violation_disposition_rate * 100, 13.7))
check("S6", "follow-up 'another 7%'", 7.0,
      round(fup.share_of_agency * 100, 2), close(fup.share_of_agency * 100, 7.0, 0.1))
check("S7", "returns (warm) 48.4%", 48.4,
      round(warm.share_of_agency * 100, 1), close(warm.share_of_agency * 100, 48.4))
check("S8", "first-contact discovery 17.1% (~one in six)", 17.1,
      round(cold.share_of_agency * 100, 1), close(cold.share_of_agency * 100, 17.1))
check("S9", "cold violation rate 45.4% vs warm 39.4%",
      "45.4/39.4",
      f"{cold.violation_disposition_rate*100:.1f}/{warm.violation_disposition_rate*100:.1f}",
      close(cold.violation_disposition_rate * 100, 45.4)
      and close(warm.violation_disposition_rate * 100, 39.4))
check("S10", "45.4% highest of the four main slices", "highest",
      f"stat {stat.violation_disposition_rate:.3f} fup {fup.violation_disposition_rate:.3f} "
      f"warm {warm.violation_disposition_rate:.3f} cold {cold.violation_disposition_rate:.3f}",
      cold.violation_disposition_rate
      == max(stat.violation_disposition_rate, fup.violation_disposition_rate,
             warm.violation_disposition_rate, cold.violation_disposition_rate))
check("S11", "~43% Class 1 of hits in both warm and cold slices", "about 43 both",
      f"warm {warm.class1_share_of_ecb_hits*100:.1f} cold {cold.class1_share_of_ecb_hits*100:.1f}",
      close(warm.class1_share_of_ecb_hits * 100, 43, 0.7)
      and close(cold.class1_share_of_ecb_hits * 100, 43, 0.7))
p7g = deco[deco.slice == "prefix_7G"].iloc[0]
check("S12", "area sweeps end without a violation ~9 in 10",
      "~10% with violation", f"7G violation rate {p7g.violation_disposition_rate*100:.1f}%",
      p7g.violation_disposition_rate * 100 < 12)

print("=" * 78)
print("SECTION 2 — origin information (agency vs 311 gap)")
print("=" * 78)

o = origin[origin.model == "lpm_pooled_headline"].iloc[0]
check("O1", "gap 26.4pp, CI 25.6-27.2",
      "26.4 [25.6, 27.2]",
      f"{o.estimate:.2f} [{o['25pct']:.2f}, {o['975pct']:.2f}]",
      close(o.estimate, 26.4) and close(o["25pct"], 25.6) and close(o["975pct"], 27.2))
raw = origin[(origin.model == "raw_rates") & (origin.label == "all 10 mixed categories")].iloc[0]
check("O2", "raw rates 52.5% vs 16.1%", "52.5/16.1",
      f"{raw.viol_agency_pct:.1f}/{raw.viol_caller_pct:.1f}",
      close(raw.viol_agency_pct, 52.5) and close(raw.viol_caller_pct, 16.1))
sub = origin[origin.model == "lpm_pooled_substantive"].iloc[0]
check("O3", "substantive-look gap 24.0", 24.0, round(sub.estimate, 2),
      close(sub.estimate, 24.0))
cats = origin[origin.model.str.match(r"lpm_cat__\d+", na=False)]
check("O4", "gap positive in all ten categories",
      "all 10 positive", f"{(cats.estimate > 0).sum()}/{len(cats)} positive",
      len(cats) == 10 and (cats.estimate > 0).all())

print("=" * 78)
print("SECTION 3 — monitoring per permit (targeting)")
print("=" * 78)

m = mon[(mon.outcome == "n_disc") & (mon.conv_def == "conversion_ge10")
        & (mon.term == "conversion_ge10")].iloc[0]
check("M1", "conversions >=10u draw 72% more, CI 32-124%",
      "+72 [32, 124]",
      f"+{(m.irr-1)*100:.0f} [{(m.irr_lo-1)*100:.0f}, {(m.irr_hi-1)*100:.0f}]",
      close((m.irr - 1) * 100, 72, 0.6) and close((m.irr_lo - 1) * 100, 32, 0.6)
      and close((m.irr_hi - 1) * 100, 124, 0.6))
check("M2", "485,126 permitted jobs", 485126, int(m.n_jobs_sample),
      m.n_jobs_sample == 485126)
c = mon[(mon.outcome == "n_disc") & (mon.conv_def == "conversion_ge10")
        & (mon.term == "log_cost")].iloc[0]
check("M3", "log cost +16%/unit, t=35.7, CI 15-17%",
      "16 [15,17] t35.7",
      f"{(c.irr-1)*100:.1f} [{(c.irr_lo-1)*100:.1f},{(c.irr_hi-1)*100:.1f}] t{c.t:.1f}",
      close((c.irr - 1) * 100, 16, 0.6) and close(c.t, 35.7, 0.1))
dm = mon[(mon.outcome == "n_disc") & (mon.conv_def == "conversion_ge10")
         & (mon.term == "jt_full_demolition")].iloc[0]
check("M4", "demolitions 4.9x, CI 4.2-5.6",
      "4.9 [4.2, 5.6]", f"{dm.irr:.2f} [{dm.irr_lo:.2f}, {dm.irr_hi:.2f}]",
      close(dm.irr, 4.9, 0.06) and close(dm.irr_lo, 4.2, 0.06) and close(dm.irr_hi, 5.6, 0.06))
s = mon[(mon.outcome == "n_stat") & (mon.conv_def == "conversion_ge10")
        & (mon.term == "conversion_ge10")].iloc[0]
check("M5", "statutory conversion excess +4%, CI -31 to +56",
      "+4 [-31, 56]",
      f"+{(s.irr-1)*100:.0f} [{(s.irr_lo-1)*100:.0f}, {(s.irr_hi-1)*100:.0f}]",
      close((s.irr - 1) * 100, 4, 0.6) and close((s.irr_lo - 1) * 100, -31, 0.6)
      and close((s.irr_hi - 1) * 100, 56, 0.9))
d10 = ppfig[ppfig.decile == 10].iloc[0]
check("M6", "top decile: conversions 6.7 vs 2.1 per 100 job-months",
      "6.7 vs 2.1", f"{d10.conv_rate_per_100:.2f} vs {d10.rate_per_100:.2f}",
      close(d10.conv_rate_per_100, 6.7, 0.06) and close(d10.rate_per_100, 2.1, 0.06))

life_s = life[life.block == "summary"].set_index(["split", "series"])
mon_all = life_s.loc[("all", "monitoring")]
inc_all = life_s.loc[("all", "incidents")]
check("M7", "monitoring crests months 16-17",
      "16-17", f"peak {mon_all.peak_month:.0f}, ma3 {mon_all.peak_month_ma3:.0f}",
      {mon_all.peak_month, mon_all.peak_month_ma3} <= {16.0, 17.0})
check("M8", "caller incidents peak in month 1 after permit",
      "month 1", f"peak month {inc_all.peak_month:.0f}",
      inc_all.peak_month == 1)

print("=" * 78)
print("SECTION 4 — dilution / capacity")
print("=" * 78)

dv = dict(zip(dil.term, dil.estimate))
j23, j25, j26 = (dv["active_jobs_monthly_avg_2023"], dv["active_jobs_monthly_avg_2025"],
                 dv["active_jobs_monthly_avg_2026"])
m23, m25 = dv["monitoring_monthly_avg_2023"], dv["monitoring_monthly_avg_2025"]
p23, p25, p26 = dv["per100_avg_2023"], dv["per100_avg_2025"], dv["per100_avg_2026"]
check("D1", "jobs ~96,500 (2023) to ~106,700 (2025)",
      "96,500 / 106,700", f"{j23:,.0f} / {j25:,.0f}",
      close(j23, 96500, 100) and close(j25, 106700, 100))
check("D2", "jobs rose 'about 10%' 2023-2025", "10%",
      f"{(j25/j23-1)*100:.1f}%", close((j25 / j23 - 1) * 100, 10, 0.7))
check("D3", "monitoring fell 26% (1,399 to 1,034)",
      "-26% (1399->1034)", f"{(m25/m23-1)*100:.1f}% ({m23:,.0f}->{m25:,.0f})",
      close((m25 / m23 - 1) * 100, -26, 0.6) and close(m23, 1399, 1) and close(m25, 1034, 1))
check("D4", "per-100 rate 1.45 -> 0.97; 2026 1.04",
      "1.45/0.97/1.04", f"{p23:.2f}/{p25:.2f}/{p26:.2f}",
      close(p23, 1.45, 0.006) and close(p25, 0.97, 0.006) and close(p26, 1.04, 0.006))
check("D5", "2025 trough 'a third less' per site", "-33%",
      f"{dv['per100_change_pct_2023_to_2025']:.1f}%",
      close(dv["per100_change_pct_2023_to_2025"], -33.2, 0.2))
pct_2026_vs_2023 = (p26 / p23 - 1) * 100
check("D6", "title/TLDR/body/closing: 2026 rate '28% below 2023'",
      "-28%", f"{pct_2026_vs_2023:.1f}% (rounds to -29%)",
      close(pct_2026_vs_2023, -28, 0.5), warn=close(pct_2026_vs_2023, -28, 1.0))
el = dil[(dil.model == "ppml_trend_cl_month") & (dil.term == "log_jobs")].iloc[0]
check("D7", "elasticity 0.28, CI -0.01 to 0.57",
      "0.28 [-0.01, 0.57]",
      f"{el.estimate:.3f} [{el.ci_low:.3f}, {el.ci_high:.3f}]",
      close(el.estimate, 0.28, 0.006) and close(el.ci_low, -0.01, 0.006)
      and close(el.ci_high, 0.57, 0.006))
punit = dil.p_vs_unit_elasticity.dropna()
check("D8", "one-for-one scaling rejected p<0.001 in every spec",
      "all p<0.001", f"max p = {punit.max():.2e} over {len(punit)} specs",
      (punit < 0.001).all())
ss20, ss24, ss25 = (dv["ecb_site_safety_total_2020"], dv["ecb_site_safety_total_2024"],
                    dv["ecb_site_safety_total_2025"])
check("D9", "site safety 1,053 -> 169 (-84%), rebound to 311",
      "1053/169/-84%/311",
      f"{ss20:.0f}/{ss24:.0f}/{(ss24/ss20-1)*100:.1f}%/{ss25:.0f}",
      ss20 == 1053 and ss24 == 169 and ss25 == 311
      and close((ss24 / ss20 - 1) * 100, -84, 0.5))
cr20, cr24, cr25 = (dv["ecb_cranes_total_2020"], dv["ecb_cranes_total_2024"],
                    dv["ecb_cranes_total_2025"])
check("D10", "cranes fell 62% 'over the same span' (CSV span = 2020-2025)",
      "-62%",
      f"2020-2025 {(cr25/cr20-1)*100:.1f}%; 2020-2024 {(cr24/cr20-1)*100:.1f}%",
      close((cr25 / cr20 - 1) * 100, -62, 0.6),
      warn=close((cr24 / cr20 - 1) * 100, -62, 1.0))
check("D11", "closing: 2025 site-safety 'below a third of 2020'",
      "<1/3", f"{ss25/ss20:.3f}", ss25 / ss20 < 1 / 3)
check("D12", "closing: 'roughly 107,000 monthly active jobs'",
      "107,000", f"{j25:,.0f}", close(j25, 107000, 500))

print("=" * 78)
print("SECTION 5 — complaint indexing")
print("=" * 78)

hx = idx[(idx.outcome == "proactive_total") & (idx.model == "lag1_jobs")
         & (idx.term == "log_caller_l1") & (idx.vcov == "CRV1 tract")].iloc[0]
check("X1", "0.075 with jobs control, CI 0.062-0.088",
      "0.075 [0.062, 0.088]",
      f"{hx.estimate:.4f} [{hx['25pct']:.4f}, {hx['975pct']:.4f}]",
      close(hx.estimate, 0.075, 0.0006) and close(hx["25pct"], 0.062, 0.0006)
      and close(hx["975pct"], 0.088, 0.0006))
nx = idx[(idx.outcome == "proactive_total") & (idx.model == "lag1_nojobs")
         & (idx.term == "log_caller_l1")].iloc[0]
check("X2", "0.085 without jobs control", 0.085, round(nx.estimate, 4),
      close(nx.estimate, 0.085, 0.0006))
st = idx[(idx.outcome == "proactive_statutory_periodic") & (idx.model == "lag1_jobs")
         & (idx.term == "log_caller_l1") & (idx.vcov == "CRV1 tract")].iloc[0]
check("X3", "statutory moves 'just as much' (0.066)", 0.066,
      round(st.estimate, 4), close(st.estimate, 0.066, 0.0006))

print("=" * 78)
print("SECTION 6 — incident event study")
print("=" * 78)

foc = inc[inc.series == "focal"]
base = foc.baseline_mean_treated.iloc[0]
m0 = foc[(foc.term == "event_month") & (foc.event_time == 0)].iloc[0]
pct0, lo0, hi0 = (100 * m0.coef / base, 100 * m0.ci_low / base, 100 * m0.ci_high / base)
check("I1", "month-0 424% above baseline, CI 381-466 ('about five times baseline')",
      "424 [381, 466]; 5x",
      f"{pct0:.0f} [{lo0:.0f}, {hi0:.0f}]; level {1+pct0/100:.2f}x",
      close(pct0, 424, 0.9) and close(lo0, 381, 0.9) and close(hi0, 466, 0.9)
      and 4.7 <= 1 + pct0 / 100 <= 5.5)
av = foc[foc.term == "avg_months_0_3"].iloc[0]
pav, plo, phi = (100 * av.coef / base, 100 * av.ci_low / base, 100 * av.ci_high / base)
check("I2", "avg months 0-3 = 144% above baseline, CI 117-170",
      "144 [117, 170]", f"{pav:.0f} [{plo:.0f}, {phi:.0f}]",
      close(pav, 144, 0.9) and close(plo, 117, 0.9) and close(phi, 170, 0.9))
mean_m1_3 = foc[(foc.term == "event_month")
                & (foc.event_time.isin([1, 2, 3]))].coef.mean()
check("I3", "TLDR says 144% 'over the following three months' — but window is "
      "months 0-3 incl. the report-month spike; months 1-3 alone are far lower",
      "144% (following 3 months)",
      f"months 1-3 avg = {100*mean_m1_3/base:.0f}% above baseline",
      False, warn=True)
check("I4", "8,448 buildings", 8448, int(foc.n_treated_units.iloc[0]),
      foc.n_treated_units.iloc[0] == 8448)
check("I5", "back to normal by month 3", 3,
      int(foc.decay_first_ci0_month.iloc[0]), foc.decay_first_ci0_month.iloc[0] == 3)
blk = inc[inc.series == "block_spillover"]
bav = blk[blk.term == "avg_months_0_3"].iloc[0]
bbase = blk.baseline_mean_treated.iloc[0]
check("I6", "block: -0.002 on 0.017 baseline, CI -0.005 to +0.001",
      "-0.002/0.017 [-0.005, 0.001]",
      f"{bav.coef:.4f}/{bbase:.4f} [{bav.ci_low:.4f}, {bav.ci_high:.4f}]",
      close(bav.coef, -0.002, 0.0005) and close(bbase, 0.017, 0.0005)
      and close(bav.ci_low, -0.005, 0.0006) and close(bav.ci_high, 0.001, 0.0006))
check("I7", "block pre-trends clean", "clean",
      f"pretrend p = {blk.pretrend_p.iloc[0]:.3f}", blk.pretrend_p.iloc[0] > 0.05)
check("I8", "focal attention 'already drifting up' pre-event", "ramp",
      f"pretrend p = {foc.pretrend_p.iloc[0]:.1e} (rejects flat)",
      foc.pretrend_p.iloc[0] < 0.001)

print("=" * 78)
print("SECTION 7 — Becker margin / reallocation")
print("=" * 78)

b1 = beck[(beck.model == "raw_decile_mean") & (beck.term == "decile_01")].iloc[0]
b10 = beck[(beck.model == "raw_decile_mean") & (beck.term == "decile_10")].iloc[0]
check("B1", "yield 34.1% bottom vs 29.3% top decile",
      "34.1/29.3", f"{b1.estimate:.1f}/{b10.estimate:.1f}",
      close(b1.estimate, 34.1) and close(b10.estimate, 29.3))
check("B2", "intensity spans 'roughly 240x'", "~240",
      f"{b10.mean_intensity/b1.mean_intensity:.0f}x",
      close(b10.mean_intensity / b1.mean_intensity, 240, 5))
lin = beck[beck.model == "lpm_fe_linear"].iloc[0]
check("B3", "slope 0.44 pp/decile, CI 0.27-0.60",
      "0.44 [0.27, 0.60]",
      f"{abs(lin.estimate):.3f} [{abs(lin['975pct']):.3f}, {abs(lin['25pct']):.3f}]",
      close(abs(lin.estimate), 0.44, 0.006) and close(abs(lin["975pct"]), 0.27, 0.006)
      and close(abs(lin["25pct"]), 0.60, 0.006))
dd = beck[beck.model == "lpm_fe_decile"].set_index("term")
sig_neg = [t for t in dd.index if dd.loc[t, "975pct"] < 0]
check("B4", "decline sits entirely in the two most-inspected deciles",
      "deciles 9,10 only", f"significant negative: {sig_neg}",
      set(sig_neg) == {"decile_09", "decile_10"})

rv = dict(zip(reall[reall.block == "headline"].term,
              reall[reall.block == "headline"].value))
check("B5", "swept lots average 52.9% predicted risk", 52.9,
      round(rv["mean_pred_swept_pct"], 2), close(rv["mean_pred_swept_pct"], 52.9))
check("B6", "top never-swept predicted 98.8%, realized 98.5%",
      "98.8/98.5",
      f"{rv['mean_pred_top_unswept_pct']:.1f}/{rv['actual_rate_top_unswept_pct']:.1f}",
      close(rv["mean_pred_top_unswept_pct"], 98.8) and
      close(rv["actual_rate_top_unswept_pct"], 98.5))
check("B7", "459 extra violations per 1,000, CI 443-475",
      "459 [443, 475]",
      f"{rv['extra_viol_per_1000']:.0f} [{rv['extra_viol_per_1000_ci_lo']:.0f}, "
      f"{rv['extra_viol_per_1000_ci_hi']:.0f}]",
      close(rv["extra_viol_per_1000"], 459, 0.6)
      and close(rv["extra_viol_per_1000_ci_lo"], 443, 0.6)
      and close(rv["extra_viol_per_1000_ci_hi"], 475, 0.6))
check("B8", "AUC 0.88", 0.88, round(rv["model_auc"], 3), close(rv["model_auc"], 0.88, 0.005))
check("B9", "median 65 units; all with 2010-19 violation history",
      "65 / 100%",
      f"{rv['units_median_top_unswept']:.0f} / {rv['share_prior_viol_top_unswept']*100:.0f}%",
      rv["units_median_top_unswept"] == 65 and rv["share_prior_viol_top_unswept"] == 1.0)

print("=" * 78)
print("SECTION 8 — sweep structure")
print("=" * 78)

sw = sweep[sweep.analysis == "batching_permutation"].set_index("term")
check("W1", "18.7% same-block within 3 days vs 12.1% benchmark",
      "18.7 vs 12.1",
      f"{sw.loc['share_other_lot_pm3d','value']*100:.1f} vs "
      f"{sw.loc['share_other_lot_pm3d','null_mean']*100:.1f}",
      close(sw.loc["share_other_lot_pm3d", "value"] * 100, 18.7)
      and close(sw.loc["share_other_lot_pm3d", "null_mean"] * 100, 12.1))
sd = sw.loc["share_other_lot_same_day"]
check("W2", "'mostly literal same-day batching'",
      "same-day dominates",
      f"same-day {sd.value*100:.1f} vs null {sd.null_mean*100:.1f}",
      sd.value / sw.loc["share_other_lot_pm3d", "value"] > 0.75)
wb = sweep[sweep.analysis == "within_block_lpm"].set_index("term")
pw = wb.loc["era_pre1940"]
check("W3", "prewar 7.6pp less likely, CI 7.0-8.2",
      "7.6 [7.0, 8.2]",
      f"{abs(pw.value):.2f} [{abs(pw.ci_high):.2f}, {abs(pw.ci_low):.2f}]",
      close(abs(pw.value), 7.6) and close(abs(pw.ci_high), 7.0)
      and close(abs(pw.ci_low), 8.2))
check("W4", "housing-violation history predicts slightly lower inclusion",
      "negative",
      f"log1p_dobviol {wb.loc['log1p_dobviol_hist','value']:.2f}, "
      f"any_prior {wb.loc['any_prior_viol','value']:.2f}",
      wb.loc["log1p_dobviol_hist", "value"] < 0 and wb.loc["any_prior_viol", "value"] < 0)
es = sweep[(sweep.analysis == "substitution_eventstudy")
           & (sweep.term == "avg_months_0_12")].iloc[0]
p24 = 100 * es.value / es.baseline
check("W5", "caller complaints ~24% below controls in year after sweep, CI 12-36",
      "-24 [-36, -12]",
      f"{p24:.1f} [{100*es.ci_low/es.baseline:.1f}, {100*es.ci_high/es.baseline:.1f}]",
      close(p24, -24, 0.9) and close(100 * es.ci_low / es.baseline, -36, 0.9)
      and close(100 * es.ci_high / es.baseline, -12, 0.9))

print("=" * 78)
print("SECTION 9 — yield turn / yearly yield")
print("=" * 78)

yyd = dict(zip(yy.year, yy.violation_yield))
yyn = dict(zip(yy.year, yy.n))
check("Y1", "yield 31% in 2024", "31%", f"{yyd[2024]*100:.2f}%",
      round(yyd[2024] * 100) == 31)
check("Y2", "yield '41%' in 2025 (CSV 41.53 rounds to 42)",
      "41%", f"{yyd[2025]*100:.2f}%", round(yyd[2025] * 100) == 41,
      warn=41 <= yyd[2025] * 100 <= 42)
check("Y3", "yield 44% in first five months of 2026", "44%",
      f"{yyd[2026]*100:.2f}%", round(yyd[2026] * 100) == 44)
check("Y4", "discretionary inspections kept falling in 2025",
      "falling", f"n 2024 {yyn[2024]:,} -> 2025 {yyn[2025]:,}", yyn[2025] < yyn[2024])
check("Y5", "yield climbed 13 points (2024 -> 2026)",
      "13", f"{(yyd[2026]-yyd[2024])*100:.1f}",
      close((yyd[2026] - yyd[2024]) * 100, 13, 0.6))
w25 = yt[(yt.term == "2024_to_2025__within_share_of_change")].estimate.iloc[0]
w26 = yt[(yt.term == "2024_to_2026jm__within_share_of_change")].estimate.iloc[0]
check("Y6", "'about half' composition / half within",
      "~50/50", f"within share: 2025 {w25:.0f}%, 2026jm {w26:.0f}%",
      40 <= w25 <= 60 and 40 <= w26 <= 60)
l25 = yt[(yt.model == "pooled_cat_fe") & (yt.term == "2025")].iloc[0]
l26 = yt[(yt.model == "pooled_cat_fe") & (yt.term == "2026")].iloc[0]
check("Y7", "within-program +4.3 [2.8, 5.9] in 2025; +6.1 [4.5, 7.7] in 2026",
      "4.3 [2.8,5.9]; 6.1 [4.5,7.7]",
      f"{l25.estimate:.2f} [{l25['25pct']:.2f},{l25['975pct']:.2f}]; "
      f"{l26.estimate:.2f} [{l26['25pct']:.2f},{l26['975pct']:.2f}]",
      close(l25.estimate, 4.3, 0.06) and close(l25["25pct"], 2.8, 0.06)
      and close(l25["975pct"], 5.9, 0.06) and close(l26.estimate, 6.1, 0.06)
      and close(l26["25pct"], 4.5, 0.06) and close(l26["975pct"], 7.7, 0.06))
rs = yt[(yt.block == "risk_scores") & (yt.model == "mean_p100")]
rs24 = rs[rs.term.isin(["2024H1", "2024H2"])].estimate.mean()
rs2526 = rs[rs.term.isin(["2025H1", "2025H2", "2026H1"])].estimate.mean()
check("Y8", "mean predicted risk drifted 'from about 56 to 54'",
      "56 -> 54", f"{rs24:.1f} -> {rs2526:.1f}",
      close(rs24, 56, 0.6) and close(rs2526, 54, 0.7))

print("=" * 78)
print("SECTION 10 — equity")
print("=" * 78)

alloc = eq[(eq.part == "allocation") & (eq.outcome == "n_disc_constr")].set_index("term")
for cid, term, pv, plo_, phi_ in [
        ("E1", "pct_black", 5.1, 2.7, 7.5),
        ("E2", "pct_hispanic", 6.8, 4.0, 9.7),
        ("E3", "pct_asian", 5.9, 2.7, 9.3)]:
    r = alloc.loc[term]
    check(cid, f"{term} +{pv}% per 10pp, CI {plo_}-{phi_}",
          f"{pv} [{plo_}, {phi_}]",
          f"{r.per_10pp:.2f} [{r.per_10pp_lo:.2f}, {r.per_10pp_hi:.2f}]",
          close(r.per_10pp, pv, 0.06) and close(r.per_10pp_lo, plo_, 0.06)
          and close(r.per_10pp_hi, phi_, 0.06))
statq = eq[(eq.part == "allocation") & (eq.outcome == "n_stat")
           & (eq.exposure_offset.str.contains("job-months"))].set_index("term")
nulls = all(statq.loc[t, "per_10pp_lo"] < 0 < statq.loc[t, "per_10pp_hi"]
            for t in ["pct_black", "pct_hispanic", "pct_asian"])
check("E4", "statutory contrast: no loading on any of the three", "all null",
      f"CIs straddle zero: {nulls}", nulls)

print("=" * 78)
print("SECTION 11 — map (Spearman)")
print("=" * 78)

mt = mapt[(mapt.ok_a == True) & (mapt.ok_b == True)]  # noqa: E712
rho = mt.disc_per_1000.corr(mt.hr_never_swept_per_1000, method="spearman")
check("P1", "Spearman +0.81 across 1,976 tracts lit on both",
      "0.81 / 1,976", f"{rho:.3f} / {len(mt):,}",
      close(rho, 0.81, 0.006) and len(mt) == 1976)

print("=" * 78)
print("SECTION 12 — Pfizer case study (CSV recomputation)")
print("=" * 78)

pfz["dt"] = pd.to_datetime(pfz.event_date)
pfz["year"] = pfz.dt.dt.year
ecb = pfz[pfz.event_type == "ecb_violation"]
ecb25 = ecb[ecb.year == 2025]
check("F1", "27 penalty (ECB) violations site-wide in 2025", 27, len(ecb25),
      len(ecb25) == 27)
check("F2", "$133,780 imposed in 2025", 133780,
      f"{ecb25.penalty_imposed.sum():,.0f}", ecb25.penalty_imposed.sum() == 133780)
check("F3", "$30,000 of it paid", 30000, f"{ecb25.penalty_paid.sum():,.0f}",
      ecb25.penalty_paid.sum() == 30000)
tower25 = ecb25[ecb25.bin == 1037552]
check("F4", "news 'seven construction-safety violations in 2025' at the tower (BIN 1037552)",
      7, len(tower25), len(tower25) == 7)
n_c1 = tower25.severity.astype(str).str.contains("CLASS - 1").sum()
check("F5", "'all but one classed immediately hazardous' (6 of 7 Class 1)",
      "6/7", f"{n_c1}/{len(tower25)}", n_c1 == 6 and len(tower25) == 7)
first_tower25 = tower25.dt.min()
buckle = pd.Timestamp("2026-07-07")
within_1yr = (tower25.dt >= buckle - pd.DateOffset(years=1)).sum()
check("F6", "closing says 'cited seven times in the year before its columns buckled' "
      "(the seven are calendar-2025 citations; count within 12 months pre-buckling)",
      "7 in prior year", f"{within_1yr} of 7 fall in 2025-07-07..2026-07-07; "
      f"earliest tower citation {first_tower25.date()}",
      within_1yr == 7, warn=within_1yr >= 5)

comp = pfz[pfz.event_type == "complaint"].copy()
comp25 = comp[comp.year == 2025]
check("F7", "37 complaints site-wide in 2025", 37, len(comp25), len(comp25) == 37)

caller = comp[comp.origin == "311"]
caller_by_yr = caller.year.value_counts().sort_index().to_dict()
check("F8", "'all 24 caller complaints' — CSV count of caller complaints at the site",
      24, f"total {len(caller)}; by year {caller_by_yr} "
      f"(24 = through 2025 only; window runs to May 2026)",
      len(caller) == 24, warn=len(caller[caller.year <= 2025]) == 24)

disp_u = caller.status_or_disposition.astype(str).str.upper()
no_viol = disp_u.str.contains("NO VIOLATION WARRANTED")
dupe = disp_u.str.contains("PLEASE SEE COMPLAINT")
swo = disp_u.str.contains("STOP WORK ORDER")
check("F9", "every caller complaint 'closed with no violation'",
      "all no-violation",
      f"of {len(caller)}: {no_viol.sum()} no-violation, {dupe.sum()} duplicate-referral, "
      f"{swo.sum()} closed 'STOP WORK ORDER FULLY RESCINDED'",
      swo.sum() == 0 and no_viol.sum() + dupe.sum() == len(caller))

# closure speed: parse the leading MM/DD/YYYY of status_or_disposition
disp_date = pd.to_datetime(
    caller.status_or_disposition.astype(str).str.extract(
        r"^(\d{2}/\d{2}/\d{4})")[0], errors="coerce")
days = (disp_date - caller.dt).dt.days.dropna()
share_2d = (days <= 2).mean() if len(days) else np.nan
check("F10", "'most within a day or two'",
      ">50% closed <=2 days",
      f"{share_2d*100:.0f}% of {len(days)} dated closures <=2 days "
      f"(median {days.median():.0f}d)", share_2d > 0.5)

jobs = pfz[(pfz.event_type == "permit_job_filed")
           & pfz.record_id.isin(["M01075131-I1", "M01075133-I1"])]
units = jobs.description.str.extract(r"-> (\d+)")[0].astype(float).sum()
cost = jobs.amount.sum()
check("F12", "conversion jobs filed July 2024; 481+927=1,408 units; ~$195M",
      "2024-07; 1,408; $195M",
      f"{sorted(jobs.dt.dt.date.astype(str))}; {units:.0f}; ${cost/1e6:.1f}M",
      set(jobs.dt.dt.strftime("%Y-%m")) == {"2024-07"} and units == 1408
      and close(cost / 1e6, 195, 0.5))

text = (pfz.description.fillna("") + " " + pfz.inspector_comments.fillna("")
        + " " + pfz.status_or_disposition.fillna("")).str.upper()
pfz["_text"] = text
swo25 = pfz[(pfz.year == 2025) & text.str.contains("STOP WORK|STOP ALL WORK")]
swo_dates = sorted(swo25.dt.dt.date.unique())
check("F13", "'roughly ten stop-work episodes' in 2025",
      "~10", f"{len(swo_dates)} distinct 2025 dates with stop-work text "
      f"({len(swo25)} rows)", 8 <= len(swo_dates) <= 12)

# incident narratives
def has_row(mask, cid, desc, post_val):
    sel = pfz[mask]
    got = f"{len(sel)} row(s): " + "; ".join(
        f"{r['dt'].date()} {r['event_type']}" for _, r in sel.head(3).iterrows())
    check(cid, desc, post_val, got if len(sel) else "NOT FOUND", len(sel) > 0)

has_row((pfz.year == 2025) & (pfz.dt.dt.month == 5) & text.str.contains("CHIMNEY"),
        "F14", "May 2025: demolished chimney material fell onto delivery truck",
        "May 2025 chimney event")
has_row((pfz.year == 2025) & (pfz.dt.dt.month == 6)
        & text.str.contains("70,000|70000|70K"),
        "F15", "June 2025: 70,000-pound crane hoist ('70K LBS' in ECB text)",
        "June 2025 crane event")
crane_ecb = ecb25[(ecb25.dt.dt.month == 6)
                  & ecb25.severity.astype(str).str.contains("CLASS - 1")]
check("F16", "crane episode: five immediately-hazardous violations, $30,000 (site-wide June)",
      "5 Class-1, $30k",
      f"{len(crane_ecb)} Class-1 June ECBs (BINs {sorted(crane_ecb.bin.unique())}), "
      f"imposed ${crane_ecb.penalty_imposed.sum():,.0f}",
      len(crane_ecb) == 5 and crane_ecb.penalty_imposed.sum() == 30000)
has_row((pfz.year == 2025) & (pfz.dt.dt.month == 8)
        & text.str.contains("33RD|33 RD|PANEL|DOOR FALLING"),
        "F17", "August 2025: metal panel fell from 33rd floor; full SWO "
        "(DOB text says 'DOOR FALLING'; '33rd floor'/'panel' is news-sourced)",
        "Aug 2025, 33rd floor")
has_row((pfz.year == 2025) & (pfz.dt.dt.month == 9)
        & text.str.contains("27TH") & text.str.contains("26TH|26 FLOOR|TO 26"),
        "F18", "September 2025: worker fell 27th->26th through unguarded opening",
        "Sep 2025 fall")
has_row((pfz.year == 2025) & text.str.contains("FAIL(ED|URE) TO (IMMEDIATELY )?"
                                               "(NOTIFY|REPORT)"),
        "F19", "separate violation for not reporting the fall",
        "failure-to-report violation")

jun16 = pfz[(pfz.dt == "2026-06-16") & (pfz.event_type == "complaint")]
check("F20", "June 16 (2026) worker-endangerment (cat 91) complaint exists; "
      "NOTE its origin is 'unknown (BIS not yet scraped)'",
      "2026-06-16 cat 91",
      "; ".join(f"{r['category_or_type']} origin={r['origin']}"
                for _, r in jun16.iterrows()) if len(jun16) else "NOT FOUND",
      len(jun16) > 0 and jun16.category_or_type.astype(str)
      .str.contains("91|WORKER|ENDANGER", case=False).any())
resc = pfz[pfz._text.str.contains("RESCIND")]
resc26 = resc[resc.year == 2026]
resc_dates = sorted(set(re.findall(r"RESCINDED?\s*(?:ON)?\s*(\d{1,2}/\d{1,2}/\d{2,4})",
                                   " ".join(resc26._text))))
check("F21", "SWO rescinded June 26, eleven days before July 7",
      "rescind 6/26/2026; 11 days",
      f"rescind mentions dated {resc_dates or [str(d.date()) for d in resc26.dt]}; "
      f"gap to 7/7 = {(date(2026,7,7)-date(2026,6,26)).days} days",
      (date(2026, 7, 7) - date(2026, 6, 26)).days == 11
      and (any("6/26" in d for d in resc_dates) or len(resc26) > 0))

agency_first = comp[(comp.origin == "agency") & (comp.dt >= "2024-07-03")].dt.min()
gap_months = (agency_first - pd.Timestamp("2024-07-03")).days / 30.44
check("F22", "first proactive sweep nine months after jobs filed",
      "9 months", f"first agency complaint {agency_first.date()} = "
      f"{gap_months:.1f} months after 2024-07-03", 8.5 <= gap_months <= 9.9)

pre2024 = pfz[(pfz.year <= 2023) & pfz.event_type.isin(["complaint", "ecb_violation"])]
check("F23", "'Through 2023 the site generated almost nothing' "
      "(complaints+ECB before 2024)", "almost nothing",
      f"{len(pre2024)} rows ({pre2024.event_type.value_counts().to_dict()})",
      len(pre2024) <= 5)

viol_2yr = pfz[pfz.event_type.isin(["ecb_violation"])
               & (pfz.dt >= "2024-07-07") & (pfz.dt < "2026-07-07")]
check("F24", "closing window: ECB violations in the two years before the buckling "
      "(origin resolved against DB complaint linkage in Q7/Q8 below)",
      "all agency", f"{len(viol_2yr)} ECB rows in window", len(viol_2yr) > 0)

print("=" * 78)
print("SECTION 13 — verbatim quotes vs DB (byte-identity) + DB cross-checks")
print("=" * 78)

# The three strings quoted verbatim in the post + the two script-verified
# comments the post paraphrases (falling door SWO, watchpersons). All five
# must match the DB byte-for-byte as substrings.
QUOTES = {
    "Q1": ("post quote, super log book (sweep 1723097, 11 wks pre = 2026-04-21)",
           "the superintendent log book had not been signed by the super for "
           "more than 3 weeks", "comments"),
    "Q2": ("post quote, windborne debris (compliance 1724937, 8 wks pre = 2026-05-12)",
           "pose windborne risk and create a tripping hazard", "comments"),
    "Q3": ("post quote, caller 'large item' (subject of 1706633, Oct 2025)",
           "a large item fell and broke through 5 floors and almost hit someone",
           "subject"),
    "Q4a": ("script-verified falling-door/SWO comment (1701635)",
            "FULL STOP WORK ORDER ISSUED. PLEASE SEE COMPLAINT # 1701631.",
            "comments"),
    "Q4b": ("script-verified watchperson comment (1711921)",
            "upon request for proof of FDNY watchperson certificate",
            "comments"),
}
con = sqlite3.connect(DB)
rows = pd.read_sql(
    "SELECT complaint_number, received_date, ref_311, category_code, "
    "category_description, disposition, ecb_violation, comments, subject "
    "FROM bis_scrape WHERE bin IN ('1037551','1037552')", con)
blob = rows.fillna("")
for cid, (label, q, field) in QUOTES.items():
    hit = blob[blob[field].str.contains(re.escape(q), regex=True)]
    ci_hit = blob[blob[field].str.lower().str.contains(re.escape(q.lower()))]
    when = "; ".join(hit.received_date.head(2)) if len(hit) else (
        "; ".join(ci_hit.received_date.head(2)) if len(ci_hit) else "-")
    check(cid, f"{label}: byte-identical in DB",
          f"'{q[:45]}...'",
          f"exact={len(hit)} case-insensitive={len(ci_hit)} on {when}",
          len(hit) > 0, warn=len(ci_hit) > 0)

inapp = pfz[pfz._text.str.contains("INAPPROPRIATE RIGGING")]
check("Q5", "'rigging the inspectors called inappropriate' (verbatim in ECB text)",
      "inappropriate rigging",
      f"{len(inapp)} row(s) with 'INAPPROPRIATE RIGGING', dates "
      f"{sorted(set(inapp.event_date))[:2]}", len(inapp) > 0)

# quote timing anchors: 11 weeks / 8 weeks before July 7
d1 = pd.to_datetime(blob[blob.complaint_number == "1723097"].received_date.iloc[0])
d2 = pd.to_datetime(blob[blob.complaint_number == "1724937"].received_date.iloc[0])
w1 = (pd.Timestamp("2026-07-07") - d1).days / 7
w2 = (pd.Timestamp("2026-07-07") - d2).days / 7
check("Q6", "'Eleven weeks before' (sweep) and 'Eight weeks before' (compliance)",
      "11 wks / 8 wks", f"{w1:.1f} wks ({d1.date()}) / {w2:.1f} wks ({d2.date()})",
      close(w1, 11, 0.5) and close(w2, 8, 0.5))

# CORE CLAIM: every violation agency-initiated / no caller complaint yielded one.
rows["rd"] = pd.to_datetime(rows.received_date, errors="coerce")
recent = rows[rows.rd >= "2025-01-01"]
ecb_linked = recent[recent.ecb_violation.fillna("").str.strip() != ""]
caller_linked = ecb_linked[ecb_linked.ref_311.fillna("").str.strip() != ""]
detail = "; ".join(
    f"{r.complaint_number} ({r.rd.date()}, ref {r.ref_311}, cat "
    f"{str(r.category_code)[:2]}) -> ECB {r.ecb_violation}"
    for _, r in caller_linked.iterrows())
check("Q7", "TLDR/body/closing: 'every violation came from an inspection DOB "
      "initiated' — DB linkage of ECB numbers to complaints with a 311 ref",
      "0 caller-linked violations",
      f"{len(ecb_linked)} ECB-linked complaints since 2025; "
      f"{len(caller_linked)} caller-linked: {detail or 'none'}",
      len(caller_linked) == 0)
ecbcsv = pfz[pfz.event_type == "ecb_violation"]
for _, r in caller_linked.iterrows():
    for ref in str(r.ecb_violation).split():
        sel = ecbcsv[ecbcsv.record_id == ref]
        if len(sel):
            v = sel.iloc[0]
            check("Q8", f"caller-linked ECB {ref} sits inside the post's own "
                  "'27 penalty violations in 2025'",
                  "not in the 27",
                  f"{v['event_date']} {v['severity']} imposed "
                  f"${v['penalty_imposed']:,.0f} — IS in pfizer_case_study.csv",
                  False)

# agency share by year (post: 'stable since 2020')
shares = pd.read_sql(
    "SELECT substr(received_date,7,4) AS yr, "
    "AVG(CASE WHEN ref_311 IS NULL OR trim(ref_311)='' THEN 1.0 ELSE 0 END) AS agency_share "
    "FROM bis_scrape WHERE substr(received_date,7,4) BETWEEN '2020' AND '2026' "
    "GROUP BY 1 ORDER BY 1", con)
con.close()
rng = shares.agency_share.astype(float)
check("Q9", "'the share has been stable since 2020' (31-34%/yr; 2026 partial dips)",
      "stable", ", ".join(f"{y}:{s*100:.0f}%" for y, s in
                          zip(shares.yr, shares.agency_share)),
      rng.min() > 0.30, warn=rng.min() > 0.27)

print("=" * 78)
print("SECTION 14 — internal consistency of the post text")
print("=" * 78)

for cid, needle, times in [
        ("T1", "28%", 4), ("T2", "249,707", None), ("T3", "1.45", None),
        ("T4", "0.97", None), ("T5", "1.04", None), ("T6", "96,500", None),
        ("T7", "106,700", None), ("T8", "45.4%", None), ("T9", "459", None)]:
    n = post_text.count(needle)
    check(cid, f"'{needle}' appears consistently", f">=1 use", f"{n} uses", n >= 1)

figs = re.findall(r"FIGURE: (\S+)", post_text)
art = ROOT / "data" / "analysis" / "blog_posts" / "artifacts"
missing = [f for f in figs if not (art / f).exists()]
check("T10", "all referenced figure files exist", "0 missing",
      f"{len(figs)} figures, missing: {missing}", not missing)

scripts_named = set(re.findall(r"(?:scripts/)?(proactive_\w+\.py|pfizer_case_study\.py)",
                               post_text))
missing_s = [s for s in scripts_named if not (ROOT / "scripts" / s).exists()]
check("T11", "all scripts named in methodology exist", "0 missing",
      f"{len(scripts_named)} named, missing: {missing_s}", not missing_s)

# 33% vs 28% framing: both must appear and be tied to distinct anchors
has_third = "a third less attention per site at the 2025 trough" in post_text
has_28 = "28% below" in post_text
check("T12", "-33% (2025 trough) and -28% (2026) both present, distinct anchors",
      "both", f"third@2025 trough: {has_third}; 28% below: {has_28}",
      has_third and has_28)

tldr_seven = "The site had received seven construction-safety violations in 2025" in post_text
body_seven = "seven construction-safety violations in 2025 at the tower" in post_text
check("T13", "TLDR attributes the news 'seven violations' to 'the site'; body "
      "correctly scopes it to the tower BIN; the site-wide 2025 count is 27",
      "tower-scoped everywhere", f"TLDR site-scoped: {tldr_seven}; body tower-scoped: {body_seven}",
      not tldr_seven, warn=body_seven)

print("=" * 78)
print("SUMMARY")
print("=" * 78)
n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
n_warn = sum(1 for r in RESULTS if r[0] == "WARN")
n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
print(f"{len(RESULTS)} checks: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")
for st, cid, desc, pv, cv in RESULTS:
    if st != "PASS":
        print(f"  [{st}] {cid}: {desc} | post={pv} | computed={cv}")
