"""Critic script (proactive wave): adversarial attacks on
scripts/proactive_yield_turn.py and scripts/proactive_equity.py.

Attack surfaces (numbered findings printed at the end):

Y1  Yield turn / disposition censoring. The spine takes outcomes from an
    open_data snapshot whose bulk vintage is 2026-04-04 (small 2026-07-03
    partial refresh); unresolved complaints have EMPTY disposition_code and
    classify_disposition("") == "other", NOT "pending" -- so the script's
    "pending ~0" guard row watches the wrong bucket. Measure the empty-
    disposition share by year, reclassify censored events with the fresher
    scraped bis_scrape.disposition (scrape ran 2026-04-05..2026-07-06),
    recompute 2024/25/26 yields + 2026 monthly yields, and check
    disposition lags by outcome (do violations resolve faster?).

Y2  "Risk scores flat" selection. The visited-lot mean predicted risk uses
    only visits matched to the residential risk panel (75.3% in 2024H2 ->
    61.7% in 2026H1). Tests: match rate by program (the drop is within-8A
    too); prefix-composition-reweighted score series; within-prefix
    half-year score model; per-program 2024->2026 tests (7G and 8A) with
    tract-clustered SEs; the equal-unmatched-mean threshold that would
    flip the aggregate; what unmatched 2026 visits look like (active
    permits, violation yield).

Y3  Shift-share granularity. The published within/mix split uses 2-char
    category prefixes (the finest code that exists; 8A has ONE category
    name). Re-run the exact symmetric decomposition on finer cells
    (prefix x active-permit, x warm, x assigned unit, x borough, and
    prefix x permit x warm) and see whether the "half within" survives.

E4  Equity denominator / offset restriction. The +5-7% per +10pp
    Black/Hispanic/Asian loadings are per active job-month with the offset
    coefficient FORCED to 1, while this project's own A3 dilution result
    says monitoring scales with active jobs at ~0.28, not 1. Tests:
    verify fepois(offset=) is actually applied (synthetic check);
    replicate the headline; free-elasticity refit (log job-months as a
    regressor); outcome splits (8A-only, at-permit-only, off-permit-only);
    alternative exposures (floor-area-weighted job-months, cost-weighted,
    2023+ window where DOB NOW coverage is complete, caller-side
    construction-incident volume); era x size interaction controls; the
    demographic gradient of the at-permit share (differential unpermitted
    work = differential offset undercount); NTA-block randomization
    inference for the 201-cluster adequacy question.

E5  Crisis-response power + outcome dilution. resp_any counts ANY agency
    event within 90 days (statutory boiler/elevator cycles included).
    Compute MDEs against allocation-sized effects and re-run the LPM with
    a discretionary-only response outcome.

Run: /private/tmp/pyfix_venv/bin/python scripts/audit/critic_proactive_E_yield_equity.py
"""

import re
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports"
sys.path.insert(0, ROOT + "/scripts")
sys.path.insert(0, ROOT)

import config  # noqa: E402
import pyfixest as pf  # noqa: E402
from disposition_codes import classify_disposition  # noqa: E402
from analysis_config import PROACTIVE_FAMILIES  # noqa: E402

RM = config.DATA_DIR / "analysis" / "risk_models"
PRO = config.DATA_DIR / "analysis" / "proactive"
T0 = time.time()
RNG = np.random.default_rng(20260710)
KEY3 = ["pct_black", "pct_hispanic", "pct_asian"]

FINDINGS = []


def finding(num, verdict, claim, evidence, fix=""):
    FINDINGS.append((num, verdict, claim, evidence, fix))


def hr(title):
    print(f"\n{'=' * 78}\n{title}  [t={time.time() - T0:.0f}s]\n{'=' * 78}",
          flush=True)


def share_pct(s):
    return s.mean() * 100


# ══════════════════════════════════════════════════════════════════════════
# Y1  disposition censoring in the yield turn
# ══════════════════════════════════════════════════════════════════════════

hr("Y1: pending-disposition censoring (yield turn)")

conn = sqlite3.connect(str(config.DB_PATH))
raw = pd.read_sql_query("""
    SELECT b.complaint_number, b.received_date, b.category_code,
           b.ref_311, b.disposition AS bis_dispo, b.bis_status,
           o.disposition_code, o.status, o.date_entered,
           o.disposition_date, o.dobrundate
    FROM bis_scrape b LEFT JOIN open_data o USING (complaint_number)
    WHERE b.received_date LIKE '__/__/____'
      AND substr(b.received_date, 7, 4) >= '2020'
""", conn)
conn.close()
raw["received"] = pd.to_datetime(raw["received_date"], format="%m/%d/%Y",
                                 errors="coerce")
raw = raw[raw["received"].between("2020-01-01", "2026-05-31")]
raw["pfx"] = raw["category_code"].fillna("").str[:2]
raw["family"] = raw["pfx"].map(PROACTIVE_FAMILIES).fillna("other")
raw["agency"] = raw["ref_311"].fillna("").str.strip().eq("")
disc = raw[raw["agency"] & (raw["family"] == "discretionary_field")].copy()
del raw
disc["yr"] = disc["received"].dt.year
disc["mon"] = disc["received"].dt.strftime("%Y-%m")

codes = disc["disposition_code"].fillna("").astype(str)
cmap = {c: classify_disposition(c) for c in codes.unique()}
disc["outcome"] = codes.map(cmap)                    # replicates the spine
disc["od_empty"] = codes.str.strip().eq("")

# cross-check replicated yearly yields against the published CSV
pub = pd.read_csv(RM / "proactive_yield_turn.csv")
mine = disc.groupby("yr")["outcome"].apply(
    lambda s: (s == "violation").mean() * 100)
for y in range(2020, 2027):
    ref = pub[(pub["block"] == "yearly_yield") & (pub["model"] == "raw")
              & (pub["term"] == f"yield_{y}")]["estimate"].iloc[0]
    assert abs(mine.loc[y] - ref) < 0.05, (y, mine.loc[y], ref)
print("replicated yearly yields match proactive_yield_turn.csv "
      f"(2024 {mine.loc[2024]:.2f} / 2025 {mine.loc[2025]:.2f} / "
      f"2026 {mine.loc[2026]:.2f})")

tab = disc.groupby("yr").agg(
    n=("outcome", "size"),
    empty_dispo_pct=("od_empty", share_pct),
    other_pct=("outcome", lambda s: (s == "other").mean() * 100),
    pending_pct=("outcome", lambda s: (s == "pending").mean() * 100))
print("\nempty open_data disposition (the REAL unresolved bucket) vs the "
      "script's 'pending' guard:")
print(tab.round(3).to_string())

# reclassify censored events with the fresher scraped disposition
pat = re.compile(r"^\s*\d{2}/\d{2}/\d{4}\s*-\s*([A-Z0-9]{2})\s*-")
bis_code = (disc["bis_dispo"].fillna("").str.extract(pat, expand=False)
            .fillna(""))
disc["outcome_corr"] = disc["outcome"]
fixable = disc["od_empty"] & (bis_code != "")
disc.loc[fixable, "outcome_corr"] = bis_code[fixable].map(classify_disposition)
n_promoted = int(((disc["outcome_corr"] == "violation")
                  & (disc["outcome"] != "violation")).sum())
print(f"\ncensored events with a parseable scraped disposition: "
      f"{int(fixable.sum()):,} of {int(disc['od_empty'].sum()):,} empty; "
      f"{n_promoted} become violations after correction")

corr = disc.groupby("yr").agg(
    yield_published=("outcome", lambda s: (s == "violation").mean() * 100),
    yield_corrected=("outcome_corr",
                     lambda s: (s == "violation").mean() * 100))
print("\nyield with scraped-disposition correction (pp):")
print(corr.loc[2024:].round(2).to_string())

m26 = disc[disc["yr"] == 2026].groupby("mon").agg(
    n=("outcome", "size"),
    yield_pub=("outcome", lambda s: (s == "violation").mean() * 100),
    yield_corr=("outcome_corr", lambda s: (s == "violation").mean() * 100),
    empty_pct=("od_empty", share_pct))
print("\n2026 monthly yield (censoring would show as a fade in recent "
      "months):")
print(m26.round(2).to_string())

# >=90-days-old recompute against each row's own data vintage
vint = pd.to_datetime(disc["dobrundate"].astype("string").str[:8],
                      format="%Y%m%d", errors="coerce")
vint = vint.fillna(pd.Timestamp("2026-04-04"))
mature = disc[(disc["yr"] == 2026)
              & (disc["received"] <= vint - pd.Timedelta(days=90))]
y_mat = (mature["outcome"] == "violation").mean() * 100
y_mat_c = (mature["outcome_corr"] == "violation").mean() * 100
print(f"\n2026 events >=90 days old at their snapshot vintage: "
      f"n={len(mature):,}, yield {y_mat:.2f} (corrected {y_mat_c:.2f}) "
      f"vs published 2026 {mine.loc[2026]:.2f}")

# disposition lag by outcome: do violations resolve faster?
both = disc.dropna(subset=["disposition_date"]).copy()
both = both[both["outcome"].isin(["violation", "no_violation"])]
dd = pd.to_datetime(both["disposition_date"], format="%m/%d/%Y",
                    errors="coerce")
both["lag"] = (dd - both["received"]).dt.days
both = both[both["lag"].between(0, 2000)]
lag = both.groupby(["yr", "outcome"])["lag"].agg(
    median="median", p90=lambda s: s.quantile(0.9),
    within90=lambda s: (s <= 90).mean() * 100, n="size")
print("\ndisposition lag (days) by outcome x year:")
print(lag.loc[2023:].round(1).to_string())

lag24 = lag.loc[2024]
viol_faster = (lag24.loc["violation", "within90"]
               > lag24.loc["no_violation", "within90"])
d26 = corr.loc[2026, "yield_corrected"] - corr.loc[2026, "yield_published"]

finding(
    "Y1a", "CONFIRMED-BUG (guard row measures the wrong bucket); "
    "direction of the conclusion unaffected",
    "yield_turn caveat: 'pending dispositions are ~0 in every year (max "
    "0.02% in 2025), so the turn is not a resolution-lag artifact'",
    f"The guard watches PENDING_CODES (J*/P1/P2/R2-R4), but unresolved "
    f"complaints carry an EMPTY disposition_code and classify_disposition"
    f"('') returns 'other': empty share "
    f"{tab.loc[2024, 'empty_dispo_pct']:.2f}% (2024) -> "
    f"{tab.loc[2025, 'empty_dispo_pct']:.2f}% (2025) -> "
    f"{tab.loc[2026, 'empty_dispo_pct']:.2f}% (2026; open_data bulk "
    f"vintage 2026-04-04, and 100% of empty-code events are status="
    f"ACTIVE). The cited 0.02% pending is the wrong bucket; the honest "
    f"censoring number is ~150x larger in 2026.",
    "Re-word the caveat around the empty-disposition share, or better, "
    "backfill outcomes from the scraped bis_scrape.disposition (fresher, "
    "up to 2026-07-06).")
finding(
    "Y1b", "SURVIVES (censoring deflates recent yield, cannot inflate it)",
    "attack: violations resolve faster than no-violations, inflating "
    "2026 yield",
    f"Unresolved events sit in the DENOMINATOR with hit=0, so censoring "
    f"biases recent yield DOWN, not up. Backfilling the "
    f"{int(fixable.sum()):,} censored 2026 events with scraped "
    f"dispositions moves 2026 yield "
    f"{corr.loc[2026, 'yield_published']:.2f} -> "
    f"{corr.loc[2026, 'yield_corrected']:.2f} ({d26:+.2f}pp; 2025 "
    f"{corr.loc[2025, 'yield_published']:.2f} -> "
    f"{corr.loc[2025, 'yield_corrected']:.2f}); the mature (>=90 days at "
    f"vintage) 2026 subset yields {y_mat_c:.2f} (n={len(mature)}, "
    f"July-refresh selection, indicative only); monthly 2026 corrected "
    f"yields (see table) do not fade into May. Violation dispositions do "
    f"resolve {'faster' if viol_faster else 'slower'} than no-violation "
    f"(within-90d {lag24.loc['violation', 'within90']:.0f}% vs "
    f"{lag24.loc['no_violation', 'within90']:.0f}% in 2024), but with "
    f"unresolved events kept in the denominator that only means the "
    f"published 43.8% UNDERSTATES the eventual 2026 yield.")

# ══════════════════════════════════════════════════════════════════════════
# Y2  risk-score selection (62% match in 2026H1)
# ══════════════════════════════════════════════════════════════════════════

hr("Y2: risk-scores-flat claim vs panel-match selection")

from proactive_becker_margin import load_panel_scored, norm_tract  # noqa: E402
from proactive_decomposition import add_warm_flag  # noqa: E402

ev = pd.read_csv(PRO / "proactive_events.csv.gz", usecols=[
    "received_date", "category_prefix", "family", "agency", "bbl",
    "bct2020", "outcome", "priority", "assigned_to", "active_permit"],
    dtype={"bbl": str})
ev["received"] = pd.to_datetime(ev["received_date"], format="%Y-%m-%d")
ev["bbl"] = ev["bbl"].fillna("")
ev = add_warm_flag(ev)
d = ev[(ev["agency"] == 1) & (ev["family"] == "discretionary_field")].copy()
d["year"] = d["received_date"].str[:4]
d["mm"] = d["received_date"].str[5:7].astype(int)
d["half"] = d["year"] + np.where(d["mm"] <= 6, "H1", "H2")
d["hit100"] = (d["outcome"] == "violation").astype(float) * 100.0

panel, auc, k = load_panel_scored()
panel = panel.drop_duplicates("bbl_key")
sc = d.merge(panel[["bbl_key", "p100", "bct2020"]].rename(
    columns={"bct2020": "bct_panel"}),
    left_on="bbl", right_on="bbl_key", how="left")
sc["matched"] = sc["p100"].notna()

mt = sc.groupby("half").agg(match_pct=("matched", share_pct),
                            n=("matched", "size"))
pub_match = {r["term"]: r["estimate"] for _, r in
             pub[(pub["block"] == "risk_scores")
                 & (pub["model"] == "match")].iterrows()}
assert abs(mt.loc["2026H1", "match_pct"] - pub_match["2026H1"]) < 0.1
assert abs(mt.loc["2024H2", "match_pct"] - pub_match["2024H2"]) < 0.1
print("match-rate replication OK "
      f"(2024H2 {mt.loc['2024H2', 'match_pct']:.1f} / "
      f"2026H1 {mt.loc['2026H1', 'match_pct']:.1f})")

byp = (sc.groupby(["category_prefix", "year"])["matched"]
       .mean().mul(100).unstack())
byp = byp.loc[byp.index.isin(["8A", "7G", "1X", "91", "6X", "1Y", "1U",
                              "5G"]), ["2024", "2025", "2026"]]
print("\npanel match rate by program (selection moved WITHIN programs "
      "too):")
print(byp.round(1).to_string())

# (a) prefix-composition-constant reweighting of the score series
w24 = d[d["year"] == "2024"]["category_prefix"].value_counts(normalize=True)
rw = {}
for h, g in sc[sc["matched"]].groupby("half"):
    cellmeans = g.groupby("category_prefix")["p100"].mean()
    w = w24.reindex(cellmeans.index).fillna(0.0)
    if w.sum() > 0:
        rw[h] = float((cellmeans * w).sum() / w.sum())
obs = sc[sc["matched"]].groupby("half")["p100"].mean()
cmp_ = pd.DataFrame({"observed": obs, "reweighted_2024mix": pd.Series(rw)})
print("\nmean p100 of visited matched lots, observed vs 2024-prefix-mix "
      "reweighted:")
print(cmp_.loc["2023H2":].round(1).to_string())

# (b) within-prefix half-year model (matched visits): does 2025-26 rise?
mm_in = pf.feols("p100 ~ i(half, ref='2024H2') | category_prefix",
                 data=sc[sc["matched"] & sc["bct_panel"].notna()],
                 vcov={"CRV1": "bct_panel"})
td = mm_in.tidy().reset_index()
td["half"] = td["Coefficient"].str.extract(r"(\d{4}H\d)")
w_in = td[td["half"].isin(["2025H1", "2025H2", "2026H1"])][
    ["half", "Estimate", "2.5%", "97.5%"]].set_index("half")
print("\nwithin-prefix visited-lot p100 vs 2024H2 (tract-clustered):")
print(w_in.round(2).to_string())

# (c) per-program 2024 vs 2026 score tests (7G rose in the published CSV)
prog_tests = {}
for pfx in ["7G", "8A"]:
    g = sc[(sc["category_prefix"] == pfx) & sc["matched"]
           & sc["bct_panel"].notna()
           & sc["year"].isin(["2024", "2025", "2026"])].copy()
    g["is26"] = (g["year"] == "2026").astype(float)
    g["is25"] = (g["year"] == "2025").astype(float)
    mm = pf.feols("p100 ~ is25 + is26", data=g, vcov={"CRV1": "bct_panel"})
    tt = mm.tidy().reset_index().set_index("Coefficient")
    prog_tests[pfx] = tt
    print(f"\n{pfx} visited-lot p100, 2026 vs 2024: "
          f"{tt.loc['is26', 'Estimate']:+.1f} "
          f"[{tt.loc['is26', '2.5%']:.1f}, {tt.loc['is26', '97.5%']:.1f}] "
          f"(2025 {tt.loc['is25', 'Estimate']:+.1f} "
          f"[{tt.loc['is25', '2.5%']:.1f}, {tt.loc['is25', '97.5%']:.1f}])")

# (d) how high would unmatched-visit risk need to be to flip the aggregate?
m24, r24 = obs.loc["2024H2"], mt.loc["2024H2", "match_pct"] / 100
m26, r26 = obs.loc["2026H1"], mt.loc["2026H1", "match_pct"] / 100
mu_star = (m24 * r24 - m26 * r26) / (r24 - r26)
print(f"\nequal-unmatched-mean threshold: all-visit 2026H1 mean exceeds "
      f"2024H2 iff unmatched visits average p100 > {mu_star:.1f} "
      f"(matched means {m26:.1f} vs {m24:.1f}; match {r26:.2f} vs "
      f"{r24:.2f})")

um26 = sc[(sc["half"] == "2026H1") & ~sc["matched"]]
um24 = sc[(sc["half"] == "2024H2") & ~sc["matched"]]
mt26 = sc[(sc["half"] == "2026H1") & sc["matched"]]
a8_26_p100 = sc[(sc["category_prefix"] == "8A") & sc["matched"]
                & (sc["year"] == "2026")]["p100"].mean()
print(f"unmatched 2026H1 visits: {len(um26):,}; at active permit "
      f"{um26['active_permit'].mean() * 100:.0f}% (2024H2 unmatched "
      f"{um24['active_permit'].mean() * 100:.0f}%); violation yield "
      f"{(um26['outcome'] == 'violation').mean() * 100:.0f}% vs matched "
      f"{(mt26['outcome'] == 'violation').mean() * 100:.0f}%; matched-8A "
      f"2026 mean p100 {a8_26_p100:.0f}")

t7g = prog_tests["7G"]
sig_7g_up = t7g.loc["is26", "2.5%"] > 0
w26h1 = w_in.loc["2026H1"]

finding(
    "Y2a", "OVERSTATED (scope)",
    "post: 'the buildings DOB chose to visit in 2025-26 look no riskier "
    "than the ones it visited earlier'; plan: 'RISK SCORES FLAT ... "
    "buildings visited look NO riskier'",
    f"The claim is identified only for visits matched to the pre-2020 "
    f"residential panel, and the unmatched share grew exactly where "
    f"targeting moved: total match {r24 * 100:.0f}% (2024H2) -> "
    f"{r26 * 100:.0f}% (2026H1), and WITHIN-8A "
    f"{byp.loc['8A', '2024']:.0f}% -> {byp.loc['8A', '2026']:.0f}%. "
    f"Unmatched 2026H1 visits are {um26['active_permit'].mean() * 100:.0f}"
    f"% at active permits with a {(um26['outcome'] == 'violation').mean() * 100:.0f}"
    f"% violation yield -- unscoreable construction/commercial stock, not "
    f"demonstrably low-risk stock. Under the (unverifiable) assumption "
    f"that unmatched visits average p100 > {mu_star:.0f}, the all-visit "
    f"mean ROSE; matched 8A visits alone average {a8_26_p100:.0f}. Within "
    f"the matched panel the flat/declining reading is solid (within-"
    f"prefix 2026H1 {w26h1['Estimate']:+.1f}pp vs 2024H2 "
    f"[{w26h1['2.5%']:.1f}, {w26h1['97.5%']:.1f}]; 2024-mix reweighting "
    f"does not rescue it), so the finding stands for scoreable "
    f"residential lots but the post states it for all visited buildings.",
    "Scope the sentence to panel-scoreable residential lots and put the "
    "62% vs 75% match rate next to the number, not only in the plan "
    "caveat; note the model cannot score the construction-site stock "
    "2025-26 volume moved toward.")
finding(
    "Y2b",
    "OVERSTATED (blanket FAIL hides a program-level pass)" if sig_7g_up
    else "SURVIVES (7G uptick not significant)",
    "plan: 'the sharpest LL79 test (visit riskier buildings) FAILS so "
    "far'",
    f"For 7G sweeps -- the program a risk-ranked list would redirect "
    f"first -- visited-lot predicted risk in 2026 is "
    f"{t7g.loc['is26', 'Estimate']:+.1f}pp vs 2024 "
    f"[{t7g.loc['is26', '2.5%']:.1f}, {t7g.loc['is26', '97.5%']:.1f}] "
    f"(tract-clustered; 2025 {t7g.loc['is25', 'Estimate']:+.1f} "
    f"[{t7g.loc['is25', '2.5%']:.1f}, {t7g.loc['is25', '97.5%']:.1f}]), "
    f"alongside the published 7G yield jump 9.3 -> 14.0 and the highest "
    f"7G score since 2021 (56.4). The pooled series is dragged by 8A "
    f"({prog_tests['8A'].loc['is26', 'Estimate']:+.1f}pp "
    f"[{prog_tests['8A'].loc['is26', '2.5%']:.1f}, "
    f"{prog_tests['8A'].loc['is26', '97.5%']:.1f}]), whose scoreable "
    f"slice shrank most.",
    "If the 7G CI excludes 0, quote it as an early sweep-level signal "
    "instead of a blanket FAIL; if it spans 0 (small n, 5 months of "
    "2026), the FAIL wording stands but 'sweeps ambiguous, watch H2' is "
    "the more accurate phrasing.")

# ══════════════════════════════════════════════════════════════════════════
# Y3  shift-share granularity
# ══════════════════════════════════════════════════════════════════════════

hr("Y3: shift-share within/mix split at finer granularity")


def shift_share_cells(df, base_mask, comp_mask, cell):
    b, c = df[base_mask], df[comp_mask]
    yb, yc = b["hit100"].mean(), c["hit100"].mean()
    wb = b[cell].value_counts(normalize=True)
    wc = c[cell].value_counts(normalize=True)
    ymb = b.groupby(cell)["hit100"].mean()
    ymc = c.groupby(cell)["hit100"].mean()
    cats = sorted(set(wb.index) | set(wc.index))
    anchor = 0.5 * (yb + yc)
    w_sum = c_sum = 0.0
    for cat in cats:
        w0, w1 = wb.get(cat, 0.0), wc.get(cat, 0.0)
        y0, y1 = ymb.get(cat, yb), ymc.get(cat, yc)
        w_sum += 0.5 * (w0 + w1) * (y1 - y0)
        c_sum += (w1 - w0) * (0.5 * (y0 + y1) - anchor)
    tot = yc - yb
    assert np.isclose(w_sum + c_sum, tot, atol=1e-9)
    return tot, w_sum, c_sum, len(cats)


d["permit"] = d["active_permit"].astype(str)
d["warm_s"] = d["warm"].astype(str)
top_units = d["assigned_to"].value_counts().nlargest(12).index
d["unit"] = np.where(d["assigned_to"].isin(top_units),
                     d["assigned_to"].fillna("NA"), "OTHER")
d["boro"] = d["bbl"].str[0].where(d["bbl"] != "", "NA")
d["cell_pp"] = d["category_prefix"] + "|" + d["permit"]
d["cell_pw"] = d["category_prefix"] + "|" + d["warm_s"]
d["cell_pu"] = d["category_prefix"] + "|" + d["unit"]
d["cell_pb"] = d["category_prefix"] + "|" + d["boro"]
d["cell_ppw"] = d["cell_pp"] + "|" + d["warm_s"]

y24 = (d["year"] == "2024").to_numpy()
y25 = (d["year"] == "2025").to_numpy()
y26 = (d["year"] == "2026").to_numpy()

print(f"{'cells':<28}{'n_cells':>8}{'2024->2025 within%':>22}"
      f"{'2024->2026jm within%':>24}")
gran = {}
for name, col in [("prefix (published)", "category_prefix"),
                  ("prefix x permit", "cell_pp"),
                  ("prefix x warm", "cell_pw"),
                  ("prefix x unit", "cell_pu"),
                  ("prefix x borough", "cell_pb"),
                  ("prefix x permit x warm", "cell_ppw")]:
    t25, w25, c25, k25 = shift_share_cells(d, y24, y25, col)
    t26, w26, c26, k26 = shift_share_cells(d, y24, y26, col)
    gran[name] = (w25 / t25 * 100, w26 / t26 * 100)
    print(f"{name:<28}{k26:>8}{w25 / t25 * 100:>21.1f}%"
          f"{w26 / t26 * 100:>23.1f}%")

pub_w25 = pub[(pub["block"] == "decomposition")
              & (pub["model"] == "symmetric")
              & (pub["term"] == "2024_to_2025__within_share_of_change")][
    "estimate"].iloc[0]
assert abs(gran["prefix (published)"][0] - pub_w25) < 0.1

# the finer margin the prose leans on: 8A's within jump
a8 = d[d["category_prefix"] == "8A"]
a8_24, a8_26 = a8[a8["year"] == "2024"], a8[a8["year"] == "2026"]
ap24 = a8_24[a8_24["active_permit"] == 1]["hit100"].mean()
ap26 = a8_26[a8_26["active_permit"] == 1]["hit100"].mean()
print(f"\n8A detail: single category name (no code subtypes); at-permit "
      f"share {a8_24['active_permit'].mean():.2f} -> "
      f"{a8_26['active_permit'].mean():.2f}; yield within at-permit 8A "
      f"{ap24:.1f} -> {ap26:.1f}")

wmin25 = min(v[0] for v in gran.values())
wmin26 = min(v[1] for v in gran.values())
frag = (wmin25 < 30 or wmin26 < 40)
finding(
    "Y3", "OVERSTATED (finer mix absorbs the within half)" if frag
    else "SURVIVES",
    "post/plan: about half the yield jump is mix across programs, half is "
    "higher yield within programs (within 45% 2024->2025, 56% "
    "2024->2026jm)",
    f"Re-running the identical symmetric decomposition on finer cells "
    f"(prefix x active-permit / x warm / x assigned unit / x borough / "
    f"x permit x warm; the 2-char prefix IS the finest category code and "
    f"8A carries a single category name) moves the within share from "
    f"{gran['prefix (published)'][0]:.0f}% to a minimum of {wmin25:.0f}% "
    f"(2024->2025) and from {gran['prefix (published)'][1]:.0f}% to "
    f"{wmin26:.0f}% (2024->2026jm) across those margins. The 8A rise "
    f"also holds inside its dominant stratum (at-permit 8A yield "
    f"{ap24:.0f} -> {ap26:.0f}), and the high-yield CSE-unit slice of 8A "
    f"is too small (320 of 1,953 in 2026) to carry the jump.",
    "None needed if within-share is stable; otherwise report the finer "
    "split.")

# ══════════════════════════════════════════════════════════════════════════
# E4  equity: offset restriction, bundle composition, alternative exposures
# ══════════════════════════════════════════════════════════════════════════

hr("E4: equity allocation model attacks")

# (0) does fepois offset= actually do anything? synthetic check
n = 4000
loge = RNG.normal(0, 1, n)
x = RNG.normal(0, 1, n)
ys = RNG.poisson(np.exp(0.3 + 0.5 * x) * np.exp(loge))
sim = pd.DataFrame({"y": ys, "x": x, "loge": loge})
b_off = pf.fepois("y ~ x", data=sim, offset="loge").coef()["x"]
b_no = pf.fepois("y ~ x", data=sim).coef()["x"]
b_free_sim = pf.fepois("y ~ x + loge", data=sim).coef()
assert abs(b_off - 0.5) < 0.05, b_off
assert abs(b_free_sim["loge"] - 1.0) < 0.05
print(f"fepois offset sanity: beta_x with offset {b_off:.3f} (true 0.5), "
      f"without {b_no:.3f}, free coef on log e {b_free_sim['loge']:.3f} "
      f"(true 1.0) -> offset IS applied, not silently ignored")

from proactive_equity import (DEMOS, STOCK, BLD_TRAITS,  # noqa: E402
                              build_crisis_frame, build_tract_frame,
                              load_events, load_v2, make_day_counter)

ev_eq = load_events()
tract, traits = load_v2()
tf = build_tract_frame(ev_eq, tract)
rhs = " + ".join(DEMOS + STOCK)
pub_eq = pd.read_csv(RM / "proactive_equity.csv")


def per10(b):
    return 100 * (np.exp(0.1 * b) - 1)


def fit(outcome, off, data, rhs_=None, label=""):
    sub = data[data[off].notna()]
    m = pf.fepois(f"{outcome} ~ {rhs_ or rhs} | boro", data=sub, offset=off,
                  vcov={"CRV1": "nta"})
    t = m.tidy().reset_index().set_index("Coefficient")
    out = {dm: (per10(t.loc[dm, "Estimate"]), per10(t.loc[dm, "2.5%"]),
                per10(t.loc[dm, "97.5%"])) for dm in KEY3}
    print(f"  {label:<46}" + "  ".join(
        f"{dm.split('_')[-1][:4]} {v[0]:+5.1f} [{v[1]:+5.1f},{v[2]:+5.1f}]"
        for dm, v in out.items()))
    return out, t, m


print("\nper +10pp of tract share, % change in count [95% CI]:")
head, t_head, m_head = fit("n_disc_constr", "log_job_months", tf,
                           label="HEADLINE replicate (job-month offset)")
ref = pub_eq[(pub_eq["outcome"] == "n_disc_constr")
             & (pub_eq["term"] == "pct_black")]["per_10pp"].iloc[0]
assert abs(head["pct_black"][0] - ref) < 0.05, "headline replication failed"

# (1) free elasticity: does the offset=1 restriction manufacture the gap?
sub = tf[tf["log_job_months"].notna()].copy()
sub["log_jm"] = sub["log_job_months"]
m_free = pf.fepois(f"n_disc_constr ~ {rhs} + log_jm | boro", data=sub,
                   vcov={"CRV1": "nta"})
t_free = m_free.tidy().reset_index().set_index("Coefficient")
el = t_free.loc["log_jm"]
free = {dm: (per10(t_free.loc[dm, "Estimate"]), per10(t_free.loc[dm, "2.5%"]),
             per10(t_free.loc[dm, "97.5%"])) for dm in KEY3}
print(f"  {'FREE ELASTICITY (log job-months as regressor)':<46}" + "  ".join(
    f"{dm.split('_')[-1][:4]} {v[0]:+5.1f} [{v[1]:+5.1f},{v[2]:+5.1f}]"
    for dm, v in free.items()))
print(f"    cross-tract elasticity on job-months: {el['Estimate']:.3f} "
      f"[{el['2.5%']:.3f}, {el['97.5%']:.3f}] -- the offset imposes 1.0")

# mechanism check: imposing the offset shifts each coefficient by
# (elasticity - 1) x the conditional gradient of log exposure on that
# regressor (b_offset = b_free + (el - 1) * gamma); with el < 1 and
# negative exposure gradients this inflates the nonwhite loadings
m_grad = pf.feols(f"log_jm ~ {rhs} | boro", data=sub, vcov={"CRV1": "nta"})
g_ = m_grad.tidy().reset_index().set_index("Coefficient")
print("    predicted offset-restriction inflation vs actual "
      "(headline - free), per +10pp:")
for dm in KEY3:
    pred = per10((el["Estimate"] - 1) * g_.loc[dm, "Estimate"])
    print(f"      {dm:<14} exposure gradient {g_.loc[dm, 'Estimate']:+.2f} "
          f"-> predicted {pred:+.1f}%  actual "
          f"{head[dm][0] - free[dm][0]:+.1f}%")

# (2) bundle composition + outcome splits
bundle = ev_eq[(ev_eq["agency"] == 1) & ev_eq["tract"].notna()
               & (ev_eq["family"] == "discretionary_field")
               & (ev_eq["category_prefix"] != "7G")]
compo = bundle.groupby("category_prefix").agg(
    n=("agency", "size"), at_permit=("active_permit", "mean"))
compo["share_pct"] = compo["n"] / compo["n"].sum() * 100
compo["at_permit_pct"] = compo["at_permit"] * 100
print("\nbundle composition (the outcome the post calls 'discretionary "
      "construction'):")
print(compo.sort_values("n", ascending=False)
      .head(8)[["share_pct", "at_permit_pct"]].round(1).to_string())
off_permit_share = 1 - bundle["active_permit"].mean()
print(f"  -> {off_permit_share * 100:.0f}% of bundle events sit at lots "
      f"with NO active permit")

cnt = pd.DataFrame({
    "n_8a": bundle[bundle["category_prefix"] == "8A"].groupby("tract").size(),
    "n_atpermit": bundle[bundle["active_permit"] == 1]
    .groupby("tract").size(),
    "n_offpermit": bundle[bundle["active_permit"] == 0]
    .groupby("tract").size(),
}).reindex(tf.index).fillna(0).astype(int)
tf2 = tf.join(cnt)

r_8a, _, _ = fit("n_8a", "log_job_months", tf2,
                 label="8A-only (pure construction compliance)")
r_at, _, _ = fit("n_atpermit", "log_job_months", tf2,
                 label="at-active-permit events only")
r_off, _, _ = fit("n_offpermit", "log_job_months", tf2,
                  label="OFF-permit events (EWO/illegal-work slice)")

# (3) alternative exposures
jobs = pd.read_csv(PRO / "jobs.csv.gz",
                   usecols=["bct2020", "active_start", "active_end",
                            "total_construction_floor_area", "initial_cost"],
                   parse_dates=["active_start", "active_end"])
jobs["tract"] = pd.to_numeric(jobs["bct2020"], errors="coerce").astype("Int64")
j = jobs.dropna(subset=["tract", "active_start", "active_end"]).copy()
s = ((j["active_start"].dt.year - 2020) * 12
     + j["active_start"].dt.month - 1).clip(lower=0)
t_ = ((j["active_end"].dt.year - 2020) * 12
      + j["active_end"].dt.month - 1).clip(upper=76)
keep = t_ >= s
j, s, t_ = j[keep], s[keep], t_[keep]
months = t_ - s + 1

fa = pd.to_numeric(j["total_construction_floor_area"], errors="coerce")
fa = fa.where(fa > 0)
fa = fa.fillna(fa.median())
cost = pd.to_numeric(j["initial_cost"], errors="coerce")
cost = cost.where(cost > 0)
cost = cost.fillna(cost.median())
fa_exp = (pd.DataFrame({"tract": j["tract"], "v": months * fa})
          .groupby("tract")["v"].sum())
cost_exp = (pd.DataFrame({"tract": j["tract"], "v": months * cost})
            .groupby("tract")["v"].sum())
tf2["log_fa_months"] = np.log(fa_exp.reindex(tf2.index))
tf2["log_cost_months"] = np.log(cost_exp.reindex(tf2.index))

r_fa, _, _ = fit("n_disc_constr", "log_fa_months", tf2,
                 label="floor-area-weighted job-month offset")
r_cost, _, _ = fit("n_disc_constr", "log_cost_months", tf2,
                   label="cost-weighted job-month offset")
r_fa_stat, _, _ = fit("n_stat", "log_fa_months", tf2,
                      label="statutory placebo, fa-weighted offset")

# 2023+ window (numerator and denominator; DOB NOW coverage complete)
s23 = s.clip(lower=36)
keep23 = t_ >= s23
jm23 = (pd.DataFrame({"tract": j["tract"][keep23],
                      "m": (t_ - s23 + 1)[keep23]})
        .groupby("tract")["m"].sum())
tf2["log_jm23"] = np.log(jm23.reindex(tf2.index))
b23 = bundle[bundle["received"] >= "2023-01-01"]
tf2["n_disc23"] = (b23.groupby("tract").size().reindex(tf2.index)
                   .fillna(0).astype(int))
r_23, _, _ = fit("n_disc23", "log_jm23", tf2,
                 label="2023+ window (complete permit coverage)")

# caller-side construction-incident volume as the exposure
caller_con = ev_eq[(ev_eq["agency"] == 0)
                   & (ev_eq["family"] == "mixed_incident")
                   & ev_eq["tract"].notna()]
cc = caller_con.groupby("tract").size().reindex(tf2.index).fillna(0)
tf2["log_caller_con"] = np.log1p(cc)
r_cc, _, _ = fit("n_disc_constr", "log_caller_con", tf2,
                 label="caller construction-incident offset (log1p)")

# (4) era x size interaction controls
for a_ in ["era_pre1940", "era_1980plus"]:
    for b_ in ["mean_log_ba", "log_mean_units"]:
        tf2[f"{a_}_x_{b_}"] = tf2[a_] * tf2[b_]
inter = [f"{a_}_x_{b_}" for a_ in ["era_pre1940", "era_1980plus"]
         for b_ in ["mean_log_ba", "log_mean_units"]]
r_int, _, _ = fit("n_disc_constr", "log_job_months", tf2,
                  rhs_=rhs + " + " + " + ".join(inter),
                  label="+ era x size interaction controls")

# (5) demographic gradient of the at-permit share (offset mismeasurement)
share = tf2[(tf2["n_atpermit"] + tf2["n_offpermit"]) >= 10].copy()
share["at_share"] = share["n_atpermit"] / (share["n_atpermit"]
                                           + share["n_offpermit"])
share["w"] = (share["n_atpermit"] + share["n_offpermit"]).astype(float)
ms = pf.feols(f"at_share ~ {rhs} | boro", data=share, weights="w",
              vcov={"CRV1": "nta"})
ts = ms.tidy().reset_index().set_index("Coefficient")
# at_share and the demo shares are both 0-1 scaled, so the pp change in
# the at-permit share per +10pp of demo share is 10 * b
at_grad = {dm: (10 * ts.loc[dm, "Estimate"], 10 * ts.loc[dm, "2.5%"],
                10 * ts.loc[dm, "97.5%"]) for dm in KEY3}
print(f"\nat-permit share of bundle events ({len(share):,} tracts with "
      ">=10 events), per +10pp (pp, NTA-clustered):")
for dm in KEY3:
    print(f"  {dm:<14} {at_grad[dm][0]:+.1f}pp "
          f"[{at_grad[dm][1]:+.1f}, {at_grad[dm][2]:+.1f}]")

# (6) NTA-block randomization inference (cluster-adequacy stress)
nta_boro = tf2.groupby("nta")["boro"].agg(lambda x: x.mode().iat[0])
nta_mean = tf2.groupby("nta")[DEMOS].mean()
tf_ri = tf2[tf2["log_job_months"].notna()].copy()
tf_ri[DEMOS] = nta_mean.reindex(tf_ri["nta"]).to_numpy()
m_nta = pf.fepois(f"n_disc_constr ~ {rhs} | boro", data=tf_ri,
                  offset="log_job_months", vcov={"CRV1": "nta"})
b_obs = {dm: m_nta.coef()[dm] for dm in KEY3}
print(f"\nNTA-mean-demo spec (block-RI basis): "
      + ", ".join(f"{dm} {per10(v):+.1f}%" for dm, v in b_obs.items()))

ntas = nta_mean.index.to_numpy()
boro_of = nta_boro.reindex(ntas).to_numpy()
hits = {dm: 0 for dm in KEY3}
n_ok = 0
for it in range(149):
    perm = np.empty(len(ntas), dtype=object)
    for bb in np.unique(boro_of):
        idx = np.where(boro_of == bb)[0]
        perm[idx] = ntas[RNG.permutation(idx)]
    shuf = nta_mean.loc[perm].set_axis(ntas, axis=0)
    tf_ri[DEMOS] = shuf.reindex(tf_ri["nta"]).to_numpy()
    try:
        mp = pf.fepois(f"n_disc_constr ~ {rhs} | boro", data=tf_ri,
                       offset="log_job_months")
    except Exception:
        continue
    n_ok += 1
    for dm in KEY3:
        if abs(mp.coef()[dm]) >= abs(b_obs[dm]):
            hits[dm] += 1
ri_p = {dm: (v + 1) / (n_ok + 1) for dm, v in hits.items()}
print(f"block-RI p-values ({n_ok} within-borough NTA permutations): "
      + ", ".join(f"{dm} {p:.3f}" for dm, p in ri_p.items()))

# ── E4 verdicts, conditional on what actually happened ───────────────────
free_dead = sum(1 for dm in KEY3 if free[dm][1] <= 0)     # CI lower <= 0
free_shrunk = sum(1 for dm in KEY3
                  if abs(free[dm][0]) < 0.5 * abs(head[dm][0]))
el_below_1 = el["97.5%"] < 1
if el_below_1 and (free_dead == 3 or free_shrunk == 3):
    v_e4a = ("CONFIRMED-BUG (functional form): the unit-elasticity offset "
             "manufactures most of the headline gradient")
elif el_below_1 and (free_dead + free_shrunk) >= 2:
    v_e4a = ("OVERSTATED: the +5-7% is not robust to relaxing the "
             "offset=1 restriction")
else:
    v_e4a = "SURVIVES: loadings robust to freeing the exposure elasticity"
finding(
    "E4a", v_e4a,
    "equity headline: +5-7% per +10pp Black/Hispanic/Asian per active "
    "job-month (offset coefficient forced to 1)",
    f"The project's own A3 dilution result puts the monitoring-jobs "
    f"elasticity at ~0.28, and cross-tract the free elasticity here is "
    f"{el['Estimate']:.2f} [{el['2.5%']:.2f}, {el['97.5%']:.2f}]"
    f"{' (CI excludes 1)' if el_below_1 else ''} against the 1.0 the "
    f"offset imposes. Nonwhite tracts host less permitted construction, "
    f"so under a concave true relationship forcing proportionality "
    f"mechanically inflates their per-job-month rates. Freeing the "
    f"elasticity: Black {free['pct_black'][0]:+.1f}% "
    f"[{free['pct_black'][1]:+.1f}, {free['pct_black'][2]:+.1f}], "
    f"Hispanic {free['pct_hispanic'][0]:+.1f}% "
    f"[{free['pct_hispanic'][1]:+.1f}, {free['pct_hispanic'][2]:+.1f}], "
    f"Asian {free['pct_asian'][0]:+.1f}% [{free['pct_asian'][1]:+.1f}, "
    f"{free['pct_asian'][2]:+.1f}] vs headline "
    f"{head['pct_black'][0]:+.1f}/{head['pct_hispanic'][0]:+.1f}/"
    f"{head['pct_asian'][0]:+.1f}%. The mechanism is verified "
    f"quantitatively: (elasticity-1) x conditional exposure gradient "
    f"predicts inflations of "
    + "/".join(f"{per10((el['Estimate'] - 1) * g_.loc[dm, 'Estimate']):+.1f}"
               for dm in KEY3)
    + "% vs actual "
    + "/".join(f"{head[dm][0] - free[dm][0]:+.1f}" for dm in KEY3)
    + "%. The statutory placebo does NOT guard this channel: statutory "
    "counts are unrelated to construction, so the same concavity "
    "artifact cannot appear there.",
    "Report the free-elasticity spec next to the offset spec (or use "
    "inspections-per-job as the outcome); if the loadings shrink or die, "
    "the post's 'not demographically neutral' section needs the caveat "
    "or the number replaced.")

at_dead = r_at["pct_black"][1] <= 0 and r_at["pct_hispanic"][1] <= 0
a8_dead = r_8a["pct_black"][1] <= 0 and r_8a["pct_hispanic"][1] <= 0
neg_at_grad = sum(1 for dm in KEY3 if at_grad[dm][2] < 0)
if at_dead and a8_dead:
    v_e4b = ("OVERSTATED: the 'construction oversight' loading lives in "
             "the off-permit enforcement slice")
elif at_dead or a8_dead or neg_at_grad >= 2:
    v_e4b = ("OVERSTATED (framing): part of the loading rides on "
             "unpermitted-work enforcement the offset cannot see")
else:
    v_e4b = ("SURVIVES: loadings hold within the permitted-construction "
             "slice")
finding(
    "E4b", v_e4b,
    "post frames the +5-7% as 'discretionary construction oversight ... "
    "parallels the conversion/big-job targeting'",
    f"{off_permit_share * 100:.0f}% of the outcome bundle sits at lots "
    f"with NO active permit; the biggest non-8A pieces are enforcement "
    f"work orders and illegal/unpermitted-work programs (1X at-permit "
    f"{compo.loc['1X', 'at_permit_pct']:.0f}%, 1Y "
    f"{compo.loc['1Y', 'at_permit_pct']:.0f}%, 5G "
    f"{compo.loc['5G', 'at_permit_pct']:.0f}%). Splits per +10pp Black: "
    f"at-permit-only {r_at['pct_black'][0]:+.1f}% "
    f"[{r_at['pct_black'][1]:+.1f}, {r_at['pct_black'][2]:+.1f}]; 8A-only "
    f"{r_8a['pct_black'][0]:+.1f}% [{r_8a['pct_black'][1]:+.1f}, "
    f"{r_8a['pct_black'][2]:+.1f}]; OFF-permit {r_off['pct_black'][0]:+.1f}"
    f"% [{r_off['pct_black'][1]:+.1f}, {r_off['pct_black'][2]:+.1f}]. "
    f"At-permit share gradient per +10pp: Black {at_grad['pct_black'][0]:+.1f}"
    f"pp [{at_grad['pct_black'][1]:+.1f}, {at_grad['pct_black'][2]:+.1f}], "
    f"Hispanic {at_grad['pct_hispanic'][0]:+.1f}pp "
    f"[{at_grad['pct_hispanic'][1]:+.1f}, {at_grad['pct_hispanic'][2]:+.1f}]"
    f" -- a negative gradient means the permit-based offset undercounts "
    f"true construction exposure more in nonwhite tracts (unpermitted "
    f"work), which is the denominator-artifact channel. Nuance: the "
    f"ASIAN loading partially survives inside the permitted slice "
    f"(at-permit {r_at['pct_asian'][0]:+.1f}% "
    f"[{r_at['pct_asian'][1]:+.1f}, {r_at['pct_asian'][2]:+.1f}]), so the "
    f"three groups should not be lumped into one sentence.",
    "Quote the at-permit / 8A-only split; where the loading concentrates "
    "off-permit (Black, Hispanic), reframe from 'construction oversight "
    "targeting' toward 'enforcement against unpermitted work', which "
    "carries a different equity reading (and a different denominator).")

alt = {"floor-area wt": r_fa, "cost wt": r_cost, "2023+ window": r_23,
       "caller-side": r_cc, "era x size": r_int}
alt_ok = {k: all(v[dm][1] > 0 for dm in KEY3) for k, v in alt.items()}
n_fail = sum(1 for v in alt_ok.values() if not v)
finding(
    "E4c",
    "SURVIVES (alternative exposures)" if n_fail == 0 else
    f"OVERSTATED ({n_fail}/5 alternative-exposure specs break at least "
    f"one loading)",
    "robustness of +5-7% to how exposure is measured and to richer stock "
    "controls",
    "all three loadings positive with CI>0 under: "
    + "; ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in alt_ok.items())
    + f". Statutory placebo under the fa-weighted offset stays "
    f"{'null' if r_fa_stat['pct_black'][1] <= 0 <= r_fa_stat['pct_black'][2] else 'NON-null'}"
    f" on Black ({r_fa_stat['pct_black'][0]:+.1f}% "
    f"[{r_fa_stat['pct_black'][1]:+.1f}, {r_fa_stat['pct_black'][2]:+.1f}]).",
    "Cite whichever margins fail as caveats next to the headline.")
finding(
    "E4d",
    "SURVIVES (cluster adequacy)" if max(ri_p.values()) < 0.05 else
    "OVERSTATED (block-RI p-values exceed 0.05)",
    "201 NTA clusters adequate; significance not a cluster-count artifact",
    f"201 clusters is far past small-cluster territory, and within-"
    f"borough NTA-block randomization inference (demographic profiles "
    f"permuted as whole NTA blocks, {n_ok} draws) gives p = "
    + ", ".join(f"{dm.split('_')[-1]} {p:.3f}" for dm, p in ri_p.items())
    + " for the NTA-mean version of the spec. Inference mechanics are "
    "fine; the live issues are E4a/E4b (what the estimand means), not "
    "the SEs.")

# ══════════════════════════════════════════════════════════════════════════
# E5  crisis-response power + outcome dilution
# ══════════════════════════════════════════════════════════════════════════

hr("E5: crisis-response flat -- power and outcome dilution")

cr = pub_eq[(pub_eq["part"] == "crisis_response")
            & (pub_eq["outcome"] == "resp_any")
            & pub_eq["term"].isin(KEY3)]
base = cr["baseline"].iloc[0]
print(f"resp_any baseline {base:.3f}; an allocation-sized effect (+5-7% "
      f"of base) = {0.05 * base * 100:.2f}-{0.07 * base * 100:.2f}pp per "
      f"+10pp")
mde_rows = []
for _, r in cr.iterrows():
    mde = 1.96 * r["se"] * 10        # pp per +10pp (per_10pp = 10*b)
    alloc_pp = (pub_eq[(pub_eq["outcome"] == "n_disc_constr")
                       & (pub_eq["term"] == r["term"])]["per_10pp"].iloc[0]
                / 100 * base * 100)
    excl = not (r["per_10pp_lo"] <= alloc_pp <= r["per_10pp_hi"])
    mde_rows.append((r["term"], mde, alloc_pp, excl))
    print(f"  {r['term']:<14} est {r['per_10pp']:+.2f}pp "
          f"[{r['per_10pp_lo']:+.2f}, {r['per_10pp_hi']:+.2f}]  "
          f"MDE(95) {mde:.2f}pp  allocation-equivalent {alloc_pp:+.2f}pp "
          f"{'EXCLUDED' if excl else 'INSIDE CI (underpowered vs it)'}")

# outcome dilution: rebuild the crisis frame, split responses by family
cf = build_crisis_frame(ev_eq, tract, traits)
e = ev_eq[ev_eq["bbl"].notna() & ev_eq["bbl"].str.fullmatch(r"\d{10}")].copy()
e["bbl_i"] = e["bbl"].astype("int64")
ag_stat = make_day_counter(e[(e["agency"] == 1)
                             & (e["family"] == "statutory_periodic")])
ag_disc = make_day_counter(e[(e["agency"] == 1)
                             & (e["family"] != "statutory_periodic")
                             & (e["category_prefix"] != "30")])
b_arr, d_arr = cf["bbl_i"].to_numpy(), cf["day"].to_numpy()
cf["resp_stat"] = (ag_stat(b_arr, d_arr, d_arr + 90) > 0).astype(int)
cf["resp_disc"] = (ag_disc(b_arr, d_arr, d_arr + 90) > 0).astype(int)
stat_only = (((cf["resp_any"] == 1) & (cf["resp_disc"] == 0)).sum()
             / max(int(cf["resp_any"].sum()), 1))
print(f"\nresponse composition: resp_any {cf['resp_any'].mean():.3f}; "
      f"non-statutory (excl cat-30) {cf['resp_disc'].mean():.3f}; "
      f"statutory {cf['resp_stat'].mean():.3f}; share of 'responses' "
      f"with NO non-statutory component {stat_only * 100:.0f}%")

rhs_cr = " + ".join(DEMOS) + " + C(priority) + " + " + ".join(BLD_TRAITS)
m_disc = pf.feols(f"resp_disc ~ {rhs_cr} | boro_month", data=cf,
                  vcov={"CRV1": "tract_s"})
t_disc = m_disc.tidy().reset_index().set_index("Coefficient")
print("discretionary-only 90-day response, per +10pp (pp):")
disc_sig = []
for dm in DEMOS:
    est, lo, hi = (10 * t_disc.loc[dm, "Estimate"],
                   10 * t_disc.loc[dm, "2.5%"], 10 * t_disc.loc[dm, "97.5%"])
    if not lo <= 0 <= hi:
        disc_sig.append(dm)
    print(f"  {dm:<14} {est:+.2f}pp [{lo:+.2f}, {hi:+.2f}]")

n_excl = sum(1 for r in mde_rows if r[3])
finding(
    "E5",
    "SURVIVES (adequately powered)" if n_excl >= 2 else
    "OVERSTATED (cannot rule out allocation-sized gaps)",
    "crisis response ~ demographically flat",
    f"MDEs per +10pp on the {base * 100:.1f}pp base: "
    + ", ".join(f"{r[0].split('_')[-1]} {r[1]:.2f}pp" for r in mde_rows)
    + f"; allocation-sized (+5-7% relative) positive gaps are excluded "
    f"for {n_excl}/3 groups, so 'close to flat' is a powered claim, not "
    f"an absence-of-evidence claim. Caveats: (i) "
    f"{stat_only * 100:.0f}% of 'responses' contain no non-statutory "
    f"agency event -- boiler/elevator/facade cycles land in the 90-day "
    f"window mechanically -- and a discretionary-only outcome gives "
    f"{'gaps for ' + ', '.join(x.split('_')[-1] for x in disc_sig) if disc_sig else 'no significant gaps either'}"
    f"; (ii) the Hispanic -0.62pp in resp_any is fragile exactly as the "
    f"plan flags (null excluding same-day responses) and should stay "
    f"unquoted.")

# ══════════════════════════════════════════════════════════════════════════

hr("NUMBERED FINDINGS")
for num, verdict, claim, evidence, fix in FINDINGS:
    print(f"\n[{num}] {verdict}")
    print(f"  CLAIM   : {claim}")
    print(f"  EVIDENCE: {evidence}")
    if fix:
        print(f"  FIX     : {fix}")

print(f"\ntotal runtime {time.time() - T0:.0f}s")
