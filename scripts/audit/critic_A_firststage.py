"""Critic script A: first-stage inference, geography, LOO contamination, decomposition."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/scripts")
sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
import config
import pyfixest as pf

t0 = time.time()
PANEL = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/data/analysis/master_panel.csv"
usecols = ["complaint_number","disposition_code","date_entered","community_board","borough",
           "priority","category_description","assigned_to","inspector_badge","received_time",
           "bbl","boro_block","year_month","violation_found","loo_strictness",
           "numfloors","yearbuilt","unitsres","assesstot","bldgarea"]
df = pd.read_csv(PANEL, usecols=usecols, low_memory=False)
df["complaint_number"] = df["complaint_number"].astype(str)
print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

# NTA merge
import sqlite3
conn = sqlite3.connect(str(config.DB_PATH))
nta = pd.read_sql_query("SELECT complaint_number, nta FROM complaint_nta", conn)
conn.close()
nta["complaint_number"] = nta["complaint_number"].astype(str)
df = df.merge(nta, on="complaint_number", how="left")
print(f"NTA matched {df['nta'].notna().mean():.3f}", flush=True)

# FE cells
cat = df["category_description"].fillna("UNK")
unit = df["assigned_to"].fillna("UNK")
ym = df["year_month"].fillna("UNK")
df["cell4"] = (cat + "|" + unit + "|" + ym)
df["cell4_nta"] = df["cell4"] + "|" + df["nta"].fillna("UNK")
df["cell4_cb"] = df["cell4"] + "|" + df["community_board"].fillna("UNK").astype(str)

# kitchen sink cell (spec 6): + time_block + priority + day_of_week
ent = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce")
dow = ent.dt.dayofweek.fillna(-1).astype(int).astype(str)
def parse_hour(t):
    if not isinstance(t, str) or t in ("0:00","00:00"): return np.nan
    try: return int(t.split(":")[0])
    except Exception: return np.nan
hr = df["received_time"].apply(parse_hour)
tb = pd.cut(hr, bins=[-1,6,12,17,24], labels=["night_early","morning","afternoon","evening"]).astype(str)
tb = tb.where(hr.notna(), "unknown")
df["cell6"] = df["cell4"] + "|" + tb + "|" + df["priority"].fillna("UNK") + "|" + dow

for c in ["cell4","cell4_nta","cell4_cb","cell6","inspector_badge","bbl","boro_block"]:
    df[c] = pd.Categorical(df[c].astype(str)).codes

print("\n=== 1. FIRST STAGE: violation_found ~ loo_strictness ===", flush=True)
def run(fml, data, cluster=None, label=""):
    v = {"CRV1": cluster} if cluster else "iid"
    m = pf.feols(fml, data=data, vcov=v)
    b = m.coef().iloc[0]; se = m.se().iloc[0]
    print(f"  {label:<52} b={b:+.4f} se={se:.4f} t={b/se:8.2f} N={m._N:,}", flush=True)
    return b, se

d = df.dropna(subset=["loo_strictness"]).copy()
run("violation_found ~ loo_strictness | cell4", d, None, "cat x unit x ym, iid (replicates published)")
run("violation_found ~ loo_strictness | cell4", d, "inspector_badge", "cat x unit x ym, CLUSTER inspector")
run("violation_found ~ loo_strictness | cell6", d, "inspector_badge", "kitchen sink (spec6), CLUSTER inspector")
run("violation_found ~ loo_strictness | cell4_nta", d, "inspector_badge", "cat x unit x ym x NTA, CLUSTER inspector")

print("\n=== 2. LOO CONTAMINATION: repeat buildings ===", flush=True)
g_ib = df.groupby(["inspector_badge","bbl"])["violation_found"]
ib = g_ib.agg(v_ib="sum", n_ib="count").reset_index()
df = df.merge(ib, on=["inspector_badge","bbl"], how="left")
g_i = df.groupby("inspector_badge")["violation_found"]
ins = g_i.agg(V_i="sum", N_i="count").reset_index()
df = df.merge(ins, on="inspector_badge", how="left")
df["loo_xbbl"] = (df["V_i"] - df["v_ib"]) / (df["N_i"] - df["n_ib"])
# same-block version
g_iblk = df.groupby(["inspector_badge","boro_block"])["violation_found"]
iblk = g_iblk.agg(v_iblk="sum", n_iblk="count").reset_index()
df = df.merge(iblk, on=["inspector_badge","boro_block"], how="left")
df["loo_xblock"] = (df["V_i"] - df["v_iblk"]) / (df["N_i"] - df["n_iblk"])

print(f"  share of cases where inspector has 2+ cases at same BBL: {(df['n_ib']>=2).mean():.3f}")
print(f"  mean # same-inspector same-BBL cases (incl self):        {df['n_ib'].mean():.2f}")
print(f"  share where inspector has 2+ cases on same block:        {(df['n_iblk']>=2).mean():.3f}")
print(f"  corr(loo, loo_xbbl)   = {df[['loo_strictness','loo_xbbl']].corr().iloc[0,1]:.4f}")
print(f"  corr(loo, loo_xblock) = {df[['loo_strictness','loo_xblock']].corr().iloc[0,1]:.4f}")
d2 = df.dropna(subset=["loo_xbbl"]).copy()
run("violation_found ~ loo_xbbl | cell4", d2, "inspector_badge", "LOO excl same-BBL, CLUSTER inspector")
d3 = df.dropna(subset=["loo_xblock"]).copy()
run("violation_found ~ loo_xblock | cell4", d3, "inspector_badge", "LOO excl same-block, CLUSTER inspector")
run("violation_found ~ loo_xblock | cell4_nta", d3, "inspector_badge", "LOO excl same-block + NTA FE, CLUSTER insp")

print("\n=== 3. GEOGRAPHIC CONCENTRATION of inspector caseloads ===", flush=True)
dd = df[df["nta"].notna()]
conc = dd.groupby("inspector_badge")["nta"].agg(
    n_cases="count",
    n_ntas="nunique",
    top_share=lambda s: s.value_counts(normalize=True).iloc[0],
    hhi=lambda s: (s.value_counts(normalize=True)**2).sum(),
)
conc = conc[conc["n_cases"]>=30]
print(conc[["n_ntas","top_share","hhi"]].describe().loc[["mean","25%","50%","75%"]].round(3).to_string())
# borough concentration
bconc = df.groupby("inspector_badge")["borough"].agg(top_boro_share=lambda s: s.value_counts(normalize=True).iloc[0])
print(f"  median share of caseload in inspector's top borough: {bconc['top_boro_share'].median():.3f}")
# CB concentration
cbconc = dd.groupby("inspector_badge")["community_board"].agg(top_cb=lambda s: s.value_counts(normalize=True).iloc[0])
print(f"  median share in top community board: {cbconc['top_cb'].median():.3f}")

print("\n=== 4. BALANCE within unit x ym vs unit x ym x NTA (both sides demeaned) ===", flush=True)
from scipy import stats as sps
df["unit_ym"] = unit + "|" + ym
df["unit_ym_nta"] = df["unit_ym"] + "|" + df["nta"].fillna("UNK")
def dm(s, g):
    return s - s.groupby(g).transform("mean")
for gcol in ["unit_ym","unit_ym_nta"]:
    g = df[gcol]
    rs = dm(df["loo_strictness"], g)
    print(f"  --- residualized on {gcol} ---")
    for c in ["numfloors","yearbuilt","bldgarea","assesstot","unitsres"]:
        x = dm(pd.to_numeric(df[c], errors="coerce"), g)
        v = x.notna() & rs.notna()
        r, p = sps.pearsonr(x[v], rs[v])
        print(f"    {c:<12} r={r:+.4f} p={p:.4f}  (N={v.sum():,})")

print("\n=== 5. VARIANCE DECOMPOSITION honesty check ===", flush=True)
y = df["violation_found"].astype(float)
vy = np.var(y)
def r2_of(gcol_series):
    m = y.groupby(gcol_series).transform("mean")
    return 1 - np.var(y-m)/vy
r2_insp = r2_of(df["inspector_badge"]); r2_cat = r2_of(cat); r2_unit = r2_of(unit)
print(f"  R2 inspector alone: {r2_insp:.4f} ({df['inspector_badge'].nunique()} groups)")
print(f"  R2 category alone:  {r2_cat:.4f} ({cat.nunique()} groups)")
print(f"  R2 unit alone:      {r2_unit:.4f} ({unit.nunique()} groups)")
df["cat_unit"] = cat + "|" + unit
r2_catunit = r2_of(df["cat_unit"]); r2_cell4 = r2_of(df["cell4"])
print(f"  R2 cat x unit:      {r2_catunit:.4f}")
print(f"  R2 cat x unit x ym: {r2_cell4:.4f}")
# joint cell4 + inspector via alternating projections
r = y.copy()
for _ in range(30):
    r = r - r.groupby(df["cell4"]).transform("mean")
    r = r - r.groupby(df["inspector_badge"]).transform("mean")
r2_joint = 1 - np.var(r)/vy
print(f"  R2 cat x unit x ym + inspector (joint): {r2_joint:.4f}")
print(f"  => incremental R2 of inspector beyond case mix: {r2_joint-r2_cell4:.4f}")
# and category increment beyond inspector
r = y.copy()
for _ in range(30):
    r = r - r.groupby(cat).transform("mean")
    r = r - r.groupby(df["inspector_badge"]).transform("mean")
r2_ci = 1 - np.var(r)/vy
print(f"  R2 inspector + category (joint): {r2_ci:.4f}; category increment beyond inspector: {r2_ci-r2_insp:.4f}; inspector increment beyond category: {r2_ci-r2_cat:.4f}")

print("\n=== 6. ROLE COMPOSITION: closure/rescission codes inside 'no_violation' ===", flush=True)
CLOSURE = {"I3","X1","AB","AF","K7","K8","Q4","RG","RL","S0","L2","L3","Y2","Y4","P4"}
FRESH_NV = {"I1","I2","R1","V1","MA","U2","WA","X3"}
dc = df["disposition_code"].str.strip().str.upper()
nv = df["violation_found"]==0
print(f"  no_violation cases: {nv.sum():,}; of which closure/rescission codes: {(nv & dc.isin(CLOSURE)).mean()/nv.mean():.3f}")
print(dc[nv].value_counts().head(8).to_string())
# inspector-level: corr(strictness, closure share of caseload)
df["is_closure"] = (nv & dc.isin(CLOSURE)).astype(int)
prof = df.groupby("inspector_badge").agg(strict=("violation_found","mean"), closure_share=("is_closure","mean"), n=("violation_found","count"))
prof = prof[prof["n"]>=30]
r, p = sps.pearsonr(prof["strict"], prof["closure_share"])
print(f"  inspector-level corr(violation rate, closure-code share): r={r:+.3f} p={p:.2g}")
# strictness spread excluding closure codes
d4 = df[(df["violation_found"]==1) | (nv & dc.isin(FRESH_NV))].copy()
cnt = d4.groupby("inspector_badge")["violation_found"].agg(["mean","count"])
cnt = cnt[cnt["count"]>=30]
print(f"  excluding closure codes: N={len(d4):,}, inspectors={len(cnt)}, violation-rate P10={cnt['mean'].quantile(.1):.3f} P50={cnt['mean'].quantile(.5):.3f} P90={cnt['mean'].quantile(.9):.3f}")
full = df.groupby("inspector_badge")["violation_found"].agg(["mean","count"])
full = full[full["count"]>=30]
print(f"  full sample:              inspectors={len(full)}, violation-rate P10={full['mean'].quantile(.1):.3f} P50={full['mean'].quantile(.5):.3f} P90={full['mean'].quantile(.9):.3f}")
# first stage on fresh-only sample
st = d4.groupby("inspector_badge")["violation_found"].agg(V="sum", N="count")
d4 = d4.drop(columns=["V_i","N_i"]).merge(st, on="inspector_badge")
d4 = d4[d4["N"]>=30]
d4["loo_fresh"] = (d4["V"]-d4["violation_found"])/(d4["N"]-1)
run("violation_found ~ loo_fresh | cell4", d4, "inspector_badge", "FRESH-ONLY sample first stage, CLUSTER insp")

print(f"\ndone in {time.time()-t0:.0f}s")
