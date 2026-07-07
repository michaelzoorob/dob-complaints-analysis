"""Critic script D: verify own-BBL linkage, focal-ECB timing, spillovers with clustered SEs."""
import sys, time, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/scripts")
sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
import config
import pyfixest as pf

t0 = time.time()
PANEL = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/data/analysis/master_panel.csv"
usecols = ["complaint_number","category_description","assigned_to","inspector_badge","bbl","boro_block",
           "year_month","inspection_dt","violation_found","loo_strictness",
           "any_permit_30d","any_permit_365d","future_ecb_30d","future_ecb_365d"]
df = pd.read_csv(PANEL, usecols=usecols, low_memory=False)
df["complaint_number"] = df["complaint_number"].astype(str)
df["inspection_dt"] = pd.to_datetime(df["inspection_dt"], errors="coerce")
df = df.dropna(subset=["inspection_dt"])
print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

conn = sqlite3.connect(str(config.DB_PATH))
nta = pd.read_sql_query("SELECT complaint_number, nta FROM complaint_nta", conn)
nta["complaint_number"] = nta["complaint_number"].astype(str)
df = df.merge(nta, on="complaint_number", how="left")

# ---- load permits and ECB ----
permits = pd.read_sql_query("SELECT bbl, issued_date FROM permits WHERE bbl IS NOT NULL AND issued_date IS NOT NULL", conn)
permits["dt"] = pd.to_datetime(permits["issued_date"], errors="coerce")
pnow = pd.read_sql_query("SELECT bbl, filing_date FROM permits_now WHERE bbl IS NOT NULL AND filing_date IS NOT NULL", conn)
pnow["dt"] = pd.to_datetime(pnow["filing_date"], errors="coerce")
allp = pd.concat([permits[["bbl","dt"]], pnow[["bbl","dt"]]], ignore_index=True).dropna(subset=["dt"])
ecb = pd.read_sql_query("SELECT boro, block, lot, issue_date, dob_violation_number FROM ecb_violations WHERE issue_date IS NOT NULL AND boro IS NOT NULL", conn)
conn.close()
def mkbbl(r):
    try: return f"{r['boro']}{int(float(r['block'])):05d}{int(float(r['lot'])):04d}"
    except Exception: return ""
ecb["bbl"] = ecb.apply(mkbbl, axis=1)
ecb["dt"] = pd.to_datetime(ecb["issue_date"], format="%Y%m%d", errors="coerce")
ecb = ecb[(ecb["bbl"]!="") & ecb["dt"].notna()]
print(f"permits {len(allp):,}  ecb {len(ecb):,}", flush=True)

# ---- vectorized counter: events at key within (t+lo, t+hi] days ----
def count_window(events_key, events_dt, focal_key, focal_dt, lo, hi):
    """Count events per focal row with key match and dt in (focal_dt+lo, focal_dt+hi]."""
    ev = pd.DataFrame({"k": events_key.values, "d": events_dt.values.astype("datetime64[ns]")}).sort_values(["k","d"])
    ev["i"] = np.arange(len(ev))
    starts = ev.groupby("k")["i"].min()
    ends = ev.groupby("k")["i"].max() + 1
    dsec = ev["d"].values.astype("int64")
    fk = pd.Series(focal_key.values)
    s = fk.map(starts); e = fk.map(ends)
    lo_t = (focal_dt + pd.Timedelta(days=lo)).values.astype("datetime64[ns]").astype("int64")
    hi_t = (focal_dt + pd.Timedelta(days=hi)).values.astype("datetime64[ns]").astype("int64")
    out = np.zeros(len(fk), dtype=np.int64)
    ok = s.notna().values
    si = s.values[ok].astype(np.int64); ei = e.values[ok].astype(np.int64)
    lot = lo_t[ok]; hit = hi_t[ok]
    res = np.zeros(ok.sum(), dtype=np.int64)
    # per-focal binary search within its key's slice
    for j in range(len(si)):
        seg = dsec[si[j]:ei[j]]
        res[j] = np.searchsorted(seg, hit[j], side="right") - np.searchsorted(seg, lot[j], side="right")
    out[ok] = res
    return out

# ---- 1. VERIFY own-BBL linkage against panel columns ----
print("\n=== 1. VERIFY build_analysis_dataset own-BBL linkage ===", flush=True)
df["bbl"] = df["bbl"].astype(str)
chk = df.sample(n=60000, random_state=1).copy()
chk["my_p365"] = count_window(allp["bbl"].astype(str), allp["dt"], chk["bbl"], chk["inspection_dt"], 0, 365)
chk["my_any_p365"] = (chk["my_p365"]>0).astype(int)
agree = (chk["my_any_p365"]==chk["any_permit_365d"]).mean()
print(f"  any_permit_365d agreement with independent recount: {agree:.4f}")
chk["my_e365"] = count_window(ecb["bbl"], ecb["dt"], chk["bbl"], chk["inspection_dt"], 0, 365)
agree_e = (chk["my_e365"]==chk["future_ecb_365d"]).mean()
corr_e = np.corrcoef(chk["my_e365"], chk["future_ecb_365d"])[0,1]
print(f"  future_ecb_365d exact agreement: {agree_e:.4f}  corr: {corr_e:.4f}")

# ---- 2. FOCAL ECB TIMING: is the focal citation excluded? ----
print("\n=== 2. Focal-citation timing (violation_found=1 rows) ===", flush=True)
v1 = df[df["violation_found"]==1].sample(n=60000, random_state=2).copy()
for lo, hi, lab in [(-1,0,"day 0 (same day)"), (0,7,"days 1-7"), (7,30,"days 8-30"), (30,365,"days 31-365")]:
    c = count_window(ecb["bbl"], ecb["dt"], v1["bbl"], v1["inspection_dt"], lo, hi)
    print(f"  share of violation cases w/ ECB at same BBL {lab:<18}: {(c>0).mean():.3f}")
v0 = df[df["violation_found"]==0].sample(n=60000, random_state=2).copy()
c0 = count_window(ecb["bbl"], ecb["dt"], v0["bbl"], v0["inspection_dt"], -1, 0)
print(f"  [no-violation cases, day 0 rate for reference]: {(c0>0).mean():.3f}")

# ---- 3. SPILLOVERS with clustering ----
print("\n=== 3. SPILLOVERS: block-minus-own counts, clustered ===", flush=True)
# map bbl -> block for permits/ecb via PLUTO universe: block key = boro+block from bbl string
allp["blk"] = allp["bbl"].astype(str).str[:6]
ecb["blk"] = ecb["bbl"].astype(str).str[:6]
df["blk"] = df["bbl"].str[:6]
sp = df.dropna(subset=["loo_strictness"]).copy()

t1 = time.time()
sp["blk_p365"] = count_window(allp["blk"], allp["dt"], sp["blk"], sp["inspection_dt"], 0, 365)
sp["own_p365"] = count_window(allp["bbl"].astype(str), allp["dt"], sp["bbl"], sp["inspection_dt"], 0, 365)
sp["nbr_permits_365"] = sp["blk_p365"] - sp["own_p365"]
sp["blk_e365"] = count_window(ecb["blk"], ecb["dt"], sp["blk"], sp["inspection_dt"], 0, 365)
sp["own_e365"] = count_window(ecb["bbl"], ecb["dt"], sp["bbl"], sp["inspection_dt"], 0, 365)
sp["nbr_ecb_365"] = sp["blk_e365"] - sp["own_e365"]
# lagged window (exclude first 14 days: same-visit mechanical channel)
sp["blk_e_l"] = count_window(ecb["blk"], ecb["dt"], sp["blk"], sp["inspection_dt"], 14, 365)
sp["own_e_l"] = count_window(ecb["bbl"], ecb["dt"], sp["bbl"], sp["inspection_dt"], 14, 365)
sp["nbr_ecb_14_365"] = sp["blk_e_l"] - sp["own_e_l"]
sp["nbr_ecb_0_14"] = sp["nbr_ecb_365"] - sp["nbr_ecb_14_365"]
print(f"  counts built in {time.time()-t1:.0f}s; mean nbr permits365={sp['nbr_permits_365'].mean():.2f}, nbr ecb365={sp['nbr_ecb_365'].mean():.2f}", flush=True)

cat = sp["category_description"].fillna("UNK"); unit = sp["assigned_to"].fillna("UNK")
sp["cellnta"] = cat+"|"+unit+"|"+sp["year_month"].fillna("UNK")+"|"+sp["nta"].fillna("UNK")
for c in ["cellnta","inspector_badge","blk"]:
    sp[c] = pd.Categorical(sp[c].astype(str)).codes

for yv, lab in [("nbr_permits_365","neighbor permits 365d (script est 0.80, t=2.8)"),
                ("nbr_ecb_365","neighbor ECB 365d (script est 0.20, t=3.9)"),
                ("nbr_ecb_0_14","neighbor ECB days 1-14 (same-visit window)"),
                ("nbr_ecb_14_365","neighbor ECB days 15-365 (lagged)")]:
    m1 = pf.feols(f"{yv} ~ loo_strictness | cellnta", data=sp, vcov="iid")
    m2 = pf.feols(f"{yv} ~ loo_strictness | cellnta", data=sp, vcov={"CRV1":"inspector_badge"})
    m3 = pf.feols(f"{yv} ~ loo_strictness | cellnta", data=sp, vcov={"CRV1":"blk"})
    b = m1.coef().iloc[0]
    print(f"  {lab:<48} b={b:+.4f}  t_iid={b/m1.se().iloc[0]:6.2f}  t_CLinsp={b/m2.se().iloc[0]:6.2f}  t_CLblock={b/m3.se().iloc[0]:6.2f}  N={m1._N:,}", flush=True)

print(f"\ndone in {time.time()-t0:.0f}s")
