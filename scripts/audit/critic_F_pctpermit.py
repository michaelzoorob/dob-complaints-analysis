"""Critic script F: replicate spillover 'pct neighbors w/ permit 365d' (post: +0.05pp/10pp, t=4.2) with clustering."""
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
           "year_month","inspection_dt","violation_found","loo_strictness"]
df = pd.read_csv(PANEL, usecols=usecols, dtype={"bbl":"string"}, low_memory=False)
df["complaint_number"] = df["complaint_number"].astype(str)
df["inspection_dt"] = pd.to_datetime(df["inspection_dt"], errors="coerce")
df = df.dropna(subset=["inspection_dt","loo_strictness"])

conn = sqlite3.connect(str(config.DB_PATH))
nta = pd.read_sql_query("SELECT complaint_number, nta FROM complaint_nta", conn)
nta["complaint_number"] = nta["complaint_number"].astype(str)
df = df.merge(nta, on="complaint_number", how="inner")  # their script inner-joins NTA

permits = pd.read_sql_query("SELECT bbl, issued_date as d FROM permits WHERE bbl IS NOT NULL AND issued_date IS NOT NULL", conn)
pnow = pd.read_sql_query("SELECT bbl, filing_date as d FROM permits_now WHERE bbl IS NOT NULL AND filing_date IS NOT NULL", conn)
allp = pd.concat([permits, pnow], ignore_index=True)
allp["dt"] = pd.to_datetime(allp["d"], errors="coerce")
allp = allp.dropna(subset=["dt"])
pluto = pd.read_sql_query("SELECT borocode, block, lot FROM pluto WHERE latitude IS NOT NULL", conn)
conn.close()
pluto["bbl"] = pluto.apply(lambda r: f"{r['borocode']}{int(r['block']):05d}{int(r['lot']):04d}", axis=1)
pluto["boro_block"] = pluto["borocode"].astype(str) + "_" + pluto["block"].astype(str)
block_to_bbls = pluto.groupby("boro_block")["bbl"].apply(list).to_dict()
print(f"setup {time.time()-t0:.0f}s; permits {len(allp):,}", flush=True)

# permit dates per bbl as sorted int64 ns arrays
allp["ns"] = allp["dt"].values.astype("datetime64[ns]").astype("int64")
pdates = {k: np.sort(v.values) for k, v in allp.groupby("bbl")["ns"]}

# subsample
sub = df.sample(n=min(160000, len(df)), random_state=7).copy()
W = 365 * 86400 * 10**9
tvals = sub["inspection_dt"].values.astype("datetime64[ns]").astype("int64")
bbls = sub["bbl"].astype(str).values
blocks = sub["boro_block"].astype(str).values
pct = np.full(len(sub), np.nan)
nn = np.zeros(len(sub), dtype=np.int64)
t1 = time.time()
for i in range(len(sub)):
    members = block_to_bbls.get(blocks[i])
    if not members: continue
    cnt = 0; tot = 0
    lo = tvals[i]; hi = tvals[i] + W
    fb = bbls[i]
    for b in members:
        if b == fb: continue
        tot += 1
        arr = pdates.get(b)
        if arr is not None:
            if np.searchsorted(arr, hi, side="right") > np.searchsorted(arr, lo, side="right"):
                cnt += 1
    if tot > 0:
        pct[i] = cnt / tot; nn[i] = tot
sub["pct_permit_365"] = pct
sub["n_neighbors"] = nn
sub = sub[sub["n_neighbors"] > 0]
print(f"loop {time.time()-t1:.0f}s; N={len(sub):,}; mean pct={sub['pct_permit_365'].mean():.4f} (their log: 0.154); mean nbrs={sub['n_neighbors'].mean():.1f} (their 36.6)", flush=True)

sub["cell"] = sub["category_description"].fillna("U")+"|"+sub["assigned_to"].fillna("U")+"|"+sub["year_month"].fillna("U")+"|"+sub["nta"].fillna("U")
sub["blk"] = blocks[sub.index.map(lambda x: True)] if False else sub["boro_block"]
for c in ["cell","inspector_badge","blk"]:
    sub[c] = pd.Categorical(sub[c].astype(str)).codes
m1 = pf.feols("pct_permit_365 ~ loo_strictness | cell", data=sub, vcov="iid")
m2 = pf.feols("pct_permit_365 ~ loo_strictness | cell", data=sub, vcov={"CRV1":"inspector_badge"})
m3 = pf.feols("pct_permit_365 ~ loo_strictness | cell", data=sub, vcov={"CRV1":"blk"})
b = m1.coef().iloc[0]
print(f"pct neighbors w/ permit 365d (published 0.0048, t=4.19 full sample):")
print(f"  b={b:+.4f}  t_iid={b/m1.se().iloc[0]:5.2f}  t_CLinsp={b/m2.se().iloc[0]:5.2f}  t_CLblock={b/m3.se().iloc[0]:5.2f}  N={m1._N:,}")
print(f"done in {time.time()-t0:.0f}s")
