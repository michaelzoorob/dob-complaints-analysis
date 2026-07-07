"""Critic script E: does future_ecb_365d recount the focal complaint's own citation?"""
import sys, time, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/scripts")
sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
import config

t0 = time.time()
PANEL = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/data/analysis/master_panel.csv"
df = pd.read_csv(PANEL, usecols=["complaint_number","inspection_dt","violation_found","future_ecb_365d","future_ecb_30d"],
                 dtype={"complaint_number":"string"}, low_memory=False)
df["inspection_dt"] = pd.to_datetime(df["inspection_dt"], errors="coerce")

conn = sqlite3.connect(str(config.DB_PATH))
own = pd.read_sql_query("SELECT complaint_number, ecb_violation FROM bis_scrape WHERE ecb_violation IS NOT NULL AND ecb_violation != ''", conn)
ecb = pd.read_sql_query("SELECT ecb_violation_number, issue_date FROM ecb_violations WHERE issue_date IS NOT NULL", conn)
conn.close()
own["complaint_number"] = own["complaint_number"].astype(str)
df["complaint_number"] = df["complaint_number"].astype(str)

# explode the focal complaint's own ECB numbers
own["nums"] = own["ecb_violation"].str.split()
own = own.explode("nums")
own["num"] = own["nums"].str.strip().str.upper()
# ecb table numbers carry a trailing check letter too; normalize both with and without
ecb["num"] = ecb["ecb_violation_number"].astype(str).str.strip().str.upper()
ecb["dt"] = pd.to_datetime(ecb["issue_date"], format="%Y%m%d", errors="coerce")
m = own.merge(ecb[["num","dt"]], on="num", how="left")
print(f"focal ECB numbers: {len(own):,}; matched to ECB table: {m['dt'].notna().mean():.3f}")
# also try without check letter
if m["dt"].notna().mean() < 0.5:
    own["num2"] = own["num"].str[:-1]
    ecb["num2"] = ecb["num"].str[:-1]
    m = own.merge(ecb[["num2","dt"]].drop_duplicates("num2"), left_on="num2", right_on="num2", how="left")
    print(f"  retry without check digit: matched {m['dt'].notna().mean():.3f}")

m = m.dropna(subset=["dt"]).merge(df[["complaint_number","inspection_dt","violation_found"]], on="complaint_number", how="inner")
m = m.dropna(subset=["inspection_dt"])
m["lag"] = (m["dt"] - m["inspection_dt"]).dt.days
print(f"\nfocal-complaint own ECB violations linked to panel: {len(m):,}")
print("distribution of (own ECB issue date - inspection date), days:")
print(m["lag"].describe(percentiles=[.05,.25,.5,.75,.95,.99]).round(1).to_string())
print(f"  share with lag == 0:      {(m['lag']==0).mean():.3f}")
print(f"  share with lag in [1,30]: {m['lag'].between(1,30).mean():.3f}")
print(f"  share with lag in [31,365]: {m['lag'].between(31,365).mean():.3f}")
print(f"  share with lag < 0:       {(m['lag']<0).mean():.3f}")

# how much of future_ecb_365d at violation cases is the OWN citation?
own_late = m[m["lag"].between(1,365)].groupby("complaint_number").size().rename("own_in_window")
d = df.merge(own_late, on="complaint_number", how="left")
d["own_in_window"] = d["own_in_window"].fillna(0)
v = d[d["violation_found"]==1]
print(f"\nviolation cases: mean future_ecb_365d = {v['future_ecb_365d'].mean():.3f}; mean own-citation-in-window = {v['own_in_window'].mean():.4f}")
print(f"share of violation cases where own citation appears inside the 365d window: {(v['own_in_window']>0).mean():.4f}")
print(f"\ndone in {time.time()-t0:.0f}s")
