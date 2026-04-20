"""
Spatial Spillover Analysis: Does strict enforcement at one property
induce compliance at nearby properties?

Design:
- For each inspected property, identify neighbors (same block)
- Measure whether neighbors file more permits or receive fewer violations
  after a strict inspector visits the focal property
- Use the same LOO inspector strictness instrument

Key outcome: neighbor permit filings within 90/180/365 days
"""

import sqlite3, sys, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from disposition_codes import classify_disposition

conn = sqlite3.connect(str(config.DB_PATH), timeout=30)

BC = "CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2' WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' END"

# ── Load focal complaints ───────────────────────────────────────────────
print("Loading focal complaints...", flush=True)
df = pd.read_sql_query(f"""
    SELECT o.complaint_number, o.disposition_code, o.date_entered, o.inspection_date,
           o.community_board, b.borough, b.category_description, b.assigned_to,
           b.inspector_badge, b.priority, b.block, b.lot, cn.nta,
           p.latitude, p.longitude
    FROM open_data o
    JOIN bis_scrape b ON o.complaint_number = b.complaint_number
    JOIN complaint_nta cn ON cn.complaint_number = o.complaint_number
    LEFT JOIN pluto p ON p.borocode = {BC} AND p.block = b.block AND p.lot = b.lot
    WHERE b.inspector_badge IS NOT NULL
      AND o.disposition_code IS NOT NULL AND o.disposition_code != ''
      AND b.block IS NOT NULL AND b.lot IS NOT NULL
""", conn)

df["outcome"] = df["disposition_code"].apply(classify_disposition)
df["violation_found"] = (df["outcome"] == "violation").astype(int)
df["inspection_dt"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y", errors="coerce")
df["year_month"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce").dt.to_period("M").astype(str)

bc_map = {"MANHATTAN":"1","BRONX":"2","BROOKLYN":"3","QUEENS":"4","STATEN ISLAND":"5"}
df["boro_code"] = df["borough"].map(bc_map)
df["bbl"] = df.apply(lambda r: f"{r['boro_code']}{int(r['block']):05d}{int(r['lot']):04d}"
    if pd.notna(r['boro_code']) and pd.notna(r['block']) and pd.notna(r['lot']) and r['block']!='' and r['lot']!=''
    else "", axis=1)
df["boro_block"] = df["boro_code"].astype(str) + "_" + df["block"].astype(str)

# Analysis sample
sub = df[df["outcome"].isin(["violation","no_violation"])].copy()
counts = sub["inspector_badge"].value_counts()
sub = sub[sub["inspector_badge"].isin(counts[counts>=30].index)]

stats = sub.groupby("inspector_badge")["violation_found"].agg(["sum","count"])
sub = sub.merge(stats.rename(columns={"sum":"tv","count":"tn"}), on="inspector_badge")
sub["loo"] = (sub["tv"]-sub["violation_found"])/(sub["tn"]-1)
sub = sub.drop(columns=["tv","tn"])
sub = sub.dropna(subset=["inspection_dt"])

print(f"Focal sample: {len(sub):,} complaints across {sub['bbl'].nunique():,} BBLs")
print(f"Unique blocks: {sub['boro_block'].nunique():,}")

# ── Load neighbor permit data ───────────────────────────────────────────
print("\nLoading permits...", flush=True)
permits = pd.read_sql_query(
    "SELECT bbl, issued_date FROM permits WHERE bbl IS NOT NULL AND issued_date IS NOT NULL", conn)
permits["dt"] = pd.to_datetime(permits["issued_date"], errors="coerce")
permits = permits.dropna(subset=["dt"])

pnow = pd.read_sql_query(
    "SELECT bbl, filing_date FROM permits_now WHERE bbl IS NOT NULL AND filing_date IS NOT NULL", conn)
pnow["dt"] = pd.to_datetime(pnow["filing_date"], errors="coerce")
pnow = pnow.dropna(subset=["dt"])

# Combine all permits
all_permits = pd.concat([
    permits[["bbl","dt"]],
    pnow[["bbl","dt"]],
], ignore_index=True)
print(f"Total permits: {len(all_permits):,}")

# ECB violations
ecb = pd.read_sql_query(
    "SELECT boro,block,lot,issue_date FROM ecb_violations WHERE issue_date IS NOT NULL AND boro IS NOT NULL AND block IS NOT NULL AND lot IS NOT NULL", conn)
ecb["bbl"] = ecb.apply(lambda r: f"{r['boro']}{int(float(r['block'])):05d}{int(float(r['lot'])):04d}"
    if r['boro'] and r['block'] and r['lot'] else "", axis=1)
ecb = ecb[ecb["bbl"]!=""]
ecb["dt"] = pd.to_datetime(ecb["issue_date"], format="%Y%m%d", errors="coerce")
ecb = ecb.dropna(subset=["dt"])
print(f"ECB violations: {len(ecb):,}")
conn.close()

# ── Build block → BBL index ─────────────────────────────────────────────
print("\nBuilding block-neighbor index...", flush=True)

# Get all BBLs per block from PLUTO (all properties, not just complained-about ones)
conn2 = sqlite3.connect(str(config.DB_PATH))
pluto_blocks = pd.read_sql_query("""
    SELECT borocode, block, lot,
           borocode || '_' || block as boro_block
    FROM pluto
    WHERE latitude IS NOT NULL
""", conn2)
pluto_blocks["bbl"] = pluto_blocks.apply(
    lambda r: f"{r['borocode']}{int(r['block']):05d}{int(r['lot']):04d}", axis=1)
conn2.close()

block_to_bbls = pluto_blocks.groupby("boro_block")["bbl"].apply(set).to_dict()
print(f"Blocks with properties: {len(block_to_bbls):,}")

# Pre-index permits and ECB by BBL
permit_idx = all_permits.groupby("bbl")
ecb_idx = ecb.groupby("bbl")

# ── Compute neighbor outcomes ───────────────────────────────────────────
print("\nComputing neighbor outcomes...", flush=True)

windows = [90, 180, 365]
results = []

for i, (_, row) in enumerate(sub.iterrows()):
    focal_bbl = row["bbl"]
    focal_block = row["boro_block"]
    idt = row["inspection_dt"]

    # Get all BBLs on the same block, excluding the focal property
    neighbor_bbls = block_to_bbls.get(focal_block, set()) - {focal_bbl}
    if not neighbor_bbls:
        continue

    rec = {
        "complaint_number": row["complaint_number"],
        "n_neighbors": len(neighbor_bbls),
    }

    for w in windows:
        cutoff = idt + pd.Timedelta(days=w)

        # Count neighbor permits
        n_permits = 0
        n_neighbors_with_permit = 0
        for nbbl in neighbor_bbls:
            if nbbl in permit_idx.groups:
                np_ = permit_idx.get_group(nbbl)
                within = np_[(np_["dt"] > idt) & (np_["dt"] <= cutoff)]
                if len(within) > 0:
                    n_permits += len(within)
                    n_neighbors_with_permit += 1

        rec[f"neighbor_permits_{w}d"] = n_permits
        rec[f"neighbor_any_permit_{w}d"] = int(n_neighbors_with_permit > 0)
        rec[f"neighbor_pct_permit_{w}d"] = n_neighbors_with_permit / len(neighbor_bbls) if neighbor_bbls else 0

        # Count neighbor ECB violations
        n_ecb = 0
        for nbbl in neighbor_bbls:
            if nbbl in ecb_idx.groups:
                ev = ecb_idx.get_group(nbbl)
                within = ev[(ev["dt"] > idt) & (ev["dt"] <= cutoff)]
                n_ecb += len(within)

        rec[f"neighbor_ecb_{w}d"] = n_ecb

    results.append(rec)

    if (i+1) % 25000 == 0:
        print(f"  {i+1:,}/{len(sub):,} ({100*(i+1)/len(sub):.0f}%)", flush=True)

spill = pd.DataFrame(results)
sub_spill = sub.merge(spill, on="complaint_number", how="inner")
print(f"\nSpillover sample: {len(sub_spill):,} complaints with neighbors")

# ── Regressions ─────────────────────────────────────────────────────────
def demean(a,g):
    s=pd.Series(a,dtype=float); return (s-s.groupby(pd.Series(g)).transform("mean")).values

def run(y,x,g):
    v=~np.isnan(y)&~np.isnan(x); yd=demean(y[v],g[v]); xd=demean(x[v],g[v])
    X=np.column_stack([np.ones(len(yd)),xd]); b,_,_,_=np.linalg.lstsq(X,yd,rcond=None)
    r=yd-X@b; se=np.sqrt(np.sum(r**2)/(len(yd)-2)*np.linalg.inv(X.T@X)[1,1])
    return b[1],se,b[1]/se if se>0 else 0,v.sum()

nta_groups = (sub_spill["category_description"].fillna("U").astype(str)+"|"+
              sub_spill["assigned_to"].fillna("U").astype(str)+"|"+
              sub_spill["year_month"].fillna("U").astype(str)+"|"+
              sub_spill["nta"].astype(str)).values
x = sub_spill["loo"].values

print(f"\n{'='*90}")
print(f"SPATIAL SPILLOVER ANALYSIS — Same-Block Neighbors")
print(f"N = {len(sub_spill):,}  |  Cat × Unit × YM × NTA FEs")
print(f"Mean neighbors per focal property: {sub_spill['n_neighbors'].mean():.1f}")
print(f"{'='*90}")

print(f"\nPANEL A: NEIGHBOR PERMIT FILING")
print(f"{'Outcome':<45} {'β':>8} {'SE':>8} {'t':>8} {'N':>8} {'Mean':>8}")
print("-"*85)
for w in windows:
    y = sub_spill[f"neighbor_any_permit_{w}d"].values.astype(float)
    b,se,t,n = run(y,x,nta_groups)
    print(f"  Any neighbor permit ({w}d){'':<19} {b:>8.4f} {se:>8.4f} {t:>8.2f} {n:>8,} {np.nanmean(y):>8.3f}")

print()
for w in windows:
    y = sub_spill[f"neighbor_pct_permit_{w}d"].values.astype(float)
    b,se,t,n = run(y,x,nta_groups)
    print(f"  Pct neighbors w/ permit ({w}d){'':<14} {b:>8.4f} {se:>8.4f} {t:>8.2f} {n:>8,} {np.nanmean(y):>8.3f}")

print()
for w in windows:
    y = sub_spill[f"neighbor_permits_{w}d"].values.astype(float)
    b,se,t,n = run(y,x,nta_groups)
    print(f"  Total neighbor permits ({w}d){'':<16} {b:>8.4f} {se:>8.4f} {t:>8.2f} {n:>8,} {np.nanmean(y):>8.3f}")

print(f"\nPANEL B: NEIGHBOR ECB VIOLATIONS")
print("-"*85)
for w in windows:
    y = sub_spill[f"neighbor_ecb_{w}d"].values.astype(float)
    b,se,t,n = run(y,x,nta_groups)
    print(f"  Neighbor ECB violations ({w}d){'':<15} {b:>8.4f} {se:>8.4f} {t:>8.2f} {n:>8,} {np.nanmean(y):>8.3f}")

# Panel C: Heterogeneity by own outcome
print(f"\nPANEL C: SPILLOVERS CONDITIONAL ON OWN VIOLATION STATUS")
print("-"*85)
for outcome_val, outcome_label in [(1, "Violation found"), (0, "No violation")]:
    osub = sub_spill[sub_spill["violation_found"]==outcome_val]
    og = (osub["category_description"].fillna("U")+"|"+osub["assigned_to"].fillna("U")+"|"+
          osub["year_month"].fillna("U")+"|"+osub["nta"]).values
    ox = osub["loo"].values
    y = osub["neighbor_any_permit_90d"].values.astype(float)
    b,se,t,n = run(y,ox,og)
    print(f"  {outcome_label}: neighbor permit (90d){'':<13} {b:>8.4f} {se:>8.4f} {t:>8.2f} {n:>8,}")

print(f"\n{'='*90}")
print("DONE")
print(f"{'='*90}")
