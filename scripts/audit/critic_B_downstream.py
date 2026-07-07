"""Critic script B: downstream permit/ECB effects — clustered inference + mediation."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/scripts")
sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
import config
import pyfixest as pf

t0 = time.time()
PANEL = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/data/analysis/master_panel.csv"
usecols = ["complaint_number","community_board","borough","category_description","assigned_to",
           "inspector_badge","bbl","year_month","violation_found","loo_strictness",
           "any_permit_30d","any_permit_60d","any_permit_90d","any_permit_180d","any_permit_365d",
           "future_ecb_30d","future_ecb_90d","future_ecb_180d","future_ecb_365d","future_penalty_365d"]
df = pd.read_csv(PANEL, usecols=usecols, low_memory=False)
df["complaint_number"] = df["complaint_number"].astype(str)
print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

import sqlite3
conn = sqlite3.connect(str(config.DB_PATH))
nta = pd.read_sql_query("SELECT complaint_number, nta FROM complaint_nta", conn)
conn.close()
nta["complaint_number"] = nta["complaint_number"].astype(str)
df = df.merge(nta, on="complaint_number", how="left")

cat = df["category_description"].fillna("UNK")
unit = df["assigned_to"].fillna("UNK")
ym = df["year_month"].fillna("UNK")
df["cell4"] = cat + "|" + unit + "|" + ym
df["cell4_nta"] = df["cell4"] + "|" + df["nta"].fillna("UNK")
df["cell4_cb"] = df["cell4"] + "|" + df["community_board"].fillna("UNK").astype(str)
for c in ["cell4","cell4_nta","cell4_cb","inspector_badge","bbl"]:
    df[c] = pd.Categorical(df[c].astype(str)).codes

df["any_ecb_30d"]  = (df["future_ecb_30d"]>0).astype(float)
df["any_ecb_90d"]  = (df["future_ecb_90d"]>0).astype(float)
df["any_ecb_180d"] = (df["future_ecb_180d"]>0).astype(float)
df["any_ecb_365d"] = (df["future_ecb_365d"]>0).astype(float)
df["log_pen_365d"] = np.log1p(df["future_penalty_365d"].astype(float))

def run(fml, data, fe, cluster=None, label=""):
    v = {"CRV1": cluster} if cluster else "iid"
    m = pf.feols(f"{fml} | {fe}", data=data, vcov=v)
    b = m.coef().loc["loo_strictness"]; se = m.se().loc["loo_strictness"]
    print(f"  {label:<58} b={b:+.4f} se={se:.4f} t={b/se:7.2f}", flush=True)
    return m

d = df.dropna(subset=["loo_strictness"]).copy()

print("\n=== PUBLISHED DOWNSTREAM CLAIMS: iid vs clustered by inspector ===", flush=True)
specs = [
    ("any_permit_30d",  "cell4",     "permit 30d | cat.unit.ym (post: +0.08pp/10pp, t=2.5)"),
    ("any_permit_30d",  "cell4_nta", "permit 30d | +NTA (csv: 0.0232)"),
    ("any_permit_90d",  "cell4_cb",  "permit 90d | +CB (post: +0.25pp/10pp t=7.1)"),
    ("any_permit_90d",  "cell4_nta", "permit 90d | +NTA (csv: 0.0137)"),
    ("any_permit_365d", "cell4_nta", "permit 365d | +NTA (csv: 0.0138)"),
    ("any_ecb_180d",    "cell4",     "any ECB 180d | cat.unit.ym (post: +0.09pp/10pp)"),
    ("any_ecb_365d",    "cell4",     "any ECB 365d | cat.unit.ym (post: +0.27pp/10pp t=7.5)"),
    ("any_ecb_365d",    "cell4_nta", "any ECB 365d | +NTA (csv: 0.0253)"),
    ("log_pen_365d",    "cell4",     "log penalty 365d | cat.unit.ym (post: t=8.15)"),
]
for yv, fe, lab in specs:
    run(f"{yv} ~ loo_strictness", d, fe, None, lab + "  [iid]")
    run(f"{yv} ~ loo_strictness", d, fe, "inspector_badge", lab + "  [CL insp]")

print("\n=== MEDIATION: control for / split by violation_found ===", flush=True)
print("-- controlling for violation_found (CL inspector) --")
for yv, fe, lab in [("any_permit_30d","cell4","permit 30d | cell4"),
                    ("any_permit_90d","cell4_cb","permit 90d | +CB"),
                    ("any_ecb_365d","cell4","any ECB 365d | cell4"),
                    ("log_pen_365d","cell4","log penalty 365d | cell4")]:
    m = pf.feols(f"{yv} ~ loo_strictness + violation_found | {fe}", data=d, vcov={"CRV1":"inspector_badge"})
    b = m.coef(); se = m.se()
    print(f"  {lab:<38} loo: b={b['loo_strictness']:+.4f} t={b['loo_strictness']/se['loo_strictness']:6.2f}   viol: b={b['violation_found']:+.4f} t={b['violation_found']/se['violation_found']:6.2f}", flush=True)

print("-- violation_found as outcome of the SAME downstream design (first stage in these cells) --")
run("violation_found ~ loo_strictness", d, "cell4", "inspector_badge", "violation_found | cell4 [CL insp]")

print("-- split samples (CL inspector) --")
for val, vlab in [(1,"violation written"), (0,"no violation")]:
    sub = d[d["violation_found"]==val]
    for yv, fe, lab in [("any_permit_30d","cell4","permit 30d"),
                        ("any_ecb_365d","cell4","any ECB 365d"),
                        ("log_pen_365d","cell4","log pen 365d")]:
        m = pf.feols(f"{yv} ~ loo_strictness | {fe}", data=sub, vcov={"CRV1":"inspector_badge"})
        b = m.coef().loc["loo_strictness"]; se = m.se().loc["loo_strictness"]
        print(f"  [{vlab:<18}] {lab:<16} b={b:+.4f} se={se:.4f} t={b/se:6.2f} N={m._N:,}", flush=True)

print("\n=== simple descriptive: outcome means by violation_found ===")
print(d.groupby("violation_found")[["any_permit_30d","any_permit_90d","any_ecb_365d","log_pen_365d"]].mean().round(4).to_string())

print(f"\ndone in {time.time()-t0:.0f}s")
