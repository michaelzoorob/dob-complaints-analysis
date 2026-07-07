"""Critic script C: selection into substantive outcomes (no-access margin), unconditional first stage, dispatch."""
import sys, time, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports/scripts")
sys.path.insert(0, "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
import config
from disposition_codes import classify_disposition
import pyfixest as pf
from scipy import stats as sps

t0 = time.time()
conn = sqlite3.connect(str(config.DB_PATH))
df = pd.read_sql_query("""
    SELECT o.complaint_number, o.disposition_code, o.date_entered,
           b.priority, b.category_description, b.assigned_to, b.inspector_badge
    FROM open_data o
    JOIN bis_scrape b ON o.complaint_number = b.complaint_number
    WHERE b.inspector_badge IS NOT NULL
      AND o.disposition_code IS NOT NULL AND o.disposition_code != ''
""", conn)
conn.close()
print(f"loaded {len(df):,} assigned complaints in {time.time()-t0:.0f}s", flush=True)

df["outcome"] = df["disposition_code"].apply(classify_disposition)
df["violation_found"] = (df["outcome"]=="violation").astype(int)
df["year_month"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce").dt.to_period("M").astype(str)
print(df["outcome"].value_counts().to_string(), flush=True)

# inspector set: >=30 substantive cases (matches published sample)
sub = df[df["outcome"].isin(["violation","no_violation"])]
cnt = sub["inspector_badge"].value_counts()
keep = cnt[cnt>=30].index
print(f"inspectors with >=30 substantive: {len(keep)}")

cat = df["category_description"].fillna("UNK")
unit = df["assigned_to"].fillna("UNK")
df["cell4"] = cat + "|" + unit + "|" + df["year_month"].fillna("UNK")
df["unit_ym"] = unit + "|" + df["year_month"].fillna("UNK")

# conditional inspector strictness (published measure, inspector-level mean)
strict = sub[sub["inspector_badge"].isin(keep)].groupby("inspector_badge")["violation_found"].mean().rename("insp_strict")

# ---- 1. Does strictness predict ACCESS (selection into the sample)? ----
print("\n=== SELECTION: no-access & substantive-outcome margins vs strictness ===", flush=True)
d_na = df[df["outcome"].isin(["violation","no_violation","no_access"]) & df["inspector_badge"].isin(keep)].copy()
d_na = d_na.merge(strict, on="inspector_badge")
d_na["no_access"] = (d_na["outcome"]=="no_access").astype(float)
m = pf.feols("no_access ~ insp_strict | cell4", data=d_na, vcov={"CRV1":"inspector_badge"})
b, se = m.coef().loc["insp_strict"], m.se().loc["insp_strict"]
print(f"  no_access ~ inspector strictness | cat.unit.ym [CL]: b={b:+.4f} se={se:.4f} t={b/se:6.2f} N={m._N:,} (mean {d_na['no_access'].mean():.3f})")

d_all = df[df["inspector_badge"].isin(keep)].copy().merge(strict, on="inspector_badge")
d_all["substantive"] = d_all["outcome"].isin(["violation","no_violation"]).astype(float)
m = pf.feols("substantive ~ insp_strict | cell4", data=d_all, vcov={"CRV1":"inspector_badge"})
b, se = m.coef().loc["insp_strict"], m.se().loc["insp_strict"]
print(f"  substantive ~ inspector strictness | cat.unit.ym [CL]: b={b:+.4f} se={se:.4f} t={b/se:6.2f} N={m._N:,} (mean {d_all['substantive'].mean():.3f})")

# ---- 2. UNCONDITIONAL first stage: violation per assigned complaint ----
print("\n=== UNCONDITIONAL MARGIN first stages (LOO on each margin) ===", flush=True)
def loo_first_stage(d, label):
    st = d.groupby("inspector_badge")["y"].agg(V="sum", N="count")
    st = st[st["N"]>=30]
    d = d.merge(st, on="inspector_badge", how="inner")
    d["loo"] = (d["V"]-d["y"])/(d["N"]-1)
    m = pf.feols("y ~ loo | cell4", data=d, vcov={"CRV1":"inspector_badge"})
    b, se = m.coef().loc["loo"], m.se().loc["loo"]
    loo_i = d.groupby("inspector_badge")["y"].mean()
    print(f"  {label:<52} b={b:+.4f} se={se:.4f} t={b/se:6.2f} N={m._N:,}")
    print(f"      inspector rate spread: P10={loo_i.quantile(.1):.3f} P50={loo_i.quantile(.5):.3f} P90={loo_i.quantile(.9):.3f}")

# (a) replicate conditional margin
d1 = df[df["outcome"].isin(["violation","no_violation"]) & df["inspector_badge"].isin(keep)].copy()
d1["y"] = d1["violation_found"].astype(float)
loo_first_stage(d1, "violation | substantive only (published margin)")
# (b) include no-access as zeros
d2 = df[df["outcome"].isin(["violation","no_violation","no_access"]) & df["inspector_badge"].isin(keep)].copy()
d2["y"] = d2["violation_found"].astype(float)
loo_first_stage(d2, "violation | substantive + no-access as 0")
# (c) fully unconditional: all assigned complaints
d3 = df[df["inspector_badge"].isin(keep)].copy()
d3["y"] = d3["violation_found"].astype(float)
loo_first_stage(d3, "violation | ALL assigned complaints as 0")

# ---- 3. DISPATCH: does case severity/priority predict inspector strictness? ----
print("\n=== DISPATCH: priority composition vs strictness within unit x ym ===", flush=True)
d1 = d1.merge(strict, on="inspector_badge")
d1["pri_A"] = (d1["priority"].str.strip().str.upper()=="A").astype(float)
d1["pri_AB"] = d1["priority"].str.strip().str.upper().isin(["A","B"]).astype(float)
for yv in ["pri_A","pri_AB"]:
    m = pf.feols(f"{yv} ~ insp_strict | unit_ym", data=d1, vcov={"CRV1":"inspector_badge"})
    b, se = m.coef().loc["insp_strict"], m.se().loc["insp_strict"]
    print(f"  {yv} ~ strictness | unit x ym [CL]: b={b:+.4f} se={se:.4f} t={b/se:6.2f} (mean {d1[yv].mean():.3f})")

# inspector-level: volume and tenure vs strictness
prof = d1.groupby("inspector_badge").agg(strict=("violation_found","mean"),
                                         n=("violation_found","count"),
                                         first=("year_month","min"), last=("year_month","max"),
                                         months=("year_month","nunique"))
r1, p1 = sps.pearsonr(prof["strict"], np.log(prof["n"]))
r2, p2 = sps.pearsonr(prof["strict"], prof["months"])
print(f"  inspector-level corr(strictness, log case volume): r={r1:+.3f} p={p1:.2g}")
print(f"  inspector-level corr(strictness, months active):   r={r2:+.3f} p={p2:.2g}")
print(f"\ndone in {time.time()-t0:.0f}s")
