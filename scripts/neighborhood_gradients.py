"""Neighborhood gradients: how DOB caller complaints, ECB citations, scheduled-inspection
violations, and inspection hit rates vary with census-tract demographics, with and
without building-stock controls.

Estimands (the post reports all of them side by side; none is causal):
  total    outcome ~ demographic | size_bin + comm_bin + borough
           building size held fixed as exposure, nothing else
  direct   total + building-stock controls (era, within-tract value rank, ownership,
           use/size detail, prior enforcement history)
  gelbach  exact, order-invariant decomposition of (total - direct) for the
           any-caller-complaint linear probability model into control groups
           (Gelbach 2016), so the reader sees how much of a gradient runs through
           observable stock without pretending that settles mediation vs confounding
  hitrate  complaint-level LPM: violation found among caller complaints where the
           inspector accessed the property, with category/size/borough FE, then
           + inspector FE (same inspector, complaints from richer vs poorer tracts),
           then + building controls
  noaccess complaint-level LPM: inspector could not get in

Demographics are 2023 ACS 5-year tract shares (per +10 percentage points) and log
median income (per +1 SD), attached to the committed panel by build_risk_dataset.py.
Caller complaints are complaints with a 311 reference number (the definition used in
post0_descriptive_stats.py); the rest are agency-initiated. Scheduled-inspection
violations are the deduplicated BIS + DOB NOW union (dob_ledger.py), 2020 - May 2026.

Outputs (data/analysis/risk_models/): neighborhood_gradients.csv,
  neighborhood_decomposition.csv, neighborhood_hitrate.csv, neighborhood_descriptives.csv
Committed extracts so the results notebook can refit without the 8 GB database:
  data/analysis/neighborhood_lot_outcomes.csv.gz  (bbl_key, n_caller, n_agency, n_dobviol_2020on)
  data/analysis/neighborhood_complaints.csv.gz    (complaint level: bbl_key, category, outcome,
                                                   inspector_badge, caller, year_month)
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import gc

LEAN = dict(lean=True, store_data=False, copy_data=False)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
import dob_ledger  # noqa: E402
from analysis_config import make_bbl  # noqa: E402
from disposition_codes import classify_disposition  # noqa: E402

DATA = config.DATA_DIR / "analysis"
RM = DATA / "risk_models"
PANEL = DATA / "property_risk_panel_v2.csv.gz"
LOT_EXTRACT = DATA / "neighborhood_lot_outcomes.csv.gz"
CPL_EXTRACT = DATA / "neighborhood_complaints.csv.gz"
WINDOW = ("2020-01-01", "2026-05-31")
YEARS_C = 77 / 12          # complaint and DOB-record window, Jan 2020 - May 2026
YEARS_E = 6.25             # ECB window, Jan 2020 - Mar 2026 (as in citation_descriptives)
DTYPE = {"bct2020": str, "size_bin": str, "borocode": str, "bbl_key": str}
FE_B = "size_bin + comm_bin + borocode"
VCOV = {"CRV1": "bct2020"}
MIN_INSPECTOR_CASES = 30

DEMOS = {
    "tract_poverty10": "Tract poverty rate (+10 pp)",
    "tract_log_income_z": "Tract log median income (+1 SD)",
    "tract_renter10": "Tract renter share (+10 pp)",
    "tract_foreign10": "Tract foreign-born share (+10 pp)",
    "tract_overcrowd10": "Tract overcrowded-household share (+10 pp)",
    "tract_black10": "Tract Black share (+10 pp)",
    "tract_hispanic10": "Tract Hispanic share (+10 pp)",
    "tract_asian10": "Tract Asian share (+10 pp)",
}
GROUPS = {
    "era": ["era_pre1940", "era_4079", "era_8099", "era_unknown"],
    "value": ["value_rank_tract"],
    "ownership": ["llc", "corp_other", "trust_estate", "nycha", "govt", "owner_occ_star",
                  "is_coop", "is_condo", "geo_nyc_other", "geo_outside_nyc", "geo_unknown",
                  "multi_prop_owner"],
    "use_size": ["com_class", "log_bldgarea", "log2_area_per_unit", "mzone", "multi_bldg"],
    "history": ["any_prior_viol"],
}
CONTROLS = [c for g in GROUPS.values() for c in g]
X = " + ".join(CONTROLS)
OUTCOMES = {
    "n_caller": "caller complaints",
    "n_agency": "agency-initiated complaints",
    "n_ecb_2020on": "ECB citations",
    "n_dobviol_2020on": "DOB violation records (scheduled inspections)",
}


def log(msg):
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"[{time.strftime('%H:%M:%S')} rss {rss:.1f}G] {msg}", flush=True)


# ── 1. frame (mirrors build_risk_notebook.py SETUP) ───────────────────────

def build_frame() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype=DTYPE, low_memory=False)
    for t in ["llc", "corp_other", "trust_estate", "nycha", "govt"]:
        df[t] = (df["owner_type"] == t).astype(int)
    df = df[df["owner_type"] != "missing"].copy()
    for b in ["owner_occ_star", "is_coop", "is_condo"]:
        df[b] = df[b].astype(int)
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])
    ut = pd.to_numeric(df["unitstotal"], errors="coerce")
    ur = pd.to_numeric(df["unitsres"], errors="coerce")
    df["unitscom"] = np.maximum(ut - ur, 0).fillna(0.0)
    cb = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 50, 100, 250, 100000]
    cl = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
          "11-15", "16-25", "26-50", "51-100", "101-250", "251+"]
    df["comm_bin"] = pd.cut(df["unitscom"], bins=cb, labels=cl).astype(str)
    ba = pd.to_numeric(df["bldgarea"], errors="coerce")
    df["log_bldgarea"] = np.log(ba.where(ba > 0))
    df["com_class"] = df["bldgclass"].astype(str).str[0].isin(["S", "K", "O"]).astype(int)
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    df["mzone"] = pd.to_numeric(df["mzone"], errors="coerce").fillna(0).astype(int)
    # assessed value per unit ranked WITHIN the tract, so the value control measures a
    # building's standing relative to its neighbors and cannot absorb between-tract
    # income differences by construction
    vpu = (df["assesstot"] / df["unitsres"]).where((df["assesstot"] > 0) & (df["unitsres"] > 0))
    df["value_rank_tract"] = vpu.groupby(df["bct2020"]).rank(pct=True)
    # demographics: shares 0-1 in the panel -> per +10 pp; log income -> z
    for share in ["tract_poverty", "tract_renter_share", "tract_foreign_born", "tract_overcrowd",
                  "tract_pct_black", "tract_pct_hispanic", "tract_pct_asian"]:
        assert df[share].dropna().between(0, 1).all(), share
    df["tract_poverty10"] = df["tract_poverty"] * 10
    df["tract_renter10"] = df["tract_renter_share"] * 10
    df["tract_foreign10"] = df["tract_foreign_born"] * 10
    df["tract_overcrowd10"] = df["tract_overcrowd"] * 10
    df["tract_black10"] = df["tract_pct_black"] * 10
    df["tract_hispanic10"] = df["tract_pct_hispanic"] * 10
    df["tract_asian10"] = df["tract_pct_asian"] * 10
    li = df["tract_log_income"]
    df["tract_log_income_z"] = (li - li.mean()) / li.std()
    need = ["log2_area_per_unit", "value_rank_tract", "size_bin", "bct2020"] + list(DEMOS)
    keep = df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])
    return df[keep].copy()


# ── 2. database extracts (cached as committed csv.gz) ─────────────────────

BORO_CASE = """CASE b.borough
    WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
    WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4'
    WHEN 'STATEN ISLAND' THEN '5' END"""


def extract(conn) -> tuple[pd.DataFrame, pd.DataFrame]:
    if LOT_EXTRACT.exists() and CPL_EXTRACT.exists():
        log(f"using cached extracts {LOT_EXTRACT.name}, {CPL_EXTRACT.name}")
        return (pd.read_csv(LOT_EXTRACT, dtype={"bbl_key": str}),
                pd.read_csv(CPL_EXTRACT, dtype={"bbl_key": str, "inspector_badge": str,
                                                "complaint_category": str, "complaint_number": str}))
    log("querying complaints (open_data x bis_scrape) ...")
    df = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.date_entered, o.disposition_code, o.complaint_category,
               {BORO_CASE} AS boro_code, b.block, b.lot, b.ref_311, b.inspector_badge
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
    """, conn)
    df["entered"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y", errors="coerce")
    df = df[(df["entered"] >= WINDOW[0]) & (df["entered"] <= WINDOW[1])].copy()
    df["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt in zip(df["boro_code"], df["block"], df["lot"])]
    df = df[df["bbl_key"] != ""]
    df["caller"] = (df["ref_311"].fillna("").astype(str).str.strip() != "").astype(int)
    df["outcome"] = df["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    df["year_month"] = df["entered"].dt.strftime("%Y-%m")
    log(f"  {len(df):,} complaints in window; caller share {df['caller'].mean():.3f}")
    lots = df.groupby("bbl_key").agg(n_caller=("caller", "sum"),
                                      n_agency=("caller", lambda s: int((s == 0).sum()))).reset_index()
    log("building DOB violation union ...")
    u = dob_ledger.union_frame(conn)
    w = u[(u["ymd"] >= WINDOW[0].replace("-", "")) & (u["ymd"] <= WINDOW[1].replace("-", ""))]
    dob = w.groupby("bbl_key").size().rename("n_dobviol_2020on").reset_index()
    log(f"  union entries in window: {len(w):,} on {len(dob):,} lots")
    lots = lots.merge(dob, on="bbl_key", how="outer").fillna(0)
    for c in ["n_caller", "n_agency", "n_dobviol_2020on"]:
        lots[c] = lots[c].astype(int)
    cpl = df[["complaint_number", "bbl_key", "complaint_category", "outcome",
              "inspector_badge", "caller", "year_month"]].copy()
    lots.to_csv(LOT_EXTRACT, index=False)
    cpl.to_csv(CPL_EXTRACT, index=False)
    log(f"  wrote {LOT_EXTRACT.name} ({len(lots):,} lots) and {CPL_EXTRACT.name} ({len(cpl):,} complaints)")
    return lots, cpl


# ── 3. helpers ────────────────────────────────────────────────────────────

def irr_row(m, term):
    b = float(m.coef()[term]); se = float(m.se()[term]); lo, hi = m.confint().loc[term].values
    return dict(estimate=b, std_error=se, ci_lo=float(lo), ci_hi=float(hi),
                pct_change=(np.exp(b) - 1) * 100, pct_lo=(np.exp(lo) - 1) * 100,
                pct_hi=(np.exp(hi) - 1) * 100, p=float(m.pvalue()[term]), n=int(m._N))


def pp_row(m, term):
    b = float(m.coef()[term]); se = float(m.se()[term]); lo, hi = m.confint().loc[term].values
    return dict(estimate=b, std_error=se, ci_lo=float(lo), ci_hi=float(hi),
                p=float(m.pvalue()[term]), n=int(m._N))


# ── 4. checkpoints ────────────────────────────────────────────────────────
# Every fit appends its rows to a JSON-lines checkpoint the moment it finishes, so an
# interrupted run resumes where it stopped and a finished stage is never recomputed.

import json

CK = RM / "checkpoints"


class Checkpoint:
    def __init__(self, name):
        CK.mkdir(exist_ok=True)
        self.path = CK / f"{name}.jsonl"
        self.rows, self.done = [], set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line); self.rows.extend(rec["rows"]); self.done.add(rec["key"])
            log(f"  checkpoint {self.path.name}: {len(self.done)} keys done")

    def add(self, key, rows):
        rows = [{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer,)) else v))
                 for k, v in r.items()} for r in rows]
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "rows": rows}) + "\n")
        self.rows.extend(rows); self.done.add(key)

    def frame(self):
        return pd.DataFrame(self.rows)


# ── 5. main ───────────────────────────────────────────────────────────────

def main():
    # stages: gradients, gelbach, hitrate, descriptives (default all). A stage whose CSV
    # already exists is skipped unless FORCE=1 is set in the environment.
    import os
    stages = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {"gradients", "gelbach", "hitrate", "descriptives"}
    force = os.environ.get("FORCE") == "1"
    def todo(stage, csv):
        if stage not in stages:
            return False
        if (RM / csv).exists() and not force:
            log(f"{csv} exists; skipping stage {stage} (FORCE=1 to redo)"); return False
        return True

    conn = sqlite3.connect(str(config.DB_PATH))
    lots, cpl = extract(conn)
    conn.close()
    frame = build_frame()
    frame = frame.merge(lots, on="bbl_key", how="left")
    for c in ["n_caller", "n_agency", "n_dobviol_2020on"]:
        frame[c] = frame[c].fillna(0).astype(int)
    frame["any_caller100"] = (frame["n_caller"] > 0).astype(float) * 100
    est = frame.dropna(subset=CONTROLS + list(OUTCOMES)).copy()
    log(f"estimation frame: {len(est):,} lots ({len(frame):,} before control dropna)")

    # ---- gradients: PPML, total and direct, bivariate and joint ----------
    if todo("gradients", "neighborhood_gradients.csv"):
        ck = Checkpoint("gradients")
        for y, ylab in OUTCOMES.items():
            for d, dlab in DEMOS.items():
                key = f"{y}|bivariate|{d}"
                if key in ck.done:
                    continue
                rows = []
                for spec, rhs in [("total", d), ("direct", f"{d} + {X}")]:
                    m = pf.fepois(f"{y} ~ {rhs} | {FE_B}", data=est, vcov=VCOV, **LEAN)
                    rows.append(dict(outcome=y, outcome_label=ylab, spec=spec, entry="bivariate",
                                     term=d, term_label=dlab, **irr_row(m, d)))
                ck.add(key, rows); gc.collect()
                log(f"  {y} / {d}: total {rows[0]['pct_change']:+.1f}%  direct {rows[1]['pct_change']:+.1f}%")
            joint = " + ".join(DEMOS)
            for spec, rhs in [("total", joint), ("direct", f"{joint} + {X}")]:
                key = f"{y}|joint|{spec}"
                if key in ck.done:
                    continue
                m = pf.fepois(f"{y} ~ {rhs} | {FE_B}", data=est, vcov=VCOV, **LEAN)
                ck.add(key, [dict(outcome=y, outcome_label=ylab, spec=spec, entry="joint",
                                  term=d, term_label=dlab, **irr_row(m, d)) for d, dlab in DEMOS.items()])
            log(f"  {y}: joint specs done")
        grad = ck.frame()
        grad["fe"] = FE_B; grad["cluster"] = "bct2020"; grad["window"] = f"{WINDOW[0]}..{WINDOW[1]}"
        grad.to_csv(RM / "neighborhood_gradients.csv", index=False)
        log(f"wrote neighborhood_gradients.csv ({len(grad)} rows)")

    # ---- Gelbach decomposition on the any-caller-complaint LPM ----------
    if todo("gelbach", "neighborhood_decomposition.csv"):
        ck = Checkpoint("gelbach")
        for d, dlab in DEMOS.items():
            if d in ck.done:
                continue
            base = pf.feols(f"any_caller100 ~ {d} | {FE_B}", data=est, vcov=VCOV, **LEAN)
            full = pf.feols(f"any_caller100 ~ {d} + {X} | {FE_B}", data=est, vcov=VCOV, **LEAN)
            b_base, b_full = float(base.coef()[d]), float(full.coef()[d])
            gap = b_base - b_full
            contrib = {}
            for k in CONTROLS:
                aux = pf.feols(f"{k} ~ {d} | {FE_B}", data=est, vcov="iid", **LEAN)
                contrib[k] = float(aux.coef()[d]) * float(full.coef()[k])
            total_contrib = sum(contrib.values())
            assert np.isclose(total_contrib, gap, rtol=1e-3, atol=1e-6), (d, total_contrib, gap)
            base_r, full_r = pp_row(base, d), pp_row(full, d)
            rows = [dict(term=d, term_label=dlab, component="total", points=b_base,
                         ci_lo=base_r["ci_lo"], ci_hi=base_r["ci_hi"], share_of_total=1.0, n=base_r["n"]),
                    dict(term=d, term_label=dlab, component="direct", points=b_full,
                         ci_lo=full_r["ci_lo"], ci_hi=full_r["ci_hi"],
                         share_of_total=b_full / b_base if b_base else np.nan, n=full_r["n"])]
            for g, ks in GROUPS.items():
                v = sum(contrib[k] for k in ks)
                rows.append(dict(term=d, term_label=dlab, component=f"via_{g}", points=v,
                                 ci_lo=np.nan, ci_hi=np.nan, share_of_total=v / b_base if b_base else np.nan,
                                 n=full_r["n"]))
            ck.add(d, rows); gc.collect()
            log(f"  gelbach {d}: total {b_base:+.2f} pp, direct {b_full:+.2f} pp, "
                + ", ".join(f"{g} {sum(contrib[k] for k in ks):+.2f}" for g, ks in GROUPS.items()))
        dec = ck.frame()
        dec["outcome"] = "any caller complaint (pp)"; dec["fe"] = FE_B
        dec.to_csv(RM / "neighborhood_decomposition.csv", index=False)
        log("wrote neighborhood_decomposition.csv")

    # ---- complaint-level frames (hit rate, no access, descriptives) -----
    keep_cols = ["bbl_key", "size_bin", "comm_bin", "borocode", "bct2020"] + list(DEMOS) + CONTROLS
    c = cpl.merge(est[keep_cols], on="bbl_key", how="inner")
    c = c[(c["caller"] == 1) & c["outcome"].isin(["violation", "no_violation", "no_access"])].copy()
    c["noaccess100"] = (c["outcome"] == "no_access").astype(float) * 100
    acc = c[c["outcome"] != "no_access"].copy()
    acc["viol100"] = (acc["outcome"] == "violation").astype(float) * 100
    acc["inspector_badge"] = acc["inspector_badge"].fillna("").astype(str).str.strip()
    counts = acc["inspector_badge"].value_counts()
    ok = counts[counts >= MIN_INSPECTOR_CASES].index.difference([""])
    acc = acc[acc["inspector_badge"].isin(ok)].copy()
    log(f"hit-rate sample: {len(acc):,} accessed caller complaints, {acc['inspector_badge'].nunique():,} inspectors "
        f"(>= {MIN_INSPECTOR_CASES} cases); no-access sample: {len(c):,}; baseline hit rate {acc['viol100'].mean():.1f}%, "
        f"no-access rate {c['noaccess100'].mean():.1f}%")
    FE_H0 = "complaint_category + size_bin + comm_bin + borocode"
    FE_H1 = FE_H0 + " + inspector_badge"

    if todo("hitrate", "neighborhood_hitrate.csv"):
        ck = Checkpoint("hitrate")
        for d, dlab in DEMOS.items():
            for spec, rhs, fe, data, y in [
                ("hit_base", d, FE_H0, acc, "viol100"),
                ("hit_inspector_fe", d, FE_H1, acc, "viol100"),
                ("hit_inspector_fe_controls", f"{d} + {X}", FE_H1, acc, "viol100"),
                ("noaccess_base", d, FE_H0, c, "noaccess100"),
                ("noaccess_controls", f"{d} + {X}", FE_H0, c, "noaccess100"),
            ]:
                key = f"{d}|{spec}"
                if key in ck.done:
                    continue
                m = pf.feols(f"{y} ~ {rhs} | {fe}", data=data, vcov=VCOV, **LEAN)
                r = dict(outcome=y, spec=spec, term=d, term_label=dlab, fe=fe,
                         baseline_rate=float(data[y].mean()), **pp_row(m, d))
                ck.add(key, [r]); gc.collect()
                log(f"  {spec} {d}: {r['estimate']:+.2f} pp [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]")
        hit = ck.frame()
        hit["cluster"] = "bct2020"; hit["min_inspector_cases"] = MIN_INSPECTOR_CASES
        hit.to_csv(RM / "neighborhood_hitrate.csv", index=False)
        log("wrote neighborhood_hitrate.csv")

    if not todo("descriptives", "neighborhood_descriptives.csv"):
        return
    qrows = []
    cl = c.merge(est[["bbl_key"]], on="bbl_key")  # complaint-level, caller, with outcome
    for var, raw in [("poverty", "tract_poverty"), ("income", "med_income"),
                     ("foreign_born", "tract_foreign_born"), ("black", "tract_pct_black"),
                     ("hispanic", "tract_pct_hispanic"), ("asian", "tract_pct_asian"),
                     ("renter", "tract_renter_share"), ("overcrowd", "tract_overcrowd")]:
        q = pd.qcut(est[raw].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        est["_q"] = q.astype(int)
        cq = cl.merge(est[["bbl_key", "_q"]], on="bbl_key")
        for k, g in est.groupby("_q"):
            gq = cq[cq["_q"] == k]
            acc_k = gq[gq["outcome"] != "no_access"]
            qrows.append(dict(
                variable=var, quintile=int(k), range_lo=float(g[raw].min()), range_hi=float(g[raw].max()),
                n_lots=len(g),
                caller_per100_yr=g["n_caller"].sum() / len(g) / YEARS_C * 100,
                agency_per100_yr=g["n_agency"].sum() / len(g) / YEARS_C * 100,
                ecb_per100_yr=g["n_ecb_2020on"].sum() / len(g) / YEARS_E * 100,
                dobviol_per100_yr=g["n_dobviol_2020on"].sum() / len(g) / YEARS_C * 100,
                any_caller_share=(g["n_caller"] > 0).mean(),
                hit_rate=(acc_k["outcome"] == "violation").mean() if len(acc_k) else np.nan,
                noaccess_rate=(gq["outcome"] == "no_access").mean() if len(gq) else np.nan,
                n_caller_complaints=len(gq)))
    est.drop(columns="_q", inplace=True)
    acc_all = cl[cl["outcome"] != "no_access"]
    qrows.append(dict(variable="all", quintile=0, range_lo=np.nan, range_hi=np.nan, n_lots=len(est),
                      caller_per100_yr=est["n_caller"].sum() / len(est) / YEARS_C * 100,
                      agency_per100_yr=est["n_agency"].sum() / len(est) / YEARS_C * 100,
                      ecb_per100_yr=est["n_ecb_2020on"].sum() / len(est) / YEARS_E * 100,
                      dobviol_per100_yr=est["n_dobviol_2020on"].sum() / len(est) / YEARS_C * 100,
                      any_caller_share=(est["n_caller"] > 0).mean(),
                      hit_rate=(acc_all["outcome"] == "violation").mean(),
                      noaccess_rate=(cl["outcome"] == "no_access").mean(),
                      n_caller_complaints=len(cl)))
    desc = pd.DataFrame(qrows)
    desc.to_csv(RM / "neighborhood_descriptives.csv", index=False)
    log("wrote neighborhood_descriptives.csv")
    print(desc[desc.variable.isin(["poverty", "foreign_born", "all"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
