"""
SCRATCH sensitivity check (not a published pipeline script).

Question: do the published ownership effects (LLC, owner-occupied, absentee/
outside-NYC) on complaints and violations survive once commercial building
EXPOSURE is accounted for? The published models control only for a binary
`mixed_use` flag and `log2_area_per_unit`; they never include a commercial-
unit count, total floor area, commercial floor area, or a commercial class.

Five specifications per outcome, ownership coefficients reported for each. All
use exact residential unit-count (`size_bin`) + census-tract (`bct2020`) fixed
effects and SEs clustered by tract, matching the published owner-augmented spec.

  1  CURRENT/PUBLISHED  -- BUILDING_COVARS + OWNER_COVARS (incl. binary
     `mixed_use` and `log2_area_per_unit`). Reproduces the committed estimates.
  2  LINEAR +COMMERCIAL -- spec 1 plus unitscom = max(unitstotal-unitsres,0)
     (LINEAR), log(bldgarea), and a commercial/mixed-use class dummy
     (bldgclass in S/K/O). Full universe.
  2b PARALLEL comm_bin FE -- treats commercial units SYMMETRICALLY to
     residential units: `comm_bin` bins unitscom exactly as size_bin bins
     unitsres (exact 0..10, then 11-15/16-25/.../251+) and enters as FIXED
     EFFECTS alongside size_bin + bct2020, plus the S/K/O class dummy. The
     binary `mixed_use` is dropped (subsumed by comm_bin FE). Full universe.
  2c +COMAREA (gold standard) -- spec 1 plus log1p(comarea), the TRUE
     commercial floor area (PLUTO 64uk-42ks), pulled for the full residential
     universe. Captures commercial SPACE even when a storefront is not counted
     as a PLUTO "unit". Lots not matched to current PLUTO comarea are DROPPED
     (comarea not assumed 0); match rate reported.
  3  RESIDENTIAL-ONLY SAMPLE -- spec-1 covariates, sample restricted to
     unitstotal == unitsres AND bldgclass in {A,B,C,D}.

Reads property_risk_panel_v2.csv.gz; DOB-ledger count joined from the DB opened
READ-ONLY; comarea pulled over HTTP from Socrata and cached to a plain CSV
(never the DB). Never touches the browser/scrape or any published model script.
Writes data/analysis/owner_commercial_sensitivity.{md,csv}.
"""

import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

REPO = Path("/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import config
import dob_ledger

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
OUT_MD = config.DATA_DIR / "analysis" / "owner_commercial_sensitivity.md"
OUT_CSV = config.DATA_DIR / "analysis" / "owner_commercial_sensitivity.csv"
COMAREA_ALL_CACHE = config.DATA_DIR / "analysis" / "pluto_comarea_all_res.csv"
NDOB_CACHE = Path("/tmp/ndobviol_2020on_cache.csv")   # scratch cache, not in repo
PLUTO_API = "https://data.cityofnewyork.us/resource/64uk-42ks.json"

BUILDING_COVARS = [
    "llc", "corp_other", "trust_estate", "nycha", "govt",
    "owner_occ_star", "is_coop", "is_condo",
    "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "mixed_use", "mzone", "multi_bldg",
    "log2_area_per_unit", "value_rank", "any_prior_viol",
]
OWNER_COVARS = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner"]
CAT_SHARES = ["sh_conv", "sh_constr", "sh_elev", "sh_boiler"]
OWNER_VARS = ["llc", "owner_occ_star", "geo_outside_nyc"]

# exact-count-then-binned scheme for residential units (build_risk_dataset.py:314)
SIZE_BINS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 50, 100, 250, 100000]
SIZE_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
               "11-15", "16-25", "26-50", "51-100", "101-250", "251+"]
# parallel scheme for commercial units: exact 0..10 then the same coarse bins
COMM_BINS = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 50, 100, 250, 100000]
COMM_LABELS = ["0"] + SIZE_LABELS

PUBLISHED = {  # committed spec-1 coefficients (owner_/citation_tidy_estimates.csv)
    ("ncomp", "llc"): 0.4242, ("ncomp", "owner_occ_star"): -0.0045, ("ncomp", "geo_outside_nyc"): 0.0752,
    ("anyc", "llc"): 8.9322, ("anyc", "owner_occ_star"): -0.4191, ("anyc", "geo_outside_nyc"): 1.6253,
    ("dobviol", "llc"): 0.3163, ("dobviol", "owner_occ_star"): -0.2471, ("dobviol", "geo_outside_nyc"): 0.0664,
    ("violrate", "llc"): 1.1561, ("violrate", "owner_occ_star"): -2.8199, ("violrate", "geo_outside_nyc"): 0.3896,
    ("ecb", "llc"): 0.6112, ("ecb", "owner_occ_star"): -0.0132, ("ecb", "geo_outside_nyc"): 0.1110,
}


# ---- Spec 2c: true commercial floor area (comarea), full residential universe ----
def fetch_comarea_all() -> pd.DataFrame:
    if COMAREA_ALL_CACHE.exists():
        df = pd.read_csv(COMAREA_ALL_CACHE, dtype={"bbl_key": str})
        df["comarea"] = pd.to_numeric(df["comarea"], errors="coerce")
        print(f"comarea: using cache {COMAREA_ALL_CACHE} ({len(df):,} lots)")
        return df.drop_duplicates("bbl_key")[["bbl_key", "comarea"]]
    rows, offset = {}, 0
    print("comarea: pulling full residential universe (unitsres>=1) from Socrata 64uk-42ks ...")
    while True:
        params = {"$select": "bbl,comarea", "$where": "unitsres >= 1",
                  "$limit": 50000, "$offset": offset, "$order": "bbl"}
        url = PLUTO_API + "?" + urllib.parse.urlencode(params)
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    data = json.load(r)
                break
            except Exception:  # network flakiness -> retry with backoff
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        if not data:
            break
        for d in data:
            b = d.get("bbl")
            if b is None:
                continue
            try:
                rows[str(int(float(b)))] = d.get("comarea")
            except (TypeError, ValueError):
                continue
        offset += 50000
        if offset % 250000 == 0 or len(data) < 50000:
            print(f"  comarea pull: {len(rows):,} lots (offset {offset:,})")
        if len(data) < 50000:
            break
    df = pd.DataFrame({"bbl_key": list(rows), "comarea": list(rows.values())})
    COMAREA_ALL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMAREA_ALL_CACHE, index=False)
    print(f"comarea: pulled {len(df):,} residential lots -> {COMAREA_ALL_CACHE}")
    df["comarea"] = pd.to_numeric(df["comarea"], errors="coerce")
    return df.drop_duplicates("bbl_key")[["bbl_key", "comarea"]]


def load() -> pd.DataFrame:
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str},
                     low_memory=False)
    df["bbl_key"] = df["bbl_key"].astype(str)

    for t in ["llc", "corp_other", "trust_estate", "nycha", "govt"]:
        df[t] = (df["owner_type"] == t).astype(int)
    df = df[df["owner_type"] != "missing"].copy()
    for b in ["owner_occ_star", "is_coop", "is_condo"]:
        df[b] = df[b].astype(int)

    yb = pd.to_numeric(df["yearbuilt"], errors="coerce")
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (pd.to_numeric(df["numbldgs"], errors="coerce") >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(pd.to_numeric(df["area_per_unit"], errors="coerce"))

    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)

    df["any100"] = df["any_complaint"].astype(float) * 100.0
    df["anyecb100"] = df["any_ecb_2020on"].astype(float) * 100.0
    ns = pd.to_numeric(df["n_substantive"], errors="coerce")
    nvd = pd.to_numeric(df["n_viol_disp"], errors="coerce")
    nc = pd.to_numeric(df["n_complaints"], errors="coerce")
    with np.errstate(invalid="ignore", divide="ignore"):
        df["violrate100"] = np.where(ns > 0, nvd / ns * 100.0, np.nan)
        for g in ["conv", "constr", "elev", "boiler"]:
            df[f"sh_{g}"] = np.where(nc > 0, pd.to_numeric(df[f"n_{g}"], errors="coerce") / nc, 0.0)

    # commercial exposure
    ut = pd.to_numeric(df["unitstotal"], errors="coerce")
    ur = pd.to_numeric(df["unitsres"], errors="coerce")
    df["unitscom"] = np.maximum(ut - ur, 0).fillna(0.0)                    # linear (spec 2)
    df["comm_bin"] = pd.cut(df["unitscom"], bins=COMM_BINS,
                            labels=COMM_LABELS).astype(str)                 # parallel FE (spec 2b)
    ba = pd.to_numeric(df["bldgarea"], errors="coerce")
    df["log_bldgarea"] = np.log(ba.where(ba > 0))
    cls0 = df["bldgclass"].astype(str).str[0]
    df["com_class"] = cls0.isin(["S", "K", "O"]).astype(int)               # storefront/office/mixed class
    df["res_only"] = ((ut == ur) & cls0.isin(["A", "B", "C", "D"])).fillna(False)

    # published estimation sample
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    ok = df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])
    df = df[ok].copy()

    # gold-standard comarea (spec 2c): join true commercial floor area
    com = fetch_comarea_all()
    df = df.merge(com, on="bbl_key", how="left")
    df["comarea_matched"] = df["comarea"].notna()
    df["log1p_comarea"] = np.log1p(df["comarea"])                          # NaN where unmatched

    n = len(df)
    print(f"panel complete-case sample: {n:,}")
    print(f"  res-only subsample: {int(df['res_only'].sum()):,} ({df['res_only'].mean():.1%})")
    print(f"  comarea matched: {int(df['comarea_matched'].sum()):,} "
          f"({df['comarea_matched'].mean():.2%}); unmatched dropped in spec 2c: "
          f"{int((~df['comarea_matched']).sum()):,}")
    m = df["comarea_matched"]
    print(f"  comarea among matched: >0 in {(df.loc[m,'comarea']>0).mean():.1%} of lots; "
          f"mean {df.loc[m,'comarea'].mean():.0f} sqft; "
          f"mean among res-only {df.loc[m & df['res_only'],'comarea'].mean():.0f} sqft")
    print(f"  commercial-unit dist (comm_bin!='0'): {(df['comm_bin']!='0').mean():.1%} of lots")
    return df


def add_dobviol(df: pd.DataFrame) -> pd.DataFrame:
    if NDOB_CACHE.exists():
        counts = (pd.read_csv(NDOB_CACHE, dtype={"bbl_key": str})
                  .set_index("bbl_key")["n_dobviol_2020on"])
        print(f"DOB-ledger: using scratch cache {NDOB_CACHE}")
    else:
        uri = f"file:{config.DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=120)
        conn.execute("PRAGMA busy_timeout=120000;")
        try:
            u = dob_ledger.union_frame(conn, verbose=True)
        finally:
            conn.close()
        u = u[(u["year"] >= 2020) & (u["year"] <= 2026)]
        counts = u.groupby("bbl_key").size().rename("n_dobviol_2020on")
        counts.reset_index().to_csv(NDOB_CACHE, index=False)
    df = df.merge(counts.rename("n_dobviol_2020on"), on="bbl_key", how="left")
    df["n_dobviol_2020on"] = df["n_dobviol_2020on"].fillna(0).astype(int)
    print(f"DOB-ledger 2020-26: mean {df['n_dobviol_2020on'].mean():.4f}/lot, "
          f"any {(df['n_dobviol_2020on']>0).mean():.4f}")
    return df


OUTCOMES = [
    dict(key="ncomp", label="Complaint count", kind="pois", y="n_complaints", scale="irr", sample="full"),
    dict(key="anyc", label="Any complaint", kind="ols", y="any100", scale="pp", sample="full"),
    dict(key="dobviol", label="DOB-ledger violation count", kind="pois", y="n_dobviol_2020on", scale="irr", sample="full"),
    dict(key="violrate", label="Violations / substantive inspection", kind="ols", y="violrate100",
         scale="pp", sample="inspected", weights="n_substantive", cats=True),
    dict(key="ecb", label="ECB citation count", kind="pois", y="n_ecb_2020on", scale="irr", sample="full"),
]

# spec 1 | 2 linear | 2b parallel-FE | 2c comarea | 3 residential-only
SPEC_DEFS = [
    dict(key="s1", name="Spec 1 (current)", add=[], drop=[], fe="size_bin + bct2020", resonly=False, need=[]),
    dict(key="s2", name="Spec 2 (linear +comm)", add=["unitscom", "log_bldgarea", "com_class"],
         drop=[], fe="size_bin + bct2020", resonly=False, need=[]),
    dict(key="s2b", name="Spec 2b (comm_bin FE + class)", add=["com_class"], drop=["mixed_use"],
         fe="size_bin + comm_bin + bct2020", resonly=False, need=[]),
    dict(key="s2c", name="Spec 2c (+log1p comarea)", add=["log1p_comarea"], drop=[],
         fe="size_bin + bct2020", resonly=False, need=["comarea_matched"]),
    dict(key="s3", name="Spec 3 (residential-only)", add=[], drop=[], fe="size_bin + bct2020",
         resonly=True, need=[]),
]
SPEC_ORDER = [s["key"] for s in SPEC_DEFS]


def drop_zero_var(data, covars):
    keep, dropped = [], []
    for c in covars:
        if data[c].nunique(dropna=True) >= 2:
            keep.append(c)
        else:
            dropped.append(c)
    return keep, dropped


def fit(oc, spec, base):
    d = base
    if spec["resonly"]:
        d = d[d["res_only"]]
    for col in spec["need"]:
        d = d[d[col]]
    covars = [c for c in (BUILDING_COVARS + OWNER_COVARS) if c not in spec["drop"]] + spec["add"]
    if oc.get("cats"):
        covars = covars + CAT_SHARES
    covars, dropped = drop_zero_var(d, covars)
    X = " + ".join(covars)
    fml = f"{oc['y']} ~ {X} | {spec['fe']}"
    if oc["kind"] == "pois":
        m = pf.fepois(fml, data=d, vcov={"CRV1": "bct2020"})
    else:
        m = pf.feols(fml, data=d, vcov={"CRV1": "bct2020"}, weights=oc.get("weights"))
    return m, dropped


def effect(scale, b):
    return (np.exp(b) - 1.0) * 100.0 if scale == "irr" else b


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


DF_CACHE = Path("/tmp/owner_sens_df.pkl")            # prepared analysis frame (scratch)
PROG_CACHE = Path("/tmp/owner_sens_progress.pkl")    # per-outcome results (scratch)
BUDGET_S = 420  # start no new outcome after this wall-clock; keeps each call < 600s Bash cap


def get_df():
    if DF_CACHE.exists():
        print(f"loading prepared df from cache {DF_CACHE}")
        return pd.read_pickle(DF_CACHE)
    df = load()
    df = add_dobviol(df)
    df.to_pickle(DF_CACHE)
    print(f"cached prepared df -> {DF_CACHE}  ({len(df):,} rows)")
    return df


def run_outcome(oc, df, t0):
    base = df if oc["sample"] == "full" else df[df["n_substantive"] > 0]
    fitted = {}
    for spec in SPEC_DEFS:
        m, dropped = fit(oc, spec, base)
        fitted[spec["key"]] = m
        print(f"[{oc['key']:8} {spec['key']:4}] N={m._N:,} ({time.time()-t0:.0f}s)"
              + (f"  dropped: {dropped}" if dropped else ""))
    rows, sanity = [], []
    for ov in OWNER_VARS:
        rec = dict(outcome=oc["label"], okey=oc["key"], owner=ov, scale=oc["scale"])
        for k in SPEC_ORDER:
            m = fitted[k]
            rec[f"{k}_b"] = float(m.coef().get(ov, np.nan))
            rec[f"{k}_se"] = float(m.se().get(ov, np.nan))
            rec[f"{k}_p"] = float(m.pvalue().get(ov, np.nan))
            rec[f"{k}_eff"], rec[f"{k}_N"] = effect(oc["scale"], rec[f"{k}_b"]), int(m._N)
        b1 = rec["s1_b"]
        for k in SPEC_ORDER[1:]:
            rec[f"{k}_dpct"] = ((rec[f"{k}_b"] - b1) / abs(b1) * 100.0) if abs(b1) > 0.02 else np.nan
        rows.append(rec)
        pub = PUBLISHED.get((oc["key"], ov))
        sanity.append(dict(outcome=oc["key"], owner=ov, published=pub, reproduced=b1,
                           diff=(b1 - pub) if pub is not None else np.nan))
    return rows, sanity


def main():
    t0 = time.time()
    df = get_df()

    if PROG_CACHE.exists():
        prog = pd.read_pickle(PROG_CACHE)
        rows, sanity, done = prog["rows"], prog["sanity"], set(prog["done"])
        print(f"resuming; outcomes already done: {sorted(done)}")
    else:
        rows, sanity, done = [], [], set()

    for oc in OUTCOMES:
        if oc["key"] in done:
            continue
        if time.time() - t0 > BUDGET_S:
            print(f"time budget hit before {oc['key']}; saving progress and exiting")
            break
        r, s = run_outcome(oc, df, t0)
        rows += r
        sanity += s
        done.add(oc["key"])
        pd.to_pickle({"rows": rows, "sanity": sanity, "done": list(done)}, PROG_CACHE)
        print(f"  saved progress: {len(done)}/{len(OUTCOMES)} outcomes")

    if done >= {o["key"] for o in OUTCOMES}:
        res = pd.DataFrame(rows)
        res.to_csv(OUT_CSV, index=False)
        san = pd.DataFrame(sanity)
        print("\n=== SANITY CHECK: spec-1 reproduction vs committed estimates ===")
        print(san.to_string(index=False,
                            formatters={"published": "{:+.4f}".format,
                                        "reproduced": "{:+.4f}".format, "diff": "{:+.5f}".format}))
        maxdiff = san["diff"].abs().max()
        print(f"max |reproduced - published| = {maxdiff:.5f} ({'OK' if maxdiff < 0.02 else 'CHECK'})")
        decomp = decompose(df)
        write_md(res, df, maxdiff, decomp)
        print(f"\nwrote {OUT_MD}\nwrote {OUT_CSV}\nALL DONE")
    else:
        print(f"PARTIAL: {len(done)}/{len(OUTCOMES)} outcomes done; re-run to continue")


def decompose(df):
    """Which piece of commercial exposure moves the (uniquely sensitive) DOB-ledger
    count? (a) add each control ALONE to Spec 1; (b) the 2b+ variant = comm_bin FE
    + S/K/O class + log(bldgarea) for every outcome."""
    vc = {"CRV1": "bct2020"}
    base = BUILDING_COVARS + OWNER_COVARS
    out = {"oneterm": [], "twobplus": []}
    print("\n=== decomposition: one commercial term at a time on DOB-ledger count ===")
    for add, lbl in [([], "Spec 1 (baseline)"), (["com_class"], "+ S/K/O class dummy"),
                     (["log_bldgarea"], "+ log(bldgarea)"), (["unitscom"], "+ unitscom (linear)"),
                     (["log1p_comarea"], "+ log1p(comarea)")]:
        d = df[df["comarea_matched"]] if "log1p_comarea" in add else df
        covs, _ = drop_zero_var(d, base + add)
        m = pf.fepois(f"n_dobviol_2020on ~ {' + '.join(covs)} | size_bin + bct2020", data=d, vcov=vc)
        row = {"add": lbl, "N": int(m._N)}
        for ov in ["llc", "owner_occ_star"]:
            b = float(m.coef().get(ov, np.nan))
            row[ov], row[ov + "_p"] = (np.exp(b) - 1) * 100, float(m.pvalue().get(ov, np.nan))
        out["oneterm"].append(row)
        print(f"  DOB LLC {row['llc']:+6.1f}%  owner-occ {row['owner_occ_star']:+6.1f}%   {lbl}")
    print("=== 2b+ (comm_bin FE + S/K/O class + log bldgarea) across outcomes ===")
    for oc in OUTCOMES:
        d = df if oc["sample"] == "full" else df[df["n_substantive"] > 0]
        covs = [c for c in base if c != "mixed_use"] + ["com_class", "log_bldgarea"] \
            + (CAT_SHARES if oc.get("cats") else [])
        covs, _ = drop_zero_var(d, covs)
        fml = f"{oc['y']} ~ {' + '.join(covs)} | size_bin + comm_bin + bct2020"
        m = (pf.fepois(fml, data=d, vcov=vc) if oc["kind"] == "pois"
             else pf.feols(fml, data=d, vcov=vc, weights=oc.get("weights")))
        row = {"outcome": oc["label"], "okey": oc["key"], "scale": oc["scale"], "N": int(m._N)}
        for ov in OWNER_VARS:
            b = float(m.coef().get(ov, np.nan))
            row[ov], row[ov + "_p"] = effect(oc["scale"], b), float(m.pvalue().get(ov, np.nan))
        out["twobplus"].append(row)
    return out


LABELS = {"llc": "LLC (vs individual)", "owner_occ_star": "Owner-occupied (STAR)",
          "geo_outside_nyc": "Absentee (owner outside NYC)"}
MODELTAG = {"ncomp": "Poisson", "dobviol": "Poisson", "ecb": "Poisson",
            "anyc": "LPM", "violrate": "LPM, wtd"}
SPEC_HDR = {"s1": "Spec 1<br>current", "s2": "Spec 2<br>linear +comm",
            "s2b": "Spec 2b<br>comm-unit FE", "s2c": "Spec 2c<br>+comarea",
            "s3": "Spec 3<br>res-only"}


def cell(scale, eff, p):
    if np.isnan(eff):
        return "--"
    unit = "%" if scale == "irr" else "pp"
    return f"{eff:+.1f}{unit}{stars(p)}"


def main_table(res):
    L = ["| Outcome (model) | Ownership | " + " | ".join(SPEC_HDR[k] for k in SPEC_ORDER) + " |",
         "|---|---|" + "---|" * len(SPEC_ORDER)]
    order = {o["key"]: i for i, o in enumerate(OUTCOMES)}
    res = res.sort_values(by=["okey", "owner"],
                          key=lambda s: s.map(order) if s.name == "okey"
                          else s.map({v: i for i, v in enumerate(OWNER_VARS)}))
    last = None
    for _, r in res.iterrows():
        lbl = f"{r['outcome']} ({MODELTAG[r['okey']]})" if r["okey"] != last else ""
        last = r["okey"]
        cells = " | ".join(cell(r["scale"], r[f"{k}_eff"], r[f"{k}_p"]) for k in SPEC_ORDER)
        L.append(f"| {lbl} | {LABELS[r['owner']]} | {cells} |")
    return L


def write_md(res, df, maxdiff, decomp=None):
    L = ["# Ownership effects vs. commercial-exposure controls (five specifications)\n"]
    L.append("Each ownership coefficient estimated under five specs; all use exact "
             "residential unit-count (`size_bin`) + census-tract (`bct2020`) fixed effects "
             "and tract-clustered SEs (the published owner-augmented spec).\n")
    L.append("| Spec | What it adds to the published spec |\n|---|---|")
    L.append("| **1 current** | `BUILDING_COVARS + OWNER_COVARS`: binary `mixed_use` + `log2_area_per_unit` only (reproduces committed estimates). |")
    L.append("| **2 linear** | + `unitscom = max(unitstotal-unitsres,0)` (linear), `log(bldgarea)`, S/K/O class dummy. Full universe. |")
    L.append("| **2b comm-unit FE** | commercial units treated SYMMETRICALLY to residential: `comm_bin` (exact 0..10 then binned, mirroring `size_bin`) as FIXED EFFECTS + S/K/O class dummy; `mixed_use` dropped (subsumed). Full universe. |")
    L.append("| **2c +comarea** | + `log1p(comarea)`, true commercial floor area (PLUTO 64uk-42ks, full-universe pull). Captures commercial space even when the storefront is not a counted PLUTO unit. |")
    L.append("| **3 res-only** | spec-1 covariates on the sample `unitstotal==unitsres` AND `bldgclass in {A,B,C,D}`. |\n")
    L.append(f"Spec-1 reproduction: max |reproduced - published| = **{maxdiff:.4f}** across all 15 "
             f"owner coefficients (exact). Cells show the effect on the post's native scale - Poisson "
             f"counts as incidence-rate change `(exp(b)-1)`, LPMs in percentage points `pp` - with "
             f"significance `*`/`**`/`***` = p<.05/.01/.001. Coefficient SEs, p-values, and % movement "
             f"vs spec 1 are in `owner_commercial_sensitivity.csv`.\n")

    nmatch = int(df["comarea_matched"].sum())
    L.append(f"**Spec 2c comarea join:** matched {nmatch:,} of {len(df):,} lots "
             f"({df['comarea_matched'].mean():.1%}) to current PLUTO commercial floor area; the "
             f"{int((~df['comarea_matched']).sum()):,} unmatched lots are DROPPED from spec 2c "
             f"(comarea not assumed 0). Among matched lots comarea>0 for "
             f"{(df.loc[df['comarea_matched'],'comarea']>0).mean():.1%}.\n")

    L += main_table(res)

    # focused coefficient detail for the sensitive outcome
    L.append("\n### DOB-ledger violation count - coefficient detail (the confounded outcome)\n")
    L.append("Raw Poisson coefficient `b (se)` [IRR%], LLC and owner-occupied, across the five specs.\n")
    L.append("| Ownership | " + " | ".join(SPEC_HDR[k].replace("<br>", " ") for k in SPEC_ORDER) + " |")
    L.append("|---|" + "---|" * len(SPEC_ORDER))
    dv = res[res["okey"] == "dobviol"].set_index("owner")
    for ov in ["llc", "owner_occ_star"]:
        r = dv.loc[ov]
        cs = " | ".join(f"{r[f'{k}_b']:+.3f} ({r[f'{k}_se']:.3f}){stars(r[f'{k}_p'])} "
                        f"[{r[f'{k}_eff']:+.0f}%]" for k in SPEC_ORDER)
        L.append(f"| {LABELS[ov]} | {cs} |")

    if decomp:
        L.append("\n### What drives the DOB-ledger attenuation? (mechanism)\n")
        L.append("The DOB-ledger count is the only sensitive outcome. Adding each commercial "
                 "control ALONE to Spec 1 shows the operative confounder is the commercial/"
                 "mixed-use building **class** (S/K/O) and total **floor area** - NOT commercial "
                 "units or commercial floor area, which move the coefficient ~0.\n")
        L.append("| Added to Spec 1 (alone) | DOB-ledger LLC | DOB-ledger owner-occupied |")
        L.append("|---|---|---|")
        for r in decomp["oneterm"]:
            L.append(f"| {r['add']} | {r['llc']:+.1f}%{stars(r['llc_p'])} | "
                     f"{r['owner_occ_star']:+.1f}%{stars(r['owner_occ_star_p'])} |")
        L.append("\n`2b+` = comm_bin FE + S/K/O class + `log(bldgarea)` (the parallel-FE spec plus "
                 "total floor area) across all outcomes:\n")
        L.append("| Outcome | LLC | Owner-occupied | Absentee |")
        L.append("|---|---|---|---|")
        for r in decomp["twobplus"]:
            u = "%" if r["scale"] == "irr" else "pp"
            L.append(f"| {r['outcome']} | {r['llc']:+.1f}{u}{stars(r['llc_p'])} | "
                     f"{r['owner_occ_star']:+.1f}{u}{stars(r['owner_occ_star_p'])} | "
                     f"{r['geo_outside_nyc']:+.1f}{u}{stars(r['geo_outside_nyc_p'])} |")

    L.append("\n**Sample sizes (N):**\n")
    L.append("| Outcome | " + " | ".join(SPEC_HDR[k].replace("<br>", " ") for k in SPEC_ORDER) + " |")
    L.append("|---|" + "---:|" * len(SPEC_ORDER))
    for oc in OUTCOMES:
        r = res[res["okey"] == oc["key"]].iloc[0]
        L.append(f"| {oc['label']} | " + " | ".join(f"{int(r[f'{k}_N']):,}" for k in SPEC_ORDER) + " |")

    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
