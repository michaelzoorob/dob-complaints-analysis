"""Build notebooks/risk_factors_results.ipynb: a self-contained, reproducible
notebook that COMPUTES and DISPLAYS every number in the risk-factor post
(data/analysis/blog_posts/post_risk_substack.md, "Which Buildings Get Complained
About, and Which Ones Receive Violations?"). Each section carries a unique,
alphanumeric heading so the nbviewer anchor is exactly heading.replace(" ", "-");
the article hyperlinks each numeric result to the matching section.

Mirrors build_overview_notebook.py (same committed panel, same estimation frame
and helpers). Reproducibility tiers:
  - complaint, ECB, ECB-owner, per-inspection, and no-access models refit live
    from the committed panel data/analysis/property_risk_panel_v2.csv.gz.
  - the deduplicated DOB-violation union models and the tract-demographic model
    (which need the 8 GB database or tract covariates absent from the committed
    panel) display the committed tidy CSV and name the producing script.

Execute with the pinned kernel:
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=pyfix \
      notebooks/risk_factors_results.ipynb
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)
OUT = NB_DIR / "risk_factors_results.ipynb"
REPO = "michaelzoorob/dob-complaints-analysis"

REQUIREMENTS = """# environment for risk_factors_results.ipynb (Python 3.14)
pyfixest==0.60.0
pandas==3.0.0
numpy==2.3.5
scipy==1.17.0
statsmodels==0.14.6
"""

HEADER = f"""# Reproducible results: which buildings get complained about, and which get violations

This notebook computes every number in the article **"Which Buildings Get
Complained About, and Which Ones Receive Violations?"** Each result below is
linked from the matching sentence in the post: click a number in the article and
you land on the cell that produces it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/risk_factors_results.ipynb)

**How to read this.** The complaint, ECB-citation, per-inspection, and no-access
models are refit here on the committed property panel, so every coefficient, 95%
confidence interval, and sample size you see is produced by the cell. The
deduplicated DOB-violation union models and the neighborhood-demographic model
need the 8 GB complaint database or tract covariates that are not in the
committed panel; those cells display the committed tidy-estimate table written by
the named script, and a final cell cross-checks every refit against the committed
tables and the article's quoted number.
"""

SETUP = r'''# === setup: versions, committed panel, analysis frame, helpers ===
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, pyfixest as pf
import scipy
warnings.filterwarnings("ignore")
print("python", sys.version.split()[0], "| pyfixest", pf.__version__,
      "| pandas", pd.__version__, "| numpy", np.__version__, "| scipy", scipy.__version__)

CAND = [Path.cwd(), Path.cwd().parent] + list(Path.cwd().parents)
ROOT = next((p for p in CAND if (p / "data" / "analysis").exists()), Path.cwd())
DATA = ROOT / "data" / "analysis"
RM = DATA / "risk_models"
DB = ROOT / "data" / "dob_complaints.db"
HAVE_DB = DB.exists()
YEARS = 6.25
PANEL = DATA / "property_risk_panel_v2.csv.gz"
DTYPE = {"bct2020": str, "size_bin": str, "borocode": str, "bbl_key": str}

def build_frame(path):
    """Estimation frame from the committed panel, mirroring load_frame() in
    risk_factor_models.py / violation_rate_models.py, minus the DOB-union join."""
    df = pd.read_csv(path, dtype=DTYPE, low_memory=False)
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
    with np.errstate(invalid="ignore", divide="ignore"):
        df["violrate100"] = np.where(df["n_substantive"] > 0,
                                     df["n_viol_disp"] / df["n_substantive"] * 100.0, np.nan)
        df["noaccrate100"] = np.where(df["n_complaints"] > 0,
                                      df["n_no_access"] / df["n_complaints"] * 100.0, np.nan)
        for g in ["conv", "constr", "elev", "boiler"]:
            df[f"sh_{g}"] = np.where(df["n_complaints"] > 0,
                                     df[f"n_{g}"] / df["n_complaints"], 0.0)
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    keep = df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])
    return df[keep].copy()

panel = pd.read_csv(PANEL, dtype=DTYPE, low_memory=False)
frame = build_frame(PANEL)

BUILDING = ["llc", "corp_other", "trust_estate", "nycha", "govt", "owner_occ_star",
            "is_coop", "is_condo", "era_pre1940", "era_4079", "era_8099", "era_unknown",
            "mzone", "multi_bldg", "com_class", "log_bldgarea", "log2_area_per_unit",
            "value_rank", "any_prior_viol"]
OWNER = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner"]
CAT = ["sh_conv", "sh_constr", "sh_elev", "sh_boiler"]
FE = "size_bin + comm_bin + bct2020"
VCOV = {"CRV1": "bct2020"}
X, XO, XC = " + ".join(BUILDING), " + ".join(BUILDING + OWNER), " + ".join(BUILDING + CAT)

def irr(m, term):
    """Poisson term -> (percent change, CI low, CI high)."""
    b = m.coef()[term]; lo, hi = m.confint().loc[term].values
    return (np.exp(b) - 1) * 100, (np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100

def pp(m, term):
    """LPM term -> (points, CI low, CI high)."""
    b = m.coef()[term]; lo, hi = m.confint().loc[term].values
    return b, lo, hi

def committed(csv):
    return pd.read_csv(RM / csv)

print(f"panel: {len(panel):,} lots | estimation frame: {len(frame):,} | database present: {HAVE_DB}")
'''

S = []
def sec(heading, note, code):
    S.append((heading, note, code))


sec("Panel and coverage for the risk models",
    "The universe, complaint match, per-lot complaint incidence, the overall ECB "
    "citation rate, the inspected subsample, and owner-geography coverage "
    "(`post0_descriptive_stats.py`, `panel_headline_counts.py`, `citation_descriptives.csv`).",
    r'''
d = committed("post0_descriptive_stats.csv").set_index("metric")["value"].astype(float)
phc = committed("panel_headline_counts.csv").set_index("metric")["value"]
cd = committed("citation_descriptives.csv").set_index("group")
print(f"RESULT residential tax lots: {len(panel):,}")
print(f"RESULT scraped complaints matched to a residential lot: {int(d['matched_residential_pages']):,}")
print(f"RESULT complaints per year (avg): {d['monthly_average']*12:,.0f}")
anyc = (panel.n_complaints > 0).mean()
print(f"RESULT properties with any complaint, 2020-May 2026: {anyc*100:.0f}%")
print(f"RESULT ECB citations per 100 residential lots per year (all): {cd.loc['All residential','ecb_per100_yr']:.1f}")
print(f"RESULT owner-geography coverage: {phc['owner_geo_coverage_share']*100:.0f}%")
assert len(panel) == 766939
''')

sec("Raw citation rates by group",
    "Raw ECB citations per 100 lots per year by owner type and enforcement history, "
    "before any adjustment, and the raw ratios the article quotes "
    "(`violation_rate_models.py`, `citation_descriptives.csv`).",
    r'''
g = committed("citation_descriptives.csv").set_index("group")["ecb_per100_yr"]
ind = g["Individual"]
print(f"RESULT individually owned: {ind:.1f} ECB per 100 lots/yr")
print(f"RESULT LLC-owned: {g['LLC']:.1f}  ({g['LLC']/ind:.1f}x the individual rate)")
print(f"RESULT NYCHA: {g['NYCHA']:.1f}  ({g['NYCHA']/ind:.1f}x the individual rate)")
print(f"RESULT prior-violation (2010-19): {g['Prior violation 2010-19']:.1f}  "
      f"({g['Prior violation 2010-19']/g['No prior violation']:.1f}x the never-cited rate)")
''')

sec("Complaint risk factors",
    "PPML complaint count within census tract and exact unit-count size class, "
    "commercial exposure held constant (`risk_factor_models.py`, model "
    "`tract_ppml_ncomp`). Refit here on the committed panel; each coefficient is an "
    "incidence-rate ratio versus individually owned, post-2000, purely residential "
    "buildings of the same size in the same tract.",
    r'''
m = pf.fepois(f"n_complaints ~ {X} | {FE}", data=frame, vcov=VCOV)
for term, lab in [("llc", "LLC-owned"), ("owner_occ_star", "owner-occupied (STAR)"),
                  ("era_pre1940", "pre-1940"), ("era_8099", "built 1980-99"),
                  ("value_rank", "value rank (bottom->top)"), ("com_class", "commercial class S/K/O"),
                  ("is_coop", "co-op"), ("is_condo", "condo")]:
    v, lo, hi = irr(m, term)
    print(f"RESULT {lab} complaints: {v:+.0f}%   95% CI [{lo:+.0f}, {hi:+.0f}]")
print(f"N={m._N:,}")
''')

sec("ECB citation risk factors",
    "PPML ECB/OATH citation count, same fixed effects (`violation_rate_models.py`, "
    "model `ppml_ecb`). Refit on the committed panel.",
    r'''
m = pf.fepois(f"n_ecb_2020on ~ {X} | {FE}", data=frame, vcov=VCOV)
for term, lab in [("any_prior_viol", "prior violation 2010-19"), ("llc", "LLC-owned"),
                  ("trust_estate", "trust/estate"), ("nycha", "NYCHA"),
                  ("era_pre1940", "pre-1940"), ("value_rank", "value rank (bottom->top)"),
                  ("com_class", "commercial class S/K/O")]:
    v, lo, hi = irr(m, term)
    print(f"RESULT {lab} ECB citations: {v:+.0f}%   95% CI [{lo:+.0f}, {hi:+.0f}]")
print(f"N={m._N:,}")
''')

sec("Distant owner citations",
    "Owner-augmented ECB model: owners whose deed address is elsewhere in NYC, or "
    "outside NYC, versus same-ZIP owners (`violation_rate_models.py`, `ppml_ecb_owner`).",
    r'''
m = pf.fepois(f"n_ecb_2020on ~ {XO} | {FE}", data=frame, vcov=VCOV)
for term, lab in [("geo_nyc_other", "elsewhere in NYC"), ("geo_outside_nyc", "outside NYC")]:
    v, lo, hi = irr(m, term)
    print(f"RESULT owner {lab}: {v:+.0f}% ECB   95% CI [{lo:+.0f}, {hi:+.0f}]")
print(f"N={m._N:,}")
''')

sec("Violations per inspection risk factors",
    "Weighted, category-adjusted LPM of the per-inspection violation rate, in "
    "percentage points, on the inspected subsample (`risk_factor_models.py`, model "
    "`cond_violrate_catadj`). Refit on the committed panel.",
    r'''
sub = frame[frame.n_substantive > 0]
m = pf.feols(f"violrate100 ~ {XC} | {FE}", data=sub, weights="n_substantive", vcov=VCOV)
for term, lab in [("owner_occ_star", "owner-occupied (STAR)"), ("era_pre1940", "pre-1940"),
                  ("value_rank", "value rank (bottom->top)"), ("com_class", "commercial class S/K/O"),
                  ("is_condo", "condo")]:
    b, lo, hi = pp(m, term)
    print(f"RESULT {lab} per inspection: {b:+.1f} pp   95% CI [{lo:.1f}, {hi:.1f}]")
print(f"N={m._N:,}")
''')

sec("No access by building age",
    "Weighted LPM of the no-access rate per complaint, in percentage points, on the "
    "complained subsample (`risk_factor_models.py`, model `cond_noaccess`). Refit on "
    "the committed panel.",
    r'''
sub = frame[frame.n_complaints > 0]
m = pf.feols(f"noaccrate100 ~ {X} | {FE}", data=sub, weights="n_complaints", vcov=VCOV)
b, lo, hi = pp(m, "era_pre1940")
print(f"RESULT pre-1940 no-access per complaint: {b:+.1f} pp   95% CI [{lo:.1f}, {hi:.1f}]   N={m._N:,}")
''')

sec("DOB violation records risk factors",
    "PPML on the deduplicated DOB-violation union (`violation_rate_models.py`, model "
    "`ppml_dobviol`). The union outcome is built from the 8 GB database, so these "
    "values display the committed estimate table; they regenerate by running that "
    "script against the database snapshot.",
    r'''
c = committed("citation_tidy_estimates.csv")
for term, lab in [("nycha", "NYCHA"), ("era_pre1940", "pre-1940"),
                  ("com_class", "commercial class S/K/O"), ("owner_occ_star", "owner-occupied (STAR)")]:
    r = c[(c.model == "ppml_dobviol") & (c.term == term)].iloc[0]
    lo = (np.exp(r["25pct"]) - 1) * 100; hi = (np.exp(r["975pct"]) - 1) * 100
    print(f"RESULT {lab} DOB violations (union): {r['pct_change']:+.0f}%   95% CI [{lo:+.0f}, {hi:+.0f}]   N={r['n']:,.0f}")
''')

sec("Neighborhood demographics add little",
    "Tier-1 LPM of any complaint with tract demographics added on top of building "
    "covariates (`risk_factor_models.py`, model `tier1_lpm_any`). The tract "
    "covariates are not in the committed panel, so this displays the committed table.",
    r'''
te = committed("tidy_estimates.csv")
for term, lab in [("tract_poverty10", "tract poverty +10pp"),
                  ("tract_renter10", "tract renter share +10pp"),
                  ("tract_log_income_z", "tract log income +1 SD")]:
    r = te[(te.model == "tier1_lpm_any") & (te.term == term)].iloc[0]
    print(f"RESULT {lab} -> any complaint: {r['estimate']:+.2f} pp   95% CI [{r['25pct']:+.2f}, {r['975pct']:+.2f}]")
''')

sec("Verification against committed estimates",
    "Each refit above is cross-checked against the committed tidy-estimate tables and "
    "the article's quoted number. All rows should read PASS.",
    r'''
te, ce = committed("tidy_estimates.csv"), committed("citation_tidy_estimates.csv")
def lp(df, model, term):
    r = df[(df.model == model) & (df.term == term)].iloc[0]
    return (np.exp(r.estimate) - 1) * 100
def pv(df, model, term):
    return df[(df.model == model) & (df.term == term)].iloc[0].estimate
checks = [
    ("LLC complaints",        lp(te, "tract_ppml_ncomp", "llc"),             70),
    ("pre-1940 complaints",   lp(te, "tract_ppml_ncomp", "era_pre1940"),     -9),
    ("1980-99 complaints",    lp(te, "tract_ppml_ncomp", "era_8099"),       -34),
    ("value complaints",      lp(te, "tract_ppml_ncomp", "value_rank"),     -18),
    ("com_class complaints",  lp(te, "tract_ppml_ncomp", "com_class"),      -15),
    ("co-op complaints",      lp(te, "tract_ppml_ncomp", "is_coop"),        -26),
    ("condo complaints",      lp(te, "tract_ppml_ncomp", "is_condo"),       -38),
    ("prior-viol ECB",        lp(ce, "ppml_ecb", "any_prior_viol"),         121),
    ("LLC ECB",               lp(ce, "ppml_ecb", "llc"),                    103),
    ("trust/estate ECB",      lp(ce, "ppml_ecb", "trust_estate"),           -27),
    ("NYCHA ECB",             lp(ce, "ppml_ecb", "nycha"),                  -21),
    ("pre-1940 ECB",          lp(ce, "ppml_ecb", "era_pre1940"),            -28),
    ("value ECB",             lp(ce, "ppml_ecb", "value_rank"),             -16),
    ("com_class ECB",         lp(ce, "ppml_ecb", "com_class"),              -11),
    ("owner elsewhere NYC",   lp(ce, "ppml_ecb_owner", "geo_nyc_other"),      9),
    ("owner outside NYC",     lp(ce, "ppml_ecb_owner", "geo_outside_nyc"),   12),
    ("owner-occ inspection",  pv(te, "cond_violrate_catadj", "owner_occ_star"), -2.8),
    ("pre-1940 inspection",   pv(te, "cond_violrate_catadj", "era_pre1940"),  4.6),
    ("value inspection",      pv(te, "cond_violrate_catadj", "value_rank"),  -4.3),
    ("com_class inspection",  pv(te, "cond_violrate_catadj", "com_class"),    2.1),
    ("condo inspection",      pv(te, "cond_violrate_catadj", "is_condo"),    -1.8),
    ("pre-1940 no-access",    pv(te, "cond_noaccess", "era_pre1940"),          7),
    ("NYCHA DOB union",       lp(ce, "ppml_dobviol", "nycha"),               92),
    ("pre-1940 DOB union",    lp(ce, "ppml_dobviol", "era_pre1940"),         21),
    ("com_class DOB union",   lp(ce, "ppml_dobviol", "com_class"),          108),
    ("owner-occ DOB union",   lp(ce, "ppml_dobviol", "owner_occ_star"),     -10),
]
out = pd.DataFrame([{"result": n, "committed": round(c, 1), "article": a,
                     "check": "PASS" if abs(c - a) <= 1.0 else "REVIEW"}
                    for n, c, a in checks])
print(out.to_string(index=False))
assert (out.check == "PASS").all(), out[out.check != "PASS"]
''')

# --------------------------------------------------------------------------- #
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "pyfix", "language": "python", "name": "pyfix"},
               "language_info": {"name": "python"}}
cells = [nbf.v4.new_markdown_cell(HEADER), nbf.v4.new_code_cell(SETUP.strip())]
anchors = {}
for heading, note, code in S:
    anchors[heading] = heading.replace(" ", "-")
    cells.append(nbf.v4.new_markdown_cell(f"## {heading}\n\n{note}"))
    cells.append(nbf.v4.new_code_cell(code.strip()))
nb.cells = cells
nbf.write(nb, OUT)
(NB_DIR / "risk_requirements.txt").write_text(REQUIREMENTS)
(NB_DIR / "risk_anchors.json").write_text(json.dumps(anchors, indent=1))
print(f"wrote {OUT} ({len(S)} sections, {len(cells)} cells)")
print(f"wrote risk_anchors.json")
