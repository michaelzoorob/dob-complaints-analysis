"""Build notebooks/neighborhood_results.ipynb: a self-contained notebook that RECOMPUTES
every number in the neighborhood post (data/analysis/blog_posts/post_neighborhoods_substack.md)
from the committed panel (property_risk_panel_v2.csv.gz) and the two committed extracts
written by neighborhood_gradients.py (neighborhood_lot_outcomes.csv.gz,
neighborhood_complaints.csv.gz). No database access is needed. Each section carries a
unique alphanumeric heading so the nbviewer anchor is heading.replace(" ", "-"); the post
hyperlinks each number to its section. The frame-building code is lifted verbatim from
neighborhood_gradients.py at build time so the two cannot drift.

Execute with the pinned kernel:
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3000 --ExecutePreprocessor.kernel_name=pyfix \
      notebooks/neighborhood_results.ipynb
"""
import inspect
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import neighborhood_gradients as ng  # noqa: E402

NB_DIR = ROOT / "notebooks"
OUT = NB_DIR / "neighborhood_results.ipynb"
REPO = "michaelzoorob/dob-complaints-analysis"

REQUIREMENTS = """# environment for neighborhood_results.ipynb (Python 3.14)
pyfixest==0.60.0
pandas==3.0.0
numpy==2.3.5
scipy==1.17.0
statsmodels==0.14.6
"""

HEADER = f"""# Reproducible results: which neighborhoods complain, and which ones have violations

This notebook recomputes every number in the article on how DOB complaints, ECB
citations, scheduled-inspection violations, and inspection hit rates vary with
census-tract demographics. Click a number in the article and you land on the cell
that produces it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/neighborhood_results.ipynb)

**How to read this.** Everything is refit here from the committed property panel and
two committed extracts (per-lot caller, agency, and DOB-violation-record counts; a
complaint-level file with outcome and inspector). The producing script is
`scripts/neighborhood_gradients.py`; a final cell cross-checks every refit against
the committed CSVs in `data/analysis/risk_models/`.
"""

CONSTANTS = "\n".join([
    'DTYPE = {"bct2020": str, "size_bin": str, "borocode": str, "bbl_key": str}',
    f"WINDOW = {ng.WINDOW!r}",
    f"YEARS_C = {ng.YEARS_C!r}",
    f"YEARS_E = {ng.YEARS_E!r}",
    f"FE_B = {ng.FE_B!r}",
    f"VCOV = {ng.VCOV!r}",
    f"MIN_INSPECTOR_CASES = {ng.MIN_INSPECTOR_CASES!r}",
    f"DEMOS = {ng.DEMOS!r}",
    f"GROUPS = {ng.GROUPS!r}",
    "CONTROLS = [c for g in GROUPS.values() for c in g]",
    'X = " + ".join(CONTROLS)',
    f"OUTCOMES = {ng.OUTCOMES!r}",
])

SETUP = r'''# === setup: versions, committed panel and extracts, analysis frame, helpers ===
import sys, warnings
from pathlib import Path
import gc
import numpy as np, pandas as pd, pyfixest as pf
import scipy
warnings.filterwarnings("ignore")
print("python", sys.version.split()[0], "| pyfixest", pf.__version__,
      "| pandas", pd.__version__, "| numpy", np.__version__, "| scipy", scipy.__version__)

CAND = [Path.cwd(), Path.cwd().parent] + list(Path.cwd().parents)
ROOT = next((p for p in CAND if (p / "data" / "analysis").exists()), Path.cwd())
DATA = ROOT / "data" / "analysis"
RM = DATA / "risk_models"
PANEL = DATA / "property_risk_panel_v2.csv.gz"
LOT_EXTRACT = DATA / "neighborhood_lot_outcomes.csv.gz"
CPL_EXTRACT = DATA / "neighborhood_complaints.csv.gz"

''' + CONSTANTS + "\n\n" + inspect.getsource(ng.build_frame) + "\n" + inspect.getsource(ng.irr_row) + "\n" + inspect.getsource(ng.pp_row) + r'''

def committed(csv):
    return pd.read_csv(RM / csv)

lots = pd.read_csv(LOT_EXTRACT, dtype={"bbl_key": str})
cpl = pd.read_csv(CPL_EXTRACT, dtype={"bbl_key": str, "inspector_badge": str, "complaint_category": str})
frame = build_frame().merge(lots, on="bbl_key", how="left")
for c in ["n_caller", "n_agency", "n_dobviol_2020on"]:
    frame[c] = frame[c].fillna(0).astype(int)
frame["any_caller100"] = (frame["n_caller"] > 0).astype(float) * 100
est = frame.dropna(subset=CONTROLS + list(OUTCOMES)).copy()

# complaint-level frames (caller complaints only): all with an outcome, and accessed
keep_cols = ["bbl_key", "size_bin", "comm_bin", "borocode", "bct2020"] + list(DEMOS) + CONTROLS
c_all = cpl.merge(est[keep_cols], on="bbl_key", how="inner")
c_all = c_all[(c_all["caller"] == 1) & c_all["outcome"].isin(["violation", "no_violation", "no_access"])].copy()
c_all["noaccess100"] = (c_all["outcome"] == "no_access").astype(float) * 100
acc = c_all[c_all["outcome"] != "no_access"].copy()
acc["viol100"] = (acc["outcome"] == "violation").astype(float) * 100
acc["inspector_badge"] = acc["inspector_badge"].fillna("").astype(str).str.strip()
_counts = acc["inspector_badge"].value_counts()
acc = acc[acc["inspector_badge"].isin(_counts[_counts >= MIN_INSPECTOR_CASES].index.difference([""]))].copy()
FE_H0 = "complaint_category + size_bin + comm_bin + borocode"
FE_H1 = FE_H0 + " + inspector_badge"
REFIT = {}
print(f"estimation frame: {len(est):,} lots | caller complaints with an outcome: {len(c_all):,} | accessed, inspectors with >= {MIN_INSPECTOR_CASES} cases: {len(acc):,}")
'''

S = []
def sec(heading, note, code):
    S.append((heading, note, code))


sec("Neighborhood panel and coverage",
    "Sample sizes behind the neighborhood post: lots with tract demographics and building "
    "controls, caller versus agency-initiated complaints (a complaint is caller-initiated when "
    "it carries a 311 reference number, as in `post0_descriptive_stats.py`), and the "
    "complaint-level hit-rate sample (`neighborhood_gradients.py`).",
    r'''
print(f"RESULT residential lots in the estimation frame: {len(est):,}")
print(f"RESULT census tracts represented: {est['bct2020'].nunique():,}")
n_c, n_a = int(cpl["caller"].sum()), int((cpl["caller"] == 0).sum())
print(f"RESULT complaints in window with a lot: {len(cpl):,}  caller {n_c:,} ({n_c/len(cpl)*100:.1f}%)  agency-initiated {n_a:,} ({n_a/len(cpl)*100:.1f}%)")
print(f"RESULT caller complaints per 100 lots per year: {est['n_caller'].sum()/len(est)/YEARS_C*100:.1f}")
print(f"RESULT lots with any caller complaint, 2020 - May 2026: {(est['n_caller']>0).mean()*100:.0f}%")
print(f"RESULT accessed caller inspections in the hit-rate sample: {len(acc):,} across {acc['inspector_badge'].nunique():,} inspectors")
print(f"RESULT baseline hit rate (violation found | accessed): {acc['viol100'].mean():.1f}%   baseline no-access rate: {c_all['noaccess100'].mean():.1f}%")
d = committed("neighborhood_descriptives.csv"); a = d[d.variable == "all"].iloc[0]
assert abs(a["caller_per100_yr"] - est['n_caller'].sum()/len(est)/YEARS_C*100) < 0.05
''')

sec("Raw rates by tract quintile",
    "Lot-weighted quintiles of tract poverty, foreign-born share, and median income; caller "
    "complaints, ECB citations, and DOB violation records per 100 lots per year, plus the "
    "hit rate and no-access rate among caller complaints, before any adjustment "
    "(`neighborhood_descriptives.csv`).",
    r'''
d = committed("neighborhood_descriptives.csv")
def quintile_table(raw):
    q = pd.qcut(est[raw].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    cq = c_all.merge(pd.DataFrame({"bbl_key": est["bbl_key"], "_q": q.values}), on="bbl_key")
    rows = []
    for k in range(1, 6):
        g = est[q.values == k]; gc = cq[cq["_q"] == k]; ak = gc[gc["outcome"] != "no_access"]
        rows.append(dict(quintile=k, lo=g[raw].min(), hi=g[raw].max(), n_lots=len(g),
                         caller_per100_yr=g["n_caller"].sum()/len(g)/YEARS_C*100,
                         ecb_per100_yr=g["n_ecb_2020on"].sum()/len(g)/YEARS_E*100,
                         dobviol_per100_yr=g["n_dobviol_2020on"].sum()/len(g)/YEARS_C*100,
                         hit_rate=(ak["outcome"] == "violation").mean(), noaccess_rate=(gc["outcome"] == "no_access").mean()))
    return pd.DataFrame(rows)
for var, raw in [("poverty", "tract_poverty"), ("foreign_born", "tract_foreign_born"), ("income", "med_income")]:
    t = quintile_table(raw); lo, hi = t.iloc[0], t.iloc[-1]
    print(f"RESULT {var}: bottom quintile caller complaints {lo.caller_per100_yr:.1f} vs top quintile {hi.caller_per100_yr:.1f} per 100 lots/yr; "
          f"DOB violation records {lo.dobviol_per100_yr:.1f} vs {hi.dobviol_per100_yr:.1f}; hit rate {lo.hit_rate*100:.1f}% vs {hi.hit_rate*100:.1f}%; "
          f"no access {lo.noaccess_rate*100:.1f}% vs {hi.noaccess_rate*100:.1f}%")
    cm = d[d.variable == var].sort_values("quintile")
    assert np.allclose(cm["caller_per100_yr"].values, t["caller_per100_yr"].values, atol=0.05)
    print(t.round(3).to_string(index=False)); print()
''')

for y, ylab, heading in [
    ("n_caller", "caller complaints", "Caller complaint gradients by tract demographics"),
    ("n_ecb_2020on", "ECB citations", "ECB citation gradients by tract demographics"),
    ("n_dobviol_2020on", "DOB violation records", "DOB violation record gradients by tract demographics"),
    ("n_agency", "agency-initiated complaints", "Agency initiated complaint gradients by tract demographics"),
]:
    sec(heading,
        f"Poisson pseudo-maximum-likelihood models of {ylab} per lot on each tract demographic, "
        "one at a time. `total` holds building size fixed only (unit-count, commercial-unit, and "
        "borough fixed effects); `direct` adds the building-stock controls (era, within-tract value "
        "rank, ownership, use and size detail, prior violations). Standard errors clustered by tract "
        "(`neighborhood_gradients.py`, `neighborhood_gradients.csv`).",
        rf'''
y = "{y}"
for d, dlab in DEMOS.items():
    out = {{}}
    for spec, rhs in [("total", d), ("direct", f"{{d}} + {{X}}")]:
        m = pf.fepois(f"{{y}} ~ {{rhs}} | {{FE_B}}", data=est, vcov=VCOV, lean=True, store_data=False, copy_data=False)
        r = irr_row(m, d); out[spec] = r; REFIT[(y, spec, d)] = r["pct_change"]
    t, dr = out["total"], out["direct"]
    print(f"RESULT {{dlab}}: total {{t['pct_change']:+.1f}}% [{{t['pct_lo']:+.1f}}, {{t['pct_hi']:+.1f}}]   "
          f"direct {{dr['pct_change']:+.1f}}% [{{dr['pct_lo']:+.1f}}, {{dr['pct_hi']:+.1f}}]   N={{t['n']:,}}")
    del m; gc.collect()
''')

sec("Building stock decomposition of the caller complaint gradient",
    "Linear probability model of any caller complaint (percentage points) on each demographic, "
    "with and without the building-stock controls, and the exact Gelbach (2016) decomposition of "
    "the gap into control groups: each control's coefficient in the full model times its own "
    "gradient on the demographic. The group contributions sum to the gap by construction "
    "(`neighborhood_decomposition.csv`).",
    r'''
dec = committed("neighborhood_decomposition.csv")
for d, dlab in DEMOS.items():
    base = pf.feols(f"any_caller100 ~ {d} | {FE_B}", data=est, vcov=VCOV, lean=True, store_data=False, copy_data=False)
    full = pf.feols(f"any_caller100 ~ {d} + {X} | {FE_B}", data=est, vcov=VCOV, lean=True, store_data=False, copy_data=False)
    b0, b1 = float(base.coef()[d]), float(full.coef()[d])
    contrib = {k: float(pf.feols(f"{k} ~ {d} | {FE_B}", data=est, vcov="iid", lean=True, store_data=False, copy_data=False).coef()[d]) * float(full.coef()[k]) for k in CONTROLS}
    assert np.isclose(sum(contrib.values()), b0 - b1, rtol=1e-3, atol=1e-6)
    parts = {g: sum(contrib[k] for k in ks) for g, ks in GROUPS.items()}
    REFIT[("gelbach", "total", d)] = b0; REFIT[("gelbach", "direct", d)] = b1
    for g, v in parts.items(): REFIT[("gelbach", f"via_{g}", d)] = v
    br, fr = pp_row(base, d), pp_row(full, d)
    print(f"RESULT {dlab}: total {b0:+.2f} pp [{br['ci_lo']:+.2f}, {br['ci_hi']:+.2f}]  left over {b1:+.2f} pp [{fr['ci_lo']:+.2f}, {fr['ci_hi']:+.2f}]  "
          + "  ".join(f"via {g} {v:+.2f}" for g, v in parts.items())
          + (f"   (left over = {b1/b0*100:.0f}% of total)" if abs(b0) > 1e-9 else ""))
    del base, full; gc.collect()
''')

sec("Hit rate by tract demographics with inspector fixed effects",
    "Complaint-level linear probability model: violation found (percentage points) among "
    "caller complaints where the inspector accessed the property, on each tract demographic. "
    "Fixed effects for complaint category, unit-count, commercial-unit, and borough; then adding "
    "inspector fixed effects (same inspector, complaints from different tracts); then adding the "
    "building-stock controls. Inspectors with at least 30 accessed caller inspections. Standard "
    "errors clustered by tract (`neighborhood_hitrate.csv`).",
    r'''
print(f"baseline hit rate: {acc['viol100'].mean():.1f}%  N={len(acc):,}")
for d, dlab in DEMOS.items():
    out = {}
    for spec, rhs, fe in [("hit_base", d, FE_H0), ("hit_inspector_fe", d, FE_H1), ("hit_inspector_fe_controls", f"{d} + {X}", FE_H1)]:
        m = pf.feols(f"viol100 ~ {rhs} | {fe}", data=acc, vcov=VCOV, lean=True, store_data=False, copy_data=False)
        r = pp_row(m, d); out[spec] = r; REFIT[("hit", spec, d)] = r["estimate"]
    print(f"RESULT {dlab}: base {out['hit_base']['estimate']:+.2f} pp   with inspector FE {out['hit_inspector_fe']['estimate']:+.2f} pp "
          f"[{out['hit_inspector_fe']['ci_lo']:+.2f}, {out['hit_inspector_fe']['ci_hi']:+.2f}]   "
          f"plus building controls {out['hit_inspector_fe_controls']['estimate']:+.2f} pp [{out['hit_inspector_fe_controls']['ci_lo']:+.2f}, {out['hit_inspector_fe_controls']['ci_hi']:+.2f}]")
    del m; gc.collect()
''')

sec("No access by tract demographics",
    "Complaint-level linear probability model: the inspection ended without access (percentage "
    "points) among caller complaints with an outcome, on each tract demographic, with and without "
    "the building-stock controls; category, unit-count, commercial-unit, and borough fixed effects, "
    "standard errors clustered by tract (`neighborhood_hitrate.csv`).",
    r'''
print(f"baseline no-access rate: {c_all['noaccess100'].mean():.1f}%  N={len(c_all):,}")
for d, dlab in DEMOS.items():
    out = {}
    for spec, rhs in [("noaccess_base", d), ("noaccess_controls", f"{d} + {X}")]:
        m = pf.feols(f"noaccess100 ~ {rhs} | {FE_H0}", data=c_all, vcov=VCOV, lean=True, store_data=False, copy_data=False)
        r = pp_row(m, d); out[spec] = r; REFIT[("noaccess", spec, d)] = r["estimate"]
    print(f"RESULT {dlab}: base {out['noaccess_base']['estimate']:+.2f} pp [{out['noaccess_base']['ci_lo']:+.2f}, {out['noaccess_base']['ci_hi']:+.2f}]   "
          f"with building controls {out['noaccess_controls']['estimate']:+.2f} pp [{out['noaccess_controls']['ci_lo']:+.2f}, {out['noaccess_controls']['ci_hi']:+.2f}]")
    del m; gc.collect()
''')

sec("Verification against committed estimates",
    "Every refit above compared with the committed CSVs written by `neighborhood_gradients.py`. "
    "Refits use the same committed data and code, so they should agree to rounding.",
    r'''
g = committed("neighborhood_gradients.csv"); g = g[g.entry == "bivariate"]
h = committed("neighborhood_hitrate.csv"); dec = committed("neighborhood_decomposition.csv")
rows = []
for (kind, spec, d), v in REFIT.items():
    if kind in OUTCOMES:
        c = float(g[(g.outcome == kind) & (g.spec == spec) & (g.term == d)]["pct_change"].iloc[0]); tol = 0.05
    elif kind == "gelbach":
        c = float(dec[(dec.term == d) & (dec.component == spec)]["points"].iloc[0]); tol = 0.005
    else:
        c = float(h[(h.spec == spec) & (h.term == d)]["estimate"].iloc[0]); tol = 0.005
    rows.append({"result": f"{kind} / {spec} / {d}", "refit": round(v, 3), "committed": round(c, 3),
                 "check": "PASS" if abs(v - c) <= tol else "REVIEW"})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
print(f"\n{(out.check == 'PASS').sum()} of {len(out)} PASS")
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
(NB_DIR / "neighborhood_requirements.txt").write_text(REQUIREMENTS)
(NB_DIR / "neighborhood_anchors.json").write_text(json.dumps(anchors, indent=1))
print(f"wrote {OUT} ({len(S)} sections, {len(cells)} cells)")
