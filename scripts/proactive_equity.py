#!/usr/bin/env python3
"""
Hypotheses #7 and #10 (proactive_enforcement_plan.md): equity in proactive
enforcement, conditional on exposure.

Part 1 — ALLOCATION (#7). Tract-level PPML of agency-initiated complaint
counts (2020-01..2026-05 totals) on tract demographics, holding the
relevant exposure fixed with an offset and controlling building-stock mix:

  discretionary construction   discretionary_field family excluding 7G
    (8A compliance, 1X EWO, 91 worker endangerment, ...), offset
    log(active job-months 2020-26) aggregated from jobs.csv.gz spans
    (month-index expansion identical to proactive_spine.build_tract_month;
    the script asserts the two agree tract by tract);
  7G area sweeps               offset log(residential units, PLUTO);
  statutory periodic contrast  same demographics and controls, run under
    BOTH offsets. Statutory cycles are calendar-driven, so demographic
    loadings there read as building-stock mechanics the controls missed,
    the same placebo logic as proactive_complaint_indexing.py.

Demographics: tract_pct_black / hispanic / asian, tract_poverty,
tract_renter_share (ACS via property_risk_panel_v2, 0-1 shares, jointly
entered). Building-stock mix per CLAUDE.md commercial-exposure convention,
aggregated to the tract: era shares from yearbuilt (pre-1940 and 1980+,
base 1940-79, plus a missing-year share), mean log building area,
commercial-class share (PLUTO class S/K/O), and log mean residential
units per lot. Borough fixed effects; SEs clustered on NTA (coarser than
the tract-level treatment, CLAUDE.md convention).

Critic-E additions (critic_proactive_E_yield_equity.py, E4a-c):
  free elasticity        the offset forces the exposure elasticity to 1.0,
                         but the fitted cross-tract elasticity is ~0.72
                         [0.67, 0.77]; under that concavity the offset
                         mechanically inflates per-exposure rates in
                         low-exposure (nonwhite) tracts. Every margin is
                         re-run with log exposure entered as a REGRESSOR;
                         these free-elasticity rows are the headline
                         (loadings roughly halve), and the original rows
                         are kept labeled "unit-elasticity offset
                         (superseded as headline)".
  at/off-permit split    the discretionary-construction bundle split by
                         whether the event sits at a lot with an active
                         permit (plus an 8A-only cut). The Black and
                         Hispanic loadings live OFF-permit (enforcement
                         against unpermitted work, where the permit-based
                         offset undercounts true construction exposure);
                         the Asian loading partially survives at-permit.
                         Do not lump the three groups into one sentence.
  alternative exposures  floor-area-weighted job-months and caller-side
                         construction-incident volume (log1p) as offsets.

Part 2 — CRISIS RESPONSE (#10). Among each lot's FIRST caller-originated
category-30 unstable-building report (event definition reused from
proactive_incident_eventstudy.py, day-level, restricted so the 90-day
window fits inside the 2020-01..2026-05 spine), LPM of any
agency-originated complaint event at the same BBL within 90 days
(day-level searchsorted counter; agency events proxy proactive
inspections) on tract demographics + priority dummies + building traits
(log units, log building area, prewar, commercial class,
missing-trait indicator, active-permit flag at the trigger) with
borough x month FE, SEs clustered by tract. Variants: excluding agency
cat-30 follow-ups from the outcome, and excluding same-day responses.

Writes
    data/analysis/risk_models/proactive_equity.csv
    data/analysis/blog_posts/artifacts/proactive_equity.png

Run:  /private/tmp/pyfix_venv/bin/python scripts/proactive_equity.py
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
PRO = ROOT / "data" / "analysis" / "proactive"
PANEL_V2 = ROOT / "data" / "analysis" / "property_risk_panel_v2.csv.gz"
RM = ROOT / "data" / "analysis" / "risk_models"
ART = ROOT / "data" / "analysis" / "blog_posts" / "artifacts"

WINDOW_START = pd.Timestamp("2020-01-01")
WINDOW_END = pd.Timestamp("2026-05-31")
N_MONTHS = 77                     # 2020-01 .. 2026-05
RESPONSE_DAYS = 90
DAY_STRIDE = 4096                 # > max day index, packs (bbl, day) in int64

DEMOS = ["pct_black", "pct_hispanic", "pct_asian", "poverty", "renter"]
DEMO_LABELS = {
    "pct_black": "Black share",
    "pct_hispanic": "Hispanic share",
    "pct_asian": "Asian share",
    "poverty": "poverty rate",
    "renter": "renter share",
}
STOCK = ["era_pre1940", "era_1980plus", "era_missing", "mean_log_ba",
         "com_share", "log_mean_units"]
BLD_TRAITS = ["log_units", "log_ba", "prewar", "com_class",
              "bldg_missing", "active_permit"]

GEO_NOTE = ("agency events geo-match to tracts at 94.1% (callers 97.9%); "
            "proactive counts slightly undercounted")

# ── house style (constants from scripts/make_descriptive_figures.py) ────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
ZERO_C = "#b9b7ac"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ── shared inputs ────────────────────────────────────────────────────────

def load_events() -> pd.DataFrame:
    ev = pd.read_csv(
        PRO / "proactive_events.csv.gz",
        usecols=["received_date", "category_prefix", "family", "agency",
                 "bbl", "bct2020", "nta", "priority", "active_permit"],
        dtype={"category_prefix": "str", "bbl": "str", "nta": "str",
               "priority": "str"}, low_memory=False)
    ev["received"] = pd.to_datetime(ev["received_date"])
    ev["day"] = (ev["received"] - WINDOW_START).dt.days
    ev["tract"] = pd.to_numeric(ev["bct2020"], errors="coerce").astype("Int64")
    print(f"events: {len(ev):,} rows 2020-01..2026-05; "
          f"agency {int(ev['agency'].sum()):,}; "
          f"tract match {ev['tract'].notna().mean():.3f}")
    return ev


def load_v2():
    """Tract demographics + tract building-stock mix + per-BBL traits."""
    v2 = pd.read_csv(
        PANEL_V2,
        usecols=["bbl_key", "bct2020", "bldgclass", "bldgarea", "unitsres",
                 "yearbuilt", "prewar", "tract_pct_black",
                 "tract_pct_hispanic", "tract_pct_asian", "tract_poverty",
                 "tract_renter_share"],
        dtype={"bbl_key": "str"})
    v2["tract"] = pd.to_numeric(v2["bct2020"], errors="coerce").astype("Int64")
    v2 = v2[v2["tract"].notna()]

    yb = v2["yearbuilt"].fillna(0)
    v2["era_pre1940"] = ((yb > 0) & (yb < 1940)).astype(float)
    v2["era_1980plus"] = (yb >= 1980).astype(float)
    v2["era_missing"] = (yb <= 0).astype(float)
    v2["com_class"] = (v2["bldgclass"].astype(str).str[0]
                       .isin(["S", "K", "O"]).astype(int))
    v2["log_ba"] = np.log(v2["bldgarea"].where(v2["bldgarea"] > 0))

    tract = v2.groupby("tract").agg(
        pct_black=("tract_pct_black", "first"),
        pct_hispanic=("tract_pct_hispanic", "first"),
        pct_asian=("tract_pct_asian", "first"),
        poverty=("tract_poverty", "first"),
        renter=("tract_renter_share", "first"),
        era_pre1940=("era_pre1940", "mean"),
        era_1980plus=("era_1980plus", "mean"),
        era_missing=("era_missing", "mean"),
        mean_log_ba=("log_ba", "mean"),
        com_share=("com_class", "mean"),
        mean_units=("unitsres", "mean"),
        n_props=("bbl_key", "size"))
    tract["log_mean_units"] = np.log1p(tract["mean_units"])

    v2["log_units"] = np.log1p(v2["unitsres"])
    traits = (v2.drop_duplicates("bbl_key")
              .set_index("bbl_key")[["log_units", "log_ba", "prewar",
                                     "com_class"]])
    print(f"panel v2: {len(v2):,} properties -> {len(tract):,} tracts with "
          f"demographics, {len(traits):,} BBL trait rows")
    return tract, traits


# ── Part 1: allocation conditional on exposure ───────────────────────────

def job_months_by_tract() -> pd.Series:
    """Active job-months 2020-01..2026-05 per tract from jobs.csv.gz spans."""
    jobs = pd.read_csv(PRO / "jobs.csv.gz",
                       usecols=["bct2020", "active_start", "active_end"],
                       parse_dates=["active_start", "active_end"])
    jobs["tract"] = pd.to_numeric(jobs["bct2020"], errors="coerce").astype("Int64")
    j = jobs.dropna(subset=["tract", "active_start", "active_end"])
    s = ((j["active_start"].dt.year - 2020) * 12
         + j["active_start"].dt.month - 1).clip(lower=0)
    t = ((j["active_end"].dt.year - 2020) * 12
         + j["active_end"].dt.month - 1).clip(upper=N_MONTHS - 1)
    keep = t >= s
    jm = (pd.DataFrame({"tract": j["tract"][keep], "months": (t - s + 1)[keep]})
          .groupby("tract")["months"].sum())

    # must agree with the spine's tract_month expansion, tract by tract
    tm = pd.read_csv(PRO / "tract_month.csv.gz",
                     usecols=["bct2020", "active_jobs"])
    spine = tm.groupby("bct2020")["active_jobs"].sum()
    spine.index = spine.index.astype("int64")
    both = pd.concat([jm.rename("direct"), spine.rename("spine")],
                     axis=1).fillna(0)
    n_bad = int((both["direct"] != both["spine"]).sum())
    assert n_bad == 0, f"job-month aggregation diverges from spine ({n_bad} tracts)"
    print(f"job-months: {int(jm.sum()):,} across {len(jm):,} tracts "
          f"(matches tract_month spine exactly)")
    return jm


def units_by_tract() -> pd.Series:
    tm = pd.read_csv(PRO / "tract_month.csv.gz",
                     usecols=["bct2020", "units_res"])
    u = tm.groupby("bct2020")["units_res"].first()
    u.index = u.index.astype("int64")
    return u


def fa_job_months_by_tract() -> pd.Series:
    """Floor-area-weighted active job-months per tract (alternative
    exposure, critic E4c): each job's months are weighted by its total
    construction floor area (nonpositive/missing -> median)."""
    jobs = pd.read_csv(PRO / "jobs.csv.gz",
                       usecols=["bct2020", "active_start", "active_end",
                                "total_construction_floor_area"],
                       parse_dates=["active_start", "active_end"])
    jobs["tract"] = pd.to_numeric(jobs["bct2020"],
                                  errors="coerce").astype("Int64")
    j = jobs.dropna(subset=["tract", "active_start", "active_end"]).copy()
    s = ((j["active_start"].dt.year - 2020) * 12
         + j["active_start"].dt.month - 1).clip(lower=0)
    t = ((j["active_end"].dt.year - 2020) * 12
         + j["active_end"].dt.month - 1).clip(upper=N_MONTHS - 1)
    keep = t >= s
    j, s, t = j[keep], s[keep], t[keep]
    months = t - s + 1
    fa = pd.to_numeric(j["total_construction_floor_area"], errors="coerce")
    fa = fa.where(fa > 0)
    fa = fa.fillna(fa.median())
    return (pd.DataFrame({"tract": j["tract"], "v": months * fa})
            .groupby("tract")["v"].sum())


def build_tract_frame(ev: pd.DataFrame, tract: pd.DataFrame) -> pd.DataFrame:
    ag = ev[(ev["agency"] == 1) & ev["tract"].notna()]
    disc = ag["family"] == "discretionary_field"
    bundle = ag[disc & (ag["category_prefix"] != "7G")]
    cnt = pd.DataFrame({
        "n_disc_constr": bundle.groupby("tract").size(),
        "n_atpermit": bundle[bundle["active_permit"] == 1]
        .groupby("tract").size(),
        "n_offpermit": bundle[bundle["active_permit"] == 0]
        .groupby("tract").size(),
        "n_8a": bundle[bundle["category_prefix"] == "8A"]
        .groupby("tract").size(),
        "n_7g": ag[ag["category_prefix"] == "7G"].groupby("tract").size(),
        "n_stat": ag[ag["family"] == "statutory_periodic"]
        .groupby("tract").size(),
    }).fillna(0).astype(int)
    print(f"tract counts: disc-constr {cnt['n_disc_constr'].sum():,} "
          f"(at-permit {cnt['n_atpermit'].sum():,} / off-permit "
          f"{cnt['n_offpermit'].sum():,} / 8A {cnt['n_8a'].sum():,}), "
          f"7G {cnt['n_7g'].sum():,}, statutory {cnt['n_stat'].sum():,}")

    xw = (ev.dropna(subset=["nta"]).loc[lambda d: d["tract"].notna()]
          .groupby("tract")["nta"].agg(lambda s: s.mode().iat[0]))

    df = (cnt.join(job_months_by_tract().rename("job_months"), how="outer")
          .join(units_by_tract().rename("units_res"), how="left")
          .join(tract, how="inner"))
    for c in ("n_disc_constr", "n_atpermit", "n_offpermit", "n_8a",
              "n_7g", "n_stat", "job_months", "units_res"):
        df[c] = df[c].fillna(0)
    df["nta"] = df.index.map(xw)
    df["boro"] = (df.index // 1_000_000).astype(int).astype(str)
    df = df[df["nta"].notna()
            & df[DEMOS].notna().all(axis=1)
            & df[STOCK].notna().all(axis=1)].copy()
    df["log_job_months"] = np.log(df["job_months"].where(df["job_months"] > 0))
    df["log_units_res"] = np.log(df["units_res"].where(df["units_res"] > 0))
    # alternative exposures (critic E4c)
    df["log_fa_months"] = np.log(fa_job_months_by_tract().reindex(df.index))
    caller_con = ev[(ev["agency"] == 0) & (ev["family"] == "mixed_incident")
                    & ev["tract"].notna()]
    cc = caller_con.groupby("tract").size().reindex(df.index).fillna(0)
    df["log_caller_con"] = np.log1p(cc)
    print(f"tract frame: {len(df):,} tracts, {df['nta'].nunique()} NTA "
          f"clusters; zero job-months {int((df['job_months'] == 0).sum())}, "
          f"zero units {int((df['units_res'] == 0).sum())}")
    return df


SPEC_OFFSET = "unit-elasticity offset (superseded as headline)"
SPEC_FREE = "free elasticity (log exposure as regressor)"
SPEC_SPLIT = "at/off-permit split of the bundle (job-month offset)"
SPEC_ALT = "alternative exposure offset"

ALLOC_MODELS = [
    # outcome, label, exposure column, exposure label, how, spec
    ("n_disc_constr", "discretionary construction (8A/1X/91..., excl 7G)",
     "log_job_months", "log active job-months 2020-26", "offset",
     SPEC_OFFSET),
    ("n_7g", "7G area sweeps", "log_units_res", "log residential units",
     "offset", SPEC_OFFSET),
    ("n_stat", "statutory periodic (contrast)", "log_units_res",
     "log residential units", "offset", SPEC_OFFSET),
    ("n_stat", "statutory periodic (contrast)", "log_job_months",
     "log active job-months 2020-26", "offset", SPEC_OFFSET),
    # free-elasticity refits (critic E4a): log exposure as a regressor;
    # the fitted cross-tract elasticity is ~0.72, so the offset's forced
    # 1.0 mechanically inflates loadings on low-exposure tracts. These
    # are the headline rows.
    ("n_disc_constr", "discretionary construction (8A/1X/91..., excl 7G)",
     "log_job_months", "log active job-months 2020-26", "free", SPEC_FREE),
    ("n_7g", "7G area sweeps", "log_units_res", "log residential units",
     "free", SPEC_FREE),
    ("n_stat", "statutory periodic (contrast)", "log_units_res",
     "log residential units", "free", SPEC_FREE),
    ("n_stat", "statutory periodic (contrast)", "log_job_months",
     "log active job-months 2020-26", "free", SPEC_FREE),
    # at/off-permit split (critic E4b): where the loading lives
    ("n_atpermit", "bundle events at an active permit", "log_job_months",
     "log active job-months 2020-26", "offset", SPEC_SPLIT),
    ("n_offpermit", "bundle events off permit (EWO/illegal-work slice)",
     "log_job_months", "log active job-months 2020-26", "offset",
     SPEC_SPLIT),
    ("n_8a", "8A construction compliance only", "log_job_months",
     "log active job-months 2020-26", "offset", SPEC_SPLIT),
    # alternative exposures (critic E4c)
    ("n_disc_constr", "discretionary construction (8A/1X/91..., excl 7G)",
     "log_fa_months", "log floor-area-weighted job-months", "offset",
     SPEC_ALT),
    ("n_disc_constr", "discretionary construction (8A/1X/91..., excl 7G)",
     "log_caller_con", "log1p caller construction-incident count", "offset",
     SPEC_ALT),
]


def run_allocation(df: pd.DataFrame) -> pd.DataFrame:
    rhs = " + ".join(DEMOS + STOCK)
    rows = []
    for outcome, label, off, off_label, how, spec in ALLOC_MODELS:
        sub = df[df[off].notna()]
        t0 = time.time()
        if how == "offset":
            m = pf.fepois(f"{outcome} ~ {rhs} | boro", data=sub, offset=off,
                          vcov={"CRV1": "nta"})
        else:  # free elasticity: log exposure enters as a regressor
            m = pf.fepois(f"{outcome} ~ {rhs} + {off} | boro", data=sub,
                          vcov={"CRV1": "nta"})
        td = m.tidy().reset_index()
        for _, r in td.iterrows():
            term = r["Coefficient"]
            b, lo, hi = r["Estimate"], r["2.5%"], r["97.5%"]
            demo = term in DEMOS
            term_label = DEMO_LABELS.get(term, term)
            if how == "free" and term == off:
                term_label = "exposure elasticity (offset spec forces 1.0)"
            rows.append({
                "part": "allocation", "outcome": outcome,
                "outcome_label": label, "exposure_offset": off_label,
                "spec": spec, "term": term,
                "term_label": term_label,
                "b": b, "se": r["Std. Error"], "t": r["t value"],
                "p": r["Pr(>|t|)"], "ci_lo": lo, "ci_hi": hi,
                "per_10pp": 100 * (np.exp(0.1 * b) - 1) if demo else np.nan,
                "per_10pp_lo": 100 * (np.exp(0.1 * lo) - 1) if demo else np.nan,
                "per_10pp_hi": 100 * (np.exp(0.1 * hi) - 1) if demo else np.nan,
                "per_10pp_unit": "pct_change_in_count" if demo else "",
                "n_obs": int(m._N), "n_clusters": sub["nta"].nunique(),
                "cluster": "nta", "fe": "boro",
                "controls": ("demographics jointly + era shares + mean log "
                             "bldg area + commercial share + log mean units"
                             + (" + log exposure (free elasticity)"
                                if how == "free" else "")),
                "window": "2020-01..2026-05",
                "outcome_total": int(sub[outcome].sum()),
                "baseline": (float(sub[outcome].sum()
                                   / np.exp(sub[off]).sum())
                             if how == "offset" else np.nan),
                "note": GEO_NOTE,
            })
        key = td[td["Coefficient"].isin(DEMOS)]
        msg = ", ".join(f"{r['Coefficient']} {100 * (np.exp(0.1 * r['Estimate']) - 1):+.1f}%"
                        for _, r in key.iterrows())
        print(f"  {outcome:<14} {how:<6} {off_label:<34} n={m._N:,} "
              f"({time.time() - t0:.0f}s)  per +10pp: {msg}")
        if how == "free":
            el = td[td["Coefficient"] == off].iloc[0]
            print(f"    fitted exposure elasticity {el['Estimate']:.3f} "
                  f"[{el['2.5%']:.3f}, {el['97.5%']:.3f}] "
                  f"(the offset spec forces 1.0)")
    return pd.DataFrame(rows)


# ── Part 2: crisis-response equity ───────────────────────────────────────

def make_day_counter(sub: pd.DataFrame):
    keys = np.sort(sub["bbl_i"].to_numpy() * DAY_STRIDE + sub["day"].to_numpy())

    def count(bbl_arr, lo_arr, hi_arr):
        """Events per BBL with day in [lo, hi] inclusive."""
        lo = bbl_arr * DAY_STRIDE + lo_arr
        hi = bbl_arr * DAY_STRIDE + hi_arr
        return (np.searchsorted(keys, hi, side="right")
                - np.searchsorted(keys, lo, side="left"))

    return count


def build_crisis_frame(ev: pd.DataFrame, tract: pd.DataFrame,
                       traits: pd.DataFrame) -> pd.DataFrame:
    e = ev[ev["bbl"].notna() & ev["bbl"].str.fullmatch(r"\d{10}")].copy()
    e["bbl_i"] = e["bbl"].astype("int64")

    c30 = e[(e["category_prefix"] == "30") & (e["agency"] == 0)]
    first = c30.sort_values("day").drop_duplicates("bbl_i").copy()
    n_first = len(first)
    last_day = (WINDOW_END - WINDOW_START).days - RESPONSE_DAYS
    first = first[first["day"] <= last_day]
    print(f"crisis events: {len(c30):,} caller cat-30 -> {n_first:,} "
          f"first-per-lot -> {len(first):,} with {RESPONSE_DAYS}-day room "
          f"(t0 through {(WINDOW_START + pd.Timedelta(days=last_day)):%Y-%m-%d})")

    ag_all = make_day_counter(e[e["agency"] == 1])
    ag_x30 = make_day_counter(e[(e["agency"] == 1)
                                & (e["category_prefix"] != "30")])
    b = first["bbl_i"].to_numpy()
    d = first["day"].to_numpy()
    first["resp_any"] = (ag_all(b, d, d + RESPONSE_DAYS) > 0).astype(int)
    first["resp_excl30"] = (ag_x30(b, d, d + RESPONSE_DAYS) > 0).astype(int)
    first["resp_day1"] = (ag_all(b, d + 1, d + RESPONSE_DAYS) > 0).astype(int)

    df = first.merge(tract[DEMOS], on="tract", how="inner")
    df = df.merge(traits, left_on="bbl", right_index=True, how="left")
    df["bldg_missing"] = df["log_units"].isna().astype(int)
    for c in ("log_units", "log_ba", "prewar", "com_class"):
        df[c] = df[c].fillna(df[c].median())
    df["priority"] = df["priority"].fillna("missing")
    df["boro_month"] = df["bbl"].str[0] + "_" + df["received"].dt.strftime("%Y-%m")
    df["tract_s"] = df["tract"].astype(str)
    print(f"crisis frame: {len(df):,} events with tract demographics "
          f"({df['tract_s'].nunique():,} tract clusters); building traits "
          f"missing {df['bldg_missing'].mean():.1%} (median-filled + "
          f"indicator); response rates any {df['resp_any'].mean():.3f} / "
          f"excl cat-30 {df['resp_excl30'].mean():.3f} / "
          f"excl same-day {df['resp_day1'].mean():.3f}")
    return df


CRISIS_MODELS = [
    ("resp_any", "any agency event at BBL within 90 days"),
    ("resp_excl30", "agency event excl cat-30 follow-ups, 90 days"),
    ("resp_day1", "any agency event days 1-90 (excl same-day)"),
]


def run_crisis(df: pd.DataFrame) -> pd.DataFrame:
    rhs = " + ".join(DEMOS) + " + C(priority) + " + " + ".join(BLD_TRAITS)
    rows = []
    for outcome, label in CRISIS_MODELS:
        m = pf.feols(f"{outcome} ~ {rhs} | boro_month", data=df,
                     vcov={"CRV1": "tract_s"})
        td = m.tidy().reset_index()
        base = float(df[outcome].mean())
        for _, r in td.iterrows():
            term = r["Coefficient"]
            b, lo, hi = r["Estimate"], r["2.5%"], r["97.5%"]
            demo = term in DEMOS
            rows.append({
                "part": "crisis_response", "outcome": outcome,
                "outcome_label": label,
                "exposure_offset": "", "spec": "",
                "term": term,
                "term_label": DEMO_LABELS.get(term, term),
                "b": b, "se": r["Std. Error"], "t": r["t value"],
                "p": r["Pr(>|t|)"], "ci_lo": lo, "ci_hi": hi,
                "per_10pp": 10 * b if demo else np.nan,
                "per_10pp_lo": 10 * lo if demo else np.nan,
                "per_10pp_hi": 10 * hi if demo else np.nan,
                "per_10pp_unit": "pp_probability" if demo else "",
                "n_obs": int(m._N), "n_clusters": df["tract_s"].nunique(),
                "cluster": "tract", "fe": "boro_month",
                "controls": ("demographics jointly + priority dummies + "
                             "log units + log bldg area + prewar + "
                             "commercial class + missing-trait indicator + "
                             "active-permit flag"),
                "window": "first caller cat-30 per lot, 2020-01..2026-03",
                "outcome_total": int(df[outcome].sum()),
                "baseline": base,
                "note": ("agency-originated complaint events proxy "
                         "proactive inspections"),
            })
        key = td[td["Coefficient"].isin(DEMOS)]
        msg = ", ".join(f"{r['Coefficient']} {10 * r['Estimate']:+.2f}pp"
                        for _, r in key.iterrows())
        print(f"  {outcome:<12} base {base:.3f}  n={m._N:,}  "
              f"per +10pp: {msg}")
    return pd.DataFrame(rows)


# ── figure ───────────────────────────────────────────────────────────────

def pick(est, part, outcome, offset_label=None, spec=None):
    g = est[(est["part"] == part) & (est["outcome"] == outcome)
            & est["term"].isin(DEMOS)]
    if offset_label is not None:
        g = g[g["exposure_offset"] == offset_label]
    if spec is not None:
        g = g[g["spec"] == spec]
    assert len(g) == len(DEMOS), (part, outcome, offset_label, spec, len(g))
    return g.set_index("term").loc[DEMOS]


def make_figure(est: pd.DataFrame, meta: dict) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.8), dpi=160, sharey=True,
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.08,
                     "left": 0.10, "right": 0.985, "top": 0.775,
                     "bottom": 0.21})
    y = np.arange(len(DEMOS))[::-1]

    def draw(ax, g, color, dy, label=None, contrast=False):
        """Filled marker for the discretionary margin, open marker for its
        spec-matched statutory contrast (same hue carries the pairing)."""
        lw, alpha = (1.6, 0.85) if not contrast else (1.2, 0.5)
        face, edge = (color, SURFACE) if not contrast else (SURFACE, color)
        ax.hlines(y + dy, g["per_10pp_lo"], g["per_10pp_hi"], color=color,
                  lw=lw, alpha=alpha, zorder=3)
        ax.plot(g["per_10pp"], y + dy, "o", ms=6.5 if not contrast else 5.5,
                markerfacecolor=face, markeredgecolor=edge,
                markeredgewidth=1.4, zorder=4, linestyle="none", label=label)

    # Panel A: discretionary construction, free-elasticity headline plus
    # the at/off-permit split; the superseded unit-elasticity offset spec
    # stays visible as the open-marker contrast
    ax = axes[0]
    ax.axvline(0, color=ZERO_C, linewidth=1.1, zorder=1)
    draw(ax, pick(est, "allocation", "n_disc_constr", spec=SPEC_FREE), BLUE,
         +0.27, "discretionary construction · free elasticity")
    draw(ax, pick(est, "allocation", "n_atpermit", spec=SPEC_SPLIT), AQUA,
         +0.09, "at an active permit · job-month offset")
    draw(ax, pick(est, "allocation", "n_offpermit", spec=SPEC_SPLIT), RED,
         -0.09, "off permit (EWO / illegal work) · job-month offset")
    draw(ax, pick(est, "allocation", "n_disc_constr",
                  "log active job-months 2020-26", SPEC_OFFSET), MUTED,
         -0.27, "unit-elasticity offset (superseded)", contrast=True)
    style_ax(ax)
    ax.set_ylim(-0.65, 5.45)     # headroom band for the legend
    ax.set_yticks(y)
    ax.set_yticklabels([DEMO_LABELS[d] for d in DEMOS], fontsize=11.5,
                       color=INK)
    ax.set_xlabel("% change in proactive events per +10pp of tract share",
                  fontsize=10.5)
    ax.set_title("Where discretionary construction oversight goes",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=22)
    ax.text(0, 1.035, f"{meta['n_tracts']:,} tracts, borough FE, "
            "SEs clustered on NTA", transform=ax.transAxes, fontsize=9.5,
            color=MUTED)
    # opaque surface frame: the zero line and x-gridlines run through the
    # legend band otherwise
    ax.legend(loc="upper center", ncols=2, frameon=True, fontsize=8.4,
              facecolor=SURFACE, edgecolor="none", framealpha=1.0,
              handletextpad=0.35, columnspacing=1.1, borderaxespad=0.1)

    # Panel B: crisis response
    ax = axes[1]
    ax.axvline(0, color=ZERO_C, linewidth=1.1, zorder=1)
    draw(ax, pick(est, "crisis_response", "resp_any"), RED, 0.0)
    style_ax(ax)
    ax.set_xlabel("pp change in 90-day follow-up probability per +10pp",
                  fontsize=10.5)
    ax.set_title("Follow-through after an unstable-building report",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=22)
    ax.text(0, 1.035, f"{meta['n_events']:,} first reports, baseline "
            f"{meta['crisis_base']:.0%} get an agency visit within 90 days",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)

    fig.suptitle(meta["title"], x=0.01, y=0.985, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.935, meta["subtitle"], fontsize=10, color=MUTED,
             va="top", linespacing=1.45)
    fig.text(0.01, 0.012,
             "Left: tract-level PPML of agency-initiated discretionary-"
             "construction counts (excl 7G) 2020-2026 on tract demographics "
             "(jointly entered), controlling era shares, mean log\nbuilding "
             "area, commercial share, and log mean units, borough FE, SEs "
             "clustered on NTA. Blue enters log active job-months as a "
             "regressor (fitted elasticity ~0.7); green and\nred split the "
             "bundle by active-permit status under the job-month offset; "
             "open gray forces the unit-elasticity offset (superseded as "
             "headline). Right: LPM among each lot's first\ncaller-reported "
             "category-30 complaint, controlling priority, building traits, "
             "and an active-permit flag, borough x month FE, SEs clustered "
             "by tract. Whiskers are 95% CIs.",
             fontsize=8.6, color=MUTED, linespacing=1.5)
    out = ART / "proactive_equity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out}")


# ── verdict ──────────────────────────────────────────────────────────────

def print_verdict(est: pd.DataFrame) -> None:
    def fmt(r):
        return (f"{r['per_10pp']:+.1f}% [{r['per_10pp_lo']:+.1f}, "
                f"{r['per_10pp_hi']:+.1f}]")

    print("\n== Demographic loadings per +10pp of tract share, 95% CI ==")
    blocks = [
        ("HEADLINE discretionary construction / free elasticity",
         pick(est, "allocation", "n_disc_constr", spec=SPEC_FREE)),
        ("7G sweeps / free elasticity (unit exposure)",
         pick(est, "allocation", "n_7g", spec=SPEC_FREE)),
        ("statutory contrast / free elasticity (unit exposure)",
         pick(est, "allocation", "n_stat", "log residential units",
              SPEC_FREE)),
        ("statutory contrast / free elasticity (job-month exposure)",
         pick(est, "allocation", "n_stat", "log active job-months 2020-26",
              SPEC_FREE)),
        ("at-permit bundle events / job-month offset",
         pick(est, "allocation", "n_atpermit", spec=SPEC_SPLIT)),
        ("OFF-permit bundle events / job-month offset",
         pick(est, "allocation", "n_offpermit", spec=SPEC_SPLIT)),
        ("8A-only / job-month offset",
         pick(est, "allocation", "n_8a", spec=SPEC_SPLIT)),
        ("alt exposure: floor-area-weighted job-months",
         pick(est, "allocation", "n_disc_constr",
              "log floor-area-weighted job-months", SPEC_ALT)),
        ("alt exposure: caller construction-incident volume",
         pick(est, "allocation", "n_disc_constr",
              "log1p caller construction-incident count", SPEC_ALT)),
        ("SUPERSEDED discretionary construction / unit-elasticity offset",
         pick(est, "allocation", "n_disc_constr",
              "log active job-months 2020-26", SPEC_OFFSET)),
        ("SUPERSEDED 7G sweeps / unit-elasticity offset",
         pick(est, "allocation", "n_7g", spec=SPEC_OFFSET)),
        ("SUPERSEDED statutory / unit offset",
         pick(est, "allocation", "n_stat", "log residential units",
              SPEC_OFFSET)),
        ("SUPERSEDED statutory / job-month offset",
         pick(est, "allocation", "n_stat", "log active job-months 2020-26",
              SPEC_OFFSET)),
    ]
    for name, g in blocks:
        print(f"\n  {name}")
        for term in DEMOS:
            r = g.loc[term]
            star = "" if r["ci_lo"] <= 0 <= r["ci_hi"] else "  <- CI excludes 0"
            print(f"    {DEMO_LABELS[term]:<15} {fmt(r)}{star}")

    el = est[(est["part"] == "allocation")
             & (est["outcome"] == "n_disc_constr")
             & (est["spec"] == SPEC_FREE)
             & (est["term"] == "log_job_months")].iloc[0]
    print(f"\n  fitted job-month elasticity (discretionary construction): "
          f"{el['b']:.3f} [{el['ci_lo']:.3f}, {el['ci_hi']:.3f}]; the "
          f"offset spec forces 1.0, which inflates loadings on "
          f"low-exposure tracts")

    print("\n  crisis response, pp per +10pp (baseline in CSV)")
    for outcome, _ in CRISIS_MODELS:
        g = pick(est, "crisis_response", outcome)
        print(f"    [{outcome}]")
        for term in DEMOS:
            r = g.loc[term]
            star = "" if r["ci_lo"] <= 0 <= r["ci_hi"] else "  <- CI excludes 0"
            print(f"    {DEMO_LABELS[term]:<15} {r['per_10pp']:+.2f}pp "
                  f"[{r['per_10pp_lo']:+.2f}, {r['per_10pp_hi']:+.2f}]{star}")

    # spec-matched, direction-aware placebo reading on the HEADLINE
    # (free-elasticity) rows: a discretionary loading is discounted only
    # when the statutory contrast under the IDENTICAL exposure loads with
    # the same sign and its CI excludes 0
    stat_for = {
        "n_disc_constr": pick(est, "allocation", "n_stat",
                              "log active job-months 2020-26", SPEC_FREE),
        "n_7g": pick(est, "allocation", "n_stat", "log residential units",
                     SPEC_FREE),
    }
    print("\n== Plain verdict (free elasticity, spec-matched statutory "
          "placebo) ==")
    for outcome, name in [("n_disc_constr", "discretionary construction"),
                          ("n_7g", "7G sweeps")]:
        g = pick(est, "allocation", outcome,
                 "log active job-months 2020-26" if outcome == "n_disc_constr"
                 else None, SPEC_FREE)
        s = stat_for[outcome]
        clean, shared = [], []
        for t in DEMOS:
            if g.loc[t, "ci_lo"] <= 0 <= g.loc[t, "ci_hi"]:
                continue
            stat_same_sign = (not s.loc[t, "ci_lo"] <= 0 <= s.loc[t, "ci_hi"]
                              and np.sign(s.loc[t, "b"]) == np.sign(g.loc[t, "b"]))
            (shared if stat_same_sign else clean).append(t)
        print(f"  {name}")
        print(f"    reads as allocation (statutory null or opposite-signed): "
              f"{[DEMO_LABELS[t] for t in clean] or 'none'}")
        print(f"    shared with statutory, reads as residual stock "
              f"mechanics: {[DEMO_LABELS[t] for t in shared] or 'none'}")

    at = pick(est, "allocation", "n_atpermit", spec=SPEC_SPLIT)
    off = pick(est, "allocation", "n_offpermit", spec=SPEC_SPLIT)
    print("  at/off-permit reading")
    for t in ["pct_black", "pct_hispanic", "pct_asian"]:
        a, o = at.loc[t], off.loc[t]
        print(f"    {DEMO_LABELS[t]:<15} at-permit {a['per_10pp']:+.1f}% "
              f"[{a['per_10pp_lo']:+.1f}, {a['per_10pp_hi']:+.1f}]  "
              f"off-permit {o['per_10pp']:+.1f}% "
              f"[{o['per_10pp_lo']:+.1f}, {o['per_10pp_hi']:+.1f}]")
    print("    Black/Hispanic loadings concentrate off-permit (enforcement "
          "against unpermitted work; the permit-based offset undercounts "
          "true construction exposure there); the Asian loading partially "
          "survives at-permit. Do not lump the groups.")

    cr = pick(est, "crisis_response", "resp_any")
    cr_sig = [t for t in DEMOS
              if not cr.loc[t, "ci_lo"] <= 0 <= cr.loc[t, "ci_hi"]]
    print(f"  crisis response")
    print(f"    90-day follow-up gaps with CI excluding 0: "
          f"{[DEMO_LABELS[t] for t in cr_sig] or 'none'} "
          f"(check robustness rows resp_excl30 / resp_day1 before quoting)")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    RM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    ev = load_events()
    tract, traits = load_v2()

    print("\nPart 1: allocation PPML (offsets, borough FE, NTA clusters)")
    tf = build_tract_frame(ev, tract)
    est_a = run_allocation(tf)

    print("\nPart 2: crisis-response LPM (borough x month FE, tract clusters)")
    cf = build_crisis_frame(ev, tract, traits)
    est_c = run_crisis(cf)

    est = pd.concat([est_a, est_c], ignore_index=True)
    out_csv = RM / "proactive_equity.csv"
    est.to_csv(out_csv, index=False)
    print(f"\nEstimates -> {out_csv} ({len(est)} rows)")

    print_verdict(est)

    free = pick(est, "allocation", "n_disc_constr", spec=SPEC_FREE)
    at = pick(est, "allocation", "n_atpermit", spec=SPEC_SPLIT)
    off = pick(est, "allocation", "n_offpermit", spec=SPEC_SPLIT)
    el = est[(est["part"] == "allocation")
             & (est["outcome"] == "n_disc_constr")
             & (est["spec"] == SPEC_FREE)
             & (est["term"] == "log_job_months")].iloc[0]
    meta = {
        "n_tracts": len(tf),
        "n_events": len(cf),
        "crisis_base": float(cf["resp_any"].mean()),
        "title": ("Construction oversight tilts modestly toward nonwhite "
                  "tracts, mostly at unpermitted work"),
        "subtitle": (
            "freeing the exposure elasticity (the offset forces 1.0; the "
            f"data say {el['b']:.2f}) roughly halves the loadings · per "
            f"+10pp of tract share, Black {free.loc['pct_black', 'per_10pp']:+.1f}%, "
            f"Hispanic {free.loc['pct_hispanic', 'per_10pp']:+.1f}%, Asian "
            f"{free.loc['pct_asian', 'per_10pp']:+.1f}%\n· the Black and "
            "Hispanic tilt sits at lots without an active permit "
            f"(off-permit Black {off.loc['pct_black', 'per_10pp']:+.1f}%, "
            f"at-permit {at.loc['pct_black', 'per_10pp']:+.1f}%); the Asian "
            f"tilt persists at permitted jobs "
            f"({at.loc['pct_asian', 'per_10pp']:+.1f}%) · crisis "
            "follow-through close to flat"),
    }
    make_figure(est, meta)
    print(f"\nTOTAL {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
