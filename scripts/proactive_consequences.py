#!/usr/bin/env python3
"""
Post-slated findings from the New-hypothesis hunt
(data/analysis/blog_posts/proactive_enforcement_plan.md, 2026-07-10),
regenerated through the script->CSV convention. One CSV, four labeled
blocks; every modeled number carries a CI; every pilot number from the
hunt is re-derived here and verified in the printed summary.

Blocks (column `block` in the output CSV)

1. swo_revolving_door
   Stop-work episodes reconstructed from bis_scrape dispositions
   (format "MM/DD/YYYY - CODE - TEXT"). An EPISODE is a complaint whose
   disposition text contains "STOP WORK ORDER FULLY RESCINDED" with a
   parseable leading date (codes L2 full rescission and L4 after-hours
   full rescission). The SWO in-force proxy runs from the complaint's
   received date to the rescission (disposition) date. SWO-family
   evidence codes: A3 full SWO served, L1 partial SWO, L3 partial
   rescission, H3 violation for failure to obey an SWO, plus the
   rescission rows themselves.
   Re-SWO headline: among episodes rescinded 2021-23 (so the 365-day
   window sits well inside scrape coverage, and the later complaint has
   had time to resolve to a disposition), the share of BINs where a
   DIFFERENT complaint RECEIVED within (rescission, rescission+365d]
   went on to draw an SWO-family disposition -- i.e., a genuinely new
   stoppage, not a companion complaint of the same one. Alternates in
   the CSV: keying on the other complaint's disposition date (counts
   overlapping companions, upper bound) and windowing episodes by
   received year instead.
   H3 defiance: share of episodes where a different complaint at the
   same BIN drew an H3 disposition dated inside the in-force window.

2. default_economy
   ECB (OATH) violations, construction family, issued 2020-24.
   Construction family = violation_type IN ('Construction',
   'Site Safety', 'Cranes and Derricks') -- the hunt's definition:
   'Construction' plus the device/site-safety construction families
   (pinned by exact reproduction of the pilot numbers 40,144 / $274M /
   22.2%; 'Construction' alone gives 39,793 / $270M). Counts, imposed
   and paid dollars, and pay rate (sum amount_paid / sum
   penality_imposed, source spelling) by hearing_status; DEFAULT vs
   adjudicated-appeared (= IN VIOLATION: respondent appeared and was
   found in violation); certification_status 'NO COMPLIANCE RECORDED'.
   ECB-only by design: hearing/payment/certification fields exist only
   in the ECB stream. This is not an owner-level violation count, so
   the BIS+DOB NOW union ledger rule does not apply.

3. conversion_hazard
   Same PPML machinery as scripts/proactive_monitoring_per_permit.py
   (imported from it: base-project spine, event->project assignment,
   log active-months offset, nta^first-permit-quarter FE, SEs clustered
   on BBL, identical covariates), outcomes swapped to hazard measures:
     n_class1  events carrying a Class-1 ECB violation ref (spine
               first-token join, ecb_severity == 'CLASS - 1'), any
               origin
     n_injury  injury/incident complaints: category prefix 91 (worker
               endangerment) or 1E (scaffold/hoist accident), plus
               priority-A complaints in the mixed_incident families
               (30/10/12/14/03/23/67), any origin
   plus caller-only variants of both (detection-independent robustness:
   agency == 0). Conversion-flag IRRs with CIs; the monitoring
   (n_disc) and caller placebo (n_caller_incident, n_caller_all) IRRs
   are copied alongside from the existing
   proactive_monitoring_estimates.csv for the riskier-vs-more-visible
   comparison.

4. shell_respondents + swo_escalation (one-liners)
   shell_respondents: construction-family ECB violations issued 2020+
   with a valid BIN (non-null, not the borough placeholder [1-5]000000)
   whose respondent_name (uppercased, trimmed) contains 'LLC'. A
   respondent is SINGLE-BUILDING if its violations sit at exactly one
   distinct BIN. Default share and pay rate by single vs multi. An
   alternate row set collapses punctuation/whitespace in names before
   grouping (more aggressive dedup).
   swo_escalation: complaints whose disposition code is A3 (full SWO
   served), by RECEIVED year 2023-25 (received-year keying reproduces
   the pilot 206->853 exactly); the category-05 component (NO BUILDING
   PERMIT: construction/demolition) and caller-origin shares
   (ref_311 non-empty).

Output: data/analysis/risk_models/proactive_consequences.csv
Run:    /private/tmp/pyfix_venv/bin/python scripts/proactive_consequences.py
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analysis_config import _PROACTIVE_FAMILY_PREFIXES  # noqa: E402
from proactive_monitoring_per_permit import (  # noqa: E402  (PPML machinery)
    CONV_DEFS, JOB_TYPE_DUMMIES, X_TERMS, assign_events, load_jobs,
    tract_nta_crosswalk)

DB = ROOT / "data" / "dob_complaints.db"
SPINE = ROOT / "data" / "analysis" / "proactive"
RM = ROOT / "data" / "analysis" / "risk_models"
OUT = RM / "proactive_consequences.csv"
MONITORING_CSV = RM / "proactive_monitoring_estimates.csv"

DISP_RE = r"^(\d{2}/\d{2}/\d{4}) - ([A-Z0-9]{2}) - (.*)$"
RESC_TEXT = "STOP WORK ORDER FULLY RESCINDED"
SWO_EVENT_CODES = ("A3", "L1", "L3", "H3")
CONSTRUCTION_ECB_TYPES = ("Construction", "Site Safety", "Cranes and Derricks")
BIN_PLACEHOLDER = r"[1-5]000000"

rows: list[dict] = []
checks: list[tuple[str, str, str, bool, str]] = []  # block, label, pilot, got-ok, got


def add_row(block, metric, group, value, *, ci_lo=np.nan, ci_hi=np.nan,
            ci_method="", b=np.nan, se=np.nan, p=np.nan, n=np.nan,
            n_events=np.nan, window="", definition="", pilot="", source=""):
    rows.append({
        "block": block, "metric": metric, "group": group, "value": value,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "ci_method": ci_method,
        "b": b, "se": se, "p": p, "n": n, "n_events": n_events,
        "window": window, "definition": definition, "pilot": pilot,
        "source": source,
    })


def check(block, label, pilot_str, got_str, ok, note=""):
    checks.append((block, label, pilot_str, got_str, ok, note))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p_ = k / n
    denom = 1 + z ** 2 / n
    center = p_ + z ** 2 / (2 * n)
    half = z * np.sqrt(p_ * (1 - p_) / n + z ** 2 / (4 * n ** 2))
    return (center - half) / denom, (center + half) / denom


def cluster_prop_ci(hit, cluster, z=1.96):
    """Cluster-robust (linearized ratio) CI for a proportion."""
    g = pd.DataFrame({"h": hit.astype(float).to_numpy(),
                      "c": cluster.to_numpy()}).groupby("c")["h"]
    ks, ns = g.sum(), g.count()
    p_ = ks.sum() / ns.sum()
    m = len(ks)
    se = np.sqrt(m / (m - 1) * ((ks - p_ * ns) ** 2).sum()) / ns.sum()
    return p_ - z * se, p_ + z * se


# ── block 1: swo_revolving_door ──────────────────────────────────────────

def block1_swo(con) -> None:
    blk = "swo_revolving_door"
    print("\n== Block 1: swo_revolving_door ==")
    d = pd.read_sql_query(
        "SELECT complaint_number, bin, received_date, disposition "
        "FROM bis_scrape WHERE disposition IS NOT NULL AND disposition != ''",
        con)
    ext = d["disposition"].str.extract(DISP_RE)
    d["disp_date"] = pd.to_datetime(ext[0], format="%m/%d/%Y", errors="coerce")
    d["disp_code"] = ext[1]
    d["received"] = pd.to_datetime(d["received_date"], errors="coerce")
    n_unparsed = int(d["disp_code"].isna().sum())
    print(f"  {len(d):,} disposed complaints; {n_unparsed:,} dispositions "
          f"do not match 'MM/DD/YYYY - CODE - TEXT' (excluded)")

    is_resc = (d["disposition"].str.contains(RESC_TEXT, na=False)
               & d["disp_date"].notna())
    resc = d[is_resc].copy()
    code_mix = resc["disp_code"].value_counts().to_dict()
    n_ep = len(resc)
    days = (resc["disp_date"] - resc["received"]).dt.days
    med, p90 = days.median(), days.quantile(0.90)
    print(f"  episodes with parseable rescission date: {n_ep:,} "
          f"(codes {code_mix}); complaint-to-rescission median "
          f"{med:.0f}d, p90 {p90:.0f}d")

    ep_def = ("episode = complaint whose disposition contains 'STOP WORK "
              "ORDER FULLY RESCINDED' with parseable leading date "
              f"(codes: {code_mix}); days = rescission (disposition) date "
              "minus complaint received date")
    add_row(blk, "n_episodes", "all", n_ep, window="full scrape",
            definition=ep_def, pilot="41,385", source="bis_scrape")
    add_row(blk, "median_days_complaint_to_rescission", "all", med,
            n=n_ep, window="full scrape", definition=ep_def, pilot="35",
            source="bis_scrape")
    add_row(blk, "p90_days_complaint_to_rescission", "all", p90,
            n=n_ep, window="full scrape", definition=ep_def, pilot="307",
            source="bis_scrape")
    check(blk, "episodes with rescission date", "41,385", f"{n_ep:,}",
          n_ep == 41385)
    check(blk, "median days complaint->rescission", "35", f"{med:.0f}",
          round(med) == 35)
    check(blk, "p90 days complaint->rescission", "307", f"{p90:.0f}",
          round(p90) == 307)

    # SWO-family evidence events (any complaint), valid rows only
    swo_ev = d[(d["disp_code"].isin(SWO_EVENT_CODES) | is_resc)
               & d["disp_date"].notna() & d["bin"].notna()]
    ev = swo_ev[["complaint_number", "bin", "disp_date", "received"]].rename(
        columns={"complaint_number": "cn_ev", "disp_date": "ev_disp",
                 "received": "ev_recv"})

    def reswo_share(w, key, strict_after=True):
        m = w[["complaint_number", "bin", "disp_date"]].merge(ev, on="bin")
        m = m[m["cn_ev"] != m["complaint_number"]]
        k = m[key]
        lo = m["disp_date"]
        hi = m["disp_date"] + pd.Timedelta(days=365)
        m = m[(k > lo) & (k <= hi)] if strict_after else m[(k >= lo) & (k <= hi)]
        return w["complaint_number"].isin(m["complaint_number"])

    # headline: episodes rescinded 2021-23; new complaint received in
    # (rescission, rescission+365d] that drew an SWO-family disposition
    w = resc[resc["disp_date"].dt.year.between(2021, 2023)].copy()
    hit = reswo_share(w, "ev_recv")
    lo, hi = cluster_prop_ci(hit, w["bin"])
    reswo = hit.mean()
    print(f"  re-SWO within 365d of rescission (2021-23 episodes, new "
          f"complaint received after rescission): {reswo:.1%} "
          f"[{lo:.1%}, {hi:.1%}] of {len(w):,}")
    add_row(blk, "n_episodes_2021_23", "rescinded 2021-23", len(w),
            window="rescission 2021-23", definition=ep_def,
            source="bis_scrape")
    add_row(blk, "share_reswo_within_365d", "headline", reswo,
            ci_lo=lo, ci_hi=hi, ci_method="cluster-robust (BIN)",
            n=len(w), n_events=int(hit.sum()), window="rescission 2021-23",
            definition=("different complaint at same BIN, RECEIVED in "
                        "(rescission, rescission+365d], that drew an "
                        "SWO-family disposition (A3/L1/L3/H3 or another "
                        "full rescission); received-keying excludes "
                        "companion complaints of the same stoppage"),
            pilot="37.4%", source="bis_scrape")
    check(blk, "re-SWO share within 365d (2021-23)", "37.4%",
          f"{reswo * 100:.1f}%", abs(reswo - 0.374) < 0.0005,
          "definition-sensitive; alternates in CSV span 35.1-41.0%")

    # alternates
    hit_a = reswo_share(w, "ev_disp")
    la, ha = cluster_prop_ci(hit_a, w["bin"])
    add_row(blk, "share_reswo_within_365d", "alt: SWO disposition dated in window",
            hit_a.mean(), ci_lo=la, ci_hi=ha,
            ci_method="cluster-robust (BIN)", n=len(w),
            n_events=int(hit_a.sum()), window="rescission 2021-23",
            definition=("different complaint's SWO-family disposition DATED "
                        "in (rescission, rescission+365d]; counts "
                        "overlapping companion complaints of the same "
                        "stoppage, upper bound"), source="bis_scrape")
    w2 = resc[resc["received"].dt.year.between(2021, 2023)].copy()
    hit_b = reswo_share(w2, "ev_recv")
    lb, hb = cluster_prop_ci(hit_b, w2["bin"])
    add_row(blk, "share_reswo_within_365d", "alt: episodes by received year 2021-23",
            hit_b.mean(), ci_lo=lb, ci_hi=hb,
            ci_method="cluster-robust (BIN)", n=len(w2),
            n_events=int(hit_b.sum()), window="received 2021-23",
            definition="headline event rule, episode window on received year",
            source="bis_scrape")
    print(f"  alternates: disposition-dated {hit_a.mean():.1%}; "
          f"received-year window {hit_b.mean():.1%}")

    # H3 defiance while the SWO was in force (all episodes)
    h3 = d[(d["disp_code"] == "H3") & d["disp_date"].notna()
           & d["bin"].notna()][["complaint_number", "bin", "disp_date"]]
    h3 = h3.rename(columns={"complaint_number": "cn_h3",
                            "disp_date": "h3_disp"})
    m = resc[["complaint_number", "bin", "received", "disp_date"]].merge(
        h3, on="bin")
    m = m[(m["cn_h3"] != m["complaint_number"])
          & (m["h3_disp"] >= m["received"])
          & (m["h3_disp"] <= m["disp_date"])]
    hit_h = resc["complaint_number"].isin(m["complaint_number"])
    lh, hh = cluster_prop_ci(hit_h, resc["bin"])
    print(f"  H3 defiance while in force: {hit_h.mean():.1%} "
          f"[{lh:.1%}, {hh:.1%}] of {n_ep:,}")
    add_row(blk, "share_h3_while_in_force", "all episodes", hit_h.mean(),
            ci_lo=lh, ci_hi=hh, ci_method="cluster-robust (BIN)",
            n=n_ep, n_events=int(hit_h.sum()), window="full scrape",
            definition=("different complaint at same BIN with H3 (failure "
                        "to obey SWO) disposition dated inside "
                        "[received, rescission] of the episode"),
            pilot="4.7%", source="bis_scrape")
    check(blk, "H3 defiance while in force", "4.7%",
          f"{hit_h.mean() * 100:.1f}%", abs(hit_h.mean() - 0.047) < 0.0005,
          "regenerated definition lands 0.1pp under the pilot")


# ── block 2: default_economy ─────────────────────────────────────────────

def block2_default(con) -> None:
    blk = "default_economy"
    print("\n== Block 2: default_economy ==")
    types = ", ".join(f"'{t}'" for t in CONSTRUCTION_ECB_TYPES)
    e = pd.read_sql_query(
        f"SELECT hearing_status, certification_status, penality_imposed, "
        f"amount_paid FROM ecb_violations "
        f"WHERE violation_type IN ({types}) "
        f"AND substr(issue_date, 1, 4) BETWEEN '2020' AND '2024'", con)
    e["imposed"] = pd.to_numeric(e["penality_imposed"], errors="coerce")
    e["paid"] = pd.to_numeric(e["amount_paid"], errors="coerce")
    fam_def = (f"construction family = violation_type IN "
               f"({', '.join(CONSTRUCTION_ECB_TYPES)}), issued 2020-24; "
               "pay rate = sum(amount_paid)/sum(penality_imposed)")
    win = "issued 2020-24"

    g = (e.assign(hs=e["hearing_status"].fillna("").replace("", "(blank)"))
         .groupby("hs").agg(n=("hs", "size"), imposed=("imposed", "sum"),
                            paid=("paid", "sum")))
    g["pay_rate"] = g["paid"] / g["imposed"]
    for hs, r in g.sort_values("n", ascending=False).iterrows():
        add_row(blk, "violations_by_hearing_status", hs, int(r["n"]),
                window=win, definition=fam_def, source="ecb_violations",
                pilot="40,144" if hs == "DEFAULT" else "")
        add_row(blk, "dollars_imposed_by_hearing_status", hs, r["imposed"],
                n=int(r["n"]), window=win, definition=fam_def,
                source="ecb_violations",
                pilot="$274M" if hs == "DEFAULT" else "")
        add_row(blk, "dollars_paid_by_hearing_status", hs, r["paid"],
                n=int(r["n"]), window=win, definition=fam_def,
                source="ecb_violations")
        add_row(blk, "pay_rate_by_hearing_status", hs, r["pay_rate"],
                n=int(r["n"]), window=win,
                definition=fam_def + ("; DISMISSED/CURED rows can exceed 1 "
                                      "(payments against near-zero imposed)"
                                      if r["pay_rate"] > 1 else ""),
                source="ecb_violations",
                pilot=("22.2%" if hs == "DEFAULT" else
                       "70.7%" if hs == "IN VIOLATION" else ""))

    dflt, inviol = g.loc["DEFAULT"], g.loc["IN VIOLATION"]
    overall = e["paid"].sum() / e["imposed"].sum()
    ncr = int((e["certification_status"] == "NO COMPLIANCE RECORDED").sum())
    add_row(blk, "pay_rate_overall", "all construction-family", overall,
            n=len(e), window=win, definition=fam_def,
            pilot="42.9%", source="ecb_violations")
    add_row(blk, "n_no_compliance_recorded", "all construction-family", ncr,
            n=len(e), window=win,
            definition=fam_def + "; certification_status = "
                                 "'NO COMPLIANCE RECORDED'",
            pilot="36,753", source="ecb_violations")
    add_row(blk, "share_no_compliance_recorded", "all construction-family",
            ncr / len(e), ci_lo=wilson_ci(ncr, len(e))[0],
            ci_hi=wilson_ci(ncr, len(e))[1], ci_method="Wilson",
            n=len(e), window=win, definition=fam_def,
            source="ecb_violations")

    print(f"  {len(e):,} construction-family violations 2020-24; DEFAULT "
          f"{int(dflt['n']):,} (${dflt['imposed'] / 1e6:.1f}M imposed, pay "
          f"{dflt['pay_rate']:.1%}); adjudicated-appeared IN VIOLATION "
          f"{int(inviol['n']):,} (pay {inviol['pay_rate']:.1%}); overall pay "
          f"{overall:.1%}; NO COMPLIANCE RECORDED {ncr:,}")
    check(blk, "DEFAULT violations", "40,144", f"{int(dflt['n']):,}",
          int(dflt["n"]) == 40144)
    check(blk, "DEFAULT dollars imposed", "$274M",
          f"${dflt['imposed'] / 1e6:.0f}M",
          round(dflt["imposed"] / 1e6) == 274)
    check(blk, "DEFAULT pay rate", "22.2%", f"{dflt['pay_rate'] * 100:.1f}%",
          abs(dflt["pay_rate"] - 0.222) < 0.0005)
    check(blk, "adjudicated-appeared pay rate", "70.7%",
          f"{inviol['pay_rate'] * 100:.1f}%",
          abs(inviol["pay_rate"] - 0.707) < 0.0005)
    check(blk, "overall pay rate", "42.9%", f"{overall * 100:.1f}%",
          abs(overall - 0.429) < 0.0005)
    check(blk, "NO COMPLIANCE RECORDED", "36,753", f"{ncr:,}", ncr == 36753)


# ── block 3: conversion_hazard ───────────────────────────────────────────

def block3_hazard() -> None:
    blk = "conversion_hazard"
    print("\n== Block 3: conversion_hazard (PPML, monitoring-per-permit "
          "machinery) ==")
    jobs = load_jobs()
    ev = pd.read_csv(
        SPINE / "proactive_events.csv.gz",
        usecols=["complaint_number", "received_date", "category_prefix",
                 "family", "agency", "priority", "ecb_severity", "bbl",
                 "bct2020", "nta", "active_job_key"],
        dtype={"bbl": "str", "bct2020": "str", "nta": "str",
               "active_job_key": "str", "category_prefix": "str"},
        parse_dates=["received_date"], low_memory=False)
    xw = tract_nta_crosswalk(ev)

    mixed = set(_PROACTIVE_FAMILY_PREFIXES["mixed_incident"])
    is_c1 = ev["ecb_severity"].eq("CLASS - 1")
    is_inj = (ev["category_prefix"].isin({"91", "1E"})
              | (ev["priority"].eq("A") & ev["category_prefix"].isin(mixed)))
    caller = ev["agency"].eq(0)
    outcome_defs = {
        "n_class1": (is_c1,
                     "events carrying a Class-1 ECB ref (spine first-token "
                     "join), any origin"),
        "n_injury": (is_inj,
                     "injury/incident complaints: cat 91 or 1E, plus "
                     "priority-A in mixed_incident cats "
                     "(30/10/12/14/03/23/67), any origin"),
        "n_class1_caller": (is_c1 & caller,
                            "Class-1-linked events, caller-originated only "
                            "(detection-independent robustness)"),
        "n_injury_caller": (is_inj & caller,
                            "injury/incident complaints, caller-originated "
                            "only"),
    }
    print("Event -> project assignment:")
    for col, (mask, _) in outcome_defs.items():
        jobs[col] = (jobs["job_key"]
                     .map(assign_events(ev[mask], jobs, col))
                     .fillna(0).astype(int))

    # covariates exactly as in proactive_monitoring_per_permit.py
    jobs["nta"] = jobs["bct2020"].map(xw)
    jobs["permit_q"] = jobs["first_permit_date"].dt.to_period("Q").astype(str)
    jobs["log_cost"] = np.log1p(jobs["initial_cost"].clip(lower=0))
    floor = jobs["total_construction_floor_area"]
    jobs["floor_missing"] = floor.isna().astype("int8")
    jobs["log_floor"] = np.log1p(floor.fillna(0).clip(lower=0))
    jobs["log_months"] = np.log(jobs["months"])
    for col, val in JOB_TYPE_DUMMIES.items():
        jobs[col] = (jobs["job_type"] == val).astype("int8")
    mf = jobs[jobs["nta"].notna()].reset_index(drop=True)
    print(f"Model frame: {len(mf):,} projects; outcome events kept "
          + ", ".join(f"{k} {mf[k].sum():,}" for k in outcome_defs))

    # context rates (descriptive)
    conv = mf["conversion"] == 1
    for col in ("n_class1", "n_injury"):
        for grp, sel in (("all projects", mf.index == mf.index),
                         ("conversions (any size)", conv)):
            rate = 100 * mf.loc[sel, col].sum() / mf.loc[sel, "months"].sum()
            add_row(blk, "rate_per_100_project_months", f"{col}|{grp}",
                    rate, n=int(sel.sum()),
                    n_events=int(mf.loc[sel, col].sum()),
                    window="2020-01..2026-05",
                    definition=outcome_defs[col][1], source="this script")

    spec = ("PPML y ~ conv_flag + log_cost + log_floor + floor_missing + "
            "job-type dummies | nta^first_permit_quarter, offset log(active "
            "months), SE clustered on BBL; base-project unit")
    print("PPML models:")
    for outcome, (_, out_def) in outcome_defs.items():
        for flag, flag_label in CONV_DEFS.items():
            t0 = time.time()
            m = pf.fepois(f"{outcome} ~ {flag} + {X_TERMS} | nta^permit_q",
                          data=mf, offset="log_months",
                          vcov={"CRV1": "bbl"})
            td = m.tidy().reset_index()
            r = td[td["Coefficient"] == flag].iloc[0]
            irr, ilo, ihi = (np.exp(r["Estimate"]), np.exp(r["2.5%"]),
                             np.exp(r["97.5%"]))
            add_row(blk, "irr", f"{outcome}|{flag}", irr,
                    ci_lo=ilo, ci_hi=ihi,
                    ci_method="PPML cluster-robust (BBL)",
                    b=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"],
                    n=int(m._N), n_events=int(mf[outcome].sum()),
                    window="2020-01..2026-05",
                    definition=f"{out_def}; {flag_label}; {spec}",
                    source="this script")
            print(f"  {outcome} ~ {flag}: IRR {irr:.2f} [{ilo:.2f}, "
                  f"{ihi:.2f}] (n={int(m._N):,}, {time.time() - t0:.0f}s)")

    # comparison IRRs from the existing monitoring CSV
    est = pd.read_csv(MONITORING_CSV)
    comp = est[(est["term"] == est["conv_def"])
               & est["outcome"].isin(["n_disc", "n_caller_incident",
                                      "n_caller_all"])]
    for _, r in comp.iterrows():
        add_row(blk, "irr_comparison", f"{r['outcome']}|{r['conv_def']}",
                r["irr"], ci_lo=r["irr_lo"], ci_hi=r["irr_hi"],
                ci_method="PPML cluster-robust (BBL)", b=r["b"],
                se=r["se"], p=r["p"], n=r["n_obs"], n_events=r["n_events"],
                window=r["window"],
                definition=f"{r['outcome_label']}; identical spec",
                source="proactive_monitoring_estimates.csv (existing)")
    print(f"  copied {len(comp)} comparison IRRs "
          f"(n_disc / n_caller_incident / n_caller_all) from "
          f"{MONITORING_CSV.name}")
    check(blk, "comparison IRRs copied from existing CSV",
          "9 rows (3 outcomes x 3 conv defs)", f"{len(comp)} rows",
          len(comp) == 9)

    # riskier-vs-more-visible verdict table
    hz = pd.DataFrame(rows)
    hz = hz[(hz["block"] == blk) & hz["metric"].isin(["irr",
                                                      "irr_comparison"])]
    hz[["outcome", "flag"]] = hz["group"].str.split("|", expand=True)
    show = ["n_class1", "n_injury", "n_class1_caller", "n_injury_caller",
            "n_disc", "n_caller_incident", "n_caller_all"]
    print("\n  Riskier or just more visible? (IRR [95% CI] by conversion "
          "flag; hazard outcomes this script, comparison rows existing CSV)")
    for flag in CONV_DEFS:
        print(f"    {flag}:")
        for o in show:
            r = hz[(hz["outcome"] == o) & (hz["flag"] == flag)].iloc[0]
            tag = "  (comparison)" if r["metric"] == "irr_comparison" else ""
            print(f"      {o:<18} IRR {r['value']:.2f} "
                  f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]{tag}")


# ── block 4a: shell_respondents ──────────────────────────────────────────

def block4_shells(con) -> None:
    blk = "shell_respondents"
    print("\n== Block 4a: shell_respondents ==")
    types = ", ".join(f"'{t}'" for t in CONSTRUCTION_ECB_TYPES)
    e = pd.read_sql_query(
        f"SELECT respondent_name, bin, hearing_status, penality_imposed, "
        f"amount_paid FROM ecb_violations "
        f"WHERE violation_type IN ({types}) "
        f"AND substr(issue_date, 1, 4) >= '2020'", con)
    e["imposed"] = pd.to_numeric(e["penality_imposed"], errors="coerce")
    e["paid"] = pd.to_numeric(e["amount_paid"], errors="coerce")
    e = e[e["bin"].notna()
          & ~e["bin"].str.fullmatch(BIN_PLACEHOLDER)].copy()

    def run_grouping(nm, label, pilot_n="", pilot_share=""):
        sub = e.assign(nm=nm)
        sub = sub[sub["nm"].str.contains("LLC", na=False)]
        n_bins = sub.groupby("nm")["bin"].nunique()
        n_resp = len(n_bins)
        single_names = n_bins[n_bins == 1].index
        share = (n_bins == 1).mean()
        slo, shi = wilson_ci((n_bins == 1).sum(), n_resp)
        base_def = (f"{label}; construction-family ECB (types "
                    f"{', '.join(CONSTRUCTION_ECB_TYPES)}) issued 2020+, "
                    "valid BIN; respondent grouped on name; single-building "
                    "= exactly 1 distinct BIN")
        add_row(blk, "n_llc_respondents", label, n_resp,
                n_events=len(sub), window="issued 2020+",
                definition=base_def, pilot=pilot_n, source="ecb_violations")
        add_row(blk, "share_single_building", label, share,
                ci_lo=slo, ci_hi=shi, ci_method="Wilson", n=n_resp,
                window="issued 2020+", definition=base_def,
                pilot=pilot_share, source="ecb_violations")
        stats = {}
        for tag, sel in (("single_site", sub["nm"].isin(single_names)),
                         ("multi_site", ~sub["nm"].isin(single_names))):
            s = sub[sel]
            dflt = (s["hearing_status"] == "DEFAULT")
            dlo, dhi = cluster_prop_ci(dflt, s["nm"])
            pay = s["paid"].sum() / s["imposed"].sum()
            pilots = {"single_site": ("20.9%", "32.1%"),
                      "multi_site": ("13.3%", "44.9%")}
            p_d, p_p = pilots[tag] if label.startswith("headline") else ("", "")
            add_row(blk, "n_violations", f"{tag}|{label}", len(s),
                    window="issued 2020+", definition=base_def,
                    source="ecb_violations")
            add_row(blk, "default_share", f"{tag}|{label}", dflt.mean(),
                    ci_lo=dlo, ci_hi=dhi,
                    ci_method="cluster-robust (respondent)", n=len(s),
                    n_events=int(dflt.sum()), window="issued 2020+",
                    definition=base_def + "; share of violations with "
                                          "hearing_status DEFAULT",
                    pilot=p_d, source="ecb_violations")
            add_row(blk, "pay_rate", f"{tag}|{label}", pay, n=len(s),
                    window="issued 2020+",
                    definition=base_def + "; sum paid / sum imposed",
                    pilot=p_p, source="ecb_violations")
            stats[tag] = (len(s), dflt.mean(), pay)
        return n_resp, share, stats

    nm_raw = e["respondent_name"].fillna("").str.upper().str.strip()
    n_resp, share, st = run_grouping(
        nm_raw, "headline: raw uppercased name",
        pilot_n="23,507", pilot_share="~93%")
    print(f"  headline: {n_resp:,} LLC respondents, single-building "
          f"{share:.1%}; single default {st['single_site'][1]:.1%} / pay "
          f"{st['single_site'][2]:.1%}; multi default "
          f"{st['multi_site'][1]:.1%} / pay {st['multi_site'][2]:.1%}")

    nm_norm = (e["respondent_name"].fillna("").str.upper()
               .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
               .str.replace(r"\s+", " ", regex=True).str.strip())
    n2, share2, st2 = run_grouping(
        nm_norm, "alt: punctuation/whitespace-collapsed name")
    print(f"  alt normalization: {n2:,} respondents, single-building "
          f"{share2:.1%}; multi default {st2['multi_site'][1]:.1%} / pay "
          f"{st2['multi_site'][2]:.1%}")

    check(blk, "LLC respondents since 2020", "23,507", f"{n_resp:,}",
          n_resp == 23507,
          f"name-normalization sensitive; alt grouping gives {n2:,}")
    check(blk, "single-building share", "~93%", f"{share * 100:.1f}%",
          round(share * 100) == 93)
    check(blk, "single-site default / pay", "20.9% / 32.1%",
          f"{st['single_site'][1] * 100:.1f}% / {st['single_site'][2] * 100:.1f}%",
          abs(st["single_site"][1] - 0.209) < 0.0005
          and abs(st["single_site"][2] - 0.321) < 0.0005)
    check(blk, "multi-site default / pay", "13.3% / 44.9%",
          f"{st['multi_site'][1] * 100:.1f}% / {st['multi_site'][2] * 100:.1f}%",
          abs(st["multi_site"][1] - 0.133) < 0.0005
          and abs(st["multi_site"][2] - 0.449) < 0.0005,
          f"pilot's exact name grouping not recoverable; alt grouping "
          f"gives {st2['multi_site'][1] * 100:.1f}% / "
          f"{st2['multi_site'][2] * 100:.1f}%")


# ── block 4b: swo_escalation ─────────────────────────────────────────────

def block4_escalation(con) -> None:
    blk = "swo_escalation"
    print("\n== Block 4b: swo_escalation ==")
    d = pd.read_sql_query(
        "SELECT complaint_number, category_code, ref_311, received_date, "
        "disposition FROM bis_scrape WHERE disposition LIKE '%- A3 -%'", con)
    ext = d["disposition"].str.extract(DISP_RE)
    d = d[ext[1] == "A3"].copy()
    d["ry"] = pd.to_datetime(d["received_date"], errors="coerce").dt.year
    d["cat05"] = d["category_code"].astype(str).str.slice(0, 2) == "05"
    d["caller"] = d["ref_311"].fillna("").str.strip().ne("")
    t = d[d["ry"].between(2023, 2025)]
    base_def = ("complaints with final disposition code A3 (FULL STOP WORK "
                "ORDER SERVED), keyed by complaint RECEIVED year; cat-05 = "
                "category prefix 05, NO BUILDING PERMIT "
                "(construction/demolition); caller = ref_311 non-empty; "
                "dispositions observed through the 2026-06 scrape")
    got = {}
    for y, g in t.groupby("ry"):
        n_all, n_05 = len(g), int(g["cat05"].sum())
        cs = g["caller"].mean()
        clo, chi = wilson_ci(int(g["caller"].sum()), n_all)
        c05 = g.loc[g["cat05"], "caller"].mean()
        c05lo, c05hi = wilson_ci(int(g.loc[g["cat05"], "caller"].sum()), n_05)
        got[y] = (n_all, n_05, cs)
        add_row(blk, "n_a3_full_swo_served", str(y), n_all,
                window=str(y), definition=base_def,
                pilot={2023: "206", 2025: "853"}.get(y, ""),
                source="bis_scrape")
        add_row(blk, "n_a3_cat05_component", str(y), n_05, n=n_all,
                window=str(y), definition=base_def,
                pilot={2023: "34", 2025: "565"}.get(y, ""),
                source="bis_scrape")
        add_row(blk, "caller_share_a3", str(y), cs, ci_lo=clo, ci_hi=chi,
                ci_method="Wilson", n=n_all, window=str(y),
                definition=base_def,
                pilot="~70%" if y == 2025 else "", source="bis_scrape")
        add_row(blk, "caller_share_a3_cat05", str(y), c05, ci_lo=c05lo,
                ci_hi=c05hi, ci_method="Wilson", n=n_05, window=str(y),
                definition=base_def, source="bis_scrape")
        print(f"  {y}: A3 {n_all:,} (cat-05 {n_05:,}); caller share "
              f"{cs:.1%} (cat-05 {c05:.1%})")
    check(blk, "A3 full SWOs served 2023 -> 2025", "206 -> 853",
          f"{got[2023][0]} -> {got[2025][0]}",
          got[2023][0] == 206 and got[2025][0] == 853)
    check(blk, "cat-05 component 2023 -> 2025", "34 -> 565",
          f"{got[2023][1]} -> {got[2025][1]}",
          got[2023][1] == 34 and got[2025][1] == 565)
    check(blk, "caller-origin share (2025 A3s)", "~70%",
          f"{got[2025][2] * 100:.1f}%", round(got[2025][2], 1) == 0.7)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    RM.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    block1_swo(con)
    block2_default(con)
    block4_shells(con)
    block4_escalation(con)
    con.close()
    block3_hazard()

    order = {"swo_revolving_door": 1, "default_economy": 2,
             "conversion_hazard": 3, "shell_respondents": 4,
             "swo_escalation": 5}
    df = pd.DataFrame(rows).sort_values(
        "block", key=lambda s: s.map(order), kind="stable")
    df.to_csv(OUT, index=False)
    print(f"\nCSV -> {OUT} ({len(df)} rows, "
          f"{df['block'].nunique()} blocks)")

    print("\n== Verification vs hunt pilot numbers "
          "(proactive_enforcement_plan.md) ==")
    n_flag = 0
    cur = None
    for blk, label, pilot, got, ok, note in checks:
        if blk != cur:
            print(f"  [{blk}]")
            cur = blk
        verdict = "PASS" if ok else "FLAG"
        n_flag += (not ok)
        line = f"    {verdict}  {label}: pilot {pilot} | regenerated {got}"
        if note and not ok:
            line += f"  ({note})"
        print(line)
    print(f"\n  {len(checks) - n_flag}/{len(checks)} checks match at pilot "
          f"precision; {n_flag} flagged.")
    print(f"Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
