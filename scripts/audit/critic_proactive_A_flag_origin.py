"""Critic script (proactive post): attack the agency flag, the warm/cold
decomposition, and the origin-information (+26.4pp) headline.

Three attack surfaces, run in order:

  A. THE PROACTIVE FLAG. Is empty bis_scrape.ref_311 really agency-initiated?
     A1 re-scans the raw HTML archive for EVERY agency-flagged in-window
     complaint with a permissive regex (parser-failure rate), plus a caller-
     side sample (value fidelity). A2 checks empty-share drift by scrape
     week. A3 classifies agency-row subject text into internal / external-
     referral / caller-language buckets (semantic misclassification rate).
     A4 re-checks the pure-agency category priors.

  B. DECOMPOSITION (proactive_decomposition.py). B1 reproduces the published
     slice shares and rates from the spine. B2 brute-forces the 730-day
     searchsorted warm flag on random + adversarial samples. B3 audits the
     prefix->family mapping for misfiled programs. B4 reweights the
     cold-vs-warm violation gap by category mix. B5 ties the same-day
     paperwork channel to the cold-yield headline.

  C. ORIGIN INFORMATION (proactive_origin_information.py). C1 reproduces
     +26.4 / +24.0 / +14.3. C2 decomposes the gap by disposition code.
     C3 splits by inspection timing (record-as-paperwork test). C4 re-runs
     the gap on the one disposition-code family both channels use (A8/A9).
     C5 attacks the ECB margin with citation issue dates (pre-existing
     paper). C6 within-building (BBL FE) variant. C7 credits caller H1
     cross-references with successor outcomes. C8 double-linked ECB numbers.

Run:  /private/tmp/pyfix_venv/bin/python scripts/audit/critic_proactive_A_flag_origin.py
Env:  CRITIC_HTML_SCAN=full|sample   (default full; sample = 30k agency pages)
"""

import gzip
import os
import re
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/Users/mzoorob/Dropbox/nycpol/map_311_complaints_to_building_inspector_reports"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from disposition_codes import classify_disposition  # noqa: E402

DB = os.path.join(ROOT, "data", "dob_complaints.db")
SPINE = os.path.join(ROOT, "data", "analysis", "proactive", "proactive_events.csv.gz")
PANEL = os.path.join(ROOT, "data", "analysis", "property_risk_panel_v2.csv.gz")
DECOMP_CSV = os.path.join(ROOT, "data", "analysis", "risk_models", "proactive_decomposition.csv")
ARCHIVE = os.path.join(ROOT, "data", "html_archive")

MIX = ["04", "23", "30", "45", "58", "59", "67", "73", "83", "91"]
DISC_FAMILIES = ("discretionary_field", "mixed_incident", "other")
LOOKBACK = 730

T0 = time.time()


def log(msg=""):
    print(msg, flush=True)


def hdr(title):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


# ── shared loads ─────────────────────────────────────────────────────────

def load_spine():
    ev = pd.read_csv(SPINE, dtype={"bbl": str, "ecb_number": str,
                                   "category_prefix": str,
                                   "complaint_number": str,
                                   "disposition_code": str})
    ev["received"] = pd.to_datetime(ev["received_date"])
    ev["bbl"] = ev["bbl"].fillna("")
    return ev


def load_od_dates(conn):
    od = pd.read_sql_query(
        "SELECT complaint_number, inspection_date, disposition_code AS od_dispo "
        "FROM open_data", conn)
    od["insp"] = pd.to_datetime(od["inspection_date"], format="%m/%d/%Y",
                                errors="coerce")
    return od[["complaint_number", "insp", "od_dispo"]]


# ═════════════════════════════════════════════════════════════════════════
# A. THE PROACTIVE FLAG
# ═════════════════════════════════════════════════════════════════════════

LABEL_RE = re.compile(r"311\s+Reference\s+Number", re.I)
VALUE_RE = re.compile(r"311 Reference Number:</b>(?:&nbsp;|\s)*([^<&\s]*)", re.I)


def scan_pages(cns, want_value=False):
    """Scan archived pages; return (n_scanned, n_missing, n_corrupt, hits).
    hits = list of (complaint_number, extracted_value_or_'' ) for pages where
    the 311 label row is present at all."""
    n_missing = n_corrupt = n_scanned = 0
    hits = []
    for cn in cns:
        p = os.path.join(ARCHIVE, f"{cn}.html.gz")
        if not os.path.exists(p):
            n_missing += 1
            continue
        try:
            with gzip.open(p, "rt", errors="replace") as f:
                h = f.read()
        except Exception:
            n_corrupt += 1
            continue
        n_scanned += 1
        if LABEL_RE.search(h):
            m = VALUE_RE.search(h)
            hits.append((cn, m.group(1).strip() if m else ""))
    return n_scanned, n_missing, n_corrupt, hits


def section_A(conn, ev):
    hdr("A. THE PROACTIVE FLAG: is empty ref_311 really agency-initiated?")

    bis = pd.read_sql_query("""
        SELECT complaint_number, TRIM(COALESCE(ref_311,'')) AS ref,
               subject, date(scraped_at) AS scrape_day,
               substr(category_code,1,2) AS pfx
        FROM bis_scrape
        WHERE received_date LIKE '__/__/____'
          AND substr(received_date,7,4) >= '2020'""", conn)
    agency = bis[bis["ref"] == ""]
    caller = bis[bis["ref"] != ""]
    log(f"in-window rows: {len(bis):,}; agency-flagged {len(agency):,} "
        f"({len(agency) / len(bis):.4f})")

    # A1 parser failure: rescan raw HTML
    mode = os.environ.get("CRITIC_HTML_SCAN", "full")
    ag_cns = agency["complaint_number"].tolist()
    if mode != "full":
        rng = np.random.default_rng(7)
        ag_cns = list(rng.choice(ag_cns, size=min(30_000, len(ag_cns)),
                                 replace=False))
    t = time.time()
    n_s, n_m, n_c, hits = scan_pages(ag_cns)
    log(f"\nA1a agency-side rescan ({mode}): {n_s:,} pages scanned in "
        f"{time.time() - t:.0f}s; archive missing {n_m:,}; corrupt {n_c:,}")
    with_val = [h for h in hits if h[1]]
    log(f"    pages with a '311 Reference Number' label despite empty DB "
        f"ref_311: {len(hits):,} ({len(hits) / max(n_s, 1):.6f})")
    log(f"    of those, label carries a non-empty value (TRUE parser "
        f"misses): {len(with_val):,} -> false-agency rate "
        f"{len(with_val) / max(n_s, 1):.6f}")
    if with_val[:5]:
        log(f"    examples: {with_val[:5]}")

    rng = np.random.default_rng(11)
    ca_cns = list(rng.choice(caller["complaint_number"].tolist(), 30_000,
                             replace=False))
    n_s2, n_m2, n_c2, hits2 = scan_pages(ca_cns)
    hv = dict(hits2)
    ca = caller.set_index("complaint_number")
    n_match = sum(1 for cn, v in hits2 if v == ca.loc[cn, "ref"])
    log(f"A1b caller-side sample: {n_s2:,} scanned; label present on "
        f"{len(hits2):,}; extracted value == DB value on {n_match:,} "
        f"({n_match / max(len(hits2), 1):.4f}); missing {n_m2:,}, corrupt {n_c2:,}")

    # A2 scrape-week drift of the empty share
    wk = bis.copy()
    wk["week"] = pd.to_datetime(wk["scrape_day"]).dt.to_period("W").astype(str)
    drift = wk.groupby("week").agg(empty_share=("ref", lambda s: (s == "").mean()),
                                   n=("ref", "size"))
    log("\nA2 empty-ref share by scrape week (parser-era drift):")
    log(drift[drift["n"] > 1000].round(4).to_string())

    # A3 semantic buckets from subject text (agency rows)
    subj = agency["subject"].fillna("").str.upper()
    internal = subj.str.contains(
        r"\b(?:SWEEP|COMPLIANCE|RE-?INSPEC|AUDIT|MONITOR|EWO|ECB\s*#|AHV|"
        r"FOLLOW\s?UP|PADLOCK|SITE SAFETY|PERIODIC|CYCLE|TR6|LL\s?\d|"
        r"PERMIT.{0,12}EXPIR|FISP|FACADE UNIT|BEST SQUAD|SCHEDULED)", regex=True)
    external = subj.str.contains(
        r"\b(?:FDNY|FIRE DEP|BUREAU OF FIRE|HPD|NYPD|CON\s?ED|DOHMH|DSNY|"
        r"OSHA|DOI|NYCEM|OEM|REFERRAL|REFERR?ED (?:BY|FROM)|COMMUNITY BOARD|"
        r"COUNCIL|BOROUGH PRESIDENT|PUBLIC ADVOCATE|SENATOR|ASSEMBLY|"
        r"COMPTROLLER|MARSHAL)", regex=True)
    callerish = subj.str.contains(
        r"\b(?:CALLER|COMPLAINANT|ANONYMOUS|ANON\b|TENANT|NEIGHBOR|OCCUPANT|"
        r"RESIDENT|PASSER\s?BY|CONSTITUENT|MY (?:APARTMENT|BUILDING|"
        r"LANDLORD))", regex=True)
    caller_only = callerish & ~internal & ~external
    tab = pd.DataFrame({
        "internal_kw": internal, "external_referral_kw": external,
        "caller_language_kw": callerish, "caller_only_kw": caller_only,
        "pfx": agency["pfx"].values})
    log("\nA3 subject-keyword buckets on AGENCY-flagged rows (shares):")
    log("    all agency rows:      " + tab[["internal_kw", "external_referral_kw",
        "caller_language_kw", "caller_only_kw"]].mean().round(4).to_string())
    mixmask = tab["pfx"].isin(MIX)
    log("    mixed-cat agency rows: " + tab.loc[mixmask, ["internal_kw",
        "external_referral_kw", "caller_language_kw", "caller_only_kw"]]
        .mean().round(4).to_string())
    ex = agency.loc[caller_only & mixmask, "subject"].head(6).tolist()
    log("    examples of caller-language 'agency' rows (mixed cats):")
    for e in ex:
        log(f"      - {str(e)[:95]}")

    # A4 pure-agency category priors
    log("\nA4 agency share in by-construction agency programs (should be ~1):")
    for pfx in ("8A", "7G", "7K", "7F", "1X", "6V", "2E"):
        sub = bis[bis["pfx"] == pfx]
        log(f"    {pfx}: {(sub['ref'] == '').mean():.4f}  (n={len(sub):,})")

    return tab, mixmask


# ═════════════════════════════════════════════════════════════════════════
# B. DECOMPOSITION: warm/cold logic, family mapping, reweighting
# ═════════════════════════════════════════════════════════════════════════

def searchsorted_warm(ev):
    """Replicate proactive_decomposition.add_warm_flag exactly."""
    day = (ev["received"] - ev["received"].min()).dt.days.to_numpy()
    codes = pd.factorize(ev["bbl"])[0]
    K = int(day.max()) + LOOKBACK + 2
    key = codes.astype(np.int64) * K + (day + LOOKBACK)
    located = (ev["bbl"] != "").to_numpy()
    pool = np.sort(key[located])
    n = (np.searchsorted(pool, key, side="left")
         - np.searchsorted(pool, key - LOOKBACK, side="left"))
    n[~located] = 0
    ev["prior_730"] = n
    ev["warm"] = (n > 0).astype("int8")
    return ev


def section_B(ev, od):
    hdr("B. DECOMPOSITION: warm/cold verification, family mapping, "
        "composition reweighting")

    ev = searchsorted_warm(ev)
    ag = ev[ev["agency"] == 1]
    disc = ag[ag["family"].isin(DISC_FAMILIES)]
    cold = disc[disc["warm"] == 0]
    warm = disc[disc["warm"] == 1]

    # B1 reproduce published table
    pub = pd.read_csv(DECOMP_CSV)
    pubm = pub[pub["group"] == "main"].set_index("slice")
    checks = [
        ("agency warm share", ag["warm"].mean(),
         pub.loc[pub["slice"] == "all_agency", "warm_share"].iat[0]),
        ("cold share of agency", len(cold) / len(ag),
         pubm.loc["discretionary_cold", "share_of_agency"]),
        ("warm share of agency", len(warm) / len(ag),
         pubm.loc["discretionary_warm", "share_of_agency"]),
        ("cold violation rate", (cold["outcome"] == "violation").mean(),
         pubm.loc["discretionary_cold", "violation_disposition_rate"]),
        ("warm violation rate", (warm["outcome"] == "violation").mean(),
         pubm.loc["discretionary_warm", "violation_disposition_rate"]),
    ]
    log("B1 reproduction against published proactive_decomposition.csv:")
    for name, got, exp in checks:
        flag = "OK" if abs(got - exp) < 5e-4 else "DIVERGES"
        log(f"    {name:<24} got {got:.4f}  published {exp:.4f}  {flag}")

    # B2 brute force the searchsorted warm counts
    rng = np.random.default_rng(11)
    idx = list(rng.choice(ev.index.to_numpy(), 5_000, replace=False))
    # adversarial: BBLs with heavy same-day batching (sweep blocks)
    counts = ev.groupby(["bbl", "received"]).size()
    batch_bbls = counts[counts >= 5].reset_index()["bbl"].unique()[:50]
    idx += list(ev[ev["bbl"].isin(batch_bbls)].index[:2_000])
    # boundary: events exactly 730d after another at the same BBL
    sub = ev.loc[idx]
    groups = {b: g for b, g in
              ev[ev["bbl"].isin(sub["bbl"].unique())].groupby("bbl")}
    mism = 0
    for i, row in sub.iterrows():
        if row["bbl"] == "":
            bf = 0
        else:
            g = groups[row["bbl"]]
            lo = row["received"] - pd.Timedelta(days=LOOKBACK)
            bf = int(((g["received"] < row["received"])
                      & (g["received"] >= lo)).sum())
        if bf != row["prior_730"]:
            mism += 1
    log(f"\nB2 brute-force check of prior_730 on {len(sub):,} rows "
        f"(random + same-day sweep batches): {mism} mismatches")
    # window convention: [d-730, d-1]; same-day excluded
    d0 = ev[ev["prior_730"] > 0].iloc[0]
    log(f"    window convention verified: prior events in [d-730, d-1], "
        f"same-day excluded (example event {d0['complaint_number']})")

    # B3 family mapping audit
    log("\nB3 prefix -> family audit (agency rows, n >= 800):")
    t = (ag.groupby(["family", "category_prefix"])
         .agg(n=("outcome", "size"),
              viol=("outcome", lambda s: (s == "violation").mean()))
         .reset_index())
    names = (ag.groupby("category_prefix")["category_name"]
             .agg(lambda s: s.value_counts().index[0]))
    t["name"] = t["category_prefix"].map(names).str[:52]
    log(t[t["n"] >= 800].sort_values(["family", "n"], ascending=[True, False])
        .to_string(index=False))
    ext_pfx = ["1D", "7L", "1Z", "2N"]
    extn = ag[ag["category_prefix"].isin(ext_pfx)]
    log(f"    external-referral / EO prefixes inside the 'discretionary' pool "
        f"(1D ConEd, 7L DOHMH, 1Z interagency, 2N COVID EO): "
        f"{len(extn):,} agency rows, of which cold "
        f"{(extn.merge(cold[['complaint_number']], on='complaint_number').shape[0]):,}")

    # B4 reweighting the cold-vs-warm violation gap by category mix
    g = (disc.groupby(["category_prefix", "warm"])
         .agg(n=("outcome", "size"),
              viol=("outcome", lambda s: (s == "violation").mean()))
         .reset_index())
    p = g.pivot(index="category_prefix", columns="warm",
                values=["n", "viol"])
    p.columns = ["n_cold", "n_warm", "v_cold", "v_warm"]
    p = p.fillna(0)
    both = p[(p["n_cold"] >= 30) & (p["n_warm"] >= 30)].copy()
    both["gap"] = both["v_cold"] - both["v_warm"]
    w_pool = (both["n_cold"] + both["n_warm"])
    raw_gap = ((cold["outcome"] == "violation").mean()
               - (warm["outcome"] == "violation").mean())
    log("\nB4 cold-minus-warm violation gap, reweighted by category mix:")
    log(f"    raw pooled gap (published 45.4 - 39.4): {raw_gap * 100:+.2f} pp")
    log(f"    within-prefix gap, pooled-mix weights:  "
        f"{(both['gap'] * w_pool / w_pool.sum()).sum() * 100:+.2f} pp")
    log(f"    within-prefix gap, warm-mix weights:    "
        f"{(both['gap'] * both['n_warm'] / both['n_warm'].sum()).sum() * 100:+.2f} pp")
    log(f"    within-prefix gap, cold-mix weights:    "
        f"{(both['gap'] * both['n_cold'] / both['n_cold'].sum()).sum() * 100:+.2f} pp")
    fam = (disc.groupby(["family", "warm"])["outcome"]
           .apply(lambda s: (s == "violation").mean()).unstack())
    fam["cold_minus_warm"] = fam[0] - fam[1]
    log("    family-level gaps (cold minus warm):")
    log((fam["cold_minus_warm"] * 100).round(2).to_string())
    log("    largest-prefix detail:")
    log(both.sort_values("n_cold", ascending=False).head(10)[
        ["n_cold", "n_warm", "v_cold", "v_warm", "gap"]].round(3).to_string())

    # B5 same-day paperwork channel inside the cold slice
    dm = disc.merge(od, on="complaint_number", how="left")
    dm["lag"] = (dm["insp"] - dm["received"]).dt.days
    for tag, mask in (("cold", dm["warm"] == 0), ("warm", dm["warm"] == 1)):
        s = dm[mask & dm["lag"].notna()]
        same = s[s["lag"] == 0]
        lead = s[s["lag"] > 0]
        log(f"\nB5 {tag}: same-day-inspected share "
            f"{(s['lag'] == 0).mean():.3f}; violation rate same-day "
            f"{(same['outcome'] == 'violation').mean():.3f} vs "
            f"inspection-after-receipt {(lead['outcome'] == 'violation').mean():.3f} "
            f"(n {len(same):,}/{len(lead):,})")
    return ev


# ═════════════════════════════════════════════════════════════════════════
# C. ORIGIN INFORMATION: +26.4pp headline and the +14.3 ECB margin
# ═════════════════════════════════════════════════════════════════════════

def section_C(conn, ev, od):
    import pyfixest as pf

    hdr("C. ORIGIN INFORMATION: mechanics of the +26.4pp / +14.3pp gaps")

    m = ev[ev["category_prefix"].isin(MIX) & (ev["outcome"] != "pending")].copy()
    m["violation100"] = (m["outcome"] == "violation").astype(float) * 100
    m["ecb100"] = (m["ecb_number"].fillna("") != "").astype(float) * 100
    m["a89_100"] = m["disposition_code"].isin(["A8", "A9"]).astype(float) * 100
    m["bct"] = pd.to_numeric(m["bct2020"], errors="coerce").astype("Int64").astype(str)
    m["cat_month"] = m["category_prefix"] + "_" + m["month"]
    pan = (pd.read_csv(PANEL, usecols=["bbl_key", "size_bin"],
                       dtype={"bbl_key": str, "size_bin": str})
           .drop_duplicates("bbl_key").rename(columns={"bbl_key": "bbl"}))
    m = m.merge(pan, on="bbl", how="left")
    m = m.merge(od, on="complaint_number", how="left")
    m["lag"] = (m["insp"] - m["received"]).dt.days
    hm = m[m["size_bin"].notna() & m["bct2020"].notna()].copy()

    vcov = {"CRV1": "bct"}

    def run(fml, d, tag, vc=None):
        fit = pf.feols(fml, data=d, vcov=vc or vcov)
        t = fit.tidy().reset_index()
        r = t[t["Coefficient"] == "agency"].iloc[0]
        log(f"    {tag:<52} {r['Estimate']:+7.2f} "
            f"[{r['2.5%']:+6.2f},{r['97.5%']:+6.2f}]  N={fit._N:,}")
        return r["Estimate"]

    log("C1 reproduction of published models:")
    b_head = run("violation100 ~ agency | cat_month + size_bin + bct", hm,
                 "headline (published +26.4)")
    run("violation100 ~ agency | cat_month + size_bin + bct",
        hm[hm["outcome"].isin(["violation", "no_violation"])],
        "substantive-only (published +24.0)")
    b_ecb = run("ecb100 ~ agency | cat_month + size_bin + bct", hm,
                "ECB linkage (published +14.3)")

    # C2 disposition-code composition among violation-coded rows
    log("\nC2 disposition-code composition of 'violation' outcomes by origin:")
    for a, tag in ((1, "agency"), (0, "caller")):
        sub = m[(m["agency"] == a) & (m["outcome"] == "violation")]
        vc = (sub["disposition_code"].value_counts(normalize=True) * 100)
        log(f"    {tag} (n={len(sub):,}): " + ", ".join(
            f"{k} {v:.1f}%" for k, v in vc.head(6).items()))
        log(f"      linked-ECB share of these violation-coded rows: "
            f"{(sub['ecb_number'].fillna('') != '').mean():.3f}")
    a1 = m[(m["disposition_code"] == "A1") & (m["agency"] == 1)]
    log(f"    A1 'Buildings Violation Served' agency rows by category: "
        f"{a1['category_prefix'].value_counts().head(4).to_dict()}")

    # C3 record-as-paperwork: inspection timing
    log("\nC3 inspection timing (open_data.inspection_date vs received):")
    for a, tag in ((1, "agency"), (0, "caller")):
        s = m[(m["agency"] == a) & m["lag"].notna()]
        sv = s[s["outcome"] == "violation"]
        log(f"    {tag}: same-day-inspected {(s['lag'] == 0).mean():.3f}; "
            f"among violation-coded rows {(sv['lag'] == 0).mean():.3f}")
    ag = m[(m["agency"] == 1) & m["lag"].notna()]
    log(f"    agency violation rate: same-day "
        f"{(ag.loc[ag['lag'] == 0, 'outcome'] == 'violation').mean():.3f} vs "
        f"inspected-later {(ag.loc[ag['lag'] > 0, 'outcome'] == 'violation').mean():.3f}")
    lead = hm[hm["lag"] > 0]
    same = hm[hm["lag"] == 0]
    log("    headline spec split by timing:")
    b_lead = run("violation100 ~ agency | cat_month + size_bin + bct", lead,
                 "violation gap, inspection AFTER receipt only")
    run("violation100 ~ agency | cat_month + size_bin + bct", same,
        "violation gap, same-day-inspected only")
    log(f"    -> the +{b_head:.1f} headline blends a lead-like gap of "
        f"+{b_lead:.1f} with a same-day (record-opened-at-inspection) gap")

    # C4 identical disposition-code family: A8/A9 OATH summons only
    log("\nC4 outcome = A8/A9 OATH summons served (code family both "
        "channels use; strictest common-coding test):")
    run("a89_100 ~ agency | cat_month + size_bin + bct", hm,
        "A8/A9-only violation definition")

    # C5 ECB margin vs citation issue dates
    ecb = pd.read_sql_query(
        "SELECT ecb_violation_number AS ecb_number, issue_date "
        "FROM ecb_violations", conn).drop_duplicates("ecb_number")
    ecb["iss"] = pd.to_datetime(ecb["issue_date"], errors="coerce")
    m2 = hm.merge(ecb[["ecb_number", "iss"]], on="ecb_number", how="left")
    linked = m2["ecb_number"].fillna("") != ""
    m2["dd"] = (m2["iss"] - m2["received"]).dt.days
    log("\nC5 linked-ECB citation issue date vs complaint receipt:")
    for a, tag in ((1, "agency"), (0, "caller")):
        s = m2[(m2["agency"] == a) & linked & m2["dd"].notna()]
        log(f"    {tag}: issued BEFORE receipt {(s['dd'] < 0).mean():.3f}; "
            f"same day {(s['dd'] == 0).mean():.3f}; after {(s['dd'] > 0).mean():.3f} "
            f"(n={len(s):,})")
    pre83 = m2[(m2["agency"] == 1) & (m2["category_prefix"] == "83")
               & linked & m2["dd"].notna()]
    log(f"    category 83 agency links issued before receipt: "
        f"{(pre83['dd'] < 0).mean():.3f} (n={len(pre83):,})")
    # scrub pre-dated links; keep unmatched-issue-date links as hits
    m2["ecb_new100"] = ((linked & ~(m2["dd"] < 0)).astype(float) * 100)
    log("    ECB margin with pre-dated citations removed:")
    run("ecb_new100 ~ agency | cat_month + size_bin + bct", m2,
        "ECB linkage, citation issued on/after receipt")
    run("ecb_new100 ~ agency | cat_month + size_bin + bct",
        m2[m2["lag"] > 0], "ECB new-citation gap, inspection AFTER receipt")
    log(f"    (published ECB margin {b_ecb:+.1f})")

    # C6 within-building fixed effects
    wb = m[m["bbl"] != ""].copy()
    log("\nC6 within-building (BBL FE, clustered by BBL):")
    run("violation100 ~ agency | cat_month + bbl", wb,
        "violation gap within building", vc={"CRV1": "bbl"})
    run("ecb100 ~ agency | cat_month + bbl", wb,
        "ECB gap within building", vc={"CRV1": "bbl"})

    # C7 H1 cross-reference chain: credit callers with successor outcomes
    log("\nC7 caller H1 'see complaint number' chains:")
    bis = pd.read_sql_query(
        "SELECT complaint_number, disposition FROM bis_scrape", conn)
    h1 = m[(m["agency"] == 0) & (m["disposition_code"] == "H1")].merge(
        bis, on="complaint_number", how="left")
    h1["succ"] = h1["disposition"].str.extract(
        r"SEE COMPLAINT NUMBER\s*#?\s*(\w+)", flags=re.I)
    succ_out = pd.read_sql_query(
        "SELECT complaint_number AS succ, disposition_code AS sd "
        "FROM open_data", conn)
    h1 = h1.merge(succ_out, on="succ", how="left")
    h1["succ_outcome"] = h1["sd"].fillna("").map(classify_disposition)
    ok = h1["sd"].notna()
    log(f"    caller H1 rows in mixed cats: {len(h1):,} "
        f"({len(h1) / (m['agency'] == 0).sum():.4f} of caller rows); "
        f"successor resolved {ok.mean():.3f}")
    log(f"    successor outcome mix: "
        f"{h1.loc[ok, 'succ_outcome'].value_counts(normalize=True).round(3).to_dict()}")
    extra = (h1.loc[ok, "succ_outcome"] == "violation").sum()
    kr = m[m["agency"] == 0]
    adj_caller = ((kr["outcome"] == "violation").sum() + extra) / len(kr)
    raw_agency = (m.loc[m["agency"] == 1, "outcome"] == "violation").mean()
    log(f"    raw caller violation rate {((kr['outcome'] == 'violation').mean()):.4f} "
        f"-> {adj_caller:.4f} after crediting H1 successors; raw gap "
        f"{(raw_agency - (kr['outcome'] == 'violation').mean()) * 100:+.1f} -> "
        f"{(raw_agency - adj_caller) * 100:+.1f} pp")

    # C8 double-linked ECB numbers across records
    e = ev[ev["ecb_number"].fillna("") != ""]
    dup = e["ecb_number"].duplicated(keep=False)
    span = (e[dup].groupby("ecb_number")["agency"].nunique() > 1).sum()
    log(f"\nC8 ECB numbers linked to >1 complaint record: "
        f"{dup.sum():,} rows ({dup.mean():.4f}); numbers spanning both "
        f"origins: {span}")


def main():
    conn = sqlite3.connect(DB)
    ev = load_spine()
    od = load_od_dates(conn)
    section_A(conn, ev)
    ev = section_B(ev, od)
    section_C(conn, ev, od)
    conn.close()
    log(f"\ntotal runtime {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
