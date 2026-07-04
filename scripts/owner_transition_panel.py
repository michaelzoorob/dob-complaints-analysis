"""
Within-property robustness check: ownership transitions and complaints.

Builds an owner panel from ACRIS deed sequences (master: deed types/dates;
legals: document -> lot; parties: grantee names), predicts each incoming
owner's race from grantee surnames (surname-only BISG, matching the
article's robustness variant), and asks whether the SAME property's
complaint rate changes when a deed transfers it between predicted-white
and predicted-Asian individual owners.

Design: event study / difference-in-differences around deed dates with
transition fixed effects and calendar-year fixed effects. Race-changing
sales (white->Asian) are benchmarked against race-stable sales
(white->white), which absorbs generic new-owner effects (renovation,
tenancy turnover). Symmetric check for Asian->white vs Asian->Asian.
Outcomes: DOB complaints per property-year (open_data joined via the
BIN->BBL crosswalk, 1990-2025) and ECB violations per property-year.

Sample rules: residential lots with 1-15 units; deeds with valid dates;
transitions where consecutive grantees are both surname-classifiable
individuals (P>0.7); clean +/-4 year windows (no other deed).

Outputs: risk_models/transition_tidy_estimates.csv,
         risk_models/transition_eventstudy.csv, console diagnostics.
"""

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl
from disposition_codes import classify_disposition

OUT = config.DATA_DIR / "analysis" / "risk_models"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"

Y0, Y1 = 1990, 2025          # outcome years
WINDOW = 4                    # event window +/- years
THR = 0.7                     # surname classification threshold

RE_CORP = re.compile(
    r"\b(LLC|L L C|CORP|INC|BANK|TRUST|TRUSTEE|LP|L P|LLP|ASSOC|HOLDING|CHURCH|"
    r"COUNCIL|CITY|AUTHORITY|FUND|PARTNERS|REALTY|GROUP|PROPERTIES|ESTATE|"
    r"HDFC|CONDO|FEDERAL|NATIONAL|MORTGAGE|COMPANY|CO\b|SERVICES|DEPT|USA|"
    r"UNITED STATES|SECRETARY|VETERANS|HUD|FANNIE|FREDDIE|RELOCATION)\b", re.I)


def parse_surname(name: str) -> str:
    if not isinstance(name, str) or RE_CORP.search(name):
        return ""
    s = name.split(",")[0].strip() if "," in name else name.strip().split(" ")[0]
    s = re.sub(r"[^A-Z\-']", "", s.upper())
    return s if len(s) >= 2 else ""


def load_universe():
    df = pd.read_csv(PANEL, usecols=["bbl_key", "unitsres"], dtype={"bbl_key": str})
    u = set(df.loc[df["unitsres"].between(1, 15), "bbl_key"])
    print(f"universe lots (1-15 units): {len(u):,}")
    return u


def load_deeds(conn, universe):
    m = pd.read_sql_query("""
        SELECT document_id, doc_type, substr(doc_date,1,10) AS d
        FROM acris_master WHERE doc_date IS NOT NULL""", conn)
    m["date"] = pd.to_datetime(m["d"], errors="coerce")
    m = m.dropna(subset=["date"])
    m = m[(m["date"].dt.year >= 1985) & (m["date"].dt.year <= 2026)]
    print(f"dated deed documents: {len(m):,}")

    l = pd.read_sql_query(
        "SELECT document_id, borough, block, lot FROM acris_legals", conn)
    l["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                    in zip(l["borough"], l["block"], l["lot"])]
    l = l[l["bbl_key"].isin(universe)][["document_id", "bbl_key"]].drop_duplicates()
    deeds = m.merge(l, on="document_id")
    print(f"deed-lot rows in universe: {len(deeds):,} "
          f"({deeds['bbl_key'].nunique():,} lots)")
    return deeds


def classify_grantees(conn, deeds):
    import surgeo
    p = pd.read_sql_query("SELECT document_id, name FROM acris_parties", conn)
    p = p[p["document_id"].isin(set(deeds["document_id"]))]
    p["surname"] = p["name"].apply(parse_surname)
    p = p[p["surname"] != ""]
    uniq = pd.Series(sorted(p["surname"].unique()))
    print(f"grantee rows on universe deeds: {len(p):,}; unique surnames {len(uniq):,}")
    doc_surnames = p.groupby("document_id")["surname"].agg(set)
    sm = surgeo.SurnameModel().get_probabilities(uniq).set_index("name")
    sm = sm[~sm.index.duplicated()]
    sm["asian"] = sm["api"]
    four = sm[["white", "black", "asian", "hispanic"]]
    four = four.div(four.sum(axis=1), axis=0)
    for c in four.columns:
        p[f"p_{c}"] = p["surname"].map(four[c])
    p = p.dropna(subset=["p_white"])
    doc = p.groupby("document_id")[["p_white", "p_black", "p_asian", "p_hispanic"]].mean()
    doc["n_grantees"] = p.groupby("document_id").size()
    doc["surnames"] = doc_surnames
    print(f"documents with classified individual grantee(s): {len(doc):,}")
    return doc


def build_transitions(deeds, doc):
    d = deeds.merge(doc, on="document_id", how="left")
    d = d.sort_values(["bbl_key", "date", "document_id"])
    # collapse re-records within 180 days
    d["gap"] = d.groupby("bbl_key")["date"].diff().dt.days
    d = d[(d["gap"].isna()) | (d["gap"] > 180)].copy()

    d["prev_pw"] = d.groupby("bbl_key")["p_white"].shift()
    d["prev_pa"] = d.groupby("bbl_key")["p_asian"].shift()
    d["prev_date"] = d.groupby("bbl_key")["date"].shift()
    d["next_date"] = d.groupby("bbl_key")["date"].shift(-1)
    d["prev_surnames"] = d.groupby("bbl_key")["surnames"].shift()

    t = d.dropna(subset=["prev_pw", "p_white"]).copy()
    t["year"] = t["date"].dt.year

    def cls(pw, pa):
        return np.select([pw > THR, pa > THR], ["W", "A"], default="")
    t["from"] = cls(t["prev_pw"], t["prev_pa"])
    t["to"] = cls(t["p_white"], t["p_asian"])
    t = t[(t["from"] != "") & (t["to"] != "")]
    t["ttype"] = t["from"] + "2" + t["to"]
    # family transfer: any shared surname between consecutive grantee sets
    t["family"] = [
        int(isinstance(a, set) and isinstance(b, set) and len(a & b) > 0)
        for a, b in zip(t["prev_surnames"], t["surnames"])]

    # clean window: no prior deed within WINDOW years before, none after
    ok_pre = (t["date"] - t["prev_date"]).dt.days > WINDOW * 365
    ok_post = t["next_date"].isna() | ((t["next_date"] - t["date"]).dt.days > WINDOW * 365)
    t = t[ok_pre & ok_post]
    t = t[(t["year"] >= Y0 + WINDOW) & (t["year"] <= Y1 - WINDOW + 1)]
    t = t.drop_duplicates(subset=["bbl_key", "document_id"])
    t["trans_id"] = np.arange(len(t))
    print("\ntransition counts (clean windows):")
    print(pd.crosstab(t["ttype"], t["family"]).to_string())
    return t[["trans_id", "bbl_key", "year", "ttype", "family"]]


def load_outcomes(conn, universe):
    xw = pd.read_sql_query("SELECT bin, bbl_key FROM bin_bbl_all", conn)
    xw = xw.drop_duplicates("bin").set_index("bin")["bbl_key"]

    c = pd.read_sql_query("""
        SELECT bin, substr(date_entered,7,4) AS yr, complaint_category,
               disposition_code
        FROM open_data""", conn)
    c["bbl_key"] = c["bin"].map(xw)
    c["yr"] = pd.to_numeric(c["yr"], errors="coerce")
    c = c[c["bbl_key"].isin(universe) & c["yr"].between(Y0, Y1)]
    print(f"\ncomplaints matched to universe lots {Y0}-{Y1}: {len(c):,} "
          f"(match rate incl. non-universe bins: {c['bbl_key'].notna().mean():.2f})")
    comp = (c.groupby(["bbl_key", "yr"]).size().rename("n_comp").reset_index())
    conv = (c[c["complaint_category"] == "45"]
            .groupby(["bbl_key", "yr"]).size().rename("n_conv").reset_index())
    c["outc"] = c["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    viol = (c[c["outc"] == "violation"]
            .groupby(["bbl_key", "yr"]).size().rename("n_viol").reset_index())

    e = pd.read_sql_query("""
        SELECT boro, block, lot, CAST(substr(issue_date,1,4) AS INTEGER) AS yr
        FROM ecb_violations WHERE length(issue_date) >= 8""", conn)
    e["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                    in zip(e["boro"], e["block"], e["lot"])]
    e = e[e["bbl_key"].isin(universe) & e["yr"].between(Y0, Y1)]
    ecb = e.groupby(["bbl_key", "yr"]).size().rename("n_ecb").reset_index()
    return comp, conv, ecb, viol


def build_frame(trans, comp, conv, ecb, viol):
    rows = trans.loc[trans.index.repeat(2 * WINDOW + 1)].copy()
    rows["event_t"] = np.tile(np.arange(-WINDOW, WINDOW + 1), len(trans))
    rows["yr"] = rows["year"] + rows["event_t"]
    rows = rows[(rows["yr"] >= Y0) & (rows["yr"] <= Y1)]
    for df_, col in [(comp, "n_comp"), (conv, "n_conv"), (ecb, "n_ecb"),
                     (viol, "n_viol")]:
        rows = rows.merge(df_, on=["bbl_key", "yr"], how="left")
        rows[col] = rows[col].fillna(0)
    rows["post"] = (rows["event_t"] >= 0).astype(int)
    return rows


def estimate(frame):
    import pyfixest as pf
    res = []

    def run(sample, treat_type, label, arms_length=False):
        s = frame[frame["ttype"].isin(sample)].copy()
        if arms_length:
            s = s[s["family"] == 0]
        s["treat"] = (s["ttype"] == treat_type).astype(int)
        s["treat_post"] = s["treat"] * s["post"]
        n_t = s.loc[s["treat"] == 1, "trans_id"].nunique()
        n_c = s.loc[s["treat"] == 0, "trans_id"].nunique()
        ets = []
        for outc in ["n_comp", "n_conv", "n_ecb", "n_viol"]:
            m = pf.feols(f"{outc} ~ treat_post + post | trans_id + yr",
                         data=s, vcov={"CRV1": "bbl_key"})
            t = m.tidy().loc["treat_post"]
            res.append({"contrast": label, "outcome": outc,
                        "arms_length": arms_length,
                        "estimate": t["Estimate"], "se": t["Std. Error"],
                        "p": t["Pr(>|t|)"], "n_treat": n_t, "n_ctrl": n_c,
                        "N": m._N})
            print(f"  {label:<14} {'AL' if arms_length else 'all':<4} {outc:<7} "
                  f"b={t['Estimate']:+.4f} (se {t['Std. Error']:.4f}, "
                  f"p {t['Pr(>|t|)']:.3f}) [{n_t:,}/{n_c:,}]")
            es = pf.feols(f"{outc} ~ i(event_t, treat, ref=-1) + C(event_t)"
                          " | trans_id + yr", data=s, vcov={"CRV1": "bbl_key"})
            et = es.tidy().reset_index()
            et["contrast"] = label
            et["outcome"] = outc
            et["arms_length"] = arms_length
            ets.append(et)
            # joint Wald tests: all post-sale (t>=0) and all pre-sale (t<-1)
            # treatment-interaction coefficients jointly zero, using the
            # cluster-robust covariance (chi2)
            names = [str(n) for n in es._coefnames]
            for window, keep in [("post", lambda k: k >= 0), ("pre", lambda k: k < -1)]:
                idx = []
                for i, n in enumerate(names):
                    if n.startswith("event_t::") and n.endswith(":treat"):
                        k = int(n.split("::")[1].split(":")[0])
                        if keep(k):
                            idx.append(i)
                R = np.zeros((len(idx), len(names)))
                for j, i in enumerate(idx):
                    R[j, i] = 1.0
                w = es.wald_test(R=R)
                WALD.append({"contrast": label, "outcome": outc,
                             "arms_length": arms_length, "window": window,
                             "df": len(idx), "chi2": float(w["statistic"]),
                             "p": float(w["pvalue"])})
        return pd.concat(ets)

    print("\n== DiD: treat x post, transition FE + year FE, cluster by lot ==")
    all_es = []
    global WALD
    WALD = []
    for al in [False, True]:
        all_es.append(run(["W2A", "W2W"], "W2A", "W->A vs W->W", al))
        all_es.append(run(["A2W", "A2A"], "A2W", "A->W vs A->A", al))

    pd.DataFrame(res).to_csv(OUT / "transition_tidy_estimates.csv", index=False)
    pd.concat(all_es).to_csv(OUT / "transition_eventstudy.csv", index=False)
    wd = pd.DataFrame(WALD)
    wd.to_csv(OUT / "transition_wald_tests.csv", index=False)
    print("\njoint Wald tests (event-study coefficients jointly zero):")
    print(wd.to_string(index=False))
    print(f"\nsaved -> {OUT/'transition_tidy_estimates.csv'} and transition_eventstudy.csv")


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    universe = load_universe()
    deeds = load_deeds(conn, universe)
    doc = classify_grantees(conn, deeds)
    trans = build_transitions(deeds, doc)
    comp, conv, ecb, viol = load_outcomes(conn, universe)
    conn.close()
    frame = build_frame(trans, comp, conv, ecb, viol)
    print(f"\npanel rows: {len(frame):,}")
    # pre-period balance: mean complaints in t<0 by type
    pre = frame[frame["event_t"] < 0].groupby(["ttype", "family"])[
        ["n_comp", "n_conv"]].mean()
    print("pre-period means per year by transition type x family:")
    print(pre.round(4).to_string())
    estimate(frame)


if __name__ == "__main__":
    main()
