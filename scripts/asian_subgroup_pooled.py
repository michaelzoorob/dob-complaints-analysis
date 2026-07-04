"""
Pooled subgroup models: the individual Asian-origin subgroups are small, so
"not significant" cannot be read as "zero". This script pools them into
three blocs and re-estimates the same models with the same specification as
scripts/asian_subgroups.py:

  chinese          Chinese (73,003)
  south_asian      Bangladeshi/Pakistani + Indo-Caribbean + Singh +
                   Sikh/Punjabi + Indian + Nepali/Himalayan (24,009)
  korean_viet      Korean + Vietnamese (3,705)

Reference = classified-white owners; unclassified-Asian, small pooled
(Filipino/Japanese), classified-Black, classified-Hispanic, and
mixed/uncertain enter as separate dummies. 95% confidence intervals are
reported so that null results read as ranges, not as zeros.

Outputs -> risk_models/asian_subgroup_pooled.csv + console report.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import asian_subgroups as asg
from analysis_config import make_bbl
from disposition_codes import classify_disposition

OUT = config.DATA_DIR / "analysis" / "risk_models"

SOUTH_ASIAN = ["muslim_sa", "indo_caribbean", "singh", "sikh_punjabi",
               "indian", "nepali_himalayan"]
KOREAN_VIET = ["korean", "vietnamese"]
SMALL = ["filipino", "japanese"]


def main():
    import pyfixest as pf
    bs = asg.load_panel()
    bs = asg.assign(bs)

    bs["sg_chinese"] = (bs["subgroup"] == "chinese").astype(int)
    bs["sg_south_asian"] = bs["subgroup"].isin(SOUTH_ASIAN).astype(int)
    bs["sg_korean_viet"] = bs["subgroup"].isin(KOREAN_VIET).astype(int)
    bs["sg_asian_small"] = bs["subgroup"].isin(SMALL).astype(int)
    dummies = ["sg_chinese", "sg_south_asian", "sg_korean_viet", "sg_asian_small",
               "asian_uncl", "black_c", "hisp_c"]
    bs["mixed_uncertain"] = ((bs[dummies].sum(axis=1) == 0) & (bs["white_c"] == 0)).astype(int)
    X = " + ".join(dummies + ["mixed_uncertain"] + asg.BUILDING_COVARS + asg.OWNER_COVARS)
    vcov = {"CRV1": "bct2020"}
    print(f"\npooled ns: south_asian {bs['sg_south_asian'].sum():,}, "
          f"korean_viet {bs['sg_korean_viet'].sum():,}, chinese {bs['sg_chinese'].sum():,}")

    res = []

    def collect(m, model, outcome):
        t = m.tidy().reset_index()
        t.columns = [x.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                     for x in t.columns]
        t = t.rename(columns={t.columns[0]: "term"})
        t["model"], t["outcome"], t["n"] = model, outcome, m._N
        res.append(t)

    def report(m, label, kind):
        t = m.tidy()
        for g in ["sg_chinese", "sg_south_asian", "sg_korean_viet"]:
            b, se, p = t.loc[g, "Estimate"], t.loc[g, "Std. Error"], t.loc[g, "Pr(>|t|)"]
            if kind == "pct":
                e, lo, hi = (np.exp(b)-1)*100, (np.exp(b-1.96*se)-1)*100, (np.exp(b+1.96*se)-1)*100
                print(f"  {label:<26} {g:<16} {e:+7.1f}%  [{lo:+6.1f}, {hi:+6.1f}]  p {p:.4f}")
            else:
                print(f"  {label:<26} {g:<16} {b:+6.2f}pp [{b-1.96*se:+6.2f}, {b+1.96*se:+6.2f}]  p {p:.4f}")

    print("\n== pooled PPML counts (vs classified-white) ==")
    for outc, label in [("n_complaints", "complaints"), ("n_conv", "conversion complaints"),
                        ("n_ecb_2020on", "ECB citations"), ("n_viol_disp", "disposition violations")]:
        m = pf.fepois(f"{outc} ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
        collect(m, f"ppml_pooled_{outc}", label)
        report(m, label, "pct")

    conn = sqlite3.connect(str(config.DB_PATH))
    boro_case = """CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
        WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' END"""
    c = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.disposition_code, o.complaint_category,
               {boro_case} AS boro_code, b.block, b.lot
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                    in zip(c["boro_code"], c["block"], c["lot"])]
    c["outcome"] = c["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    c["viol100"] = (c["outcome"] == "violation").astype(float) * 100
    c["noacc100"] = (c["outcome"] == "no_access").astype(float) * 100
    keep = ["bbl_key", "size_bin", "bct2020"] + dummies + ["mixed_uncertain", "white_c"] + \
        asg.BUILDING_COVARS + asg.OWNER_COVARS
    insp = c.merge(bs[keep], on="bbl_key")
    sub = insp[insp["outcome"].isin(["violation", "no_violation"])]
    print("\n== pooled inspection-level LPMs (pp, within complaint code) ==")
    for frame, outc, label in [(sub, "viol100", "violation | substantive"),
                               (insp, "noacc100", "no access")]:
        m = pf.feols(f"{outc} ~ {X} | complaint_category + size_bin + bct2020",
                     data=frame, vcov=vcov)
        collect(m, f"lpm_pooled_{outc}", label)
        report(m, label, "pp")

    out = pd.concat(res, ignore_index=True)
    out["pct_change"] = (np.exp(out["estimate"]) - 1) * 100
    out.to_csv(OUT / "asian_subgroup_pooled.csv", index=False)
    print(f"\nsaved -> {OUT/'asian_subgroup_pooled.csv'}")


if __name__ == "__main__":
    main()
