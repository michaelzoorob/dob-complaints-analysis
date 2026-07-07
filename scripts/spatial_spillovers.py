"""
Spatial Spillover Analysis: does strict enforcement at one property predict
permit filings or new ECB violations at same-tax-block neighbors?

Design:
- For each substantively inspected property, neighbors = all other PLUTO lots
  on the same tax block.
- Outcomes: neighbor permit filings and neighbor ECB violations within
  90/180/365 days after the focal inspection, plus the share of neighbor
  lots with at least one permit.
- Treatment: the focal inspector's leave-one-out strictness.
- Inference: standard errors clustered by focal inspector (strictness varies
  across ~640 inspectors, so unclustered SEs are drastically overconfident).

Implementation notes: neighbor counts are computed with per-block
numpy.searchsorted over pre-sorted event-date arrays (no Python loop over
inspections), which runs in seconds; the assembled analysis frame is cached
to data/analysis/spillover_frame.pkl for re-analysis.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from disposition_codes import classify_disposition

WINDOWS = [90, 180, 365]
CACHE = config.DATA_DIR / "analysis" / "spillover_frame.pkl"
BC = ("CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2' "
      "WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' END")
BC_MAP = {"MANHATTAN": "1", "BRONX": "2", "BROOKLYN": "3", "QUEENS": "4", "STATEN ISLAND": "5"}


def _bbl(boro: pd.Series, block: pd.Series, lot: pd.Series) -> pd.Series:
    """Vectorized 10-digit BBL from string/float parts; empty string when invalid."""
    b = pd.to_numeric(block, errors="coerce")
    l = pd.to_numeric(lot, errors="coerce")
    ok = boro.notna() & b.notna() & l.notna()
    out = pd.Series("", index=boro.index, dtype=object)
    out[ok] = (boro[ok].astype(str)
               + b[ok].astype(int).astype(str).str.zfill(5)
               + l[ok].astype(int).astype(str).str.zfill(4))
    return out


def load_focal(conn) -> pd.DataFrame:
    """Substantive inspections with LOO strictness, BBL, and block key."""
    df = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.disposition_code, o.date_entered, o.inspection_date,
               b.borough, b.category_description, b.assigned_to,
               b.inspector_badge, b.block, b.lot, cn.nta
        FROM open_data o
        JOIN bis_scrape b ON o.complaint_number = b.complaint_number
        JOIN complaint_nta cn ON cn.complaint_number = o.complaint_number
        WHERE b.inspector_badge IS NOT NULL
          AND o.disposition_code IS NOT NULL AND o.disposition_code != ''
          AND b.block IS NOT NULL AND b.lot IS NOT NULL
    """, conn)
    df["outcome"] = df["disposition_code"].apply(classify_disposition)
    df["violation_found"] = (df["outcome"] == "violation").astype(int)
    df["inspection_dt"] = pd.to_datetime(df["inspection_date"], format="%m/%d/%Y", errors="coerce")
    df["year_month"] = pd.to_datetime(df["date_entered"], format="%m/%d/%Y",
                                      errors="coerce").dt.to_period("M").astype(str)
    df["boro_code"] = df["borough"].map(BC_MAP)
    df["bbl"] = _bbl(df["boro_code"], df["block"], df["lot"])
    blk = pd.to_numeric(df["block"], errors="coerce")
    df["boro_block"] = df["boro_code"].astype(str) + "_" + blk.astype("Int64").astype(str)

    sub = df[df["outcome"].isin(["violation", "no_violation"])].copy()
    counts = sub["inspector_badge"].value_counts()
    sub = sub[sub["inspector_badge"].isin(counts[counts >= 30].index)]
    stats = sub.groupby("inspector_badge")["violation_found"].agg(["sum", "count"])
    sub = sub.merge(stats.rename(columns={"sum": "tv", "count": "tn"}), on="inspector_badge")
    sub["loo"] = (sub["tv"] - sub["violation_found"]) / (sub["tn"] - 1)
    sub = sub.drop(columns=["tv", "tn"]).dropna(subset=["inspection_dt"])
    # drop a handful of corrupt pre-2019 inspection dates in a 2020+ entered sample
    sub = sub[sub["inspection_dt"] >= "2019-01-01"]
    print(f"Focal sample: {len(sub):,} complaints across {sub['bbl'].nunique():,} BBLs, "
          f"{sub['boro_block'].nunique():,} blocks")
    return sub


def load_events(conn):
    """Permit filings (BIS issued + DOB NOW filed) and ECB violations, as (bbl, dt)."""
    permits = pd.read_sql_query(
        "SELECT bbl, issued_date AS d FROM permits WHERE bbl IS NOT NULL AND issued_date IS NOT NULL "
        "UNION ALL "
        "SELECT bbl, filing_date AS d FROM permits_now WHERE bbl IS NOT NULL AND filing_date IS NOT NULL",
        conn)
    permits["dt"] = pd.to_datetime(permits["d"], errors="coerce")
    permits = permits.dropna(subset=["dt"])[["bbl", "dt"]]
    print(f"Total permits: {len(permits):,}")

    ecb = pd.read_sql_query(
        "SELECT boro, block, lot, issue_date FROM ecb_violations "
        "WHERE issue_date IS NOT NULL AND boro IS NOT NULL AND block IS NOT NULL AND lot IS NOT NULL",
        conn)
    ecb["bbl"] = _bbl(ecb["boro"].astype(str), ecb["block"], ecb["lot"])
    ecb["dt"] = pd.to_datetime(ecb["issue_date"], format="%Y%m%d", errors="coerce")
    ecb = ecb[(ecb["bbl"] != "")].dropna(subset=["dt"])[["bbl", "dt"]]
    print(f"ECB violations: {len(ecb):,}")
    return permits, ecb


def load_blocks(conn):
    """PLUTO block -> array of BBLs (the neighbor universe)."""
    pl = pd.read_sql_query(
        "SELECT borocode, block, lot FROM pluto WHERE latitude IS NOT NULL", conn)
    pl["bbl"] = _bbl(pl["borocode"].astype(str), pl["block"], pl["lot"])
    blk = pd.to_numeric(pl["block"], errors="coerce")
    pl["boro_block"] = pl["borocode"].astype(str) + "_" + blk.astype("Int64").astype(str)
    pl = pl[pl["bbl"] != ""]
    idx = pl.groupby("boro_block")["bbl"].apply(lambda s: np.array(sorted(set(s))))
    print(f"Blocks with properties: {len(idx):,}")
    return idx.to_dict()


def _dates_by_key(events: pd.DataFrame, key: str) -> dict:
    """key -> sorted int64 event-date array."""
    ev = events.sort_values("dt")
    return {k: g["dt"].values.astype("datetime64[ns]").astype(np.int64)
            for k, g in ev.groupby(key, sort=False)}


def neighbor_outcomes(sub: pd.DataFrame, permits: pd.DataFrame,
                      ecb: pd.DataFrame, block_bbls: dict) -> pd.DataFrame:
    """Vectorized neighbor counts: per-block searchsorted over sorted date arrays."""
    print("\nComputing neighbor outcomes (vectorized)...", flush=True)
    bbl_to_block = {}
    for blk, arr in block_bbls.items():
        for b in arr:
            bbl_to_block[b] = blk
    for name, ev in (("permit", permits), ("ecb", ecb)):
        ev["boro_block"] = ev["bbl"].map(bbl_to_block)

    p_by_block = _dates_by_key(permits.dropna(subset=["boro_block"]), "boro_block")
    p_by_bbl = _dates_by_key(permits, "bbl")
    e_by_block = _dates_by_key(ecb.dropna(subset=["boro_block"]), "boro_block")
    e_by_bbl = _dates_by_key(ecb, "bbl")
    # per-block list of (bbl, its permit-date array), for the distinct-neighbor share
    p_bbl_lists = {}
    for b, dates in p_by_bbl.items():
        blk = bbl_to_block.get(b)
        if blk is not None:
            p_bbl_lists.setdefault(blk, []).append((b, dates))

    DAY = 86_400_000_000_000  # ns
    EMPTY = np.array([], dtype=np.int64)
    out = {c: [] for c in ["complaint_number", "n_neighbors"]}
    for w in WINDOWS:
        for c in (f"neighbor_permits_{w}d", f"neighbor_any_permit_{w}d",
                  f"neighbor_pct_permit_{w}d", f"neighbor_ecb_{w}d"):
            out[c] = []

    sub = sub.sort_values("boro_block")
    for blk, g in sub.groupby("boro_block", sort=False):
        universe = block_bbls.get(blk)
        if universe is None or len(universe) == 0:
            continue
        uni_set = set(universe)
        lo = g["inspection_dt"].values.astype("datetime64[ns]").astype(np.int64)
        focal_bbls = g["bbl"].values
        in_uni = np.fromiter((b in uni_set for b in focal_bbls), bool, len(focal_bbls))
        n_nb = len(universe) - in_uni.astype(int)
        keep = n_nb > 0
        if not keep.any():
            continue
        bp = p_by_block.get(blk, EMPTY)
        be = e_by_block.get(blk, EMPTY)
        own_p = [p_by_bbl.get(b, EMPTY) for b in focal_bbls]
        own_e = [e_by_bbl.get(b, EMPTY) for b in focal_bbls]
        out["complaint_number"].extend(g["complaint_number"].values[keep])
        out["n_neighbors"].extend(n_nb[keep])
        for w in WINDOWS:
            hi = lo + w * DAY
            blk_p = np.searchsorted(bp, hi, "right") - np.searchsorted(bp, lo, "right")
            blk_e = np.searchsorted(be, hi, "right") - np.searchsorted(be, lo, "right")
            own_pc = np.array([np.searchsorted(d, h, "right") - np.searchsorted(d, l, "right")
                               for d, l, h in zip(own_p, lo, hi)])
            own_ec = np.array([np.searchsorted(d, h, "right") - np.searchsorted(d, l, "right")
                               for d, l, h in zip(own_e, lo, hi)])
            nb_p = blk_p - own_pc
            nb_e = blk_e - own_ec
            # distinct neighbor lots with >=1 permit: loop over the block's
            # permit-holding lots (few), vectorized over this block's focals
            with_permit = np.zeros(len(g), dtype=np.int64)
            for b, dates in p_bbl_lists.get(blk, ()):
                cnt = np.searchsorted(dates, hi, "right") - np.searchsorted(dates, lo, "right")
                with_permit += ((cnt > 0) & (focal_bbls != b)).astype(np.int64)
            out[f"neighbor_permits_{w}d"].extend(nb_p[keep])
            out[f"neighbor_any_permit_{w}d"].extend((with_permit[keep] > 0).astype(int))
            out[f"neighbor_pct_permit_{w}d"].extend(with_permit[keep] / n_nb[keep])
            out[f"neighbor_ecb_{w}d"].extend(nb_e[keep])

    spill = pd.DataFrame(out)
    merged = sub.merge(spill, on="complaint_number", how="inner")
    print(f"Spillover sample: {len(merged):,} complaints with neighbors "
          f"(mean neighbors {merged['n_neighbors'].mean():.1f})")
    return merged


def estimate(sub_spill: pd.DataFrame):
    """Within-cell estimates with inspector-clustered SEs (pyfixest)."""
    import pyfixest as pf
    sub_spill = sub_spill.copy()
    sub_spill["cell"] = (sub_spill["category_description"].fillna("U").astype(str) + "|"
                         + sub_spill["assigned_to"].fillna("U").astype(str) + "|"
                         + sub_spill["year_month"].fillna("U").astype(str) + "|"
                         + sub_spill["nta"].astype(str))
    print(f"\n{'='*95}")
    print("SPATIAL SPILLOVERS - same-block neighbors, cat x unit x ym x NTA cells, "
          "SEs clustered by inspector")
    print(f"{'='*95}")
    print(f"{'Outcome':<36} {'beta':>9} {'clust SE':>9} {'t':>7} {'N':>9} {'mean':>8}")
    print("-" * 85)
    for w in WINDOWS:
        for col in (f"neighbor_pct_permit_{w}d", f"neighbor_permits_{w}d", f"neighbor_ecb_{w}d"):
            d = sub_spill[[col, "loo", "cell", "inspector_badge"]].dropna()
            d.columns = ["y", "x", "fe", "insp"]
            m = pf.feols("y ~ x | fe", data=d, vcov={"CRV1": "insp"})
            b, se = float(m.coef()["x"]), float(m.se()["x"])
            print(f"  {col:<34} {b:>9.4f} {se:>9.4f} {b/se:>7.2f} {m._N:>9,} "
                  f"{d['y'].mean():>8.3f}")


def main():
    conn = sqlite3.connect(str(config.DB_PATH), timeout=30)
    sub = load_focal(conn)
    permits, ecb = load_events(conn)
    block_bbls = load_blocks(conn)
    conn.close()
    sub_spill = neighbor_outcomes(sub, permits, ecb, block_bbls)
    sub_spill.to_pickle(CACHE)
    print(f"cached spillover frame -> {CACHE}")
    estimate(sub_spill)


if __name__ == "__main__":
    main()
