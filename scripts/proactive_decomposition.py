#!/usr/bin/env python3
"""
Hypothesis #3 (proactive_enforcement_plan.md): decompose the 249,707
agency-initiated complaints (empty ref_311, received 2020-01..2026-05)
into what sent the inspector.

Partition (exact, sums to the agency total):
  statutory_periodic   cyclical / legally mandated families (7K, 7F, 6V, ...)
  followup_family      re-inspections keyed to earlier enforcement
                       (7R, 4G, 1L, 2H)
  discretionary_warm   everything else (discretionary_field + mixed_incident
                       + other families) at a WARM building: any DOB
                       complaint, caller or agency, at the same BBL in the
                       prior 730 days (strictly before the received date)
  discretionary_cold   same pool, DOB-cold: no complaint at the BBL in the
                       prior 730 days (de-novo targeting)

Warm/cold uses the per-BBL sorted searchsorted pattern from
scripts/spatial_spillovers.py, collapsed into a single global
np.searchsorted over integer (bbl_code * K + day) keys — no Python loop.

Left-truncation caveat: events received before 2022-01-01 have less than a
full 730-day lookback inside the spine window, which overstates "cold"
early on. A sensitivity block restricts focal events to 2022-01+ (full
lookback available for every event).

Per slice: count, share of agency work, warm share, violation-disposition
rate (classify_disposition == "violation"), ECB-hit rate (linked ECB
violation number present), CLASS-1 share of severity-matched ECB hits, and
the unconditional CLASS-1 rate per event.

Outputs
  data/analysis/risk_models/proactive_decomposition.csv
  data/analysis/blog_posts/artifacts/proactive_decomposition.png

Run:  /private/tmp/pyfix_venv/bin/python scripts/proactive_decomposition.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPINE_DIR = os.path.join(ROOT, "data", "analysis", "proactive")
RISK_DIR = os.path.join(ROOT, "data", "analysis", "risk_models")
ART_DIR = os.path.join(ROOT, "data", "analysis", "blog_posts", "artifacts")

OUT_CSV = os.path.join(RISK_DIR, "proactive_decomposition.csv")
OUT_PNG = os.path.join(ART_DIR, "proactive_decomposition.png")

LOOKBACK_DAYS = 730
FULL_LOOKBACK_START = pd.Timestamp("2022-01-01")  # 730d inside the window
DISCRETIONARY_FAMILIES = ("discretionary_field", "mixed_incident", "other")
TOP_PREFIXES = 6  # prefix detail rows for the biggest discretionary programs

# ── house style (copied from scripts/make_descriptive_figures.py) ────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_events() -> pd.DataFrame:
    """Spine events with the columns this decomposition needs."""
    cols = ["received_date", "category_prefix", "family", "agency", "bbl",
            "outcome", "ecb_number", "ecb_severity"]
    pq = os.path.join(SPINE_DIR, "proactive_events.parquet")
    gz = os.path.join(SPINE_DIR, "proactive_events.csv.gz")
    if os.path.exists(pq):
        ev = pd.read_parquet(pq, columns=cols)
    else:
        ev = pd.read_csv(gz, usecols=cols,
                         dtype={"bbl": str, "ecb_number": str,
                                "ecb_severity": str})
    ev["received"] = pd.to_datetime(ev["received_date"], format="%Y-%m-%d")
    ev["bbl"] = ev["bbl"].fillna("")
    ev["ecb_hit"] = (ev["ecb_number"].fillna("") != "").astype("int8")
    ev["class1"] = (ev["ecb_severity"] == "CLASS - 1").astype("int8")
    print(f"Spine events: {len(ev):,} "
          f"({ev['received'].min():%Y-%m-%d}..{ev['received'].max():%Y-%m-%d}); "
          f"agency {int((ev['agency'] == 1).sum()):,}")
    return ev


def add_warm_flag(ev: pd.DataFrame) -> pd.DataFrame:
    """prior_730 = complaints (any origin) at the same BBL in the prior
    730 days, strictly before the received date. Same-day events at the
    BBL (sweep batches, the focal row itself) never count.

    Single searchsorted over integer keys bbl_code * K + (day + 730):
    lo = key - 730 (day >= d-730), hi = key with side='left' (day < d).
    """
    day = (ev["received"] - ev["received"].min()).dt.days.to_numpy()
    codes = pd.factorize(ev["bbl"])[0]
    K = int(day.max()) + LOOKBACK_DAYS + 2
    key = codes.astype(np.int64) * K + (day + LOOKBACK_DAYS)

    located = (ev["bbl"] != "").to_numpy()

    def prior_counts(pool_mask: np.ndarray) -> np.ndarray:
        pool = np.sort(key[pool_mask & located])
        n = (np.searchsorted(pool, key, side="left")
             - np.searchsorted(pool, key - LOOKBACK_DAYS, side="left"))
        n[~located] = 0  # 3 agency rows lack a BBL
        return n

    ev["prior_730"] = prior_counts(np.ones(len(ev), dtype=bool))
    ev["warm"] = (ev["prior_730"] > 0).astype("int8")
    # interpretive guard: warmth traceable to a CALLER complaint, so
    # "warm" is not read as purely DOB's own prior program visits
    caller_pool = (ev["agency"] == 0).to_numpy()
    ev["warm_caller"] = (prior_counts(caller_pool) > 0).astype("int8")
    ag = ev[ev["agency"] == 1]
    print(f"Warm flag: {int(located.sum()):,} located events in the pool; "
          f"agency warm share {ag['warm'].mean():.3f} "
          f"(via caller complaint {ag['warm_caller'].mean():.3f}); "
          f"mean prior complaints (warm agency rows) "
          f"{ag.loc[ag['warm'] == 1, 'prior_730'].mean():.1f}")
    return ev


def metrics(sub: pd.DataFrame, group: str, slice_name: str, desc: str,
            denom: int) -> dict:
    sev_matched = sub["ecb_severity"].notna()
    n_matched = int(sev_matched.sum())
    return {
        "group": group,
        "slice": slice_name,
        "description": desc,
        "n": len(sub),
        "share_of_agency": round(len(sub) / denom, 4),
        "warm_share": round(float(sub["warm"].mean()), 4),
        "warm_caller_share": round(float(sub["warm_caller"].mean()), 4),
        "violation_disposition_rate": round(float((sub["outcome"] == "violation").mean()), 4),
        "ecb_hit_rate": round(float(sub["ecb_hit"].mean()), 4),
        "class1_share_of_ecb_hits": (round(float(sub.loc[sev_matched, "class1"].mean()), 4)
                                     if n_matched else np.nan),
        "class1_rate_per_event": round(float(sub["class1"].mean()), 4),
    }


def slice_masks(ag: pd.DataFrame) -> list:
    """(slice_name, description, mask) for the main partition."""
    disc = ag["family"].isin(DISCRETIONARY_FAMILIES)
    return [
        ("statutory_periodic",
         "cyclical/mandated families (statutory_periodic)",
         ag["family"] == "statutory_periodic"),
        ("followup_family",
         "re-inspections keyed to earlier enforcement (7R/4G/1L/2H)",
         ag["family"] == "followup"),
        ("discretionary_warm",
         "discretionary_field+mixed_incident+other, any DOB complaint at "
         "the BBL in the prior 730 days",
         disc & (ag["warm"] == 1)),
        ("discretionary_cold",
         "same pool, no DOB complaint at the BBL in the prior 730 days "
         "(de novo; includes 3 events without a BBL)",
         disc & (ag["warm"] == 0)),
    ]


def build_table(ev: pd.DataFrame) -> pd.DataFrame:
    ag = ev[ev["agency"] == 1]
    n_agency = len(ag)
    rows = [metrics(ag, "total", "all_agency",
                    "all agency-initiated complaints 2020-01..2026-05",
                    n_agency)]

    main = slice_masks(ag)
    for name, desc, mask in main:
        rows.append(metrics(ag[mask], "main", name, desc, n_agency))
    assert sum(int(m.sum()) for _, _, m in main) == n_agency, \
        "main partition does not sum to the agency total"

    # family detail: warm/cold inside each pooled discretionary family
    for fam in DISCRETIONARY_FAMILIES:
        for warm, tag in ((1, "warm"), (0, "cold")):
            sub = ag[(ag["family"] == fam) & (ag["warm"] == warm)]
            rows.append(metrics(sub, "family_detail", f"{fam}_{tag}",
                                f"{fam} family, {tag} building", n_agency))

    # prefix detail: biggest discretionary programs, one row each
    disc = ag[ag["family"].isin(DISCRETIONARY_FAMILIES)]
    for pfx in disc["category_prefix"].value_counts().head(TOP_PREFIXES).index:
        rows.append(metrics(disc[disc["category_prefix"] == pfx],
                            "prefix_detail", f"prefix_{pfx}",
                            "discretionary-pool category prefix", n_agency))

    # sensitivity: full 730-day lookback for every focal event
    ag22 = ag[ag["received"] >= FULL_LOOKBACK_START]
    n22 = len(ag22)
    for name, desc, _ in main:
        _, _, mask22 = [m for m in slice_masks(ag22) if m[0] == name][0]
        rows.append(metrics(ag22[mask22], "sensitivity_2022plus", name,
                            f"{desc}; received 2022-01+, share of the "
                            f"{n22:,} agency events in that window", n22))

    return pd.DataFrame(rows)


def make_figure(table: pd.DataFrame) -> None:
    """Single stacked horizontal bar: agency work decomposed, hit rates
    printed on the segments (desc_ house style)."""
    main = table[table["group"] == "main"].set_index("slice")
    total = table.loc[table["group"] == "total", "n"].iat[0]
    segs = [
        ("statutory_periodic", "Statutory & periodic programs", BASE),
        ("followup_family", "Enforcement follow-up", MUTED),
        ("discretionary_warm", "Discretionary · warm building", BLUE),
        ("discretionary_cold", "Discretionary · DOB-cold (de novo)", RED),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 4.0), dpi=160)

    def clamped(cx):
        """Keep edge labels inside the axes (anchor at the border)."""
        if cx > 84:
            return 100.0, "right"
        if cx < 4:
            return 0.0, "left"
        return cx, "center"

    left = 0.0
    for i, (key, label, color) in enumerate(segs):
        r = main.loc[key]
        w = r["share_of_agency"] * 100
        ax.barh(0, w, left=left, height=0.52, color=color,
                edgecolor=SURFACE, linewidth=2)
        cx = left + w / 2
        # share, printed inside the segment
        ink_on = SURFACE if color in (BLUE, RED, MUTED) else INK2
        ax.text(cx, 0, f"{w:.0f}%", ha="center", va="center",
                fontsize=14, fontweight="bold", color=ink_on)
        # name + count above (two alternating levels, leader for the far one)
        y_name = 0.42 if i % 2 == 0 else 0.68
        if i % 2 == 1:
            ax.plot([cx, cx], [0.30, y_name - 0.09], color=MUTED,
                    linewidth=0.8)
        x, ha = clamped(cx)
        ax.text(x, y_name, f"{label} · {int(r['n']):,}", ha=ha,
                fontsize=11.5, color=INK)
        # hit rates below the segment
        ax.text(x, -0.46,
                f"violation {r['violation_disposition_rate'] * 100:.0f}% · "
                f"ECB {r['ecb_hit_rate'] * 100:.0f}%",
                ha=ha, fontsize=10.5, color=INK2)
        left += w

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.62, 0.88)
    ax.xaxis.set_visible(False)
    ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    fig.suptitle("What sends a DOB inspector when nobody called",
                 x=0.012, ha="left", fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.012, 0.885, f"{total:,} agency-initiated complaints, "
             "2020–May 2026 · warm = any DOB complaint at the same tax lot "
             "in the prior 730 days", fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(RISK_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    ev = load_events()
    ev = add_warm_flag(ev)
    table = build_table(ev)
    table.to_csv(OUT_CSV, index=False)
    make_figure(table)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n== proactive_decomposition ==")
        print(table.drop(columns="description").to_string(index=False))
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
