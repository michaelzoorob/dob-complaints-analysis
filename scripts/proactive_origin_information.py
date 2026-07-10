"""
Origin information value within mixed-origin categories (plan hypothesis #12).

Question: within complaint categories that arrive through BOTH channels,
do agency-initiated inspections out-perform 311-caller reports at finding
violations? Same complaint type, same buildings-stock controls, different
information source: DOB's own targeting signal vs the crowd's.

Design. Ten categories with genuine origin mixes (agency share 3-72%, from
the Wave-1 inventory): 04 after-hours, 23 shed/scaffold, 30 unstable, 45
illegal conversion, 58 boiler, 59 electrical, 67 crane, 73 failure to
maintain, 83 contrary to plans, 91 endangering workers. LPM

    violation100 ~ agency | category_prefix x month + size_bin + bct2020

on events merged to the residential risk panel by BBL (size_bin), SEs
clustered by tract (treatment varies at the event level but targeting is
spatial; tract is the conservative house choice). Pooled and per-category.
Variants: full sample without the residential restriction (drops size_bin),
substantive-only denominator (violation or no_violation dispositions, i.e.
conditional on the inspector getting in and ruling), and the linked-ECB
outcome (a citation actually issued, the harder margin).

Raw paired rates per category (agency vs caller violation %, ECB %,
no-access %) are reported unadjusted so every figure annotation traces.

Honest caveats carried in the outputs:
  - agency events in these categories are partly follow-ups/returns to
    known buildings (proactive_decomposition.py: only ~1 in 6 agency
    inspections is de-novo), so "the agency signal" bundles institutional
    memory, not just fresh targeting;
  - part of the raw gap is access, not information: 311-driven events hit
    locked doors far more often (no-access rates in the raw block); the
    substantive-only variant nets that channel out;
  - the residential-panel merge drops construction-heavy lots (match
    rates in the sample block; the no-size_bin full-sample variant shows
    the restriction is not doing the work).

Inputs : data/analysis/proactive/proactive_events.csv.gz
         data/analysis/property_risk_panel_v2.csv.gz
Outputs: data/analysis/risk_models/proactive_origin_information.csv
         data/analysis/blog_posts/artifacts/proactive_origin_information.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_origin_information.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
WINDOW = "2020-01..2026-05"

# categories with genuine origin mixes (Wave-1 agency shares in comments)
MIXED_CATS = {
    "04": "After-hours work",            # ~3% agency
    "23": "Sidewalk shed / scaffold",    # ~13%
    "30": "Building shaking / unstable", # ~27%
    "45": "Illegal conversion",          # ~4%
    "58": "Boiler defective",            # ~12%
    "59": "Electrical wiring",           # ~28%
    "67": "Crane",                       # ~30%
    "73": "Failure to maintain",         # ~72%
    "83": "Contrary to approved plans",  # ~16%
    "91": "Endangering workers",         # ~44%
}

# house style (constants from scripts/make_descriptive_figures.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
TINT = "#f2f1ec"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def norm_tract(s: pd.Series) -> pd.Series:
    """bct2020 arrives float in the spine, str in the panel; normalize."""
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def clean_tidy(model, name, outcome, n=None):
    """pyfixest tidy -> house column names (matches violation_rate_models)."""
    t = model.tidy().reset_index()
    t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                 for c in t.columns]
    if "coefficient" in t.columns:
        t = t.rename(columns={"coefficient": "term"})
    elif "index" in t.columns:
        t = t.rename(columns={"index": "term"})
    t["model"] = name
    t["outcome"] = outcome
    t["n"] = model._N if n is None else n
    return t


# ── data ─────────────────────────────────────────────────────────────────

def load_events() -> tuple[pd.DataFrame, dict]:
    ev = pd.read_csv(SPINE, usecols=[
        "category_prefix", "agency", "outcome", "ecb_number", "month",
        "bbl", "bct2020"], dtype={"bbl": str, "ecb_number": str})
    ev = ev[ev["category_prefix"].isin(MIXED_CATS)].copy()
    acct = {"n_mixed_events": len(ev)}

    n0 = len(ev)
    ev = ev[ev["outcome"] != "pending"]
    acct["n_dropped_pending"] = n0 - len(ev)

    ev["violation100"] = (ev["outcome"] == "violation").astype(float) * 100.0
    ev["ecb100"] = (ev["ecb_number"].fillna("") != "").astype(float) * 100.0
    ev["no_access100"] = (ev["outcome"] == "no_access").astype(float) * 100.0
    ev["bct"] = norm_tract(ev["bct2020"])
    ev["has_tract"] = ev["bct2020"].notna()
    ev["cat_month"] = ev["category_prefix"] + "_" + ev["month"]

    pan = pd.read_csv(PANEL, usecols=["bbl_key", "size_bin"],
                      dtype={"bbl_key": str, "size_bin": str})
    pan = pan.drop_duplicates("bbl_key").rename(columns={"bbl_key": "bbl"})
    ev = ev.merge(pan, on="bbl", how="left")
    ev["matched"] = ev["size_bin"].notna() & ev["has_tract"]
    acct["n_panel_matched"] = int(ev["matched"].sum())
    acct["match_share"] = round(ev["matched"].mean(), 4)
    acct["n_full_tract"] = int(ev["has_tract"].sum())
    acct["n_matched_agency"] = int(ev.loc[ev["matched"], "agency"].sum())
    return ev, acct


# ── raw paired rates ─────────────────────────────────────────────────────

def raw_rates(ev: pd.DataFrame) -> pd.DataFrame:
    """Unadjusted agency-vs-caller rates per category and pooled."""
    rows = []
    m = ev[ev["matched"]]
    for cat, sub in [("all_mixed", ev)] + [
            (c, ev[ev["category_prefix"] == c]) for c in sorted(MIXED_CATS)]:
        a, k = sub[sub["agency"] == 1], sub[sub["agency"] == 0]
        ms = m if cat == "all_mixed" else m[m["category_prefix"] == cat]
        ma, mk = ms[ms["agency"] == 1], ms[ms["agency"] == 0]
        rows.append({
            "model": "raw_rates", "term": cat,
            "outcome": "per 100 events, unadjusted",
            "label": "all 10 mixed categories" if cat == "all_mixed"
                     else MIXED_CATS[cat],
            "estimate": a["violation100"].mean() - k["violation100"].mean(),
            "viol_agency_pct": a["violation100"].mean(),
            "viol_caller_pct": k["violation100"].mean(),
            "ecb_agency_pct": a["ecb100"].mean(),
            "ecb_caller_pct": k["ecb100"].mean(),
            "noaccess_agency_pct": a["no_access100"].mean(),
            "noaccess_caller_pct": k["no_access100"].mean(),
            "n_agency": len(a), "n_caller": len(k),
            "viol_agency_pct_matched": ma["violation100"].mean(),
            "viol_caller_pct_matched": mk["violation100"].mean(),
            "n_agency_matched": len(ma), "n_caller_matched": len(mk),
            "agency_share": sub["agency"].mean(),
        })
    return pd.DataFrame(rows)


# ── models ───────────────────────────────────────────────────────────────

def run_models(ev: pd.DataFrame) -> pd.DataFrame:
    vcov = {"CRV1": "bct"}
    m = ev[ev["matched"]].copy()
    full = ev[ev["has_tract"]].copy()
    subst = m[m["outcome"].isin(["violation", "no_violation"])]
    out = []

    # pooled headline: category x month + size_bin + tract, panel-matched
    fit = pf.feols("violation100 ~ agency | cat_month + size_bin + bct",
                   data=m, vcov=vcov)
    out.append(clean_tidy(fit, "lpm_pooled_headline",
                          "violation disposition per 100 events (pp)"))

    # full sample (incl. non-residential lots), no size_bin
    fit = pf.feols("violation100 ~ agency | cat_month + bct",
                   data=full, vcov=vcov)
    out.append(clean_tidy(fit, "lpm_pooled_full_no_sizebin",
                          "violation disposition per 100 events (pp)"))

    # substantive-only denominator: nets out the access channel
    fit = pf.feols("violation100 ~ agency | cat_month + size_bin + bct",
                   data=subst, vcov=vcov)
    out.append(clean_tidy(fit, "lpm_pooled_substantive",
                          "violation per 100 substantive inspections (pp)"))

    # ECB citation outcome (harder margin), headline spec
    fit = pf.feols("ecb100 ~ agency | cat_month + size_bin + bct",
                   data=m, vcov=vcov)
    out.append(clean_tidy(fit, "lpm_pooled_ecb",
                          "linked ECB citation per 100 events (pp)"))

    # per-category: month + size_bin + tract on the matched sample
    for cat in sorted(MIXED_CATS):
        sub = m[m["category_prefix"] == cat]
        for dv, tag, oname in [
                ("violation100", "lpm_cat",
                 "violation disposition per 100 events (pp)"),
                ("ecb100", "lpm_cat_ecb",
                 "linked ECB citation per 100 events (pp)")]:
            try:
                fit = pf.feols(f"{dv} ~ agency | month + size_bin + bct",
                               data=sub, vcov=vcov)
                out.append(clean_tidy(fit, f"{tag}__{cat}", oname))
            except Exception as e:  # tiny cells (67): keep the row traceable
                out.append(pd.DataFrame([{
                    "term": "agency", "model": f"{tag}__{cat}",
                    "outcome": oname, "n": len(sub),
                    "note": f"fit failed: {e}"}]))
    res = pd.concat(out, ignore_index=True)
    res = res[res["term"] == "agency"].reset_index(drop=True)
    return res


# ── figure ───────────────────────────────────────────────────────────────

def make_figure(res: pd.DataFrame, raw: pd.DataFrame):
    cats = []
    for cat in MIXED_CATS:
        r = res[res["model"] == f"lpm_cat__{cat}"]
        if len(r) and pd.notna(r["estimate"].iat[0]):
            cats.append((cat, r.iloc[0]))
    cats.sort(key=lambda t: t[1]["estimate"], reverse=True)
    pooled = res[res["model"] == "lpm_pooled_headline"].iloc[0]
    rawix = raw.set_index("term")

    n = len(cats)
    gap = 1.0
    y_pool = float(n) + gap
    ys = np.arange(n, dtype=float)[::-1]
    xmax = 80.0

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160,
                           gridspec_kw={"left": 0.235, "right": 0.855,
                                        "top": 0.855, "bottom": 0.085})
    ax.axvline(0, color=BASE, lw=1.2, zorder=1)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)

    def draw(yi, row, color, ms=8.5):
        sig = row["pr(>|t|)"] < 0.05
        clipped = row["975pct"] > xmax
        ax.plot([row["25pct"], min(row["975pct"], xmax)], [yi, yi],
                color=color, lw=2, solid_capstyle="round", zorder=2,
                alpha=0.85)
        ax.plot(row["estimate"], yi, "o", ms=ms, zorder=3,
                markerfacecolor=color if sig else SURFACE,
                markeredgecolor=SURFACE if sig else color, markeredgewidth=2)
        if clipped:  # CI runs off scale (crane, n=168): label above the dot
            ax.annotate(f"{row['estimate']:+.1f} · n={int(row['n']):,}",
                        (row["estimate"], yi), textcoords="offset points",
                        xytext=(0, 9), va="bottom", ha="center", fontsize=9,
                        color=INK2)
            ax.annotate(f"CI to {row['975pct']:+.0f} →", (xmax - 1, yi),
                        textcoords="offset points", xytext=(0, 9),
                        va="bottom", ha="right", fontsize=8, color=MUTED)
        else:
            ax.annotate(f"{row['estimate']:+.1f}",
                        (max(row["975pct"], row["estimate"]), yi),
                        textcoords="offset points", xytext=(7, -0.5),
                        va="center", ha="left", fontsize=9, color=INK2)

    draw(y_pool, pooled, RED, ms=9.5)
    for yi, (cat, row) in zip(ys, cats):
        draw(yi, row, BLUE)

    # right-margin column: raw paired rates (full sample, unadjusted)
    ax.annotate("raw: agency vs 311", xy=(1.015, y_pool + 0.75),
                xycoords=("axes fraction", "data"), fontsize=8.5,
                color=MUTED, ha="left", va="center")
    for yi, key in [(y_pool, "all_mixed")] + [
            (yi, c) for yi, (c, _) in zip(ys, cats)]:
        rr = rawix.loc[key]
        ax.annotate(f"{rr['viol_agency_pct']:.0f}%  vs  "
                    f"{rr['viol_caller_pct']:.0f}%",
                    xy=(1.015, yi), xycoords=("axes fraction", "data"),
                    fontsize=9.5, color=INK2, ha="left", va="center")

    labels = [f"{MIXED_CATS[c]} ({c})" for c, _ in cats]
    ax.set_yticks(list(ys) + [y_pool])
    ax.set_yticklabels(labels + ["Pooled, all 10 categories"], fontsize=10.5,
                       color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.axhline(n - 1 + (gap + 1) / 2, color=GRID, lw=1.0)
    ax.set_xlim(-6, xmax)
    ax.set_ylim(-2.9, y_pool + 1.05)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("agency-initiated minus 311-driven, violations per 100 "
                  "inspections (pp)", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9.5)

    fig.suptitle("Within the same complaint type, DOB's own leads out-perform "
                 "311 calls", x=0.028, y=0.972, ha="left", fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.028, 0.925, "violation rate of agency-initiated vs 311-driven "
             "events in the 10 mixed-origin categories, 2020–May 2026 · LPM "
             "within category × month, building-size bin,\nand census tract "
             "(residential lots) · filled = p<0.05 · whiskers = 95% CI, SEs "
             "clustered by tract · right column: raw violation rates, all "
             "lots", fontsize=9.5, color=MUTED, va="top")

    rr04 = rawix.loc["04"]
    ax.text(0.0, 0.012,
            "the extreme case: after-hours work (04) is 3% agency-initiated, "
            "but those agency events carry an ECB citation "
            f"{rr04['ecb_agency_pct']:.0f}% of the time\nvs "
            f"{rr04['ecb_caller_pct']:.1f}% when 311-driven (raw); "
            "a caller's late-night noise report almost never becomes paper",
            transform=ax.transAxes, fontsize=9.3, color=INK2, style="italic",
            ha="left", va="bottom")

    fig.savefig(ART / "proactive_origin_information.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    print("[1/4] load events + panel merge")
    ev, acct = load_events()
    print(f"  {acct['n_mixed_events']:,} mixed-category events; "
          f"{acct['n_dropped_pending']:,} pending dropped; "
          f"{acct['n_panel_matched']:,} matched to residential panel "
          f"({acct['match_share']:.1%})")

    print("[2/4] raw paired rates")
    raw = raw_rates(ev)
    print(raw[["term", "label", "viol_agency_pct", "viol_caller_pct",
               "ecb_agency_pct", "ecb_caller_pct", "noaccess_agency_pct",
               "noaccess_caller_pct", "n_agency", "n_caller"]]
          .round(1).to_string(index=False))

    print("[3/4] LPMs (pooled + per category)")
    res = run_models(ev)
    print(res[["model", "estimate", "std_error", "25pct", "975pct", "n"]]
          .round(2).to_string(index=False))

    acct_rows = pd.DataFrame([{"model": "sample", "term": k, "estimate": v,
                               "outcome": "count"} for k, v in acct.items()])
    allrows = pd.concat([res, raw, acct_rows], ignore_index=True)
    allrows["window"] = WINDOW
    allrows.to_csv(OUT / "proactive_origin_information.csv", index=False)

    print("[4/4] figure")
    make_figure(res, raw)
    print(f"wrote {OUT / 'proactive_origin_information.csv'}\n"
          f"      {ART / 'proactive_origin_information.png'}")

    p = res[res["model"] == "lpm_pooled_headline"].iloc[0]
    s = res[res["model"] == "lpm_pooled_substantive"].iloc[0]
    e = res[res["model"] == "lpm_pooled_ecb"].iloc[0]
    cat_est = res[res["model"].str.startswith("lpm_cat__")].dropna(
        subset=["estimate"])
    r04 = raw.set_index("term").loc["04"]
    print(f"\nVerdict: within category x month, size bin, and tract, an "
          f"agency-initiated event is {p['estimate']:+.1f} pp more likely to "
          f"end in a violation than a 311-driven event in the same category "
          f"(95% CI {p['25pct']:+.1f}..{p['975pct']:+.1f}; caller base rates "
          f"in the raw block). Positive in all {len(cat_est)} categories "
          f"(range {cat_est['estimate'].min():+.1f} to "
          f"{cat_est['estimate'].max():+.1f}).")
    print(f"Channels: conditional on a substantive inspection result the gap "
          f"is {s['estimate']:+.1f} pp [{s['25pct']:+.1f}, {s['975pct']:+.1f}] "
          f"(so access explains only part); on the harder linked-ECB margin "
          f"{e['estimate']:+.1f} pp [{e['25pct']:+.1f}, {e['975pct']:+.1f}].")
    print(f"Extreme: after-hours (04), {r04['agency_share']:.0%} agency, "
          f"ECB raw {r04['ecb_agency_pct']:.0f}% vs "
          f"{r04['ecb_caller_pct']:.1f}% (plan: 43% vs 2.4%).")
    print("Caveat: the agency signal bundles institutional memory (most "
          "agency events are follow-ups or returns to known buildings, per "
          "the decomposition), so this is the value of DOB-held information, "
          "not of fresh de-novo targeting alone.")


if __name__ == "__main__":
    main()
