"""
Is the 2025-26 discretionary yield turn real targeting improvement or a
mix shift? (LL79 question; extends proactive_yearly_yield.py, which shows
discretionary violation yield 30.7% in 2024 -> 41.5% in 2025 -> 43.8% in
Jan-May 2026.)

Four exercises on agency-initiated discretionary_field events:

(1) Shift-share decomposition of the 2024->2025 and 2024->2026 (Jan-May)
    yield change over category prefixes: symmetric (average-weight)
    within-category vs composition terms, exact by construction; a
    Laspeyres (base-2024-weight) variant and a seasonally matched
    2024 Jan-May -> 2026 Jan-May comparison as robustness.

(2) Within-category yield by year: pooled LPM hit100 ~ year dummies
    (ref 2024) | category_prefix, SEs clustered by tract (the treatment,
    "which year the inspection happened," varies at the visit level, but
    visits cluster spatially; tract is the conservative house level).
    Per-program year effects and raw tract-clustered yearly means for the
    big programs (8A, 7G, 1X, 91, 6X).

(3) Sharpest LL79 test: score every panel lot with the SAME pre-2020
    logit as proactive_becker_margin.py (imported, not re-implemented:
    era, size bins, log floor area, class letter, 2010-19 violation
    history, tract poverty; outcome any-violation-2020on). Mean predicted
    risk (and panel risk percentile) of discretionary-visited buildings
    by half-year 2020H1..2026H1, tract-clustered CIs. If LL79's
    risk-based targeting is real, visited-building scores should move up
    in 2025-26.

(4) De-novo (cold) share of discretionary work by year, reusing the
    730-day warm/cold searchsorted machinery imported from
    proactive_decomposition.py (warm = any DOB complaint at the BBL in
    the prior 730 days). Cold visits out-yield warm ones (45.4 vs 39.4 in
    A1), so a cold shift is itself a targeting channel.

Honest caveats carried in the outputs:
  - 2026 is Jan-May only; the seasonally matched comparison shows 2024
    Jan-May yield (27.8%) was BELOW the 2024 full-year figure, so
    seasonality understates rather than explains the turn;
  - risk scores exist only for lots in the residential risk panel; the
    matched share per half-year is reported so panel-selection drift is
    visible (7G/8A construction sites often sit on non-residential lots);
  - the risk model's outcome is realized enforcement, not latent risk
    (proactive_becker_margin.py caveats apply); scores are frozen
    pre-2020-features predictions, so cross-period comparisons are
    apples to apples;
  - cold shares before 2022 are overstated (lookback truncated at the
    2020-01 spine edge); rows are flagged, and the 2024-26 turn sits
    entirely in complete-lookback years;
  - pending dispositions are ~0 in every year (max 0.02% in 2025), so
    the turn is not a resolution-lag artifact; outcome-share guard rows
    are included.

Inputs : data/analysis/proactive/proactive_events.csv.gz
         data/analysis/property_risk_panel_v2.csv.gz (via becker import)
         data/dob_complaints.db (dob_ledger union, via becker import)
Outputs: data/analysis/risk_models/proactive_yield_turn.csv
         data/analysis/blog_posts/artifacts/proactive_yield_turn.png

Run: /private/tmp/pyfix_venv/bin/python scripts/proactive_yield_turn.py
"""

import re
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
from proactive_becker_margin import (clean_tidy, cluster_mean,
                                     load_panel_scored, norm_tract)
from proactive_decomposition import add_warm_flag

SPINE = config.DATA_DIR / "analysis" / "proactive" / "proactive_events.csv.gz"
OUT = config.DATA_DIR / "analysis" / "risk_models"
ART = config.DATA_DIR / "analysis" / "blog_posts" / "artifacts"
WINDOW = "2020-01..2026-05"
BIG = ["8A", "7G", "1X", "91", "6X"]
YEARS = [str(y) for y in range(2020, 2027)]

# house style (constants from scripts/make_descriptive_figures.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
AQUA = "#1baf7a"
TINT = "#f2f1ec"

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
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def row(block, model, term, estimate, std_error=np.nan, lo=np.nan, hi=np.nan,
        n=np.nan, note=""):
    return {"block": block, "model": model, "term": term,
            "estimate": float(estimate), "std_error": std_error,
            "25pct": lo, "975pct": hi, "n": n, "note": note}


# ── load: discretionary pool with warm flag ──────────────────────────────

def load_disc() -> pd.DataFrame:
    ev = pd.read_csv(SPINE, usecols=[
        "received_date", "category_prefix", "family", "agency", "bbl",
        "bct2020", "outcome"], dtype={"bbl": str})
    ev["received"] = pd.to_datetime(ev["received_date"], format="%Y-%m-%d")
    ev["bbl"] = ev["bbl"].fillna("")
    # 730-day warm/cold flag from proactive_decomposition.py, computed on
    # the FULL spine pool (caller + agency) before filtering
    ev = add_warm_flag(ev)

    d = ev[(ev["agency"] == 1) & (ev["family"] == "discretionary_field")].copy()
    d["year"] = d["received_date"].str[:4]
    d["mm"] = d["received_date"].str[5:7].astype(int)
    d["half"] = d["year"] + np.where(d["mm"] <= 6, "H1", "H2")
    d["hit100"] = (d["outcome"] == "violation").astype(float) * 100.0
    d["bct"] = norm_tract(d["bct2020"])
    d.loc[d["bct2020"].isna(), "bct"] = np.nan
    print(f"discretionary_field agency events: {len(d):,} "
          f"({d['received_date'].min()}..{d['received_date'].max()}); "
          f"tract-located {d['bct'].notna().mean():.3f}")
    return d


# ── (0) yearly series + guard rows ───────────────────────────────────────

def yearly_block(d: pd.DataFrame) -> list:
    rows = []
    for y, sub in d.groupby("year"):
        loc = sub[sub["bct"].notna()]
        m, se, g = cluster_mean(loc["hit100"].to_numpy(), loc["bct"].to_numpy())
        est = sub["hit100"].mean()  # all rows: traces to proactive_yearly_yield.csv
        rows.append(row(
            "yearly_yield", "raw", f"yield_{y}", est, se,
            est - 1.96 * se, est + 1.96 * se, len(sub),
            f"violation yield (pp), all events; CI tract-clustered on the "
            f"{len(loc):,} tract-located events ({g} tracts)"))
        for oc in ["pending", "no_access", "other", "referral"]:
            rows.append(row(
                "yearly_yield", "outcome_shares", f"{oc}_share_{y}",
                (sub["outcome"] == oc).mean() * 100, n=len(sub),
                note="share of events (pp); guard against resolution artifacts"))
    # seasonal anchor
    jm24 = d[(d["year"] == "2024") & (d["mm"] <= 5)]
    rows.append(row("yearly_yield", "raw", "yield_2024_janmay",
                    jm24["hit100"].mean(), n=len(jm24),
                    note="2024 Jan-May only; seasonal anchor for the 2026 "
                         "comparison"))
    return rows


# ── (1) shift-share decomposition over category prefixes ────────────────

def shift_share(d: pd.DataFrame, base_mask, comp_mask, label: str) -> list:
    b, c = d[base_mask], d[comp_mask]
    yb, yc = b["hit100"].mean(), c["hit100"].mean()
    wb = b["category_prefix"].value_counts(normalize=True)
    wc = c["category_prefix"].value_counts(normalize=True)
    ymb = b.groupby("category_prefix")["hit100"].mean()
    ymc = c.groupby("category_prefix")["hit100"].mean()
    cats = sorted(set(wb.index) | set(wc.index))
    anchor = 0.5 * (yb + yc)  # centering constant; totals invariant

    per = {}
    for cat in cats:
        w0, w1 = wb.get(cat, 0.0), wc.get(cat, 0.0)
        # empty cells: impute the year's overall mean (exactness unaffected)
        y0, y1 = ymb.get(cat, yb), ymc.get(cat, yc)
        within = 0.5 * (w0 + w1) * (y1 - y0)
        comp = (w1 - w0) * (0.5 * (y0 + y1) - anchor)
        # Laspeyres variant: base weights / base yields + interaction
        within_l = w0 * (y1 - y0)
        comp_l = (w1 - w0) * (y0 - yb)
        inter_l = (w1 - w0) * (y1 - y0)
        per[cat] = (within, comp, within_l, comp_l, inter_l, w0, w1, y0, y1)

    tot = yc - yb
    w_sum = sum(v[0] for v in per.values())
    c_sum = sum(v[1] for v in per.values())
    assert np.isclose(w_sum + c_sum, tot, atol=1e-9), label
    wl_sum = sum(v[2] for v in per.values())
    cl_sum = sum(v[3] for v in per.values())
    il_sum = sum(v[4] for v in per.values())
    assert np.isclose(wl_sum + cl_sum + il_sum, tot, atol=1e-9), label

    rows = [
        row("decomposition", "symmetric", f"{label}__total_change_pp", tot,
            n=len(c), note=f"yield {yb:.2f} -> {yc:.2f} pp; n base {len(b):,}"),
        row("decomposition", "symmetric", f"{label}__within_total_pp", w_sum,
            note="sum of avg-weight x yield-change terms over prefixes"),
        row("decomposition", "symmetric", f"{label}__composition_total_pp",
            c_sum, note="sum of weight-change x (avg yield - overall) terms"),
        row("decomposition", "symmetric", f"{label}__within_share_of_change",
            w_sum / tot * 100, note="within_total / total_change (pct)"),
        row("decomposition", "laspeyres", f"{label}__within_total_pp", wl_sum,
            note="2024 weights"),
        row("decomposition", "laspeyres", f"{label}__composition_total_pp",
            cl_sum, note="weight change x base yield (centered at base mean)"),
        row("decomposition", "laspeyres", f"{label}__interaction_pp", il_sum),
    ]
    other_w = sum(v[0] for k, v in per.items() if k not in BIG)
    other_c = sum(v[1] for k, v in per.items() if k not in BIG)
    for cat in BIG:
        w, cmp_, *_, w0, w1, y0, y1 = per[cat]
        rows.append(row(
            "decomposition", "symmetric", f"{label}__within__{cat}", w,
            note=f"yield {y0:.1f}->{y1:.1f} pp at avg weight "
                 f"{(w0 + w1) / 2 * 100:.1f}%"))
        rows.append(row(
            "decomposition", "symmetric", f"{label}__composition__{cat}", cmp_,
            note=f"share {w0 * 100:.1f}%->{w1 * 100:.1f}%"))
    rows.append(row("decomposition", "symmetric",
                    f"{label}__within__other_prefixes", other_w,
                    note=f"{len(cats) - len(BIG)} smaller prefixes"))
    rows.append(row("decomposition", "symmetric",
                    f"{label}__composition__other_prefixes", other_c))
    return rows


# ── (2) within-category LPMs ─────────────────────────────────────────────

def year_term(t: str) -> str:
    m = re.search(r"(\d{4})", t)
    return m.group(1) if m else t


def lpm_block(d: pd.DataFrame) -> list:
    rows = []
    loc = d[d["bct"].notna()].copy()
    vcov = {"CRV1": "bct"}

    m = pf.feols("hit100 ~ i(year, ref='2024') | category_prefix",
                 data=loc, vcov=vcov)
    t = clean_tidy(m, "pooled_cat_fe", "yield vs 2024 (pp), within prefix")
    for r in t.to_dict("records"):
        rows.append(row("lpm_year_effects", "pooled_cat_fe",
                        year_term(r["term"]), r["estimate"], r["std_error"],
                        r["25pct"], r["975pct"], r["n"],
                        note="LPM year dummies | category_prefix, tract-clustered"))

    jm = loc[loc["mm"] <= 5]
    m = pf.feols("hit100 ~ i(year, ref='2024') | category_prefix",
                 data=jm, vcov=vcov)
    t = clean_tidy(m, "pooled_cat_fe_janmay", "Jan-May yield vs 2024 (pp)")
    for r in t.to_dict("records"):
        rows.append(row("lpm_year_effects", "pooled_cat_fe_janmay",
                        year_term(r["term"]), r["estimate"], r["std_error"],
                        r["25pct"], r["975pct"], r["n"],
                        note="Jan-May months only in every year (seasonal match)"))

    for pfx in BIG:
        sub = loc[loc["category_prefix"] == pfx]
        m = pf.feols("hit100 ~ i(year, ref='2024')", data=sub, vcov=vcov)
        t = clean_tidy(m, f"program_{pfx}", "yield vs 2024 (pp)")
        for r in t.to_dict("records"):
            if r["term"] == "Intercept":
                continue
            rows.append(row("lpm_year_effects", f"program_{pfx}",
                            year_term(r["term"]), r["estimate"],
                            r["std_error"], r["25pct"], r["975pct"], r["n"]))
        # raw tract-clustered yearly levels (figure panel C)
        for y, s in sub.groupby("year"):
            mean, se, g = cluster_mean(s["hit100"].to_numpy(),
                                       s["bct"].to_numpy())
            n_all = int(((d["category_prefix"] == pfx) & (d["year"] == y)).sum())
            rows.append(row("program_year_yield", f"raw_{pfx}", y, mean, se,
                            mean - 1.96 * se, mean + 1.96 * se, len(s),
                            note=f"tract-located events; {n_all:,} total, "
                                 f"{g} tracts"))
    return rows


# ── (3) risk scores of visited buildings (becker pre-2020 model) ────────

def risk_block(d: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    panel, auc, k = load_panel_scored()
    panel = panel.drop_duplicates("bbl_key")
    panel["risk_pctile"] = panel["p_hat"].rank(pct=True) * 100.0
    sc = d.merge(
        panel[["bbl_key", "p100", "risk_pctile", "bct2020"]].rename(
            columns={"bct2020": "bct_panel"}),
        left_on="bbl", right_on="bbl_key", how="left")

    rows = [row("risk_scores", "model", "auc", auc, n=len(panel),
                note="in-sample rank AUC of the imported becker pre-2020 logit"),
            row("risk_scores", "model", "n_params", k)]
    halves = sorted(sc["half"].unique())
    for h in halves:
        sub = sc[sc["half"] == h]
        mt = sub[sub["p100"].notna()]
        rows.append(row("risk_scores", "match", h,
                        len(mt) / len(sub) * 100, n=len(sub),
                        note="share of visits matched to a scored panel lot (pp)"))
        for col, name in [("p100", "mean_p100"),
                          ("risk_pctile", "mean_risk_pctile")]:
            m, se, g = cluster_mean(mt[col].to_numpy(),
                                    mt["bct_panel"].to_numpy())
            rows.append(row(
                "risk_scores", name, h, m, se, m - 1.96 * se, m + 1.96 * se,
                len(mt),
                note="visited panel lots, event-weighted; CI clustered on "
                     f"panel tract ({g} tracts)"))
    # supplementary: per-program yearly mean predicted risk
    for pfx in BIG:
        for y in YEARS:
            mt = sc[(sc["category_prefix"] == pfx) & (sc["year"] == y)
                    & sc["p100"].notna()]
            if len(mt) < 20:
                continue
            rows.append(row("risk_scores", f"mean_p100_{pfx}", y,
                            mt["p100"].mean(), n=len(mt)))
    return rows, sc


# ── (4) cold share by year ───────────────────────────────────────────────

def cold_block(d: pd.DataFrame) -> list:
    rows = []
    for y, sub in d.groupby("year"):
        flag = ("" if y >= "2022" else
                "; lookback truncated (spine starts 2020-01), cold overstated")
        rows.append(row("cold_share", "cold_share", y,
                        (1 - sub["warm"].mean()) * 100, n=len(sub),
                        note="share with no DOB complaint at the BBL in the "
                             "prior 730 days (pp)" + flag))
        for w, name in [(0, "cold_yield"), (1, "warm_yield")]:
            s = sub[sub["warm"] == w]
            rows.append(row("cold_share", name, y, s["hit100"].mean(),
                            n=len(s), note="violation yield (pp)" + flag))
    return rows


# ── figure ───────────────────────────────────────────────────────────────

def get(res, block, model, term):
    r = res[(res["block"] == block) & (res["model"] == model)
            & (res["term"] == term)]
    return r.iloc[0]


def make_figure(res: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), dpi=160)
    fig.subplots_adjust(hspace=0.52, wspace=0.24, top=0.86, bottom=0.07,
                        left=0.06, right=0.985)
    xs = np.arange(2020, 2027)

    # A: raw yearly yield vs within-category (pooled FE, anchored at 2024)
    ax = axes[0, 0]
    raw = [get(res, "yearly_yield", "raw", f"yield_{y}")["estimate"]
           for y in YEARS]
    anchor = raw[YEARS.index("2024")]
    adj, lo, hi = [], [], []
    for y in YEARS:
        if y == "2024":
            adj.append(anchor); lo.append(np.nan); hi.append(np.nan)
        else:
            r = get(res, "lpm_year_effects", "pooled_cat_fe", y)
            adj.append(anchor + r["estimate"])
            lo.append(anchor + r["25pct"]); hi.append(anchor + r["975pct"])
    ax.plot(xs, raw, color=MUTED, linewidth=2, marker="o", markersize=6,
            label="raw yield")
    ax.errorbar(xs, adj, yerr=[np.nan_to_num(np.array(adj) - np.array(lo)),
                               np.nan_to_num(np.array(hi) - np.array(adj))],
                color=BLUE, linewidth=2, marker="o", markersize=6,
                elinewidth=1.4, capsize=3,
                label="within category (year effects, anchored 2024)")
    style_ax(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in xs[:-1]] + ["2026\nJan–May"],
                       fontsize=9.5)
    ax.set_ylabel("violations per 100 inspections", fontsize=10.5)
    ax.set_ylim(20, 50)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_title("Yield rose within programs too",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=10)

    # B: shift-share stacked bars
    ax = axes[0, 1]
    labels = [("2024_to_2025", "2024 → 2025"),
              ("2024_to_2026jm", "2024 → 2026 (Jan–May)")]
    for i, (key, lab) in enumerate(labels):
        w = get(res, "decomposition", "symmetric",
                f"{key}__within_total_pp")["estimate"]
        c = get(res, "decomposition", "symmetric",
                f"{key}__composition_total_pp")["estimate"]
        y = 1 - i
        ax.barh(y, w, left=0, height=0.5, color=BLUE, edgecolor=SURFACE,
                linewidth=1.5)
        ax.barh(y, c, left=w, height=0.5, color=RED, edgecolor=SURFACE,
                linewidth=1.5)
        ax.text(w / 2, y, f"within\n+{w:.1f}pp", ha="center", va="center",
                fontsize=10, color=SURFACE, fontweight="bold")
        ax.text(w + c / 2, y, f"mix\n+{c:.1f}pp", ha="center", va="center",
                fontsize=10, color=SURFACE, fontweight="bold")
        ax.text(w + c + 0.25, y, f"+{w + c:.1f}pp total", va="center",
                fontsize=10.5, color=INK)
        ax.text(-0.25, y + 0.37, lab, fontsize=10.5, color=INK, ha="left")
    ax.set_xlim(-0.4, 18)
    ax.set_ylim(-0.55, 1.75)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("contribution to yield change (pp)", fontsize=10.5)
    ax.set_title("Where the jump came from",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=24)
    ax.text(0, 1.03, "shift-share over category prefixes · within = yield "
            "change at average program weights · mix = volume moving across "
            "programs", transform=ax.transAxes, fontsize=8.5, color=MUTED)

    # C: program-level yields
    ax = axes[1, 0]
    colors = {"8A": BLUE, "7G": RED, "1X": AQUA, "91": MUTED, "6X": BASE}
    for pfx in BIG:
        sub = res[(res["block"] == "program_year_yield")
                  & (res["model"] == f"raw_{pfx}")].sort_values("term")
        yy = sub["term"].astype(int).to_numpy()
        ax.errorbar(yy, sub["estimate"], yerr=1.96 * sub["std_error"],
                    color=colors[pfx], linewidth=1.8, marker="o",
                    markersize=4.5, elinewidth=1.0, capsize=2, alpha=0.95)
        last = sub.iloc[-1]
        ax.text(2026.15, last["estimate"], pfx, fontsize=10, color=colors[pfx],
                va="center", fontweight="bold")
    style_ax(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(y) for y in xs[:-1]] + ["2026\nJan–May"],
                       fontsize=9.5)
    ax.set_xlim(2019.6, 2026.8)
    ax.set_ylabel("violations per 100 inspections", fontsize=10.5)
    ax.set_title("The big programs, one by one",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=24)
    ax.text(0, 1.03, "8A compliance · 7G sweeps · 1X work orders · "
            "91 worker endangerment · 6X permit watch list · 95% CIs, "
            "tract-clustered", transform=ax.transAxes, fontsize=8.5,
            color=MUTED)

    # D: predicted risk of visited buildings + cold share
    ax = axes[1, 1]
    rs = res[(res["block"] == "risk_scores")
             & (res["model"] == "mean_p100")].sort_values("term")
    hx = [int(h[:4]) + (0.25 if h.endswith("H1") else 0.75)
          for h in rs["term"]]
    ax.fill_between(hx, rs["25pct"], rs["975pct"], color=BLUE, alpha=0.15,
                    linewidth=0)
    ax.plot(hx, rs["estimate"], color=BLUE, linewidth=2, marker="o",
            markersize=4.5)
    ax.text(hx[-1] + 0.12, rs["estimate"].iloc[-1],
            "predicted risk\nof visited lots", fontsize=9.5, color=BLUE,
            va="center")
    cs = res[(res["block"] == "cold_share")
             & (res["model"] == "cold_share")].sort_values("term")
    cs = cs[cs["term"] >= "2022"]  # full 730-day lookback only
    cx = cs["term"].astype(int) + 0.5
    ax.plot(cx, cs["estimate"], color=MUTED, linewidth=1.8,
            linestyle="--", marker="o", markersize=4)
    ax.text(cx.iloc[-1] + 0.12, cs["estimate"].iloc[-1],
            "de-novo (cold)\nshare of visits", fontsize=9.5, color=MUTED,
            va="center")
    style_ax(ax)
    ax.set_xticks(xs + 0.5)
    ax.set_xticklabels([str(y) for y in xs[:-1]] + ["2026\nJan–May"],
                       fontsize=9.5)
    ax.set_xlim(2020, 2027.3)
    ax.set_ylim(0, 70)
    ax.set_ylabel("percent", fontsize=10.5)
    ax.set_title("Did visits move up the risk distribution?",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=24)
    ax.text(0, 1.03, "mean pre-2020-model predicted any-violation risk of "
            "visited lots, half-yearly · 95% CIs · cold share from 2022 "
            "(full lookback)", transform=ax.transAxes, fontsize=8.5,
            color=MUTED)

    fig.suptitle("The 2025–26 discretionary yield jump, decomposed",
                 x=0.012, ha="left", fontsize=15.5, fontweight="bold",
                 color=INK)
    fig.text(0.012, 0.925,
             "agency-initiated discretionary inspections (8A/7G/1X/91/6X/…), "
             "2020–May 2026 · violation yield 30.7% (2024) → 41.5% (2025) → "
             "43.8% (Jan–May 2026)",
             fontsize=10.5, color=MUTED)
    fig.savefig(ART / "proactive_yield_turn.png", bbox_inches="tight")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    print("[1/5] load discretionary pool + warm flags")
    d = load_disc()
    rows = yearly_block(d)

    # cross-check against the published yearly series
    pub = pd.read_csv(OUT / "proactive_yearly_yield.csv",
                      dtype={"year": str}).set_index("year")
    for y in YEARS:
        mine = [r for r in rows if r["term"] == f"yield_{y}"][0]["estimate"]
        assert abs(mine / 100 - pub.loc[y, "violation_yield"]) < 5e-4, y
    print("  yearly yields match proactive_yearly_yield.csv")

    print("[2/5] shift-share decomposition over category prefixes")
    y24 = (d["year"] == "2024").to_numpy()
    y24jm = y24 & (d["mm"] <= 5).to_numpy()
    rows += shift_share(d, y24, (d["year"] == "2025").to_numpy(),
                        "2024_to_2025")
    rows += shift_share(d, y24, (d["year"] == "2026").to_numpy(),
                        "2024_to_2026jm")
    rows += shift_share(d, y24jm, (d["year"] == "2026").to_numpy(),
                        "2024jm_to_2026jm")

    print("[3/5] within-category LPMs (tract-clustered)")
    rows += lpm_block(d)

    print("[4/5] risk scores of visited buildings (imported becker model)")
    risk_rows, _ = risk_block(d)
    rows += risk_rows

    print("[5/5] cold share by year + figure")
    rows += cold_block(d)

    res = pd.DataFrame(rows)
    res["window"] = WINDOW
    res.to_csv(OUT / "proactive_yield_turn.csv", index=False)
    make_figure(res)

    with pd.option_context("display.width", 220, "display.max_rows", 300):
        for blk in ["decomposition", "lpm_year_effects", "risk_scores",
                    "cold_share"]:
            print(f"\n== {blk} ==")
            sub = res[res["block"] == blk]
            if blk == "risk_scores":
                sub = sub[sub["model"].isin(["mean_p100", "mean_risk_pctile",
                                             "match", "model"])]
            print(sub.drop(columns=["block", "window"]).round(2)
                  .to_string(index=False))
    print(f"\nwrote {OUT / 'proactive_yield_turn.csv'}")
    print(f"wrote {ART / 'proactive_yield_turn.png'}")

    # verdict
    w25 = get(res, "decomposition", "symmetric",
              "2024_to_2025__within_share_of_change")["estimate"]
    w26 = get(res, "decomposition", "symmetric",
              "2024_to_2026jm__within_share_of_change")["estimate"]
    e25 = get(res, "lpm_year_effects", "pooled_cat_fe", "2025")
    e26 = get(res, "lpm_year_effects", "pooled_cat_fe", "2026")
    r24 = get(res, "risk_scores", "mean_p100", "2024H2")["estimate"]
    r26 = get(res, "risk_scores", "mean_p100", "2026H1")["estimate"]
    print(f"\nVerdict: within-category share of the yield change "
          f"{w25:.0f}% (2024->2025), {w26:.0f}% (2024->2026 Jan-May); "
          f"pooled within-category year effect +{e25['estimate']:.1f}pp "
          f"[{e25['25pct']:.1f}, {e25['975pct']:.1f}] in 2025 and "
          f"+{e26['estimate']:.1f}pp [{e26['25pct']:.1f}, {e26['975pct']:.1f}] "
          f"in 2026; mean predicted risk of visited lots {r24:.1f} (2024H2) "
          f"-> {r26:.1f} (2026H1).")


if __name__ == "__main__":
    main()
