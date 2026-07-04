# NYC Building Complaints, Inspections, and Owner Race: Replication Package

Data and code for a series of analyses of New York City's complaint-driven
building code enforcement, including the study of predicted-Asian-owned homes
described in the accompanying blog post. The pipeline joins ~775,000 DOB
complaint records (NYC Open Data) to the detailed narratives and outcomes on
the city's BIS portal, links every complaint to a PLUTO tax lot, predicts
owner race from surnames and geography (BISG), and estimates enforcement gaps
within census tract and building-size class.

## What can be reproduced, at three tiers

**Tier 1 — figures (minutes).** Every figure in the posts regenerates from the
tidy estimate tables included in `data/analysis/risk_models/`:

```bash
pip install pandas numpy matplotlib
python scripts/make_asian_figures.py
python scripts/make_subgroup_figures.py
python scripts/make_text_figures.py
python scripts/make_transition_figure.py
python scripts/make_risk_figures.py
```

Output PNGs land in `data/analysis/blog_posts/artifacts/` (created if absent).

**Tier 2 — model re-estimation (hours).** The property-level panel
(`data/analysis/property_risk_panel_v2.csv.gz`, 460K individually owned
properties with BISG probabilities, outcomes, and covariates) is included, so
the cross-sectional models re-estimate directly:

```bash
pip install pyfixest surgeo
python scripts/owner_models.py               # headline BISG complaint models
python scripts/violation_rate_models.py      # violation ledgers (ECB, DOB, dispositions)
python scripts/asian_effect_heterogeneity.py # categories, per-inspection margins
python scripts/category_citations.py         # disposition violations by category
python scripts/asian_subgroups.py            # Asian-origin subgroup classification + models
python scripts/asian_article_stats.py        # raw rates, bounds, marginal arithmetic
python scripts/complaint_origin_models.py    # citizen vs agency channel
python scripts/agency_referral_antecedents.py# cross-agency 311 antecedents (uses cached pulls)
python scripts/text_analysis_race.py         # complaint/report text features + word log-odds
python scripts/owner_transition_panel.py     # ownership-change event studies + Wald tests
```

Scripts that touch raw complaint text or ACRIS deeds additionally need the
SQLite database (Tier 3). Each script's docstring states its inputs, design,
and outputs.

**Tier 3 — full rebuild from public sources (days).** The raw database
(`data/dob_complaints.db`, ~3M Open Data rows plus scraped BIS pages) is too
large for git. Rebuild it with:

```bash
python run_pipeline.py download   # Socrata bulk CSV -> SQLite
python run_pipeline.py scrape     # BIS complaint pages (resume-safe)
python scripts/download_pluto.py
python scripts/download_auxiliary_data.py
python scripts/download_acris_owner_addresses.py
python scripts/download_bin_bbl.py
python scripts/build_risk_dataset.py          # -> property_risk_panel_v2.csv.gz
python scripts/add_owner_characteristics.py   # BISG probabilities
```

## Data sources (all public)

- DOB Complaints Received, NYC Open Data (`eabe-havv`)
- BIS Web complaint detail pages (scraped; parser in `parser.py`)
- PLUTO / MapPLUTO (NYC Dept. of City Planning)
- ACRIS deeds and parties (NYC Dept. of Finance)
- ECB/OATH and DOB violation ledgers, NYC Open Data
- 311 Service Requests 2010-present (`erm2-nwe9`)
- ACS 5-year tables (tract demographics; tenure by race B25003;
  detailed Asian origin B02015; ancestry B04006)
- Census Bureau 2010 surname file (via the `surgeo` package)

## Layout

- `scripts/` — all analysis and figure code (docstrings document each)
- `config.py`, `run_pipeline.py`, `scraper.py`, `parser.py`, `db.py`,
  `download_open_data.py` — scrape/build pipeline
- `data/analysis/risk_models/` — tidy estimate tables (CSV) behind every figure
  and reported number
- `data/analysis/property_risk_panel_v2.csv.gz` — analysis panel (Tier 2)
- `data/analysis/owner_names_parsed.csv` — surnames/forenames parsed from
  public PLUTO owner names (input to the subgroup classification)
- `data/analysis/sr311_*.csv.gz` — cached 311 Service Request pulls used by the
  antecedent analysis

## Notes on the race measure

No property record lists an owner's race. BISG combines Census surname
probabilities with the racial composition of homeowners in the owner's ZIP
code; the result is a probability, not an identification, and it is derived
entirely from public records (PLUTO owner names, the Census surname file, and
ACS tables). Prediction error attenuates disparity estimates (McCartan et al.
2024, NBER w32373). The Asian-origin subgroup classification and its
validation against enclave geography are documented in
`scripts/asian_subgroups.py`.

## Requirements

Python 3.12+; `pandas`, `numpy`, `matplotlib`, `pyfixest`, `surgeo`,
`requests`. Estimation was run with pyfixest's Poisson (PPML) and OLS/LPM
with cluster-robust (CRV1) standard errors.
