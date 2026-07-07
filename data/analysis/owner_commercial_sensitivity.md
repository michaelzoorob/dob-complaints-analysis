# Ownership effects vs. commercial-exposure controls (five specifications)

Each ownership coefficient estimated under five specs; all use exact residential unit-count (`size_bin`) + census-tract (`bct2020`) fixed effects and tract-clustered SEs (the published owner-augmented spec).

| Spec | What it adds to the published spec |
|---|---|
| **1 current** | `BUILDING_COVARS + OWNER_COVARS`: binary `mixed_use` + `log2_area_per_unit` only (reproduces committed estimates). |
| **2 linear** | + `unitscom = max(unitstotal-unitsres,0)` (linear), `log(bldgarea)`, S/K/O class dummy. Full universe. |
| **2b comm-unit FE** | commercial units treated SYMMETRICALLY to residential: `comm_bin` (exact 0..10 then binned, mirroring `size_bin`) as FIXED EFFECTS + S/K/O class dummy; `mixed_use` dropped (subsumed). Full universe. |
| **2c +comarea** | + `log1p(comarea)`, true commercial floor area (PLUTO 64uk-42ks, full-universe pull). Captures commercial space even when the storefront is not a counted PLUTO unit. |
| **3 res-only** | spec-1 covariates on the sample `unitstotal==unitsres` AND `bldgclass in {A,B,C,D}`. |

Spec-1 reproduction: max |reproduced - published| = **0.0000** across all 15 owner coefficients (exact). Cells show the effect on the post's native scale - Poisson counts as incidence-rate change `(exp(b)-1)`, LPMs in percentage points `pp` - with significance `*`/`**`/`***` = p<.05/.01/.001. Coefficient SEs, p-values, and % movement vs spec 1 are in `owner_commercial_sensitivity.csv`.

**Spec 2c comarea join:** matched 762,799 of 766,429 lots (99.5%) to current PLUTO commercial floor area; the 3,630 unmatched lots are DROPPED from spec 2c (comarea not assumed 0). Among matched lots comarea>0 for 8.0%.

| Outcome (model) | Ownership | Spec 1<br>current | Spec 2<br>linear +comm | Spec 2b<br>comm-unit FE | Spec 2c<br>+comarea | Spec 3<br>res-only |
|---|---|---|---|---|---|---|
| Complaint count (Poisson) | LLC (vs individual) | +52.8%*** | +48.8%*** | +51.8%*** | +52.5%*** | +72.7%*** |
|  | Owner-occupied (STAR) | -0.5% | +1.7% | -0.1% | -0.4% | +3.6%* |
|  | Absentee (owner outside NYC) | +7.8%*** | +7.8%*** | +7.8%*** | +7.8%*** | +9.9%*** |
| Any complaint (LPM) | LLC (vs individual) | +8.9pp*** | +8.9pp*** | +9.0pp*** | +8.9pp*** | +10.5pp*** |
|  | Owner-occupied (STAR) | -0.4pp* | -0.4pp* | -0.4pp** | -0.4pp* | -0.4pp* |
|  | Absentee (owner outside NYC) | +1.6pp*** | +1.6pp*** | +1.6pp*** | +1.6pp*** | +1.8pp*** |
| DOB-ledger violation count (Poisson) | LLC (vs individual) | +37.2%*** | +18.3%*** | +21.7%*** | +37.1%*** | +5.2%** |
|  | Owner-occupied (STAR) | -21.9%*** | -11.2%*** | -14.5%*** | -21.6%*** | -8.1%*** |
|  | Absentee (owner outside NYC) | +6.9%*** | +5.7%*** | +6.2%*** | +7.3%*** | +5.2%** |
| Violations / substantive inspection (LPM, wtd) | LLC (vs individual) | +1.2pp*** | +1.1pp*** | +1.0pp** | +1.1pp*** | +1.1pp** |
|  | Owner-occupied (STAR) | -2.8pp*** | -2.8pp*** | -2.7pp*** | -2.8pp*** | -2.5pp*** |
|  | Absentee (owner outside NYC) | +0.4pp | +0.4pp | +0.4pp | +0.4pp | +0.2pp |
| ECB citation count (Poisson) | LLC (vs individual) | +84.3%*** | +77.9%*** | +81.5%*** | +83.2%*** | +104.5%*** |
|  | Owner-occupied (STAR) | -1.3% | +1.7% | -0.5% | -1.1% | +2.9% |
|  | Absentee (owner outside NYC) | +11.7%*** | +11.5%*** | +11.8%*** | +12.2%*** | +15.6%*** |

### DOB-ledger violation count - coefficient detail (the confounded outcome)

Raw Poisson coefficient `b (se)` [IRR%], LLC and owner-occupied, across the five specs.

| Ownership | Spec 1 current | Spec 2 linear +comm | Spec 2b comm-unit FE | Spec 2c +comarea | Spec 3 res-only |
|---|---|---|---|---|---|
| LLC (vs individual) | +0.316 (0.016)*** [+37%] | +0.168 (0.015)*** [+18%] | +0.197 (0.015)*** [+22%] | +0.316 (0.016)*** [+37%] | +0.051 (0.018)** [+5%] |
| Owner-occupied (STAR) | -0.247 (0.018)*** [-22%] | -0.119 (0.017)*** [-11%] | -0.156 (0.018)*** [-14%] | -0.244 (0.018)*** [-22%] | -0.085 (0.016)*** [-8%] |

### What drives the DOB-ledger attenuation? (mechanism)

The DOB-ledger count is the only sensitive outcome. Adding each commercial control ALONE to Spec 1 shows the operative confounder is the commercial/mixed-use building **class** (S/K/O) and total **floor area** - NOT commercial units or commercial floor area, which move the coefficient ~0.

| Added to Spec 1 (alone) | DOB-ledger LLC | DOB-ledger owner-occupied |
|---|---|---|
| Spec 1 (baseline) | +37.2%*** | -21.9%*** |
| + S/K/O class dummy | +21.4%*** | -14.3%*** |
| + log(bldgarea) | +28.7%*** | -16.9%*** |
| + unitscom (linear) | +37.1%*** | -22.0%*** |
| + log1p(comarea) | +37.1%*** | -21.6%*** |

`2b+` = comm_bin FE + S/K/O class + `log(bldgarea)` (the parallel-FE spec plus total floor area) across all outcomes:

| Outcome | LLC | Owner-occupied | Absentee |
|---|---|---|---|
| Complaint count | +48.4%*** | +1.8% | +7.8%*** |
| Any complaint | +8.9pp*** | -0.4pp* | +1.6pp*** |
| DOB-ledger violation count | +17.2%*** | -10.6%*** | +5.5%*** |
| Violations / substantive inspection | +1.1pp*** | -2.8pp*** | +0.4pp |
| ECB citation count | +77.4%*** | +1.8% | +11.5%*** |

**Sample sizes (N):**

| Outcome | Spec 1 current | Spec 2 linear +comm | Spec 2b comm-unit FE | Spec 2c +comarea | Spec 3 res-only |
|---|---:|---:|---:|---:|---:|
| Complaint count | 766,382 | 766,382 | 766,382 | 762,752 | 701,372 |
| Any complaint | 766,400 | 766,400 | 766,400 | 762,770 | 701,394 |
| DOB-ledger violation count | 765,821 | 765,821 | 765,821 | 762,195 | 700,781 |
| Violations / substantive inspection | 134,791 | 134,791 | 134,791 | 134,112 | 105,066 |
| ECB citation count | 766,376 | 766,376 | 766,376 | 762,746 | 701,343 |
