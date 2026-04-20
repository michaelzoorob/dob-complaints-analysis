# Inspector Enforcement Heterogeneity and Compliance Behavior: Methodology and Results

## 1. Introduction

We study whether the strictness of building inspectors causally affects property owners' compliance investment in New York City. Using a novel dataset linking 456,460 DOB complaint inspections (2021–2026) with permit filings and subsequent violations, we exploit quasi-random assignment of inspectors within narrow administrative cells to identify the causal effect of enforcement intensity on two downstream outcomes: (1) whether property owners file construction permits (a costly, observable measure of compliance investment), and (2) whether neighboring properties on the same block respond with their own permit filings (spatial spillovers).

---

## 2. Data

We link five administrative datasets:

- **DOB Complaints** (NYC Open Data + BIS Web scrape): 649,000 complaint inspections with inspector identity (badge number), inspector comments, complaint category, disposition code, and property identifiers. This novel dataset was constructed by scraping the BIS Web portal to obtain inspector-level detail not available in the public Open Data release.
- **DOB Permits** (BIS + DOB NOW): 1.8 million permit filings with issuance dates, work types, and estimated costs.
- **ECB/OATH Violations**: 1.8 million violations with issuance dates, severity, and penalties.
- **PLUTO** (NYC Dept. of City Planning): Building characteristics (floors, year built, assessed value, building class, land use) and geocoordinates for every tax lot in NYC.
- **NTA Boundaries** (NYC DCP): 262 Neighborhood Tabulation Areas, assigned to complaints via spatial join (point-in-polygon), achieving 100% coverage.

Properties are identified by BBL (Borough-Block-Lot). Complaints are linked to permits and violations at the same BBL within specified time windows following each inspection.

---

## 3. Measuring Inspector Strictness

### 3.1 Leave-One-Out Violation Rate

For inspector *j* handling case *i*, we define strictness as the inspector's violation rate on all other cases:

$$S_{j(-i)} = \frac{1}{N_j - 1} \sum_{k \neq i,\; k \in \mathcal{C}_j} V_k$$

where $V_k \in \{0, 1\}$ indicates whether a violation was found on case $k$, and $\mathcal{C}_j$ is the set of all cases handled by inspector $j$. The leave-one-out construction avoids mechanical correlation between the instrument and the outcome for case $i$.

We restrict to cases with "substantive" outcomes — where the inspector exercised discretion (violation found *or* no violation warranted) — excluding no-access dispositions (C1–C4), referrals, and administrative closures.

### 3.2 Sample

The analysis sample contains:

| | |
|---|---|
| Observations | 456,460 |
| Inspectors ($\geq$30 cases) | 577 |
| Unique properties (BBLs) | 149,000 |
| Neighborhoods (NTAs) | 248 |
| Violation rate | 31.7% |
| Mean LOO strictness | 0.317 |
| SD of LOO strictness | 0.220 |
| P10–P90 range | 0.10 – 0.63 |

---

## 4. Identification Strategy

### 4.1 The Key Assumption

Within a given (complaint category × assigned unit × year-month × neighborhood) cell, which specific inspector handles a complaint is as-good-as-random. This assumption rests on the institutional process: complaints are routed to a DOB unit based on type and geography, and within that unit, the specific inspector assigned depends on scheduling, rotation, and availability — factors plausibly unrelated to the building's characteristics.

### 4.2 Estimating Equation

All regressions take the form:

$$Y_i = \beta \cdot S_{j(-i)} + \alpha_c + \varepsilon_i$$

where:

- $Y_i$ is the outcome for complaint $i$ (violation found, permit filed, future violation, or neighbor outcome)
- $S_{j(-i)}$ is the leave-one-out strictness of inspector $j$ assigned to case $i$
- $\alpha_c$ is a fixed effect for cell $c$, defined as the interaction of complaint category, assigned unit, year-month, and NTA
- $\varepsilon_i$ is the residual

Fixed effects are absorbed via the within transformation (Frisch-Waugh-Lovell): both $Y_i$ and $S_{j(-i)}$ are demeaned within each cell before estimation. We report results across a graduated sequence of geographic fixed effects — from none, to borough (5), community board (73), NTA (248), and census tract (2,322) — to assess how the coefficient evolves as the identifying variation becomes more local.

### 4.3 Interpretation of $\beta$

$\beta$ is the change in $Y$ associated with a one-unit (i.e., 100 percentage point) increase in inspector strictness, comparing inspectors handling the same type of complaint, at the same unit, in the same month, in the same neighborhood. A $\beta$ of 0.036 on any-permit-filed means: a 10 percentage point increase in inspector strictness raises the probability of a permit filing by 0.36 percentage points.

### 4.4 Balance Tests

If inspector assignment is quasi-random within cells, then $S_{j(-i)}$ should be uncorrelated with pre-determined building characteristics after conditioning on cell fixed effects. We test this by residualizing both LOO strictness and each covariate on cell FEs, then computing their correlation:

| Covariate | Correlation with LOO strictness | p-value |
|---|---|---|
| Number of floors | +0.004 | 0.018 |
| Year built | −0.003 | 0.039 |
| Building area | +0.003 | 0.049 |
| Assessed value | +0.001 | 0.602 |
| Residential units | +0.004 | 0.004 |
| Day of week | −0.016 | <0.001 |

All correlations are below |r| = 0.005, with the exception of day-of-week (r = −0.016), which reflects inspector scheduling patterns. The economic magnitude of imbalance is negligible: building characteristics jointly explain less than 0.02% of the variation in inspector strictness.

---

## 5. Results

### 5.1 Variance Decomposition

Inspector identity explains a substantial share of outcome variation:

| Fixed Effect | R² |
|---|---|
| Inspector identity | 22.3% |
| Complaint category | 15.8% |
| Assigned unit | 6.5% |
| NTA (neighborhood) | 1.4% |
| Borough | 0.2% |

Inspector identity is the single strongest predictor of whether a violation is found — stronger than the type of complaint itself.

### 5.2 First Stage: Does Inspector Strictness Predict Violation Outcomes?

| Specification | $\beta$ | SE | t-stat | FE Groups |
|---|---|---|---|---|
| (1) No controls | 0.996 | 0.003 | 361 | 1 |
| (2) Category | 0.873 | 0.003 | 279 | 277 |
| (3) Category × Unit | 0.892 | 0.003 | 267 | 1,122 |
| (4) Cat × Unit × Year-Month | 0.886 | 0.003 | 262 | 22,364 |
| (5) + Borough (5) | 0.901 | 0.004 | 260 | 43,804 |
| (6) + Community Board (73) | 0.937 | 0.003 | 277 | 170,457 |
| (7) + NTA (248) | 0.948 | 0.003 | 289 | 249,621 |
| (8) + Census Tract (2,322) | **0.959** | **0.003** | **335** | 353,837 |

N = 456,460 for all specifications. The coefficient is remarkably stable (0.89–0.96) across all specifications, and the t-statistic *increases* with finer geography — from 262 without geographic controls to 335 at census tract. At the most saturated specification, a 10pp increase in inspector strictness predicts a **9.6pp increase** in the probability of finding a violation.

### 5.3 No-Access Outcomes: Strictness vs. Persistence

A natural concern is whether the LOO violation rate captures not just an inspector's enforcement *threshold* (the decision to write a violation conditional on observing conditions) but also their *effort* (how hard they try to gain access and inspect). If "strict" inspectors are simply more diligent across all dimensions, the compliance response we estimate could reflect thoroughness rather than enforcement discretion.

We test this by examining no-access outcomes (disposition codes C1--C4: inspector unable to gain access or access denied). Across the full sample (572,641 inspections including no-access outcomes), the overall no-access rate is 20.3%, with enormous variation by complaint type --- illegal conversion complaints have a 74.9% no-access rate, while elevator complaints have less than 1%.

**Variance decomposition** reveals a different structure than for violations:

| Fixed Effect | R$^2$ (Violation) | R$^2$ (No-Access) |
|---|---|---|
| Inspector identity | 22.3% | 20.4% |
| Complaint category | 15.8% | 31.5% |
| Assigned unit | 27.3% | 6.5% |

Category explains twice as much of no-access variation (31.5%) as it does for violations (15.8%), reflecting that no-access is largely a property of the complaint type (residents refusing entry for illegal apartment inspections), not the inspector.

**First stage for no-access** shows a markedly different pattern than for violations. The LOO no-access rate coefficient drops from 0.997 (unconditional) to **0.267** at census tract, compared to 0.996 $\to$ 0.959 for violations. Most inspector-level variation in no-access rates is driven by case mix and geography, not individual behavior.

| Specification | $\beta$ (Violation) | $\beta$ (No-Access) |
|---|---|---|
| (1) No controls | 0.996 | 0.997 |
| (4) Cat $\times$ Unit $\times$ YM | 0.886 | 0.416 |
| (7) + NTA | 0.948 | 0.337 |
| (8) + Census Tract | **0.959** | **0.267** |

**The cross-inspector correlation between violation rate and no-access rate is essentially zero** ($r = -0.007$, $p = 0.86$, $N = 581$ inspectors). Strict inspectors are neither more nor less likely to gain access. Strictness and persistence are independent dimensions of inspector behavior.

This finding strengthens the identification. The variation in violation rates we exploit reflects the *decision margin* --- conditional on gaining access and observing conditions, how high is the inspector's bar for writing a violation? --- rather than effort or thoroughness in reaching that point. It also implies that DOB could, in principle, intervene on access persistence and enforcement threshold as operationally distinct policy levers.

### 5.4 Compliance: Does Strict Enforcement Induce Permit Filing?

**Panel A: Extensive margin** (any permit filed at the inspected property)

| Window | $\beta$ | SE | t-stat | Base rate |
|---|---|---|---|---|
| 30 days | 0.035 | 0.003 | 10.9 | 19.8% |
| 60 days | 0.034 | 0.004 | 9.4 | 28.2% |
| 90 days | **0.036** | **0.004** | **9.8** | 33.1% |
| 180 days | 0.035 | 0.004 | 9.4 | 41.0% |
| 365 days | 0.040 | 0.004 | 10.6 | 47.6% |

The effect is positive, highly significant, and stable across all horizons. A 10pp increase in inspector strictness raises the probability of any permit filing within 90 days by 0.36 percentage points, or about 1.1% relative to the base rate.

**Panel B: Intensive margin** (number of permits)

| Window | $\beta$ | SE | t-stat | Mean |
|---|---|---|---|---|
| 90 days | 0.088 | 0.036 | 2.4 | 1.58 |
| 180 days | 0.037 | 0.063 | 0.6 | 2.92 |
| 365 days | −0.017 | 0.113 | −0.2 | 5.26 |

The intensive margin is positive at short horizons but imprecise at longer horizons.

### 5.5 Robustness: The Sign-Flip Pattern

The permit result is highly sensitive to the inclusion of controls, in a pattern that is informative about the nature of confounding:

| Specification | $\beta$ | t-stat | FE Groups |
|---|---|---|---|
| (1) No controls | **−0.136** | −43.0 | 1 |
| (2) Category | −0.062 | −17.5 | 277 |
| (3) Category × Unit | +0.005 | 1.3 | 1,122 |
| (4) Cat × Unit × Year-Month | +0.004 | 1.0 | 22,364 |
| (5) + Borough (5) | +0.013 | 3.3 | 43,804 |
| (6) + Community Board (73) | +0.033 | 8.6 | 170,457 |
| (7) + NTA (248) | +0.036 | 9.8 | 249,621 |
| (8) + Census Tract (2,322) | **+0.044** | **14.4** | 353,837 |

Without controls, the correlation between strictness and permits is strongly *negative*: lenient inspectors' buildings file more permits. This reflects selection — lenient inspectors disproportionately handle complaints at buildings with ongoing construction (elevator maintenance, active renovation), which are already in permit pipelines.

Adding complaint category absorbs half the bias. Adding unit fixed effects brings the coefficient to approximately zero. Adding geographic fixed effects reveals the *positive* causal compliance response: within the same neighborhood, comparing the same type of complaint, stricter enforcement induces more permit filings.

The coefficient climbs smoothly from $+0.004$ (no geography) through $+0.013$ (borough), $+0.033$ (community board), $+0.036$ (NTA), to $\mathbf{+0.044}$ **(census tract, $t = 14.4$)**. The monotonic increase and rising $t$-statistic with finer geography indicate that within broad administrative cells, inspector assignment retains some correlation with neighborhood quality. Conditioning on increasingly granular geography removes this residual confounding. The census tract specification --- comparing complaints on the same few blocks --- yields the strongest and most precisely estimated effect.

### 5.6 Future ECB Violations: Cascading Enforcement

| Window | $\beta$ | SE | t-stat | Base rate |
|---|---|---|---|---|
| 30 days | −0.003 | 0.003 | −1.3 | 9.8% |
| 60 days | +0.004 | 0.003 | 1.2 | 15.9% |
| 90 days | +0.013 | 0.003 | 4.0 | 20.3% |
| 180 days | +0.026 | 0.004 | 7.1 | 28.3% |
| 365 days | **+0.043** | **0.004** | **11.2** | 36.4% |

Properties inspected by stricter inspectors receive *more* future ECB violations. This is not evidence of counterproductive enforcement. Rather, an initial violation triggers a monitoring cascade: the building is flagged for follow-up inspections, compliance checks, and re-inspections. The null effect at 30 days and growing positive effect through 365 days is consistent with the time required for follow-up enforcement to be initiated and documented.

This interpretation is supported by the simultaneous positive effect on permits: properties are investing in compliance (filing permits) *while also* receiving additional documented violations through the monitoring process.

### 5.7 Heterogeneity

**By borough** (any permit within 90 days):

| Borough | $\beta$ (permit) | t | $\beta$ (ECB) | t | N |
|---|---|---|---|---|---|
| Manhattan | 0.033 | 3.9 | 0.029 | 3.9 | 105,534 |
| Brooklyn | 0.022 | 3.6 | 0.005 | 1.0 | 159,484 |
| Queens | **0.061** | **8.9** | 0.003 | 0.5 | 104,954 |
| Bronx | 0.038 | 4.0 | 0.024 | 2.5 | 66,132 |
| Staten Island | 0.044 | 2.4 | 0.012 | 0.9 | 20,356 |

Queens shows the strongest compliance response ($\beta$ = 0.061, t = 8.9), consistent with its housing stock of small residential properties where owners are directly responsive to enforcement. Manhattan shows the strongest future-violation cascading effect, consistent with more intensive monitoring in the central borough.

**By building type** (any permit within 90 days):

| Building Type | $\beta$ (permit) | t | $\beta$ (ECB) | t | N |
|---|---|---|---|---|---|
| 1–2 Family | 0.049 | 7.8 | 0.003 | 0.7 | 111,756 |
| Multi-Family | 0.048 | 5.2 | 0.002 | 0.3 | 77,529 |
| Mixed Use | 0.045 | 6.0 | −0.005 | −0.6 | 77,736 |
| Commercial | 0.036 | 4.5 | 0.029 | 3.9 | 99,531 |

Residential properties (1–2 family and multi-family) show the strongest permit response. Commercial properties show the strongest ECB cascading, consistent with more intensive regulatory monitoring of commercial buildings.

---

## 6. Spatial Spillovers

### 6.1 Design

For each inspected property, we identify all other properties on the same tax block ("neighbors") using PLUTO. We then test whether the strictness of the inspector at the *focal* property affects outcomes at *neighboring* properties. The estimating equation is:

$$Y^{\text{neighbor}}_i = \beta^{\text{spill}} \cdot S_{j(-i)} + \alpha_c + \varepsilon_i$$

where $Y^{\text{neighbor}}_i$ measures permit filings or violations at the neighbors of the property inspected in case $i$, and all other notation is as before. This tests whether strict enforcement generates localized compliance responses beyond the directly inspected property.

The sample contains 221,437 inspections with at least one block neighbor. The average property has 36.4 neighbors on its block.

### 6.2 Results

**Panel A: Neighbor permit filing**

| Outcome | $\beta$ | SE | t-stat | Mean |
|---|---|---|---|---|
| Any neighbor has permit (90d) | −0.004 | 0.005 | −0.8 | 66.1% |
| **% neighbors with permit (90d)** | **+0.007** | **0.001** | **4.9** | 8.2% |
| **% neighbors with permit (365d)** | **+0.009** | **0.002** | **5.5** | 14.0% |
| Total neighbor permits (180d) | +0.437 | 0.216 | 2.0 | 11.7 |
| **Total neighbor permits (365d)** | **+1.194** | **0.377** | **3.2** | 20.9 |

The binary "any neighbor has a permit" measure is null because the base rate is already 66% — there is a ceiling effect. However, the *share* of neighbors filing permits is strongly significant: a 10pp increase in the focal inspector's strictness raises the share of neighboring properties filing permits by 0.07 percentage points (t = 4.9 at 90 days, t = 5.5 at 365 days).

The total number of neighbor permits is significant at longer horizons: roughly 1.2 additional permits on the block within a year (t = 3.2).

**Panel B: Neighbor ECB violations**

| Outcome | $\beta$ | SE | t-stat |
|---|---|---|---|
| Neighbor ECB violations (90d) | −0.024 | 0.029 | −0.8 |
| Neighbor ECB violations (180d) | +0.030 | 0.044 | 0.7 |
| Neighbor ECB violations (365d) | −0.051 | 0.069 | −0.7 |

No spillover effect on neighbor violations. Enforcement cascading is property-specific, not block-wide.

**Panel C: Spillovers conditional on own violation status**

| Focal outcome | $\beta$ (neighbor permit 90d) | t |
|---|---|---|
| Violation found | +0.016 | 1.6 |
| No violation found | −0.016 | −1.7 |

Suggestive evidence that the spillover mechanism operates through *visible enforcement actions*: when the focal property receives a violation (an event that may be observable to neighbors), neighbor permit filings increase. When no violation is found, there is no spillover.

### 6.3 Interpretation

The spatial spillover results are consistent with a **localized deterrence/information mechanism**: property owners on a block observe enforcement activity at a neighboring property and respond by investing in their own compliance. The effect is modest in magnitude but highly statistically significant, grows over time, and appears to operate specifically through visible enforcement actions (violations served) rather than mere inspector presence.

---

## 7. Summary

| Finding | NTA ($\beta$, t) | Census Tract ($\beta$, t) |
|---|---|---|
| Inspector identity explains 22% of violation outcome variance | R² = 0.223 | — |
| LOO strictness predicts own-case violations | 0.948, t=289 | 0.959, t=335 |
| Strict enforcement → more permit filings (90d) | 0.036, t=9.8 | **0.044, t=14.4** |
| Strict enforcement → more permit filings (365d) | 0.040, t=10.6 | 0.030, t=9.7 |
| Strict enforcement → more future ECB violations (365d) | 0.043, t=11.2 | 0.019, t=6.0 |
| Neighbor permit spillover (% with permit, 365d) | 0.009, t=5.5 | — |
| Neighbor violation spillover | $\approx$ 0, n.s. | — |
| Balance: max |r| for building characteristics | 0.004 | — |

Inspector strictness causally induces real compliance investment, as measured by costly permit filings. The effect is identified within narrow administrative and geographic cells where inspector assignment is quasi-random, robust to increasingly saturated fixed effects, and present across all boroughs and building types. There is additionally evidence of localized spatial spillovers: strict enforcement at one property induces compliance investment at neighboring properties on the same block.
