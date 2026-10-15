# Paper 2 — Causal Spillover Effects of NRB's 2015 Capital-Floor Mandate

**Status:** design settled, not yet implemented. **Target venue:** econometrics/finance venue, or
ICAIF's applied track.

**One-line description:** Estimate the causal spillover effect of NRB's 2015 capital-floor
mandate on Nepali BFIs, using Paper 1's reconstructed interbank exposure matrix $\hat W_t$ as the
exposure-mapping weight in an Aronow–Samii/Leung-style interference design.

**Dependency on Paper 1 is strictly one-directional**: Paper 1 consumes covariates + marginals +
capital and outputs $\hat W_t$. Paper 2 takes $\hat W_t$ as a fixed input and combines it with the
capital-floor treatment dose and P&L outcomes to estimate the causal spillover effect. None of
Paper 2's causal-identification machinery (DiD/RDD, parallel trends, no-sorting) appears anywhere
in Paper 1's model — see `paper1_network_reconstruction.md`.

---

## 1. Identification design

- **Primary**: event-study / difference-in-differences around the 2015 capital-floor mandate year
  — treat the mandate as a quasi-random shock, under a parallel-trends assumption.
- **Secondary / robustness**: regression discontinuity near the capital-floor threshold itself —
  under a no-sorting-near-cutoff assumption.

## 2. Interference structure (Aronow–Samii / Leung style)

Outcome for bank $i$ depends on two exposure components. Formalized as a panel regression (the
natural functional form implied by the DiD design in §1 — stated explicitly here so every term
can be matched one-to-one to a raw variable in §4):

$$y_{i,t} = \beta \cdot \text{dose}_{i,t} + \gamma \cdot \text{spillover}_{i,t} + \delta^\top Z_{i,t} + \alpha_i + \tau_t + \varepsilon_{i,t} \qquad \text{(Eq. A)}$$

- **Direct treatment dose** ($\text{dose}_{i,t}$): own distance below the floor
- **Network exposure** (the spillover carrier): neighbors' treatment dose, weighted by Paper 1's
  reconstructed exposure matrix:
  $$\text{spillover}_{i,t} = \sum_j \hat W_{ij,t} \cdot \text{dose}_{j,t} \qquad \text{(Eq. B)}$$
- $Z_{i,t}$: the control vector (§4d)
- $\alpha_i, \tau_t$: bank and period fixed effects (absorb time-invariant bank heterogeneity and
  common shocks, standard in DiD)

This is the exposure-mapping function $f(z,\theta_i) \to D_i$ in Aronow-Samii's framework — here
instantiated with $\hat W_t$ as the weighting structure instead of a known/designed network.

## 3. The Aronow–Samii observational-vs-randomized caveat (applies throughout)

Aronow & Samii (2017)'s unbiasedness and conservative-variance guarantees (Horvitz-Thompson /
Hájek estimators, the generalized propensity score $\pi_i(d_k)$) are **design-based**: they hold
because $\Pr(Z=z)$ is known by construction in a randomized experiment.

The NRB capital-floor mandate is **not** a randomized assignment — it's a regulator-driven,
observational shock. So throughout this paper, $\pi_i(d_k)$ must instead be **estimated** from
observables (size, capital ratio, BFI class) under a **selection-on-observables / conditional
ignorability** assumption — the paper itself gestures at this extension via Tchetgen Tchetgen &
VanderWeele (2012).

**Consequence**: every property that would be "guaranteed" in Aronow-Samii's original randomized
setting (unbiasedness, conservative variance) becomes **conditional on the propensity model being
correctly specified** in this paper. This downgrade applies to *every* regressor and estimator
below, not just the headline spillover term — it should be stated once, prominently, and then
referenced rather than re-derived at each step.

## 4. Variable inventory

Eq. (A) is estimated once per outcome — $y_{i,t}$ is instantiated separately for each row in §4b
(same $\text{dose}$, $\text{spillover}$, $Z_{i,t}$ on the right-hand side each time).

### 4a. Treatment dose — $\text{dose}_{i,t}$ in Eq. (A), (B)

| Variable | Built from | Symbol |
|---|---|---|
| Capital-floor distance | `1 CAPITAL FUND` (Balance Sheet, Liabilities) ÷ size, measured against the 2015 regulatory floor | $\text{dose}_{i,t} = \text{floor} - \dfrac{\text{CAPITAL FUND}_{i,t}}{\text{size}_{i,t}}$ |

### 4b. Direct outcomes — $y_{i,t}$ in Eq. (A), one row per model run

| NRB field | Symbol instantiation |
|---|---|
| `12. Net Profit` / `9. Net Loss` | $y_{i,t} = \text{NetProfit}_{i,t}$ (primary outcome) |
| `7.1.1 / 7.1.2 / 7.1.3` loan-loss provision breakdown | $y_{i,t} = \text{Provision}_{i,t}$ (General Pass/Watch, Special, Additional — run separately or summed) |
| `8. Loan Written Off` | $y_{i,t} = \text{WriteOff}_{i,t}$ |
| `1. Interest Income − 1. Interest Expense` | $y_{i,t} = \text{NIM}_{i,t}$ |

`10. Provision for Income Tax` is **not** an outcome — it's a control, moved to §4d below (it was
mislabeled as an outcome in the first draft of this file).

### 4c. Spillover / network-exposure term — Eq. (B)

| Component | Source | Symbol |
|---|---|---|
| $\hat W_{ij,t}$ | **output of Paper 1** — not re-estimated here, treated as fixed | weight in Eq. (B) |
| $\text{dose}_{j,t}$ | same construction as §4a, applied to neighbor $j$ | summed term in Eq. (B) |

### 4d. Controls — components of $Z_{i,t}$ in Eq. (A)

Same underlying raw fields as Paper 1's $X_{i,t}$ (`paper1_network_reconstruction.md` §8e), reused
here as regression controls rather than SBM reconstruction inputs — kept as a *separate* symbol
($Z_{i,t}$, not $X_{i,t}$) since here they enter a linear control vector, not a fitness exponent.

| Component | Built from | Symbol |
|---|---|---|
| $Z^{(1)}_{i,t}$ | `1 LIQUID FUNDS` (Assets) ÷ size | liquidity ratio |
| $Z^{(2)}_{i,t}$ | `4 LOANS & ADVANCES` ÷ `3 DEPOSITS` | loan-to-deposit pressure |
| $Z^{(3)}_{i,t}$ | `2.a NRB` ÷ `2 BORROWINGS` | NRB-dependence |
| $Z^{(4)}_{i,t}$ | `3 DEPOSITS` composition (Current/Savings/Fixed/Call) | funding stability |
| $Z^{(5)}_{i,t}$ | `10. Provision for Income Tax` ÷ pre-tax profit | effective tax rate (moved here from §4b) |
| $Z^{(6)}_{i,t}$ (optional) | `3.e Bonds and Securities` ÷ `2 BORROWINGS` | wholesale market access |
| $Z^{(7)}_{i,t}$ (optional) | `2.c Foreign Banks and Fin. Ins.` (Liabilities) | domestic vs. foreign interbank split |

### 4e. Statement of Credit — secondary outcome / concentration control

| Group | Role | Symbol |
|---|---|---|
| Sector-wise credit (12 sectors + Others) | secondary outcome, or concentration control | $y_{i,t} = \text{SectorCredit}_{i,t}$, or an added $Z^{(8)}_{i,t}$ term |
| Product-wise credit | control (loan-product mix) | component of $Z_{i,t}$ |
| Collateral-wise credit | asset-quality control candidate | component of $Z_{i,t}$ if used |

**Coverage caveat**: `nrb_statement_of_credit.csv` currently has only 2 years of coverage in the
extracted data — usable as a cross-sectional control at the mandate year if available, but **not**
as a full event-study panel unless more years are collected.

### 4f. Terms with no raw-field source (latent/learned)

| Symbol | Appears in | Source |
|---|---|---|
| $\beta, \gamma, \delta$ | Eq. (A) | **estimated** regression coefficients |
| $\alpha_i, \tau_t$ | Eq. (A) | **estimated** fixed effects |
| $\varepsilon_{i,t}$ | Eq. (A) | residual |
| $\pi_i(d_k)$ | §3 | **estimated** generalized propensity score, modeled as a function of $X^{(1)}_{i,t}, X^{(2)}_{i,t}$ (size, capital ratio — same raw fields as Paper 1's covariates, reused as the propensity-model inputs) and BFI class |

## 5. Data-pipeline caveats relevant to this paper

- The `item_code` collision documented in `paper1_network_reconstruction.md` §9 (Liabilities vs.
  Assets side both restarting at 1) affects the covariate/control construction in §4d as well,
  since several controls here pull from both sides of the Balance Sheet. Needs the same `side`
  column fix before any automated join.
- P&L `item_code`s do not collide across Income/Expense sides in the same way balance-sheet
  Liabilities/Assets do (confirmed from the same extraction run) — the outcome variables in §4b
  are safe to join on `item_code` + `statement == "profit_and_loss"` as-is.

## 6. Paper structure (not yet drafted in detail — placeholder)

Econometrics-venue structure: Intro/motivation → institutional background (2015 mandate) →
identification strategy (DiD + RDD) → data (NRB panel + $\hat W_t$ from Paper 1) → results →
robustness (RDD, alternative propensity specifications per §3) → discussion.
