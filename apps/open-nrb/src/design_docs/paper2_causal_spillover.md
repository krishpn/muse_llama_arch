# Paper 2 — Causal Spillover Effects of NRB's 2015 Capital-Floor Mandate

**Status:** implemented and run (2026-11-04); results in §5b; Paper 1 consistency check in §0b;
drafting next. **Target venue:** field journal (J. Financial Stability / Emerging Markets Review),
full length with appendices.

**One-line description:** Estimate the causal spillover effect of NRB's 2015 capital-floor
mandate on Nepali BFIs, using Paper 1's reconstructed interbank exposure matrix $\hat W_t$ as the
exposure-mapping weight in an Aronow–Samii/Leung-style interference design.

**Dependency on Paper 1 is strictly one-directional**: Paper 1 consumes covariates + marginals +
capital and outputs $\hat W_t$. Paper 2 takes $\hat W_t$ as a fixed input and combines it with the
capital-floor treatment dose and P&L outcomes to estimate the causal spillover effect. None of
Paper 2's causal-identification machinery (DiD/RDD, parallel trends, no-sorting) appears anywhere
in Paper 1's model — see `paper1_network_reconstruction.md`.

**Note on Paper 1's block-identity ($z$) result, 2026-10-29:** Paper 1's `kl_beta` ablation
(bracketing $\beta_{\max}=0.15$ vs. $0$) confirmed block-identity recovery collapses to a
degenerate, information-free posterior on real NRB data regardless of hyperparameter setting —
uniform $(0.5, 0.5)$ for every bank at one end, a unanimous single-block assignment for every bank
at the other — because real data never supplies a reconstruction signal to supervise $z$ (no
bilateral ground truth exists). **This does not weaken Paper 2.** Paper 2 consumes only $\hat
W_t$ (Eq. B, §4c) — never $z$ or block membership — so the collapse is structurally outside this
paper's dependency surface. The $\hat W_t$ reconstruction result itself remains robust (beats the
classical Sinkhorn/RAS baseline on temporal stability) and is unaffected.

~~If anything, the two-sided collapse is useful supporting evidence for treating $\hat W_t$ as a
trustworthy fixed input here: the same architecture recovers real block structure on synthetic
data when a genuine reconstruction signal exists, ...~~ **Withdrawn 2026-11-04 (§0b): Paper 1's own
later results don't support this paragraph.** Synthetic block recovery beats the trivial baseline
in only 3/5 seeds and not significantly (p = 0.1136, Paper 1 §10). The uniform / single-block
bracket was superseded by Paper 1's `lambda_mi` findings. And on real data, $\hat W_t$ is
determined by the projection onto the observed marginals plus the smoothness prior, not by $z$
(Paper 1 §10). Don't use this paragraph to argue that $\hat W_t$ is reliable; use §0b.

## 0a. Interface contract: what Paper 1 actually hands off

Concrete artifacts, not just the conceptual $\hat W_t$ dependency stated above — pinned down now
since Paper 1 has real synthetic-validation results and a real-data run in progress
(`paper1_network_reconstruction.md` §6, §12).

- **Primary artifact**: $\hat W_t$ built from Paper 1's **posterior-mean** decode (§6 of that doc:
  $q_\psi(z)$'s mean, $\sigma(\eta)$ for existence, $\lambda\cdot e^{\sigma^2/2}$ for magnitude, then
  projected through Algorithm 1) — never a single reparameterized training-time sample. Saved per
  (bank $i$, bank $j$, period $t$) for exactly Paper 1's complete-case panel.
- **Index set must match exactly**: Paper 1's real-data panel is currently N=16 of 33 commercial
  banks, T≈133 months (as of 2026-10-28, re-measured per §9.9 of that doc). Paper 2's regression
  panel (§4) is therefore restricted to the same N/T unless Paper 2 explicitly handles banks Paper
  1 couldn't reconstruct — e.g. treating them as missing neighbors (zero spillover weight) vs.
  dropping them from $i$ entirely — a decision not yet made here.
- **Carry over the RAS/Gandy-Veraart baseline network too, not just $\hat W_t$**: Paper 1 already
  computes $W_t^{\text{RAS}}$ as its mandatory classical comparator (§7, §12 of that doc).
  Re-running Eq. (A)/(B) with $W_t^{\text{RAS}}$ in place of $\hat W_t$ as the spillover weight is a
  specification-robustness check Paper 2 gets for free — if the spillover effect $\hat\gamma$
  survives under both weightings, that's real evidence the causal finding isn't an artifact of
  which reconstruction method was used; if it doesn't survive, that's the single most important
  robustness result in the paper and should be reported either way. **Qualified 2026-11-04
  (§0b B2):** on real data both networks are pinned to the same marginals, so agreement is
  expected; it shows robustness to the cap and the smoothing, not to the reconstruction method.
- **Inherited caveat — closed-system bias in $\hat W_t$** (from Paper 1's 2026-11-01 reviewer-mode
  review, §10a item 1): Paper 1's marginal-constraint projection is only exact over its own N-bank
  complete-case submatrix; real lending/borrowing totals almost certainly include exposure to
  counterparties outside that set (discarded banks, NRB, foreign banks). Eq. (B)'s spillover sum
  $\sum_j \hat W_{ij,t}\cdot\text{dose}_{j,t}$ is therefore only a partial accounting of bank $i$'s
  true network exposure — any true spillover routed through a non-modeled counterparty is invisible
  to this term by construction, not just unmeasured noise. Worth one sentence in Paper 2's
  limitations, framed as inherited from Paper 1 rather than independently discovered here.
- **Not yet addressed — $\hat W_t$ is an estimated, not observed, regressor.** Eq. (B)'s spillover
  term is built from a point estimate, with no propagation of Paper 1's own estimation uncertainty
  into Paper 2's standard errors on $\gamma$. This is a classical errors-in-variables problem
  (attenuation bias on $\gamma$, understated SEs) — analogous to, and should sit alongside, §3's
  existing propensity-score-estimation downgrade. **Possible fix, not yet implemented**: Paper 1
  exports posterior draws of $(\eta,\lambda)$ (not just the posterior mean) for a handful of
  bootstrap replicates of $\hat W_t$; Paper 2 re-estimates Eq. (A) across those replicates and
  reports the added variance from $\hat W_t$'s own uncertainty, not just $\varepsilon_{i,t}$'s.
- **NOT carried over**: Paper 1's block assignment $z$ / `BlockPosteriorMLP` output — already
  correctly scoped out above (2026-10-29 note); restated here only so this section stays the single
  place listing everything that does/doesn't cross the Paper 1 → Paper 2 boundary.

## 0b. Consistency with Paper 1's results (2026-11-04, before drafting)

This compares what Paper 2 claims or assumes about Paper 1 against Paper 1's design doc
(`paper1_network_reconstruction.md`, read 2026-11-04) and the paper1_model project notes. **Paper 1
is not edited.** Items that Paper 1 should pick up are listed in (C) for the user to carry over.

**(A) Consistent: the same numbers in both papers.**
- *Panel:* N = 16 of 33 commercial banks, T = 133 months from Asar 2072 (Paper 1 §9.9 "T≈133",
  `MIN_PERIOD`; Paper 2 §5a). The pre-2072 gap Paper 1 deferred (no balance-sheet rows Chaitra
  2071 – Jestha 2072) is exactly the three missing months that `total_effect` prints.
  - Arithmetic to check: Baisakh 2070 – Shrawan 2083 is 160 calendar months. 160 − 3 known gaps =
    157, but §5a counts 156 bulletins. And 156 − 23 = 133 = T, while Asar 2072 – Shrawan 2083 is
    134 calendar months. So one more month seems to be missing *inside* the estimation panel.
    This holds only if the CSV starts at Baisakh 2070; verify. Identify the month: it matters for
    Paper 1's smoothness term (W_t vs W_{t−1} across a gap) and for Paper 2's de-cumulation.
- *Covariates:* Paper 2's controls (liquidity ratio, LDR, NRB dependence, plus log size) are
  Paper 1's X_{i,t} (§8b) without the capital ratio, which is Paper 2's dose. `size` is the
  same item sum in both (Paper 1 §8b; Paper 2 §5a notes it inherits the digit fragments).
- *Statement totals:* Paper 1's audit found TOTAL Liabilities = TOTAL ASSETS with 0 mismatches
  over 2,615 (file, bank) pairs, and Paper 2's §5a found it in 100% of the 4,094 sheets that have
  both rows. The counts differ because Paper 2 covers all 42 institutions and both layouts; the
  conclusion is the same.
- *Survivorship:* both papers use 16/33. Paper 1 §10a item 5 asks whether the discard is
  systematic, and Paper 2 item 2 answers it: selection on size (median NPR 64.9bn vs 44.5bn,
  p 0.005), not on dose (p 0.25). Use the same numbers in both papers.
- *z / block identity:* not used by Paper 2 (§0, §0a).

**(B) Paper 2 claims that Paper 1's results weaken: fix before drafting.**
1. **§0's "architecture recovers block structure on synthetic data"** is withdrawn (see §0).
2. **W_hat vs RAS is not an independent robustness check.** §0a calls a γ that survives both
   networks "real evidence the causal finding isn't an artifact of which reconstruction method
   was used". Paper 1 §10 found that on real data $\hat W_t$ is the Dykstra projection onto the
   same observed marginals (plus the cap and a smoothness prior): two runs with opposite
   posteriors give $\hat W_t$ within rel. diff 1.5e-5. RAS is also a function of those marginals.
   The two networks therefore differ only by the cap and the smoothing, so agreement between them
   is expected and is weak evidence. The near-identical Eq. A′ estimates (NIM dose 0.082 W_hat vs
   0.088 RAS) fit this. Paper wording: "robust to dropping the regulatory cap and the temporal
   smoothing", not "robust to the reconstruction method".
   **Check [cheap]:** corr(E_it under W_hat, E_it under RAS) over post cells, and the mean
   |difference| relative to the mean exposure. If corr ≈ 1, say so.
3. **Item 7 ("W_hat is a generated regressor").** The argument stands (ignoring first-stage noise
   makes SEs too small, so it can only reinforce a null), but it can be sharpened with Paper 1's
   result. On real data $\hat W_t$ barely depends on the model state, so its *sampling* variability
   is probably small. Its *model* error (which of the many matrices that satisfy the marginals is
   true) is not measured at all, because no bilateral ground truth exists (Paper 1 §0, §12). State
   both. Only Paper 1's seed 0 real run exists, so seed sensitivity is unmeasured; re-running
   Paper 1 with other seeds (no code edit) and re-fitting γ would measure it.
4. **Paper 1's headline real-data number.** Temporal stability is 1.16 vs RAS 2.32 (2026-10-31
   run, `lambda_mi` = 0.05). Paper 1 §10a item 1 calls it circular: the model is penalised for
   period-to-period change and RAS is not. If Paper 2 cites it, cite it with that caveat. Also,
   after `lambda_mi` was set to 0 (2026-11-01), Paper 1's doc records no updated stability figure
   or synthetic mean error (0.4154 and 0.4219 are from earlier settings; only p = 0.0101 is
   recorded for `lambda_mi` = 0). Paper 2 needs Paper 1's final numbers.
   **Check:** confirm that the artifact Paper 2 loads (`results/baseline/real_run_seed0.pt`) is the
   post-2026-11-01 run (`lambda_mi` = 0); its mtime and the saved config should show this.
5. **The cap mechanically links W_hat to the treatment.** Paper 1's cap is κ·K_{i,t} with K =
   CAPITAL FUND (§5, §8e). The paid-up mandate raised capital 1.05–4.38× (item 1), so wherever
   the cap binds, the mandate changes $\hat W_t$ through the cap itself. This is a concrete
   mechanism behind the item 3 point that the exposure terms are post-treatment controls for β,
   and it is the only part of the network where W_hat and RAS differ by construction (RAS has no
   cap). The cap-binding fraction is still unmeasured (Paper 1 §10a item 2). **Check [cheap]:**
   share of (i, j, t) entries with $\hat W_{ij,t}$ ≥ κK_{i,t}(1 − 1e-6), by fiscal year. Read κ
   from Paper 1's config; don't assume it.
6. **The extraction corruption feeds W_hat.** §5a shows the bulletins are clean through Falgun
   2075 and corrupted from Chaitra 2075 (digit fragments; size, CAPITAL FUND and sub-items wrong,
   0–8% passing from Magh 2079). Paper 1's doc does not record this. Paper 1 judged the data
   "good enough" on 2026-10-28, before the §5a audit, and its 1.16 vs 2.32 is computed over all
   133 months. Every Paper 1 input can be hit: r, c are lettered sub-items (the class the fragments
   hit), K = CAPITAL FUND, and size. For Paper 2's main spec (clean window) the risk is small but
   not zero. Each month's $\hat W_t$ is pinned to that month's (clean) marginals by the
   projection. But the decoder and GRU are trained on all 133 months, and the smoothness penalty
   couples neighbouring months, so Falgun 2075 is linked to Chaitra 2075. Given (B2) the effect
   should be small. **Checks:** (a) the share of single-digit r, c, K values by fiscal year, with
   the `audit_bs_extraction.py` logic applied to the two marginal rows; (b) re-run Paper 1 on
   Asar 2072 – Falgun 2075 only (a config change, not a code edit, if `run_real_experiment` takes
   an end period; otherwise defer) and compare $\hat W_t$ on the overlap.

**(C) Paper 2 findings that Paper 1 should cite (user to carry over; Paper 1 not edited here).**
- §5a extraction audit (clean through Falgun 2075, corrupted from Chaitra 2075), together with
  (B6). This is a limitation of Paper 1's real-data result and belongs in its §9.9 / §12
  limitations.
- Merger survivors in the 16-bank panel (Paper 1 §10a item 9, open there): SBL (joint operation
  2016-06-21 ≈ Asar 2073) and RBB (2018-05-02 ≈ Baisakh 2075), from NRB's List of BFIs
  (user-supplied). Which institutions they absorbed is not verified (item 2).
- Survivorship numbers (A), answering Paper 1 §10a item 5.
- The label relabels Citizens→Citizen, NBB→NBBL, PRABHU→Prabhu, Civil→CBL (Magh 2075), which are
  continuous in size and capital (item 2). Harmonising them in Paper 1 would add 4 banks to its
  panel. Out of scope for both papers now.

**Drafting rule from this section:** Paper 2's text says $\hat W_t$ is "the marginal-constrained,
capped reconstruction of Paper 1". It does not call it a validated network, and it does not
present W_hat vs RAS as independent methods.

---

## 1. Identification design

- **Primary**: event-study / difference-in-differences around the 2015 capital-floor mandate year
  — treat the mandate as a quasi-random shock, under a parallel-trends assumption.
- **Secondary / robustness**: regression discontinuity near the capital-floor threshold itself —
  under a no-sorting-near-cutoff assumption.

**Specification revised 2026-11-03, after the first real end-to-end run** (user-approved). Eq.
(A) below with a *contemporaneous* $\text{dose}_{i,t}$ is not a DiD around the mandate, and it is
mechanically contaminated: $\text{CAPITAL FUND}_{i,t}$ absorbs retained profit and is drawn down
by provisions/write-offs, so regressing P&L outcomes on it partly recovers an accounting identity.
The estimated equation is therefore a **treatment-intensity DiD with a frozen pre-mandate dose**
(Eq. A′, §2). Concretely (`open_nrb_causal/mandate_design.py`):
- *Timing*: pre-mandate window = panel start (Asar 2072) .. Mangsir 2072, the last month before
  the parallel run (Poush end 2072). Transition = Poush 2072 .. Asar 2073 (dropped from the pooled
  DiD, its own event-study bin). Post = Shrawan 2073 onward.
- *Event study*: bins by Nepali fiscal year (Shrawan..Asar), reference = the pre-window fiscal
  year (FY 2072/73's pre-parallel-run months). **Pre-trend power is very limited**: $\hat W_t$
  only exists from Asar 2072, so the only lead bin is a single month (Asar 2072) — and after YTD
  de-cumulation that month has no previous month and is NaN, so **the event study has no
  estimable lead bin at all** (reported as a dropped bin). A longer
  dose-only pre-trend check (no spillover term, back to 2070 Baisakh) is possible from the
  balance-sheet/P&L CSVs alone and is an open option.
- *Outcomes*: divided by size (ROA-style ratios), de-cumulated first if the P&L is fiscal-YTD
  cumulative (diagnostic printed by the run).
- *Inference*: bank-clustered (G=16) CR1 SEs + WCR wild-cluster-bootstrap p-values, Webb
  six-point weights.
- *RDD*: a single cross-section has ~1 bank per side of the threshold at bandwidth 0.02 (first run:
  all NaN), so it pools Shrawan 2073 .. Asar 2075, the explicitly-mandated 4% period. With N=16 it
  stays a secondary/descriptive check; distinct banks per side are reported alongside cell counts.

## 1a. Framing note: potential outcomes, not a Pearlian SCM

This paper's causal apparatus — Aronow–Samii interference, Horvitz–Thompson/Hájek weighting, the
generalized propensity score, DiD/RDD identification — is stated entirely in the **potential-
outcomes (Neyman–Rubin)** idiom ($Y_i(d)$, $\Pr(D=d)$, ignorability), not Pearl's **structural-
causal-model (SCM)** idiom of DAGs and do-calculus. The two are known to be formally equivalent
under standard regularity conditions (Pearl 2009; Richardson & Robins' SWIGs give an explicit
translation between them), so nothing in the identification argument below hinges on preferring
one over the other. The reason to flag it at all is reviewer-facing, not substantive: Aronow &
Samii (2017) and the interference/network-spillover literature this paper builds on (§2-§3) write
exclusively in PO notation, and switching framing mid-paper — e.g. introducing a DAG in the intro
— would require re-deriving the exposure-mapping and ignorability assumptions in SCM terms for no
identification gain. So this paper stays in PO notation throughout; this note exists so a
Pearl-trained reviewer isn't left wondering why no DAG appears.

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

**Estimated form (Eq. A′, 2026-11-03; see §1).** With $\overline{\text{dose}}_i$ = bank $i$'s mean
dose over the pre-mandate window, $\text{post}_t = 1[t > \text{Asar 2073}]$, and
$E_{i,t} = \sum_j \hat W_{ij,t}\,\overline{\text{dose}}_j$:

$$y_{i,t} = \beta\,\overline{\text{dose}}_i\,\text{post}_t + \gamma\,E_{i,t}\,\text{post}_t + \rho\,E_{i,t} + \delta^\top (\bar Z_i\,\text{post}_t) + \alpha_i + \tau_t + \varepsilon_{i,t} \qquad \text{(Eq. A′)}$$

$\rho E_{i,t}$ keeps $\gamma$ identified off the pre/post change rather than off cross-sectional
connectedness to low-capital banks; $\bar Z_i$ (pre-window means of §4d's controls plus log size)
interacted with post replaces contemporaneous, possibly post-treatment, controls. The generalized
propensity score (§3) models $E_{i,t}$ given $(\overline{\text{dose}}_i, \bar Z_i)$, and is
re-estimated separately for the RAS-baseline network (§0a). An earlier implementation reused
$\hat W_t$'s HT weights for the RAS run, which is a bug.

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

**Floor value resolved, 2026-11-02, from the real NRB "Capital Adequacy Framework, 2015"
document (supplied by the user):** $\text{floor} = 0.04$, NRB's Basel III **leverage ratio**
minimum (§2.7.B.a: minimum Tier 1/Exposure Measure of 4%, parallel run Mid July 2016 – Mid July
2018) — chosen over the Total/CET1 Capital Ratio floors (§2.4, §1.8's 8.5%–11% phase-in table)
because those are defined against **risk-weighted** exposure, which Paper 1 never computes (would
require replicating §3–§5's full credit/operational/market risk-weighting machinery), whereas
Paper 1's `capital_ratio` covariate (`CAPITAL FUND ÷ size`, size = total on-balance-sheet assets)
is structurally the leverage ratio's denominator, not the CAR's.

**This is an acknowledged approximation, not an exact replication — state as a limitation:**
1. *Numerator*: Paper 1's `capital_ratio` uses CAPITAL FUND (Tier 1 + Tier 2); the regulatory
   leverage ratio is Tier 1 only. Overstates the numerator for any bank holding Tier 2 capital,
   biasing $\text{dose}$ downward (understates distance below the floor).
2. *Denominator*: `size` is not exactly the regulatory "Exposure Measure" (§2.7.B.II nets specific
   provisions/valuation adjustments and adds off-balance-sheet items at a 100% CCF) — not
   replicated here.
3. *Period applicability*: the 4% figure is explicitly mandated only for the parallel run
   (mid-2016–mid-2018); "the final leverage ratio requirement might be revised by NRB" afterward,
   with no later figure given in the source document. Applied as a constant across Paper 1's full
   panel (which starts before the mandate, 2072 Asar) — a modeling choice, not a verified
   post-2018 fact.

**Effective date**: Mid July 2016 = **Asar end 2073 BS** (§1.9); parallel run Poush end 2072 BS →
Asar end 2073 BS. This supersedes an earlier background-agent draft's unverified "NPR 8bn / Asar
2074 BS" figure — which was a different quantity (paid-up capital, not a leverage/CAR ratio) and
does not appear anywhere in the actual framework document; it was correctly rejected as a
leverage-floor value. (Update 2026-11-04: as a *paid-up capital* figure it is confirmed by Unified
Directive 2073, dir. 23 pt. 4; see the 800-crore regime below. It remains irrelevant to
`capital_floor`.)

**Paid-up capital requirement, the regime before the window (2026-11-04).** Both circulars were
supplied by the user as Preeti-font text and decoded here; check the decoding against the PDFs
before quoting.
- *NRB circular no. 25 (2063/64), ref. बै.वि.नि.वि./नीति/परिपत्र/२५/०६३/६४, dated 2064-01-02,
  "पूँजी वृद्धि बारे" (on raising capital).*
  - It applies the "Policy and procedural arrangements for licensing BFIs, 2063" (NRB decision
    of 2063-12-13).
  - Minimum paid-up capital: Class "क" (commercial banks), national level, **Rs 200 crore =
    NPR 2,000 mn**. The other classes are lower: "ख" Rs 64 crore, "ग" Rs 30/20 crore, "घ" Rs 10
    crore and less.
  - Existing BFIs had to reach it by end of Asar 2070. Asar 2064 was the base year, with
    proportional annual increases from FY 2064/65.
  - At least 80% had to be paid-up capital; the remaining 20% could come from other primary
    capital sources.
  - Sanctions grew each year a bank fell short: dividends withheld (FY 2064/65), then a ban on
    taking deposits (2065/66), on lending (2066/67), and on all business except loan recovery
    and deposit repayment (2067/68). After that, licence cancellation.
  - Banks that are short and merge may count their combined capital.
- *NRB circular no. 02 (2070/71, Classes A/B/C), ref. बै.वि.नि.वि./नीति/परिपत्र/कखग/०२/०७०/७१,
  dated 2070-04-27, §2 "prescribed minimum paid-up capital".*
  - It implements the FY 2070/71 Monetary Policy and removes the 80/20 allowance (Unified
    Directive 2070, no. 22, point 5(2)).
  - BFIs still short had to meet the minimum **in paid-up capital (ordinary shares)** by **end
    of Asar 2071**, with capital plans due by end of Asoj 2070.
  - It restates no amount, so the figure is the NPR 2,000 mn above.
- **Consequence for the design:**
  - All 16 banks are already above NPR 2,000 mn in the pre window (minimum: SBL, 2,031).
    This regime had ended (deadline Asar 2071) before Asar 2072 and doesn't bind.
  - With 2,000, dose_pu < 0 for every bank: headroom again, the same problem as the leverage
    floor.
  - The convergence to ≈ NPR 8,000–10,300 mn by Falgun 2075 (§5b item 1) has to come from a
    **later** increase: the 800-crore regime below.
  - The ordinary-share rule means bonus and rights shares count, but Tier 2 doesn't. So the
    bulletin's 'A. PAID-UP CAPITAL' row is the regulated quantity.

**Paid-up capital requirement, the regime inside the window (2026-11-04).** Source: NRB Unified
Directive 2073 (एकीकृत निर्देशन २०७३, consolidated through 2073-06-11), supplied by the user as
Preeti-font text and decoded here; check against the PDF before quoting.
- *Directive 23 (विविध व्यवस्था), point 4 "पूँजी वृद्धि सम्बन्धमा"*, implementing the FY 2072/73
  Monetary Policy.
  - (1) Minimum paid-up capital, Rs crore:

    | Class | National | 4–10 districts | 1–3 districts |
    |---|---|---|---|
    | क (commercial banks) | **800 = NPR 8,000 mn** | – | – |
    | ख | 250 | 120 | 50 |
    | ग | 80 | 80 | 40 |

    The district-level columns exclude the Kathmandu valley.
  - (2) Existing BFIs must reach it **by end of Asar 2074**. A capital plan was due **by end of
    Bhadra 2072**. Bonus shares proposed from the FY 2073/74 financial statements may count.
  - (3) If unmet by Asar 2074: no cash dividend or bonus, and bans on deposit collection, lending
    and branch expansion.
  - (4) The capital plan must be published on the bank's website.
- *Directive 23, point 6:* for the capital plan, paid-up capital means ordinary-share paid-up
  capital including bonus and rights shares, and may include proposed ordinary bonus shares.
- *Directive 10, point 16(ख)* still cites the old FY 2064/65 – Asar 2070 schedule, consistent with
  the 200-crore regime above.
- **Consequence for the design:**
  - `paid_up_capital_requirement: 8000` in all three configs. The observed convergence (15 of 16
    banks at ≈ 8,000–10,300 by Falgun 2075) matches it, but the figure is set from the text, not
    from the data.
  - **Timing caveat.** The requirement comes from the FY 2072/73 Monetary Policy, and the capital
    plan was due by Bhadra end 2072 (stated in the text). So the announcement is at or before the
    start of the Asar–Mangsir 2072 pre window, and the frozen dose_pu is partly
    *post-announcement*. Banks could already be raising capital (rights issues, bonus plans)
    inside the pre window: anticipation biases dose_pu toward 0. The exact announcement month is
    not in the directive and is inferred, not asserted.
  - The compliance deadline (Asar 2074) is inside the clean window (post = Shrawan 2073 onward),
    one year after the leverage-floor effective date. dose_pu's natural timing differs from the
    leverage dose's (§5b item 1, next step iii).
  - Kumari ≈ 7,200 at Falgun 2075 is below 8,000. Not explained here. Candidates to check
    (a merger in progress, a merger-related exemption, a sanctioned shortfall) are not
    asserted.

**Other NRB circulars in or near the pre window (annexed to Unified Directive 2073).** The
Baisakh 12, 2072 earthquake (`@)&@ a}zfv !@ ut]sf] e"sDk`) is the source for that date.
- Circulars 22–30/071/72 (2072-01-14 to 2072-03-09): relief, CRR week merging, a reconstruction
  refinance facility (up to Rs 25 lakh in the valley, 15 lakh elsewhere, at 2%), and
  loan-classification relief.
- Circulars 07/072/73 (2072-07-10) and 12/072/73 (2072-09-30): relief for "the recently arisen
  difficult situation" (हालै उत्पन्न असहज परिस्थिति; the text does not name the cause, so it is
  not asserted). Loans repaid by Poush/Chaitra 2072 may stay pass, watch-list provisioning stays
  at 2%, and rescheduling, 180-day TR loans and grace-period extensions are allowed.
- These fall inside the pre window and transition. They can move the provisions and written-off
  outcomes for all banks at once (absorbed by the time effects) but, if the relief uptake
  correlates with the dose, they bias the pre-window means. Report in the limitations.

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

## 5a. Data extraction: process and known errors (reviewer-facing, 2026-11-03)

Repairing the extraction is **out of scope** (the agreed limitation): this paper takes Paper 1's
extracted panel as given, reports how it was produced and where it is known to be wrong, and
makes the estimation code robust to those errors rather than correcting them.

**Process.** Source = NRB's monthly statistical bulletins for banks and financial institutions
(one PDF per month; file naming changes over the sample, e.g.
`Monthly-Statistics-_-2077_03-July-2020.pdf`, `Poush-2078_Publish2.pdf`, `Asar_2080_Publish.pdf`).
Paper 1's `pdf_extract.py` turns each bank's balance sheet and P&L into long-format rows (bank,
BS year, BS month, line item, value NPR mn, source file) —
`nrb_balance_sheet.csv` / `nrb_profit_and_loss.csv`. Panel construction, with the counts from the
2026-11-03 run:
1. Bulletins re-issued as a "-NEW" resubmission supersede the original: 1,844 balance-sheet rows
   dropped.
2. Required fields missing → cell NaN: 22 (bank, month) cells (e.g. Karnali, Mangsir–Chaitra
   2081: no `1 CAPITAL FUND` row).
3. Commercial ("A"-class) banks only: 9 of 42 institutions excluded.
4. Periods from Asar 2072 (the first month $\hat W_t$ exists): 23 of 156 months excluded.
5. Paper 1's complete-case filter: **N = 16 banks × T = 133 months** (Asar 2072 – Shrawan 2083).
6. Paper 2 only: P&L lines are fiscal-year-to-date cumulative (median |Shrawan|/|prior Asar| of
   NIM = 0.087 ≈ 1/12), de-cumulated to monthly flows; each panel's first month is lost.

**Known extraction errors** (found by inspecting raw rows, `diagnose_capital_break.py`):
- *Digit fragments.* In some bulletin files, line items are extracted as single digits (exactly
  1.0 … 9.0) where the true value is in the thousands — e.g. EBL, Asar 2077: paid-up capital 8.0,
  fixed assets 2.0. Sub-items then no longer sum to their parent (EBL's capital sub-items sum to
  253.5 vs. CAPITAL FUND 16,121.1). Severity varies by file: Poush-2078_Publish2.pdf satisfies
  every accounting identity exactly; Asar_2080_Publish.pdf / Poush_2081_Publish_2.pdf corrupt
  top-level items too.
- *Statement totals survive.* In every inspected sheet, `TOTAL Liabilities` = `TOTAL ASSETS`.
- *Size is an item sum.* Paper 1's `size` adds the top-level asset items rather than reading
  `TOTAL ASSETS`, so it inherits the fragments: EBL Asar 2080 item sum 184,323 vs. reported total
  253,224; SCBNL 20,400 vs. 155,258.
- *Visible consequence.* Median CAPITAL FUND/size by bank is plausible in FY2071/72–FY2076/77
  (≈0.05–0.17), then breaks in both directions: from FY2079/80, EBL ≈ 0.001–0.03 (corrupted
  numerator) and SCBNL ≈ 0.4–0.57 (corrupted denominator), plus isolated FY2077/78 outliers
  (≈0.5). These are extraction artefacts, not capital movements.
- Extent: `audit_bs_extraction.py` checks, for every (bank, month, file) sheet, whether
  liabilities and assets sum to their TOTAL rows, capital sub-items sum to CAPITAL FUND, and the
  two totals balance, and counts single-digit values ("frag"). 2026-11-03: 4,094 sheets, 42
  institutions, 156 bulletins. Two layouts: bulletins up to Chaitra 2074 (1,537 sheets) have no
  `TOTAL Liabilities` row, so assets are split off where the top-level numbering restarts and
  liabilities are checked against `TOTAL ASSETS` (the balance check is n/a there). No bulletins
  exist in the CSV for Chaitra 2071 – Jestha 2072 (before the panel starts).
  - *Baisakh 2070 – Falgun 2075 (incl. the whole pre-mandate window and transition):* liabilities
    and assets 96–100% pass in every file, frag 0.0–0.3, median asset-sum gap 0.00%. Exception
    on the capital sub-item check only: Mangsir 2072 – Mangsir 2073 pass 4–8% while the
    top-level checks still pass and frag ≈ 0 — consistent with a change in how those bulletins
    label CAPITAL FUND's sub-items rather than with digit fragments (not verified). It doesn't
    affect `1 CAPITAL FUND` itself, which is part of the passing liabilities sum.
  - *Chaitra 2075 – Asar 2077 (except Bhadra/Ashwin 2076):* 0–31% pass, frag 12–27, median
    asset-sum gap 9–20% (Asar 2076: 222%).
  - *Chaitra 2077; Jestha 2078 – Poush 2079 (except Baisakh and Ashwin 2078):* 88–100% pass on
    liabilities/assets, 64–92% on capital sub-items, frag ≈ 0.
  - *Magh 2079 – Shrawan 2083:* 0–8% pass, frag 11–21, median asset-sum gap 28–48% (up to 729% in
    Baisakh 2080, and one extreme sheet in Falgun 2080).
  - *Totals balance (TOTAL Liabilities = TOTAL ASSETS) in 100% of sheets that have both rows*, in
    every file, so totals are reliable even where line items are not.
  By fiscal year, the share of sheets passing the asset identity is 100% for FY2069/70 –
  FY2074/75, then FY2075/76 71%, FY2076/77 22%, FY2077/78 28%, FY2078/79 91%, FY2079/80 50%,
  FY2080/81 onward ≤ 2%.

**What this does to the estimates (state in the limitations section).**
- The dose is frozen over Asar–Mangsir 2072 only, so dose accuracy depends on those six bulletins'
  extraction quality alone. All six pass the liabilities and asset identities for every bank
  (frag ≤ 0.1), so $\overline{\text{dose}}_i$, the frozen controls and pre-window `size` rest on clean
  files. Errors in later files do not move $\overline{\text{dose}}_i$.
- The first ~32 post-mandate months (Shrawan 2073 – Falgun 2075) are also clean; corruption starts
  with the Chaitra 2075 bulletin. Robustness spec `experiments/configs/clean_window.yaml`
  (`post_window_end: [2075, "Falgun"]`) re-estimates everything on that clean sample only — a
  sample restriction, not a repair. Results (2026-11-03, T = 45, post = 32 months): every pooled
  DiD coefficient is null under both networks, as in the full panel (smallest wild p: provision
  exposure 0.062 W_hat / 0.079 RAS, same sign in both, but ≈0 or negative in the full panel, so not
  robust). The clean-window CR1 SEs are *smaller* than the full panel's for every dose coefficient
  (e.g. NIM 0.021 vs 0.069, net profit 0.016 vs 0.024), so the null is not a loss of power from
  dropping ~88 months; the corrupted months add noise. Two full-panel event-study patterns don't
  survive: the constant written-off dose path (≈ −0.052 in every bin) becomes ≈ +0.003, and
  the NIM exposure path keeps its point estimates (≈ −5e-5, including the transition bin) but
  loses significance (p 0.18–0.43), which is consistent with a level shift against the short
  reference bin rather than a dynamic effect. Loans written off has only 7 clusters in the clean
  window (582/720 cells missing), so its wild-bootstrap p-values are unreliable there. Below-floor
  cells are 0 in FY2073/74–FY2075/76 on clean files, which supports reading the later counts as
  extraction artefacts.
- Outcomes are divided by `size`, and the Eq. A′ controls are pre-window means. Post-period `size`
  corruption therefore enters the outcomes as non-classical measurement error that is correlated
  within files (i.e. within periods), partly absorbed by $\tau_t$ only if it hits all banks
  equally, which it doesn't.
- `capital_ratio` errors and the post-period below-floor counts (FY2080+) are artefacts; they
  don't enter the frozen-dose DiD but do enter Paper 1's covariates and hence $\hat W_t$ (inherited,
  see §0a).
- The 4% leverage floor binds for 1 of 16 banks before the mandate even on the (cleaner) early
  files — a substantive limit on identification, separate from extraction quality.

**Code-level safeguards (no data repair).** `fit_twfe` drops cells where any regressor is
non-finite (counted), reports regressors with no variation after FE removal as NaN instead of
failing, and returns a NaN p-value when the CR1 SE is zero or undefined. `fit_event_study` drops
bins with no valid outcome cell and returns an empty result, with a note, if the reference bin has
none. The run sets `size ≤ 0` to NaN before dividing or taking logs. Every drop is printed and
saved in the results JSON.

## 5b. Reviewer pre-mortem: skeptical questions per result (2026-11-03, before drafting)

Same exercise as Paper 1's §10a, written from a skeptical econometrics referee's seat against the
results actually obtained (full panel + clean window, §5a). Every item says what we can already
answer from the existing output, and what (if anything) still needs to be run. Status tags:
**[answered]** = existing output suffices, just write it up; **[cheap]** = a small script or
printout over the existing results/CSVs; **[new analysis]** = new estimation code;
**[limitation]** = can't be fixed in scope, so state it plainly. Ranked by how damaging each
objection would be if left unanswered.

**Headline result: pooled DiD null for all four outcomes, both networks, both samples.**

1. **"Your treatment barely treats anyone. What is β identifying?"** (most dangerous)
   Only 1 of 16 banks is below the 4% floor before the mandate (dose_pre range [−0.081, 0.0003]),
   and 0 clean-file cells are below it in FY2073/74–FY2075/76. So β is the effect of *headroom
   above* a floor that doesn't bind, not of being forced to recapitalise. A null is then the
   expected outcome, not evidence that capital regulation has no spillovers.
   - Reframe the estimand as "intensity of regulatory pressure (headroom)" and say explicitly that
     the leverage floor was slack. **[answered]**
   - Minimum detectable effect: MDE ≈ 2.8 × SE (5% two-sided, 80% power), scaled by SD(dose_pre)
     and SD(exposure) and expressed in outcome units (e.g. bp of ROA), so the null is quantified as
     "effects larger than X are ruled out". **[cheap]** (SEs and dose_pre are in the JSONs)
     **Done 2026-11-03 (`power_mde.py`, `results/power_mde.md`).** MDE = (t₀.₉₇₅ + t₀.₈₀; G−1 df)
     × SE_CR1 ≈ 3.0 × SE at G = 16. Standardised MDE = the within-bank-SD change in y from a 1-SD
     higher regressor that the design detects with 80% power. SD(dose_pre) = 0.020 across banks;
     SD(exposure × post) = 37 (clean) / 47 (full).
     - *Dose* (net profit, provisions, NIM): std. MDE 0.10–0.56 in the full panel, 0.33–0.70
       in the clean window. The clean window has smaller SEs, but its within-bank SD of y is
       smaller too, since there is less extraction noise. So the clean-window null rules out
       large own-dose effects (≳ 0.3–0.7 within-bank SD per SD of dose), not moderate ones.
     - *Exposure*: std. MDE 0.21–0.91 (full) and 0.48–1.56 (clean). The spillover null is
       weak; it only rules out effects of roughly one within-bank SD or more. State it that way.
     - *Loans written off*: std. MDE 2–4,000. The outcome is sparse (7 clusters in the clean
       window), so it carries no information. Report it as descriptive only.
     - All |β|/MDE < 1 except **clean-window provision exposure under W_hat: |β|/MDE = 1.86**
       (CR1 t ≈ 5.6) but wild p = 0.062. That much disagreement between CR1 and the wild
       bootstrap is the signature of one or two high-leverage banks: leave-one-out (item 6)
       should identify them. RAS gives |β|/MDE = 0.70.
     - Paper wording: "the design can detect own-dose effects of about half a within-bank SD;
       it is underpowered for spillovers of less than about one SD."
   - Whether the *binding* 2015-16 constraint was a different instrument: the paid-up-capital
     increase in the 2072/73 Monetary Policy, **confirmed 2026-11-04 from Unified Directive 2073,
     dir. 23 pt. 4: NPR 8,000 mn by Asar end 2074** (§4a). It is the natural alternative dose:
     (required − actual paid-up capital) / size.
     - **Coded 2026-11-04 (`reviewer_checks --only paid_up_dose`, config field
       `paid_up_capital_requirement`, set to 8000 in all three configs).**
       - Definition: dose_pu = (requirement − paid-up) / size, frozen at the pre-window mean.
         It uses the leverage dose's timing; if the directive has its own dates, it needs its
         own timing.
       - Row matching: the balance-sheet row is found by the keywords PAID + CAPITAL. It
         matched only 'A. PAID-UP CAPITAL', with 2,128 cells (the whole N × T panel) and no
         conflicting values. The only other CAPITAL label is '1 CAPITAL FUND' (the main dose).
       - Pre window (Asar–Mangsir 2072, 6 cells per bank):
         - paid-up capital: median NPR 2,985 mn, range 2,031 (SBL) to 9,861 (ADBNL);
         - paid-up / size: 0.023–0.082;
         - corr(paid-up / size, dose_pre) = −0.26, so it is largely a different treatment
           from the leverage dose.
       - **What the data show (not a policy claim):** paid-up capital grows by 1.05–4.38× from
         the pre window to Falgun 2075. Growth is largest where paid-up started lowest (SBL
         4.38, EBL 3.79, SCBNL 3.56) and smallest where it started highest (RBB 1.05,
         ADBNL 1.41, NBL 1.52).
         - Pre mean × ratio gives roughly NPR 8,000–10,300 mn for 15 of 16 banks by Falgun
           2075 (several close to 8,000; Kumari ≈ 7,200; ADBNL ≈ 13,900).
         - That convergence to a common level is what a uniform *absolute* paid-up floor would
           produce. If such a floor existed, it would answer item 1: the leverage floor was
           slack, but the paid-up constraint bound for most banks.
         - The figure and its deadline come from the policy text (§4a), not from this
           convergence.
       - Next: (i) done, 8000 from Unified Directive 2073; (ii) re-run the check for Eq. A′ and
         the total effect; (iii) the deadline (Asar 2074) differs from the leverage timing and the
         announcement precedes the pre window (§4a timing caveat), so dose_pu may need its own
         pre/transition/post. A per-bank monthly paid-up path (when each bank crosses 8,000)
         would show the effective timing in the data. **[pending the re-run]**
       - **Re-run 2026-11-04 (main.yaml, req = 8,000, n_boot 999).** 14/16 banks below 8,000
         pre-mandate (ADBNL, RBB above); dose_pu_pre range [−0.016, 0.134], SD 0.044 (leverage
         dose SD 0.020).
         - Eq. A′ (from Asar 2072): nothing significant. NIM dose −0.080 (CR1 SE 0.017, wild p
           0.12 W_hat; −0.074, p 0.17 RAS). A CR1 t ≈ 4.6 against a wild p of 0.12 points to a
           few influential clusters.
         - Total effect (dose-only, from Baisakh 2070), paid-up vs leverage dose:

           | outcome | β (paid-up) | wild p | lead p | 1-SD effect / pre mean | leverage β, p |
           |---|---|---|---|---|---|
           | NIM | −0.057 | **0.003** | 0.84 | −0.0025 / 0.0080 (≈ −31%) | 0.020, 0.57 |
           | net profit | −0.041 | **0.023** | 0.87 | −0.0018 / 0.0042 (≈ −43%) | −0.007, 0.70 |
           | provisions | −0.001 | 0.92 | 0.94 | ≈ 0 | 0.012, 0.33 |
           | written off | −0.058 | 0.97 | 0.16 | 7 clusters, uninformative | 0.077, 0.97 |

         - Within the 4 paid-up tests: Holm keeps NIM only (0.012); BH keeps NIM and net profit.
           The leads are flat.
         - **Not yet a causal claim.** dose_pu = 8,000/size − paid-up/size. The 8,000/size term
           ranges 0.053–0.192 and paid-up/size only 0.023–0.082, so dose_pu is mostly *inverse
           bank size*. The result could be a small-bank-vs-large-bank trend over 2073–75 that
           the flat leads alone don't rule out. Other open threats: the effect sizes are large;
           a size-denominator channel (capital raising grows assets and shrinks the ratio
           outcomes, cf. item 11c); SBL's merger (≈ Asar 2073) inside the window; anticipation
           (§4a timing caveat).
         - Checks to run before the paper uses it: (a) split dose_pu into its two terms
           (8,000/size_pre and paid-up/size_pre, each × post); (b) add log size_pre × post;
           (c) frozen-size outcomes; (d) leave-one-out; (e) `paid_up_path`. **[pending]**
           - Correction: (b) is already in place. `build_inputs` puts frozen log size among
             the Z_pre × post controls of every fit, so the result holds after a linear log-size
             control. (a) is the stronger version (it lets size enter as 1/size, the dose's own
             functional form).
           - Coded 2026-11-04: (a), (c), (d) in `reviewer_checks --only paid_up_dose`
             (`total_effect.fit_total_effect` gained `extra_doses`, `denominator`, `drop_bank`);
             (e) is `--only paid_up_path` (paid-up at each Asar, first month ≥ 8,000 vs the
             Asar 2074 deadline, share of the increase in place by Mangsir 2072).
             **[pending the user's run]**
           - **Run 2026-11-04 (main.yaml, n_boot 999):**
             - (a) Decomposition. The paid-up term carries the effect: NIM −0.062 (wild p 0.011),
               net profit −0.041 (p 0.017). The size term doesn't: NIM 0.011 (p 0.65), net
               profit −0.106 (p 0.16). corr(size term, log size_pre) = −0.984, so the log-size
               control absorbs the size term. The test therefore reads *conditional on size*:
               among banks of the same size, those with less paid-up capital per asset had
               lower NIM and net profit after the mandate. That is the comparison an absolute
               floor implies, and it is not a size effect. It is not a "same β on both terms"
               confirmation either, because the size term is not separately identified.
             - (c) Frozen pre-window size as the denominator: the effects shrink and lose
               significance. NIM −0.021 (p 0.38, vs −0.057), net profit −0.020 (p 0.56, vs
               −0.041). The SEs roughly double. Reading: much of the ratio effect runs through
               the denominator. High-dose banks' assets grew faster than their net interest
               income (capital raising financing balance-sheet growth at lower margins is one
               candidate, not tested). This is a real channel but a different claim from "the
               floor cut profits".
             - (d) Leave-one-out (CR1): NIM is robust (range −0.071 to −0.049, no sign flips,
               |t| ≥ 5.1). Net profit hinges on NBL: −0.020 (t −1.28) without NBL, and t −2.07
               without ADBNL. Provisions and written off stay null.
             - (e) `paid_up_path`: 15/16 banks reach 8,000 inside the panel, but only 4 by the
               Asar 2074 deadline. ADBNL and RBB were already above; Global and NBL reach it in
               Asar 2074. Eleven cross 2–15 months late, mostly Bhadra–Chaitra 2074, then SBL
               (Baisakh 2075, +10) and NMB (Ashwin 2075, +15). Kumari never does (7,163 at
               Falgun 2075). Late crossings fit the rule that proposed bonus shares from the
               FY 2073/74 accounts count (dir. 23 pt. 4(2)): they would appear in the bulletin
               only after issue. Not verified. Most of the increase happens in FY 2073/74 and
               2074/75, inside post. Within the pre window (Asar → Mangsir 2072), only Global
               (+1,152), Laxmi (+701) and NMB (+1,755) raise paid-up capital, so anticipation
               is limited to 3 banks. The printed share started at Baisakh 2070, which counts
               pre-announcement raises (NBL's 58% is its FY 2070/71 raise); it now starts at
               Asar 2072.
             - **Reading for the paper:** the paid-up floor bound (most banks raised 2–4× in
               FY 2073–75, close to the deadline), unlike the leverage floor. Banks further
               below it, at a given size, saw NIM per asset fall. That NIM result is robust to
               leave-one-out but mostly a denominator / asset-growth channel. The net-profit
               result is fragile (NBL). Before the paper can call it a mandate effect, it
               needs: a first stage (dose_pu → paid-up capital raised), a direct asset-growth
               test (dose_pu → log size), and a horse race with the leverage dose.
               Coded 2026-11-04 in `paid_up_dose`: (f) first stage = total effect of dose_pu
               on paid-up / size_pre (dose units) and log paid-up, quarter-end stocks, no
               differencing; (g) asset growth = the same on log size; (h) horse race = dose_pu
               and the leverage dose entered jointly on the four P&L outcomes, each with its
               own lead / transition / post.
           - **Run 2026-11-04, (e) re-run and (f)–(h) (main.yaml, n_boot 999):**
             - (e) The corrected anticipation share (from Asar 2072): median 0.00, range
               0.00–0.24, so nearly all of each bank's increase comes after the pre window.
             - (f) **First stage holds.** paid-up / size_pre: β 0.358 (wild p 0.005), lead
               −0.037 (p 0.27). log paid-up: β 9.99 (p 0.007), lead −0.83 (p 0.24); a 1-SD
               higher dose means ≈ +0.44 log points (≈ 55%) more paid-up capital in post. β is
               averaged over all post quarters, most of them before the banks crossed 8,000, so
               0.36 is not a compliance rate. Caveat: with bank FE this also fits plain
               convergence (low-capital banks catching up); the flat lead argues against a
               pre-existing convergence trend but can't rule out one that starts in post.
             - (g) **Asset growth, borderline.** log size: β 3.04 (p 0.053), 1-SD ≈ +0.13 log
               points (≈ 14%) faster balance-sheet growth. The lead is also borderline:
               −0.82 (p 0.066). The lead compares the early pre bins with the reference bin,
               so a negative lead means high-dose banks were smaller *before* the reference
               bin relative to it: they were already growing faster. (Corrected 2026-11-04:
               the first write-up read the sign backwards.) The post rise may therefore
               continue a pre-existing small-bank growth trend; dose_pu is mostly inverse
               size, so this is size convergence, not a reversal.
             - (h) **Horse race: the paid-up dose carries NIM.** NIM: paid-up −0.057 (p 0.013),
               leverage 0.015 (p 0.18). Net profit: paid-up −0.042 (p 0.135; same point
               estimate, the wild p rises), leverage −0.010 (p 0.48). Provisions and written
               off null on both. corr(dose_pu_pre, leverage dose_pre) = −0.18, so the two are
               nearly separate sources of variation.
             - **Reading for the paper (updated):** the paid-up floor bound (first stage,
               flat lead, almost no pre-window anticipation), the leverage floor did not, and
               the paid-up dose, not the leverage dose, predicts the post-mandate NIM per asset
               fall (horse race, LOO-robust). The fall is mostly a denominator effect: income
               relative to the frozen pre-window size is not significantly lower (c), and
               high-dose banks' assets grew faster (g, p 0.053). So the defensible claim is
               "the paid-up mandate made more-exposed banks raise capital and expand their
               balance sheets faster than their net interest income", not "the mandate cut
               profits". The net-profit result is not robust (NBL in d, p 0.135 in h). Open:
               the log-size lead (p 0.066), SBL's merger (≈ Asar 2073) inside the window, and
               convergence as an alternative reading of (f).
           - **Run 2026-11-04, `paid_up_threats` (main.yaml, n_boot 999), the three open points:**
             - **Pre-trend (stock event study, ref bin −1 = Shrawan–Mangsir 2072).** Every
               pre bin is negative for all three stocks; log size is −1.12 (p 0.081) in FY2070/71
               and −0.45 (p 0.061) in FY2071/72, so high-dose (small) banks were already growing
               faster before the mandate. Paid-up/size_pre has small pre bins (all p ≥ 0.15),
               then a jump in FY2074/75, the year after the Asar 2074 deadline: 0.11 (p 0.39) in
               FY2073/74, then 0.49 (p 0.000) and 0.60 (p 0.001).
             - **With dose × trend and log size_pre × trend** (linear in years since Mangsir
               2072, whole panel): the first stage survives (paid-up/size_pre 0.39, p 0.022; log
               paid-up 7.87, p 0.007). Asset growth does not (log size 3.04 → 1.67, p 0.24). **NIM
               does not** (−0.057 p 0.003 → −0.017 p 0.53, SE 0.008 → 0.021), and net profit
               flips sign (+0.028, p 0.088). With trends, the leads of the first-stage outcomes
               become significant (0.22, p 0.000). This doesn't mean a new pre-trend: the
               whole-panel line is pulled by the post jump, so the pre cells sit above it. That
               is the stated caveat, a whole-panel trend absorbs part of a ramping effect.
             - **Mergers.** Clean-month |Δlog size|: median 0.021, p99 0.146 (n = 1,679).
               Lumbini's last bulletin is Jestha 2073. SBL's Asar 2073 change is +0.144 (just
               under p99; Asar is fiscal year-end) and paid-up +585. If the whole of Lumbini (size
               27.8k, paid-up 2.3k) had been added, the jump would be ≈ +0.34 log. So the bulletins
               don't show a full absorption in Asar 2073; when and how the merger enters SBL's
               balance sheet is unverified. RBB shows no merger-sized change around Baisakh 2075
               (largest +0.099). Robustness: the first stage holds when SBL or RBB is dropped and
               in the SBL + Lumbini pro-forma (paid-up/size_pre p 0.008–0.034; SBL's dose_pu_pre
               0.106 → 0.049 pro-forma). log size loses significance (p 0.15 drop SBL / pro-forma).
               NIM in the horse race holds everywhere (−0.057 to −0.061, p 0.001–0.014; leverage
               null). Net profit is significant only when RBB is dropped (p 0.013), p 0.14–0.20
               otherwise.
             - **Floor, not catch-up.** At Asar 2075, 10 of 16 banks sit in [8,000, 8,400) and 11
               in [8,000, 8,800); every placebo threshold has at most 1 (2 at 7,000). For the 14
               banks below 8,000 at Mangsir 2072, the rank of the start level doesn't predict the
               end level (Spearman 0.011, p 0.97), and the coefficient of variation falls 0.41 →
               0.05 (end median 8,043, range 7,163–8,888). Raise / shortfall is 1.00–1.04 for 10
               banks (median 1.01 over the 14): they raised almost exactly up to 8,000. The
               exceptions are Global 1.48, SBL 1.08, Kumari 0.85 (still −837) and NMB 0.90 (−397).
               Pooled OLS: raise = 1,627 + 0.68 × shortfall (R² 0.73, permutation p 0.0004). The
               slope is below 1 and the intercept positive because of the two banks with no
               shortfall that raised anyway (ADBNL +4,077, RBB +416). Raises by fiscal year: NPR
               3.4–6.8k a year in FY2069/70–2071/72, 13.5k in FY2072/73, 33.6k in FY2073/74 (the
               deadline year), 26.7k in FY2074/75, then 7.2k (partial year).
             - **Reading for the paper (after the threats check):**
               - The paid-up floor bound. The first stage is robust to dose-specific trends and
                 mergers, and its shape (bunching just above 8,000, start rank erased, raise =
                 shortfall) is a floor, not catch-up.
               - The downstream NIM-per-asset effect can't be told apart from a pre-existing
                 small-bank growth trend. Its channel is the denominator (frozen size kills it,
                 c), log size has negative pre bins, and both lose significance with
                 dose-specific trends.
               - Defensible claim: "the paid-up mandate bound and made exposed banks raise
                 capital to the floor; we find no robust effect on profitability or margins
                 beyond pre-existing convergence".
               - Net profit is not robust.
               - Still open: the whole-panel trend is conservative. The decisive version fits
                 the trend on pre cells only and extrapolates it (or a Rambachan–Roth bound),
                 together with an NIM event study by bin.
           - **Run 2026-11-04, `paid_up_trend` (main.yaml, n_boot 999): pre-only trend.** Its
             no-trend and whole-panel rows reproduce `paid_up_threats` exactly. The dose and log
             size_pre slopes are fitted on Baisakh 2070 – Mangsir 2072 only; transition and post
             get their own level and slope; τ (mean post date) = 1.96 years.
             - **First stage survives:** paid-up/size_pre 0.89 (p 0.013); bins 0–2 are 0.60,
               1.05, 1.11 (p 0.010–0.014). Smoothness breakdown M* = 0.118/yr = 0.70 × the pre
               slope. But the magnitude is not credible: bins above 1 mean raising more than the
               whole shortfall per unit of dose. That comes from extrapolating a noisy negative
               pre slope (−0.168/yr, SE 0.11) two years out. Robust: the sign and significance
               under all three versions. Magnitude: no-trend 0.36 together with the cross-section
               raise/shortfall ≈ 1.01.
             - **NIM, no trend: a clean-looking event study.** Pre bins 0.0014 (p 0.80) and
               −0.0003 (p 0.94); post bins −0.064, −0.060, −0.040 (p 0.004, 0.001, 0.006).
               Bin −4 has no NIM cells. With the pre-only trend: pooled +0.020 (p 0.65, SE 0.036
               vs 0.008), bins −0.027 / +0.041 / +0.078, all p ≥ 0.17. The fitted pre slope is
               −0.030/yr (SE 0.012) even though the pre bin means are flat. So it comes from
               within-year variation in an unbalanced pre sample: about 5–7 pre quarters per
               bank, FY2071/72 has only Q1–Q2 (no BS rows Chaitra 2071 – Jestha 2072), and bin −1
               is the Ashwin quarter alone. A dose-specific seasonal pattern there could produce
               it. CI(M = 0) [−0.056, 0.096] contains both 0 and the no-trend −0.057.
             - **Net profit and provisions:** null under the pre-only trend (p 0.81, 0.28). Net
               profit's no-trend bins are −0.039 / −0.047 / −0.033 (p 0.016–0.049) with flat pre
               bins. Log size: pre-only 1.37 (p 0.68), bins 1.1–1.6 with SE 2–4.
             - **One argument against "continued convergence":** in the pre period, high-dose
               banks' assets already grew faster (log size pre bins −1.12 → −0.45 → 0), yet
               their NIM/size did not fall (flat pre bins). Pre-mandate size convergence alone
               didn't lower NIM per asset; after the mandate it did.
             - **Reading for the paper (final for item 1):**
               - The paid-up floor bound. This is robust to every check: trends, mergers, catch-up
                 and leave-one-out.
               - NIM per asset fell for more-exposed banks after the mandate in the no-trend
                 event study (flat pre bins, all post bins p < 0.01; net profit similar but
                 weaker). The channel is mostly asset growth (c).
               - This is not robust to dose-specific linear trends. About 2.5 years of
                 unbalanced pre quarters can't pin down such a trend (SE ×4.6), so the data
                 neither confirm nor rule out an effect beyond trend.
               - Report it as suggestive, with the no-trend event study as the main exhibit
                 and the trend versions and the bound beside it.
               - Not done: a trend fitted on same-quarter (Ashwin) cells only, which would
                 remove the within-year mix behind the NIM pre slope.
2. **"Survivorship: did capital-pressured banks merge and drop out of your panel?"**
   N = 16 of 33 commercial banks pass Paper 1's complete-case filter. If low-capital banks merged
   during 2073–2076 (merger activity around the capital increase is to be verified, not
   asserted), the panel keeps the banks least affected by the mandate, which would bias β toward
   zero. Compare the excluded banks' pre-window dose and size with the included ones' (Paper 1's
   `eda_survivorship.py` already buckets by size/activity), and list the exits with their dates.
   **[cheap]** **Done 2026-11-03 (`reviewer_checks.py survivorship`, baseline):**
   - Of the 17 excluded banks, 8 have no pre-window data at all. Four pairs look like the
     *same bank under a new short name* from Magh 2075: Citizens→Citizen, PRABHU→Prabhu,
     NBB→NBBL and Civil→CBL. In each pair the first series ends in Poush 2075 and the second
     starts in Magh 2075. That points to a bulletin label change, not an exit; verify it
     against the bulletins before stating it.
   - A second group ends in Mangsir 2079 – Jestha 2080 (BOK, Century, Mega, NCC, NIBL, NBBL,
     Sunrise), and NIMB starts in Magh 2079. That fits merger activity (to be verified
     against NRB merger records, not asserted).
   - Lumbini ends in Jestha 2073, Sanima starts in Chaitra 2074, and Century starts in
     Shrawan 2074.
   - **Selection on dose: none detectable.** Pre-window dose_pre median −0.0438 (included,
     n=16) vs −0.0438 (excluded, n=9); Mann-Whitney p=0.25. All nine excluded banks are also
     above the floor.
   - **Selection on size: yes.** Pre-window size median NPR 64.9bn vs 44.5bn, p=0.005.
   - So survivorship doesn't bias β through dose, but the results describe the larger
     commercial banks: state it as an external-validity limit.
   - The label-split banks could only enter Paper 2 if Paper 1 harmonised names and re-ran
     (W_hat covers the 16 banks only), which is out of scope. **[limitation]**
   - **Label-continuity check 2026-11-04 (`reviewer_checks --only label_continuity`;
     continuity in the data only, not verified against NRB records).**
     - 33 commercial-bank labels, Baisakh 2070 – Shrawan 2083. Reference distribution: |log
       size change| between a bank's consecutive complete months, median 0.026 and 99th pct
       2.17 (n = 3,713).
     - Label pairs: all four (Citizens→Citizen, NBB→NBBL, PRABHU→Prabhu, Civil→CBL) have
       0 overlap months and a gap of 1 (Poush → Magh 2075), with size new/old 1.003–1.042 and
       capital fund new/old 0.97–1.00. **Consistent with a relabel.** None of the 8 labels is
       in the 16-bank estimation panel.
     - Exits: Lumbini (Jestha 2073) is the only one inside the clean window, and no bank's
       size jumps above p99 the next month. The others exit in Mangsir 2079 – Jestha 2080, after
       the clean window ends. Two anomalies there:
       - NABIL's size *falls* by NPR 415,114 mn as NIBL exits (Magh 2079);
       - four banks jump together as Sunrise exits (Asar 2080).
       Both look like bulletin-level extraction problems (§5a) rather than single absorbers.
       They are outside the main specification and not pursued (agreed limitation).
     - The p99 of 2.17 (≈ 9× in size) is inflated by the corrupted bulletins (§5a). It is a lax
       threshold, which doesn't change the result, since nothing inside the window needs it.
     - Entrants: Century (Chaitra 2070), Sanima (Kartik 2071), NIMB (Magh 2079). **These
       don't match the survivorship dates above** (Sanima Chaitra 2074, Century Shrawan
       2074). The two checks define "first month" differently (panel start Baisakh 2070 here),
       so reconcile them before quoting either.
     - **Conclusion: no label split, exit or absorption touches the 16 estimation banks in the
       clean window.** Survivorship reduces to the external-validity limit above.
       **[answered]**
     - **Revised 2026-11-04, from NRB's "List of BFIs" (user-supplied; paid-up capital as of
       mid-Jan 2025; `*` = "joint operation date after merger / acquisition"):**
       - Two panel banks have a joint operation date inside the clean window: **Siddhartha
         (SBL), 2016-06-21 ≈ Asar 2073 BS**, and **RBB, 2018-05-02 ≈ Baisakh 2075 BS** (AD→BS
         converted by hand, ±1–2 days).
       - Which institutions were absorbed isn't on the list. Lumbini's exit in Jestha 2073 is
         a candidate for SBL, but unverified.
       - The check above missed both because its p99 threshold (2.17, inflated by corrupted
         bulletins) is far too lax. **The conclusion above is withdrawn** until SBL's and RBB's
         size around those dates is checked against a clean-months-only threshold.
       - The dose is frozen pre-mandate, so it's unaffected; the post-period outcomes and size
         may not be. **[open]**
3. **"There is no pre-trend test."** $\hat W_t$ starts in Asar 2072, and de-cumulation NaNs that
   month, so the event study has no lead bin (§1). But the audit shows the bulletins from Baisakh
   2070 onward are clean, so a **dose-only** event study (no exposure term, which needs
   $\hat W_t$) on Baisakh 2070 – Falgun 2075 would give ~2 years of leads. It tests parallel trends
   for β, though not for γ. **[new analysis, moderate: reuse `fit_event_study` with the exposure
   regressors removed and the panel not cut at Asar 2072]**
   - **Implemented 2026-11-03 as reviewer check `pre_trend`, on quarterly outcomes** (the
     monthly P&L is unreliable off quarter ends, item 11a):
     - Panel: the 16 estimation banks, Baisakh 2070 to the config's end, built with Paper 1's
       `load_panel` + `filter_commercial_banks` + `filter_from_period(start=Baisakh 2070)`.
       Paper 1 code is called, not edited.
     - Same frozen dose_pre and Z_pre × post controls as the main run. Bank + period FE,
       unweighted (item 8 recommendation).
     - Coverage cost: Paper 1's `MIN_PERIOD` comment records no balance-sheet rows for Chaitra
       2071 – Jestha 2072. So FY2071 Q3–Q4 are NaN. The available pre quarters are FY2070
       Q1–Q4, FY2071 Q1–Q2 and Ashwin 2072 (the reference), up to 7 per bank. The check
       prints the missing months and per-bank cell counts.
     - Output: per-bin dose coefficients; a pooled lead (bins < −1) next to the post
       coefficient, for all quarters and for Q1 only; and the 1-SD economic magnitude
       relative to the pre-period mean.
     - **Run 1 (clean window, 2026-11-03):**
       - Missing months: exactly Chaitra 2071, Baisakh 2072 and Jestha 2072. 13 banks have 7
         pre quarters each. Global, NMB and Prime have none for NIM, as in the main run (a
         possible name change in the old bulletins, unverified).
       - **No pre-trend on any outcome.** Pooled leads: NIM −0.007 (p 0.47), net profit 0.008
         (0.71), provisions 0.002 (0.77). All per-bin leads p ≥ 0.29. **[answered for β]**
       - **But the NIM dose shrinks:** post β = 0.020 (p 0.57), with bins 0–2 at
         0.007/0.034/0.015, against ≈ 0.085 in the main quarterly DiD. 1-SD effect: 5% of the
         pre-period mean quarterly NIM, against ≈ 21% at β = 0.085.
       - Loans written off is degenerate (7 clusters, SE 3e-09, so p = 0.000 is meaningless).
       - The spec differs from the main DiD in two ways: no exposure terms, and the 2070–71
         cells feeding the bank FE. Added rows fitting from Asar 2072 only (dose-only, then
         + exposure, the latter should reproduce ≈ 0.082) to tell which one moves β.
         **[answered by run 2]**
     - **Run 2 (clean window, 2026-11-03): the NIM dose effect depends on the specification.**
       - Reconciliation: "from Asar 2072 + exposure" gives 0.0822 (p 0.051), and 0.0547 for
         Q1 only. That matches the main unweighted quarterly DiD and `same_quarter`, so the
         code is consistent.
       - NIM dose post β, all quarters / Q1 only:

         | spec | all quarters | Q1 only |
         |---|---|---|
         | main window + exposure terms | 0.082 (p 0.051) | 0.055 (p 0.16) |
         | main window, dose-only | 0.043 (p 0.21) | 0.024 (p 0.42) |
         | from Baisakh 2070, dose-only | 0.020 (p 0.57) | 0.008 (p 0.80) |

       - Each step roughly halves β, so both causes contribute: the exposure terms, and the
         single Ashwin 2072 reference quarter.
       - Net profit and provisions show the same pattern: adding the exposure terms raises the
         dose coefficient (net profit Q1 0.036 → 0.065; provisions Q1 0.007 → 0.050, p 0.044).
       - Reading:
         - The exposure regressors are built from $\hat W_t$ after the mandate. That network can
           itself respond to the mandate, so for β they are post-treatment ("bad") controls.
           The dose-only β is the cleaner total-effect estimand. The Eq. A′ β is a direct
           effect conditional on the post-mandate network.
         - The single pre quarter is not representative. The 2070–71 quarters change the
           estimate, which is the item 4 risk (earthquake/blockade).
       - **Conclusion: no robust dose effect on any outcome.** The headline stays a null on
         both β and γ, with flat pre-trends (run 1). Report the NIM ≈ 0.08 as a
         specification-dependent estimate, together with this table.
       - The pre-trend sample has no exposure terms (no $\hat W$ before Asar 2072), so γ still
         has no pre-trend test. **[limitation]**
       - Loans written off is degenerate in every row (7 clusters). **[descriptive only]**
       - **[user decision 2026-11-03: the headline β is the dose-only total effect from Baisakh
         2070; Eq. A′ is kept for γ.]** Implemented as `total_effect.py`, called by
         `run_causal_experiment` for quarterly configs. It prints and saves `total_effect` per
         outcome (post β, lead β, 1-SD effect, pre mean), and the report shows it in the DiD
         forest/table. `pre_trend` shares its panel code. The Eq. A′ β (NIM ≈ 0.08) goes into the
         decomposition table above, as the direct effect conditional on the post-mandate network.
       - **Headline run (`main.yaml`, 2026-11-03), dose × post / lead, wild p:**

         | outcome | total effect β | lead β | n_obs, clusters |
         |---|---|---|---|
         | NIM | 0.0196 (0.027), p 0.57 | −0.0066, p 0.44 | 284, 16 |
         | net profit | −0.0073 (0.018), p 0.71 | 0.0078, p 0.69 | 290, 16 |
         | provisions | 0.0119 (0.010), p 0.33 | 0.0019, p 0.76 | 268, 16 |
         | loans written off | 0.077 (2.65), p 0.97 | 0.0084, p 0.11 | 59, 7 — descriptive only |

         It reproduces `pre_trend` run 2 exactly. **Headline: no dose effect on any outcome, and
         no pre-trend.** Figures and table regenerated with three runs (main + the two monthly
         appendix runs).
4. **"Your pre-window overlaps big common shocks."** The pre-mandate window (Asar–Mangsir 2072) falls
   just after the April 2015 Gorkha earthquake and inside the 2015-16 India–Nepal border blockade
   (cite the dates from a source in the draft). τ_t absorbs common shocks; the threat is shocks
   whose bank-level incidence correlates with dose_pre (e.g. earthquake-zone loan books). We have
   no bank-level exposure data, so: state it, and note that a 6-month frozen mean averages over
   the blockade months. **[limitation]**
5. **"You dismiss your significant event-study coefficients ad hoc."** Full panel: NIM exposure
   p ≈ 0.03 in most bins, written-off dose ≈ −0.052 in every bin. Answers we already have: both
   are constant across bins *including the transition*, which is the signature of a level shift
   against a 5-month reference bin; the written-off path disappears and the NIM p-values rise to
   0.18–0.43 in the clean window. The figures (`event_study.png`) make the mechanism explicit.
   Bins 0–1 (FY2073/74–2074/75) contain only clean months in *both* runs, yet the written-off
   dose estimate there is ≈ −0.052 in the full panel and ≈ 0 in the clean window. Corrupted later
   months move the bank fixed effects, which shifts every bin relative to the reference. So the
   clean window is the event study to report, and the full panel goes in an appendix.
   **[answered]** Still missing:
   - a joint test that all post bins = 0, per outcome × coefficient (wild bootstrap Wald), plus a
     count of tests (≈ 4 outcomes × 2 coefficients × 12 bins ≈ 96 in the full panel, ~5 false
     positives expected at 5%) or a Holm/Romano-Wolf adjustment. **[new analysis, small]**
     **Holm/BH done (baseline):**
     - Pooled DiD: 0 of 16 tests have p<0.05.
     - Event study: 16 of 96 have p<0.05, against 4.8 expected under the null. But **0 survive
       Holm or BH** (the smallest Holm p is 0.077, NIM dose bin 5).
     - The excess over 4.8 is not independent evidence: bins share one reference bin, so a
       level shift against it makes many bins significant together (the mechanism above).
     - **Joint wild-bootstrap Wald test done 2026-11-04** (`estimator.fit_twfe(joint=...)`,
       run by `fit_event_study` for every outcome; `main.yaml`):
       - Method: H0 = every post bin in the family is 0 (families `dose_post` and
         `exposure_post`, bins 0–2; the transition bin is in neither family). W = b′V⁻¹b with
         the CR1 V. The WCR Webb bootstrap imposes the joint null on the residuals and
         recomputes each draw's own V. Unit test `joint_wald_rejects_real_effect_not_null_case`
         passes (p_real 0.000, p_null 0.85, and with one bin W = t²).
       - Results (W, wild p):

         | outcome | dose_post | exposure_post |
         |---|---|---|
         | NIM | 26.9, p 0.137 | 30.6, **p 0.038** |
         | net profit | 5.05, p 0.498 | 19.0, p 0.548 |
         | provisions | 7.0, p 0.477 | 26.3, p 0.124 |
         | loans written off | 311, p 0.048 | 2.53, p 0.448 |

       - **None survives Holm or BH within these 8 tests** (smallest Holm p 0.302, NIM
         exposure_post). Over all four families there are 0 Holm/BH rejections. The headline
         total effect gives 0 of 4 at p<0.05.
       - The NIM exposure_post p = 0.038 is the same near-constant level shift seen in every
         bin, transition included (main-run note under item 11), so it isn't a dynamic effect.
         Loans written off has 6 clusters and identical exposure coefficients in every bin,
         so it is unidentified and its p = 0.048 doesn't count **[descriptive only]**.
       - W is large next to a χ²(3) reference (e.g. net-profit exposure, W = 19 but wild
         p = 0.55). With 16 clusters the CR1 Wald statistic is heavy-tailed, so report the
         wild p and never a χ² p. **[answered]**
6. **"16 clusters, one dominant bank."** The WCR Webb wild bootstrap is the standard answer for
   G ≈ 16 **[answered]**, and CR1 vs wild-p disagreements (e.g. written-off β/SE ≈ 2.2 but p ≈ 0.56)
   should be shown, not hidden. Still needed:
   - leave-one-bank-out re-estimation (16 runs of the pooled DiD) to show no single bank drives
     any coefficient. **[cheap–moderate: a loop around `compare_against_ras_baseline`, n_boot can
     be small]** **Done (baseline, CR1, design and weights fixed):**
     - No single bank flips the sign of the net-profit or NIM dose coefficient under W_hat
       (0/16 flips), and at most 1–2 of 16 flip the other coefficients.
     - Exception: W_hat provision exposure flips sign in 10/16 runs. It is centred on zero and
       has no stable sign.
     - The most influential banks are NBL and ADBNL. They are the two dose extremes: NBL is the
       only bank above dose 0 (+0.0003), ADBNL the lowest (−0.081). That is expected
       for a continuous-dose design.
     - Dropping one bank lifts some CR1 |t| to 3.0–3.7 (e.g. RAS NIM exposure without NBL).
       Individually these are not robust (no wild p, and the full-sample estimate is null).
     - Loans written off is driven by NMB: dropping it moves the W_hat dose from −0.025 to
       +0.82. That is more support for treating this outcome as descriptive only.
   - written-off in the clean window has only 7 clusters, so call that outcome descriptive.
     **[answered]**
7. **"$\hat W_t$ is a generated regressor; your SEs ignore its estimation error."** Propagating W
   uncertainty is out of scope (no bootstrap replicates; §0a). Two arguments: (i) the RAS network
   gives the same null; (ii) ignoring first-stage noise makes SEs too *small*, so accounting for it
   would only reinforce a null. **[limitation, with argument]**
8. **"Are the HT/GPS weights well-behaved?"** Extreme stabilised weights would make the estimates
   rest on a handful of cells. Report the weight distribution (max, 99th pct), the Kish effective
   sample size, and an unweighted TWFE comparison. **[cheap: weights are built in `did_design`;
   print them]** **Done 2026-11-03 (baseline): the most serious finding of the pre-mortem.**
   - The stabilised HT weights are extremely concentrated: max/mean = 76 (W_hat) and 68 (RAS),
     and p99/mean ≈ 70.
   - **Kish ESS = 30 (W_hat) and 34 (RAS) out of 2,016 cells (1.5–2%).** The weighted
     estimates rest on the equivalent of ~30 observations.
   - The unweighted TWFE is null for every outcome as well, so the headline null does not
     depend on the weights. But the point estimates move a lot: e.g. NIM dose −0.097 weighted
     vs −0.010 unweighted, and RAS net-profit dose +0.046 vs −0.021.
   - The largest unweighted p-value shift: net profit exposure under W_hat, p=0.058
     unweighted vs 0.30 weighted.
   - Next run (added to the script): per-bank weight shares, plus trimmed weights capped at
     the 95th and 90th percentiles. A likely cause is a GPS model that is overfitted with
     N=16 banks and 7 covariates; the per-bank shares will show which banks take the weight.
   - Drafting decision this forces: either report the unweighted (or trimmed) TWFE as
     co-primary, or show the weighted estimate is not driven by the ~2 banks holding the
     weight. **[open, decision after the per-bank shares]**
   - **Per-bank shares and trimming (both configs, 2026-11-03):**
     - Baseline W_hat shares: NMB 0.32, SCBNL 0.14, Global 0.12, NBL 0.09, Prime 0.09.
     - Clean window: ESS 9 of 608 cells; NBL 0.45, NMB 0.28, ADBNL 0.26. Three banks hold
       essentially all the weight.
     - **In both configs the pre-window cells hold 0.00 of the weight and the post cells
       1.00.** The DiD contrast is therefore implicitly weighted against a pre period that
       carries no weight.
     - Trimmed (q = 0.95, 0.90) and unweighted results: null throughout.
       - The clean-window provision exposure (weighted wild p = 0.062) moves to p ≈ 0.29–0.48
         trimmed and 0.47 unweighted. **It is a weighting artefact of NBL/NMB/ADBNL**, which
         also settles item 1's |β|/MDE = 1.86.
       - Clean-window NIM dose: −0.044 weighted vs +0.003 unweighted.
       - The only p < 0.05 is RAS written-off dose at q = 0.95 in the clean window (0.033). That
         outcome is uninformative (item 1), and the result is one of many tests (item 5).
   - **Diagnosis: a GPS specification flaw, not overfitting.** Every GPS covariate is frozen
     pre-mandate, so the fitted mean exposure is constant over time for each bank. Exposure
     itself is Σ_j W_ij,t dose_pre_j, with W in nominal NPR mn, so it trends with nominal
     interbank volume. A single pooled σ then puts whole stretches of months in the density
     tails, and their 1/density weights explode. The weighting tracks *calendar time*, not
     conditional exposure surprise.
   - Remedy implemented as reviewer check `gps_period`:
     - GPS mean X_i β + α_t, with a per-period σ_t (`propensity.estimate_propensity_period_effects`).
       Invariant to any common per-period shift in exposure (unit test).
     - β stays pooled, because N = 16 is too few to fit it per period.
     - Same regressors; only the weights change.
   - Proposed drafting rule, **[pending the `gps_period` run]**:
     - if ESS recovers and the estimates match the unweighted ones, make period-effects HT
       the main specification;
     - report the unweighted and trimmed estimates alongside;
     - present the pooled-GPS results as the original specification and explain why it was
       replaced.
   - **`gps_period` run (2026-11-03): it fixes the calendar problem, not the bank
     concentration.**
     - Pre-window weight share: 0.00 → 0.08 (baseline, W_hat) and 0.23 (clean).
     - Kish ESS: 30 → 53 (baseline) and 9 → 18 (clean). But max/mean gets *worse* (102–169).
     - NBL still holds 0.50–0.58 of the clean-window weight. This is a positivity/overlap limit
       with N = 16: NBL's exposure is unlike any other bank's given its covariates. No GPS
       specification removes it.
     - Baseline period-effects NIM dose: −0.159, p = 0.002 (both networks). It doesn't survive
       unweighted (p = 0.82) or the clean window (p = 0.90). It is a weighting result, to be
       reported as such.
   - **Recommendation [user decision]:**
     - unweighted TWFE becomes the main estimator;
     - period-effects HT is the Aronow–Samii-faithful robustness check, with its ESS printed;
     - pooled-GPS HT is reported as the superseded original specification.
     - Wording: "with 16 banks, HT weighting cannot achieve overlap; we report it for
       completeness".
   - **[user decision 2026-11-03: accepted.]** Implemented as config field `weighting`
     (`unweighted` | `ht`, `mandate_design.apply_weighting`). Every config is now `unweighted`.
     The weight diagnostics (`reviewer_checks --only weights gps_period`) still build the HT
     weights, so they are reported in full.
9. **"Your dose isn't the regulatory ratio."** Numerator CAPITAL FUND (Tier 1 + 2) vs Tier 1;
   denominator size vs the Exposure Measure (§4a caveats). If the mismatch is roughly
   proportional across banks it mostly rescales β; if not, it's measurement error that
   attenuates β toward zero, which matters for a null. Extraction quality isn't the issue here:
   the six dose-window bulletins are clean and size = TOTAL ASSETS there (gap 0.00%).
   **[limitation + answered]**
10. **"Extraction errors drive your results."** Answered by §5a: totals balance everywhere,
    the dose window is clean, and the clean-window run gives the same null with *smaller* SEs.
    `extraction_audit.png` is the figure for this. **[answered]**
11. **"Outcome construction."** (a) De-cumulation: one misreported month corrupts two monthly
    flows. Report the share of negative de-cumulated flows for lines that should be non-negative
    (interest income, provisions). **[cheap]** **Done (baseline):** negative monthly flows:
    net profit 20.7%, provisions 23.6%, written off 4.6%, NIM 11.9%.
    - Provisions (write-backs) and net profit can legitimately be negative.
    - NIM should almost never be negative, yet 28% of FY2074 cells are. FY2074 (Shrawan 2074 –
      Asar 2075) is a *clean* fiscal year, so this is not the balance-sheet digit-fragment
      problem. It looks like a P&L reporting quirk: YTD restatements, or some bulletins
      reporting a non-cumulative figure.
    - Next run: negative share by calendar month, to show which.
    - The clean-window results depend on this, because FY2074 is inside the clean window.
    - **By calendar month (both configs, 2026-11-03): the problem is tied to the calendar.**
      - Clean window, NIM: negative in 0.00 of cells at Ashwin, Poush, Chaitra and Asar (the
        quarter-end months) vs ≈ 32–41% in every other month.
      - Net profit shows the same pattern: ≈ 0.02–0.05 at quarter ends vs ≈ 0.38–0.48
        elsewhere.
      - Baseline: a weaker version (NIM 0.00–0.02 at quarter ends vs 0.13–0.22 elsewhere).
      - Shrawan is never differenced, yet its NIM is negative in 41% of clean-window cells, so
        the non-quarter-end bulletins' P&L figures are themselves unreliable. This is not a
        differencing artefact.
      - Reading: the early bulletins' P&L is a dependable fiscal-YTD figure only at quarter
        ends, consistent with banks' quarterly financial reporting. The monthly figures in
        between are a different or garbled quantity. This is unverified against the bulletin
        PDFs; it is an inference from the sign pattern.
    - Remedy, a Paper 2-side outcome construction like the clean window (the data is not
      repaired), implemented as reviewer check `quarterly`
      (`mandate_design.quarterly_from_fiscal_ytd`):
      - quarterly flow = quarter-end YTD minus the previous quarter end of the same fiscal year
        (Ashwin = YTD itself), divided by quarter-end size;
      - non-quarter months are never read.
      - Cost: ⅓ of the cells, and only one pre-window quarter (Ashwin 2072; Asar 2072 has no
        preceding Chaitra in the panel). The pre-period contrast is then thin; report it.
    - If the quarterly estimates differ materially from the monthly ones, the quarterly
      construction should become the main outcome. **[pending the `quarterly` run]**
    - **`quarterly` run (2026-11-03): the quarter-end reading is confirmed.**
      - Negative quarterly NIM: 0.035 (baseline) and **0.000** (clean). Net profit: 0.043 and
        0.005. Provisions stay at 0.17–0.25, which write-backs can explain.
      - Pre-window quarterly cells: 13–15 per outcome (≈ one per bank, Ashwin 2072);
        written-off has 2.
      - **Baseline quarterly:** the significant estimates are scattered and none holds across
        weightings (net-profit dose p = 0.004 under main HT but 0.58 unweighted; provision
        exposure p < 0.001 unweighted only). Not robust, as expected with the corrupted later
        years.
      - **Clean-window quarterly NIM dose: the first estimate stable across weighting and
        network.** β ≈ +0.082–0.090 in all six fits: p = 0.052/0.011 (main HT), 0.051/0.017
        (unweighted), 0.038/0.017 (period-effects), W_hat/RAS.
        - Sign: banks with lower pre-mandate capital ratios have higher NIM after the mandate.
          That is consistent with rebuilding capital through margins. The monthly estimate
          (−0.044) had the opposite sign and was built on unreliable months.
        - **Not yet a finding:**
          - Holm within the 16-test family gives ≈ 0.18.
          - Leave-one-out hasn't been run on it, and NBL flips the monthly NIM dose.
          - The pre window is Q1 only, while post spans all quarters, so a dose-related
            seasonal pattern would load onto dose × post.
          - Checks added: `--quarterly` (every check on quarterly outcomes) and `same_quarter`
            (Q1 cells only).
      - Recommendation: clean window × quarterly outcomes as the main specification. The
        monthly construction goes to an appendix, with the calendar-month negative-flow table as
        the reason. **[user decision, after the leave-one-out and same_quarter runs]**
      - **Quarterly leave-one-out + `same_quarter`, clean window (2026-11-03):**
        - NIM dose: **0/16 sign flips** under both networks (W_hat 0.044–0.092, RAS
          0.055–0.098). NBL is the most influential bank, but the estimate stays positive
          without it.
        - Net-profit dose: 0/16 flips (0.010–0.044).
        - NIM exposure flips without ADBNL, and provisions and written-off flip too. The
          spillover stays null.
        - Q1 only (n = 61): NIM dose 0.055–0.089, the same sign and size as the full quarterly
          0.082–0.087, so dose-related seasonality doesn't explain it. p = 0.063–0.165 (the
          sample is a third of the size).
        - Remaining threats: Holm ≈ 0.18, and no pre-trend with one pre quarter (mean reversion
          after the earthquake/blockade, item 4). Item 3 is the decisive check.
      - **[user decision 2026-11-03: accepted.]** Main specification = clean window × quarterly
        outcomes × unweighted TWFE, as `experiments/configs/main.yaml` (config field
        `outcome_construction: quarterly`). `baseline.yaml` and `clean_window.yaml` are the
        monthly appendix runs. Item 3 run 2 then makes the NIM ≈ 0.08 from this specification
        specification-dependent. Still open: whether the dose-only β from Baisakh 2070 is
        reported as the primary β (the total effect), with Eq. A′ kept for γ.
      - **Main run (`main.yaml`, 2026-11-03), tests pass:**
        - NIM dose 0.082 (p 0.040, W_hat) / 0.088 (p 0.011, RAS), matching the reviewer checks.
          Every other pooled coefficient is null (net profit, provisions; p ≥ 0.17).
        - Event study, NIM exposure: p 0.019–0.039 in every bin, *including the transition*,
          with a near-constant level (−1.5e-4 → −1.2e-4), while the pooled exposure is null
          (p 0.18 / 0.08). Same signature as item 5: a level shift against a reference bin that
          holds one quarter per bank (Ashwin 2072). Not dynamics.
        - Loans written off: 6 clusters, identical exposure coefficients in every bin (SE 1.2),
          so it is unidentified. **[descriptive only]**
        - Appendix runs (monthly, unweighted): full panel and clean window null on every
          pooled coefficient. The scattered full-panel event-study p < 0.05 (e.g. NIM dose bins
          2, 6, 8 with alternating signs) fall in the corrupted years and are consistent with
          multiple testing.
      **[open]** (b) Loans written off is missing on 656/2128 cells
    (582/720 in the clean window). Whether a blank means 0 is still open; if yes, it's a coding
    choice to state; if no, the missingness is plausibly not at random. **[open, user decision]**
    **Both versions run 2026-11-04 (`reviewer_checks --only written_off_zero`, `main.yaml`).**
    - The alternative (blank = 0) zero-fills only where the bank reported some other P&L line
      that month. With no P&L row at all, the cell stays missing. Zero-filling is applied to
      the extracted fiscal-YTD figure, before differencing.
    - Extraction (clean window, 16 × 45 cells):
      - 155 cells report written-off, and only 8 of them are an explicit 0 (all ADBNL);
      - 531 cells are blank while other P&L lines are present;
      - 34 have no P&L row (Global, NMB, Prime, 11–12 each).
      - Five banks (HBL, Kumari, NBL, NSBI, RBB) never report the line. ADBNL both prints 0
        and leaves blanks, so for that bank at least, a blank isn't simply an unprinted 0.
        The data can't settle what a blank means.
    - Quarterly outcome: 45 finite cells / 6 banks (missing) vs 210 / 16 (zero). Negative
      share 0.044 vs 0.019.
    - Eq. A′ (blank = 0): null on every coefficient (dose p 0.57/0.87, exposure p 0.73/0.44
      for W_hat/RAS), n_obs 169, 16 clusters instead of 6.
    - Total effect (blank = 0): β = −0.0020 (SE 0.0015), wild p 0.26, lead p 0.39, 290 obs,
      16 clusters. With blank = missing: 0.077 (SE 2.65), p 0.97, 7 clusters.
    - Reading: blank = 0 is the only version with usable inference, and it is null too. So
      the null for loans written off doesn't depend on the coding; only its precision does.
    - Options: (a) keep blank = missing as the main coding (no imputation; outcome stays
      descriptive) and report blank = 0 as a robustness row; or (b) make blank = 0 the main
      coding and state the assumption.
    - **Decision 2026-11-04 (user): (b), blank = 0 is the main coding.** Config field
      `blank_as_zero: [loan_written_off]` in all three yamls, applied in `build_inputs`, the
      total effect and `pre_trend`; `written_off_zero` now shows both versions explicitly and
      labels which one is main. Assumption to state in the paper: a blank written-off row in a
      month where the bank reported other P&L lines means nothing written off so far this fiscal
      year (no P&L row at all stays missing). Blank = missing is the robustness row. The step 4
      JSONs and figures need a re-run.
    - Run 2026-11-04 under the new main: total effect −0.0020 (p 0.26), Eq. A′ null, 16
      clusters. Caveat for the paper: 7 banks (Global, HBL, Kumari, NBL, NSBI, Prime, RBB) never
      report the row in the clean window, so for them written off is identically 0. The
      variation comes from about 6 banks (ADBNL, Laxmi, NABIL, SCBNL, MBL, NMB), so the
      effective number of clusters is lower than the nominal 16. The only explicit zeros are
      ADBNL's 8.
    (c) Scaling by contemporaneous size puts post-treatment size in the denominator; a robustness
    check divides by the frozen pre-window size instead. **[cheap: one line in
    run_causal_experiment]** **Done (baseline):** null for every outcome under both networks
    (smallest p: RAS provision exposure, 0.082). **[answered]**
12. **"Exposure mapping is ad hoc."** First-order neighbours only, linear in $\hat W_{ij,t}$, and
    interbank-only (closed-system bias inherited from Paper 1, §0a). The low corr(dose × post,
    exposure × post) of 0.10–0.18 shows β and γ are separately identified. **[answered]** A
    row-normalised exposure (share-weighted rather than level-weighted) is a natural alternative
    mapping. **[cheap–moderate]** **Done (baseline):** null for every outcome under both
    networks (smallest p 0.107, RAS net-profit exposure). The written-off SEs explode
    (36.9; 3,120 under RAS), so that outcome is unidentified under this mapping. **[answered]**
13. **"Why report an RDD that can't be estimated?"** 0 cells right of the cutoff in every
    specification. Move it to a footnote or appendix as "infeasible: the floor doesn't bind" and
    drop it from the abstract and the method list. It also reinforces item 1. **[answered]**

**Suggested order before drafting:** 1 (MDE) → 2 (survivorship) → 8 (weights) → 11c (frozen size)
→ 6 (leave-one-out) → 5 (joint test) → 3 (dose-only pre-trend). Items 1, 2, 8 and 11c are small
and directly protect the headline null; 3 is the largest piece of code and the main
identification check.

## 6. Paper structure

**Drafted 2026-11-05: see `paper2_draft.md` (+ `book.bib`).** Framing "binding floor, no
spillover"; sections 1–9, data availability, Appendices A–F (monthly outcomes, HT weights,
extraction audit, RDD, written-off coding, full panel). The RDD and alternative propensity
specifications are appendix material, not a robustness section. Outstanding items are in the
draft's pre-submission checklist.

~~Econometrics-venue structure: Intro/motivation → institutional background (2015 mandate) →
identification strategy (DiD + RDD) → data (NRB panel + $\hat W_t$ from Paper 1) → results →
robustness (RDD, alternative propensity specifications per §3) → discussion.~~
