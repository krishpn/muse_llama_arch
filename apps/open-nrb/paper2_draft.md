# A Binding Floor Without Detectable Spillovers: Nepal's Paid-Up Capital Mandate, Bank Profitability and the Interbank Network

**Status:** first full draft, all sections and Appendices A–F (2026-11-05), built from the design doc `paper2_causal_spillover.md`
(§0b, §4a, §5a, §5b). Numbers updated to the 9,999-draw `reproduce_paper.sh` run (2026-11-05). The companion Paper 1 draft
(`paper1_draft.md`) exists and is cited as `paper1companion`. **Target venue:** field
journal (Journal of Financial Stability / Emerging Markets Review), full length with appendices.

**One-line description:** NRB's NPR 8,000 mn paid-up capital floor bound and banks raised capital
to it; we find no robust effect on profitability and no detectable spillover through the
reconstructed interbank network, and we state how large an effect the design rules out.

> **Pre-submission checklist (delete before submission)**
> 1. Every policy quotation (Unified Directive 2073 dir. 23 pt. 4; circulars 25/063-64 and
>    02/070-71; Capital Adequacy Framework 2015 §1.9, §2.7.B) was decoded from Preeti-font text.
>    Check each one against the PDF.
> 2. Every `.bib` entry marked `TO VERIFY` needs checking against the publisher record. Fill in
>    `paper1companion`.
> 3. Add a source for the 2015–16 border blockade dates, marked `[CITATION NEEDED]` in §3.4.
> 4. Literature gap: Nepal-specific work on bank capital and the merger wave. Not sourced; §2.4
>    holds a placeholder.
> 5. Insert the figures from `results/figures/` after `reproduce_paper.sh`: `extraction_audit.png`
>    (Figure 1, `run_event_study_report`), `paid_up_compliance.png` (Figure 2) and
>    `paid_up_event_study.png` (Figure 3), both from `plot_paid_up_figures`. The numbers in
>    `paid_up_figures.md` (2026-11-05 run) match §6.2–6.3.
> 6. Checks (a)–(d) are now numbers in §8, from `reproduce_paper.sh` (2026-11-05). For (c), the
>    absent month is Kartik 2078 (`panel_window_checks`, 2026-11-05); closed. Still not run,
>    stated in §8 without numbers:
>    - (e) the fragment share of the marginals;
>    - (f) Paper 1 re-run on the clean window.
> 7. Unexplained or unverified facts, kept as such in the text:
>    - why Kumari stays below the floor;
>    - which institutions SBL and RBB absorbed;
>    - the Sanima/Century entry dates;
>    - what a blank written-off row means.
> 8. AD dates converted from BS by hand (±1–2 days); check them.
> 9. ~~**Number reconciliation:** the frozen-denominator NIM p-value is given as 0.37 in §6.3 and
>    0.38 in §7.2.~~ **RESOLVED 2026-11-07:** set to **p 0.38** in both §6.3 and §7.2 (the value
>    §7.2 and the design doc already carried; the 0.37 in §6.3 was a transcription slip). Re-confirm
>    against `reviewer_checks_main.md`'s 9,999-draw run if re-verifying. Null either way.

## Highlights

- Nepal's NPR 8,000 mn paid-up capital floor bound; banks raised capital to meet it.
- No robust effect of the mandate on bank profitability is detected.
- No spillover is detectable through the reconstructed interbank network — but the test is weak: it
  rules out only effects of about two or more within-bank SDs per SD of exposure.
- We quantify how large an effect the design can rule out (standardised MDE).
- Published lending and borrowing totals do not balance, limiting any reconstruction.

## Abstract

Capital regulation in small banking systems is often an absolute floor on paid-up capital
rather than a risk-based ratio, and its effects are rarely studied. We study Nepal Rastra Bank's
requirement that every commercial bank hold at least NPR 8,000 million of paid-up capital by mid-July
2017. Our data are a monthly panel of 16 commercial banks hand-extracted from NRB statistical
bulletins, combined with a reconstructed interbank exposure network. We estimate a treatment-intensity
difference-in-differences design with a frozen pre-mandate dose.

The floor bound. Ten of 16 banks ended within NPR 400 million above the floor, and the median
raise was 1.01 times the bank's shortfall. Starting rank did not predict the end level, so this
was compliance with a floor, not catch-up. In the regression, banks further below the floor raised
more paid-up capital. The estimate survives merger adjustments and leaving out any one bank, and
keeps its sign and significance under dose-specific linear trends, although with a whole-panel
trend its pre-period lead is also significant. Profitability shows no robust response.
More-exposed banks saw net interest income per unit of assets fall after the mandate (a one-SD
higher dose lowers this per-asset measure by about 31% of its pre-period mean), with flat
pre-period coefficients.
But the effect runs mostly through faster asset growth, and the data cannot separate it from a
dose-specific linear trend: with the trend fitted on the pre-period, the 95% interval contains
both zero and the no-trend estimate. We therefore report it as suggestive. The Basel III leverage
floor that took effect at the same time was slack for 15 of 16 banks, and we find no effect of
leverage headroom. Network exposure to counterparties further below the paid-up floor has no
detectable effect on any outcome, and neither does exposure built from leverage headroom. This
null is weak: in the main specification the design can detect only spillovers of about two to
five within-bank standard deviations of the outcome per standard deviation of exposure. The two
published interbank marginals do not balance within the sample in 43 of 45 months, so no
reconstruction can match them exactly; detecting smaller spillovers would require bilateral
exposure data.

**Keywords:** capital requirements; paid-up capital; interference; difference-in-differences;
interbank networks; Nepal. **JEL:** G21, G28, C21, C23.

## 1. Introduction

Most of what we know about how banks respond to higher capital requirements comes from large,
risk-based regimes: the EBA capital exercise @mesonnier2015did @gropp2019banks, UK bank-specific
requirements @aiyar2014does, and Spain's dynamic provisioning @jimenez2017macroprudential. Many
emerging-market regulators use a blunter tool, an absolute floor on paid-up capital that every
licensed bank must meet by a deadline. Such floors are easy to supervise, and their incidence is
very uneven: a uniform absolute floor is a small burden for a large bank and a large one for a
small bank. Because banks lend to each other, a floor that forces some banks to recapitalise could
also reach banks that were not themselves short of capital, through their interbank exposures
@allen2000financial @acemoglu2015systemic.

We study one such floor. Under Directive 23 of NRB's Unified Directive 2073, implementing the FY
2072/73 Monetary Policy, every commercial ("Class A") bank had to reach NPR 8,000 million of
paid-up capital by the end of Asar 2074 BS (mid-July 2017) @nrb2073unified. This was four times
the NPR 2,000 million floor that applied until Asar 2071 @nrb2064circular25 @nrb2070circular02.
Over the same period NRB introduced a Basel III leverage-ratio minimum of 4%, with a parallel run
from Poush 2072 to Asar end 2073 (mid-July 2016) @nrb2015caf.

We ask three questions. Did the floor bind? Did banks further below it change their
profitability, margins or provisioning? Did banks whose interbank counterparties were further below
it respond, holding their own position fixed?

**Design.** We build a monthly panel of 16 commercial banks from NRB's statistical bulletins
@nrbbulletins, Baisakh 2070 to Shrawan 2083 BS. Each bank's treatment dose is frozen at its mean
over the six months before the leverage floor's parallel run (Asar–Mangsir 2072). We use two
doses:
- the paid-up shortfall, (8,000 − paid-up capital)/assets;
- the leverage-ratio headroom, 0.04 − capital fund/assets.

The outcomes are quarterly flows of net interest income (NIM), net profit, loan-loss provisions
and loans written off, each scaled by assets. The headline estimand is the dose-only total effect,
estimated from Baisakh 2070 so that it has pre-period leads. Network exposure is the exposure-weighted
sum of each bank's counterparties' doses @aronow2017estimating. The weights come
from the interbank network of our companion paper @paper1companion, which reconstructs each
month's bilateral matrix from the two published interbank marginals under a regulatory exposure
cap.

**Findings.**
1. *The paid-up floor bound.* The main evidence is bank-level compliance, which shows a floor,
   not catch-up:
   - 10 of 16 banks bunch in [8,000, 8,400) at Asar 2075, against at most 2 at any placebo
     threshold;
   - the raise matches the shortfall almost one for one (median 1.01);
   - starting rank is unrelated to the end level (Spearman 0.011).

   The regression first stage supports it. One within-sample SD of dose raises paid-up capital by
   about 55% (wild-cluster bootstrap p = 0.009), with a flat lead. The estimate holds without
   either merging bank and leaving out any single bank. It keeps its sign and significance with
   dose-specific linear trends, but it is not trend-robust in the full sense: with a whole-panel
   trend the pre-period lead is significant (0.22, p 0.001), and with a pre-only trend the
   magnitude comes from extrapolation (Section 6.2).
2. *No robust profitability effect.* In the no-trend event study, banks further below the floor saw
   NIM per unit of assets fall by 0.040–0.064 per unit of dose in every post-mandate year (p ≤
   0.006), with pre-period coefficients of 0.0014 and −0.0003. We do not read this as a causal
   effect on margins. Scaled by frozen pre-mandate assets, the effect shrinks by 63% and
   loses significance. More-exposed banks grew their balance sheets faster, and with a pre-period
   linear trend the confidence interval contains both zero and the no-trend estimate. The
   net-profit effect depends on one bank.
3. *The leverage floor was slack.* Only one bank was below 4% before the mandate. The leverage
   headroom has no effect on any outcome, with flat leads. The design was first specified with
   this dose; we added the paid-up dose after these results came back null (Section 4.4). We
   therefore read the leverage null as a design check, not as a finding about the leverage floor:
   the same estimator and sample find nothing where the constraint did not bind.
4. *No detectable spillover.* Exposure built from the paid-up dose has no effect on any outcome
   under either network (wild p ≥ 0.29). Exposure built from the leverage dose has none either,
   and no exposure coefficient survives a multiple-testing correction. The test is weak. In the
   main specification it can detect only spillovers of about 2–5 within-bank SDs of the outcome
   per SD of paid-up exposure (4–8 for leverage exposure), against 0.4–2 for the own dose.

**Contribution.**
- We document how an absolute paid-up floor binds, bank by bank, using hand-extracted bulletins
  and an extraction audit that we report in full. Small-system floors are seldom studied. That
  banks comply with a sanctioned floor is unsurprising; the contribution is to measure the floor's
  incidence bank by bank — bunching, raise against shortfall, erased rank — with diagnostics that
  need no control group.
- We show that a floor's measured effect on ratio outcomes can run largely through the
  denominator: banks that raise capital also grow their assets. Measured against frozen
  pre-mandate assets, the NIM effect shrinks by 63% and loses significance. The frozen-denominator
  check is cheap, and we suggest it as a routine diagnostic.
- We bring an exposure-mapping interference design
  @aronow2017estimating @leung2020treatment to bank regulation where the network is reconstructed
  rather than observed. At 16 banks the design-based Horvitz–Thompson estimator cannot achieve
  overlap (Section 7.3), so the working estimator is unweighted two-way fixed effects with a
  network-weighted exposure regressor; that the design-based estimator breaks down at this scale is
  itself a finding. We measure what such a network can support.
  NRB's two published
  interbank marginals are zero in most bank-months and do not balance within the 16 banks in 43
  of 45 months, so no reconstruction can match them exactly, and a spillover test built on one can
  detect only large effects.
- With 16 clusters, we state every null together with its minimum detectable effect, use
  wild-cluster bootstrap inference throughout @cameron2008bootstrap @webb2023reworking, and
  correct for multiple testing @holm1979simple @benjamini1995controlling.

The rest of the paper is organised as follows. Section 2 reviews the literature and Section 3 the
institutional background. Section 4 describes the data and the extraction audit, Section 5 the
empirical strategy. Section 6 reports results and Section 7 robustness. Section 8 discusses
limitations and Section 9 concludes.

## 2. Literature Review

### 2.1 Bank responses to higher capital requirements

A bank facing a higher requirement can issue equity, retain earnings, or shrink or re-weight its
assets. Equity issuance is costly when investors are less informed than the bank
@myers1984corporate. The policy debate turns on whether the adjustment falls on lending
@admati2013bankers. Quasi-experimental evidence from Europe finds that banks hit harder by the
2011 EBA capital exercise met it largely by cutting risk-weighted assets rather than raising
equity @gropp2019banks @mesonnier2015did. UK evidence shows that the adjustment can leak to
unregulated lenders @aiyar2014does. Spanish evidence shows that capital buffers built in good times
support credit in bad times @jimenez2017macroprudential. All of these are risk-based ratio
regimes. An absolute paid-up floor is different in two ways. It cannot be met by re-weighting
assets, only by issuing ordinary shares, including bonus and rights shares. And its burden
relative to the balance sheet falls in inverse proportion to bank size. Our first-stage results
show the issuance channel directly.

### 2.2 Interbank networks and spillovers

Theory predicts that shocks travel along interbank exposures @allen2000financial
@eisenberg2001systemic @acemoglu2015systemic, and early simulation studies use observed or
estimated exposures to measure contagion @furfine2003interbank @upper2011simulation. Where
bilateral exposures are not published, the network has to be reconstructed from marginals
@squartini2018reconstruction @anand2018filling, most often by iterative proportional fitting /
RAS @deming1940least, or by sampling consistent networks @gandy2013financial. Our companion paper
@paper1companion builds such a reconstruction for Nepal. Here we use it only as an
exposure-weighting matrix and do not treat it as a validated network (Section 4.3).

### 2.3 Causal inference under interference

When one unit's treatment affects another unit's outcome, the no-interference part of SUTVA
fails @hudgens2008toward. Aronow and Samii @aronow2017estimating define exposure mappings and
design-based Horvitz–Thompson estimators. Leung @leung2020treatment gives inference for treatment
and spillover effects on networks. Tchetgen Tchetgen and VanderWeele @tchetgen2012causal extend
the framework to observational data under conditional ignorability. Our setting is
observational, so any propensity-score weighting has to estimate the generalised propensity score
@hirano2004propensity rather than know it by design. With 16 banks it cannot achieve overlap
(Section 7.3), and our main estimator is unweighted two-way fixed effects. We work throughout in
potential-outcomes notation @imbens2015causal, the idiom of this literature.

### 2.4 Difference-in-differences with a continuous dose

Our dose is continuous and the timing is common to all banks. Two-way fixed-effects estimators
with heterogeneous effects can misweight units under staggered timing
@goodmanbacon2021difference @dechaisemartin2020two. With common timing that concern is smaller,
but with a continuous dose the coefficient is a weighted average of dose-response derivatives, and
identification needs a strong parallel-trends assumption @callaway2024continuous. We test pre-trends,
allow dose-specific trends, and report a smoothness bound in the spirit of @rambachan2023more. We
interpret the pre-tests with the caution in @roth2022pretest @freyaldenhoven2019pre. Inference
with few clusters follows @bertrand2004much @cameron2008bootstrap @mackinnon2017wild
@webb2023reworking @roodman2019fast.

*[LIT GAP, to source before submission: Nepal-specific studies of bank capital, the 2015/16
capital increase and the merger wave that followed. Do not cite from recall.]*

## 3. Institutional Background

*All quotations in this section were decoded from Preeti-font text supplied to the author.
[CHECK EACH AGAINST THE PDF BEFORE SUBMISSION.] Dates are Bikram Sambat (BS); the Nepali fiscal
year runs from Shrawan to Asar (mid-July to mid-July).*

### 3.1 The earlier paid-up regime (to Asar 2071)

NRB circular no. 25 (2063/64, dated 2064-01-02 BS) set a minimum paid-up capital of NPR 200 crore
(NPR 2,000 mn) for national-level commercial banks, to be reached by the end of Asar 2070. It
allowed up to 20% from other primary-capital sources and set sanctions that grew year by year:
- dividends withheld;
- then a ban on taking deposits;
- then a ban on lending;
- then a ban on all business except loan recovery and deposit repayment;
- finally, licence cancellation.

@nrb2064circular25

Circular no. 02 (2070/71, dated 2070-04-27 BS) removed the 20% allowance and required any bank
still short to meet the minimum in ordinary paid-up shares by the end of Asar 2071
@nrb2070circular02. All 16 banks in our panel were above NPR 2,000 mn in the pre-mandate window
(minimum NPR 2,031 mn), so this regime no longer bound.

### 3.2 The NPR 8,000 mn floor

Directive 23, point 4 of Unified Directive 2073, implementing the FY 2072/73 Monetary Policy,
sets the minimum paid-up capital for national-level commercial banks at NPR 800 crore
(NPR 8,000 mn) @nrb2073unified.
- Existing banks had to reach it by the end of Asar 2074 (mid-July 2017). Each bank had to submit
  a capital plan by the end of Bhadra 2072 and publish it on its website.
- Bonus shares proposed from the FY 2073/74 accounts may count toward the floor.
- A bank short at the deadline may not pay cash dividends or bonus shares, and is barred from
  collecting deposits, lending and opening branches.
- Point 6 defines paid-up capital as ordinary-share capital, including bonus and rights shares.

This is the "A. PAID-UP CAPITAL" row of the bulletins' balance sheets. Mergers were a permitted
route: under the earlier circular, merging banks could count their combined capital.

**Timing.** The capital plan was due by Bhadra end 2072, so the requirement was known at or
before the start of our pre-mandate window (Asar–Mangsir 2072). The directive does not give the
month it was announced, and we do not assert one. Banks could therefore have started raising
capital inside the window. In the data only three banks did so, and the median share of a bank's
total raise that took place in the window is 0.00 (Section 6.2).

### 3.3 The leverage-ratio floor

NRB's Capital Adequacy Framework 2015 introduced a Basel III leverage ratio @bcbs2014leverage
(Tier 1 capital over an exposure measure) with a minimum of 4%. It ran in parallel from Poush end
2072 to Asar end 2073 (mid-July 2016) @nrb2015caf. The framework says the final requirement "might
be revised" afterwards and gives no later figure; we apply 4% throughout. Our measure (capital
fund over total assets) is an approximation: the numerator includes Tier 2, and the denominator is
not the regulatory exposure measure (Section 8). The two floors overlap in time but are different
instruments. Across banks, the paid-up dose and the leverage dose correlate at only −0.26.

### 3.4 Concurrent shocks

The Gorkha earthquake of Baisakh 12, 2072 (April 2015) @npc2015pdna occurred two months before our
pre-mandate window. The India–Nepal border blockade of 2015–16 *[CITATION NEEDED for dates]* falls
inside it. NRB circulars 22–30 of 2071/72 provided relief to borrowers: a reconstruction
refinance facility and loan-classification relief. Circulars 07 and 12 of 2072/73 (dated
2072-07-10 and 2072-09-30) gave further relief for "the recently arisen difficult situation". The
text does not name its cause, and we do not attribute one.

Common shocks are absorbed by period fixed effects. A threat remains if their bank-level incidence
correlates with the dose, and we have no bank-level data on earthquake or blockade exposure
(Section 8).

## 4. Data

### 4.1 Source and extraction

NRB publishes a monthly statistical bulletin with each licensed institution's balance sheet and
profit-and-loss statement @nrbbulletins. We extract them from the PDFs into long format (bank,
month, line item, value in NPR mn) with the extraction code of our companion paper
@paper1companion. The corpus has 156 bulletins, 4,094 (institution, month) balance sheets and 42
institutions. Panel construction:
1. Bulletins re-issued as "-NEW" files replace the original (1,844 balance-sheet rows superseded).
2. Cells with a required field missing are set to missing (22 bank-month cells).
3. Only commercial ("Class A") banks are kept; 9 of 42 institutions are excluded.
4. The network sample starts at Asar 2072, the first month the reconstructed network exists.
5. Banks must be observed in every month (the companion paper's complete-case rule).

The result is N = 16 commercial banks over T = 133 months. Asar 2072 – Shrawan 2083 spans 134
calendar months; Kartik 2078 is absent from the bulletins, leaving 133 observed months. The
total-effect panel extends the same 16 banks back to Baisakh 2070. For Chaitra 2071, Baisakh 2072 and
Jestha 2072 the extraction yields no balance-sheet rows, so these months are missing.

### 4.2 Extraction audit

We audit the extraction rather than repair it. For every sheet we check three identities:
- the liabilities sum to their total;
- the assets sum to their total;
- the two totals balance.

We also count single-digit values ("fragments"), a signature of mis-parsed numbers (Figure 1,
Appendix C):
- **Baisakh 2070 – Falgun 2075** (the whole pre-mandate window, the transition and 32
  post-mandate months): 96–100% of sheets pass in every bulletin, with 0–0.3 fragments per sheet
  and a median asset-sum gap of 0.00%.
- **Chaitra 2075 onward:** quality collapses. Chaitra 2075 – Asar 2077 passes 0–31%, and Magh
  2079 onward passes 0–8%, with median asset-sum gaps of 28–48%.
- Statement totals balance in 100% of sheets that report both totals, so totals stay reliable
  where line items are not.
- By fiscal year, the asset identity holds in 100% of sheets through FY 2074/75, then in 71%,
  22%, 28%, 91%, 50% and ≤ 2%.

**Consequence: our main sample ends at Falgun 2075.** The frozen dose uses only the six
pre-window bulletins, all of which pass every identity for every bank. Within the clean window the
sample restriction is a choice, not a repair. The full panel to Shrawan 2083 is reported in the
appendix only. Outside the clean window its estimates mix real variation with extraction noise:
the clean-window standard errors are smaller than the full panel's for every dose coefficient,
even though the clean window has 88 fewer months.

### 4.3 The reconstructed interbank network

Bilateral interbank exposures are not published. Our companion paper @paper1companion
reconstructs a monthly matrix $\hat W_t$ for the 16 banks. Each matrix is fitted to the two
published interbank marginals (lending to and borrowing from banks) as closely as the cap and the
data allow (they cannot be matched exactly in 43 of 45 main-window months; Section 8), respects a
regulatory exposure cap proportional to each bank's capital fund, and is smoothed over time. We call $\hat W_t$ the
*marginal-constrained, capped reconstruction* and use it only as an exposure-weighting matrix. We
do not treat it as a validated network, for three reasons:
- No bilateral ground truth exists, so the reconstruction cannot be validated on Nepali data.
- On real data the companion paper finds that $\hat W_t$ is determined almost entirely by the
  projection onto the marginals, the cap and the smoothness prior. The baseline fit and an
  ablation without KL pressure on the latent posterior give matrices within a relative
  difference of $1.5\times10^{-5}$.
- For the same reason, the RAS reconstruction @deming1940least, which we use
  as an alternative weighting, targets the same marginals. It fits them differently (Section 8),
  but agreement between the two still shows robustness to the cap, the temporal smoothing and
  how the marginals are fitted, not to the reconstruction method.

The companion paper's real-data comparison with RAS uses a temporal-stability criterion that the
model is trained to optimise. We therefore do not cite that comparison as validation.

### 4.4 Variables

**Doses.** With $\text{PU}_{it}$ the paid-up capital, $K_{it}$ the capital fund and $A_{it}$ the
sum of the top-level asset items:
$$d^{pu}_{it} = \frac{8000 - \text{PU}_{it}}{A_{it}}, \qquad d^{lev}_{it} = 0.04 - \frac{K_{it}}{A_{it}},$$
each frozen at its bank mean over Asar–Mangsir 2072: $\bar d_i$.
- 14 of 16 banks are below the paid-up floor in the pre-window (ADBNL and RBB are above). Paid-up
  capital ranges from NPR 2,031 to 9,861 mn (median 2,985), and $\bar d^{pu}_i$ ranges over
  [−0.016, 0.134], SD 0.044.
- One bank is below the leverage floor. $\bar d^{lev}_i$ ranges over [−0.081, 0.0003], SD 0.020.
- *Order of analysis.* The design document specified the leverage dose only. We added the paid-up
  dose after the leverage-dose estimates came back null and showed that the leverage floor was
  slack (Section 6.1). The paid-up estimates are therefore not pre-specified. The timing, sample,
  controls, estimator and inference are the same for both doses; only the dose changed.

**Outcomes.** The P&L lines are fiscal-year-to-date cumulative. The median ratio of Shrawan
(first-month) NIM to the previous Asar (twelfth-month) NIM is 0.087 ≈ 1/12. The monthly figures
between quarter ends are unreliable: in the clean window, NIM is negative in 0% of cells at
quarter-end months and in 32–41% of cells in other months. That includes Shrawan, which needs no
differencing (Appendix A). We therefore build quarterly flows: the quarter-end YTD figure minus the
previous quarter end of the same fiscal year, divided by quarter-end assets. Non-quarter months are
never read. The outcomes are:
- NIM (interest income − interest expense);
- net profit;
- loan-loss provisions;
- loans written off.

A blank written-off row in a month where the bank reports other P&L lines is coded as zero, with
blank = missing as a robustness row (Appendix E).

**Controls.** Pre-window means of the liquidity ratio, loan-to-deposit ratio, NRB-borrowing share
and log assets, each interacted with the post indicator. These are the companion paper's
covariates, without the capital ratio, which is our leverage dose.

**Exposure.** $E_{it} = \sum_j \hat W_{ij,t}\,\bar d_j$, the exposure-weighted sum of
counterparties' frozen doses @aronow2017estimating. Across post-mandate cells, the correlation
between dose × post and exposure × post is 0.10–0.18, so own and network terms are separately
identified.

## 5. Empirical Strategy

### 5.1 Timing

| phase | months (BS) | role |
|---|---|---|
| early pre | Baisakh 2070 – Asar 2072 | leads (total-effect panel only) |
| pre-mandate window | Asar – Mangsir 2072 | dose frozen here; reference bin = FY 2072/73 Q1 (Ashwin 2072 quarter) |
| transition | Poush 2072 – Asar 2073 | leverage parallel run; own coefficient |
| post | Shrawan 2073 – Falgun 2075 | includes the paid-up deadline (Asar 2074) |

The event-study bins are Nepali fiscal years.

### 5.2 Total effect (headline)

$$y_{it} = \beta\,\bar d_i\,\text{post}_t + \lambda\,\bar d_i\,\text{lead}_t + \theta\,\bar d_i\,\text{trans}_t + \delta^\top(\bar Z_i\,\text{post}_t) + \alpha_i + \tau_t + \varepsilon_{it} \qquad (1)$$

Here $\text{lead}_t$ marks pre-period quarters before the reference bin, and $\alpha_i$, $\tau_t$
are bank and quarter fixed effects. Under parallel trends in dose, $\beta$ is the total effect of
a unit of dose and $\lambda = 0$. We do not condition on network exposure in (1). The exposure is
built from the post-mandate network, which the mandate can move: the cap is proportional to each
bank's capital, and capital is what the mandate changes. Conditioning on it would make it a bad
control @angrist2009mostly. Dropping it also lets (1) start in Baisakh 2070, before the network
exists, which gives up to seven pre-period quarters per bank.

### 5.3 Direct effect and spillover

$$y_{it} = \beta\,\bar d_i\,\text{post}_t + \gamma\,E_{it}\,\text{post}_t + \rho\,E_{it} + \delta^\top(\bar Z_i\,\text{post}_t) + \alpha_i + \tau_t + \varepsilon_{it} \qquad (2)$$

$\gamma$ is the spillover: the change in a bank's outcome when its counterparties are further
below the floor, holding its own dose fixed. $\rho E_{it}$ identifies $\gamma$ from the pre/post
change rather than from cross-sectional connectedness. Equation (2) can only start at Asar 2072,
so $\gamma$ has no lead test. Its $\beta$ is a direct effect conditional on the post-mandate
network and is reported for comparison only. Under interference, ignorability is conditional on
observables (frozen covariates), not guaranteed by design @tchetgen2012causal.

### 5.4 Inference

All standard errors are CR1, clustered by bank (G = 16). Every p-value is from a WCR wild-cluster
bootstrap with Webb six-point weights and 9,999 replications @cameron2008bootstrap
@webb2023reworking @roodman2019fast. Joint hypotheses (all post bins = 0) use a bootstrap Wald
test that imposes the joint null; we never use a χ² reference. Multiple testing is controlled by
Holm @holm1979simple and Benjamini–Hochberg @benjamini1995controlling within families.

**Minimum detectable effect.** MDE = $(t_{0.975} + t_{0.80};\, G-1) \times \text{SE} \approx 3.0
\times$ SE at G = 16. We express each MDE as the change in the outcome from a one-SD higher
regressor, in within-bank SDs of the outcome.

The weighted Horvitz–Thompson estimator with an estimated generalised propensity score
@aronow2017estimating @hirano2004propensity is reported as robustness (Section 7.3). The main
estimator is unweighted, because with 16 banks the weights cannot achieve overlap.

## 6. Results

### 6.1 The leverage floor was slack

Before the mandate, 1 of 16 banks was below the 4% leverage floor. In the clean bulletins, no
bank-month is below it in FY 2073/74 – 2075/76. Table 1 reports the total effect (1) of leverage
headroom.

**Table 1. Total effect of the leverage dose, quarterly outcomes, Baisakh 2070 – Falgun 2075.**

| outcome | β (SE) | wild p | lead β (wild p) | obs | clusters |
|---|---|---|---|---|---|
| NIM | 0.0196 (0.027) | 0.57 | −0.0066 (0.44) | 284 | 16 |
| net profit | −0.0073 (0.018) | 0.71 | 0.0078 (0.69) | 290 | 16 |
| provisions | 0.0119 (0.010) | 0.33 | 0.0019 (0.76) | 268 | 16 |
| loans written off | −0.0020 (0.0015) | 0.28 | — (0.40) | 290 | 16 |

*Notes:*
- Outcomes are quarterly flows over quarter-end assets.
- Bank and quarter fixed effects; pre-window controls × post.
- CR1 SEs clustered by bank; p-values from the WCR wild-cluster bootstrap (Webb weights, 9,999
  draws).
- Loans written off: blank = 0 where other P&L lines are reported (Appendix E).

No coefficient is significant, and none of the four survives multiple-testing correction. The
leads are flat. A one-SD higher leverage dose changes quarterly NIM by about 5% of its pre-period
mean. With SE 0.027 and MDE ≈ 3.0 × SE, the design would have detected a one-SD effect of about
0.0016 per quarter, or 20% of the pre-period mean NIM of 0.0080 (net profit: about 26% of its
pre mean of 0.0042). *[Derived: 3.0 × SE × SD(dose) / pre mean; recompute exactly from
`power_mde.py` before submission.]*

Because the floor did not bind, this β measures the effect of *headroom* above a slack constraint,
not of a forced recapitalisation. A null is what one would expect. We report it for two reasons.
It separates the two instruments: whatever moved Nepali banks in 2016–18 was not the leverage
floor. And it is a design check. The leverage dose was the pre-specified dose and the paid-up dose
came later (Section 4.4), so the leverage null shows that the same estimator, sample and inference
do not produce an effect where the constraint did not bind, with power to detect one-SD effects
of about 20–26% of the pre-period mean. It is a weak check: a null from a slack constraint cannot
rule out every source of bias in the paid-up estimates.

### 6.2 The paid-up floor bound

**First stage.** Table 2 reports the total effect of $\bar d^{pu}$ on quarter-end stocks, not
differenced.

**Table 2. First stage and asset growth, paid-up dose.**

| outcome | β | wild p | lead β (wild p) |
|---|---|---|---|
| paid-up / pre-window assets | 0.358 | 0.005 | −0.037 (0.28) |
| log paid-up | 9.99 | 0.009 | −0.83 (0.25) |
| log assets | 3.04 | 0.062 | −0.82 (0.088) |

A one-SD higher dose raises paid-up capital by about 0.44 log points (≈ 55%) over the post
period. β averages over every post quarter, most of them before banks crossed the floor, so 0.36
is not a compliance rate. The log-assets lead is borderline and negative: banks with a higher dose
(mostly smaller banks) were already growing faster before the mandate. We return to this in
Section 6.3.

**Compliance at the bank level (Figure 2).**
- **Bunching.** At Asar 2075, 10 of 16 banks hold paid-up capital in [8,000, 8,400) and 11 in
  [8,000, 8,800). Every placebo threshold has at most one bank in the matching window (two at
  7,000).
- **Raise matches shortfall.** For the 14 banks below the floor in Mangsir 2072, the ratio of the
  amount raised to the shortfall is 1.00–1.04 for 10 banks (median 1.01). OLS over all 16 banks:
  raise = 1,627 + 0.68 × shortfall (R² 0.73, permutation p = 0.0004). The intercept and the
  below-one slope reflect the two banks without a shortfall that raised anyway: ADBNL +4,077,
  RBB +416. Exceptions: Global 1.48, SBL 1.08, Kumari 0.85, NMB 0.90.
- **Convergence to the floor, not catch-up.** Starting rank does not predict ending level among
  the 14 (Spearman 0.011, p = 0.97). The coefficient of variation falls from 0.41 to 0.05 (end
  median 8,043, range 7,163–8,888). Catch-up would preserve rank order; a floor erases it.
- **Timing.**
  - Raises total NPR 3.4–6.8 bn a year in FY 2069/70 – 2071/72, then 13.5 bn in FY 2072/73,
    33.6 bn in FY 2073/74 (the deadline year) and 26.7 bn in FY 2074/75.
  - Four banks were at 8,000 by the Asar 2074 deadline: ADBNL and RBB already above, Global and
    NBL reaching it in Asar 2074. Eleven crossed 2–15 months late, mostly in Bhadra–Chaitra 2074.
    That is consistent with bonus shares from the FY 2073/74 accounts counting toward the floor,
    but we have not verified it.
  - Fifteen of 16 reached it within the clean window. Kumari did not (NPR 7,163 mn at Falgun
    2075). We have not established why.
- **Anticipation was small.** Only Global, Laxmi and NMB raised paid-up capital inside the
  pre-mandate window. The median share of a bank's total raise that took place there is 0.00
  (maximum 0.24).

[Figure 2: `paid_up_compliance.png`. (a) Paid-up capital by bank, Baisakh 2070 – Falgun 2075,
with the NPR 8,000 mn floor, the pre-window end (Mangsir 2072) and the deadline (Asar 2074).
(b) Amount raised between Mangsir 2072 and Asar 2075 against the Mangsir 2072 shortfall, with the
45° line. (c) Paid-up capital at Mangsir 2072 against Asar 2075, with the band [8,000, 8,400).]

**Is it a floor or a size trend?** $\bar d^{pu}_i = 8000/A_i - \text{PU}_i/A_i$, and the first
term (0.053–0.192) varies more than the second (0.023–0.082), so the dose is largely inverse
size. The strongest evidence against a small-bank trend is the bank-level shape; the regression
checks support it but do not settle it on their own:
- The bank-level shape (bunching, raise = shortfall, rank erased) is what a floor produces and
  what convergence does not.
- Frozen log assets × post is among the controls in every fit.
- It survives dropping either merging bank and an SBL-plus-Lumbini pro-forma (p = 0.008–0.034;
  Section 7.4).
- With a dose-specific linear trend over the whole panel the first stage is 0.39 (p = 0.018), but
  the pre-period lead is then significant (0.22, p 0.001), so this version fails its own pre-trend
  test. With a trend fitted on pre-period cells only it is 0.89 (p = 0.010), a magnitude that is
  not credible: it comes from extrapolating a noisy pre slope (−0.168/yr, SE 0.11) about two years
  out. The sign and significance hold in all three versions, but neither trend version is clean.
  The smoothness breakdown value is M* = 0.118 per year, 0.70 × the estimated pre slope.

### 6.3 Profitability: suggestive, not robust

**Table 3. Total effect of the paid-up dose on P&L outcomes.**

| outcome | β | wild p | lead p | 1-SD effect / pre mean |
|---|---|---|---|---|
| NIM | −0.057 | 0.002 | 0.84 | −0.0025 / 0.0080 (−31%) |
| net profit | −0.041 | 0.021 | 0.87 | −0.0018 / 0.0042 (−43%) |
| provisions | −0.001 | 0.91 | 0.94 | ≈ 0 |
| loans written off (blank = 0) | [TO FILL: paid-up dose × written-off, main blank=0 coding, from main.yaml run] | [TO FILL] | [TO FILL] | [TO FILL] |

Within the four tests, Holm keeps NIM (adjusted p 0.008 = 4 × 0.002, computed by hand from the
wild p-values). Benjamini–Hochberg keeps NIM and net
profit. Taken alone, Table 3 would say the floor cut margins and profits. We do not draw that
conclusion, for three reasons.

**(i) Event study.** Figure 3 plots the NIM event study by fiscal year without trends:
- pre-period bins: 0.0014 (p 0.80) and −0.0003 (p 0.94);
- transition bin: −0.011 (p 0.38);
- post bins: −0.064, −0.060 and −0.040 (p 0.002, 0.001, 0.006).

Net profit is similar but weaker: post bins −0.039, −0.047 and −0.033 (p 0.047, 0.013, 0.019),
transition ≈ 0. This
looks clean.

[Figure 3: `paid_up_event_study.png`. NIM and net profit per unit of assets on the paid-up dose,
by fiscal year relative to the mandate, no trend. Reference bin −1 (FY 2072/73 pre-window
quarters) at 0; bars are 95% CR1 intervals; filled markers have wild-bootstrap p < 0.05; the
shaded bin is the transition.]

**(ii) The effect runs through the denominator.** Dividing by frozen pre-window assets instead
of current assets:
- NIM: −0.021 (p 0.38), down from −0.057;
- net profit: −0.020 (p 0.56).

Together with the positive asset-growth coefficient in Table 2, most of the fall in NIM per unit
of assets comes from faster balance-sheet growth at high-dose banks. The rest, −0.021 against
pre-mandate assets, is not significant, but it is not shown to be zero either. Its 95% interval,
derived as β ± t(15) × CR1 SE (SE 0.016), is [−0.055, 0.012]. It contains zero and only just
excludes the no-trend estimate −0.057. One candidate mechanism is new capital financing asset growth at
lower margins; we have not tested it.

**(iii) Not separable from a dose-specific trend.**
- *Whole-panel linear trend:* NIM goes to −0.017 (p 0.52) and net profit to +0.028 (p 0.100).
- *Trend fitted on pre-period cells only:* NIM is +0.020 (p 0.66). The SE rises from 0.008 to
  0.036, and the 95% interval at M = 0 is [−0.056, 0.096]. It contains both zero and the no-trend
  estimate.
- *Why the pre-period trend is so imprecise:*
  - The fitted NIM pre slope is −0.030/yr (SE 0.012), even though the pre-period bin means are
    flat.
  - The slope comes from within-year variation in an unbalanced pre-sample: 5–7 quarters per
    bank, only Q1–Q2 of FY 2071/72 (no bulletins Chaitra 2071 – Jestha 2072), and a reference bin
    that is a single quarter.
  - About 2.5 years of such data cannot pin down a dose-specific trend.

**One fact cuts against "continued convergence".** Before the mandate, high-dose banks already
grew faster (log-assets pre-period bins −1.12, −0.45, 0), yet their NIM per unit of assets did
not fall: the pre-period bins are flat. Asset convergence alone did not lower NIM per unit of
assets before the mandate; after it, it did.

**Other checks.**
- *Horse race:* entering both doses jointly, NIM loads on the paid-up dose (−0.057, p 0.017),
  not on leverage headroom (0.015, p 0.16).
- *Leave-one-bank-out:* the NIM coefficient ranges from −0.071 to −0.049, with |t| ≥ 5.1 for
  every bank left out.
- *Net profit is fragile.* Without NBL it is −0.020 (t −1.28). In the horse race its p rises to
  0.127. With the pre-only trend it is null (p 0.81).

**Reading.** The data are consistent with a fall in margin per unit of assets at more-exposed
banks after the mandate, driven mostly by asset growth. They cannot separate it from a
dose-specific trend. Measured against pre-mandate assets, net interest income falls by less and
not significantly, so we cannot say whether net interest income itself fell. We report the
effect as suggestive: the estimate is neither robust nor shown to be zero.

### 6.4 Spillovers through the interbank network

**Paid-up exposure (primary test).** The paid-up floor is the constraint that bound, so the
spillover test that matters uses exposure built from the paid-up dose,
$E^{pu}_{it} = \sum_j W_{ij,t} \bar d^{pu}_j$, in Eq. (2) (Eq. A′; same sample, controls, estimator
and inference). Like the paid-up dose itself, this test was not pre-specified (Section 4.4). The
pooled spillover coefficient γ is null for every outcome under both networks:

| outcome | wild p, W_hat | wild p, RAS |
|---|---|---|
| NIM | 0.405 | 0.439 |
| net profit | 0.853 | 0.673 |
| provisions | 0.994 | 0.711 |
| loans written off (blank = 0) | 0.516 | 0.294 |

No other coefficient in this fit is significant either. The own NIM dose is −0.080 (wild p 0.12)
under W_hat and −0.074 (p 0.18) under RAS.

**Leverage exposure (secondary).** Equation (2) with exposure built from the leverage dose, the
pre-specified version, gives the same answer. Its spillover coefficient γ is null for every
outcome:
- NIM: wild p 0.18 under W_hat and 0.08 under RAS;
- net profit and provisions: p ≥ 0.17;
- loans written off (blank = 0): p 0.73 / 0.44.

Every other pooled coefficient is null too, except one: the own NIM dose, 0.082 (wild p 0.040)
under W_hat and 0.088 (p 0.011) under RAS. It is the direct effect *conditional on the
post-mandate network*. The network can itself respond to the mandate, so for β the exposure
terms are post-treatment controls. The estimate is also specification-dependent (Table 5).
Dropping the exposure terms halves it (0.043, p 0.21). Adding the 2070–71 pre-period quarters
halves it again, to the total effect of Table 1 (0.020, p 0.57). We therefore report the total
effect as the headline β and use Eq. (2) for γ only.

Joint tests of all post bins in the leverage-exposure fit (Table 4) reject at 5% only for NIM
exposure (p 0.038). It does not survive Holm or BH within the eight tests (smallest Holm p
0.302), and across every family we test there are no Holm or BH rejections (smallest Holm p
among the 16 pooled DiD tests: 0.178).
- *NIM exposure.* The event-study coefficients are nearly constant across bins, including the
  transition bin (−1.5e-4 to −1.2e-4). That is the signature of a level shift against a
  reference bin of one quarter per bank (Ashwin 2072), not a dynamic effect.
- *The monthly full-panel event studies (Appendix F).* 16 of 96 bin-level tests have p < 0.05,
  against 4.8 expected under the null, and none survives Holm or BH. Bins share a reference
  period, so these rejections are not independent.

**Table 4. Joint wild-bootstrap Wald tests, H0: all post bins = 0 (main specification, leverage
dose and exposure).**

| outcome | dose × post: W, p | exposure × post: W, p |
|---|---|---|
| NIM | 26.9, 0.137 | 30.6, 0.038 |
| net profit | 5.05, 0.498 | 19.0, 0.548 |
| provisions | 7.0, 0.477 | 26.3, 0.124 |
| loans written off | 0.408, 0.970 | 4.08, 0.300 |

*Note:* With 16 clusters the CR1 Wald statistic is heavy-tailed: a W of 19 has bootstrap p 0.55.
We report only bootstrap p-values.

For both doses the estimates under the RAS network are close to those under W_hat (e.g. own
NIM dose 0.082 vs 0.088 for leverage, −0.080 vs −0.074 for paid-up). The two networks target the
same marginals (Section 4.3), so this shows only that the result does not depend on the
regulatory cap, the temporal smoothing or how each fits the marginals (Section 8).

**How large a spillover is ruled out?** Table 6 gives the standardised MDE: the change in the
outcome, in within-bank SDs, from a one-SD higher regressor that the design detects 80% of the
time. For exposure the SD is that of exposure × post over post cells.

**Table 6. Standardised MDE, range over the four outcomes and both networks.**

| specification | paid-up exposure | leverage exposure | paid-up dose | leverage dose |
|---|---|---|---|---|
| main (quarterly, from Asar 2072) | 1.97–5.05 | 3.85–7.83 | 0.63–1.99 | 0.42–0.99 |
| clean window (monthly, Appendix F) | 0.45–1.10 | 0.59–1.12 | 0.31–0.64 | 0.13–0.36 |
| full panel (monthly, Appendix F) | 0.15–0.48 | 0.15–0.48 | 0.09–0.31 | 0.07–0.16 |

*Note:* From `power_mde.py`; the dose columns are the own-dose coefficient in the same Eq. (2)
fit, not the headline Eq. (1) total effect.

Scaled instead by the SD of exposure that remains after the fixed effects, the controls and the
own-dose term are partialled out, the main-specification MDE is 0.06–0.17 within-bank SDs for
paid-up exposure and 0.08–0.16 for leverage exposure (`power_mde.py`, "std. MDE (resid)"). The
two scalings differ because only 3.4–3.8% of paid-up exposure's spread, and 1.9–2.6% of leverage
exposure's, is left once these terms are removed (same cells, weighted). The fixed effects remove
about a third of it and the own dose almost none. The exposure-level term $\rho E_{it}$ removes
most of the rest: on its own it leaves 13–14% of what remains after the fixed effects (7–8% for
leverage exposure), while the pre-mandate characteristics × post leave 98% or more. Within a
bank, $E_{it}\,\text{post}_t$ therefore moves almost one-for-one with $E_{it}$. The level term is
there so that $\gamma$ is identified from the pre/post change rather than from connectedness
(Section 5.3). The cost is that $\gamma$ rests on the small part of exposure that shifts at the
mandate, and a one-SD difference in exposure lies far outside that variation.

In the main specification the design could detect a spillover only if a one-SD higher paid-up
exposure moved the outcome by about two to five within-bank SDs. Effects of that size are
implausible a priori, so the spillover null rules out very little. The monthly specifications
have smaller MDEs, but they are appendix specifications with different samples. We therefore
read the spillover result as *no detectable* spillover, not as evidence that there was none.

## 7. Robustness

### 7.1 The leverage-dose NIM estimate depends on the specification

**Table 5. Leverage-dose NIM coefficient across specifications (wild p).**

| specification | all quarters | Q1 cells only |
|---|---|---|
| from Asar 2072, with exposure terms (Eq. 2) | 0.082 (0.051) | 0.055 (0.16) |
| from Asar 2072, dose only | 0.043 (0.21) | 0.024 (0.42) |
| from Baisakh 2070, dose only (Eq. 1, headline) | 0.020 (0.57) | 0.008 (0.80) |

*Note:* The first row is the reviewer-check reproduction of the main Eq. (2) fit (0.082, p 0.040
in the main run), with a different bootstrap draw.

Each step roughly halves β, so both changes matter:
- the exposure terms, which are post-treatment controls for β;
- the single Ashwin 2072 reference quarter, which the 2070–71 quarters replace with a longer
  pre-period.

Net profit and provisions show the same pattern: adding the exposure terms raises the Q1 dose
coefficient (net profit 0.036 → 0.065; provisions 0.007 → 0.050, p 0.044). In the Eq. (2) fit,
no single bank flips the sign of the NIM dose (W_hat 0.044–0.092, RAS 0.055–0.098), and on Q1
cells only (n = 61) it is 0.055–0.089, so dose-related seasonality does not explain it. What
moves it is the comparison period, not any one bank. We do not report it as an effect.

### 7.2 Paid-up dose: decomposition and denominators

Splitting the paid-up dose into its two terms, each interacted with post:
- *Paid-up term* (−paid-up/assets): carries the effect. NIM −0.062 (wild p 0.012), net profit
  −0.041 (p 0.016).
- *Size term* (8,000/assets): null. NIM 0.011 (p 0.65), net profit −0.106 (p 0.16).

The size term correlates −0.984 with log pre-window assets, which is among the controls, so it is
not separately identified. The test therefore reads *conditional on size*: among banks of the
same size, those with less paid-up capital per unit of assets had lower NIM per unit of assets
after the mandate. That is the comparison an absolute floor implies.

Section 6.3 reports the frozen-denominator results (NIM −0.021, p 0.38; net profit −0.020,
p 0.56; SEs about twice as large).

### 7.3 Horvitz–Thompson weighting

The Aronow–Samii estimator weights each cell by the inverse of its generalised propensity
@aronow2017estimating @hirano2004propensity. With 16 banks these weights fail.
- **Pooled GPS (the original specification).**
  - Max/mean weight is 76.
  - Kish effective sample size @kish1965survey is 30 of 2,016 cells in the monthly full panel
    and 9 of 608 in the monthly clean window. In the clean window, three banks hold almost all
    the weight (NBL 0.45, NMB 0.28, ADBNL 0.26).
  - The pre-window cells carry 0.00 of the weight, so the DiD contrast is weighted against a pre
    period that has no weight.
  - Cause: every GPS covariate is frozen pre-mandate, while exposure grows with nominal
    interbank volume. A single σ therefore puts whole stretches of calendar time in the tails of
    the density. The weights track the calendar, not conditional exposure.
- **Period-effects GPS** (period intercepts and a per-period σ).
  - This fixes the calendar problem: the pre-window weight share rises to 0.08 and 0.23, and ESS
    to 53 and 18.
  - It does not fix the bank concentration: NBL still holds 0.50–0.58 of the clean-window
    weight. That is a positivity limit, since NBL's exposure is unlike any other bank's given its
    covariates.
  - Under these weights the full-panel NIM dose is −0.159 (p 0.002). It disappears unweighted
    (p 0.82) and in the clean window (p 0.90). It is a weighting result.

The unweighted TWFE is therefore the main estimator, and every HT variant (pooled, trimmed at the
95th/90th percentile, period-effects) is null except where a few banks hold the weight. Appendix B
gives the full diagnostics.

### 7.4 Mergers and survivorship

**Mergers inside the window.** NRB's list of institutions gives a joint operation date after
merger for two panel banks: SBL (21 June 2016, ≈ Asar 2073) and RBB (2 May 2018, ≈ Baisakh 2075)
@nrblistbfi. Which institutions they absorbed is not on the list and we have not verified it.
- *SBL.* Lumbini's last bulletin is Jestha 2073. SBL's log assets rise by 0.144 in Asar 2073,
  just under the 99th percentile of clean-month changes (0.146; median 0.021, n = 1,679). A full
  absorption of Lumbini would imply about +0.34. The bulletins therefore do not show when or how a
  merger enters SBL's balance sheet.
- *RBB.* Its largest monthly change near the date is +0.099.

We re-estimate under three versions: without SBL, without RBB, and with an SBL + Lumbini
pro-forma (SBL's dose falls from 0.106 to 0.049).
- The first stage holds in all three (paid-up/assets p 0.008–0.034).
- The NIM horse-race coefficient holds in all three (−0.057 to −0.061, p 0.001–0.014).
- Net profit is significant only without RBB (p 0.013), and p 0.14–0.20 otherwise.

**Survivorship.** Sixteen of the 33 commercial-bank labels pass the complete-case filter of the
network reconstruction, which runs over all 133 months. Over the main window alone 21 labels are
complete; BOK, Mega, NCC, NIBL and Sunrise are excluded only by months after Falgun 2075.
Including them would need the reconstruction re-run on the clean window (Section 8). The excluded banks do not differ in leverage dose (median −0.0438 in both groups;
Mann–Whitney p 0.25), but they are smaller (median pre-window assets NPR 64.9 bn included vs 44.5 bn excluded,
p 0.005). Four excluded series look like relabels of one bank under a new short name from Magh
2075 (Citizens→Citizen, NBB→NBBL, PRABHU→Prabhu, Civil→CBL), continuous in size and capital; we
have not verified them against NRB records. Our results describe the larger commercial banks.

### 7.5 Other checks

- **Loans written off coding** (Appendix E). Blank = 0 is the main coding. With blank = missing,
  the total effect is 0.077 (SE 2.65, p 0.97, 7 clusters). The null does not depend on the coding,
  only its precision does. For the paid-up dose the blank = missing written-off total effect is
  −0.058 (p 0.97, lead p 0.16, 7 clusters); Table 3 reports the main blank = 0 coding for this row.
- **Frozen-size denominator, leverage dose** (monthly full panel). Null for every outcome under
  both networks; the smallest p is 0.082.
- **Row-normalised exposure** (shares instead of levels; main specification). Banks that report no
  interbank lending in a month get zero exposure. The exposure coefficient is null for every
  outcome under both networks; the smallest p is 0.161. With this exposure in the model, the own
  provisions dose has p 0.035 (W_hat) and 0.023 (RAS). That is 2 of 16 coefficients, from a
  post-treatment-conditional fit, and we do not read it as an effect. In the earlier monthly
  full-panel version every coefficient was null (smallest p 0.107).
- **Indirect exposure** (exploratory; main specification). Katz–Bonacich exposure
  $\tilde W(I - 0.5\tilde W)^{-1}\bar d$, with $\tilde W$ row-normalised: it adds counterparties'
  counterparties, discounted by half per step. We fixed the discount before looking at results. It
  correlates 0.965 (W_hat) and 0.978 (RAS) with the row-normalised direct exposure, so indirect
  links add little. No exposure coefficient is significant (smallest p 0.163; Holm-adjusted p 1.00
  for all eight).
- **RAS network.** Every Eq. (2) estimate is nearly the same with the RAS network in place of
  W_hat. Both networks target the same observed marginals; they differ by the regulatory cap, the
  temporal smoothing and how each fits the marginals (Section 8). So this shows the results are
  robust to those choices, not to the choice of reconstruction method.
- **Separate identification of β and γ.** corr(dose × post, exposure × post) is 0.10–0.18.
- **Regression discontinuity.** Infeasible: no cell lies on the binding side of the leverage
  cutoff in any specification (Appendix D).

## 8. Limitations

**Identification.**
- *Concurrent shocks.* The pre-mandate window follows the Baisakh 2072 earthquake @npc2015pdna
  and overlaps the 2015–16 border disruption [CITATION NEEDED], and NRB relief circulars fall
  inside the pre window and transition (Section 3.4). Bank fixed effects and time effects absorb
  common shocks. A shock whose incidence correlates with the dose would not be absorbed, and we
  have no bank-level exposure data to test for one.
- *Announcement timing.* The paid-up requirement comes from the FY 2072/73 Monetary Policy, and
  capital plans were due by Bhadra 2072. The frozen dose is therefore partly post-announcement.
  Measured anticipation is small (Section 6.2), but the exact announcement month is inferred, not
  documented.
- *Trends.* The pre-period is about 2.5 years of unbalanced quarters. It cannot pin down a
  dose-specific trend, which is why the NIM result stays suggestive (Section 6.3).
- *No pre-trend test for γ.* The network exists only from Asar 2072, so the exposure terms have
  no leads.
- *Not pre-specified.* We analysed the paid-up dose after the leverage-dose null and after
  finding the policy text. We report every check we ran on it, including those that weaken it.

**Measurement.**
- *The leverage dose is not the regulatory ratio.* Its numerator is capital fund (Tier 1 plus
  Tier 2), not Tier 1, and its denominator is total on-balance-sheet assets, not the regulatory
  exposure measure @nrb2015caf. Mismatch that is not proportional across banks attenuates β.
- *Outcomes.* Bulletin P&L figures are reliable only at quarter ends (Appendix A), so we use a
  third of the months. A blank written-off row is coded as zero (Appendix E). For seven banks the
  outcome is therefore identically zero, and the effective number of clusters is about six.

**The reconstructed network.** W_hat is the marginal-constrained, capped reconstruction of
@paper1companion.
- *Closed system.* The projection covers only the 16 banks. Lending to and borrowing from
  other institutions, NRB and foreign banks are outside the network, so any spillover routed
  through them is invisible to E_it by construction.
- *Generated regressor.* We treat W_hat as fixed. Ignoring its estimation error makes our SEs too
  small, which can only reinforce a null. Its sampling variability is probably small, because on
  real data it depends little on the model state. Its model error is not measured at all: many
  matrices fit the same marginals and no bilateral ground truth exists. Only one seed of the
  reconstruction was run, so seed sensitivity is also unmeasured.
- *The cap links W_hat to the treatment.* The cap is proportional to each bank's capital fund,
  and the mandate raised capital. Wherever the cap binds, the treatment changes the network
  directly. This is the only part where W_hat and RAS differ by design; they also fit the
  marginals differently (below).
- *Extraction errors.* The reconstruction is trained on all 133 months, including bulletins
  corrupted from Chaitra 2075. Each month's matrix is fitted to that month's marginals, and the
  clean-window marginals are clean, so we expect the effect to be small. The temporal smoothing
  does link Falgun 2075 to Chaitra 2075.
- *Interface checks (run, `check_paper1_interface.py`).* The artefact is the final Paper 1 run
  (`lambda_mi` 0.0, `lambda_smooth` 0.05, 500 epochs, seed 0), shape (133, 16, 16), Asar 2072 –
  Shrawan 2083, with no non-finite or degenerate months. That span has 134 calendar months; the
  one absent from the artefact is Kartik 2078, outside the main window, which has all 45 of its
  months. The marginals stored in the artefact match the bulletins, and the implied cap is
  κ = 0.05. The cap binds for 7.0% of entries over all months and 1.9% in the main window.
- *How closely each matrix fits the marginals (main window, 720 bank-months).* Most banks report
  no interbank lending or borrowing in most months: lending is zero in 398 bank-months, borrowing
  in 465. The two reconstructions fit these totals differently:
  - RAS matches the lending totals closely (largest deviation NPR 41 mn), including every zero.
    Its borrowing totals deviate by up to NPR 8,290 mn.
  - W_hat matches the borrowing totals more closely (largest deviation NPR 438 mn), but puts
    some borrowing on every bank, including those that report none: on all 465 such bank-months
    the column sum exceeds NPR 1 mn (median NPR 101 mn). Its lending totals deviate by up to
    NPR 2,311 mn. Where a bank reports no lending, W_hat's row sum exceeds NPR 1 mn in 249 of the
    398 bank-months (median NPR 27 mn).

  The two matrices therefore share the same targets but not the same fit, so a reader should
  not take them as identical apart from the cap. Exposure under W_hat and under RAS still
  correlates 0.968 on the 400 post-mandate cells of the main window (0.939 on all 720; 0.849 on
  the 1,576 post cells of the full panel). This shows that the spillover estimates do not depend
  on these fitting differences, the cap or the smoothing. It does not show that either network is
  right.

  The fits differ because the two published marginals do not balance within the 16 banks (run,
  `reviewer_checks` `marginal_balance`). Total interbank lending and total interbank borrowing
  differ by more than 1% in 43 of 45 months (median relative gap 0.32, maximum 0.93), and in 43
  months no non-negative matrix with a zero diagonal can match both. Both reconstructions' worst
  marginal misses track the gap (Spearman 0.93 for each). RAS matches the lending totals and
  leaves the whole gap on borrowing, while W_hat splits it between the two. The size of the misfit
  therefore reflects the published data; how it is split reflects each method. The within-panel
  imbalance is itself partly expected because the panel is not a closed system — panel banks trade
  with institutions outside the 16, so their two totals need not balance within the panel
  (Paper 1 §6.2) — which is a further reason not to read it as a pure data-quality defect.
- *Checks not run.* Two checks would sharpen these points; we have not run them, so we state
  them without numbers:
  - the share of digit fragments in the marginals;
  - a reconstruction run on the clean window only.

**Scope.** Sixteen large commercial banks. The excluded banks are smaller (Section 7.4), and
small banks are where an absolute floor bites hardest. Kumari stays below the floor through
Falgun 2075, and we have not established why.

## 9. Conclusion

Nepal's NPR 8,000 million paid-up floor bound. Banks raised capital almost exactly to the floor,
and close to the deadline. Starting rank did not predict the end level, and bank-level raises
matched shortfalls one for one. The regression first stage agrees, though it is not clean under
dose-specific trends. The Basel III leverage floor introduced at the same time did not bind, and
leverage headroom predicts nothing. That null is also a check on the design, which was first
specified with the leverage dose.

Whether the floor changed profitability we cannot say with confidence. Net interest income per
unit of assets fell at more-exposed banks after the mandate, with flat pre-period coefficients.
But the fall runs mostly through faster asset growth: measured against pre-mandate assets it is
smaller and not significant. The data cannot separate it from a dose-specific trend, and they do
not show it to be zero. We see no spillover through the reconstructed interbank network, whether
exposure is built from the paid-up or the leverage dose. But in the main specification the design
can detect only spillovers of about two or more within-bank SDs per SD of exposure, so this null
rules out only large effects.

Three lessons go beyond Nepal.
- An absolute floor's bite can be measured directly from the distribution of paid-up capital
  (bunching, raise against shortfall, rank). These diagnostics need no control group.
- Ratio outcomes can move through the denominator alone, so a frozen-denominator check belongs
  in any study of capital regulation that scales by assets.
- With a reconstructed network, exposure designs inherit the limits of the published data.
  Nepal's two interbank marginals are mostly zero and do not balance within the sample, so no
  reconstruction matches them exactly, and our design can detect only spillovers of about two or
  more within-bank SDs. Detecting smaller ones would need bilateral or large-exposure interbank
  reporting. For a supervisor, that is the case for collecting it.

## Data and code availability

The bulletins are public @nrbbulletins. The extracted panel, the reconstructed network and all
estimation code (packages `open_nrb` and `open_nrb_causal`) will be deposited at [REPOSITORY TO
FILL] on acceptance. Every number in this paper is produced by the configurations `main.yaml`
(main), `baseline.yaml` and `clean_window.yaml` (monthly appendix runs) and the
`reviewer_checks` options named in the appendices.

## Declarations

- **Funding:** [TO FILL.]
- **Competing interests:** [TO FILL.]
- **CRediT author contributions:** [TO FILL.]
- **Data availability:** see Data and code availability above.
- **Use of AI tools:** [TO FILL: disclose any use in preparation.]

## References

[Generated from `book.bib`.]

## Appendix A. Monthly outcomes and the quarter-end construction

The P&L lines in the bulletins are fiscal-year-to-date cumulative: median |Shrawan| / |previous
Asar| for NIM is 0.087 ≈ 1/12. De-cumulating them gives monthly flows. In the monthly full panel
(`baseline.yaml`), the share of negative flows is:
- net profit 20.7%;
- provisions 23.6%;
- loans written off 4.6%;
- NIM 11.9%.

Write-backs can make provisions and net profit negative, but NIM should almost never be. Yet 28%
of NIM cells in FY 2073/74 are negative, and that is a clean fiscal year.

The pattern follows the calendar:
- *Clean window, NIM:* 0.00 of cells negative at the quarter-end months (Ashwin, Poush, Chaitra,
  Asar), against 32–41% in every other month.
- *Clean window, net profit:* 0.02–0.05 at quarter ends against 0.38–0.48 elsewhere.
- *Full panel, NIM:* the same, weaker (0.00–0.02 against 0.13–0.22).
- *Shrawan:* never differenced, yet NIM is negative in 41% of clean-window Shrawan cells. So the
  non-quarter-end figures are unreliable in themselves; this is not an artefact of differencing.

Our reading, inferred from the sign pattern and not verified against the PDFs, is that the
bulletin P&L is a dependable YTD figure only at quarter ends, matching banks' quarterly
reporting. The quarterly construction uses quarter-end YTD minus the previous quarter end in the
same fiscal year (Ashwin = YTD), divided by quarter-end assets, and never reads non-quarter
months. Negative quarterly NIM falls to 0.035 (full panel) and 0.000 (clean window); net profit
to 0.043 and 0.005. Provisions stay at 0.17–0.25, consistent with write-backs.

The monthly runs (full panel and clean window, unweighted) are null on every pooled coefficient.

## Appendix B. Horvitz–Thompson weight diagnostics

| | pooled GPS | period-effects GPS |
|---|---|---|
| max / mean weight | 76 (W_hat), 68 (RAS) | 102–169 |
| p99 / mean | ≈ 70 | — |
| Kish ESS, full panel | 30 / 2,016 (W_hat), 34 (RAS) | 53 |
| Kish ESS, clean window | 9 / 608 | 18 |
| pre-window weight share | 0.00 | 0.08 (full), 0.23 (clean) |
| largest bank shares | full: NMB 0.32, SCBNL 0.14, Global 0.12, NBL 0.09, Prime 0.09; clean: NBL 0.45, NMB 0.28, ADBNL 0.26 | clean: NBL 0.50–0.58 |

Monthly runs; "—" = not reported.

- *Weighting moves point estimates but not inference.* In the monthly full panel, weighted vs
  unweighted: NIM dose −0.097 vs −0.010; RAS net-profit dose +0.046 vs −0.021. All of them null.
- *Trimming at the 95th and 90th percentiles:* null throughout.
  - The one near-rejection, clean-window provision exposure (weighted p 0.062; CR1 t ≈ 5.6,
    |β|/MDE 1.86), moves to 0.29–0.48 trimmed and 0.47 unweighted. It is a weighting artefact
    of NBL, NMB and ADBNL.
  - The only p < 0.05 is the RAS written-off dose at q = 0.95 in the clean window (0.033), an
    uninformative outcome and one of many tests.
- *Period-effects GPS:* full-panel NIM dose −0.159 (p 0.002, both networks), not robust
  (unweighted p 0.82, clean window p 0.90).

## Appendix C. Extraction audit

`audit_bs_extraction.py` checks every (bank, month, file) sheet:
- liabilities sum to their total;
- assets sum to their total;
- capital sub-items sum to the capital fund;
- the two totals balance;
- it also counts single-digit values ("fragments").

Coverage is 4,094 sheets, 42 institutions and 156 bulletins. The CSV has no bulletins for
Chaitra 2071 – Jestha 2072.

| bulletins | liabilities/assets pass | fragments | median asset-sum gap |
|---|---|---|---|
| Baisakh 2070 – Falgun 2075 | 96–100% | 0.0–0.3 | 0.00% |
| Chaitra 2075 – Asar 2077 (except Bhadra/Ashwin 2076) | 0–31% | 12–27 | 9–20% (Asar 2076: 222%) |
| Chaitra 2077; Jestha 2078 – Poush 2079 (except Baisakh, Ashwin 2078) | 88–100% | ≈ 0 | — |
| Magh 2079 – Shrawan 2083 | 0–8% | 11–21 | 28–48% |

- *Totals balance in 100% of sheets that have both total rows, in every file.*
- *Asset identity by fiscal year:* 100% pass for FY 2069/70 – 2074/75, then 71%, 22%, 28%, 91%,
  50%, and ≤ 2% from FY 2080/81.
- *Capital sub-item exception.* In Mangsir 2072 – Mangsir 2073 only 4–8% of sheets pass the
  capital sub-item check, while the top-level checks pass and fragments are ≈ 0. This fits a
  change in sub-item labelling (unverified). The capital fund itself is unaffected.
- *The six dose-window bulletins* (Asar–Mangsir 2072) pass the liability and asset identities
  for every bank (fragments ≤ 0.1).

Panel construction:
- 1,844 balance-sheet rows superseded by "-NEW" resubmissions;
- 22 (bank, month) cells missing a required field;
- 9 of 42 institutions not commercial banks;
- 23 of 156 months before Asar 2072, in the network sample only;
- the complete-case filter, leaving N = 16, T = 133.

Example of a fragment: EBL, Asar 2077, paid-up capital 8.0 and fixed assets 2.0. Its capital
sub-items sum to 253.5 against a capital fund of 16,121.1.

[Figure 1: `extraction_audit.png`. Share of balance sheets whose asset items sum to TOTAL ASSETS,
and mean single-digit values per sheet, by bulletin; the dose window, the transition and the end
of the clean window are marked.]

## Appendix D. Regression discontinuity

We planned a secondary RDD at the 4% leverage cutoff. Pooling Shrawan 2073 – Asar 2075, no cell
lies on the binding side of the cutoff in any specification, so it cannot be estimated. This
reflects the same fact as Section 6.1: the leverage floor did not bind.

## Appendix E. Coding of loans written off

Clean window, 16 banks × 45 months:
- 155 cells report the line, and only 8 of them are an explicit zero (all ADBNL);
- 531 are blank while other P&L lines are present;
- 34 have no P&L row (Global, NMB, Prime).

ADBNL both prints zeros and leaves blanks, so a blank is not simply an unprinted zero, and the
data cannot settle what it means.

**Main coding (blank = 0).** A blank row in a month with other P&L lines means nothing written
off so far this fiscal year; with no P&L row at all, the cell stays missing. The coding is applied
to the YTD figure before differencing.
- It gives 210 quarterly cells over 16 banks, against 45 over 6 banks when blanks are missing.
- Seven banks (Global, HBL, Kumari, NBL, NSBI, Prime, RBB) never report the line in the clean
  window, so for them the outcome is identically zero.
- The variation comes from about six banks (ADBNL, Laxmi, NABIL, SCBNL, MBL, NMB).

| coding | total effect β (SE), wild p | Eq. (2) dose p, W_hat / RAS | Eq. (2) exposure p, W_hat / RAS |
|---|---|---|---|
| blank = 0 (main) | −0.0020 (0.0015), 0.28; 290 obs, 16 clusters | 0.57 / 0.87 | 0.73 / 0.44 |
| blank = missing | 0.077 (2.65), 0.97; 59 obs, 7 clusters | — | — |

In the monthly full panel, dropping NMB moves the written-off dose from −0.025 to +0.82. We treat
the outcome as descriptive.

## Appendix F. Full-panel (monthly) results

The full panel (`baseline.yaml`, all 133 months) and the clean window (`clean_window.yaml`,
T = 45, 32 post months) are monthly, unweighted, appendix runs.
- *Pooled DiD:* null for every coefficient under both networks in both samples. The smallest
  clean-window p is provision exposure (0.062 W_hat / 0.079 RAS, under the original weights). It
  is ≈ 0 or negative in the full panel and is a weighting artefact (Appendix B).
- *Precision.* The clean window has *smaller* CR1 SEs for every dose coefficient (e.g. NIM 0.021
  vs 0.069, net profit 0.016 vs 0.024), so the corrupted months add noise rather than power.
- *Event-study patterns that do not survive the clean window:*
  - The written-off dose path is ≈ −0.052 in every full-panel bin and ≈ +0.003 in the clean
    window. The FY 2073/74 – 2074/75 bins contain only clean months in both runs. Corrupted later
    months move the bank fixed effects and shift every bin.
  - The NIM exposure path keeps its point estimates (≈ −5e-5, including the transition bin) but
    p rises to 0.18–0.43.
  - The scattered full-panel p < 0.05 (e.g. NIM dose bins 2, 6 and 8, alternating in sign) fall in
    the corrupted years.
- *Multiple testing:* 16 of 96 event-study tests have p < 0.05, against 4.8 expected; none
  survives Holm or BH (smallest Holm p 0.077).
