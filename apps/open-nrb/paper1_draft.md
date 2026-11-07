# Who Lends to Whom? Reconstructing Interbank Exposures from Public Balance Sheets

**Status:** first full draft (2026-11-05), built from the design doc
`paper1_network_reconstruction.md` (§0–§12, Appendices A–B) and the Paper 2 runs it cites. No new
runs. **Target venue:** Journal of Network Theory in Finance, full length with appendices.

**One-line description:** A covariate-informed blockmodel with a differentiable capped projection
reconstructs bilateral interbank exposures from row and column totals. On synthetic networks it
beats RAS; on Nepal's published bulletins the totals do not balance, the projection sets the
result, and the bulletins cannot say who is exposed to whom.

> **Pre-submission checklist (delete before submission)**
> 1. ~~**Synthetic headline number for the reported config.**~~ DONE 2026-11-06: §5.2 filled
>    from `results/sweep_lambda_mi_0/multiseed_summary.json` (`lambda_mi` = 0) — model 0.430 ± 0.043
>    vs. RAS 0.566 ± 0.026, 5/5 seeds beat baseline, p = 0.0101. (No "Table 2" exists in the body;
>    the headline is prose in §5.2.) The 0.4219 ± 0.0357 figure from `synthetic_baseline`
>    (`lambda_mi` = 0.05) was NOT used.
> 2. **Not run, stated as not run:** a reconstruction run on the clean window
>    Asar 2072 – Falgun 2075 (§6.1, §7).
>    (All four referee checks are now RUN — capped-RAS §5.2, single-block $K=1$ §5.4, non-SBM
>    misspecification §5.4, persistence §6.5 — plus the parameter count §7: 2,133 parameters.)
> 3. ~~**Real-data stability numbers**~~ DONE 2026-11-06: §6.5 filled from
>    `results/baseline/real_run_summary_seed0.json` (`model_stability_mean` 1.16 vs.
>    `baseline_stability_mean` 2.32; no `eda_model_checks` file exists). Reported as consistency
>    with the smoothness prior only.
> 4. **Bib entries missing** (keys used below, not in `book.bib`):
>    `mistrulli2011assessing`, `littlerubin2019missing`, `fsbimf2009infogaps`,
>    `bcbs2013principles`, `boyle1986method`, `nrbunified`, `paper2companion`. Add each with
>    `note = {TO VERIFY}` and check against the publisher record.
> 5. **Regulatory basis of κ = 0.05.** The value is the run configuration's. Check the
>    single-obligor limit and its base (capital fund vs. core capital) against NRB's Unified
>    Directive before stating it as NRB's rule (§3.3).
> 6. **Open author decisions:** (a) balance the marginals before projecting, or keep the
>    least-violation output (this draft keeps it and reports the misfit, §6.2); (b) whether to
>    merge the four possible relabel pairs (§6.1); (c) whether to rerun on the clean window,
>    which could raise N from 16 to 21 (§6.1).
> 7. Figures to insert: synthetic ground truth vs. reconstruction (`plot_synthetic_network`),
>    real-panel marginals (`eda_real_panel`), marginal gap vs. $\hat W_t$ miss (from Paper 2
>    `marginal_balance`).
> 8. Code and data release: fill the repository URL in the Data availability statement.

## Abstract

Bilateral interbank exposures are rarely published, so studies of contagion and spillovers
reconstruct them from each bank's total interbank lending and borrowing. We propose a
reconstruction that combines a covariate-informed stochastic blockmodel, an amortized recurrent
inference network and a differentiable projection onto the published totals and a per-pair
regulatory exposure cap, solved by Dykstra's alternating projections. On synthetic blockmodel
networks with known ground truth (24 banks, 16 periods, 5 seeds), the reconstruction error is lower
than that of the classical RAS method in every seed (paired t-test p = 0.0101), but the latent
block labels are not recovered better than a majority guess (p = 0.1136). We then apply the method
to a hand-extracted monthly panel of 16 Nepali commercial banks over 133 months. Here the published
data, not the model, decide the result. In 43 of 45 months of the clean window, total lending and
total borrowing within the 16 banks differ by more than 1% (median relative gap 0.32), so no
non-negative matrix can match both totals; the projection returns its least-violation compromise,
and its miss tracks the gap (Spearman 0.926). Two model fits whose latent posteriors end at opposite
degenerate states give matrices within a relative difference of $1.5\times10^{-5}$. The cap binds
for 1.9% of entries in the clean window. We conclude that the method works where its inputs are
consistent, and that Nepal's two published marginals do not support a bilateral reconstruction:
answering "who is exposed to whom" would need bilateral or large-exposure reporting.

**Keywords:** interbank networks; network reconstruction; maximum entropy; RAS; stochastic
blockmodel; capped matrix balancing; Dykstra's algorithm; emerging markets; Nepal.

## 1. Introduction

Contagion analysis needs a network, and the network is usually not observed. Central banks and
researchers observe each bank's total interbank lending and total interbank borrowing, and must
reconstruct the bilateral matrix whose row and column sums these are [@upper2011simulation;
@squartini2018reconstruction]. The classical answer spreads the totals as evenly as possible (RAS
or maximum entropy, @deming1940least; @gandy2013financial), which is known to produce networks that
are too dense and to understate systemic risk where true bilateral data exist for comparison
[@mistrulli2011assessing; @anand2018filling].

This paper makes three contributions.

1. **A capped, covariate-informed reconstruction.** We model the matrix with a stochastic
   blockmodel whose edge intensities depend on bank covariates, infer it with an amortized
   recurrent network, and force the output onto the published totals and a per-pair regulatory
   cap with a differentiable Dykstra projection. The cap is applied to each entry, as an exposure
   limit is, not to each row total; an earlier capped-Sinkhorn version that capped row totals is
   documented as a pitfall (Section 3.4).
2. **A synthetic test with honest scope.** With ground truth available, the method's
   reconstruction error is lower than RAS's in every seed. Its block labels are not better than a
   majority guess, so we make no claim about recovering core and periphery.
3. **A measured data limit on real data.** On Nepal's monthly bulletins, the two published totals
   do not balance within the observed banks in almost every month. We show what this does to any
   reconstruction that targets them: the misfit is set by the data, the projection sets the matrix,
   and the model's latent structure has no detectable effect on the output. For a regulator this is
   a concrete finding. The imbalance is first of all a sign that the 16 banks are not a closed
   system — a panel bank can lend to or borrow from institutions outside the panel, so its two
   totals need not match within the panel even if every figure is correct — and only possibly a
   sign that the two bulletin lines are defined inconsistently; we cannot separate the two from the
   published aggregates. Either way a bilateral reconstruction restricted to the panel is not
   identified from them.

We report the third result as a finding, not a failure. Missing bilateral data is the norm among
central banks [@upper2011simulation; @fsbimf2009infogaps], and data aggregation problems persist
even inside large banks [@bcbs2013principles]. A method paper that reports what real published
data can and cannot support is useful to the many emerging-market regulators who publish similar
aggregate returns. A companion paper [@paper2companion] uses the reconstruction as an exposure
weight in a study of Nepal's paid-up capital mandate; we do not discuss causal questions here.

Section 2 reviews related work. Section 3 sets out the model and the projection. Section 4
describes the data. Section 5 reports the synthetic results and Section 6 the real-data results.
Section 7 discusses limitations and Section 8 concludes.

## 2. Related work

**Matrix balancing and maximum entropy.** Iterative proportional fitting [@deming1940least;
@stone1961input] rescales a prior matrix until its margins match. Applied to interbank markets
[@upper2011simulation; @gandy2013financial], it gives the RAS baseline we use.
@squartini2017maximum and @squartini2018reconstruction place these methods in a maximum-entropy
ensemble framework. @anand2018filling show that such ensembles spread exposures over too many
links relative to observed networks, and @mistrulli2011assessing, with Italian bilateral data,
shows how far maximum-entropy contagion estimates can be from those on the observed network.

**Block structure and learned inference.** Interbank markets are often tiered
[@clementi2019tiered], which motivates a stochastic blockmodel [@holland1983stochastic;
@nowicki2001estimation]. Amortized variational inference over interaction graphs
[@kipf2018neural] and differentiable relaxations of discrete structure [@mena2018learning] make
such models trainable by gradient descent. Our covariate term plays the role of degree correction
or fitness, as in @peixoto2020latent.

**Reconstruction as Bayesian inference.** A separate line infers a network from noisy or partial
reports of its edges [@guimera2009missing; @martin2016structural; @peixoto2018reconstructing;
@peixoto2019reconstruction]. Our setting gives no edge-level signal at all: only row and column
sums. These methods are therefore close in spirit but not drop-in baselines; adapting them to
margin-only input is outside our scope.

**Model selection versus recovery.** @vallescatala2018modelselection show that a model's own
selection criterion and its link-recovery performance can disagree. Our real-data stability
comparison has the same problem, since the model is trained to be smooth (Section 6.5).

## 3. Method

### 3.1 Notation

Banks are $i, j \in \{1,\dots,N\}$ and months $t \in \{1,\dots,T\}$. $W_t \in
\mathbb{R}^{N\times N}_{\ge 0}$ is the latent exposure matrix, with $W_{ij,t}$ the exposure of bank
$i$ to bank $j$ and a zero diagonal. We observe the lending total $r_{i,t}$, the borrowing total
$c_{i,t}$, the capital fund $K_{i,t}$ and covariates $X_{i,t} \in \mathbb{R}^d$; $z_i \in
\{1,\dots,K\}$ is a latent block.

### 3.2 Generative model

$$z_i \sim \text{Categorical}(\pi), \qquad \phi_{i,t} = \exp(w^\top X_{i,t}), \tag{1–2}$$
$$\lambda_{ij,t} = \Lambda_{z_i z_j}\,\phi_{i,t}\,\phi_{j,t}, \tag{3}$$
$$\Pr(\tilde W_{ij,t} = 0) = \sigma(-\eta_{ij,t}), \qquad \tilde W_{ij,t} \mid \tilde W_{ij,t} > 0
\sim \text{LogNormal}(\log \lambda_{ij,t}, \sigma^2), \tag{4}$$
$$r_{i,t} = \textstyle\sum_j W_{ij,t}, \qquad c_{j,t} = \textstyle\sum_i W_{ij,t}. \tag{5}$$

The zero inflation in (4) reflects that most bank pairs have no exposure. Because (5) is a
deterministic constraint, not a noisy observation, we enforce it in the architecture through a
projection (Section 3.3) rather than in the likelihood. $\Lambda$, $w$ and the block-level
existence logits are parameters of the synthetic generator only; the trained model replaces (3)–(4)
with a learned decoder.

### 3.3 Capped projection

Given a raw matrix $\tilde W_t$, the output is its Euclidean projection onto the intersection of
the row-sum set $\mathcal{R}$, the column-sum set $\mathcal{C}$ and the box $\mathcal{B}$:
$$W_t = \text{Proj}_{\mathcal{R}\cap\mathcal{C}\cap\mathcal{B}}(\tilde W_t), \qquad
\mathcal{B} = \{W : 0 \le W_{ij} \le \kappa K_{i,t}\ \forall j\}. \tag{8}$$
The box caps each bilateral exposure at a fraction $\kappa$ of the lender's capital fund, the
form of a single-obligor limit. We use $\kappa = 0.05$ [TO VERIFY against @nrbunified].

We compute (8) with Dykstra's alternating projections [@boyle1986method] (Algorithm 1). Each of the
three set projections is closed form: a uniform shift of each row, a uniform shift of each column,
and a clip. The iteration converges to the exact projection when the intersection is non-empty, and
gradients pass through it by automatic differentiation (the clip has the usual subgradient). When
the intersection is empty the iteration returns a least-violation point that keeps the box but
misses the margins. We treat that case as a property of the inputs to be reported, not as a
solution; on real data it is the usual case (Section 6.2).

```
Algorithm 1  CapacitatedProjection(W̃, r, c, cap, n_iters, tol)
 1: W ← W̃;  I_row, I_col, I_box ← 0
 2: for iter = 1 .. n_iters:
 3:     W_prev ← W
 4:     Y ← W + I_row;  W ← Y + (r − rowsums(Y)) / N;  I_row ← Y − W
 5:     Y ← W + I_col;  W ← Y + (c − colsums(Y)) / N;  I_col ← Y − W
 6:     Y ← W + I_box;  W ← clip(Y, 0, cap_i) for every entry of row i;  I_box ← Y − W
 7:     if ‖W − W_prev‖ < tol: break
 8: return W
```

### 3.4 A pitfall: capping rows instead of entries

A natural first attempt adds a cap to Sinkhorn scaling by clipping the row scalar $u_i$. A row
scalar rescales a whole row, so the clip enforces $\sum_j W_{ij} \le \kappa K_{i,t}$, a limit on
total interbank lending, instead of $W_{ij} \le \kappa K_{i,t}$ for each counterparty, a limit on
concentration. A unit test with a binding per-entry cap exposed the difference. Plain, uncapped
Sinkhorn remains our RAS baseline [@cuturi2013sinkhorn; @gandy2013financial].

### 3.5 Inference network

A GRU encodes each bank's history, $h_{i,t} = \text{GRU}_\theta([X_{i,t}; r_{i,t}; c_{i,t}],
h_{i,t-1})$ (6). The marginals enter the encoder because in (1)–(2) the block is drawn
independently of the covariates, so covariates alone carry no information about $z$; the marginals
do, through $\Lambda$. A small network gives $q_\psi(z_i) = \text{softmax}(\text{MLP}_\psi(h_{i,t}))$
(7), and a pairwise decoder maps $(h_{i,t}, h_{j,t}, z_i, z_j)$ to $(\eta_{ij,t}, \lambda_{ij,t})$.
Training samples use the Gumbel-softmax, Gumbel-sigmoid and log-normal reparameterizations; the
reported $\hat W_t$ uses posterior means ($q_\psi(z)$, $\sigma(\eta)$ and $\lambda e^{\sigma^2/2}$)
before projection, which removes sampling noise from the point estimate. We use a GRU rather than a
temporal graph network because the panel is small (16 banks).

### 3.6 Training objective

$$\mathcal{L} = \underbrace{-\mathbb{E}_q[\log p(W_t^{\text{true}})]}_{\text{synthetic only}} +
\beta_t\,\text{KL}\big(q_\psi(z)\,\|\,p(z)\big) + \lambda_{\text{smooth}} \sum_t \|W_t -
W_{t-1}\|_F^2. \tag{9}$$

$\beta_t$ rises linearly to $\beta_{\max} = 0.15$ over 300 epochs, with a free-bits floor of 0.05
per bank and period, and $\lambda_{\text{smooth}} = 0.05$. The code also contains a
mutual-information clustering term; a sweep over its weight (Appendix C) found no benefit, and it
is set to zero in every reported run.

On real data the first term of (9) is identically zero, because no bilateral ground truth exists.
The KL term depends only on $q_\psi(z)$ and gives the decoder no gradient. The only term that
reaches the decoder is the smoothness penalty on the projected matrices. Real-data training is
therefore a regularized fit to the margins, not likelihood estimation, and Section 6.4 shows its
consequence.

## 4. Data

**Source.** NRB's monthly *Banking and Financial Statistics* bulletins, published as PDFs. We
extract each commercial bank's balance sheet. The two margins are sub-items of the balance sheet:
$r_{i,t}$ is `a. Interbank Lending` under `3 SHARE & OTHER INVESTMENT` (assets), and $c_{i,t}$ is
`b. Interbank Borrowing` under `2 BORROWINGS` (liabilities). $K_{i,t}$ is `1 CAPITAL FUND`. The five
covariates are size (the sum of the ten top-level asset rows), the capital and liquidity ratios,
loans over deposits, and NRB borrowing over total borrowing (Appendix B).

**Extraction.** Item numbers drift across eras, lettered sub-items repeat under different parents,
early bulletins lack a liabilities total, and rows are interleaved across banks. The extractor
keys on normalized label text and parent section, tracks the balance-sheet side per bank, and treats
an absent optional row as zero only where it is a component of a sum. An audit of the accounting
identities (Paper 2, Section 4.2) finds 96–100% of sheets passing through Falgun 2075 and a sharp
collapse from Chaitra 2075; we call the months through Falgun 2075 the clean window. We have not
traced the later failures to the PDFs rather than to the extraction.

**Complete-case rule.** We never impute. A bank enters only if every field is present in every
month of the modelling window. Missingness here has identifiable causes (an era's numbering, a bank
absent from a page) and is not missing completely at random, so imputation would inject bias tied
to the failure modes [@littlerubin2019missing]. The rule leaves N = 16 of 33 commercial-bank labels
over T = 133 months, Asar 2072 – Shrawan 2083. One calendar month, Kartik 2078, is absent. Section
6.1 reports who is excluded and why.

## 5. Synthetic results

### 5.1 Design

The generator (Appendix A, Algorithm 2) draws 24 banks over 16 periods with $K = 2$ blocks,
$\pi = (0.3, 0.7)$, $\Lambda = \begin{psmallmatrix}2.5 & 0.6\\ 0.6 & 0.2\end{psmallmatrix}$,
block existence logits $\begin{psmallmatrix}2 & 0\\ 0 & -2\end{psmallmatrix}$, five covariates
with weights $w = (0.05, 0.8, 0.5, -0.3, 0.4)$, $\sigma^2 = 0.15$ and cap 6.0. The ground truth is
itself projected through Algorithm 1, so it satisfies the cap the model must satisfy; RAS does not
use the cap, an asymmetry that favours our method and that the capped-RAS check in Section 5.4
would isolate. We train for
3,000 epochs on seeds 0–4 and compare (i) the relative Frobenius error of $\hat W_t$ against RAS on
the same margins and (ii) block accuracy against the majority-class guess (0.70).

### 5.2 Reconstruction

The model's mean relative error is lower than RAS's in 5/5 seeds
(0.430 ± 0.043 vs. 0.566 ± 0.026); a paired t-test across seeds gives
p = 0.0101. (With five seeds, Wilcoxon's smallest attainable two-sided p is 0.0625, so we report
the t-test.) A capped-RAS baseline — plain Sinkhorn given the same per-pair cap the method uses
(Section 5.4) — lets us split that gain into its two sources. Of the 0.136 total improvement over
RAS, the cap supplies 0.101 (RAS 0.566 → capped RAS 0.465, paired p = 0.028) and the
covariate-informed learned prior supplies the remaining 0.036 (capped RAS 0.465 → model 0.430),
the latter small but consistent across all five seeds (paired p = 0.0011). (The two components
are computed from the unrounded seed means and so need not equal the differences of the
three-digit figures shown here.) About three-quarters of
the win is therefore the regulatory cap and one-quarter the learned prior; and that learned-prior
quarter is not the block structure. A single-block ($K=1$) ablation leaves reconstruction error
statistically unchanged (0.420 vs. 0.430, paired p = 0.36; Section 5.4) — and still below capped
RAS — so the gain beyond the cap comes from the covariates, the temporal encoder and the learned
decoder, not from the covariate-informed blockmodel.

### 5.3 Block recovery

Block accuracy exceeds the majority guess by 0.0781 on average (sd 0.0774), in 3 of 5 seeds
(paired t-test p = 0.1136). The margin is smaller than its spread, and two failure modes recur
across configurations: collapse to the uniform prior, and confident but wrong assignments. We
therefore do not claim that the model identifies core and periphery banks. Table C1 shows that the
mutual-information term does not change this at any weight tried.

### 5.4 What the synthetic win is and is not

The synthetic ground truth comes from the same blockmodel the method assumes, which favours the
method. The three checks this invites are now run. The capped-RAS baseline (Section 5.2)
attributes about three-quarters of the gain over RAS to the per-pair cap and one-quarter to the
learned prior. A single-block ablation ($K=1$) then isolates what the block channel itself
contributes: with the number of blocks cut from two to one, reconstruction error is statistically
unchanged (0.420 at $K=1$ vs. 0.430 at $K=2$, paired t-test p = 0.36, and still below the 0.465 of
capped RAS), while block accuracy collapses mechanically to a single block. Since the block labels
are not recovered in the first place (Section 5.3), this says the covariate-informed blockmodel
adds nothing to reconstruction beyond the cap, the smoothness prior, the temporal encoder and the
learned decoder: the learned-prior quarter of the gain is those components, not the communities.
A misspecification check completes the picture. When the generator has no block structure
(homogeneous blocks — a pure covariate/degree fitness model), the method still beats RAS in all
five seeds, but by far less: model 0.730 vs. RAS 0.759, a mean gap of 0.029 against the 0.136 it
has on block-structured data. Block accuracy sits exactly at the trivial majority guess (0.524 vs.
0.522, gap 0.002), so the model does not invent communities that are not there. Imposing the
blockmodel therefore does not hurt under misspecification: the reconstruction advantage shrinks
toward the cap-only margin and the latent structure stays inert, consistent with Sections 5.2–5.3.

A persistence baseline is reported on the real panel in Section 6.5; on the synthetic panel it is
uninformative, because the generator redraws blocks and covariates independently each period, so
consecutive $W_t$ carry no temporal dependence for a persistence rule to exploit.

## 6. Real-data results

### 6.1 The panel and who is missing

The 16 banks are ADBNL, EBL, Global, HBL, Kumari, Laxmi, MBL, NABIL, NBL, NIC, NMB, NSBI, Prime,
RBB, SBL and SCBNL. Three facts limit what the panel represents.
- **It is a closed system.** Exposures to the other commercial banks, development banks, NRB and
  foreign banks are outside the matrix.
- **The discard is not size-neutral.** Excluded banks are smaller (pre-window size medians 64,910
  NPR mn included vs. 44,510 excluded; Mann–Whitney p = 0.005).
- **Some exclusions come from later months.** Over the clean window alone (45 months), 21 labels
  are complete. BOK (first incomplete month Poush 2079), NCC (Poush 2079), NIBL (Magh 2079),
  Sunrise (Asar 2080) and Mega (Asar 2076) are dropped only because the filter runs over all 133
  months. Why these series stop has not been verified.
- **Some exclusions may be relabels.** Citizens→Citizen, NBB→NBBL, PRABHU→Prabhu and Civil→CBL
  each switch label at Poush/Magh 2075 with no overlap and size and capital ratios of 0.969–1.042
  across the switch. Each half fails the complete-case rule. We have not checked these against NRB
  records, so "33" counts labels; there are at most 29 distinct banks if all four are relabels.

### 6.2 The published margins do not balance

In the clean window (45 months, 720 bank-months), lending is zero in 398 bank-months and borrowing
in 465. Within the 16 banks, total lending and total borrowing differ by more than 1% in 43 of 45
months and by more than 10% in 37; the median relative gap is 0.32 and the maximum 0.93 (Mangsir
2072: lending NPR 10,322 mn vs. borrowing NPR 743 mn). In 16 bank-months one bank lends more than
all the others borrow, and in 2 one bank borrows more than all the others lend. A non-negative
matrix with zero diagonal must have equal totals and must not let one row exceed the other
columns' total, so in 43 of 45 months no such matrix matches both margins, before the cap is
considered.

Within-panel imbalance is expected whenever the 16 banks are not a closed system: a panel bank's
lending total counts loans to banks outside the panel, whose matching borrowing is not in the
panel's column totals, so the two panel totals need not be equal even if every figure is correct.
This is the leading explanation. A difference in how the two bulletin lines are defined would also
produce it, and we cannot say from the published aggregates how much of the gap is open-system
leakage and how much, if any, is definitional; we have not checked the line definitions with NRB.

### 6.3 What the projection does with infeasible margins

With an empty intersection, Algorithm 1 returns its least-violation point, and that is the
$\hat W_t$ we report. Its worst monthly miss tracks the gap (Spearman 0.93): the median worst miss
is NPR 200.8 mn in the 43 infeasible months and 22.28 mn in the other 2. The largest misses are NPR
2,311 mn on lending and 437.8 mn on borrowing. The projection does not keep zeros: $\hat W_t$
assigns more than NPR 1 mn of borrowing to all 465 bank-months that report none (16 of 16 banks;
median 100.9, maximum 437.8) and more than NPR 1 mn of lending to 249 of the 398 that report none
(13 banks; median 27.1, maximum 833.4).

RAS fits the same margins differently. It rescales entries, so it keeps every zero, matches lending
to within NPR 41 mn, and leaves the whole gap on borrowing (deviations up to NPR 8,290 mn). Both
methods' worst misses track the gap (Spearman 0.93 for each, Paper 2 Section 8). The size of the
misfit is therefore a property of the published data; how it is split is a property of each
method. The two networks are not independent estimates: they share their targets, and a downstream
exposure measure built from them correlates 0.968 over the companion paper's post-mandate cells.

The cap rarely binds: 7.0% of entries over all 133 months and 1.9% in the clean window
($\kappa = 0.05$). On real data the cap adds little information beyond the margins; its role rests
on the synthetic results.

### 6.4 The model's latent structure does not reach the output

We ran two fits that differ only in the KL weight. With $\beta_{\max} = 0.15$ every bank's
posterior is near uniform (about 0.45/0.55); with $\beta_{\max} = 0$ the KL term saturates at 0.691
($\approx \ln 2$) and all 16 banks sit at $\ge 0.999$ on the same block. The two posteriors are at
opposite degenerate ends, yet the two $\hat W_t$ series differ by a relative Frobenius difference of
$1.5\times10^{-5}$ (maximum absolute difference NPR 0.18 mn). On a matched synthetic pair, where the
reconstruction term is active, a much smaller posterior change (mean absolute difference 0.044)
moves $\hat W_t$ by a relative difference of 0.285.

This follows from Section 3.6. Without ground truth, no loss term rewards any particular matrix
among those near the margins, and the decoder receives gradient only through the smoothness
penalty. On real data, $\hat W_t$ is the projection of a smooth, nearly arbitrary raw matrix onto
inconsistent margins, and the blockmodel and covariates do not detectably shape it.

### 6.5 Temporal stability

$\hat W_t$ changes less from month to month than RAS: the mean month-to-month change of $\hat W_t$
is 1.16 against RAS's 2.32 (lower is more stable; seed 0, 16 banks over 133 months). But this edge
is not evidence that the model recovers stable structure. A persistence baseline makes the point:
carrying last month's reconstruction forward and re-imposing the current month's margins and cap
through Algorithm 1 already gives a month-to-month change of 1.11 — essentially the model's own
1.16. The model departs from that carried-forward matrix by only 13%, so it adds almost nothing
beyond persistence plus marginal-fitting. The gap to RAS therefore reflects that RAS over-reacts to
month-to-month noise in the published margins, not that the model's smoothness prior or latent
structure contributes stability of its own [@vallescatala2018modelselection]. This is consistent
with Section 6.4: on real data the decoder receives gradient only through the smoothness penalty.

## 7. Discussion and limitations

**What the method offers.** Where margins are consistent, the capped projection gives the
nearest matrix that matches the totals and respects a per-counterparty limit, and the synthetic
results show that the learned prior improves on RAS when there is a signal to learn. The projection
is a standalone, differentiable layer that other reconstruction models can use.

**What Nepal's bulletins allow.** Three findings bound any reconstruction from these data, not
only ours. First, the two published lending and borrowing totals do not balance within the
observed banks in 43 of 45 clean months — most likely because the panel is not a closed system,
possibly also because the two lines are defined differently — so an exact bilateral reconstruction
restricted to the panel does not exist and every method must choose where to put the misfit. Second, most bank-months report zero, so the margins carry little
information about who trades with whom. Third, without bilateral ground truth, nothing in the data
selects one matrix over another near the margins; our model's latent structure has no detectable
effect on its output. For NRB, the practical implication is that bilateral or large-exposure
reporting, even at low frequency, would be needed to make interbank exposure maps informative;
reconciling the definitions of the two published interbank lines would be a cheaper first step.

**Limitations.**
- *Ground truth.* All accuracy evidence is synthetic and from the assumed model class. The
  capped-RAS decomposition (Section 5.2), the single-block ($K=1$) ablation and the non-SBM
  misspecification check (Section 5.4), and the real-data persistence baseline (Section 6.5) are
  all now run.
  Even where bilateral data
  exist, reconstructions differ substantially from them [@mistrulli2011assessing].
- *Block identity.* Not recovered on synthetic data and not identifiable on real data; we make no
  claim about it.
- *Sample.* A closed system of 16 larger banks; the complete-case rule over the full span drops
  five banks that are complete in the clean window, and four possible relabels are unresolved.
- *Training window.* The model is trained on all 133 months, including bulletins whose line items
  fail the accounting identities from Chaitra 2075. Each month's matrix is fitted to that month's
  margins, but the smoothness penalty links Falgun 2075 to Chaitra 2075. A run on the clean window
  only would remove this and could raise N to 21; a larger panel would also show how much of the
  Section 6.2 imbalance is open-system leakage rather than definitional. It is not done.
- *Single seed on real data.* Seed sensitivity of the real-data $\hat W_t$ is not measured, and
  reconstruction uncertainty is not propagated to downstream uses.
- *Model size.* The reconstruction network has 2,133 trainable parameters (GRU encoder 1,200;
  block-posterior MLP 306; pairwise decoder 626; one scalar dispersion), fixed entirely by the
  architecture ($d=5$ covariates, hidden width 16, $K=2$ blocks) and independent of panel size — the
  same network serves the synthetic ($N=24$, $T=16$) and real ($N=16$, $T=133$) panels. On the real
  panel this is comparable to $N \times T = 2{,}128$ and smaller than the $2NT = 4{,}256$ observed
  marginals, and about two orders of magnitude below the supervised synthetic target entries used in
  training. The network is therefore under- rather than over-parameterized: amortization replaces the
  $N(N-1)T \approx 3.2 \times 10^4$ per-period latent unknowns a non-amortized reconstruction would
  carry with $\approx 2.1 \times 10^3$ shared weights, a $>10\times$ reduction in effective degrees of
  freedom. The binding constraint is not model capacity but identifiability of the bilateral matrix
  from its margins (Section 6.4), which no parameter budget resolves.

## 8. Conclusion

We give a differentiable, capped projection for interbank reconstruction and embed it in a
covariate-informed blockmodel. On synthetic data the combination reconstructs better than RAS but
does not recover blocks. On Nepal's bulletins the published lending and borrowing totals do not
balance within the observed banks, the projection's misfit tracks that imbalance, and the model's
latent structure leaves the output unchanged. The result is a measured data limit: the published
marginals cannot say who is exposed to whom, and we state what reporting would be needed to change
that.

## Data and code availability

The bulletins are public on NRB's website. The extraction, model, synthetic generator, all
configurations and the evaluation scripts are released at [TO FILL: repository URL]. Every number
in Sections 5–6 is produced by a named script: synthetic results by `run_synthetic_experiment` and
`seed_significance`; real-data fits by `run_real_experiment` (`baseline.yaml`,
`ablation_no_kl_pressure.yaml`); margin balance, zero-cell fit, cap binding, panel window and
survivorship checks by the companion paper's `reviewer_checks`, `check_paper1_interface`,
`panel_window_checks` and `survivorship` scripts, which read the saved reconstruction without
retraining.

## Declarations

[TO FILL: funding; competing interests; author contributions; use of AI tools in preparation.]

## Appendix A — Algorithms 2 and 3

```
Algorithm 2  SampleSyntheticNetwork(N, K, π, Λ, H, w, σ², X, κ, K_cap)
 1: for i: z_i ~ Categorical(π);  φ_i ← exp(wᵀX_i)
 2: for i ≠ j: λ_ij ← Λ[z_i,z_j] φ_i φ_j;  e_ij ~ Bernoulli(σ(H[z_i,z_j]))
 3:            W̃_ij ← e_ij · LogNormal(log λ_ij, σ²)
 4: W_true ← CapacitatedProjection(W̃, rowsums(W̃), colsums(W̃), κ·K_cap)
 5: return W_true, rowsums(W_true), colsums(W_true)

Algorithm 3  TrainStep(batch {X_t, r_t, c_t, K_t}, optional W_true_t)
 1: for t in temporal order:
 2:     h_i,t ← GRU([X_i,t; r_i,t; c_i,t], h_i,t−1);  q_i ← softmax(MLP(h_i,t))
 3:     z_i ← GumbelSoftmax(q_i)
 4:     (η_ij,t, λ_ij,t) ← Decoder(h_i,t, h_j,t, z_i, z_j)
 5:     W̃_ij,t ← GumbelSigmoid(η_ij,t) · LogNormal(log λ_ij,t, σ²)
 6:     W_t ← CapacitatedProjection(W̃_t, r_t, c_t, κ·K_t)
 7:     L_recon,t ← −log p(W_true_t | λ_t, η_t, σ²) if W_true_t given, else 0
 8:     L_KL,t ← KL(q(z_t) ‖ p(z))
 9: L ← Σ_t (L_recon,t + β·L_KL,t) + λ_smooth Σ_t ‖W_t − W_t−1‖²_F
10: backpropagate through Algorithm 1; optimizer step
```

## Appendix B — Variables

| Symbol | Bulletin line | Side | Transformation |
|---|---|---|---|
| $r_{i,t}$ | `a. Interbank Lending` (under `3 SHARE & OTHER INVESTMENT`) | Assets | as reported |
| $c_{i,t}$ | `b. Interbank Borrowing` (under `2 BORROWINGS`) | Liabilities | as reported |
| $K_{i,t}$ | `1 CAPITAL FUND` | Liabilities | as reported; cap = $\kappa K_{i,t}$ |
| $X^{(1)}$ size | sum of the ten top-level asset rows (`1 LIQUID FUNDS` … `10 NON BANKING ASSETS`) | Assets | sum; $r_{i,t}$ not added (already inside item 3) |
| $X^{(2)}$ | `1 CAPITAL FUND` | Liabilities | ÷ size |
| $X^{(3)}$ | `1 LIQUID FUNDS` | Assets | ÷ size |
| $X^{(4)}$ | `4 LOANS & ADVANCES` ÷ `3 DEPOSITS` | both | ratio |
| $X^{(5)}$ | `a. NRB` ÷ `2 BORROWINGS` | Liabilities | ratio; denominator = sum of sub-items a.–e. when the subtotal is not printed |

## Appendix C — Mutual-information term sweep (synthetic, 5 seeds each)

| `lambda_mi` | block-accuracy gap, mean | sd | seeds beating majority guess |
|---|---|---|---|
| 0.00 | 0.0781 | 0.0774 | 3/5 |
| 0.05 | 0.0688 | 0.0823 | 3/5 |
| 0.15 | 0.0682 | 0.0790 | 3/5 |
| 0.30 | 0.0656 | 0.0814 | 3/5 |

Paired tests of 0 against each nonzero weight: model error p = 0.3945, 0.2101, 0.3251; block
accuracy p = 0.4047, 0.1722, 0.0441 (the last in favour of 0, mean paired difference 0.0125). No
correction for three comparisons. We set the weight to 0.
