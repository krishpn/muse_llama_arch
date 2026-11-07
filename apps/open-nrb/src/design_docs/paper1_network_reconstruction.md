# Paper 1 — Who Lends to Whom? Reconstructing Interbank Exposures from Public Balance Sheets

**Status:** core algorithms (capacitated projection, synthetic generator, amortized training step)
implemented and unit-tested; first end-to-end synthetic validation experiment (§6/§12) running and
under active iteration — not yet a final result. **Target venue:** Journal of Network Theory in
Finance (decided 2026-11-05; fallbacks TMLR / ICAIF / Physical Review E / Applied Network Science /
Journal of Financial Stability). ICLR/NeurIPS dropped 2026-11-04; the ≤7-page budget below was set
for those venues and needs revisiting for the chosen venue.

**One-line description:** Reconstruct the latent bilateral interbank exposure matrix $W_t$ for
Nepali BFIs from NRB bulletin data, using an SBM-structured, covariate-informed generative model
with a differentiable regulatory-constraint projection layer, amortized by a GNN/GRU inference
network.

This supersedes the earlier version of `feature_model_mapping.md`, which had drifted from the
actual design (it invented a generic "Bayesian-net structural-constraint" layer built from
accounting identities — that was never part of the real design and is corrected below).


## 0. Problem framing and contribution to policy

**Plain-language summary.** Nobody reports who lent to whom between banks — only each bank's
total lending and total borrowing. This paper estimates the full bank-to-bank map from those
totals plus each bank's public financial profile, with no new data collection required. On
synthetic networks with known answers, the method recovers the structure and beats the classical
RAS baseline (§6, §10). On NRB's real bulletins, it also measures how far the published totals
can go: they are mostly zero and do not balance within the 16 banks, so they largely fix the
reconstructed network themselves, and "who is exposed to whom" cannot be read off them in detail
(§9.9, §5). The paper treats the real-world messiness of the raw bulletins (§9) as an honest
constraint to engineer around and report, not something to hide or idealize away.

**Problem framing (what "the network" means here).** We do not pre-specify the edges of the
interbank network. Following the network reconstruction literature — both its
econometric/maximum-entropy lineage (@squartini2018reconstruction) and its Bayesian
inference-under-unknown-observation-process lineage (@peixoto2018reconstructing,
@guimera2009missing) — we treat the bilateral exposure structure as latent and infer it
from publicly observable aggregates — balance-sheet marginals and bank-level covariates. Unlike
that Bayesian line, which typically reconstructs from noisy or partial pairwise reports of a
single network, the observation process here is a deterministic marginal constraint (Eq. 5): we
never see even a noisy edge report, only each bank's row/column sums. A node is
a bank; a directed edge $i \to j$ is $W_{ij,t}$, the exposure of bank $i$ to bank $j$; neither the
edge set nor the weights are observed, only the row/column marginals and covariates are. This is
inference under partial observability, not analysis over an already-known topology.

**Contribution to policy.** No central bank in a resource-constrained setting will mandate new
bilateral exposure reporting purely for research. This method doesn't require that: a
continuously-updatable estimate of systemic interconnectedness is produced from data NRB already
publishes monthly, with zero new reporting burden, and a pipeline other analyses (stress tests,
contagion simulations, Paper 2's spillover estimation) can reuse. This generalizes beyond Nepal:
most emerging-market central banks publish the same kind of aggregate call-report data and face
the identical constraint.

**What the real data allow (2026-11-05, from Paper 2's runs on the main window: 16 banks, Asar
2072 – Falgun 2075, 45 months, 720 bank-months).** Lending is zero in 398 and borrowing in 465 of
the 720 bank-months, and the median of each is 0 [Paper 2 `row_normalised` diagnostic]. On real
data $\hat W_t$ barely depends on the model state (relative difference 1.5e-5, §10), so it is set
mainly by the projection, the cap and the smoothing. A spillover test built on $\hat W_t$ could
detect only effects of about 2–5 within-bank SDs of the outcome per SD of paid-up exposure (Paper
2 Table 6) [`power_mde`]. The policy result is therefore a measured data gap: NRB's two published
marginals do not support an exact bilateral reconstruction, and bilateral or large-exposure
interbank reporting would be needed for that. The question "who is exposed to whom" is not made
answerable by this method on the published marginals alone.

**A second contribution: the method is demonstrated under real regulatory data friction, not
idealized data access.** Withheld or never-collected bilateral exposure data is the documented norm
across central banks, not an NRB idiosyncrasy: @upper2011simulation surveys this constraint across
the contagion-simulation literature; @mistrulli2011assessing is one of the few studies with genuine
bilateral ground truth (Italy) and shows how far reconstruction methods diverge from it — real
bilateral validation is the gold standard precisely because it's this informative, and precisely
why it's structurally unavailable for a setting like Nepal's (§10, §12 Discussion/limitations);
@gandy2013financial builds a full methodology around the same absence; and the maximum-entropy
ensembles of @squartini2017maximum/@squartini2018reconstruction are a direct response to the same
gap, though @anand2018filling shows they over-disperse edges and underestimate systemic risk
relative to true bilateral structure — exactly the divergence this paper's own Gandy-Veraart/RAS
baseline comparison (§7/§12) is designed to surface. Against that backdrop, NRB's bulletins
(scanned PDFs, inconsistent item-numbering across eras (§9.3), reused lettered sub-items (§9.4), an
unexplained multi-bank blackout (§9.9)) are a representative case, not a worst case, and carry no
bilateral exposure field at all, by construction (§10) — the same absence @anand2018filling and
@gandy2013financial already built their methods around.

This project had to build its own input table from NRB's release format and commit to an explicit
rule for unrecoverable fields. §9's missingness is traceable to specific causes (an era's
numbering scheme, a bank absent from one page), not independent of bank/period/label identity —
i.e., not missing completely at random [@littlerubin2019missing]. Under non-random missingness,
naive imputation would inject bias correlated with the extraction failure modes in §9; this
pipeline instead uses the classical complete-case alternative — discard the cell, never impute,
report the discard rate and cause (§9.9).

This data friction is structural, not Nepal-specific: the G20 Data Gaps Initiative
[@fsbimf2009infogaps] was launched because the 2008 crisis revealed even advanced-economy
regulators lacked granular interconnectedness data, and BCBS 239 [@bcbs2013principles] documents
that granular data aggregation remains a known weakness inside systemically important banks
themselves. The resulting panel size (N=16 of 33 commercial banks, §9.9) is reported with every
discard traceable to a named cause, following the missing-data convention the literature
prescribes. The discard is not size-neutral (excluded banks are smaller, §9.9), and the panel is a
closed system of 16 banks, so this is a stated limitation rather than evidence that the method
works.

**Citation status**: @upper2011simulation, @gandy2013financial, @squartini2017maximum,
@squartini2018reconstruction, @anand2018filling, @peixoto2018reconstructing,
@peixoto2019reconstruction, @guimera2009missing, @martin2016structural, @peixoto2020latent,
@peixoto2022disentangling, and @vallescatala2018modelselection all now have real entries in
`book.bib` (added 2026-11-01). **Update, 2026-11-01**: @martin2016structural's
conflicting details — author `Brian Ball` vs. `Sharon X. Ball`, page/DOI `012306` vs. `012308` —
are resolved: the paper's own text (Martin, Ball, Newman, *Structural inference for uncertain
networks*, arXiv:1506.05490, Phys. Rev. E 93, 012306 (2016)) confirms `Brian Ball`/`012306`, the
version already kept in the `.bib`; `Sharon X. Ball`/`012308` was simply wrong. Still missing
real bib entries: @mistrulli2011assessing, @littlerubin2019missing, @fsbimf2009infogaps, and
@bcbs2013principles — treat these as plausible-from-training-data recall only until confirmed:
Mistrulli, *Assessing financial contagion in the interbank market: Maximum entropy versus observed
interbank lending patterns*, Journal of Banking & Finance, 2011; Little & Rubin, *Statistical
Analysis with Missing Data*, 3rd ed., Wiley, 2019; FSB & IMF, *The Financial Crisis and
Information Gaps*, 2009 (the G20 Data Gaps Initiative's founding report); BCBS 239, Basel
Committee on Banking Supervision, *Principles for effective risk data aggregation and risk
reporting*, 2013.

## 1. Why this is its own paper

Separated from the causal/policy question (Paper 2) because the two contributions differ: this
paper's contribution is a reconstruction method with no causal-identification content; Paper 2's
contribution is a causal-identification design that *consumes* this paper's output as a fixed
input. The link as it stands (2026-11-05): Paper 2 uses one seed's $\hat W_t$ as a fixed exposure
weight, so reconstruction uncertainty is not propagated into its estimates (§10's venue/scope
hook names that propagation as possible future work; it is not done). Bundling them would force a methods paper to carry a policy
narrative it doesn't need, and vice versa.

## Literature Review

**Plain-language summary.** Three existing approaches try to guess a hidden bank network. (1)
Classical balancing methods spread known totals across a plausible network using entropy/balancing
math (RAS-style) — simple, but tends to over-guess weak connections. (2) A newer line uses
block/community structure plus deep learning, faster than older Bayesian sampling. (3) A separate
Bayesian line treats reconstruction as inference from noisy *direct* reports of individual
connections. This paper's setting is harder than all three: there is no report about any
individual connection at all, only the two totals — so the comparable baseline is the classical
balancing method (#1), while the Bayesian line (#3) is conceptually close but not a drop-in
alternative (§7 explains why it isn't benchmarked directly).

Regulatory balance sheets provide aggregate asset/liability totals, but exact bilateral
counterparty exposures remain unobserved — researchers address this missing-data problem via
classical maximum-entropy optimization or, more recently, amortized variational inference.

Matrix-balancing techniques adapted from @deming1940least and @stone1961input were applied to
interbank markets by @upper2011simulation and @gandy2013financial; @squartini2017maximum and
@squartini2018reconstruction synthesized these into entropy-maximizing ensembles, which
@anand2018filling shows over-disperse edges and underestimate systemic risk via overly dense
topologies. To capture core-periphery hierarchies (@clementi2019tiered), a second line builds on
the stochastic blockmodels of @holland1983stochastic and @nowicki2001estimation, combined with
deep relational learning to avoid slow MCMC — e.g. @kipf2018neural's Neural Relational Inference
(amortized variational inference over interaction graphs) and @mena2018learning's Gumbel-Sinkhorn
networks (differentiable discrete matching). A third, Bayesian line poses reconstruction itself as
inference under an unknown observation process: @peixoto2018reconstructing and
@martin2016structural formalize this directly for indirect/uncertain reports of a single network;
@guimera2009missing and @peixoto2019reconstruction jointly infer missing/spurious edges with block
or community structure — the same joint edge-existence/block-assignment problem this paper poses,
but from noisy pairwise reports rather than row/column marginals. @peixoto2020latent's
heterogeneous-density weighted-edge model is the same degree-correction/fitness idea behind this
paper's $\phi_{i,t}=\exp(w^\top X_{i,t})$ term (Eq. 2), applied there to Poisson rather than
log-normal weights.

These ideas combine to let models capture temporal dynamics and regulatory constraints (e.g.
exposure caps) while remaining gradient-trainable. Because true bilateral exposures are unobserved
in central bank panels, validation relies on indirect metrics — temporal stability, divergence from
classical maximum-entropy benchmarks (@anand2018filling).


## 2. Notation

| Symbol | Meaning |
|---|---|
| $i, j \in \{1,...,N\}$ | institutions (BFIs) |
| $t \in \{1,...,T\}$ | periods (NRB bulletin months) |
| $X_{i,t} \in \mathbb{R}^d$ | node covariates for bank $i$ at time $t$ |
| $r_{i,t}, c_{i,t} \ge 0$ | observed interbank lending / borrowing marginals |
| $K_{i,t}$ | core capital (used to set the regulatory single-obligor cap) |
| $W_t \in \mathbb{R}_{\ge 0}^{N \times N}$ | **latent** bilateral exposure matrix — never observed, the reconstruction target |
| $z_i \in \{1,...,K\}$ | latent block/tier membership (e.g. core vs. periphery) |

## 3. Generative model

**Plain-language summary.** This is a toy story for how the hidden network *could* have been
generated, so the model has something concrete to learn to invert. Each bank secretly belongs to
a type (core/periphery); each pair's expected exposure depends on both banks' types and their own
financial profile; most pairs have zero exposure, a few have a nonzero amount drawn from a skewed
distribution; and whatever matrix comes out must still add up to the two totals NRB actually
publishes — that last requirement is why there's a dedicated fix-up layer (§5) rather than just a
probability term in the loss.

- Block membership: $z_i \sim \text{Categorical}(\pi)$ — (Eq. 1)
- Block-pair intensity: $\Lambda \in \mathbb{R}_{\ge0}^{K\times K}$ — the SBM core
- Fitness/degree-correction from covariates: $\phi_{i,t} = \exp(w^\top X_{i,t})$ — (Eq. 2)
- Latent edge intensity:
  $$\lambda_{ij,t} = \Lambda_{z_i z_j} \cdot \phi_{i,t}\phi_{j,t} \cdot \exp(h_{i,t}^\top h_{j,t}) \qquad \text{(Eq. 3)}$$
- Zero-inflated edge weight (most bank pairs have *no* exposure):
  $$\Pr(\tilde W_{ij,t}=0) = \sigma(-\eta_{ij,t}), \qquad \tilde W_{ij,t} \mid \tilde W_{ij,t}>0 \sim \text{LogNormal}(\log\lambda_{ij,t}, \sigma^2) \qquad \text{(Eq. 4)}$$
- Marginals are a **deterministic function** of $W_t$:
  $$r_{i,t} = \sum_j W_{ij,t}, \qquad c_{j,t} = \sum_i W_{ij,t} \qquad \text{(Eq. 5)}$$
  This means the marginal constraint belongs in the *architecture* (a hard projection), not the likelihood — see §5.

## 4. Inference network (amortized, PyG GNN + GRU)

**Plain-language summary.** This section is the model's "guessing machine": given only what's
actually observed (bank covariates, plus the two marginals $r,c$), it has to guess (1) each bank's
type — core vs. periphery — and (2) how much exposure exists between every pair of banks. It is
never allowed to look at $W_t$ itself, since that's exactly the thing being reconstructed. A GRU
builds a running per-bank summary from covariates and marginals each month; a small classifier
reads that summary to guess block type; a pairwise decoder reads two banks' summaries plus their
guessed types to guess whether/how much exposure exists between them. $r,c$ were added to the
GRU's input because covariates alone carry zero information about block membership (§3 draws them
independently) while the marginals do, via $\Lambda$ — not a tuning choice, a logical necessity.
The decoder replaces the literal $\Lambda,w$ computation from §3 with a learned function, so
$\Lambda,w$ remain synthetic-data-generation-only parameters, never trained.

The encoder runs on covariates/history only — it cannot message-pass over $W_t$ itself, since
that would be circular (the whole point is that $W_t$ is unobserved).

- Temporal encoder: $h_{i,t} = \text{GRU}_\theta([X_{i,t}; r_{i,t}; c_{i,t}], h_{i,t-1}) \qquad \text{(Eq. 6)}$,
  optionally refined by a GNN layer over an auxiliary *observed* graph if one exists (e.g.
  common-sector exposure — see §8). $r_{i,t}, c_{i,t}$ (observed interbank lending/borrowing
  marginals — real, measured aggregate fields, not derived from the model's own inference of
  the latent $W_t$, so this doesn't violate the no-message-passing-over-$W_t$ rule above) were
  added to the encoder's input after the first synthetic validation run showed block recovery is
  *impossible* from $X$ alone: $z$ is drawn independently of $X$ (Eq. 1 vs. Eq. 2), so
  $q_\psi(z)$ conditioned on $X$-history only has no signal to learn from, no matter how the KL
  term in Eq. 9 is weighted (§10).
- Amortized block-assignment posterior: $q_\psi(z_i \mid X_{i,1:t}, r_{i,1:t}, c_{i,1:t}) = \text{Categorical}(\text{softmax}(\text{MLP}_\psi(h_{i,t}))) \qquad \text{(Eq. 7)}$
- Pairwise decoder produces $(\eta_{ij,t}, \lambda_{ij,t})$ from $(h_{i,t}, h_{j,t}, z_i, z_j)$ —
  full amortized variational posterior $q_\psi(W_t \mid X_{1:t}, r_t, c_t)$. This supersedes the
  literal Eq. 3-4 computation: the decoder never reads $\Lambda$ or $w$, so those stay
  generative-model/Algorithm-2-only parameters, never trained (§6, Appendix A Algorithm 3).
- Reparameterized sampling: log-normal reparam trick for the continuous weight, Gumbel-softmax
  for the discrete existence gate.

## 5. Constraint-satisfaction layer (the technical core of the paper)

**Plain-language summary.** Whatever raw matrix the model produces almost certainly won't match
the two published totals (lending, borrowing) or the regulatory exposure cap exactly. This layer
is the fix-up step: take that raw guess and nudge it, in the smallest way possible, until it
exactly satisfies both totals and the cap — mathematically guaranteed to find that exact nudge
(not an approximation) whenever one exists, and fully compatible with gradient-based training. On
NRB's real marginals one usually does not exist (see "Feasibility on real data" below), and the
layer then returns a compromise that misses the totals. An
earlier, simpler attempt at this fix-up (a one-line Sinkhorn clip) turned out to enforce the wrong
constraint entirely (a cap on each bank's row total, instead of a cap on each individual pair) and
was replaced with the approach below.

Differentiable **capacitated marginal projection**, via Dykstra's alternating-projections
algorithm (an ADMM-family method) onto the intersection of three convex sets:

$$W_t = \text{Proj}_{\mathcal{R}\cap\mathcal{C}\cap\mathcal{B}}(\tilde W_t) \qquad \text{(Eq. 8)}$$

where $\mathcal{R} = \{W : \sum_j W_{ij}=r_{i,t}\}$, $\mathcal{C} = \{W : \sum_i W_{ij}=c_{j,t}\}$,
$\mathcal{B} = \{W : 0 \le W_{ij} \le \kappa K_{i,t}\ \forall j\}$ — i.e. the nearest-in-Euclidean-
norm matrix to $\tilde W_t$ that satisfies the observed marginals and the regulatory
single-obligor cap *exactly*, not approximately — **when the intersection is non-empty**. If it is
empty, Dykstra's iteration does not reach a point in it, and §11 (Algorithm 1 note) treats that
case as an infeasible data/parameter error that must not be silently accepted.

**Feasibility on real data (2026-11-05, Paper 2's runs on the main window: 16 banks, Asar 2072 –
Falgun 2075, 45 months).** The intersection is empty in most months, so the exactness above does
not hold for the real-data $\hat W_t$ [Paper 2 `reviewer_checks --only marginal_balance`,
`row_normalised` diagnostic]:
- total lending and total borrowing within the 16 banks differ by more than 1% in 43 of 45 months
  and by more than 10% in 37; median relative gap 0.32, max 0.93 (Mangsir 2072: lending 10,322 vs.
  borrowing 743, NPR mn);
- in 16 bank-months one bank lends more than all the others borrow, and in 2 one bank borrows more
  than all the others lend;
- 43 of 45 months therefore have no exact solution (non-negative, zero diagonal) even before the
  cap is considered;
- $\hat W_t$'s worst monthly miss tracks the gap (Spearman 0.926): median 200.8 in the failing
  months vs. 22.28 in the other 2; largest misses 2,311 on lending and 437.8 on borrowing;
- where a bank reports zero, $\hat W_t$ does not keep the zero: it puts borrowing above NPR 1 mn
  on all 465 bank-months that report no borrowing (16 of 16 banks; median 100.9, max 437.8) and
  lending above NPR 1 mn on 249 of the 398 that report no lending (13 banks; median row sum 27.1,
  max 833.4). RAS keeps every zero in both directions [Paper 2 `panel_window_checks`].

So the real-data $\hat W_t$ is the least-violation output that §11 says should not be silently
accepted, and the paper must report it as such. **Open decision for the authors**: balance the
marginals before projecting (e.g. rescale both to a common total), or keep the least-violation
output and report its misfit as above. The cause of the imbalance is not known: counterparties
outside the 16 banks, or different definitions of the two bulletin lines (`a. Interbank Lending`
vs. `b. Interbank Borrowing`, Appendix B 8a) are both possible. Check the line definitions before
naming one.

**How often the cap binds on real data (2026-11-05)** [Paper 2 `check_paper1_interface`]: implied
$\kappa = 0.05$; the cap binds for 7.0% of entries over all months and 1.9% in the main window.
The cap is the technical core of the method, but on real data it rarely constrains $\hat W_t$;
the case for it rests on the synthetic results. This answers §10a item 2 (cap unfairness) in
part: the cap gives the model little extra information over the uncapped RAS baseline on real
data.

**Revision note (superseded design):** an earlier draft used a capacitated Sinkhorn/entropic-OT
iteration with a `min(...)` row-scalar clip to enforce the cap. `test_sinkhorn.py`'s
`binding_cap_case` showed this doesn't implement $\mathcal{B}$ at all — a single row-scalar $u_i$
can only rescale an entire row uniformly, so "clip to satisfy the cap" collapsed into capping the
*row sum* ($\sum_j W_{ij} \le \kappa K_{i,t}$) rather than *each entry*
($W_{ij} \le \kappa K_{i,t}\ \forall j$) — a materially different, and wrong, constraint (total
interbank activity vs. concentration to any one counterparty). Plain (uncapped) Sinkhorn/IPF is
still used, unmodified, as the classical Gandy-Veraart/RAS baseline in §7/§12 — that comparison
never needed the cap in the first place, so it's unaffected by this fix.

References: Cuturi (2013) entropic OT (for the uncapped baseline); Boyle & Dykstra (1986) for the
alternating-projection algorithm; Mena et al. (2018) Gumbel-Sinkhorn (unrelated to this layer —
that's the discrete existence gate, Eq. 4). Flagged explicitly as "the one genuinely nontrivial
implementation piece" in the design discussion — this is the part most likely to need real
engineering effort, not the GNN/GRU scaffolding around it.

**Correction vs. the earlier `feature_model_mapping.md`:** there is no "Bayesian-net
structural-constraint edge" tying `TOTAL Liabilities = TOTAL ASSETS` or `Net Profit` into the
graph. That was never part of this design. The accounting identities in the raw NRB statements
are a **data-quality cross-check** on the extractor (do the reported totals reconcile?), not a
component of the generative model or the constraint layer.

## 6. Training objective

**Plain-language summary.** The model is trained to do three things at once: (1) when test data
has a known right answer (synthetic only), get close to it; (2) keep each bank's guessed type
from drifting too far from a neutral default, with a tunable ceiling on how hard this pressure is
enforced; (3) don't let the reconstructed network swing wildly month to month. On real data, (1)
is impossible — no known answer exists — so training leans entirely on (2) and (3), which is the
root of the "does the model actually learn anything real about bank types from real data"
question explored at length in §10.

$$\mathcal{L} = -\mathbb{E}_q[\log p(W_t^{\text{true}})]_{\text{synthetic only}} + \beta_t\,\text{KL}(q_\psi(z)\,\|\,p(z)) + \lambda_{\text{smooth}}\sum_t \|W_t - W_{t-1}\|_F^2$$

No $\lambda_{\text{reg}}\|\Lambda, w\|^2$ term: $\Lambda$ (Eq. 3) and $w$ (Eq. 2) are generative-model/
Algorithm-2-synthetic-sampler parameters only — Algorithm 3's amortized inference replaces the
literal Eq. 3-4 computation with $\text{Decoder}_\psi(h_i, h_j, z_i, z_j)$ (§4), which never reads
$\Lambda$ or $w$, so a regularizer on them would have no forward-pass effect to actually regularize
and was dropped rather than wired in.

- **$\beta_t$ (KL annealing weight)**: ramps linearly $0 \to \beta_{\max}$ over the first
  `KL_WARMUP_EPOCHS` epochs (standard $\beta$-VAE warm-up), giving $q_\psi(z)$ room to latch onto
  block signal before the KL penalty pulls it toward $p(z)$. The ceiling matters more than ramp
  speed: $\beta_t = 1$ eventually out-competes whatever reconstruction-gradient signal keeps
  $q_\psi(z)$ informative, so $\beta_{\max} < 1$. Current value: $\beta_{\max} = 0.15$
  (`KL_BETA_MAX` in the synthetic experiment script) — **partially validated, seed-dependent**: a
  single seed (0) holds block-recovery accuracy at 0.7578–0.7682 (vs. that run's trivial baseline
  of 0.7344), but a 5-seed replication shows this does **not** generalize as a fix — block-recovery
  margin over the trivial baseline is $+0.0234, -0.1380, +0.1380, -0.1797, -0.0026$ (mean
  $-0.0318$, std $0.1148$ — std nearly 4× the mean), with only 2/5 seeds beating the trivial
  baseline at all. Two distinct failure modes recur: full collapse to the uniform prior
  ($\text{KL} \approx 0$, e.g. seed 4) and confident-but-wrong block assignment (e.g. seed 1).
  Reconstruction accuracy, by contrast, **is** robust across the same 5 seeds: model mean relative
  error beats the classical Sinkhorn/RAS baseline in 5/5 seeds (mean 0.4154 vs. 0.5657). These
  figures are from the pre-rescale 5-seed replication and are superseded: the later 5-seed run
  (§10) gives 0.4219 ± 0.0357 vs. 0.5657 ± 0.0257, 5/5 seeds, and with `lambda_mi=0` (the reported
  config) `model_mean` beats the baseline at $p=0.0101$ (§10, Step 3). Cite the §10 figures. See §10
  for what this means for the paper's claims.

- **Synthetic data** (known ground-truth $W_t$ from an SBM generator): the first term is a real
  supervised reconstruction loss — this is the main validation evidence.
- **Real data** (no $W_t$ is ever observed): the first term drops out entirely. What's left is a
  regularized/MAP-style constrained fit, *not* likelihood-based estimation. Evaluation on real
  data has to rely on indirect evidence: temporal stability of the recovered $W_t$ sequence, and
  comparison against a classical baseline (§8). The stability comparison is partly circular: the
  model is trained with the $\lambda_{\text{smooth}}$ term and RAS is not (§10a item 1), so lower
  month-to-month change is partly by construction. Report it as consistency with the smoothness
  prior, not as evidence of recovered structure; the evaluation rests on the synthetic results.
- **Point estimate at inference/evaluation time**: training (Algorithm 3) samples $z, e, W$ via
  the reparameterized Gumbel-softmax/Gumbel-sigmoid/log-normal draws so gradients can flow, but
  reporting $\hat W_t$ from a single such draw carries irreducible sampling variance even once
  $(\eta, \lambda, q)$ are well-calibrated — this showed up empirically as a flat/noisy
  `model_rel_error` in the first synthetic run while the reconstruction likelihood term improved
  steadily underneath it. At inference/evaluation time, $\hat W_t$ should instead be built from
  the posterior-predictive **mean** at each stage: $q_\psi(z)$ itself (no Gumbel-softmax draw),
  $\sigma(\eta)$ for existence (no Gumbel-sigmoid draw), and $\lambda \cdot e^{\sigma^2/2}$ for
  magnitude (the log-normal mean, no reparameterized draw) — then project through Algorithm 1 as
  usual. This is a deliberate design decision, not just an experiment-script detail: it governs
  how both the synthetic validation (§12) and the real-data baseline comparison (§7) numbers get
  computed.

## 7. Design rationale

**Plain-language summary.** Three deliberate "keep it simple" choices, each justified below:
prefer the simpler model/architecture at every fork given how little real data exists, and always
check against a classical baseline rather than assuming the deep model wins.

- **SBM over preferential-attachment/fitness models** as the sole structural backbone: core-
  periphery tiering in interbank markets is literally a latent-block story, so SBM is the natural
  fit. PA/fitness models (Caldarelli et al.) are cited in related work for contrast but not built.
- **GRU, not a temporal-GNN** (EvolveGCN, TGN, etc.) for the temporal mechanism: deliberately the
  simpler choice, given the "big model, small data" risk — roughly 50–80 institutions, at most
  ~170 periods. A fancier temporal-GNN would be over-parameterized for this panel size.
- **Mandatory baseline**: head-to-head comparison against Gandy–Veraart / RAS maximum-entropy
  network reconstruction, on both the synthetic SBM data (known ground truth) and the real panel.
  Explicit instruction from the design discussion: report honestly if the deep model doesn't beat
  the classical baseline — don't bury a negative result. **Why not Peixoto's Bayesian
  reconstruction line (@peixoto2018reconstructing, @peixoto2019reconstruction) as a second
  baseline**: that family of methods takes noisy/partial *edge-level* reports as input — some
  direct, if unreliable, signal about individual pairs. This paper's setting provides none: only
  aggregate row/column marginals (§0), zero edge-level signal of any kind. RAS/Gandy-Veraart is the
  comparable baseline because it shares that input regime; Peixoto's line solves a
  different-input-regime version of a similar problem, not a drop-in alternative without first
  adapting it to marginals-only input — an adaptation out of scope here.

## 8. Variable inventory (what actually feeds the model)

**Plain-language summary.** The model needs surprisingly little: about 5 financial-health numbers
per bank per month, plus the two totals (lending, borrowing) that are actually the thing being
reconstructed. Everything else in the NRB bulletins either belongs to Paper 2 or isn't usable with
current data.

Net statement from the design discussion: **the reconstruction model needs ~5 covariates per
node per period, plus the two interbank marginals. Everything else in the NRB bulletins is either
a Paper 2 outcome variable or not usable given current data coverage.** Full per-field provenance
— which raw bulletin line each symbol (marginals, covariates, auxiliary graph candidate) is
computed from, confirmed against `nrb_balance_sheet.csv`, the fields explicitly excluded, and the
one-to-one equation-symbol ↔ raw-variable map — moved to **Appendix B** (subsections 8a-8e) to
stay within the ≤7-page budget.

## 9. Known data-pipeline caveats (apply before this goes into the model)

**Plain-language summary.** Real NRB bulletins are messy scanned PDFs with inconsistent
numbering across years. Items 1-8 below are the specific messes the extraction code has to
handle (an engineering checklist, not reader-facing). Item 9 is the one that matters for the
paper: when a bank/period's data genuinely can't be recovered, it's dropped rather than guessed —
and that dropped-data count is reported honestly as a limitation (§12), not hidden.

**Reader-facing vs. internal split**: items 1-8 below are internal PDF-extraction/data-engineering
caveats (`item_code` collisions, era drift, interleaved banks, etc.) — implementation detail the
pipeline already handles, not something a paper reader needs. Only item 9 (the complete-case
discard policy and resulting panel size) is paper-facing **limitations** material; it, together
with §10's block-identity scope-down, is what §12's Discussion/limitations row (~0.5p) draws from.
Items 1-8 stay in this design doc only.

1. **`item_code` collision**: confirmed by re-running `nrb_extractor.py` against a real bulletin —
   `item_code` is not unique across the Balance Sheet's Liabilities and Assets sides. Both sides
   independently restart numbering at 1 (e.g. `1 CAPITAL FUND` on Liabilities vs. `1 LIQUID FUNDS`
   on Assets; same collision pattern for `2`, `3`, and lettered sub-items `a.`–`g.`). The extractor
   (`nrb_extractor.py:197-269`, `_extract_rows`) currently has no column marking which side a row
   is on — that currently has to be inferred externally from which `TOTAL` block (`TOTAL
   Liabilities` vs. `TOTAL ASSETS`) a row falls between on the page. **Fix needed**: add a `side`
   column to the extractor output before any join on `item_code` alone, or it will silently
   collide liabilities and assets rows that share a code (e.g. covariates 2 and 3 above, which
   both use an `item_code == 1` row from different sides).
2. Rotated bank-name header recovery (`recover_bank_names`, `nrb_extractor.py:130-171`) and the
   layout-mode row extraction are both working as validated against the real sample PDF — no open
   bug there as of this pass.
3. **`item_code` drift across eras**: the top-level code attached to a given labeled row is not
   stable over the panel's history — e.g. `LOANS & ADVANCES` is coded `4` in some eras and `4.1`
   in others (with `BILL PURCHASED` shifting from `5` to `4.2` correspondingly). Matching must key
   off the normalized label text (and, for lettered sub-items, parent section), never off
   `item_code` alone.
4. **Lettered sub-items (`a.`–`g.`) are reused across unrelated parent sections** (e.g. `a.` means
   `Paid-up Capital` under `1 CAPITAL FUND`, `Cash Balance` under `1 LIQUID FUNDS`, `NRB` under
   `2 BORROWINGS`, and `Interbank Lending` under `3 SHARE & OTHER INVESTMENT` — all in the same
   bulletin). The letter alone is not a key; a row's identity is (normalized label text, nearest
   preceding top-level/bare-number row in extraction order). The extractor should carry this parent
   link explicitly (a `parent_item` or `section` column) rather than requiring every consumer to
   replay the state machine itself. **Side can differ across the same reused label, too**: `e.
   Others` appears under both `3 DEPOSITS` (Liabilities) and `8 OTHER ASSETS` (Assets) in the same
   bulletin (confirmed `Asar-2074.pdf`/NBL, 2026-10-22). Side-consistency checks therefore must only
   be enforced for the small set of labels the model actually reads (§8b/§8e's 8 fields) — requiring
   every label in the bulletin to resolve to one consistent side will raise on this kind of
   legitimate, unrelated collision.
5. **Truncated-text duplicate rows**: some extractions clip a label mid-word (e.g. `4 LOANS &`
   alongside the correct `4 LOANS & ADVANCES`, or `1 CAPITAL` alongside `1 CAPITAL FUND`) for the
   same `item_code`/concept. Confirmed via `cut -d, -f7,8 ... | sort -u` on the real CSV
   (2026-10-22). Treat these as the same field (prefix match or longest-label-wins), not as
   missing data for the truncated variant.
6. **Pre-2075-era bulletins never extract a `TOTAL Liabilities` marker row at all** (confirmed
   against real `Asar-2074.pdf`, 2026-10-22) — the Liabilities section just ends and the Assets
   section starts back at `item_code "1"` with no boundary row to key off. Side inference needs a
   fallback for this era that doesn't depend on the marker existing: a drop back to a lower
   top-level bare-digit `item_code` while still in the Liabilities state is itself the side
   boundary (top-level codes strictly increase within one side).
7. **Real bulletins extract row-major by line item, interleaved across banks** — bank A's row for
   item N is immediately followed by bank B's row for item N, not bank A's entire sheet followed by
   bank B's (confirmed 2026-10-22; caused a live "inconsistent side" error before this was known).
   Side inference must replay its liabilities/assets state machine **per `(source_file, bfi_name)`
   bank-block**, re-sorted by original extraction order — not as one shared state machine over the
   raw file order — or one bank's `TOTAL LIABILITIES` marker flips the state for every other bank's
   very next row too.
8. **Blank value cells are dropped from extraction entirely, not reported as a zero.** A row with an
   empty value field (seen on optional/granular categories some banks don't report, e.g. `9 EXPENSES
   NOT WRITTEN OFF`) is simply absent from that bank/period's record, not present with value 0.
   Any field built by summing a fixed list of categories (e.g. the `size` covariate, §8b) must treat
   a missing category as 0 explicitly rather than looking it up strictly — a strict lookup wrongly
   drops the entire bank/period cell over one legitimately-absent optional row.
9. **Complete-case discard policy (data integrity over imputation)**: real bulletins will never fully
   resolve every label/era/row-numbering quirk in §9.1–§9.8 above — some bank/period cells are
   genuinely unrecoverable from the source PDFs. Confirmed examples: a required top-level field
   (`capital_fund`, `liquid_funds`, `loans_advances`, `deposits`) that no tolerant label-matching
   rule can find for that cell; and an entire bank missing from a bulletin page, e.g. BS 2080
   Asar–Kartik, where the same 13 of 33 commercial banks have **zero** extracted rows at all across
   5 consecutive months (hypothesized cause: a multi-page Commercial Banks table whose continuation
   page falls outside the TOC-derived page bound or lacks the `"BFI Name"` marker `pdf_extract.py`'s
   phase-1 bank-token detection requires — still under diagnosis as of 2026-10-28, not yet fixed).
   The pipeline's policy, deliberate by design (`data_loader.py`'s `Panel`/`_lookup_field`
   docstrings): leave the cell `NaN`, never substitute 0 or any imputed value, and let
   `_filter_complete_banks` (`run_real_experiment.py`) drop any bank that is not NaN-free across the
   *entire* modeling window before training reaches it. This trades panel size for correctness —
   e.g. starting the window at BS 2072 Asar (`MIN_PERIOD`, deferring a harder-to-diagnose pre-2072
   gap) plus the strict complete-case filter together left N=16 of 33 commercial banks over
   T≈133 months as of the 2026-10-28 session (exact N/T to be re-measured once the BS 2080 gap above
   is resolved or formally accepted as a second deferred window). **This discard count, and the
   specific named cause behind every discarded bank/period, must be reported explicitly in the
   paper (§12 Discussion/limitations)** — not smoothed over or silently interpolated — since it is
   the direct, auditable consequence of treating data integrity as non-negotiable: nothing is
   discarded without a diagnosed, citable reason (§9.1–§9.9 above, or an open item logged here until
   it is one).

   **Updates from Paper 2's runs (2026-11-05).**
   - *Panel length.* Paper 2's interface check found 133 of 134 calendar months in the saved
     artefact [`check_paper1_interface`]. The absent month is Kartik 2078, between Ashwin and
     Mangsir 2078 [Paper 2 `panel_window_checks`]. It is not the BS 2080 Asar–Kartik gap above,
     which is a bank-level gap (13 banks, 5 months), not a whole missing month. T = 133 stands;
     closed.
   - *Corrupted bulletins inside the training window.* Bulletin line items break down from Chaitra
     2075 onward; the window is clean through Falgun 2075 [Paper 2 Sec 5a audit]. Paper 1 trains
     on all 133 months, its `size` covariate sums the top-level asset rows that break, and the
     temporal smoothing links Falgun 2075 to Chaitra 2075, so the corruption can reach clean
     months too. The corruption has not been traced to NRB's PDFs rather than to the extraction,
     so it is not described as an NRB data error. Either run once on the clean window or state
     the caveat; this needs a run, so it is open.
   - *Closed system and survivorship.* The network covers only the 16 complete-case banks.
     Interbank positions with the other 17 commercial banks, development banks, NRB and foreign
     banks fall outside it. Excluded banks are smaller (pre-window size medians 64,910 NPR mn
     included vs. 44,510 excluded, Mann–Whitney p = 0.005). Paper 2's leverage dose does not
     differ between the groups (p = 0.246; 9 excluded banks with pre-window data, 8 have none);
     the same test has not been run on the paid-up dose [Paper 2 `survivorship`]. Report both in
     §12's limitations. This also partly answers §10a item 5.
     "33" counts bulletin labels, not distinct banks: it includes the four possible relabel pairs
     below (8 labels), so there are at most 29 distinct banks if those are confirmed.
     The complete-case filter runs over all 133 months, so banks are excluded even from months
     they report in full. Over Paper 2's main window alone (Asar 2072 – Falgun 2075, 45 months)
     21 labels are complete, against 16 over the full span [Paper 2 `panel_window_checks`]. The
     five dropped only by later months, with their first incomplete month: BOK (Poush 2079), NCC
     (Poush 2079), NIBL (Magh 2079), Sunrise (Asar 2080) and Mega (Asar 2076, inside the corrupted
     period). Why these series stop has not been verified (exits or mergers are possible; Mega's
     break may come from the extraction). Including them needs a run on the clean window, which
     is the same open run as the corrupted-bulletins item above.
   - *Open question: four banks may drop out because of a bulletin relabel, not missing data*
     [Paper 2 `label_continuity`]. Citizens→Citizen, NBB→NBBL, PRABHU→Prabhu and Civil→CBL each
     have 0 overlap months, a one-column gap at Poush/Magh 2075, and size and capital-fund ratios
     between 0.969 and 1.042 across the switch. Each half fails the complete-case filter, so all
     four banks drop out of N = 16. This is a data-continuity pattern only and has not been
     checked as a rename. If NRB records confirm these as relabels, merging the labels could add up
     to four banks to the panel. That changes the network and needs a re-run, so it is the
     authors' decision; nothing has been changed.

## 10. Open items (not yet decided)

**Plain-language summary.** The two biggest unresolved questions: (a) the model currently can't
reliably tell you which banks are "core" vs. "periphery" from real data — the reconstructed
network itself still looks good, but that bank-type label doesn't come free with it, so the claim
is being dropped rather than oversold; (b) it isn't yet proven that the model's good results
aren't just an artifact of testing it against fake data built from the model's own assumptions — a
planned second test, using a differently-generated fake dataset, is meant to settle this. The rest
of this section is smaller, decided/undecided design knobs.

- **Model-misspecification robustness check (not yet run)**: current synthetic validation (§12)
  generates ground truth from the same SBM the model assumes — self-fulfilling by construction, and
  the sharpest version of the "no real ground truth" objection (§0, @mistrulli2011assessing): since
  real bilateral validation doesn't exist for this setting, the next-best evidence is showing the
  method isn't just recovering its own assumptions. Planned fix: generate a second synthetic set
  from a different edge-formation process (e.g. pure covariate-fitness/preferential-attachment, no
  block structure at all) using the same Algorithm 2 infrastructure with different generative
  assumptions plugged in, and check the model still beats RAS — or degrades gracefully, not
  catastrophically — when its own structural assumption (SBM) is wrong. Not yet started.
- **$K$ (number of SBM blocks)**: start with $K=2$ (core/periphery), treat larger $K$ as a
  robustness check, not the headline specification.
- **$\sigma^2$ (log-normal dispersion)**: learned, either globally or per-block — not yet decided
  which.
- **Venue/scope hook**: ICAIF fit is comfortable at the current scope (methods + real-data case
  study). ICLR/NeurIPS main track would need an added theoretical hook — e.g. a provable statement
  on how reconstruction uncertainty in $\hat W_t$ propagates into coverage guarantees for a
  downstream interference estimator (which would also strengthen the Paper 1 → Paper 2 link).
  **Not resolved** — needs a decision before drafting. **Update, 2026-11-05**: ICLR/NeurIPS
  dropped (2026-11-04); the open choice is Physical Review E vs. Journal of Financial Stability.
  The uncertainty-propagation hook is not done (Paper 2 uses one seed's $\hat W_t$ as a fixed
  weight, §1).
- **Posterior collapse on the block-assignment KL term — root cause identified, architecture fix
  plus capped KL ceiling fixes reconstruction but NOT block recovery reliably across seeds**: root
  cause is that $q_\psi(z_i)$ conditioned on $X_{i,1:t}$ alone has zero signal to recover $z$ above
  the trivial majority-class floor, since the generative model draws $z_i$ (Eq. 1) and $X_i$ (Eq.
  2) independently — no amount of loss reweighting fixes this. Fix: widen the encoder's input to
  $[X_{i,t}; r_{i,t}; c_{i,t}]$ (the interbank marginals, which *are* functions of $z$ via
  $\Lambda[z_i,z_j]$), matching §4's stated intent of $q_\psi(W_t \mid X_{1:t}, r_t, c_t)$, plus
  capping $\beta_t$'s asymptote at $\beta_{\max}=0.15 < 1$ instead of annealing to $1$ (§6) —
  $\beta_t=1$ always eventually out-competes the reconstruction signal regardless of whether it's
  informative. **A 5-seed replication (seeds 0–4) shows this does NOT generalize**: block-recovery
  margin over the trivial baseline is $+0.0234, -0.1380, +0.1380, -0.1797, -0.0026$ (mean
  $-0.0318$, std $0.1148$ — std nearly 4× the mean), only 2/5 seeds beating the trivial baseline.
  Two failure modes recur: full collapse to uniform ($\text{KL}\approx0$, seed 4) and
  confident-but-wrong assignment (seed 1) — on average, block recovery is statistically
  indistinguishable from (if anything slightly worse than) the trivial majority-guess baseline.
  Reconstruction accuracy, in contrast, **is** robust across the same 5 seeds: model mean relative
  Frobenius error beats the classical Sinkhorn/RAS baseline in 5/5 seeds (mean 0.4154 vs. 0.5657,
  std 0.0416 vs. 0.0257). **Resolved conclusion for now**: the claim "the model identifies which
  banks are core" is **not currently supported** by this architecture + hyperparameter
  combination — either (a) drop it in favor of the already-robust reconstruction-only claim, or (b)
  keep it as an explicit goal pending further work (longer training, more `BlockPosteriorMLP`
  capacity, lower $\beta_{\max}$, per-seed diagnosis of why seeds 1/4 fail differently from 0/2).
  Not yet decided which path.
- **Posterior collapse on REAL data is structural (recon ≡ 0), not a symptom of the §9 completeness
  gaps.** `train.py`'s `total_recon` only accumulates when `Period.W_true is not None`, and every
  real `Period` sets `W_true=None` — NRB's bulletins report only the aggregate marginals
  $r_{i,t}, c_{i,t}$ (§8a), never bilateral exposures, so $W_{\text{true}}$ does not exist for real
  data under any amount of additional extraction effort. The only two loss terms with gradient
  signal for $q_\psi(z)$ on real data are therefore `kl` and `smooth_weighted`, and §6 already
  established that `kl` (even capped at $\beta_{\max}=0.15$) eventually out-competes a genuine
  reconstruction signal when one exists — with no competing signal at all, $q_\psi(z_i)\to p(z)$
  exactly is the expected outcome, not a bug: confirmed at both N=5 and N=16, `kl`$\to 0$, every
  bank's posterior exactly uniform $(0.5,0.5)$, independent of panel size.
  **Bracketing** (`run_real_experiment.py --kl-beta-max 0.0`): with KL pressure removed entirely,
  loss goes flat while the unweighted `kl` term climbs unboundedly (4.86 → 1470.16 over 500
  epochs) — $q_\psi(z)$ moves away from uniform but toward a degenerate deterministic point, not a
  genuine split: every bank lands at $\ge 0.999$ on the *same* block (one-block collapse, not
  core/periphery recovery), plausibly because a zero-entropy categorical draw removes the
  Gumbel-softmax noise that otherwise costs `smooth_weighted`. **Both endpoints of the only
  hyperparameter touching this term are degenerate**: $\beta_{\max}=0.15\Rightarrow$ uniform for
  every bank; $\beta_{\max}=0\Rightarrow$ deterministic unanimous single-block for every bank —
  strong evidence the failure is structural to the loss design on real data (no term rewards
  *correctly* separating banks, since `total_recon`$\equiv0$), not fixable by retuning this
  hyperparameter. The real-data claim is scoped down to $\hat W_t$ reconstruction/temporal-stability
  only (§12) — a different, independent reason from §9.9's discard count (which explains panel
  size, not why block-identity itself is unreportable; this would hold even with zero missing
  rows).
  **Path forward, if block-identity is still wanted**: give $q_\psi(z)$ a real observable target
  (e.g. a bank-specific `prior_probs` derived from ownership type or a size/systemic-tier proxy,
  §8b) instead of the flat $1/K$ prior — weak supervision, not ground truth, not yet started; note
  this risks conflating genuine block/community structure with covariate-driven homophily
  (ownership type doubling as both the weak-supervision signal and a homophily covariate) —
  @peixoto2022disentangling's homophily/community/triadic-closure disentangling methodology would
  be the right diagnostic if this path is pursued.
  **Decided, 2026-10-28**: scope down and write now — block-identity dropped as a Paper 1 claim;
  the weak-supervision path is deferred to future work.

  **Update, 2026-10-31 — re-run against the current codebase does NOT reproduce the exact collapse
  described above.** Two code facts confirmed by reading `train.py`/`config.py`/
  `run_real_experiment.py` directly, not inferred from training curves:
  1. `_categorical_kl` (`train.py`) means the per-node KL rather than summing it, matching
     `TrainStepConfig.kl_free_bits`'s docstring and `baseline.yaml`'s own comment — confirmed: the
     same `free_bits=0.05` that used to floor at 19.2 on the synthetic panel (N=24,T=16) vs. 106.4
     on the real panel (N=16,T=133) under the old summed version is now panel-shape-invariant.
     This alone loosens, but does not explain the full effect.
  2. **A second, undocumented loss term is active and is the more direct explanation:**
     `TrainStepConfig.lambda_mi` (`config.py`) weights `_mutual_info_regularizer` — an IIC/IMSAT-
     style term that *actively rewards* confident per-bank $q$ (low mean entropy) while penalizing
     every bank collapsing into the same block (low entropy of the mean block distribution). It is
     set to **0.05 in `baseline.yaml`, `synthetic_baseline.yaml`, and `ablation_no_kl_pressure.yaml`
     alike** — i.e. active in every config run this session, including the one meant to reproduce
     the original kl_beta_max=0.0 bracket. This term does not exist anywhere in this doc's Eq. 9 /
     §6 loss formula, and `config.py`'s own comment on it warns: "Caution: with K=2 this implicitly
     biases toward a balanced ~50/50 split" — i.e. it is designed to push $q$ away from uniform
     *regardless of whether that split is correct*, which is a materially different (and weaker)
     kind of evidence than "the model learned block structure." Confident real-data posteriors this
     session (NIC 0.8866, SCBNL 0.9925, RBB 0.8716, etc., final `kl≈0.68`) are therefore better
     explained by `lambda_mi` doing its designed job than by `kl_free_bits` alone — consistent with,
     not an alternative to, the already-stated conclusion that this is not validated block signal.
  3. **Diagnostic gap, found by reading the code, not by running it**: `_forward_pass_components`
     (`run_synthetic_experiment.py`) returns only `recon`, `kl`, and `smooth_weighted` — by its own
     docstring's explicit list — and `components_trend` (`run_real_experiment.py`) only records
     `kl` and `smooth_weighted`. **`lambda_mi`'s contribution is not tracked or printed anywhere**,
     on either real or synthetic runs. This means `baseline.yaml`'s own instruction — "re-run and
     re-check `components_trend` before assuming `kl_beta_max`/`kl_warmup_epochs` are still
     well-tuned" — cannot currently be followed to completion: the retune check needs `mi_reg`
     added to both diagnostics first, or it's checking two of the three terms that actually move
     $q_\psi(z)$ on real data.
  A second, independent synthetic 5-seed run under these same (lambda_mi-active, rescaled-kl_free_bits)
  hyperparameters gives a fresh data point, distinct from the §6 pre-rescale 5-seed replication
  above: reconstruction still robust (5/5 seeds beat baseline, mean rel. error 0.4219±0.0357 vs.
  0.5657±0.0257), block_acc again fails to reliably beat the trivial baseline (3/5 seeds, gap mean
  +0.0646, std 0.0791 — std exceeds mean). Same qualitative conclusion as 2026-10-28 (no
  block-identity claim), now for a more precisely identified reason. **Not yet done**: add `mi_reg`
  to both diagnostics, then retune/re-verify `kl_beta_max`/`kl_warmup_epochs`/`kl_free_bits`/
  `lambda_mi` together against the new scale before citing either 5-seed run as a tuned result.

  **Update, 2026-11-01 — the `lambda_mi` retune above is done; result is to turn it off.** Four
  5-seed synthetic runs (`lambda_mi` ∈ {0.00, 0.05, 0.15, 0.30}, everything else fixed) against the
  rescaled, per-node-meaned `kl_free_bits`:

  | `lambda_mi` | gap mean | gap std | seeds beating trivial baseline |
  |---|---|---|---|
  | 0.00 | 0.0781 | 0.0774 | 3/5 |
  | 0.05 | 0.0688 | 0.0823 | 3/5 |
  | 0.15 | 0.0682 | 0.0790 | 3/5 |
  | 0.30 | 0.0656 | 0.0814 | 3/5 |

  `lambda_mi=0.00` has the highest gap mean of the four, `n_beats_trivial` is tied 3/5 at every
  value, and gap mean is smaller than gap std in all four rows — no nonzero value separates from
  the others or from noise. `baseline.yaml`/`ablation_no_kl_pressure.yaml` now set `lambda_mi=0`.
  This sharpens, rather than reverses, the conclusion directly above: `lambda_mi` was never
  validated to be doing useful block-separation work, and this sweep shows it isn't earning its
  keep either on synthetic data.

  **Update, 2026-11-01 (Step 3, significance check) — confirms the `lambda_mi=0` decision rather
  than just leaving it as unseparated gap means.** `seed_significance.py` (paired t-test +
  Wilcoxon across the 5 shared seeds; at n=5, Wilcoxon's minimum achievable two-sided p is
  1/16=0.0625, so the t-test p-value and mean paired difference are the numbers that carry
  information here) gives, within `lambda_mi=0` alone: `model_mean` beats the RAS/Sinkhorn
  baseline significantly ($p=0.0101$), `block_acc` does not beat the trivial majority-guess
  baseline significantly ($p=0.1136$) — consistent with the already-stated "no block-identity
  claim." Pairwise against each nonzero candidate, `lambda_mi=0` vs.: `0.05`
  ($p=0.3945$ model_mean, $p=0.4047$ block_acc), `0.15` ($p=0.2101$, $p=0.1722$), `0.30`
  ($p=0.3251$ model_mean, $p=0.0441$ block_acc) — no nonzero value beats `lambda_mi=0` on
  anything; the one nominally "significant" difference (`0.30`'s block_acc) runs in
  `lambda_mi=0`'s *favor* (mean paired difference +0.0125, i.e. `lambda_mi=0` is higher), so it
  reinforces rather than complicates the decision. (Three pairwise comparisons at $\alpha=0.05$
  without multiple-comparison correction carries a nontrivial chance of one nominal hit by
  chance alone — noted for completeness, but moot here since the one hit favors the already-
  chosen value, not a competitor.) Closes this section's outstanding significance-check item.

  **Update, 2026-11-01 (real-data re-run, Step 7) — not literally costless to remove.** With
  `lambda_mi=0`, both real-data configs reproduce the exact degenerate endpoints this section
  already described: `ablation_no_kl_pressure.yaml` (`kl_beta_max=0`) saturates at
  `kl=0.691≈ln(2)` with all 16 banks at ≥0.999 posterior on a single block (one-block collapse);
  `baseline.yaml` (`kl_beta_max=0.15`) collapses to the `kl_free_bits=0.05` floor with every bank
  at ~0.45/0.55 (uniform). The 2026-10-31 update's confident, non-uniform posteriors (NIC 0.8866,
  etc.) were specifically a `lambda_mi=0.05` effect, now gone — it was the only thing keeping
  real-data posteriors off both endpoints. Doesn't change Paper 1's claims (block-identity was
  already dropped as a claim on 2026-10-28, for reasons independent of `lambda_mi`), but softens
  "not a loss" above to "not a loss for the claims actually made."

  **A second, more consequential finding from the same two runs**: despite one run's posteriors
  one-block-collapsing and the other's going uniform, the two runs' saved $\hat W_t$ are
  essentially identical — `(a - b).abs()` between `results/baseline/real_run_seed0.pt['W_hat']`
  and `results/ablation_no_kl_pressure/real_run_seed0.pt['W_hat']` gives a relative difference of
  1.5e-5 (max abs diff 0.18) — essentially the same matrix. $z$/block posterior has essentially no
  effect on the actual reconstructed network on real data; the Dykstra projection onto the fixed
  real marginals is doing effectively all the work regardless of what the encoder/decoder produce
  upstream. This is hard evidence of a decoder-identifiability gap: nothing in the current loss
  rewards a particular "economically correct" $\hat W_t$ among all matrices fitting the marginals
  (exactly where feasible, least-violation otherwise, §5),
  so the projection layer's own smoothness/initialization dominates and $z$ is close to inert.
  **Update, 2026-11-01 — root cause confirmed by reading `train_step` (`train.py`), not just
  inferred from the 1.5e-5 diff.** On real data, `period.W_true` is always `None`, so the branch
  that accumulates `total_recon` never executes — `total_recon` isn't approximately zero, it is
  identically zero for the entire run, by construction (there is no bilateral ground truth to
  reconstruct against on real data; that is the premise of the paper, not a bug). `total_kl` and
  `total_mi_reg` are pure functions of $q_\psi(z)$ alone and never touch $(\eta,\lambda)$ — the
  decoder's output — so they backprop into the encoder/posterior MLP but give the decoder's own
  parameters zero gradient. That leaves `lambda_smooth * L_smooth`, computed on the
  *post-projection* $\hat W_t$ trajectory, as the only loss term with any gradient path to the
  decoder's weights on real data — and all it rewards is period-to-period stability of an output
  already fitted to the real marginals by the projection layer (as closely as feasibility allows,
  §5), not any particular use of $z$ or
  $h$. So the empirical finding above isn't a training-convergence accident: **the decoder has no
  training signal on real data that could ever teach it to use $z$ meaningfully**, independent of
  seed, architecture width, or any of the hyperparameters retuned in this section.

  **Limitation, not a fix**: `total_recon≡0` on real data is structural (no bilateral ground
  truth exists to supervise it), so this isn't something the retuning in this section can reach.
  The honest claim for the real-data result is that $\hat W_t$ is produced by the projection layer
  (Algorithm 1) plus a temporal-smoothness prior, constrained to the observed marginals — the
  $z$/block-conditioned decoder's contribution to the *real-data* $\hat W_t$ is not currently
  demonstrated to be anything beyond that. Block/decoder structure is only validated on synthetic
  data, where `total_recon` is nonzero and gives $z$ a real training signal (§6's `block_acc`
  results).

  **Update, 2026-11-01 — confirmed: the insensitivity is specific to real data's
  `total_recon≡0`, not a deeper decoder/architecture defect.** Same diff check, run on a matched
  synthetic pair (`sweep_lambda_mi_0.yaml` vs. a `kl_beta_max=0` ablation of it,
  `synthetic_ablation_no_kl_pressure.yaml`, same seed, same scenario so `W_true` is identical):
  `q_final` differed by only 0.044 mean abs diff between the two runs — a much smaller posterior
  perturbation than the real-data pair's (one-block collapse at ≥0.999 vs. a near-uniform
  ~0.45/0.55 split). Despite that smaller push on $z$, `W_hat_final` differed by a relative
  diff of 0.285 (max abs diff 3.17 against `cap=6.0`, mean abs diff 0.143) — roughly 19,000×
  larger than real data's 1.5e-5 rel diff for a *bigger* posterior swing. So when `total_recon`
  is active, a small change in $z$ moves $\hat W_t$ substantially; when it's structurally absent
  (real data), a large change in $z$ moves $\hat W_t$ almost not at all. This isolates the cause
  to `total_recon≡0` specifically — closes this item. Still open: whether Eq. 3's
  `$h_i^\top h_j$` term — never actually computed by Algorithm 2 — would give the real-data
  decoder a usable signal if reintroduced (independent of the `total_recon` gap; not yet tested).

  **Limitation, 2026-11-01 — `run_real_experiment.py`'s degenerate-period handling rests on a
  false premise about $q_\psi(z)$.** `main()` NaN-masks $\hat W_t$ (not $q$) for degenerate-marginal
  periods (near-zero total interbank activity), on the stated reasoning that "block posterior comes
  from X covariates via the encoder, independent of that period's r/c marginals, so it's still
  meaningful there." That premise is false: `TemporalEncoder`'s GRU step concatenates the observed
  $r_{i,t}, c_{i,t}$ for *that same period* onto $X_{i,t}$ before producing $h_{i,t}$
  (`EncoderDecoderConfig.input_dim` is "covariates only" by design — `config.py`'s own comment
  notes the GRU's actual input width is `input_dim + 2` for exactly this reason). So $q_t$ for a
  degenerate period is produced from that period's own near-zero $r_t,c_t$ — an atypical input the
  encoder never sees on non-degenerate periods — and nothing currently checks whether $q_t$ stays
  sensible there. This $q$ still feeds descriptive real-data artifacts (e.g. `make_plots.py`'s
  time-averaged "top banks by block" figure), so a reader could be placing more trust in those
  degenerate-period block assignments than is actually justified. Same treatment as the rest of
  this section: documented here rather than fixed — block-identity was already dropped as a
  Paper 1 claim (2026-10-28, above), so this doesn't change any claim actually made, but it is a
  second, independent reason (beyond `total_recon≡0`) to keep that scope-down and not lean on $q$
  for anything beyond the already-caveated descriptive figures.

## 10a. Reviewer pre-mortem: skeptical questions and the EDA that answers them

Written from a skeptical flagship-conference reviewer's seat, before the ≤7-page extraction is
drafted (§12) — what would they doubt, and what descriptive evidence (not a new model) would
preempt it. `eda_reviewer_questions.py` computes the empirical half of each item below against the
real panel; items marked "needs a trained run" also require `results/real_run.pt` from
`run_real_experiment.py`.

**Why Sinkhorn is in this paper at all**: it is not the model's own mechanism. The model's
constraint layer is Dykstra's alternating projections (§5, Eq. 8) — a capacitated-Sinkhorn draft
was tried and dropped because its row-scalar clip implements a row-sum cap, not the per-entry cap
$\mathcal{B}$ the regulation actually specifies (`test_sinkhorn.py`'s `binding_cap_case`). Plain
*uncapped* Sinkhorn/IPF survives only as the mandatory classical comparator (§7) — Gandy–Veraart/RAS
maximum-entropy reconstruction is the field's standard baseline for interbank exposure matrices, so
beating it is the expected sanity check, not a novel contribution of this paper.

**Why GRU, not a temporal-GNN or LSTM**: the "big model, small data" argument in §7 — roughly
16–80 institutions, ~77–170 periods (the real panel: 16 banks, 133 months, §9.9) — where EvolveGCN/TGN-style temporal-GNNs would be
overparameterized. GRU-over-LSTM specifically is *asserted*, not ablated, in the current doc — a
reviewer could reasonably ask why GRU rather than an even simpler AR(1)/exponential-smoothing state
update, since GRU's fewer gates (vs. LSTM) is itself just "slightly less overparameterized," not a
principled stopping point.

Ranked by how damaging the objection would be if left unaddressed:

1. **Smoothness-penalty circularity (most dangerous).** The objective penalizes period-to-period
   change ($\lambda_{\text{smooth}}\sum_t\|W_t-W_{t-1}\|_F^2$, §6), but the headline stability claim
   compares against RAS, which has no such term — the model is scored on the axis it was told to
   optimize, the general pathology @vallescatala2018modelselection documents between a model's own
   selection criterion and genuine structural/link recovery. **EDA**: compare lag-1 autocorrelation of raw edge values between $\hat W_t$ and
   $W_{\text{baseline}}$ (needs a trained run) — if the baseline is already highly autocorrelated,
   the stability margin is weaker evidence of learned structure than it looks.
2. **Cap unfairness, same shape of problem.** The model enforces the regulatory cap
   $\mathcal{B}$; RAS is uncapped, so part of "model beats baseline" could just be extra
   information the baseline never got. **EDA**: cap-binding fraction — does $\hat W_t$'s projected
   value actually hit $\kappa K_{i,t}$ per (bank, period) (needs a trained run)? **Answered
   (2026-11-05)**: 7.0% of entries over all months, 1.9% in Paper 2's main window, $\kappa = 0.05$
   (§5).
3. **GNN branding overstates the real-data model.** The encoder's optional GNN refinement needs an
   auxiliary observed graph (§4, §8c), and the only candidate (`Statement of Credit`, sector-wise)
   has 2 years of coverage against a ~133-period panel — on real data the GNN half is likely dead
   weight outside those 2 years. **EDA**: none needed beyond stating this plainly — don't call the
   real-data model "GNN+GRU" without the caveat.
4. **Baseline-strawman.** RAS is the canonical comparator but not the only reasonable one — no
   comparison yet against a trivial persistence baseline ($\hat W_t := W_{t-1}$), which would be a
   nastier opponent than RAS given point 1. **EDA**: add a persistence baseline alongside RAS
   (needs a trained run, cheap — persistence requires no projection).
5. **Discard-sample bias.** N=16/33 commercial banks pass the complete-case filter — is that 51%
   discard systematic (bigger/smaller, state vs. private) rather than missing-at-random? **EDA**:
   extend `eda_survivorship.py`'s size/activity bucketing to full first-through-fourth moments
   (mean, variance, skew, excess kurtosis), since bank-size distributions are typically heavy-tailed.
   **Partly answered (2026-11-05)**: excluded banks are smaller (size medians 64,910 NPR mn
   included vs. 44,510 excluded, p = 0.005), so the discard is not size-neutral; see §9.9's update.
6. **Parameter count vs. effective data size.** Is the GRU+MLP+decoder stack itself still
   overparameterized relative to N×T ≈ 16×133 ≈ 2100 real observations? **EDA**: report total
   trainable parameter count next to N×T; no training run needed, just model construction.
7. **Entropic-regularization sensitivity.** `SinkhornConfig.eps=1.0` already reduces the baseline to
   plain IPF rather than true entropic OT (§7/config.py comment) — worth one sentence in the
   writeup, no new EDA.
8. **SBM-backbone validity is unverified, independent of the already-dropped block-identity
   claim.** §10 drops the *claim* that the model recovers which banks are core/periphery, but Eq.
   1-5's entire generative architecture still assumes the market *is* block/tiered-structured in
   the first place — nothing checks this against the real panel. **EDA**: check whether real
   $(r_{i,t}, c_{i,t})$ exposure concentration looks bimodal/tiered (consistent with core-periphery
   blocks) vs. a smoother size/fitness-driven pattern — descriptive statistics on the real panel
   only, no training run needed.
9. **Time-invariance of $\Lambda,\pi$ across a ~11-year, 133-month panel that plausibly spans
   NRB's BFI merger/consolidation waves.** Eq. 1-3 treat the block prior and block-pair intensity
   as fixed structural constants, not indexed by $t$, while covariates and marginals vary freely.
   **EDA**: check whether any of the N=16 complete-case banks are merger survivors with a
   discontinuous identity partway through the window (a lookup against NRB merger records, no
   training run needed), and whether splitting the panel into pre/post sub-windows changes the
   baseline comparison materially (needs a trained run for the sub-window split).
   **Data-based check (2026-11-05, Paper 2 `paid_up_threats`; dates from the data only, not
   verified merger dates)**: Lumbini's last bulletin month is Jestha 2073, and SBL's log size
   rises 0.144 at Asar 2073, just under the clean-month 99th percentile (0.146); a full absorption
   of Lumbini would imply about +0.34, so the bulletins do not show when or how a merger enters
   SBL's balance sheet. RBB's largest monthly change near Baisakh 2075 is +0.099, below the 99th
   percentile. The hinted dates come from NRB's "List of BFIs" and are approximate.
10. **Undocumented loss term in the released code (low severity).** `train.py`/`config.py` wire
    up `TrainStepConfig.lambda_mi`, an IIC/IMSAT-style mutual-information regularizer on
    $q_\psi(z)$ that does not appear in Eq. 9, and `experiments/configs/` still ships
    `sweep_lambda_mi_{0,015,03}.yaml` sweeping it to nonzero values — a reviewer reading the code
    (not just the paper) could reasonably ask why. Easy answer, already written up in full in §10
    (2026-10-31/2026-11-01 updates): it was found, ablated across 5 seeds × 4 values, shown not to
    beat the `lambda_mi=0` control on either `model_mean` or `block_acc` (§10's significance-check
    update), and set to 0 in every config actually reported (`baseline.yaml`,
    `ablation_no_kl_pressure.yaml`). **EDA**: none needed — just point reviewers at §10 if a
    reproducibility statement references the code release.

## 11. Algorithms (pseudocode)

**Plain-language summary.** Algorithm 1 is the fix-up step from §5, spelled out as code — the one
piece worth reading closely, since it's the paper's actual novel engineering. Algorithms 2 and 3
are just §3's and §4-6's math rewritten as step-by-step procedures — nothing new to learn from
them beyond what's already stated as equations, so their full pseudocode is pushed to Appendix A
to save space.

Three algorithms, in dependency order: the capacitated projection (the hard constraint layer, Eq.
8) is standalone and testable without the rest of the model; the generative forward pass (Eq. 1-5,
8) samples a synthetic $W_t$ for validation; training (Eq. 6-9) wraps the amortized
encoder/decoder of §4 around the same projection layer. Only Algorithm 1 is given in full below —
it's the paper's one genuinely nontrivial engineering contribution (§5), so it needs to be
directly auditable in the main text. Algorithms 2 and 3 each restate equations already stated
above (Eq. 1-5/8 and Eq. 6-9 respectively) as imperative control flow, with nothing new to audit
beyond what §3/§6 already state mathematically, so their full pseudocode is moved to **Appendix
A** to stay within the ≤7-page budget — see the short summaries below. All three are written over
*symbol-level* inputs ($N$, $K$, $d$, $\kappa$, etc.) — nothing below is a fixed dataset size or
constant; see §12 for how these get threaded through as config, not literals.

### Algorithm 1 — `CapacitatedProjection` (Eq. 8, Dykstra's alternating-projections algorithm)

```
Input:  W̃ ∈ R^{N×N}_{≥0}      raw/sampled edge weights
        r, c ∈ R^N_{≥0}        target row/column marginals (observed r_i,t, c_i,t)
        cap ∈ R^N_{≥0}         per-row obligor cap, cap_i = κ · K_i,t
        n_iters, tol           stopping criteria (n_iters counts full row→col→box
                                passes, not single-set applications)
Output: W ∈ R^{N×N}            satisfies row/col marginals and W_ij ≤ cap_i for all j,
                                exactly (up to tol), when the intersection is non-empty

 1:  W ← W̃
 2:  I_row, I_col, I_box ← zeros(N,N), zeros(N,N), zeros(N,N)   # Dykstra correction terms
 3:  for iter = 1 .. n_iters:
 4:      W_prev ← W
 5:      Y ← W + I_row
 6:      W ← Y + (r − row_sums(Y)) / N          # Proj onto {row sums = r}, broadcast per row
 7:      I_row ← Y − W
 8:      Y ← W + I_col
 9:      W ← Y + (c − col_sums(Y)) / N          # Proj onto {col sums = c}, broadcast per col
10:      I_col ← Y − W
11:      Y ← W + I_box
12:      W ← clip(Y, 0, cap)                     # Proj onto {0 ≤ W_ij ≤ cap_i}, per-row cap,
                                                   #   applied to EVERY entry in row i
13:      I_box ← Y − W
14:      if ‖W − W_prev‖ < tol: break
15: return W
```

Note: unlike the superseded Sinkhorn-clip draft (see §5's revision note), this provably converges
to the *exact* Euclidean projection onto the intersection of all three constraint sets whenever
that intersection is non-empty (Boyle & Dykstra, 1986) — not a heuristic. All three per-set
projections (lines 6, 9, 12) are closed-form and differentiable (the clip has the usual
subgradient at the boundary, same as a ReLU), so gradients flow through to $\tilde W_t$ exactly as
Eq. 8 requires for Algorithm 3's step 10 (Appendix A). If the intersection is empty — e.g. $\kappa K_{i,t}$ set
so tight that $N \cdot \kappa K_{i,t} < r_{i,t}$ for some $i$, meaning the cap can't physically
supply the required row total even by maxing out every entry — this converges to a
least-violation compromise instead, which should be treated as an infeasible-cap data/parameter
error, not silently accepted (`test_capacity_projection.py`'s `infeasible_case` checks for this —
the cap itself must still hold even when the row marginal can't be reached). On real data this is
the usual case, not an edge case: 43 of 45 months in Paper 2's main window have no exact
solution, even before the cap is considered (§5, "Feasibility on real data").

### Algorithm 2 — `SampleSyntheticNetwork` (Eq. 1-5, 8; ground truth for validation, §6)

Restates Eq. 1-5's generative forward pass as an imperative sampler: draw each bank's block
assignment $z_i$ (Eq. 1) and covariate-driven fitness $\phi_i$ (Eq. 2), draw each pair's intensity
$\lambda_{ij}$ and zero-inflation logit $\eta_{ij}$ from the block-pair parameters $\Lambda, H$
(Eq. 3-4) — $H$ block-structured the same way $\Lambda$ is, since existence needs the same
block-dependent pattern as magnitude for $q_\psi(z)$ to have any recon-loss signal to recover it
from — sample $\tilde W$, then project through Algorithm 1 (`CapacitatedProjection`, Eq. 8) so the
synthetic ground truth satisfies the same cap the model is trained to satisfy (an infeasible
target would invalidate the whole validation). Full pseudocode moved to **Appendix A** to keep
this section within budget.

### Algorithm 3 — `TrainStep` (Eq. 6-9; amortized inference + one optimizer step)

Restates Eq. 6-9's forward pass as an imperative training step: encode each period's covariates
and observed marginals via the §4 GRU/GNN, sample $z, e, W$ via the reparameterized
Gumbel-softmax/Gumbel-sigmoid/log-normal draws, project through Algorithm 1 (`CapacitatedProjection`,
Eq. 8), accumulate $L_{\text{recon}}$ (synthetic only) and $L_{\text{KL}}$ per period plus
$L_{\text{smooth}}$ across the batch, and take one optimizer step. Full pseudocode moved to
**Appendix A** to keep this section within budget — the one claim worth stating here directly:
gradients flow from $L$ through Algorithm 1's projection to $\tilde W_t$ via standard autodiff,
since every sub-projection in Algorithm 1 (lines 6, 9, 12) is closed-form and differentiable (the
clip has the usual subgradient at the boundary, same as a ReLU).

## 12. Paper structure (≤7 pages)

**Plain-language summary.** How the ≤7 pages break down: intro/related work, the method itself,
synthetic validation (including the misspecification check), a real-Nepal-data case study scoped
down to reconstruction only (no bank-type claim), and a short limitations section that frames the
data friction and dropped claims as honest findings, not apologies.

| Section | Length | Content |
|---|---|---|
| Intro + related work | ~1.5p | position against Aronow-Samii/Leung (network assumed *known*), Elliott-Golub-Jackson (network assumed known, no reconstruction), Gandy-Veraart (reconstruction without covariate-informed priors or regulatory constraints) |
| Method | ~2.5p | §3–6 above |
| Synthetic validation | ~1.5p | SBM-generated ground truth, baseline comparison, model-misspecification robustness check (non-SBM generator, §10) |
| Real-data case study | ~1p | Nepal panel — $\hat W_t$ reconstruction claim only; no block-identity claim (see Discussion row). Report the marginal imbalance and $\hat W_t$'s misfit (§5), cap-binding rate (§5), closed system and survivorship (§9.9); temporal stability reported as consistency with the smoothness prior, not evidence (§6) |
| Discussion/limitations | ~0.5p | frame as evidence for §0's data-friction contribution, not an apology: (1) the complete-case discard policy and resulting N (§9.9) — data integrity preserved by discarding unresolvable bank/period cells rather than imputing them, every discard tied to a named, diagnosed cause, reported as the honest cost of working against real bureaucratic release formats (§0); (2) block-identity recovery is dropped as a claim, not merely under-evidenced (§10) — true regardless of panel completeness, since real NRB panels never report bilateral exposures (`total_recon ≡ 0`), and itself a direct illustration of the regulatory-opacity point: the one field that would resolve it is exactly the field no resource-constrained regulator publishes; (3) validation relies entirely on self-generated synthetic data (SBM) — real bilateral ground truth does not exist for Nepal, or for most EM central banks (§0, @mistrulli2011assessing shows even where it exists reconstruction methods diverge substantially from it) — mitigated, not closed, by the model-misspecification robustness check (§10): a second synthetic set from a non-SBM generative process, confirming the method doesn't just recover its own fitting assumptions. **Update, 2026-10-31**: the $\beta_{\max}=0.15$-uniform / $\beta_{\max}=0$-collapse bracket this row previously cited is superseded — confirmed against `train.py`/`config.py` that confident, non-uniform real-data posteriors are driven by `lambda_mi`'s IIC/IMSAT-style regularizer (active, weight 0.05, in every current config), which rewards confident+block-diverse $q$ regardless of correctness, not by having found correct separation; see §10's 2026-10-31 update for the full code-verified account |

## Appendix A — Algorithm 2 and 3 full pseudocode

(Not counted against the ≤7-page main-text budget. See §11 for the condensed descriptions and the
autodiff-through-Algorithm-1 claim; this is the full step-by-step pseudocode that condensed prose
summarizes.)

### Algorithm 2 — `SampleSyntheticNetwork` (Eq. 1-5, 8; ground truth for validation, §6)

```
Input:  N, K, d                 problem size (num banks, num blocks, covariate dim)
        π ∈ Δ^K, Λ ∈ R^{K×K}_{≥0}, H ∈ R^{K×K}, w ∈ R^d, σ²        generative params (drawn
                                                        from a prior or fixed per synthetic
                                                        scenario); H is the block-pair
                                                        zero-inflation logit matrix (Eq. 4),
                                                        block-structured the same way Λ is
                                                        (Eq. 3) — existence, not just
                                                        magnitude, needs a block-dependent
                                                        pattern for q_psi(z) to have any
                                                        recon-loss signal to recover it from
        X_{1:N} ∈ R^{N×d}        covariates for this period (synthetic or real)
        κ, K_cap ∈ R^N           regulatory cap inputs
Output: W_true ∈ R^{N×N}_{≥0}, r, c ∈ R^N      ground-truth network + its implied marginals

1:  for i = 1 .. N: z_i ~ Categorical(π)                      # Eq. 1
2:  for i = 1 .. N: φ_i ← exp(w^T X_i)                        # Eq. 2
3:  for i, j = 1 .. N (i ≠ j):
4:      λ_ij ← Λ[z_i, z_j] · φ_i · φ_j                        # Eq. 3 (no h_i·h_j term — no
                                                               #   history yet for a one-shot sample)
5:      η_ij ← H[z_i, z_j]; e_ij ~ Bernoulli(σ(η_ij))          # Eq. 4, block-structured same as step 4
6:      W̃_ij ← e_ij · LogNormal_sample(log λ_ij, σ²)
7:  r ← row_sums(W̃); c ← col_sums(W̃)                         # pre-projection marginals
8:  W_true ← CapacitatedProjection(W̃, r, c, κ·K_cap)           # Eq. 8 — ground truth MUST
                                                               #   also satisfy the cap, or the
                                                               #   model is validated against an
                                                               #   infeasible target
9:  return W_true, row_sums(W_true), col_sums(W_true)
```

### Algorithm 3 — `TrainStep` (Eq. 6-9; amortized inference + one optimizer step)

```
Input:  batch of periods {X_{1:N,t}, r_t, c_t, K_t}_{t ∈ batch}, optional W_true_t (synthetic only)
        model params θ (GRU), ψ (block/decoder nets), σ²
        hyperparams λ_smooth, κ
State:  h_{i,t-1} carried across periods (GRU hidden state)
Output: loss value; updated θ, ψ after one gradient step

1:  for t in batch (in temporal order):
2:      for i = 1 .. N: h_{i,t} ← GRU_θ([X_{i,t}; r_{i,t}; c_{i,t}], h_{i,t-1})  # Eq. 6
3:      [optional] h_{i,t} ← GNN_refine(h_{i,t}, A_obs_t)                 # §4, §8c
4:      for i = 1 .. N: q_i ← softmax(MLP_ψ(h_{i,t}))                    # Eq. 7
5:      for i = 1 .. N: z_i ← GumbelSoftmaxSample(q_i)                    # reparameterized
6:      for i, j = 1 .. N (i ≠ j):
7:          (η_ij,t, λ_ij,t) ← Decoder_ψ(h_i,t, h_j,t, z_i, z_j)          # supersedes literal
                                                               #   Eq. 3-4's Λ/w computation
                                                               #   (§4) -- Λ, w never enter here
8:          e_ij,t ← GumbelSigmoidSample(η_ij,t)                          # reparameterized gate
9:          W̃_ij,t ← e_ij,t · LogNormalSample(log λ_ij,t, σ²)            # reparameterized weight
10:     W_t ← CapacitatedProjection(W̃_t, r_t, c_t, κ·K_t)         # Eq. 8, Algorithm 1
11:     if W_true_t is given (synthetic branch):
12:         L_recon,t ← -log p(W_true_t | λ_t, η_t, σ²)                   # Eq. 9, first term
13:     else:
14:         L_recon,t ← 0                                                 # real data: no ground truth
15:     L_KL,t ← KL(q_ψ(z_t) ‖ p(z))
16: L_smooth ← Σ_t ‖W_t - W_{t-1}‖_F²            over consecutive periods in the batch
17: L ← Σ_t (L_recon,t + L_KL,t) + λ_smooth · L_smooth
18: backprop L; update θ, ψ via optimizer step
19: return L
```

## Appendix B — Variable inventory (full detail)

(Not counted against the ≤7-page main-text budget. Subsection labels 8a-8e are preserved from
§8 for cross-reference compatibility — see §8 for the one-paragraph summary.)

### 8a. The two marginals (non-negotiable — the reconstruction target)

The names below (`2.b Interbank Borrowing`, `INTERBANK LENDING (Marginal r_i)`) were this doc's
own shorthand, not literal bulletin text, and the row-marginal name additionally implied a
standalone top-level item that doesn't exist. Confirmed against the real extracted CSV
(`nrb_balance_sheet.csv`) on 2026-10-22 — see §9.4:

| Symbol | Real bulletin text | Parent section | Statement / side |
|---|---|---|---|
| $c_{i,t}$ (column marginal) | `b. Interbank Borrowing` | `2 BORROWINGS` | Balance Sheet, **Liabilities** |
| $r_{i,t}$ (row marginal) | `a. Interbank Lending` | `3 SHARE & OTHER INVESTMENT` | Balance Sheet, **Assets** |

Both are lettered sub-items, not top-level rows — `r_{i,t}$` in particular is a breakdown
component of `3 SHARE & OTHER INVESTMENT`'s own reported total, which matters for the size
formula below (§9.4). These are the actual payoff variables for the whole method — everything else
exists to help infer the matrix whose row/column sums these fix.

### 8b. Node covariates $X_{i,t}$ (block-membership / fitness prior)

| # | Covariate | Built from | Role |
|---|---|---|---|
| 1 | Size | sum of the **10** real top-level asset rows: `1 LIQUID FUNDS`, `2 INVESTMENT IN SECURITIES`, `3 SHARE & OTHER INVESTMENT`, `4 LOANS & ADVANCES`, `5 BILL PURCHASED`, `6 LOANS AGAINST COLLECTED BILLS`, `7 FIXED ASSETS`, `8 OTHER ASSETS`, `9 EXPENSES NOT WRITTEN OFF`, `10 NON BANKING ASSETS` (**not** the `TOTAL ASSETS` label directly — see §9.1). $r_{i,t}$ is **not** added as an 11th term — per §8a/§9.4 it's a sub-item already folded into `3 SHARE & OTHER INVESTMENT`'s reported total, so adding it separately would double-count it | SBM block/fitness prior, scale control |
| 2 | Capital ratio | `1 CAPITAL FUND` (Liabilities) ÷ size | solvency/fitness term; also sets the obligor cap $K_{i,t}$ |
| 3 | Liquidity ratio | `1 LIQUID FUNDS` (Assets) ÷ size | liquidity position |
| 4 | Loan-to-deposit pressure | `4 LOANS & ADVANCES` (Assets) ÷ `3 DEPOSITS` (Liabilities) | funding-pressure signal |
| 5 | NRB-dependence | `a. NRB` ÷ `2 BORROWINGS` (both under Liabilities, `a. NRB` nested under `2 BORROWINGS`; if `2 BORROWINGS` is missing for a row — some early-era filings never print the top-level subtotal — the denominator is reconstructed as the sum of its sub-items: `a. NRB`, `b. Interbank Borrowing`, `c. Foreign Banks and Fin. Ins.`, `d. Other Financial Ins.`, `e. Bonds and Securities` — confirmed real text per §9.4; the `f.`/`g.` sub-items this doc previously assumed existed under `2 BORROWINGS` have **not** been confirmed in the real data yet) | central-bank liquidity dependence — useful for the zero-inflation gate (a bank funded mainly by NRB is less likely to have interbank exposure) |

Optional secondary signal, not in the core 5: ownership type (state/private/foreign), if available
— would add homophily structure to the block prior.

### 8c. Auxiliary observed-graph candidate (optional GNN refinement input, §4)

| Source | Role |
|---|---|
| Statement of Credit, sector-wise breakdown (12 sectors + Others) | shared sector-concentration = a correlated-shock signal independent of the interbank marginals. Candidate as the auxiliary *observed* graph the GNN layer can refine over, since it doesn't require $W_t$ itself. |

### 8d. Explicitly excluded from Paper 1

- `6 Bills Payable`, `5 Other Liabilities` (Liabilities, minor residuals) — not used
- `6/10 Reconciliation A/c` (both sides) — accounting plug, not economically meaningful
- `8 Expenses not Written off`, `9 Non Banking Assets` (Assets) — legacy/minor, not used
- Everything in Profit & Loss — these are Paper 2 outcome variables, not reconstruction inputs
  (see `paper2_causal_spillover.md`)
- `nrb_statement_of_credit.csv` as a panel source — only 2 years of coverage in current data,
  can't support a 14-year panel; not worth cleaning further unless more years turn up uncollected

### 8e. One-to-one equation-symbol ↔ raw-variable map

Every symbol that appears in Eq. (1)–(8) is listed below with exactly what real NRB field (if
any) it is computed from. Symbols with no raw field are latent or learned — they're estimated by
the model, not read off the bulletin.

| Symbol | Appears in | Raw NRB field | Statement / side | Transformation |
|---|---|---|---|---|
| $r_{i,t}$ | Eq. (5), (6), (7), (8) | `a. Interbank Lending` (sub-item of `3 SHARE & OTHER INVESTMENT`) | Balance Sheet, Assets | identity (reported value) |
| $c_{i,t}$ | Eq. (5), (6), (7), (8) | `b. Interbank Borrowing` (sub-item of `2 BORROWINGS`) | Balance Sheet, Liabilities | identity (reported value) |
| $K_{i,t}$ | Eq. (8), $\kappa K_{i,t}$ cap | `1 CAPITAL FUND` | Balance Sheet, Liabilities | identity (reported value) — note: raw value here, *not* the ratio used in $X^{(2)}_{i,t}$ below |
| $X^{(1)}_{i,t}$ | Eq. (2), component of $X_{i,t}$ | sum of the 10 real top-level asset rows (listed in §8b) | Balance Sheet, Assets | sum (not the `TOTAL ASSETS` label — see §9.1; $r_{i,t}$ excluded, see §8b/§9.3) |
| $X^{(2)}_{i,t}$ | Eq. (2) | `1 CAPITAL FUND` | Balance Sheet, Liabilities | ÷ $X^{(1)}_{i,t}$ (capital ratio) |
| $X^{(3)}_{i,t}$ | Eq. (2) | `1 LIQUID FUNDS` | Balance Sheet, Assets | ÷ $X^{(1)}_{i,t}$ (liquidity ratio) |
| $X^{(4)}_{i,t}$ | Eq. (2) | `4 LOANS & ADVANCES` ÷ `3 DEPOSITS` | Balance Sheet, Assets ÷ Liabilities | ratio across statement sides |
| $X^{(5)}_{i,t}$ | Eq. (2) | `a. NRB` ÷ `2 BORROWINGS` | Balance Sheet, Liabilities (both) | ratio within Liabilities; denominator falls back to the sum of `a.`–`e.` sub-items (§8b) when `2 BORROWINGS` is unreported |
| $w$ | Eq. (2) | — | — | **generative/scenario param only** (fitness weight vector, $\dim = 5$, no raw field) — Algorithm 2's synthetic sampler, caller-supplied; *not* learned by Algorithm 3's trained model, which uses $\text{Decoder}_\psi$ in place of Eq. 2-3 (§4, §6) |
| $z_i$ | Eq. (1), (3), (7) | — | — | **latent** (inferred via $q_\psi$, Eq. 7); optionally informed by ownership type if used (§8b) |
| $\Lambda$ | Eq. (3) | — | — | **generative/scenario param only** ($K\times K$ block-pair intensity matrix) — same caveat as $w$ above: Algorithm 2 input, not learned by Algorithm 3 |
| $h_{i,t}$ | Eq. (3), (6), (7) | $X^{(1\text{-}5)}_{i,t}$ (all 5 covariates) and the observed marginals $r_{i,t}, c_{i,t}$, concatenated as the GRU input vector | — | **learned embedding**, GRU output — not read directly off any single field |
| $\theta$ (GRU weights) | Eq. (6) | — | — | **learned** |
| $\psi$ (posterior-net weights) | Eq. (7) | — | — | **learned** |
| $\eta_{ij,t}$ | Eq. (4) | — | — | **learned** (zero-inflation logit, decoder output) |
| $\sigma^2$ | Eq. (4) | — | — | **learned** (log-normal dispersion — global or per-block, §10) |
| $\kappa$ | Eq. (8) | — | — | **exogenous regulatory constant** (single-obligor exposure limit; not reported in the bulletin itself — a known regulatory parameter) |
| $A^{\text{obs}}$ (auxiliary GNN graph, §4) | Eq. (6) refinement, optional | sector-wise credit, 12 sectors | Statement of Credit | co-exposure adjacency built from shared-sector lending shares |

**Reading guide**: $X_{i,t} = \big(X^{(1)}_{i,t}, ..., X^{(5)}_{i,t}\big) \in \mathbb{R}^5$ is the
concrete instantiation of the generic $X_{i,t}\in\mathbb{R}^d$ in §2 — i.e. $d=5$ for this model,
and the ordering above (size, capital ratio, liquidity ratio, LDR, NRB-dependence) is the fixed
component order used everywhere $X_{i,t}$ appears in Eq. (2) and (6).

