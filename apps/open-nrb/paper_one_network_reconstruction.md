# Paper 1 — Bayesian Network Reconstruction of Interbank Exposures (Nepal)

**Status:** design settled, not yet implemented. **Target venue:** ICLR / NeurIPS / ICAIF, ≤7 pages.

**One-line description:** Reconstruct the latent bilateral interbank exposure matrix $W_t$ for
Nepali BFIs from NRB bulletin data, using an SBM-structured, covariate-informed generative model
with a differentiable regulatory-constraint projection layer, amortized by a GNN/GRU inference
network.

This supersedes the earlier version of `feature_model_mapping.md`, which had drifted from the
actual design (it invented a generic "Bayesian-net structural-constraint" layer built from
accounting identities — that was never part of the real design and is corrected below).


## 0. Problem framing and contribution to policy

**Problem framing (what "the network" means here).** We do not pre-specify the edges of the
interbank network. Instead, following the network reconstruction literature (e.g., Squartini et
al., 2018), we treat the bilateral exposure structure as latent and infer it from publicly
observable aggregates — balance-sheet marginals and bank-level covariates — within a constrained
statistical framework. Concretely: a node is a bank; a directed edge $i \to j$ is $W_{ij,t}$, the
exposure of bank $i$ to bank $j$; an edge "exists" when $W_{ij,t}>0$. Neither the edge set nor the
weights are observed — only the row/column marginals and covariates are. This is a problem of
statistical inference under partial observability, not analysis over a topology that is already
known or self-reported. (Revisit/tighten this paragraph later — current form is a first pass.)

**Contribution to policy.** No central bank in a resource-constrained setting is going to mandate
new bilateral exposure reporting purely for a research exercise — that is a costly, politically
slow regulatory change. The contribution here is that this doesn't require that change: a usable,
continuously-updatable estimate of systemic interconnectedness is produced from data NRB (or any
similarly-resourced central bank) already publishes every month, with zero new reporting burden.
Put to a central banker directly: today there is no way to ask "who is exposed to whom" or "which
banks are structurally systemic" — not because the question doesn't matter, but because answering
it was assumed to require data that isn't collected. This method shows it doesn't. It converts a
previously unanswerable supervisory question into an answerable one, cheaply, and produces a
reusable pipeline that other analyses (stress tests, contagion simulations, Paper 2's spillover
estimation) can plug into. This also generalizes beyond Nepal: most emerging-market central banks
publish the same kind of aggregate call-report data and face the identical constraint, so the
method is a template for resource-constrained regulators broadly, not a Nepal-specific workaround.

## 1. Why this is its own paper

Separated from the causal/policy question (Paper 2) because the two are methodologically
unrelated: this paper's contribution is a reconstruction method with no causal-identification
content; Paper 2's contribution is a causal-identification design that *consumes* this paper's
output as a fixed input. Bundling them would force a methods paper to carry a policy
narrative it doesn't need, and vice versa.

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

The encoder runs on covariates/history only — it cannot message-pass over $W_t$ itself, since
that would be circular (the whole point is that $W_t$ is unobserved).

- Temporal encoder: $h_{i,t} = \text{GRU}_\theta(X_{i,t}, h_{i,t-1}) \qquad \text{(Eq. 6)}$, optionally refined by a GNN
  layer over an auxiliary *observed* graph if one exists (e.g. common-sector exposure — see §8).
- Amortized block-assignment posterior: $q_\psi(z_i \mid X_{i,1:t}) = \text{Categorical}(\text{softmax}(\text{MLP}_\psi(h_{i,t}))) \qquad \text{(Eq. 7)}$
- Pairwise decoder produces $(\eta_{ij,t}, \lambda_{ij,t})$ from $(h_{i,t}, h_{j,t}, z_i, z_j)$ —
  full amortized variational posterior $q_\psi(W_t \mid X_{1:t}, r_t, c_t)$
- Reparameterized sampling: log-normal reparam trick for the continuous weight, Gumbel-softmax
  for the discrete existence gate.

## 5. Constraint-satisfaction layer (the technical core of the paper)

Differentiable **capacitated Sinkhorn / entropic-OT projection**:

$$W_t = \text{Sinkhorn}_{\text{cap}}(\tilde W_t;\; r_t, c_t, \kappa K_t) \qquad \text{(Eq. 8)}$$

Enforces, exactly and differentiably:
1. row/column marginals match the observed $r_{i,t}, c_{i,t}$
2. the regulatory single-obligor cap $W_{ij,t} \le \kappa K_{i,t}$

References: Cuturi (2013) entropic OT; Mena et al. (2018) Gumbel-Sinkhorn. Flagged explicitly as
"the one genuinely nontrivial implementation piece" in the design discussion — this is the part
most likely to need real engineering effort, not the GNN/GRU scaffolding around it.

**Correction vs. the earlier `feature_model_mapping.md`:** there is no "Bayesian-net
structural-constraint edge" tying `TOTAL Liabilities = TOTAL ASSETS` or `Net Profit` into the
graph. That was never part of this design. The accounting identities in the raw NRB statements
are a **data-quality cross-check** on the extractor (do the reported totals reconcile?), not a
component of the generative model or the constraint layer.

## 6. Training objective

$$\mathcal{L} = -\mathbb{E}_q[\log p(W_t^{\text{true}})]_{\text{synthetic only}} + \text{KL}(q_\psi(z)\,\|\,p(z)) + \lambda_{\text{smooth}}\sum_t \|W_t - W_{t-1}\|_F^2 + \lambda_{\text{reg}}\|\Lambda, w\|^2$$

- **Synthetic data** (known ground-truth $W_t$ from an SBM generator): the first term is a real
  supervised reconstruction loss — this is the main validation evidence.
- **Real data** (no $W_t$ is ever observed): the first term drops out entirely. What's left is a
  regularized/MAP-style constrained fit, *not* likelihood-based estimation. Evaluation on real
  data has to rely on indirect evidence: temporal stability of the recovered $W_t$ sequence, and
  comparison against a classical baseline (§8).

## 7. Design rationale

- **SBM over preferential-attachment/fitness models** as the sole structural backbone: core-
  periphery tiering in interbank markets is literally a latent-block story, so SBM is the natural
  fit. PA/fitness models (Caldarelli et al.) are cited in related work for contrast but not built.
- **GRU, not a temporal-GNN** (EvolveGCN, TGN, etc.) for the temporal mechanism: deliberately the
  simpler choice, given the "big model, small data" risk — roughly 50–80 institutions, at most
  ~170 periods. A fancier temporal-GNN would be over-parameterized for this panel size.
- **Mandatory baseline**: head-to-head comparison against Gandy–Veraart / RAS maximum-entropy
  network reconstruction, on both the synthetic SBM data (known ground truth) and the real panel.
  Explicit instruction from the design discussion: report honestly if the deep model doesn't beat
  the classical baseline — don't bury a negative result.

## 8. Variable inventory (what actually feeds the model)

Net statement from the design discussion: **the reconstruction model needs ~5 covariates per
node per period, plus the two interbank marginals. Everything else in the NRB bulletins is either
a Paper 2 outcome variable or not usable given current data coverage.**

### 8a. The two marginals (non-negotiable — the reconstruction target)

| NRB field | Statement / side | Symbol |
|---|---|---|
| `2.b Interbank Borrowing` | Balance Sheet, **Liabilities** | $c_{i,t}$ (column marginal) |
| `3.a Interbank Lending` | Balance Sheet, **Assets** | $r_{i,t}$ (row marginal) |

These are the actual payoff variables for the whole method — everything else exists to help infer
the matrix whose row/column sums these fix.

### 8b. Node covariates $X_{i,t}$ (block-membership / fitness prior)

| # | Covariate | Built from | Role |
|---|---|---|---|
| 1 | Size | sum of top-level asset categories (**not** the `TOTAL ASSETS` label directly — see data caveat below) | SBM block/fitness prior, scale control |
| 2 | Capital ratio | `1 CAPITAL FUND` (Liabilities) ÷ size | solvency/fitness term; also sets the obligor cap $K_{i,t}$ |
| 3 | Liquidity ratio | `1 LIQUID FUNDS` (Assets) ÷ size | liquidity position |
| 4 | Loan-to-deposit pressure | `4 LOANS & ADVANCES` (Assets) ÷ `3 DEPOSITS` (Liabilities) | funding-pressure signal |
| 5 | NRB-dependence | `2.a NRB` (Liabilities, under BORROWINGS) ÷ `2 BORROWINGS` (Liabilities) | central-bank liquidity dependence — useful for the zero-inflation gate (a bank funded mainly by NRB is less likely to have interbank exposure) |

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
| $r_{i,t}$ | Eq. (5), (8) | `3.a Interbank Lending` | Balance Sheet, Assets | identity (reported value) |
| $c_{i,t}$ | Eq. (5), (8) | `2.b Interbank Borrowing` | Balance Sheet, Liabilities | identity (reported value) |
| $K_{i,t}$ | Eq. (8), $\kappa K_{i,t}$ cap | `1 CAPITAL FUND` | Balance Sheet, Liabilities | identity (reported value) — note: raw value here, *not* the ratio used in $X^{(2)}_{i,t}$ below |
| $X^{(1)}_{i,t}$ | Eq. (2), component of $X_{i,t}$ | sum of top-level asset categories | Balance Sheet, Assets | sum (not the `TOTAL ASSETS` label — see §9.1) |
| $X^{(2)}_{i,t}$ | Eq. (2) | `1 CAPITAL FUND` | Balance Sheet, Liabilities | ÷ $X^{(1)}_{i,t}$ (capital ratio) |
| $X^{(3)}_{i,t}$ | Eq. (2) | `1 LIQUID FUNDS` | Balance Sheet, Assets | ÷ $X^{(1)}_{i,t}$ (liquidity ratio) |
| $X^{(4)}_{i,t}$ | Eq. (2) | `4 LOANS & ADVANCES` ÷ `3 DEPOSITS` | Balance Sheet, Assets ÷ Liabilities | ratio across statement sides |
| $X^{(5)}_{i,t}$ | Eq. (2) | `2.a NRB` ÷ `2 BORROWINGS` | Balance Sheet, Liabilities (both) | ratio within Liabilities |
| $w$ | Eq. (2) | — | — | **learned** (fitness weight vector, $\dim = 5$, no raw field) |
| $z_i$ | Eq. (1), (3), (7) | — | — | **latent** (inferred via $q_\psi$, Eq. 7); optionally informed by ownership type if used (§8b) |
| $\Lambda$ | Eq. (3) | — | — | **learned** ($K\times K$ block-pair intensity matrix) |
| $h_{i,t}$ | Eq. (3), (6), (7) | $X^{(1\text{-}5)}_{i,t}$ (all 5 covariates, fed in as the GRU input vector) | — | **learned embedding**, GRU output — not read directly off any single field |
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

## 9. Known data-pipeline caveats (apply before this goes into the model)

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

## 10. Open items (not yet decided)

- **$K$ (number of SBM blocks)**: start with $K=2$ (core/periphery), treat larger $K$ as a
  robustness check, not the headline specification.
- **$\sigma^2$ (log-normal dispersion)**: learned, either globally or per-block — not yet decided
  which.
- **Venue/scope hook**: ICAIF fit is comfortable at the current scope (methods + real-data case
  study). ICLR/NeurIPS main track would need an added theoretical hook — e.g. a provable statement
  on how reconstruction uncertainty in $\hat W_t$ propagates into coverage guarantees for a
  downstream interference estimator (which would also strengthen the Paper 1 → Paper 2 link).
  **Not resolved** — needs a decision before drafting.

## 11. Paper structure (≤7 pages)

| Section | Length | Content |
|---|---|---|
| Intro + related work | ~1.5p | position against Aronow-Samii/Leung (network assumed *known*), Elliott-Golub-Jackson (network assumed known, no reconstruction), Gandy-Veraart (reconstruction without covariate-informed priors or regulatory constraints) |
| Method | ~2.5p | §3–6 above |
| Synthetic validation | ~1.5p | SBM-generated ground truth, baseline comparison |
| Real-data case study | ~1p | Nepal panel, temporal-stability evidence |
| Discussion/limitations | ~0.5p | honest treatment of the no-ground-truth-on-real-data limitation |
