#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

import torch

try:
    from .capacity_projection import capacitated_projection
    from .config import CapacityProjectionConfig, SyntheticConfig
except ImportError:
    from capacity_projection import capacitated_projection
    from config import CapacityProjectionConfig, SyntheticConfig


def sample_synthetic_network(
    X: torch.Tensor,
    pi: torch.Tensor,
    Lambda: torch.Tensor,
    w: torch.Tensor,
    sigma2: torch.Tensor,
    eta_block: torch.Tensor,
    cap: torch.Tensor,
    projection_config: CapacityProjectionConfig,
    config: SyntheticConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 2 (paper1_network_reconstruction.md Sec 11, Eq. 1-5, 8).

    Samples a ground-truth synthetic network W_true for validating the model against a
    known target (Sec 6) -- the only setting where the reconstruction loss has a real
    supervised target, since real NRB data never observes W_t itself.

    N (num institutions) and d (covariate dim) come from X's shape; K (num SBM blocks)
    comes from pi's shape -- nothing here is a literal problem size.

    Steps (mirrors the pseudocode exactly):
        1. z_i ~ Categorical(pi)                                    Eq. 1
        2. phi_i = exp(w^T X_i)                                     Eq. 2
        3. lambda_ij = Lambda[z_i, z_j] * phi_i * phi_j, i != j      Eq. 3 (no h_i.h_j
           term -- no history yet for a one-shot sample)
        4. eta_ij = eta_block[z_i, z_j]                              Eq. 4 existence
           e_ij ~ Bernoulli(sigmoid(eta_ij))                         logit, block-
                                                                      structured same
                                                                      as step 3's lambda
        5. W_tilde_ij = e_ij * LogNormal_sample(log(lambda_ij), sigma2)
        6. r, c = row/col sums of W_tilde (pre-projection marginals)
        7. W_true = CapacitatedProjection(W_tilde, r, c, cap)        Eq. 8, Algorithm 1

    Args:
        X: Node covariates (N, d).
        pi: Block membership probabilities (K,), sums to 1.
        Lambda: Block-pair intensity matrix (K, K), >= 0.
        w: Fitness weight vector (d,).
        sigma2: Log-normal dispersion (scalar tensor), > 0.
        eta_block: Block-pair zero-inflation logit matrix (K, K) -- Pr(W_ij != 0) =
            sigmoid(eta_block[z_i, z_j]), mirroring how Lambda sets the block-pair edge
            intensity (Eq. 3). Block-structured rather than a free (N, N) tensor so that
            existence, not just magnitude, carries a block-dependent pattern the
            amortized posterior q_psi(z) actually has a signal to recover -- an (N, N)
            eta independent of z (an earlier version of this scenario-building code) left
            the existence channel pure noise, giving training no incentive to learn z
            through that path at all.
        cap: Per-row regulatory single-obligor cap (N,) = kappa * K_cap. Must be set
            generously enough relative to the sampled row sums (N * cap_i >= r_i) or
            step 7's projection has no feasible solution for that row -- the ground
            truth itself would then be infeasible, which Sec 11's note on this
            algorithm flags explicitly as a correctness requirement, not a nicety.
        projection_config: CapacityProjectionConfig, passed through to Algorithm 1.
        config: SyntheticConfig (delta, the log-stabilizer for step 3/5).
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        (W_true, r_achieved, c_achieved, z) -- the projected ground-truth network, its
        marginals as actually achieved by W_true (not the pre-projection r, c from
        step 6 -- compare the two to check feasibility), and the latent block
        assignment z (N,) -- returned beyond the pseudocode's stated output because
        synthetic validation needs it later to check whether the amortized posterior
        q_psi(z) (Eq. 7) recovers block structure correctly.
    """
    n = X.shape[0]

    # 1. Block membership (Eq. 1)
    z = torch.multinomial(pi, num_samples=n, replacement=True, generator=generator)

    # 2. Fitness/degree correction from covariates (Eq. 2)
    phi = torch.exp(X @ w)

    # 3. Latent edge intensity (Eq. 3); i == j handled by the diagonal mask below
    Lambda_zz = Lambda[z.unsqueeze(1), z.unsqueeze(0)]
    lam = Lambda_zz * phi.unsqueeze(1) * phi.unsqueeze(0)

    # 4. Zero-inflated existence gate (Eq. 4), block-structured same as step 3's lambda
    eta_zz = eta_block[z.unsqueeze(1), z.unsqueeze(0)]
    e = torch.bernoulli(torch.sigmoid(eta_zz), generator=generator)

    # 5. Log-normal weight, gated by existence. torch.distributions.LogNormal.sample()
    # doesn't take a generator, so sample the underlying Normal directly (same
    # reparameterization LogNormal uses internally: X = exp(Normal(loc, scale))) --
    # this keeps sampling reproducible under a caller-supplied generator.
    loc = torch.log(lam + config.delta)
    scale = torch.sqrt(sigma2)
    W_tilde = e * torch.exp(torch.normal(loc, scale, generator=generator))

    # No self-exposure: bank i cannot be its own counterparty (pseudocode's i != j)
    mask = 1.0 - torch.eye(n, dtype=W_tilde.dtype, device=W_tilde.device)
    W_tilde = W_tilde * mask

    # 6. Pre-projection marginals
    r = W_tilde.sum(dim=1)
    c = W_tilde.sum(dim=0)

    # 7. Project onto the feasible set -- ground truth must satisfy the cap too
    W_true = capacitated_projection(W_tilde, r, c, cap, projection_config)

    return W_true, W_true.sum(dim=1), W_true.sum(dim=0), z
