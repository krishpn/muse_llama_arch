#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
# ]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import EncoderDecoderConfig
except ImportError:
    from config import EncoderDecoderConfig


class TemporalEncoder(nn.Module):
    """GRU temporal encoder, Eq. 6: h_{i,t} = GRU_theta(X_{i,t}, h_{i,t-1}).

    One GRUCell shared across all N banks -- N is just the batch dimension at call
    time (inferred from the input's shape), not fixed at construction. Only
    input_dim (d) and hidden_dim come from config, since those set the GRU's weight
    matrix shapes and have to be fixed when the module is built.
    """

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.cell = nn.GRUCell(config.input_dim, config.hidden_dim)

    def forward(self, X_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """X_t: (N, input_dim) covariates this period. h_prev: (N, hidden_dim)."""
        return self.cell(X_t, h_prev)

    def init_hidden(self, n: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(n, self.cell.hidden_size, device=device, dtype=dtype)


class BlockPosteriorMLP(nn.Module):
    """Amortized block-assignment posterior, Eq. 7:
    q_psi(z_i | X_{i,1:t}) = Categorical(softmax(MLP_psi(h_{i,t}))).

    Returns logits, not probabilities -- callers apply softmax (or Gumbel-softmax
    sampling, which wants raw logits) themselves.
    """

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_blocks),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (N, hidden_dim) -> logits (N, num_blocks)."""
        return self.net(h)


class PairwiseDecoder(nn.Module):
    """Pairwise decoder, Eq. 3-4: produces (eta_ij, lambda_ij) from
    (h_i, h_j, z_i, z_j) for every ordered pair (i, j).

    Vectorized over the full (N, N) grid rather than looping over index pairs --
    mathematically the same computation the pseudocode's "for i, j = 1..N" loop
    specifies, just without a Python-level double loop.
    """

    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        feat_dim = 2 * config.hidden_dim + 2 * config.num_blocks
        self.net = nn.Sequential(
            nn.Linear(feat_dim, config.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.decoder_hidden_dim, 2),  # [eta_ij, raw_lambda_ij]
        )

    def forward(self, h: torch.Tensor, z_soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (N, hidden_dim) node embeddings.
        z_soft: (N, num_blocks) a distribution over blocks per node -- soft
            (Gumbel-softmax relaxation) so gradients flow through z for Algorithm 3's
            backprop, or hard one-hot, which is just a special case of the same shape.

        Returns:
            (eta, lambda_): both (N, N). eta is the raw zero-inflation logit
            (Eq. 4); lambda_ is non-negative (Eq. 3) via softplus on the net's raw
            output. Diagonal entries (i == j) are NOT masked here -- callers apply
            the i != j mask, consistent with how synthetic.py handles it.
        """
        n = h.shape[0]
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        z_i = z_soft.unsqueeze(1).expand(n, n, -1)
        z_j = z_soft.unsqueeze(0).expand(n, n, -1)
        features = torch.cat([h_i, h_j, z_i, z_j], dim=-1)

        out = self.net(features)  # (N, N, 2)
        eta = out[..., 0]
        lambda_ = F.softplus(out[..., 1])
        return eta, lambda_
