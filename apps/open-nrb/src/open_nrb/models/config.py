#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from dataclasses import dataclass


@dataclass
class SinkhornConfig:
    """Hyperparameters for the uncapped Sinkhorn/IPF projection (Sec 5/7) -- the
    classical Gandy-Veraart/RAS baseline this paper compares against, NOT the model's
    own constraint layer (see CapacityProjectionConfig below for that). This config has
    no cap-related field because sinkhorn_projection takes no cap argument at all.
    """

    # eps=1.0 reduces the kernel to plain Sinkhorn-Knopp/IPF matrix balancing on
    # W_tilde itself (the correct op for "nearest-in-KL matrix satisfying marginals",
    # and what makes this comparable to the Gandy-Veraart/RAS baseline in Sec 7)
    # -- eps != 1 is an optional sharpening temperature, not literal OT cost regularization.
    eps: float = 1.0
    n_iters: int = 100
    tol: float = 1e-6
    delta: float = 1e-12


@dataclass
class CapacityProjectionConfig:
    """Hyperparameters for capacitated_projection (Eq. 8, Sec 5/11 Algorithm 1) -- the
    model's actual constraint layer: Dykstra's alternating-projections algorithm onto
    row marginals, column marginals, and the per-row obligor cap, simultaneously and
    exactly.

    kappa (the regulatory single-obligor cap multiplier) deliberately does NOT live
    here -- it's a modeling/business constant the caller uses to build the `cap`
    tensor (`cap = kappa * K`) before calling the projection. The projection itself
    only ever sees the resulting `cap` tensor, never kappa, so it stays reusable
    regardless of what cap policy a future caller wants.
    """

    n_iters: int = 100  # full row->col->box passes, not single-set applications
    tol: float = 1e-6




@dataclass
class SyntheticConfig:
    """Hyperparameters for sample_synthetic_network (Eq. 1-5, 8, Sec 11 Algorithm 2).

    Only holds the one numerical-stabilizer constant this algorithm needs; everything
    else (pi, Lambda, w, sigma2, eta, cap) is a generative parameter supplied by the
    caller per synthetic scenario, not a fixed hyperparameter of the algorithm itself.
    """

    delta: float = 1e-12  # log-stabilizer, avoids log(0) when a block intensity is 0



@dataclass
class EncoderDecoderConfig:
    """Layer sizes for the amortized inference network (Sec 4, Eq. 6-7, Algorithm 3).

    Unlike SinkhornConfig/CapacityProjectionConfig/SyntheticConfig, these aren't
    numerical tolerances -- they're nn.Module weight-matrix shapes, which have to be
    fixed at construction time (can't be inferred from a runtime tensor shape the way
    N is elsewhere in this package). input_dim/num_blocks should match the X and pi
    shapes used elsewhere in the same experiment.
    """

    input_dim: int           # d, covariate dimension (GRU input size)
    hidden_dim: int          # GRU hidden size, h_{i,t}'s dimension; also the block-
                              # posterior MLP's hidden-layer width
    num_blocks: int          # K, number of SBM blocks
    decoder_hidden_dim: int  # pairwise decoder MLP's hidden-layer width

@dataclass
class ReparamConfig:
    """Temperatures for the reparameterized/relaxed samplers used in Algorithm 3
    (Sec 4, Eq. 3-4, 7, steps 5/8/9) -- Gumbel-softmax for z (Jang et al., 2017;
    Maddison et al., 2017), Gumbel-sigmoid (binary concrete) for the existence gate
    e_ij. Lower tau = closer to the true discrete distribution but higher-variance
    gradients; higher tau = smoother gradients but a more biased relaxation.
    """

    gumbel_softmax_tau: float = 0.5
    gumbel_sigmoid_tau: float = 0.5
    delta: float = 1e-20  # stabilizer inside the various log(...) terms

