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

    input_dim: int           # d, covariate dimension. TemporalEncoder's actual GRU
                              # input width is input_dim + 2 internally (it also
                              # concatenates the observed r_t, c_t marginals, Eq. 6)
                              # -- this field stays "covariates only" so callers
                              # don't need to account for the +2 themselves.
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


@dataclass
class TrainStepConfig:
    """Hyperparameters for train_step (Eq. 6-9, Sec 11 Algorithm 3).

    lambda_smooth is deliberately NOT given a default -- unlike the
    numerical-stabilizer fields below, it's a real modeling choice that varies
    per experiment (Sec 7's baseline comparisons, Sec 10's open items), so the
    caller must set it explicitly rather than silently inheriting an arbitrary
    default.
    """

    lambda_smooth: float
    recon_zero_tol: float = 1e-6  # |W_true_ij| below this counts as a structural zero
    delta: float = 1e-12          # log-stabilizer
    kl_beta: float = 1.0          # Eq. 9's beta_t: KL weight, 1.0 = original unweighted
                                   # term. Mutable per-epoch by the caller to implement
                                   # annealing (Sec 10): ramping beta 0 -> 1 early in
                                   # training lets q_psi(z) latch onto block signal
                                   # before the KL penalty pulls it to the uniform prior
                                   # -- defaults to the pre-annealing behavior so existing
                                   # callers that never set it are unaffected.
    kl_free_bits: float = 0.0     # per-node KL floor (Kingma et al. "free bits"): each
                                   # node's KL(q_i || prior) is clamped from below at this
                                   # value, and the floored term is then MEANED (not
                                   # summed) over nodes and periods -- see train.py's
                                   # _categorical_kl -- so this value means the same thing
                                   # regardless of panel size N or batch length T, and q
                                   # keeps that much information about block membership
                                   # without kl_beta's gradient fighting to erase it. 0.0 =
                                   # no floor, reproduces the original unfloored behavior
                                   # -- real data has zero competing reconstruction
                                   # gradient (W_true is always None there), so at any
                                   # kl_beta > 0 an unfloored KL term drives q all the way
                                   # to the uniform prior (Sec 10).
    lambda_mi: float = 0.0        # weight on _mutual_info_regularizer (IIC/IMSAT-style):
                                   # rewards confident per-bank q (low mean entropy) while
                                   # penalizing every bank collapsing into the SAME block
                                   # (low entropy of the mean block distribution). Unlike
                                   # kl_free_bits, which only stops q from being pushed all
                                   # the way to uniform, this actively rewards sharp,
                                   # block-diverse assignments. 0.0 = no-op, reproduces
                                   # existing behavior. Caution: with K=2 this implicitly
                                   # biases toward a balanced ~50/50 split (Sec 10).
