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
