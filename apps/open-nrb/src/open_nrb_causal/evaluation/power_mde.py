#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
#     "scipy",
# ]
# ///
"""Report the minimum detectable effect of every pooled DiD coefficient.

Sec 5b item 1 ("the floor doesn't bind, so a null is uninformative"), so the paper can say
which effect sizes the design rules out rather than just "not significant".

MDE at two-sided alpha=0.05 and 80% power, with a t(G-1) reference (G = bank clusters, few-
cluster convention):
    MDE = (t_{0.975, G-1} + t_{0.80, G-1}) * SE_CR1
Same sample, design and weights (cfg.weighting) as run_causal_experiment (shared build_inputs); refits with
n_boot=0 since only the CR1 SE is needed, so it runs in seconds. CR1 SEs are deterministic, so
they equal the SEs in the saved results JSONs.

Standardised MDE = MDE * SD(regressor) / within-bank SD(y): the change in y, in within-bank SDs,
from a 1-SD higher regressor that the design could detect 80% of the time.
  SD(dose)     = cross-bank SD of dose_pre (the frozen dose varies only across banks)
  SD(exposure) = SD of exposure x post over post cells
  within-bank SD(y) = SD of y after removing each bank's mean, over the DiD estimation cells
                      (finite y, positive weight) -- the variation the bank FE leave

Residual-SD version (columns "resid SD(regressor)", "std. MDE (resid)"): the same MDE scaled by
the SD of the regressor left after partialling out bank FE, period FE, the controls and the other
main regressor (FWL), weighted by the design's weights, over the cells the fit uses (finite y and
regressors, positive weight). This is the variation that identifies the coefficient, so it is the
SD for "a 1-SD change the design can use"; the raw-SD version above also counts variation the FE
and controls absorb.

A second table splits that residual SD into steps on the same cells: weighted SD with nothing
removed, after bank + period FE, after also the other main regressor, then (each on top of FE +
the other main regressor) the exposure-level control alone and the Z_pre x post controls alone,
and finally all controls. The last column (resid SD / no-FE SD) is the share of the regressor's
spread left to identify it.

When the config sets paid_up_capital_requirement, the same rows are repeated for the paid-up dose
(reviewer_checks.paid_up_dose_pre) with the exposure and HT weights rebuilt from it, as in
reviewer_checks paid_up_dose's Eq. A' table; network column "W_hat (paid-up dose)" etc.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.power_mde
  uv run python -m open_nrb_causal.evaluation.power_mde --config src/open_nrb_causal/experiments/configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from scipy.stats import t as student_t

try:
    from ..models.estimator import _within_transform
    from ..models.experiment_config import CONFIG_DIR, load_causal_experiment_config
    from ..models.mandate_design import apply_weighting, did_design
    from ..models.robustness import compare_against_ras_baseline
    from ..models.run_causal_experiment import RESULTS_DIR, build_inputs
    from .report_format import fmt_num as _num
    from .reviewer_checks import paid_up_dose_pre
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from estimator import _within_transform  # type: ignore[no-redef]
    from experiment_config import CONFIG_DIR, load_causal_experiment_config  # type: ignore[no-redef]
    from mandate_design import apply_weighting, did_design  # type: ignore[no-redef]
    from robustness import compare_against_ras_baseline  # type: ignore[no-redef]
    from run_causal_experiment import RESULTS_DIR, build_inputs  # type: ignore[no-redef]
    from report_format import fmt_num as _num  # type: ignore[no-redef]
    from reviewer_checks import paid_up_dose_pre  # type: ignore[no-redef]

DEFAULT_CONFIGS = [CONFIG_DIR / "main.yaml", CONFIG_DIR / "baseline.yaml", CONFIG_DIR / "clean_window.yaml"]
ALPHA, POWER = 0.05, 0.80


def mde_multiplier(n_clusters: int) -> float:
    """Return the MDE multiplier t_{0.975, G-1} + t_{0.80, G-1} for G clusters.

    alpha = 0.05 two-sided, power = 0.80, t(G-1) reference; nan for G < 2. The
    MDE of a coefficient is this multiplier times its CR1 SE.
    """
    if n_clusters < 2:
        return math.nan
    df = n_clusters - 1
    return float(student_t.ppf(1 - ALPHA / 2, df) + student_t.ppf(POWER, df))


def within_bank_sd(y: torch.Tensor, weights: torch.Tensor) -> float:
    """Return SD(y_it - mean_i) over cells with finite y and positive weight.

    Each bank i's own mean is removed from its cells, the residuals are pooled
    across banks, and their standard deviation is returned -- the variation in y
    the bank fixed effects leave.
    """
    valid = y.isfinite() & (weights > 0)
    resid = []
    for i in range(y.shape[0]):
        yi = y[i][valid[i]]
        if yi.numel() >= 2:
            resid.append(yi - yi.mean())
    if not resid:
        return math.nan
    r = torch.cat(resid)
    return float(r.std()) if r.numel() >= 2 else math.nan


def _sd(x: torch.Tensor) -> float:
    x = x[x.isfinite()]
    return float(x.std()) if x.numel() >= 2 else math.nan


def residual_sd(y: torch.Tensor, d, coef: str, partial: list[str] | None = None, fe: bool = True) -> float:
    """Return the weighted SD of regressor `coef` after FWL residualization.

    `coef` ("dose" or "exposure") is residualized (Frisch-Waugh-Lovell) against
    bank FE, period FE, d.controls and the other main regressor, weighted by the
    design weights, over the cells fit_did uses (finite y and regressors,
    positive weight). With x~ the residual and w the weights, the returned value
    is

        sqrt( sum_c w_c x~_c^2 / sum_c w_c ).

    `partial` restricts the partialled-out regressors to those names (FE always
    included when fe=True); fe=False with partial=[] gives the weighted SD around
    the weighted mean on the same cells.
    """
    regs = {"dose": d.dose_x_post, "exposure": d.exposure_x_post, **d.controls}
    valid = y.isfinite() & (d.weights > 0)
    for r in regs.values():
        valid &= r.isfinite()
    if int(valid.sum()) < 2:
        return math.nan
    w = torch.where(valid, d.weights, 0.0).to(torch.float64)
    mask = valid.reshape(-1)
    w_flat = w.reshape(-1)[mask]
    sqrt_w = w_flat.sqrt()

    def _clean(r: torch.Tensor) -> torch.Tensor:
        v = torch.where(valid, r, 0.0).to(torch.float64)
        if fe:
            return _within_transform(v, w).reshape(-1)[mask]
        v = v.reshape(-1)[mask]
        return v - (v * w_flat).sum() / w_flat.sum()

    use = [nm for nm in regs if nm != coef and (partial is None or nm in partial)]
    x = _clean(regs[coef]) * sqrt_w
    others = [_clean(regs[nm]) * sqrt_w for nm in use]
    if others:
        z = torch.stack(others, dim=1)
        x = x - z @ (torch.linalg.pinv(z) @ x)
    return math.sqrt(float((x ** 2).sum() / w_flat.sum()))


def main() -> None:
    """Build the MDE and residual-SD tables for each config and save power_mde.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append",
                        help="experiment config(s); default: main.yaml, baseline.yaml and clean_window.yaml")
    args = parser.parse_args()
    configs = args.config or DEFAULT_CONFIGS

    lines = ["| config | outcome | network | coef | beta | CR1 SE | G | MDE | |beta|/MDE | "
             "SD(regressor) | within SD(y) | std. MDE (SD of y per SD of regressor) | "
             "resid SD(regressor) | std. MDE (resid) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    steps = ["| config | outcome | network | regressor | SD, same cells, no FE | after bank + period FE | "
             "+ other main regressor | ... + exposure level only | ... + Z_pre x post only | "
             "+ all controls (= resid SD) | share left (resid / no FE) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for path in configs:
        cfg = load_causal_experiment_config(path)
        inp = build_inputs(cfg, path)
        print(f"\n== {path.stem}: N={inp.n} x T={inp.t} ({inp.periods[0]} .. {inp.periods[-1]})")
        dose_sets = [("", inp.dose_pre, inp.design_main, inp.design_ras)]
        req = cfg.paid_up_capital_requirement
        if req is not None:
            try:
                dose_pu = paid_up_dose_pre(cfg, inp, req)
            except ValueError as e:
                print(f"  paid-up dose not computed: {e}")
            else:
                dose_sets.append((" (paid-up dose)", dose_pu,
                                  *(apply_weighting(did_design(W, dose_pu, inp.controls_pre, inp.timing),
                                                    cfg.weighting) for W in (inp.W_main, inp.W_ras))))
        for suffix, dose, design_w, design_r in dose_sets:
            _mde_rows(path.stem, inp, suffix, dose, {"W_hat": design_w, "RAS": design_r}, lines, steps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "power_mde.md"
    out.write_text(f"MDE at alpha={ALPHA} two-sided, power={POWER}, t(G-1) reference; see power_mde.py. "
                   "Rows marked (paid-up dose) use the paid-up dose and the exposure rebuilt from it "
                   "(reviewer_checks paid_up_dose, Eq. A'); betas are per unit of each dose. "
                   "std. MDE (resid) scales by the regressor's SD after FE, controls and the other "
                   "main regressor are partialled out (weighted).\n\n"
                   + "\n".join(lines) + "\n\n"
                   "Where each regressor's variation goes (weighted SDs on the fit's cells, partialling "
                   "out step by step; the last step is the resid SD used above):\n\n"
                   + "\n".join(steps) + "\n")
    print(f"\nsaved {out}")


def _mde_rows(stem: str, inp, suffix: str, dose: torch.Tensor, designs: dict, lines: list[str],
              steps: list[str]) -> None:
    sd_reg = {(net, "dose"): _sd(dose) for net in designs}
    for net, d in designs.items():
        post = d.dose_x_post != 0
        sd_reg[(net, "exposure")] = _sd(d.exposure_x_post[post]) if post.any() else math.nan
    print(f"  {suffix.strip() or 'leverage dose'}: SD(dose_pre) across banks = {_num(sd_reg[('W_hat', 'dose')], '.4g')}; "
          f"SD(exposure x post) W_hat = {_num(sd_reg[('W_hat', 'exposure')], '.4g')}, "
          f"RAS = {_num(sd_reg[('RAS', 'exposure')], '.4g')}")

    for name, y in inp.y.items():
        cmp = compare_against_ras_baseline(y, designs["W_hat"], designs["RAS"], n_boot=0)
        for net, res, d in [("W_hat", cmp.main, designs["W_hat"]), ("RAS", cmp.baseline, designs["RAS"])]:
            sd_y = within_bank_sd(y, d.weights)
            k = mde_multiplier(res.n_clusters)
            for coef in ["dose", "exposure"]:
                beta, se = getattr(res, f"beta_{coef}"), getattr(res, f"se_{coef}")
                mde = k * se
                ratio = abs(beta) / mde if math.isfinite(mde) and mde > 0 else math.nan
                y_ok = sd_y and math.isfinite(sd_y) and sd_y > 0
                std_mde = mde * sd_reg[(net, coef)] / sd_y if y_ok else math.nan
                sd_res = residual_sd(y, d, coef)
                std_mde_res = mde * sd_res / sd_y if y_ok else math.nan
                sd_raw_cells = residual_sd(y, d, coef, partial=[], fe=False)
                sd_fe = residual_sd(y, d, coef, partial=[])
                other = "exposure" if coef == "dose" else "dose"
                sd_fe_other = residual_sd(y, d, coef, partial=[other])
                z_names = [nm for nm in d.controls if nm != "exposure_level"]
                sd_level = residual_sd(y, d, coef, partial=[other, "exposure_level"])
                sd_z = residual_sd(y, d, coef, partial=[other, *z_names])
                share = sd_res / sd_raw_cells if math.isfinite(sd_raw_cells) and sd_raw_cells > 0 else math.nan
                steps.append(f"| {stem} | {name} | {net + suffix} | {coef} x post | {_num(sd_raw_cells, '.3g')} | "
                             f"{_num(sd_fe, '.3g')} | {_num(sd_fe_other, '.3g')} | {_num(sd_level, '.3g')} | "
                             f"{_num(sd_z, '.3g')} | {_num(sd_res, '.3g')} | {_num(share, '.3f')} |")
                print(f"  [{name}] {net + suffix:21s} {coef:8s} beta={_num(beta, '.4g'):>10s} "
                      f"SE={_num(se, '.3g'):>9s} G={res.n_clusters:2d} MDE={_num(mde, '.3g'):>9s} "
                      f"|b|/MDE={_num(ratio, '.2f'):>5s} std.MDE={_num(std_mde, '.3f')} "
                      f"std.MDE(resid)={_num(std_mde_res, '.3f')}")
                lines.append(f"| {stem} | {name} | {net + suffix} | {coef} x post | {_num(beta, '.4g')} | "
                             f"{_num(se, '.3g')} | {res.n_clusters} | {_num(mde, '.3g')} | "
                             f"{_num(ratio, '.2f')} | {_num(sd_reg[(net, coef)], '.3g')} | "
                             f"{_num(sd_y, '.3g')} | {_num(std_mde, '.3f')} | "
                             f"{_num(sd_res, '.3g')} | {_num(std_mde_res, '.3f')} |")


if __name__ == "__main__":
    main()
