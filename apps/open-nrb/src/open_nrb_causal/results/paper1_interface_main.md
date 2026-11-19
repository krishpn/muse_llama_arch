# Paper 1 interface check: main

## Artifact
- path: `src/open_nrb/results/baseline/real_run_seed0.pt` (mtime 2026-11-07T19:30:15)
- keys: W_baseline, W_hat, baseline_diff, baseline_stability, bfi_names, c, cap, components_trend, degenerate_periods, losses, model_stability, non_finite_periods, periods, q, r, seed
- W_hat (133, 16, 16), W_baseline (133, 16, 16); N=16 x T=133, Asar 2072 .. Shrawan 2083
- NaN-filled by Paper 1: non-finite 0, degenerate 0 (none)
- this config's window: Asar 2072 .. Falgun 2075 (T=45); NaN-filled months inside it: none (build_inputs passed)

## Marginals (CSV vs artifact)
- r, c match; cap / K constant, implied kappa = 0.05

## Paper 1 provenance
- config `src/open_nrb/experiments/configs/baseline.yaml` (mtime 2026-11-07T11:33:52)
- kappa 0.05, lambda_mi 0.0, lambda_smooth 0.05, epochs 500, kl_beta_max 0.15, kl_free_bits 0.05, seed 0

## Network properties used in the text
- cap-binding share of off-diagonal W_hat cells, all valid months: 0.0699
- same, inside this config's window: 0.0193
- corr(exposure W_hat, exposure RAS), post cells (400): 0.968
- corr(exposure W_hat, exposure RAS), all cells (720): 0.939
