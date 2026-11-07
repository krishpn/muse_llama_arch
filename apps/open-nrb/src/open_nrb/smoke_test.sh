#!/usr/bin/env bash
# Reviewer smoke test, Paper 1 (open_nrb): confirms the synthetic pipeline runs end to end --
# generate -> train -> capped projection -> save artifact -> evaluate -> plot -- plus the unit
# tests and the parameter-count check. Data-free (the synthetic path makes its own data) and
# ~1-2 min on CPU. Run from the apps/open-nrb repo root:
#   bash src/open_nrb/smoke_test.sh
#
# This proves the code works; it does NOT reproduce any paper number. The reconstruction quality
# from synthetic_smoke.yaml (epochs 150, 1 seed, N=16, T=8) is meaningless -- use the full
# README "## STEPS" at the real epoch budget / seed count for anything that goes in the paper.
# Stops at the first failing step. Each step's console output goes to results/logs_smoke/<step>.log.
set -euo pipefail

PKG=src/open_nrb
CFG=$PKG/experiments/configs/synthetic_smoke.yaml
LOG=$PKG/results/logs_smoke
mkdir -p "$LOG"

step() {
    local name=$1
    shift
    echo "=== $name: $*"
    "$@" 2>&1 | tee "$LOG/$name.log"
}

# A. Unit tests.
step 1_tests uv run python -m pytest "$PKG/tests" -q

# B. Parameter count (instant, no data, no GPU) -- prints TOTAL PARAMETERS: 2133.
step 2_param_count uv run python -m open_nrb.evaluation.param_count

# C. One short synthetic run -- writes results/synthetic_smoke/seed_0.pt.
step 3_synthetic uv run python -m open_nrb.models.run_synthetic_experiment \
    --config "$CFG" --workers 1 --device cpu

# D. Post-hoc evaluation on the artifact C just wrote (none of these retrain).
step 4_seed_significance uv run python -m open_nrb.evaluation.seed_significance --synthetic-config "$CFG"
step 5_capped_ras uv run python -m open_nrb.evaluation.capped_ras_baseline --synthetic-config "$CFG"
step 6_plot uv run python -m open_nrb.evaluation.plot_synthetic_network --config "$CFG" --seed 0 --period -1

echo "=== done; smoke artifacts in $PKG/results/synthetic_smoke, logs in $LOG"
