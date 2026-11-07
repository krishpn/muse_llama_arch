#!/usr/bin/env bash
# Reviewer smoke test, Paper 2 (open_nrb_causal): confirms the causal pipeline runs end to end --
# align to the Paper 1 artifact -> build panel -> DiD / total effect / event study / RDD -> save
# JSON -> reviewer checks -- plus the unit tests. Uses smoke.yaml (a clone of main.yaml with
# n_bootstrap: 0), so it finishes in under a minute. Run from the apps/open-nrb repo root:
#   bash src/open_nrb_causal/smoke_test.sh
#
# This proves the code works; it does NOT reproduce a paper p-value. The POINT ESTIMATES equal
# main.yaml's, but the wild-cluster p-values are absent (step C, n_bootstrap: 0) or only 99 draws
# (step D) -- use the full README "## STEPS" / reproduce_paper.sh at 9,999 draws for anything that
# goes in the paper.
#
# Unlike Paper 1's smoke test this is NOT data-free: this is an empirical study, so it needs the
# Paper 1 artifact and the processed NRB CSVs already present. It refuses to run (with instructions)
# rather than silently produce an incomplete result if any input is missing.
# Stops at the first failing step. Each step's console output goes to results/logs_smoke/<step>.log.
set -euo pipefail

# Stream child stdout live through the `tee` pipes below. Python block-buffers stdout when it is a
# pipe (not a TTY), so without this a step prints nothing until its buffer fills or the process
# exits -- making a healthy run look hung.
export PYTHONUNBUFFERED=1

PKG=src/open_nrb_causal
CFG=$PKG/experiments/configs/smoke.yaml
LOG=$PKG/results/logs_smoke

# Guard: must be launched from the repo root, same convention as reproduce_paper.sh.
[ -d "$PKG" ] || { echo "error: $PKG not found -- run this from the apps/open-nrb repo root" >&2; exit 2; }

mkdir -p "$LOG"

step() {
    local name=$1
    shift
    echo "=== $name: $*"
    "$@" 2>&1 | tee "$LOG/$name.log"
}

# Guard: smoke.yaml points at the real Paper 1 artifact + NRB CSVs (it can't run data-free).
# Confirm all three exist before estimating, or abort pointing back to Paper 1. Parsed with shell
# (not pyyaml) so the guard itself has no dependency to resolve -- these are plain top-level keys.
missing=()
for key in artifact_path csv_path pnl_csv_path; do
    val=$(sed -n -E "s/^${key}:[[:space:]]*//p" "$CFG" | head -1)
    val=${val%%#*}                       # strip any trailing comment
    val=$(printf '%s' "$val" | tr -d '[:space:]')
    [ -z "$val" ] && { echo "error: '$key' missing from $CFG" >&2; exit 3; }
    [ -e "$val" ] || missing+=("$val")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "error: required Paper 2 inputs not found:" >&2
    for p in "${missing[@]}"; do echo "  $p" >&2; done
    echo "Run Paper 1 first (it writes real_run_seed0.pt) and ensure the processed NRB CSVs are" >&2
    echo "present -- see README 'Reviewer quick test run'. Paper 2 is empirical; it has no" >&2
    echo "data-free smoke path." >&2
    exit 3
fi

# A. Unit tests (seconds; the estimator math is checked here on internally-generated synthetic data).
step 1_tests uv run python -m pytest "$PKG/tests" -q

# B. Paper 1 interface check -- banks, periods, marginals, cap, no NaN W_hat in the window line up.
step 2_interface uv run python -m open_nrb_causal.evaluation.check_paper1_interface --config "$CFG"

# C. One causal run, CR1 SEs only (n_bootstrap: 0) -- seconds. Writes results/smoke_seed0.json.
step 3_causal uv run python -m open_nrb_causal.models.run_causal_experiment --config "$CFG"

# D. Reviewer checks, quick bootstrap (99 draws instead of 9,999). Writes results/reviewer_checks_smoke.md.
step 4_reviewer_checks uv run python -m open_nrb_causal.evaluation.reviewer_checks --config "$CFG" --n-boot 99

echo "=== done; smoke artifacts in $PKG/results, logs in $LOG"
