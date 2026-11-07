#!/usr/bin/env bash
# Reproduces every number, table and figure in paper1_draft.md for Paper 1 (open_nrb):
# the synthetic experiment, the real-data baseline + ablation, all evaluation/EDA artifacts,
# and the five referee-revision checks. This is the scripted form of the README "## STEPS"
# walkthrough (steps 4-17); it does NOT perform the remote-machine/Syncthing setup or the NRB
# PDF download+extraction (README "Before running anything..." and "Running the PDF Downloader")
# unless you pass --download. Run from the apps/open-nrb repo root:
#   bash src/open_nrb/reproduce_paper.sh
#
# Unlike smoke_test.sh (a ~1-2 min data-free sanity check at a meaningless epoch budget), this
# runs the full real epoch budget / seed counts -- the numbers it writes are the ones in the paper.
# Expect a GPU box and a long runtime.
#
# Guards so it can't silently produce an incomplete reproduction:
#   * It confirms the manual preconditions (on the GPU/remote machine, Syncthing "Up to Date")
#     before doing any heavy work -- skip with --yes once you've checked them yourself.
#   * It will NOT run the real-data steps if the extracted NRB panel CSV is missing: it aborts
#     with instructions, or re-runs the download+extraction if you pass --download, or you can
#     run only the data-free half with --synthetic-only.
#
# Options:
#   -y, --yes          skip the interactive precondition confirmation
#       --synthetic-only  run only the data-free steps (synthetic + referee ablations + param count)
#       --download       run the NRB PDF downloader + extractor first (hits NRB's server)
#       --workers N      worker processes for the synthetic training runs (default 4)
#       --device DEV     torch device for the synthetic training runs (e.g. cpu, cuda); on a
#                        small/shared GPU pass "cpu" per the README's CUDA-OOM note
#   -h, --help         show this help and exit
#
# Stops at the first failing step. Each step's console output goes to results/logs_reproduce/<step>.log;
# the inputs' SHA-256 go to results/logs_reproduce/inputs.sha256. README "## STEPS" maps each table
# and figure to the output file it comes from.
set -euo pipefail

PKG=src/open_nrb
CFG=$PKG/experiments/configs
LOG=$PKG/results/logs_reproduce
REAL_CSV=$PKG/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv

WORKERS=4
DEVICE=""
ASSUME_YES=0
SYNTHETIC_ONLY=0
DO_DOWNLOAD=0

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1 ;;
        --synthetic-only) SYNTHETIC_ONLY=1 ;;
        --download) DO_DOWNLOAD=1 ;;
        --workers) WORKERS=${2:?--workers needs a value}; shift ;;
        --device) DEVICE=${2:?--device needs a value}; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# Guard: must be launched from the repo root, same convention as smoke_test.sh.
[ -d "$PKG" ] || { echo "error: $PKG not found -- run this from the apps/open-nrb repo root" >&2; exit 2; }

mkdir -p "$LOG"

step() {
    local name=$1
    shift
    echo "=== $name: $*"
    "$@" 2>&1 | tee "$LOG/$name.log"
}

# Manual-precondition gate. These two can't be verified reliably from here (the remote/GPU
# machine and Syncthing state live outside this checkout), so confirm them rather than assume.
confirm_preconditions() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    cat <<'MSG'
This runs the FULL Paper 1 reproduction at the real epoch budget -- long, GPU-bound.
Before continuing, confirm the manual preconditions from the README:
  1. You are on the remote (GPU) machine this project trains on.
  2. Syncthing shows this folder "Up to Date" (not "Syncing"/"Out of Sync"), so the
     remote copy isn't stale -- see README "Before running anything on the remote (GPU) machine".
MSG
    if [ ! -t 0 ]; then
        echo "error: no TTY to confirm on; re-run with --yes once the preconditions above hold." >&2
        exit 4
    fi
    read -rp "Proceed? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
}

confirm_preconditions

# Device passthrough for the synthetic training runs only (the real runs are single-process).
DEV_ARGS=()
[ -n "$DEVICE" ] && DEV_ARGS=(--device "$DEVICE")

# Optional: NRB PDF download + extraction (README "Running the PDF Downloader"). Off by default
# because it hits NRB's own server; run as PEP 723 scripts, same as the README.
if [ "$DO_DOWNLOAD" -eq 1 ]; then
    step 0a_download uv run "$PKG/utils/pdf_downloader.py"
    step 0b_extract uv run "$PKG/utils/pdf_extract.py" --all
fi

# Guard: don't start the real-data half without the extracted panel -- abort with instructions
# rather than silently reconstruct nothing.
if [ "$SYNTHETIC_ONLY" -eq 0 ] && [ ! -f "$REAL_CSV" ]; then
    echo "error: real-data panel not found: $REAL_CSV" >&2
    echo "Run the NRB PDF downloader + extractor first (README 'Running the PDF Downloader')," >&2
    echo "or re-run with --download to do it now, or with --synthetic-only to skip the real-data steps." >&2
    exit 3
fi

# Record exactly which configs (and the real panel, when used) this run consumed.
uv run python - "$LOG/inputs.sha256" "$REAL_CSV" "$SYNTHETIC_ONLY" <<'EOF'
import hashlib, sys
from datetime import datetime, timezone
from pathlib import Path
out, real_csv, synthetic_only = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "1"
cfg_dir = Path("src/open_nrb/experiments/configs")
names = ["synthetic_baseline.yaml", "sweep_lambda_mi_0.yaml",
         "synthetic_single_block.yaml", "synthetic_non_sbm.yaml"]
if not synthetic_only:
    names += ["baseline.yaml", "ablation_no_kl_pressure.yaml"]
files = [cfg_dir / n for n in names]
if not synthetic_only and real_csv.exists():
    files.append(real_csv)
lines = [f"# {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
for f in dict.fromkeys(files):
    lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}")
Path(out).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

# ---- Data-free half: unit tests, parameter count, synthetic experiment + its evals, and the
#      referee synthetic ablations (README steps 4, 11, 12, 14, 15, 17). ----

step 01_tests uv run python -m pytest "$PKG/tests" -q
step 02_param_count uv run python -m open_nrb.evaluation.param_count

step 03_synthetic uv run python -m open_nrb.models.run_synthetic_experiment \
    --config "$CFG/synthetic_baseline.yaml" --workers "$WORKERS" "${DEV_ARGS[@]}"
step 04_seed_significance uv run python -m open_nrb.evaluation.seed_significance \
    --synthetic-config "$CFG/synthetic_baseline.yaml"
step 05_plot_synthetic uv run python -m open_nrb.evaluation.plot_synthetic_network \
    --config "$CFG/synthetic_baseline.yaml" --seed 0 --period -1

# sweep_lambda_mi_0 is the lambda_mi=0 control the referee checks compare against; the README
# referee section assumes it exists but never trains it, so do that here before steps 14/15.
step 06_ctrl_train uv run python -m open_nrb.models.run_synthetic_experiment \
    --config "$CFG/sweep_lambda_mi_0.yaml" --workers "$WORKERS" "${DEV_ARGS[@]}"
step 07_capped_ras uv run python -m open_nrb.evaluation.capped_ras_baseline \
    --synthetic-config "$CFG/sweep_lambda_mi_0.yaml"

step 08_single_block_train uv run python -m open_nrb.models.run_synthetic_experiment \
    --config "$CFG/synthetic_single_block.yaml" --workers "$WORKERS" "${DEV_ARGS[@]}"
step 09_single_block_cmp uv run python -m open_nrb.evaluation.seed_significance \
    --synthetic-config "$CFG/sweep_lambda_mi_0.yaml" --compare-config "$CFG/synthetic_single_block.yaml"
step 10_non_sbm_train uv run python -m open_nrb.models.run_synthetic_experiment \
    --config "$CFG/synthetic_non_sbm.yaml" --workers "$WORKERS" "${DEV_ARGS[@]}"
step 11_non_sbm_cmp uv run python -m open_nrb.evaluation.seed_significance \
    --synthetic-config "$CFG/sweep_lambda_mi_0.yaml" --compare-config "$CFG/synthetic_non_sbm.yaml"

if [ "$SYNTHETIC_ONLY" -eq 1 ]; then
    echo "=== done (synthetic-only); outputs in $PKG/results, logs in $LOG"
    exit 0
fi

# ---- Real-data half: baseline + ablation training, config comparison, EDA/figures, and the
#      persistence baseline (README steps 5, 6, 7, 8, 9, 10, 13, 16). Baseline and ablation are
#      independent but run sequentially here to stay within one GPU's memory (README's CUDA note). ----

step 12_real_baseline uv run python -m open_nrb.models.run_real_experiment \
    --config "$CFG/baseline.yaml"
step 13_real_ablation uv run python -m open_nrb.models.run_real_experiment \
    --config "$CFG/ablation_no_kl_pressure.yaml"

step 14_compare_configs uv run python -m open_nrb.evaluation.compare_configs \
    --config-a "$CFG/baseline.yaml" --config-b "$CFG/ablation_no_kl_pressure.yaml"
step 15_eda_model_checks uv run python -m open_nrb.evaluation.eda_model_checks
step 16_eda_survivorship uv run python -m open_nrb.evaluation.eda_survivorship
step 17_make_plots uv run python -m open_nrb.evaluation.make_plots
step 18_eda_real_panel uv run python -m open_nrb.evaluation.eda_real_panel --csv "$REAL_CSV"
step 19_persistence uv run python -m open_nrb.evaluation.persistence_baseline \
    --config "$CFG/baseline.yaml"

echo "=== done; outputs in $PKG/results, logs in $LOG"
