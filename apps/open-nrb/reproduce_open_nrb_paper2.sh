#!/usr/bin/env bash
# Reproduces every number, table and figure in paper2_draft.md, Paper 1 artifact -> Paper 2
# outputs, in one pass. Run from the apps/open-nrb root:
#   bash src/open_nrb_causal/reproduce_paper.sh
# Stops at the first failing step. Each step's console output goes to results/logs/<step>.log;
# the inputs' SHA-256 go to results/logs/inputs.sha256. README "Reproducing the paper" maps each
# table and figure to the output file it comes from.
set -euo pipefail

PKG=src/open_nrb_causal
CFG=$PKG/experiments/configs
LOG=$PKG/results/logs
CONFIGS=(main baseline clean_window)
mkdir -p "$LOG"

step() {
    local name=$1
    shift
    echo "=== $name: $*"
    "$@" 2>&1 | tee "$LOG/$name.log"
}

uv run python - "$LOG/inputs.sha256" <<'EOF'
import hashlib, sys, yaml
from datetime import datetime, timezone
from pathlib import Path
cfg_dir = Path("src/open_nrb_causal/experiments/configs")
files = sorted(cfg_dir.glob("*.yaml"))
for c in ["main.yaml", "baseline.yaml", "clean_window.yaml"]:
    d = yaml.safe_load((cfg_dir / c).read_text())
    files += [Path(d["artifact_path"]), Path(d["csv_path"]), Path(d["pnl_csv_path"])]
art = Path(yaml.safe_load((cfg_dir / "main.yaml").read_text())["artifact_path"])
files.append(art.parents[2] / "experiments" / "configs" / f"{art.parent.name}.yaml")
lines = [f"# {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
for f in dict.fromkeys(files):
    lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f}")
Path(sys.argv[1]).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

step 1_tests uv run python -m pytest $PKG/tests -q
step 2_audit uv run python -m open_nrb_causal.evaluation.audit_bs_extraction
step 3_interface uv run python -m open_nrb_causal.evaluation.check_paper1_interface \
    --config $CFG/main.yaml --config $CFG/baseline.yaml --config $CFG/clean_window.yaml

pids=()
for c in "${CONFIGS[@]}"; do
    step "4_run_$c" uv run python -m open_nrb_causal.models.run_causal_experiment --config "$CFG/$c.yaml" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

step 5_figures uv run python -m open_nrb_causal.evaluation.run_event_study_report \
    $PKG/results/main_seed0.json $PKG/results/baseline_seed0.json $PKG/results/clean_window_seed0.json
step 6_power_mde uv run python -m open_nrb_causal.evaluation.power_mde

pids=()
for c in "${CONFIGS[@]}"; do
    step "7_reviewer_checks_$c" uv run python -m open_nrb_causal.evaluation.reviewer_checks --config "$CFG/$c.yaml" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=== done; outputs in $PKG/results, logs in $LOG"
