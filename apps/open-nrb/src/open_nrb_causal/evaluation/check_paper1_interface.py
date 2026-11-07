#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Check the Paper 1 -> Paper 2 interface, run once before the paper's estimation runs.

Consistency only, no new analysis: everything Paper 2 assumes about open_nrb's saved artifact,
reported in one file.

  artifact     keys, W_hat shape, N x T, period range, months Paper 1 NaN-filled (all, and inside
               this config's window -- build_inputs refuses the latter)
  marginals    aligned CSV r / c / K against the artifact's r / c / cap (check_marginals) and the
               implied kappa
  provenance   Paper 1 config next to the artifact (results/<stem>/ -> experiments/configs/
               <stem>.yaml): kappa, lambda_mi, epochs, seed; config vs artifact mtime; the
               artifact's seed against the config's
  network      cap-binding share of W_hat (Paper 1's own eda_model_checks.cap_binding_fraction)
               and corr(exposure under W_hat, exposure under RAS) over post cells -- the draft
               reports both as properties of the marginal-constrained, capped reconstruction,
               not as a validation of it

Read-only. Writes results/paper1_interface_<config>.md.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.check_paper1_interface
  uv run python -m open_nrb_causal.evaluation.check_paper1_interface --config src/open_nrb_causal/experiments/configs/baseline.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import torch
import yaml

try:
    from open_nrb.evaluation.eda_model_checks import cap_binding_fraction
    from open_nrb.evaluation.report_io import captured_output
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from evaluation.eda_model_checks import cap_binding_fraction  # type: ignore[no-redef]
    from evaluation.report_io import captured_output  # type: ignore[no-redef]

try:
    from ..models.experiment_config import CONFIG_DIR, load_causal_experiment_config
    from ..models.panel_alignment import check_marginals
    from ..models.run_causal_experiment import RESULTS_DIR, build_inputs
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from experiment_config import CONFIG_DIR, load_causal_experiment_config  # type: ignore[no-redef]
    from panel_alignment import check_marginals  # type: ignore[no-redef]
    from run_causal_experiment import RESULTS_DIR, build_inputs  # type: ignore[no-redef]

DEFAULT_CONFIGS = [CONFIG_DIR / "main.yaml"]


def _label(p) -> str:
    return f"{p[1]} {p[0]}"


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def report(out, cfg, path: Path) -> None:
    """Write the artifact, marginals, provenance and network-property sections for one config."""
    artifact_path = Path(cfg.artifact_path)
    raw = torch.load(artifact_path, map_location="cpu")
    # build_inputs runs every check Paper 2 enforces (index alignment, check_marginals, NaN months
    # in the window) and raises on the first failure
    inp = build_inputs(cfg, path)
    art = inp.artifact
    t_all = art.W_hat.shape[0]

    out("## Artifact")
    out(f"- path: `{artifact_path}` (mtime {_mtime(artifact_path)})")
    out(f"- keys: {', '.join(sorted(raw))}")
    out(f"- W_hat {tuple(art.W_hat.shape)}, W_baseline {tuple(art.W_baseline.shape)}; "
        f"N={len(art.bfi_names)} x T={t_all}, {_label(art.periods[0])} .. {_label(art.periods[-1])}")
    flagged = sorted(set(art.non_finite_periods + art.degenerate_periods))
    out(f"- NaN-filled by Paper 1: non-finite {len(art.non_finite_periods)}, degenerate "
        f"{len(art.degenerate_periods)} ({', '.join(_label(art.periods[t]) for t in flagged) or 'none'})")
    out(f"- this config's window: {_label(inp.periods[0])} .. {_label(inp.periods[-1])} (T={inp.t}); "
        "NaN-filled months inside it: none (build_inputs passed)")

    out("\n## Marginals (CSV vs artifact)")
    kappa = check_marginals(inp.panel, art)
    if kappa is None:
        out("- artifact has no r / c / cap keys: value check skipped, index check only")
    else:
        out(f"- r, c match; cap / K constant, implied kappa = {kappa:.6g}")

    out("\n## Paper 1 provenance")
    p1_cfg_path = artifact_path.parents[2] / "experiments" / "configs" / f"{artifact_path.parent.name}.yaml"
    if not p1_cfg_path.exists():
        out(f"- `{p1_cfg_path}` not found: provenance not checked")
    else:
        p1 = yaml.safe_load(p1_cfg_path.read_text())
        tr = p1.get("training", {})
        out(f"- config `{p1_cfg_path}` (mtime {_mtime(p1_cfg_path)})")
        out(f"- kappa {p1.get('kappa')}, lambda_mi {tr.get('lambda_mi')}, lambda_smooth {tr.get('lambda_smooth')}, "
            f"epochs {tr.get('epochs')}, kl_beta_max {tr.get('kl_beta_max')}, kl_free_bits {tr.get('kl_free_bits')}, "
            f"seed {p1.get('seed')}")
        if p1_cfg_path.stat().st_mtime > artifact_path.stat().st_mtime:
            out("- WARNING: the config is newer than the artifact -- the artifact may not reflect these values")
        if p1.get("seed") != art.seed:
            out(f"- WARNING: artifact seed {art.seed} != config seed {p1.get('seed')}")
        if kappa is not None and p1.get("kappa") is not None and abs(kappa - p1["kappa"]) > 1e-4 * p1["kappa"]:
            out(f"- WARNING: implied kappa {kappa:.6g} != config kappa {p1['kappa']}")

    out("\n## Network properties used in the text")
    if art.cap is not None:
        valid = torch.tensor([bool(torch.isfinite(w).all()) for w in art.W_hat])
        out(f"- cap-binding share of off-diagonal W_hat cells, all valid months: "
            f"{cap_binding_fraction(art.W_hat, art.cap, valid):.4f}")
        valid_win = valid.clone()
        valid_win[inp.t:] = False
        out(f"- same, inside this config's window: {cap_binding_fraction(art.W_hat, art.cap, valid_win):.4f}")
    d_main, d_ras = inp.design_main, inp.design_ras
    for label, a, b in [("post cells", d_main.exposure_x_post, d_ras.exposure_x_post),
                        ("all cells", d_main.exposure, d_ras.exposure)]:
        mask = a.isfinite() & b.isfinite() & ((a != 0) if label == "post cells" else True)
        corr = torch.corrcoef(torch.stack([a[mask], b[mask]]))[0, 1].item()
        out(f"- corr(exposure W_hat, exposure RAS), {label} ({int(mask.sum())}): {corr:.3f}")


def main() -> None:
    """Run the interface check for each config and save the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", help="experiment config(s); default: main.yaml")
    args = parser.parse_args()
    for path in args.config or DEFAULT_CONFIGS:
        cfg = load_causal_experiment_config(path)
        with captured_output() as buf:
            print(f"# Paper 1 interface check: {path.stem}\n")
            report(print, cfg, path)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = RESULTS_DIR / f"paper1_interface_{path.stem}.md"
        dest.write_text(buf.getvalue())
        print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
