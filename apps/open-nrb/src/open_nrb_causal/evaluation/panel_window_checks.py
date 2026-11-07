#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "pyyaml",
# ]
# ///
"""Cross-paper checks raised by the 2026-11-05 Paper 1 / Paper 2 read-through. Read-only.

  months       which calendar months are absent from Paper 1's artefact (draft §8 interface
               check: 133 of 134 months, one absent after Falgun 2075)
  complete     which commercial-bank labels pass the complete-case filter (no NaN in X, r, c, K,
               same rule as open_nrb's _filter_complete_banks) over the full artefact span vs. over
               this config's window only; lists the labels that would be added by the shorter
               window. Labels, not distinct banks (relabel pairs count twice).
  zero_fit     bank-months in this config's window where the published r (or c) is 0: how many
               get a positive row (column) sum in W_hat and RAS, with median and max. Checks the
               draft §8 sentence on where each network puts the misfit.
  w_hat_diff   relative difference between the artefact's W_hat and the ablation_no_kl_pressure
               run's (Paper 1 §10's 1.5e-5 figure), with both files' mtimes, so it is clear
               whether that figure was measured on the artefact Paper 2 uses.

Writes results/panel_window_checks_<config>.md.

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.panel_window_checks
  uv run python -m open_nrb_causal.evaluation.panel_window_checks --config src/open_nrb_causal/experiments/configs/clean_window.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

import torch

try:
    from open_nrb.evaluation.report_io import captured_output
    from open_nrb.models.data_loader import BS_MONTHS, _period_key
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "paper1_model"))
    from evaluation.report_io import captured_output  # type: ignore[no-redef]
    from models.data_loader import BS_MONTHS, _period_key  # type: ignore[no-redef]

try:
    from ..models.experiment_config import CONFIG_DIR, load_causal_experiment_config
    from ..models.panel_alignment import load_commercial_panel
    from ..models.run_causal_experiment import RESULTS_DIR, build_inputs
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from experiment_config import CONFIG_DIR, load_causal_experiment_config  # type: ignore[no-redef]
    from panel_alignment import load_commercial_panel  # type: ignore[no-redef]
    from run_causal_experiment import RESULTS_DIR, build_inputs  # type: ignore[no-redef]

DEFAULT_CONFIGS = [CONFIG_DIR / "main.yaml"]
ZERO_TOL = 1.0  # NPR mn; a row/column sum above this counts as "positive"


def _label(p) -> str:
    return f"{p[1]} {p[0]}"


def _month_no(p) -> int:
    y, m = _period_key(*p)
    return y * 12 + m


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _complete(panel, idx: list[int]) -> list[str]:
    nan_any = (panel.X[:, idx, :].isnan().any(dim=(1, 2)) | panel.r[:, idx].isnan().any(dim=1)
               | panel.c[:, idx].isnan().any(dim=1) | panel.K[:, idx].isnan().any(dim=1))
    return [nm for nm, bad in zip(panel.bfi_names, nan_any.tolist()) if not bad]


def report(out, cfg, path: Path) -> None:
    """Write the months, complete, zero_fit and w_hat_diff sections for one config.

    w_hat_diff reports two relative differences between the artefact's W_hat (a)
    and the ablation run's (b) over finite cells: the Frobenius norm
    ||a - b|| / ||a|| and the mean-absolute norm sum|a - b| / sum|a|.
    """
    inp = build_inputs(cfg, path)
    art = inp.artifact

    out("## months")
    nums = [_month_no(p) for p in art.periods]
    gaps = [(art.periods[k], art.periods[k + 1], nums[k + 1] - nums[k] - 1)
            for k in range(len(nums) - 1) if nums[k + 1] - nums[k] != 1]
    span = nums[-1] - nums[0] + 1
    out(f"- artefact: {len(nums)} months, {_label(art.periods[0])} .. {_label(art.periods[-1])}, "
        f"calendar span {span}")
    for a, b, n_missing in gaps:
        missing = [(y, BS_MONTHS[(m) % 12]) for y, m in
                   [divmod(_month_no(a) + k, 12) for k in range(1, n_missing + 1)]]
        out(f"- gap between {_label(a)} and {_label(b)}: {n_missing} month(s) absent "
            f"({', '.join(_label(p) for p in missing)})")
    if not gaps:
        out("- no gaps")

    out("\n## complete")
    panel = load_commercial_panel(cfg.csv_path)
    key = {p: t for t, p in enumerate(panel.periods)}
    full_idx = [key[p] for p in art.periods if p in key]
    win_idx = [key[p] for p in inp.periods if p in key]
    full = _complete(panel, full_idx)
    win = _complete(panel, win_idx)
    out(f"- commercial-bank labels in the CSV from {_label(panel.periods[0])}: {len(panel.bfi_names)}")
    out(f"- complete over the artefact span ({len(full_idx)} months): {len(full)} ({', '.join(full)})")
    out(f"- complete over this config's window, {_label(inp.periods[0])} .. {_label(inp.periods[-1])} "
        f"({len(win_idx)} months): {len(win)}")
    extra = [nm for nm in win if nm not in full]
    out(f"- complete in the window but dropped by the full-span filter: {len(extra)} "
        f"({', '.join(extra) or 'none'})")
    for nm in extra:
        i = panel.bfi_names.index(nm)
        bad = [panel.periods[t] for t in full_idx
               if panel.X[i, t].isnan().any() or panel.r[i, t].isnan() or panel.c[i, t].isnan()
               or panel.K[i, t].isnan()]
        out(f"  - {nm}: {len(bad)} incomplete months in the full span, first {_label(bad[0])}, "
            f"last {_label(bad[-1])}")

    out("\n## zero_fit")
    t = inp.t
    r, c = inp.panel.r[:, :t], inp.panel.c[:, :t]
    for net, W in [("W_hat", art.W_hat[:t]), ("RAS", art.W_baseline[:t])]:
        rows = W.sum(dim=2).T  # (N, T) lending implied by W
        cols = W.sum(dim=1).T  # (N, T) borrowing implied by W
        for side, pub, got in [("lending (row)", r, rows), ("borrowing (column)", c, cols)]:
            m = (pub == 0) & got.isfinite()
            vals = got[m]
            if vals.numel() == 0:
                out(f"- {net}, published {side} = 0 in no finite bank-months")
                continue
            pos = vals > ZERO_TOL
            banks = sorted({inp.bfi_names[i] for i in range(len(inp.bfi_names)) if (m[i] & (got[i] > ZERO_TOL)).any()})
            out(f"- {net}, published {side} = 0 in {int(m.sum())} bank-months: implied sum > "
                f"{ZERO_TOL:g} in {int(pos.sum())}; median {float(vals.median()):.4g}, max "
                f"{float(vals.max()):.4g}; banks with any such month: {len(banks)} of {len(inp.bfi_names)}")

    out("\n## w_hat_diff")
    a_path = Path(cfg.artifact_path)
    b_path = a_path.parent.parent / "ablation_no_kl_pressure" / a_path.name
    out(f"- artefact `{a_path}` (mtime {_mtime(a_path)})")
    if not b_path.exists():
        out(f"- `{b_path}` not found: not compared")
        return
    b = torch.load(b_path, map_location="cpu")["W_hat"]
    a = art.W_hat
    ok = a.isfinite() & b.isfinite()
    d = (a[ok] - b[ok]).abs()
    rel_fro = float((a[ok] - b[ok]).norm() / a[ok].norm())
    rel_abs = float(d.sum() / a[ok].abs().sum())
    out(f"- ablation `{b_path}` (mtime {_mtime(b_path)})")
    out(f"- relative difference, finite cells: Frobenius {rel_fro:.3g}, sum|a-b|/sum|a| {rel_abs:.3g}; "
        f"max abs diff {float(d.max()):.4g} (Paper 1 §10 reports 1.5e-5, max abs 0.18)")


def main() -> None:
    """Run the panel and window checks for each config and save the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", help="experiment config(s); default: main.yaml")
    args = parser.parse_args()
    for path in args.config or DEFAULT_CONFIGS:
        cfg = load_causal_experiment_config(path)
        with captured_output() as buf:
            print(f"# Panel and window checks: {path.stem}\n")
            report(print, cfg, path)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = RESULTS_DIR / f"panel_window_checks_{path.stem}.md"
        dest.write_text(buf.getvalue())
        print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
