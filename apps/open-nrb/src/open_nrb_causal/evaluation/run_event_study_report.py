#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
# ]
# ///
"""Build the paper figures and table from results already saved on disk.

From results/<config>_seed<k>.json (run_causal_experiment.py) and results/bs_extraction_audit.csv
(audit_bs_extraction.py) -- never re-fits anything, same "reads a saved
artifact, never retrains" convention as open_nrb.evaluation.make_plots.

Writes to results/figures/:
  event_study.png     per outcome x {dose, exposure}: event-time coefficients, 95% CR1 CI,
                      one series per run (e.g. full panel vs clean window); filled marker =
                      WCR wild-bootstrap p < 0.05, reference bin -1 drawn at 0
  did_forest.png      pooled DiD coefficients, W_hat vs RAS baseline, per run, 95% CR1 CI
  extraction_audit.png  per-bulletin share of sheets passing the asset identity + mean
                      digit-fragment count, with the pre/transition/post windows shaded
  did_table.md        the pooled DiD numbers behind did_forest.png

Usage (apps/open-nrb root):
  uv run python -m open_nrb_causal.evaluation.run_event_study_report
  uv run python -m open_nrb_causal.evaluation.run_event_study_report results/a.json results/b.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

try:
    from ..models.mandate_design import period_key
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
    from mandate_design import period_key  # type: ignore[no-redef]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
AUDIT_CSV = RESULTS_DIR / "bs_extraction_audit.csv"
Z = 1.96
OUTCOME_LABELS = {
    "net_profit_loss": "Net profit / size",
    "loan_loss_provision": "Loan-loss provisions / size",
    "loan_written_off": "Loans written off / size",
    "nim": "Net interest income / size",
}


def _run_label(run: dict, path: Path) -> str:
    end = run.get("post_window_end")
    window = f"post to {end[1]} {end[0]}" if end else "full panel"
    spec = ", ".join(v for v in (run.get("outcome_construction"), run.get("weighting")) if v)
    return f"{path.stem.rsplit('_seed', 1)[0]} ({window}{', ' + spec if spec else ''})"


def _bin_order(labels: set[str]) -> list[str]:
    ints = sorted(int(b) for b in labels if b != "transition")
    pre, post = [b for b in ints if b < 0], [b for b in ints if b >= 0]
    return [str(b) for b in pre] + (["transition"] if "transition" in labels else []) + [str(b) for b in post]


def _finite(*xs: float | None) -> bool:
    return all(x is not None and math.isfinite(x) for x in xs)


def _sci(ax, axis: str) -> None:
    # Exposure coefficients are ~1e-5: shared x10^k offset instead of long decimals.
    ax.ticklabel_format(axis=axis, style="sci", scilimits=(-2, 3), useMathText=True)
    getattr(ax, f"{axis}axis").get_offset_text().set_fontsize(7)
    getattr(ax, f"{axis}axis").set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7)


def plot_event_study(runs: list[tuple[str, dict]], out: Path) -> None:
    """Plot the per-outcome event-time coefficient panels across runs to `out`."""
    outcomes = list(runs[0][1]["outcomes"])
    fig, axes = plt.subplots(len(outcomes), 2, figsize=(11, 2.6 * len(outcomes)), squeeze=False)
    for row, name in enumerate(outcomes):
        for col, coef in enumerate(["dose", "exposure"]):
            ax = axes[row][col]
            labels: set[str] = set()
            for _, run in runs:
                es = run["outcomes"][name]["event_study"]
                labels |= set(es[coef]) | {str(es["reference_bin"])}
            order = _bin_order(labels)
            xpos = {b: i for i, b in enumerate(order)}
            offset = 0.12 * (len(runs) - 1)
            for k, (label, run) in enumerate(runs):
                es = run["outcomes"][name]["event_study"]
                dx = -offset / 2 + 0.12 * k
                ref = str(es["reference_bin"])
                pts = [(ref, 0.0, 0.0, None)] + [
                    (b, v["beta"], v["se"], v["p"]) for b, v in es[coef].items() if _finite(v["beta"], v["se"])]
                pts.sort(key=lambda p: xpos[p[0]])
                xs = [xpos[b] + dx for b, *_ in pts]
                ax.errorbar(xs, [p[1] for p in pts], yerr=[Z * p[2] for p in pts], fmt="none",
                            ecolor=f"C{k}", elinewidth=1, capsize=2)
                ax.plot(xs, [p[1] for p in pts], color=f"C{k}", linewidth=0.8, alpha=0.6,
                        label=label if row == 0 and col == 0 else None)
                for x, (b, beta, _, p) in zip(xs, pts):
                    ax.plot(x, beta, "o", color=f"C{k}", markersize=4,
                            markerfacecolor=f"C{k}" if p is not None and p < 0.05 else "white")
            ax.axhline(0, color="black", linewidth=0.5)
            if "transition" in xpos:
                ax.axvspan(xpos["transition"] - 0.5, xpos["transition"] + 0.5, color="grey", alpha=0.15)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(["trans." if b == "transition" else b for b in order], fontsize=8)
            ax.set_title(f"{OUTCOME_LABELS.get(name, name)} - {coef}", fontsize=9)
            _sci(ax, "y")
            if row == len(outcomes) - 1:
                ax.set_xlabel("fiscal years relative to mandate (ref = -1)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Reserve the top band for the legend, then place it there explicitly.
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.6)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(runs),
               fontsize=8, frameon=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _did_rows(runs: list[tuple[str, dict]]) -> list[dict]:
    rows = []
    for label, run in runs:
        for name, res in run["outcomes"].items():
            te = res.get("total_effect")
            if te:  # headline beta (quarterly configs): dose-only total effect, no network
                rows.append({"run": label, "outcome": name, "network": "dose-only from 2070", "coef": "dose",
                             "beta": te["beta_dose"], "se": te["se_dose"], "p": te["p_dose"],
                             "n_obs": te["n_obs"], "clusters": te["n_clusters"]})
            for net, key in [("W_hat", "did"), ("RAS", "did_ras_baseline")]:
                d = res[key]
                for coef in ["dose", "exposure"]:
                    rows.append({"run": label, "outcome": name, "network": net, "coef": coef,
                                 "beta": d[f"beta_{coef}"], "se": d[f"se_{coef}"], "p": d[f"p_{coef}"],
                                 "n_obs": d["n_obs"], "clusters": d["n_clusters"]})
    return rows


def plot_did_forest(rows: list[dict], out: Path) -> None:
    """Plot the pooled DiD forest (W_hat vs RAS, per run) to `out`."""
    outcomes = list(dict.fromkeys(r["outcome"] for r in rows))
    series = list(dict.fromkeys((r["run"], r["network"]) for r in rows))
    fig, axes = plt.subplots(len(outcomes), 2, figsize=(10, 1.0 + 0.45 * len(series) * len(outcomes)),
                             squeeze=False)
    for i, name in enumerate(outcomes):
        for j, coef in enumerate(["dose", "exposure"]):
            ax = axes[i][j]
            sel = [r for r in rows if r["outcome"] == name and r["coef"] == coef]
            for k, r in enumerate(sel):
                y = len(sel) - 1 - k
                if _finite(r["beta"], r["se"]):
                    ax.errorbar(r["beta"], y, xerr=Z * r["se"], fmt="o", color=f"C{series.index((r['run'], r['network']))}",
                                markersize=4, capsize=2)
                # p-values in a fixed column just right of the axes: never collides with the
                # points, the CIs or the title, whatever the x range.
                p_txt = "p=n/a" if r["p"] is None or not math.isfinite(r["p"]) else f"p={r['p']:.2f}"
                ax.text(1.02, y, p_txt, transform=ax.get_yaxis_transform(), va="center", ha="left",
                        fontsize=7, clip_on=False)
            ax.set_ylim(-0.6, len(sel) - 0.4)
            ax.set_yticks(range(len(sel)))
            ax.set_yticklabels([f"{r['run']} | {r['network']}" if j == 0 else "" for r in reversed(sel)],
                               fontsize=7)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.set_title(f"{OUTCOME_LABELS.get(name, name)} - {coef} x post", fontsize=9, pad=4)
            _sci(ax, "x")
    fig.tight_layout(w_pad=3.0, h_pad=1.4)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_did_table(rows: list[dict], out: Path) -> None:
    """Write the pooled DiD numbers behind did_forest.png to `out`."""
    def num(x: float | None, f: str) -> str:
        """Format x with spec f, or return "n/a" when x is None or non-finite."""
        return "n/a" if x is None or not math.isfinite(x) else format(x, f)

    lines = ["| run | outcome | network | coef | beta | CR1 SE | wild p | n_obs | clusters |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['outcome']} | {r['network']} | {r['coef']} x post | "
                     f"{num(r['beta'], '.4g')} | {num(r['se'], '.3g')} | {num(r['p'], '.3f')} | "
                     f"{r['n_obs']} | {r['clusters']} |")
    out.write_text("\n".join(lines) + "\n")


def plot_extraction_audit(audit_csv: Path, run: dict, out: Path) -> None:
    """Plot balance-sheet extraction quality by NRB bulletin to `out`."""
    by_period: dict[tuple[int, str], list[dict]] = defaultdict(list)
    with audit_csv.open() as fh:
        for r in csv.DictReader(fh):
            by_period[(int(r["year_bs"]), r["month_bs"])].append(r)
    periods = sorted(by_period, key=period_key)
    idx = {p: i for i, p in enumerate(periods)}

    def rate(rs: list[dict]) -> float:
        """Return the share of sheets whose asset items sum to TOTAL ASSETS."""
        vals = [r["assets"] == "True" for r in rs if r["assets"] in ("True", "False")]
        return sum(vals) / len(vals) if vals else math.nan

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(range(len(periods)), [rate(by_period[p]) for p in periods], color="C0", linewidth=1.2,
            label="share of sheets: asset items sum to TOTAL ASSETS")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("pass rate", color="C0")
    ax2 = ax.twinx()
    ax2.plot(range(len(periods)), [sum(int(r["frag"]) for r in by_period[p]) / len(by_period[p]) for p in periods],
             color="C3", linewidth=0.9, alpha=0.8, label="mean single-digit values per sheet")
    ax2.set_ylabel("mean fragments", color="C3")

    def at(p: list | None) -> float | None:
        """Map a period key to its x-axis index, or None if absent/unknown."""
        return None if p is None or tuple(p) not in idx else idx[tuple(p)]

    pre_end, eff = at(run["pre_window_end"]), at(run["effective_period"])
    start = at(run["periods"][0])
    if start is not None and pre_end is not None:
        ax.axvspan(start - 0.5, pre_end + 0.5, color="C2", alpha=0.15, label="dose window (frozen)")
    if pre_end is not None and eff is not None:
        ax.axvspan(pre_end + 0.5, eff + 0.5, color="grey", alpha=0.15, label="transition")
    clean_end = at(run.get("post_window_end"))
    if clean_end is not None:
        ax.axvline(clean_end + 0.5, color="black", linestyle="--", linewidth=0.8, label="clean-window end")
    ticks = [i for i, p in enumerate(periods) if p[1] == "Shrawan"]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"FY{periods[i][0]}/{str(periods[i][0] + 1)[-2:]}" for i in ticks], fontsize=7, rotation=45)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, frameon=False)
    ax.set_title("Balance-sheet extraction quality by NRB bulletin", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build every figure and table from the run JSONs and audit CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path,
                        help="run_causal_experiment JSONs (default: every results/*_seed*.json)")
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "figures")
    args = parser.parse_args()

    paths = args.runs or sorted(RESULTS_DIR.glob("*_seed*.json"))
    if not paths:
        raise SystemExit(f"no run JSONs given or found in {RESULTS_DIR} -- run run_causal_experiment first")
    runs = [(_run_label(json.loads(p.read_text()), p), json.loads(p.read_text())) for p in paths]
    # Full-panel run first, so it gets colour C0 in every figure.
    runs.sort(key=lambda r: r[1].get("post_window_end") is not None)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_event_study(runs, args.out_dir / "event_study.png")
    rows = _did_rows(runs)
    plot_did_forest(rows, args.out_dir / "did_forest.png")
    write_did_table(rows, args.out_dir / "did_table.md")
    written = ["event_study.png", "did_forest.png", "did_table.md"]
    if args.audit_csv.exists():
        clean = next((r for _, r in runs if r.get("post_window_end")), runs[0][1])
        plot_extraction_audit(args.audit_csv, clean, args.out_dir / "extraction_audit.png")
        written.append("extraction_audit.png")
    else:
        print(f"{args.audit_csv} not found -- run audit_bs_extraction first; skipping extraction_audit.png")
    print(f"runs: {[label for label, _ in runs]}")
    print(f"wrote {', '.join(written)} to {args.out_dir}")


if __name__ == "__main__":
    main()
