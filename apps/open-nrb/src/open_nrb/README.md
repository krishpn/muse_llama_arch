# Who Lends to Whom? Reconstructing Interbank Exposures from Public Balance Sheets

## Project Overview

`open-nrb`

## Target Audience & Applied Venues

TBD
* **Intended Impact:**

## Key Applied Audit & Replicability Criteria

- [ ] Empirical validation suite testing graph recovery robustness under non-Gaussian, real-world noise distributions and partial network observability.
- [ ] Semi-synthetic simulation engine benchmarking Structural Hamming Distance (SHD) and False Discovery Rate (FDR) on empirical graph topologies (e.g., trade flows, supply chains, social ties).
- [ ] Automated estimation pipeline for direct vs. indirect (spillover) treatment effects mapped from the discovered graphs.

## Data & License Dependencies
* **Data Class:** Public Empirical Networks (e.g., trade/mobility graphs) + Synthetic Benchmark Generators.
* **License:** MIT License.


## Reviewer quick test run

A referee who wants to confirm the code runs end to end -- without the full multi-hour synthetic
sweep or the PDF pipeline -- can run the four steps below. They need **no downloaded data** (the
synthetic path generates its own) and finish in **~1-2 min on CPU**. They prove the pipeline
works; they do **not** reproduce any paper number (that is the full `## STEPS` below, at the real
epoch budget and seed count).

All four at once (stops at the first failure; logs to `results/logs_smoke/`):

```bash
bash src/open_nrb/smoke_test.sh
```

Or step by step:

```bash
# A. Unit tests (seconds). Each is a standalone PEP723 script; run all at once with pytest:
uv run python -m pytest src/open_nrb/tests -q

# B. Parameter count (instant, no data, no GPU) -- the Section 7 over/under-parameterization
#    check. Prints TOTAL PARAMETERS: 2133 for the baseline config.
uv run python -m open_nrb.evaluation.param_count

# C. One short synthetic run on the smoke config (epochs 150, 1 seed, N=16, T=8; CPU).
#    Writes results/synthetic_smoke/seed_0.pt. Reconstruction quality here is meaningless.
uv run python -m open_nrb.models.run_synthetic_experiment --config src/open_nrb/experiments/configs/synthetic_smoke.yaml --workers 1 --device cpu

# D. Post-hoc evaluation on the artifact C just wrote -- confirms the referee-check scripts load
#    and run (none of these retrain):
uv run python -m open_nrb.evaluation.seed_significance --synthetic-config src/open_nrb/experiments/configs/synthetic_smoke.yaml
uv run python -m open_nrb.evaluation.capped_ras_baseline --synthetic-config src/open_nrb/experiments/configs/synthetic_smoke.yaml
uv run python -m open_nrb.evaluation.plot_synthetic_network --config src/open_nrb/experiments/configs/synthetic_smoke.yaml --seed 0 --period -1
```

The real-data referee checks (`persistence_baseline.py`, the real-data `eda_*`/`make_plots`
steps) read `results/baseline/real_run_seed0.pt`, so they need the processed CSVs and one
real-data run (full `## STEPS` 3 and 5) first -- they are part of the full reproduction, not this
data-free smoke test.

## STEPS

### Before running anything on the remote (GPU) machine

This folder is kept in sync with the remote machine via Syncthing, not git push/pull -- so a
local edit isn't guaranteed to be on the remote side yet by the time you switch terminals and
run something. Confirm the sync has actually caught up first, or you'll rerun stale code:

1. Open the Syncthing web GUI (default `http://localhost:8384` on either machine) and check
   this project's folder state. It must read **"Up to Date"** -- not "Syncing (n items)" and
   not "Out of Sync". If it's still syncing, wait for it to settle before launching a run.

2. If you'd rather check from a terminal than the GUI (needs this folder's Syncthing folder ID,
   visible in the GUI under the folder's settings):

   ```bash
   syncthing cli show completion --folder <folder-id>
   ```

   (or, without the newer CLI subcommand, hit the REST API directly:
   `curl -s -H "X-API-Key: $ST_API_KEY" "http://localhost:8384/rest/db/completion?folder=<folder-id>"`)
   -- either way, look for `"completion": 100` before trusting the remote copy.

3. As a quick sanity check with no Syncthing tooling at all, compare one file's modification
   time/hash on both sides (e.g. the script you just edited) -- `ls -la
   src/open_nrb/models/run_synthetic_experiment.py` or `sha256sum` on each machine should match
   once sync is done.

### Running the PDF Downloader

To download all monthly statistical reports from the Nepal Rastra Bank (NRB) website into `src/open_nrb/data/raw_pdfs/`, run the following commands from the project root:

1. **Make the downloader script executable:**

   ```bash
   chmod +x src/open_nrb/utils/pdf_downloader.py
   ```

2. **Execute the script using uv:**

   ```bash
   uv run src/open_nrb/utils/pdf_downloader.py
   ```

   (Alternatively, since it includes a `PEP 723` inline shebang, it can also be run directly as: `./src/open_nrb/utils/pdf_downloader.py`)

   Downloads run across `--workers` concurrent threads (default 4) -- each report's detail
   page and PDF are fetched independently, so this is network-I/O-bound work threads handle
   fine. Keep this modest: it hits NRB's own server, and a high worker count is
   indistinguishable from hammering it. Pass `--workers 1` for the original, strictly
   sequential, most-polite behavior.

3. **Extract the PDFs:**

   ```bash
   uv run src/open_nrb/utils/pdf_extract.py --all
   ```

   Add `--workers N` to parse PDFs across N worker processes (default: `min(n_files,
   cpu_count)`). Unlike the downloader, this is CPU-bound (each PDF's
   pypdf/pdfplumber parse is independent, no shared state), so it scales close to linearly
   with cores -- use `--workers 1` for the original sequential, ordered-log-line behavior.

4. **Run the synthetic experiment:**

   ```bash
   uv run src/open_nrb/models/run_synthetic_experiment.py --config src/open_nrb/experiments/configs/synthetic_baseline.yaml --workers 4
   ```

   Results are saved in `src/open_nrb/results/synthetic_baseline/`.

   `--workers N` runs the config's seeds across N worker processes instead of one after
   another (default: `min(n_seeds, cpu_count)`). Seeds are fully independent (fresh model,
   fresh RNG, each writes its own `seed_{seed}.pt`), so this is the same embarrassingly-
   parallel case as PDF extraction above. Omit it, or pass `--workers 1`, for the original
   sequential, ordered-print behavior.

   On a shared or small GPU, pass `--device cpu` when using `--workers > 1`: each worker
   process opens its own CUDA context (several hundred MB-1GB of VRAM apiece) on top of
   whatever else is already resident on that device, and this model's tensors are tiny enough
   (`hidden_dim=16`, `N<=~24`) that CPU workers parallelize better than GPU ones fighting over
   one device. If you see `CUDA out of memory` with multiple workers, that's this -- rerun
   with `--device cpu` or a smaller `--workers`.

5. **Run the real-data experiment (baseline config):**

   ```bash
   uv run src/open_nrb/models/run_real_experiment.py --config src/open_nrb/experiments/configs/baseline.yaml
   ```

   Results are saved in `src/open_nrb/results/baseline/`.

6. **Run the ablation study:**

   ```bash
   uv run src/open_nrb/models/run_real_experiment.py --config src/open_nrb/experiments/configs/ablation_no_kl_pressure.yaml
   ```

   Results are saved in `src/open_nrb/results/ablation_no_kl_pressure/`.

   Steps 5 and 6 don't depend on each other (different config, own output dir, no shared
   state) -- there's no `--workers` flag here since each is already a single process, but
   they can be launched as two background jobs to run at the same time, e.g. `uv run
   .../run_real_experiment.py --config .../baseline.yaml & uv run .../run_real_experiment.py
   --config .../ablation_no_kl_pressure.yaml & wait`.

7. **Run evaluation -- compare two config variants side by side:**

   ```bash
   uv run src/open_nrb/evaluation/compare_configs.py --config-a src/open_nrb/experiments/configs/baseline.yaml --config-b src/open_nrb/experiments/configs/ablation_no_kl_pressure.yaml
   ```

   Results are saved in `src/open_nrb/results/baseline_vs_ablation_no_kl_pressure/evaluation/`.

8. **Run evaluation -- reviewer pre-mortem EDA + network-structure/systemic-risk checks:**

   ```bash
   uv run src/open_nrb/evaluation/eda_model_checks.py
   ```

   Results are saved in `src/open_nrb/results/baseline/evaluation/` (real data) and
   `src/open_nrb/results/synthetic_baseline/evaluation/` (synthetic), or under whichever
   `--config`/`--synthetic-config` was passed.

9. **Run evaluation -- survivorship-bias EDA:**

    ```bash
    uv run src/open_nrb/evaluation/eda_survivorship.py
    ```

    Results are saved in `src/open_nrb/results/{csv_stem}/evaluation/` (namespaced by the input
    CSV's stem, e.g. `nrb_balance_sheet`, since this script has no `--config`).

10. **Run evaluation -- figures:**

    ```bash
    uv run src/open_nrb/evaluation/make_plots.py
    ```

    Results are saved in `src/open_nrb/results/baseline/evaluation/` (real data) and
    `src/open_nrb/results/synthetic_baseline/evaluation/` (synthetic), or under whichever
    `--config`/`--synthetic-config` was passed.

11. **Run evaluation -- seed-significance tests (synthetic only):**

    ```bash
    uv run src/open_nrb/evaluation/seed_significance.py
    ```

    Results are saved in `src/open_nrb/results/synthetic_baseline/evaluation/`, or under
    whichever `--synthetic-config` was passed. With `--compare-config`, the between-config
    comparison is saved separately under `src/open_nrb/results/{stem_a}_vs_{stem_b}/evaluation/`.

12. **Run evaluation -- qualitative synthetic network plot (ground truth vs. reconstruction):**

    ```bash
    uv run src/open_nrb/evaluation/plot_synthetic_network.py --seed 0 --period -1
    ```

    Reads only the artifact `run_synthetic_experiment.py` already saved (`seed_{seed}.pt`) --
    never retrains. Results are saved in
    `src/open_nrb/results/{config_stem}/evaluation/network_seed{seed}_period{period}.png`, or
    under whichever `--config`/`--out` was passed.

13. **Run evaluation -- real-panel EDA (asset scale/coverage + interbank marginals):**

    ```bash
    uv run src/open_nrb/evaluation/eda_real_panel.py --csv src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv
    ```

    A single run produces both plot groups (no separate invocations needed):
    - **Asset scale & coverage** -- a bank-size/reporting-coverage heatmap, reporting coverage
      over time (per-period line, answers the §10a reviewer pre-mortem discard-sample-bias
      item), a size-rank bump chart (the cheapest model-free check for "is there a persistent
      core"), and size distribution by year (log-scale boxplot + skew/kurtosis).
    - **Interbank marginals** -- zero-activity rate by year (checks real data against
      synthetic.py's zero-inflated log-normal DGP assumption), log-distribution of nonzero r/c
      by year, an r-vs-c net lender/borrower scatter, a net-position-by-year heatmap, and
      interbank activity vs. bank size (tests whether size is a good proxy for core-ness).

    Results are saved in `src/open_nrb/results/eda/`, or under whichever `--out-dir` was
    passed (not namespaced by CSV stem, unlike `eda_survivorship.py`).

### Referee checks (paper revision)

These five artifacts were added for the paper revision and back the claims in draft §5.2, §5.4,
§6.5 and §7. The two configs are one-variable clones of `sweep_lambda_mi_0.yaml` (the `lambda_mi=0`
control), so `seed_significance.py --compare-config` against that config is a clean ablation.

14. **Single-block (K=1) and non-SBM ablations** -- train the two variant configs, then compare
    each against the `lambda_mi=0` control with a paired test across shared seeds:

    ```bash
    uv run python -m open_nrb.models.run_synthetic_experiment --config src/open_nrb/experiments/configs/synthetic_single_block.yaml --workers 5
    uv run python -m open_nrb.evaluation.seed_significance --synthetic-config src/open_nrb/experiments/configs/sweep_lambda_mi_0.yaml --compare-config src/open_nrb/experiments/configs/synthetic_single_block.yaml

    uv run python -m open_nrb.models.run_synthetic_experiment --config src/open_nrb/experiments/configs/synthetic_non_sbm.yaml --workers 5
    uv run python -m open_nrb.evaluation.seed_significance --synthetic-config src/open_nrb/experiments/configs/sweep_lambda_mi_0.yaml --compare-config src/open_nrb/experiments/configs/synthetic_non_sbm.yaml
    ```

    `synthetic_single_block.yaml` sets `model.num_blocks: 1` (the generator stays K=2), isolating
    what the block channel adds. `synthetic_non_sbm.yaml` makes the generator's blocks homogeneous
    (a degree-corrected fitness model, no communities), testing robustness to misspecification.
    On an 8 GB card run them one after another at `--workers 5` (5 CUDA contexts ≈ 3 GB); the two
    `seed_significance` calls write the same `results/sweep_lambda_mi_0/evaluation/` files, so never
    launch them at the same instant. Results: `results/{synthetic_single_block,synthetic_non_sbm}/`
    and the comparison under `results/sweep_lambda_mi_0_vs_{stem}/evaluation/`.

15. **Capped-RAS decomposition** (draft §5.2) -- post-hoc, reads the saved synthetic `seed_*.pt`
    and never retrains. Splits the synthetic win over RAS into the cap contribution vs the
    learned-prior contribution (uncapped RAS → capped RAS → model), with paired t-tests:

    ```bash
    uv run python -m open_nrb.evaluation.capped_ras_baseline --synthetic-config src/open_nrb/experiments/configs/sweep_lambda_mi_0.yaml
    ```

    Results: `results/sweep_lambda_mi_0/evaluation/`.

16. **Persistence baseline** (draft §6.5) -- post-hoc on the real-data artifact; reprojects the
    previous period's $\hat W_{t-1}$ through the capped projection onto period $t$'s margins to get
    a temporal-stability floor, and reports the model's stability against it:

    ```bash
    uv run python -m open_nrb.evaluation.persistence_baseline --config src/open_nrb/experiments/configs/baseline.yaml
    ```

    Needs `results/baseline/real_run_seed0.pt` from step 5. Results: `results/baseline/evaluation/`.

17. **Parameter count** (draft §7) -- instant, no data, no GPU. Reports
    `sum(p.numel() ...)` for `ReconstructionModel` (2,133 for the baseline config) against N×T,
    the marginal observations 2NT, and the latent entries N(N−1)T:

    ```bash
    uv run python -m open_nrb.evaluation.param_count
    ```

    Prints to stdout only (reads nothing, writes nothing); safe to run alongside any training job.

## Retuning kl_free_bits / lambda_mi

Only judge `lambda_mi`/`kl_free_bits`/`kl_beta_max` against synthetic `block_acc` (there's no
ground truth on real data to judge them against).

**Step 1: Run the lambda_mi sweep on synthetic data, multi-seed each config.**

```bash
for cfg in synthetic_baseline sweep_lambda_mi_0 sweep_lambda_mi_015 sweep_lambda_mi_03; do
  uv run src/open_nrb/models/run_synthetic_experiment.py \
    --config "src/open_nrb/experiments/configs/${cfg}.yaml" --workers 5
done
```

Each writes `results/{cfg}/multiseed_summary.json`, with an `aggregates` block shaped like:

```json
{
  "aggregates": {
    "model_rel_error":    {"mean": 0.4219, "std": 0.0357},
    "baseline_rel_error": {"mean": 0.5657, "std": 0.0257},
    "block_acc":          {"mean": 0.7646, "std": 0.0646},
    "trivial_baseline":   {"mean": 0.7000, "std": 0.0000},
    "gap":                {"mean": 0.0646, "std": 0.0791},
    "n_beats_baseline": 5,
    "n_beats_trivial": 3,
    "n_seeds": 5
  }
}
```

**Step 2: Collect each config's `gap` mean/std and `n_beats_trivial` into one table.**

| `lambda_mi` | gap mean | gap std | `n_beats_trivial`/5 |
|---|---|---|---|
| 0.00 | 0.0781 | 0.0774 | 3/5 |
| 0.05 | 0.0688 | 0.0823 | 3/5 |
| 0.15 | 0.0682 | 0.0790 | 3/5 |
| 0.30 | 0.0656 | 0.0814 | 3/5 |

Result: `lambda_mi=0.00` has the highest gap mean, `n_beats_trivial` is tied 3/5 across every
value, and gap mean is smaller than gap std in all four rows -- no nonzero value clearly beats
the control. Per Step 4, use `lambda_mi=0`.

**Step 3 (optional): Check pairwise significance against the `lambda_mi=0` control.**
`compare_configs.py` only diffs point estimates (no hypothesis test) -- use
`seed_significance.py` with `--compare-config` for an actual paired test across shared seeds,
once per nonzero candidate:

```bash
for cfg in synthetic_baseline sweep_lambda_mi_015 sweep_lambda_mi_03; do
  uv run src/open_nrb/evaluation/seed_significance.py \
    --synthetic-config src/open_nrb/experiments/configs/sweep_lambda_mi_0.yaml \
    --compare-config "src/open_nrb/experiments/configs/${cfg}.yaml"
done
```

(`synthetic_baseline.yaml` is the `lambda_mi=0.05` cell -- see Step 1's table. Add
`--synthetic-config sweep_lambda_mi_0.yaml` with no `--compare-config` first to also get the
*within*-config check, i.e. whether the control itself beats the RAS/Sinkhorn and trivial
baselines.)

**Step 4: Pick a value.** Choose the smallest `lambda_mi` with the highest, most consistent gap
(mean clearly bigger than std, most seeds beating trivial). If no nonzero value clearly beats the
`lambda_mi=0` control, use `lambda_mi=0`.

**Step 5 (if needed): Sweep `kl_free_bits`/`kl_beta_max` the same way.** Copy a
`sweep_lambda_mi_*.yaml` file, change `kl_free_bits`/`kl_beta_max` instead, multi-seed, repeat
Steps 1-4.

**Step 6: Write the chosen values into `baseline.yaml` and `ablation_no_kl_pressure.yaml`.**

**Step 7: Re-run the real-data experiments with the updated configs.**

```bash
uv run src/open_nrb/models/run_real_experiment.py --config src/open_nrb/experiments/configs/baseline.yaml
uv run src/open_nrb/models/run_real_experiment.py --config src/open_nrb/experiments/configs/ablation_no_kl_pressure.yaml
```

## Parallel Processing

Audited every script in the project for independent-work loops worth parallelizing. Three
qualify, and each got a `--workers` (or equivalent) flag -- see the matching step above for
the command:

| Script | Loop parallelized | Mechanism | Why |
|---|---|---|---|
| `utils/pdf_downloader.py` | report detail pages/PDFs within a category page | `ThreadPoolExecutor` | network I/O-bound (requests release the GIL); default 4 workers to stay polite to NRB's server |
| `utils/pdf_extract.py` | PDF files (`--all`) | `ProcessPoolExecutor` | CPU-bound pypdf/pdfplumber parsing, each file fully independent |
| `models/run_synthetic_experiment.py` | seeds (`cfg.seeds`) | `ProcessPoolExecutor` | each seed is a fresh model/RNG, fully independent training run |

`run_real_experiment.py` (steps 5/6) has no internal loop to parallelize -- one process trains
one config on one seed -- but the baseline and ablation runs are independent of each other, so
they can be launched as two concurrent background jobs instead of back-to-back (see step 6).

Everything else (`compare_configs.py`, `eda_model_checks.py`, `eda_survivorship.py`,
`make_plots.py`, `seed_significance.py`, `eda_real_panel.py`, `plot_synthetic_network.py`,
`capped_ras_baseline.py`, `persistence_baseline.py`, `param_count.py`) only
reads small already-saved artifacts (JSON summaries, a handful of `(T, N, N)` tensors at this
project's scale: `N<=~24` banks, a few hundred periods) and does one fast pass of vectorized
torch/numpy math or plotting -- there's no loop in them long enough for process/thread overhead
to pay for itself, so they were left sequential.

On the CPU-bound cases (`pdf_extract.py`, `run_synthetic_experiment.py`), `--workers`
processes compete for the same cores, so don't set it above `os.cpu_count()`; on a single GPU,
`run_synthetic_experiment.py` uses a "spawn" worker pool rather than the Linux default "fork"
-- a forked child inherits the parent's CUDA driver state, but CUDA contexts cannot survive
fork (`CUDA error: initialization error` the moment a worker touches `torch.cuda.*`), so spawn
is required there, not just a style choice. Its worker processes also share one GPU (CUDA
kernels time-slice across them, and each process's CUDA context alone costs several hundred
MB-1GB of VRAM) -- still usually a net win at this model's tiny scale (`hidden_dim=16`), since
per-epoch Python/host overhead, not device FLOPs, is the actual bottleneck, but don't expect
linear scaling there the way the CPU-only case gets.

