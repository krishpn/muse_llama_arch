# Bayesian Network Reconstruction of Interbank Exposures (Nepal)

## Project Overview

`open-nrb`

## Target Audience & Applied Venues

* **Primary Target Audience:** NRB Spring Conference 2027 April 2-3, 2027

* **Intended Impact:**

## Key Applied Audit & Replicability Criteria

- [ ] Empirical validation suite testing graph recovery robustness under non-Gaussian, real-world noise distributions and partial network observability.
- [ ] Semi-synthetic simulation engine benchmarking Structural Hamming Distance (SHD) and False Discovery Rate (FDR) on empirical graph topologies (e.g., trade flows, supply chains, social ties).
- [ ] Automated estimation pipeline for direct vs. indirect (spillover) treatment effects mapped from the discovered graphs.

## Data & License Dependencies
* **Data Class:** Public Empirical Networks (e.g., trade/mobility graphs) + Synthetic Benchmark Generators.
* **License:** MIT License.


## STEPS

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

3. **Extract the PDFs:**

   ```bash
   uv run src/open_nrb/utils/pdf_extract.py
   ```

4. **Run the synthetic experiment:**

   ```bash
   uv run src/open_nrb/models/run_synthetic_experiment.py --config src/open_nrb/experiments/configs/synthetic_baseline.yaml
   ```

   Results are saved in `src/open_nrb/results/synthetic_baseline/`.

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

7. **Run evaluation -- compare two config variants side by side:**

   ```bash
   uv run src/open_nrb/evaluation/compare_configs.py --config-a src/open_nrb/experiments/configs/baseline.yaml --config-b src/open_nrb/experiments/configs/ablation_no_kl_pressure.yaml
   ```

   Results are saved in `src/open_nrb/results/baseline_vs_ablation_no_kl_pressure/evaluation/`.


uv run src/open_nrb/evaluation/compare_configs.py --config-a src/open_nrb/experiments/configs/baseline.yaml --config-b src/open_nrb/experiments/configs/ablation_no_kl_pressure.yaml --seed-a 0 --seed-b 0  

8. **Run evaluation -- network-structure/systemic-risk checks:**

   ```bash
   uv run src/open_nrb/evaluation/eda_network_structure.py
   ```

   Results are saved in `src/open_nrb/results/baseline/evaluation/` (real data) and
   `src/open_nrb/results/synthetic_baseline/evaluation/` (synthetic), or under whichever
   `--config`/`--synthetic-config` was passed.

9. **Run evaluation -- reviewer pre-mortem EDA:**

   ```bash
   uv run src/open_nrb/evaluation/eda_reviewer_questions.py
   ```

   Results are saved in `src/open_nrb/results/baseline/evaluation/`, or under whichever
   `--config` was passed.

10. **Run evaluation -- survivorship-bias EDA:**

    ```bash
    uv run src/open_nrb/evaluation/eda_survivorship.py
    ```

    Results are saved in `src/open_nrb/results/{csv_stem}/evaluation/` (namespaced by the input
    CSV's stem, e.g. `nrb_balance_sheet`, since this script has no `--config`).

11. **Run evaluation -- figures:**

    ```bash
    uv run src/open_nrb/evaluation/make_plots.py
    ```

    Results are saved in `src/open_nrb/results/baseline/evaluation/` (real data) and
    `src/open_nrb/results/synthetic_baseline/evaluation/` (synthetic), or under whichever
    `--config`/`--synthetic-config` was passed.

12. **Run evaluation -- seed-significance tests (synthetic only):**

    ```bash
    uv run src/open_nrb/evaluation/seed_significance.py
    ```

    Results are saved in `src/open_nrb/results/synthetic_baseline/evaluation/`, or under
    whichever `--synthetic-config` was passed. With `--compare-config`, the between-config
    comparison is saved separately under `src/open_nrb/results/{stem_a}_vs_{stem_b}/evaluation/`.


13. Run a synthetic plot generator :

   ```bash
   uv run src/open_nrb/evaluation/plot_synthetic_network.py --seed 0 --period -1
   ```
   Results are saved in `src/open_nrb/results/synthetic_baseline/evaluation/`


14.  uv run src/open_nrb/evaluation/eda_bank_scale_coverage.py --csv src/open_nrb/data/processed/monthly_statistical_bfis/nrb_balance_sheet.csv

- Reporting coverage over time (plot_reporting_coverage_over_time) — a plain line of "# banks reporting per period" at full monthly resolution (using the BS_MONTHS ordering already defined in data_loader.py). The heatmap only shows coverage
  gaps implicitly, at yearly resolution, via gray-cell density; this makes the coverage trend itself the direct subject of a chart — directly useful for the §10a reviewer pre-mortem item 5 (discard-sample bias) write-up.

- Size-rank bump chart (plot_size_rank_bump_chart) — each bank's rank by TOTAL ASSETS per year (rank 1 = biggest) as lines, for whichever banks were ever in the top 12. This is the one I'd actually prioritize: it's the cheapest, model-free
  way to check "is there a persistent core" — if a handful of banks hold the top ranks every year, that's independent evidence for the core/periphery story the model is supposed to recover, worth having before you trust a trained block
  assignment.


  15. uv run src/open_nrb/evaluation/eda_bank_scale_coverage.py --csv src/open_nrb/data/processed/monthly_statisti;3D;3Dcal_bfis/nrb_balance_sheet.csv

  -  Zero-activity rate by year — checks whether real data's zero-inflation matches synthetic.py's log-normal × Bernoulli-existence DGP assumption.
  - Log-distribution of nonzero r/c by year — stacked boxplots (own panel each, not dual-axis).
  - r vs. c scatter — net lender/borrower split per bank-year, colored with the project's existing red=positive/blue=negative convention (reused from the W_hat - W_true heatmap).
  - Net position ranked bar chart — each bank's time-averaged net lender/borrower status.
  - Interbank activity vs. size — reuses _load_total_assets from eda_bank_scale_coverage.py (same cross-script-import pattern eda_network_structure.py already uses) to test whether size is a good proxy for core-ness.


