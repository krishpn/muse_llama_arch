# Open-Aether: Application of Machine learning in understanding the

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
