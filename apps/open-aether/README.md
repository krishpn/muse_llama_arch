# Open-Aether: Applied Causal Graph Discovery under Real-World Network Interference

## Project Overview

`open-aether` provides a practical framework for learning causal structures (DAGs/iDAGs) from empirical network data where unit interference and network dependence break standard i.i.d. assumptions. Designed for real-world observational settings, it recovers direct treatment and spillover structures from interconnected economic, social, and policy datasets.

## Target Audience & Applied Venues

* **Primary Target Audience:** Empirical ca
usal inference researchers, applied econometricians, and domain scientists evaluating network spillovers in venues such as **AISTATS (Applied Track)**, **ICML (Applications)**, and applied economics journals.

* **Intended Impact:** Enabling practitioners to discover validated causal graphs from real-world networked data without making unrealistic unit-independence or no-interference (SUTVA) assumptions.

## Key Applied Audit & Replicability Criteria

- [ ] Empirical validation suite testing graph recovery robustness under non-Gaussian, real-world noise distributions and partial network observability.
- [ ] Semi-synthetic simulation engine benchmarking Structural Hamming Distance (SHD) and False Discovery Rate (FDR) on empirical graph topologies (e.g., trade flows, supply chains, social ties).
- [ ] Automated estimation pipeline for direct vs. indirect (spillover) treatment effects mapped from the discovered graphs.

## Data & License Dependencies
* **Data Class:** Public Empirical Networks (e.g., trade/mobility graphs) + Synthetic Benchmark Generators.
* **License:** MIT License.


