# Open-Nexus: GPU-Accelerated Continuous DAG Optimization under Network Interference

## Project Overview
`open-nexus` develops $O(d^2)$ low-rank continuous optimization algorithms for large-scale differentiable DAG discovery on graphs exceeding $D > 1,000$ variables using custom CUDA kernels and streaming tensor evaluation.

## Target Audience & Publication Venues
* **Primary Target Audience:** High-performance machine learning systems engineers, computational scientists, and readers of **ICML**, **NeurIPS (Systems Track)**, and the **Journal of Computational Science**.
* **Intended Impact:** Eliminating the $O(d^3)$ matrix exponential bottleneck in continuous DAG search via randomized low-rank trace estimators integrated into Triton/gRPC pipelines.

## Key Audit & Replicability Criteria
- [ ] CUDA C++ extensions in `csrc/` buildable via standard PyTorch `setup.py`.
- [ ] Wall-clock speedup and GPU memory consumption benchmarks against GraN-DAG and DCDI.
- [ ] Docker container environment (`docker/Dockerfile`) pinning exact CUDA/PyTorch driver versions.

## Data & License Dependencies
* **Data Class:** 100% Synthetic & Open Access.
* **License:** MIT License.