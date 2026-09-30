# Open-Hydra: Latent Representation Learning & Mechanistic Steering in Networked Environments

## Project Overview

`open-hydra` extracts monosemantic causal circuits from high-dimensional latent world models (e.g., JEPA, VAEs) using Sparse Autoencoders (SAEs) to disentangle direct treatment dynamics from non-linear neighbor spillovers.

## Target Audience & Publication Venues
* **Primary Target Audience:** Deep learning representation theorists, mechanistic interpretability researchers, and readers of **NeurIPS **, **ICLR **, and **IEEE TPAMI**.
* **Intended Impact:** Demonstrating zero-shot counterfactual intervention steering on intermediate latent states without re-training the base environment model.

## Key Audit & Replicability Criteria
- [ ] Sparse Autoencoder (SAE) dictionary training scripts with logged $L_0$ sparsity vs. loss reconstruction trade-offs.
- [ ] Causal activation patching suite demonstrating directional steering on target unit outcomes.
- [ ] Open benchmark evaluation on high-dimensional interventional video/image streams.

## Data & License Dependencies
* **Data Class:** Open Access Generative & Physics Simulation Benchmarks.
* **License:** MIT License.