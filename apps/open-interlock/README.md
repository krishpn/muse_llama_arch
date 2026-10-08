# Open-Interlock: Corporate Governance Networks, Executive Interlocks, and Peer Effects

## Project Overview
`open-interlock` estimates non-linear peer spillovers, technology adoption dynamics, and corporate behavior across executive interlock networks while addressing homophily and unobserved network confounding.

## Target Audience & Publication Venues

* **Primary Target Audience:** Applied microeconomists, corporate finance scholars, network scientists, and readers of the **Quarterly Journal of Economics (QJE)**, **Journal of Financial Economics (JFE)**, and **Network Science**.
* **Intended Impact:** Establishing causal spillover identification across board ties without imposing restrictive linear-in-means assumptions or strict SUTVA requirements.

## Key Audit & Replicability Criteria
- [ ] **Dual Execution Engine:** 100% executable out-of-the-box using synthetic BoardEx/Compustat schema generators.
- [ ] **Proprietary Data Audit:** Clear boundary separation for BoardEx and WRDS/Compustat datasets via `.env` configuration.
- [ ] Empirical robustness checks (degree heterogeneity, endogenous tie formation sensitivity).

## Data & License Dependencies
* **Data Class:** Proprietary Corporate Databases (BoardEx, WRDS/Compustat) + Public Synthetic Generator.
* **License:** MIT License (Code only; raw data excluded).