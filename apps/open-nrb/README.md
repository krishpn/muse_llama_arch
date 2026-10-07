# Open-Aether: Application of Machine learning in understanding the 

## Project Overview

`open-nrb` 
## Target Audience & Applied Venues

* **Primary Target Audience:** NRB Spring Conference 2027 April 2-3, 2027

* **Intended Impact:** 
s
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

2. Execute the script using uv:

```bash
uv run src/open_nrb/utils/pdf_downloader.py
```

(Alternatively, since it includes a `PEP 723` inline shebang, it can also run it directly as: `./src/open_nrb/utils/pdf_downloader.py`)


