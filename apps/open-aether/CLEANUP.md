# open_aether cleanup log (2026-11-15)

Pre-rerun cleanup of the local mirror of `open-aether/open_aether/*`. Everything will
be re-run on the remote box; this records WHAT changed and WHY so no reasoning is lost.
Invocation unchanged: `uv run python -m open_aether.<pkg>.<module>` from the repo root.

## New: single source of truth

- **`_spillover_common.py`** (package root) — config constants (`GRAPH_YEAR=2022`,
  `WLO,WHI=69,81`, `PRE_LO,PRE_HI=60,67`, `CUT=75`, `OUT_LO,OUT_HI=76,81`, `SEED=0`,
  `NEVER=9999`), `per_index`, path auto-detect (`BASE`), and the shared loaders:
  `load_panel`, `firm_frame`, `load_canonical_graph`, `two_hop_only`, `align`,
  `load_edges`, `director_memberships`, `add_other_min`, `ri_slope`, `strata_labels`.
  Run it as a self-test: `python -m open_aether._spillover_common` — must print
  the canonical graph (4,392 nodes / 12,757 edges / mean deg 5.8) and observed
  RI slope **+0.02978**. VALIDATE THIS FIRST on the box before trusting the refactor.
  (Validated 2026-11-15: panel 354,580 rows / 17,807 firms; graph 4,392 / 12,757 /
  5.8; in-window 1,612; slope +0.02978 exact.)

## Deletions

- **All `__pycache__/` + `*.pyc`** — build artifacts, regenerated on run. Zero info.
- **`data/mongo_test.py`** — one-off ad-hoc Mongo probe (LSEG `trandate` span, CIK
  type distribution). Its findings are already captured in memory
  (`project_insider_spillover`: LSEG span 1983→2025; CIK mixed int/string hazard).
  Not part of any pipeline. Removed as scratch. Re-derivable from `mongo_test` git
  history / the recorded facts if the CRSP-bridge work resumes.
- **`models/event_study_panel.py`** — DELETED as a file, but nothing lost: it was
  really FIVE concatenated scripts. Split (see below).

## Restructuring: `event_study_panel.py` split

The 383-line `models/event_study_panel.py` was the panel BUILDER (lines ~1-89)
concatenated with four abandoned volume event-study scripts (lines ~93-384). The
stale `data/__pycache__/event_study_panel.pyc` + `event_study.pyc` confirm the
builder's natural home is `data/`. Split into:

- **`data/build_panel.py`** — the panel builder, verbatim logic. Writes
  `_parsed/panel_firm_quarter.parquet`, the upstream dependency of every model.
  RUN THIS FIRST. (It is a data step, not a model — hence `data/`.)
- **`models/event_study_volume.py`** — the four volume event-studies, kept verbatim
  under a DEPRECATED header. These produced the early "-0.42, ~40% lower volume,
  robust" result that did NOT survive (artifact of the contaminated 2023Q1 baseline
  + Goodman-Bacon cohort weighting; collapsed to -0.133 ns once fixed). The paper's
  volume verdict is the NULL from `ht_spillover.py` (draft §6). Retained only for
  provenance of that conclusion; not in the live flow.

## Refactor: live models now import `_spillover_common`

All five slimmed to source config/graph/loaders/`ri_slope` from the shared helper.
Behavior-preserving EXCEPT the two deliberate corrections noted:

- **`decomposition.py`** (§4 PRIMARY) — unchanged numerically. Uses
  `firm_frame(actcol="n_sales")` to preserve the published ladder exactly
  (observed +0.02978; 4×4 ≈ 93% homophily / 7% contagion, p=0.028).
- **`adoption_diffusion.py`** (§5.1) — **CORRECTED GRAPH**: now on the canonical
  4,392-node 2022 graph via `load_canonical_graph` instead of the earlier denser
  multi-year graph. This fixes the documented drift (old mean 2-hop deg ≈185). The
  module prints mean `deg2` to document why the Bramoullé exclusion restriction
  fails. **Also fixed deprecated pyfixest IV syntax**:
  `... | per | nbr_frac ~ z2_frac`  →  `... + [nbr_frac ~ z2_frac] | per`.
  (Numbers will shift from the old run because the graph changed — this is intended;
  re-run and update draft §5.1.)
- **`ht_spillover.py`** (§6) — **CORRECTED GRAPH**: moved to the canonical graph per
  draft §6 ("re-run HT on the canonical 4,392-node graph"). Outcome/propensity/
  exposure logic preserved; only the node set + alignment changed. Re-run, update §6.
- **`director_fe.py`** (§5.2) — now takes degree from the canonical graph (was the
  sparse panel `degree` snapshot, only 3,212 firms). `other_min` leave-one-out logic
  moved into shared `add_other_min`. Mechanics otherwise identical.
- **`director_placebo.py`** (§5.2) — same canonical-degree sourcing; `within_dir_coef`
  kept inline (needs the per-permutation adopt map). Immortal-time placebo unchanged.

## PEP 723 inline metadata (standalone-runnable)

Every `.py` now starts with a `#!/usr/bin/env -S uv run --script` shebang + a
PEP 723 `# /// script` block listing its REAL third-party deps (numpy/pandas/
scipy/pyarrow/pyfixest/matplotlib/pymongo/networkx as applicable — NOT the `torch`
template that was pasted in; none of these use torch). Files are `chmod +x`, so
each runs standalone:  `./models/decomposition.py`  or  `uv run --script models/decomposition.py`.

The five `models/` files import `open_aether._spillover_common`, which does not
resolve when a file is run as a loose script (Python puts the script's own dir on
`sys.path`, not the repo root). Each therefore carries a 2-line bootstrap:
`_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))`. This is a
no-op under the existing `uv run python -m open_aether.models.<x>` package
invocation (repo root already importable) and makes standalone runs work too. The
`data/` files and `_spillover_common.py` import no package code, so no shim.
`data/download_raw_zips_from_sec.sh` is bash — left untouched (no PEP 723).

## New: `descriptives/` reporting subpackage (2026-11-15)

A third module group parallel to `data/` (builds) and `models/` (estimators), for
PRESENTATION artifacts that carry no estimand — they describe the clustering that
`models/decomposition.py` then decomposes. Both import `_spillover_common` (same
canonical graph + panel as the models) and carry the PEP 723 header + the
`parents[2]` sys.path bootstrap. This is NOT `data/edl` (that is the raw Form 345
parser that BUILDS data) and NOT `models/` (no identification happens here).

- **`descriptives/table1.py`** — Table 1 summary stats: Panel A sample + graph
  structure, Panel B adoption status, Panel C characteristics by adoption status,
  Panel D adoption-by-quarter. Writes `table1_summary.md` + `.csv`.
- **`descriptives/network_desc.py`** — Figure 1 intro descriptives: degree CCDF,
  adoption S-curve, exposed-vs-unexposed hazard by quarter, spatial
  autocorrelation (Moran's I + permutation join-count z/p) of adoption on the
  interlock graph, and the transitivity/clustering diagnostic (global clustering
  coefficient, mean local clustering, and the share of common-neighbor pairs that
  are also directly linked — the direct read on the Bramoullé exclusion failure in
  §5.1). Writes `fig1_network_descriptives.png` + `desc_*.csv` (incl.
  `desc_transitivity.csv`).
- **`descriptives/powerlaw_fit.py`** — APPENDIX A (optional): Clauset-Shalizi-Newman
  discrete power-law fit of the degree sequence (α, xmin, KS D) plus likelihood-ratio
  comparisons vs lognormal/exponential. NOT load-bearing and the body makes no
  scale-free claim; this exists so the appendix can state honestly whether a power
  law fits at all (expected: inconclusive, max degree ~48 => short tail). Prefers the
  `powerlaw` package, falls back to a bare discrete MLE with a note if it is absent.
  Writes `desc_powerlaw.csv`.
- **`descriptives/network_plot.py`** — Figure 2 (NEW): node-link drawing of the
  interlock network's largest connected component, nodes colored by adoption status
  (in-window / pre / never) and sized by degree, force-directed (spring) layout.
  The ONLY non-moment figure — makes the clustering visible. Prints graph-level
  homophily anchors (attribute assortativity of ever-adopt, degree assortativity,
  adopter-adopter edge share vs the p^2 random-mixing expectation) so the picture is
  quantitative, not decorative. Uses networkx (unlike the others). Writes
  `fig2_network_layout.png` + `desc_network_plot.csv`. CAPTION MUST frame the
  clustering as the phenomenon the paper explains (homophily), NOT as contagion.
- **`descriptives/diffusion_panel.py`** — Figure 3 (NEW): diffusion trail / small
  multiples of CUMULATIVE adoption on a FIXED 2022 layout, one panel per snapshot
  quarter (2022Q4/2023Q2/2023Q4/2024Q4/2026Q2), each annotated with Moran's I (full
  graph, matches Fig 1d). Shows clustering is near-max at the first post-mandate
  quarter and FLAT after — the anti-contagion signature (common shock + homophily,
  not edge-propagation). Reuses the fixed layout across panels. Writes
  `fig3_diffusion_trail.png` + `desc_diffusion_snapshots.csv`. SAME caption caution.
- **`descriptives/_layout.py`** — shared layout helper: prefers igraph
  (`layout_fruchterman_reingold`, C-fast, cleaner clusters) and falls back to
  networkx `spring_layout`. Used by both network_plot and diffusion_panel. Install
  the fast path with `uv add python-igraph`.

(The pre-existing empty `evaluation/` and `utils/` dirs are untouched scaffolding.)

## Kept as-is (distinct purpose, not redundant)

- `data/edl_basics.py` — SEC Form 345 parser → `_parsed/edges/*` + `_parsed/tx/*`
  (the `parse_sec_insider` step). Upstream of everything. (Name is cryptic; a rename
  to `parse_sec_insider.py` would be clearer but changes the module path — deferred.)
- `data/firm_firm_edges_parser.py` — builds the yearly interlock graph
  `_parsed/interlock/firm_firm_edges.parquet`. Upstream of the canonical graph.
- `data/build_firm_bridge.py`, `data/build_coverage.py`, `data/load_sec_10b_mongo.py`,
  `data/download_raw_zips_from_sec.sh` — the raw-download + CIK↔PERMNO bridge +
  Mongo-load chain for the FUTURE abnormal-return outcome work. Not in the current
  decomposition pipeline but a real, separate workstream. Kept.

## Run order on the box (current pipeline)

Run each command on its OWN line, with NO trailing `# comment` — zsh reads a
parenthesized `~` in a comment (e.g. `(~-0.17)`) as a glob qualifier and aborts
the line (`zsh: unknown file attribute: ~`). All modules run from the repo root
(`apps/open-aether/`) as `uv run python -m open_aether.<pkg>.<module>`.

### Stage A — data build (one-time; rerun only if raw SEC data changes)

```bash
bash open_aether/data/download_raw_zips_from_sec.sh      # raw Form 345 zips -> extracted/
uv run python -m open_aether.data.edl_basics             # parse -> _parsed/edges/*, _parsed/tx/*
uv run python -m open_aether.data.firm_firm_edges_parser # yearly interlock graph -> _parsed/interlock/
uv run python -m open_aether.data.build_panel            # firm x quarter panel -> panel_firm_quarter.parquet
```

### Stage B — validation gate (run first every session)

```bash
uv run python -m open_aether._spillover_common           # self-test: expect 4,392 / 12,757 / 5.8, slope +0.02978
```

### Stage C — models (order matches the draft sections)

```bash
uv run python -m open_aether.models.decomposition        # §4 PRIMARY — homophily-vs-contagion ladder
uv run python -m open_aether.models.adoption_diffusion   # §5.1 — Bramoullé friends-of-friends IV diagnostic
uv run python -m open_aether.models.director_fe          # §5.2 — within-director FE hazard (the -0.18 artifact)
uv run python -m open_aether.models.director_placebo     # §5.2 — immortal-time placebo (proves -0.18 is mechanical)
uv run python -m open_aether.models.ht_spillover         # §6 SECONDARY — Horvitz-Thompson volume spillover (null)
```

### Deprecated / provenance only (not in the live flow)

```bash
uv run python -m open_aether.models.event_study_volume   # old volume event-studies; the "-0.42 artifact" story
```

### Stage D — descriptives / reporting (figures + Table 1; run after Stage B)

Presentation artifacts, NOT identification — these describe the clustering that
Stage C decomposes. They consume the same panel + canonical graph via
`_spillover_common`, so they only need Stage A + B to have run (independent of C).

```bash
uv run python -m open_aether.descriptives.table1         # Table 1 -> table1_summary.md + .csv
uv run python -m open_aether.descriptives.network_desc   # Fig 1 + transitivity -> fig1_network_descriptives.png (+ desc_*.csv)
uv run python -m open_aether.descriptives.powerlaw_fit   # Appendix A (optional) -> desc_powerlaw.csv
uv run python -m open_aether.descriptives.network_plot   # Fig 2 node-link -> fig2_network_layout.png (+ desc_network_plot.csv)
uv run python -m open_aether.descriptives.diffusion_panel # Fig 3 diffusion trail -> fig3_diffusion_trail.png (+ desc_diffusion_snapshots.csv)
```

Notes: Stage A is the slow one-time build; once `panel_firm_quarter.parquet`
exists you only ever need B -> C. Stage C modules are mutually independent (each
runs alone after Stage B) but read top-to-bottom with the draft. VALIDATED
2026-11-15: all of Stage B + C ran clean on the box under the refactored code.

## What `edl_basics.py` does (the Stage A parser, step 2)

`data/edl_basics.py` is the Form 345 parser — upstream of EVERYTHING. It walks the
`*_form345` quarter directories under `extracted/` and, per quarter, reads three
TSVs and writes two parquet families. Key points:

- **Robust TSV read** — SEC TSVs contain stray quotes and literal empty strings
  that must NOT be coerced to NaN, so the reader disables quoting and NA inference:
  ```python
  def rd(path):
      return pd.read_csv(path, sep="\t", dtype=str, quoting=3,
                         keep_default_na=False, na_values=[], on_bad_lines="warn")
  ```
- **Join key** is `ACCESSION_NUMBER` across all tables. The SUBMISSION slice carries
  the firm identity + the treatment flag; `AFF10B5ONE` is read defensively because
  it is absent in pre-2023 quarters:
  ```python
  subk = sub[["ACCESSION_NUMBER","FILING_DATE","PERIOD_OF_REPORT","DOCUMENT_TYPE",
              "ISSUERCIK","ISSUERNAME","ISSUERTRADINGSYMBOL"]].copy()
  subk["AFF10B5ONE"] = sub["AFF10B5ONE"] if "AFF10B5ONE" in sub.columns else ""
  subk["QUARTER"]    = qid
  ```
- **Relationship flags** are substring tests on the comma-joined
  `RPTOWNER_RELATIONSHIP` tokens (`Director`/`Officer`/`TenPercentOwner`, sometimes
  glued to `Other`):
  ```python
  rel = own["RPTOWNER_RELATIONSHIP"].str.lower().fillna("")
  own["is_director"] = rel.str.contains("director")
  own["is_officer"]  = rel.str.contains("officer")
  own["is_tenpct"]   = rel.str.contains("tenpercent|ten percent|10%", regex=True)
  ```
- **Output (1) — edges** (`_parsed/edges/<qid>.parquet`): one row per (owner, filing),
  REPORTINGOWNER joined to the SUBMISSION slice. This is the insider<->firm bipartite
  graph that `firm_firm_edges_parser.py` projects into interlocks.
- **Output (2) — tx** (`_parsed/tx/<qid>.parquet`): NONDERIV_TRANS enriched with issuer,
  owner, owner-count, and the 10b5-1 flag — the transaction table behind the volume
  outcomes (`n_sales`/`sale_shr`) in `build_panel.py`.
- It also prints two diagnostics that first established the key facts in memory: the
  `RPTOWNER_RELATIONSHIP` encoding histogram, and the `AFF10B5ONE` value counts per
  quarter for 2022q4..2024q4 (showing the flag empty pre-2023, ramping 2023q1, reliable
  2023q2+).

