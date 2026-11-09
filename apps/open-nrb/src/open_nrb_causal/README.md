# Paper 2: A Floor That Binds but Doesn't Spread: Nepal's 2015 Bank Capital Mandate

## Project Overview

`open-nrb-causal`

Estimates how NRB's 2015 capital-floor mandate spilled over between Nepali commercial banks
through the interbank network. Design: a frozen pre-mandate dose x post DiD (Eq. A′) with an
Aronow-Samii exposure mapping + Horvitz-Thompson/GPS weights, a fiscal-year event study, and a
pooled local-linear RDD; inference by bank-clustered CR1 SEs and WCR wild-cluster bootstrap
(16 clusters). Full design: `paper2_causal_spillover.md`.

Depends one-directionally on Paper 1 (`open_nrb`): this package **imports** Paper 1's
data-loading code and **reads** Paper 1's already-saved real-data artifact
(`real_run_seed{seed}.pt`) for the reconstructed interbank network W_hat. It never retrains,
never copies Paper 1 files, and never edits Paper 1 source or data --
`tests/test_dependency_direction.py` enforces this.

## Layout

Same layout as Paper 1 (`src/open_nrb/`):

| Directory | Contents |
|---|---|
| `models/` | the estimation pipeline: `artifact_io`, `panel_alignment`, `variables`, `mandate_design`, `spillover`, `propensity`, `estimator`, `total_effect`, `robustness`, `experiment_config`, `run_causal_experiment` |
| `evaluation/` | scripts that read the pipeline's inputs or results and report on them: `audit_bs_extraction`, `check_paper1_interface`, `diagnose_capital_break`, `panel_window_checks`, `power_mde`, `reviewer_checks`, `run_event_study_report`, `plot_paid_up_figures`, plus the shared `report_format` helper |
| `experiments/configs/` | run configs (`main.yaml`, `baseline.yaml`, `clean_window.yaml`) |
| `results/` | every output; never written into Paper 1's `results/` |
| `tests/` | one PEP723 test script per `models/` module, plus the dependency-direction guardrail |

Imports follow Paper 1's convention: relative inside `models/` (`from .estimator import ...`),
`from ..models.x import ...` in `evaluation/`, and a script-mode fallback that puts the
`models/` directory on `sys.path`.

## Data & Known Limitations

Same inputs as Paper 1, from `src/open_nrb/data/processed/monthly_statistical_bfis/`:

| Input | Used for |
|---|---|
| `nrb_balance_sheet.csv` | dose (CAPITAL FUND / size), size, frozen pre-mandate controls; indirectly W_hat (Paper 1 built it from the interbank items) |
| `nrb_profit_and_loss.csv` | outcomes: net profit/loss, loan-loss provisions, loans written off, NIM (each divided by size) |
| `src/open_nrb/results/baseline/real_run_seed0.pt` | W_hat and the RAS baseline network |

**The extraction is taken as given, not repaired** (agreed limitation). `evaluation/audit_bs_extraction.py`
(step 2) shows the bulletins Baisakh 2070 - Falgun 2075 are clean, which covers the whole dose
window and transition. Many later bulletins have digit fragments, so line items stop summing to
their totals. The totals themselves always balance. The process, the error extent and the
consequences for the estimates are documented for reviewers in `paper2_causal_spillover.md`
§5a. The code never divides by or takes logs of bad values: size <= 0 becomes NaN, non-finite
regressor cells are dropped and counted, unidentified regressors are reported as NaN, and empty
event-study bins are dropped with a note.

## Reviewer quick test run

A referee who wants to confirm the pipeline runs end to end -- without the ~hours the full
`reproduce_paper.sh` spends on 9,999 wild-bootstrap draws per fit -- can run the four steps below
in **under a minute**. They use `smoke.yaml`, a clone of `main.yaml` with `n_bootstrap: 0`, so the
**point estimates are identical to the main specification** and only the wild p-values are reduced
(absent in C, 99 draws in D). This proves the code works; it does **not** reproduce a paper
p-value -- that is `reproduce_paper.sh` at 9,999 draws (see "Reproducing the paper" below).

Prerequisite: Paper 1 must already have written `src/open_nrb/results/baseline/real_run_seed0.pt`
and the processed CSVs must be present -- this is an empirical study, so unlike Paper 1's synthetic
smoke test it cannot run data-free.

```bash
# A. Tests (seconds).
uv run python -m pytest src/open_nrb_causal/tests -q

# B. Paper 1 interface check -- confirms the CSVs and the artifact line up (banks, periods,
#    marginals, cap, no NaN W_hat in the window) before any estimation.
uv run python -m open_nrb_causal.evaluation.check_paper1_interface --config src/open_nrb_causal/experiments/configs/smoke.yaml

# C. One causal run, CR1 SEs only (n_bootstrap: 0) -- seconds. Writes results/smoke_seed0.json.
uv run python -m open_nrb_causal.models.run_causal_experiment --config src/open_nrb_causal/experiments/configs/smoke.yaml

# D. Reviewer checks, quick bootstrap (99 draws instead of 9,999). Writes
#    results/reviewer_checks_smoke.md.
uv run python -m open_nrb_causal.evaluation.reviewer_checks --config src/open_nrb_causal/experiments/configs/smoke.yaml --n-boot 99
```

To reproduce actual paper numbers (9,999 draws, all three configs, figures, every §5b check),
run `bash src/open_nrb_causal/reproduce_paper.sh` -- the full, slow path documented under
"Reproducing the paper".

## STEPS

All commands run from the **apps/open-nrb repo root**: config paths are CWD-relative, the same
as Paper 1's `run_real_experiment.py`.

### Before running anything

1. **Paper 1 must already have produced its inputs**, i.e. Paper 1 README steps 3 (PDF
   extraction) and 5 (real-data experiment). Paper 2 reads those outputs; it never re-creates
   them.
2. **Sync check.** This folder reaches the run machine via Syncthing, so confirm the folder
   reads "Up to Date" before running (same procedure as Paper 1's README, "Before running
   anything on the remote (GPU) machine"). Otherwise you may run stale code.

### 1. Run the tests

```bash
uv run python -m open_nrb_causal.tests.test_estimator
uv run python -m open_nrb_causal.tests.test_mandate_design
uv run python -m open_nrb_causal.tests.test_variables
```

Each prints one PASS/FAIL line per case and an `OVERALL` line. The remaining files
(`test_artifact_io`, `test_panel_alignment`, `test_spillover`, `test_propensity`,
`test_dependency_direction`) run the same way, or all at once with
`uv run python -m pytest src/open_nrb_causal/tests -q` (`python -m` so pytest runs inside the
project environment; a bare `uv run pytest` can pick up a pytest without torch). Same non-pytest convention as Paper 1's tests: a
standalone PEP723 script with `CASES` + thin `def test_*()` wrappers.

### 2. Audit the balance-sheet extraction (reviewer-facing, read-only)

```bash
uv run python -m open_nrb_causal.evaluation.audit_bs_extraction
```

For every (bank, month, bulletin) sheet, it checks:
- top-level liabilities sum to their total;
- top-level assets sum to TOTAL ASSETS;
- capital sub-items sum to CAPITAL FUND;
- the two totals balance.

It also counts single-digit "fragment" values. The report prints by fiscal year and by bulletin.
Bulletins up to Chaitra 2074 have no TOTAL Liabilities row (`layout=numbering_reset`); for
those, liabilities are checked against TOTAL ASSETS and `balance` is n/a.
`--banks EBL SCBNL` restricts the audit to those banks. Output: `src/open_nrb_causal/results/bs_extraction_audit.csv`,
which step 5 also reads.

### 3. (Optional) Inspect one bank's raw rows around the capital-ratio break

```bash
uv run python -m open_nrb_causal.evaluation.diagnose_capital_break --bank EBL
```

Prints median size, CAPITAL FUND and their ratio by fiscal year, plus the raw balance-sheet
rows either side of each break. Diagnostic only; nothing downstream reads it.

### 4. Run the experiment

**4a. Main specification** (clean window to Falgun 2075 x quarterly outcomes x unweighted
TWFE; user decision 2026-11-03, `paper2_causal_spillover.md` §5b items 8 and 11a):

```bash
uv run python -m open_nrb_causal.models.run_causal_experiment --config src/open_nrb_causal/experiments/configs/main.yaml
```

**4b. Appendix: monthly de-cumulated outcomes**, full panel (`baseline.yaml`) and clean
bulletins only (`clean_window.yaml`, post window cut at Falgun 2075, the last bulletin before
the extraction corruption; a sample restriction, not a data repair). Both unweighted:

```bash
uv run python -m open_nrb_causal.models.run_causal_experiment --config src/open_nrb_causal/experiments/configs/baseline.yaml
uv run python -m open_nrb_causal.models.run_causal_experiment --config src/open_nrb_causal/experiments/configs/clean_window.yaml
```

Each run first prints diagnostics:
- the panel span and the pre/transition/post month counts (`truncated at post_window_end` for a
  clean-window config);
- the outcome construction and estimator weights;
- size cells set to NaN;
- cells below the floor by fiscal year;
- the dose_pre range;
- the P&L YTD-cumulation ratio (≈0.08 means cumulative, so keep `pnl_ytd_cumulative: true`).

Then, for each outcome, it prints:
- for quarterly configs, the **headline β**: the dose-only total effect from Baisakh 2070
  (`models/total_effect.py`, user decision 2026-11-03), with its pre-trend lead. The exposure
  regressors use the post-mandate W_hat, so they are bad controls for β;
- the pooled DiD (Eq. A′, the source of γ) under W_hat and under the RAS baseline (weights per
  `weighting`);
- the event study;
- the RDD;
- every dropped bin, regressor and cell.

Results are saved to `src/open_nrb_causal/results/<config>_seed<seed>.json`, kept separate
from Paper 1's `results/`.

`n_bootstrap` (default 9999) dominates the runtime; set it to 0 for a quick CR1-only pass.
The three runs don't depend on each other (own config, own output file), so they can run as
background jobs: `uv run ... main.yaml & uv run ... baseline.yaml & uv run ... clean_window.yaml & wait`.

### 5. Figures and table

```bash
uv run python -m open_nrb_causal.evaluation.run_event_study_report
```

Reads every `results/*_seed*.json` from step 4 (or only the JSON paths you pass as arguments)
plus the step 2 audit CSV. It never re-fits anything. Output goes to
`src/open_nrb_causal/results/figures/`:

| File | Content |
|---|---|
| `event_study.png` | per outcome x {dose, exposure}: coefficients by fiscal year with 95% CIs, full panel vs clean window overlaid; filled marker = wild p < 0.05; reference bin -1 drawn at 0 |
| `did_forest.png` | pooled DiD coefficients, W_hat vs RAS, full vs clean window, 95% CIs + p-values |
| `did_table.md` | the numbers behind `did_forest.png` (beta, CR1 SE, wild p, n_obs, clusters) |
| `extraction_audit.png` | per-bulletin asset-identity pass rate + mean fragments, dose window / transition / clean-window end marked (needs step 2) |

The figure content is final; the styling (size, fonts, greyscale) is still to be fitted to the
chosen venue.

### 6. Reviewer checks (`paper2_causal_spillover.md` §5b)

One script per §5b item. Each rebuilds the same sample as step 4 (shared
`models.run_causal_experiment.build_inputs`). `power_mde` runs all three configs by default;
`reviewer_checks` runs `main.yaml` only (`--config` to pick others).

**6.1 Minimum detectable effect (MDE)** (§5b item 1, "the floor doesn't bind"):

```bash
uv run python -m open_nrb_causal.evaluation.power_mde
```

For every pooled DiD coefficient it reports the 80%-power MDE at 5% (two-sided, t(G-1)),
|beta|/MDE, and a standardised MDE: within-bank SDs of the outcome per 1-SD change in the
regressor. CR1 only (no bootstrap), so it runs in seconds. Output: printed, plus
`src/open_nrb_causal/results/power_mde.md`.

**6.2 The remaining cheap checks** (§5b items 2, 5, 6, 8, 11a, 11c, 12):

```bash
uv run python -m open_nrb_causal.evaluation.reviewer_checks
```

| `--only` name | §5b item | What it reports |
|---|---|---|
| `survivorship` | 2 | all commercial banks: in panel or not, months with complete data, first/last month, pre-window dose and size; Mann-Whitney test, included vs excluded |
| `weights` | 8 | HT weight p99/max relative to the mean, Kish effective sample size, per-bank and pre/post weight shares; trimmed (q 0.95/0.90) and unweighted DiD vs weighted |
| `gps_period` | 8 | remedy: HT weights from a GPS with period intercepts + per-period sigma; same diagnostics + DiD |
| `negative_flows` | 11a | share of negative de-cumulated monthly flows per outcome, worst fiscal year, by calendar month |
| `quarterly` | 11a | remedy: outcomes built from quarter-end YTD figures only; DiD under main, unweighted and period-effects weights |
| `same_quarter` | 11a | quarterly DiD on Ashwin (Q1) cells only, ruling out a dose-related seasonal pattern |
| `pre_trend` | 3 | dose-only quarterly event study from Baisakh 2070 (before W_hat exists): per-bin leads, pooled lead vs post (all quarters and Q1 only), 1-SD effect relative to the pre mean |
| `frozen_size` | 11c | DiD with outcomes divided by the frozen pre-window size |
| `leave_one_out` | 6 | DiD dropping each bank in turn (CR1): range, sign flips, most influential bank |
| `row_normalised` | 12 | DiD with exposure from the row-normalised network (weights per `weighting`); rows with published lending r ≤ 0 set to 0 first; prints both matrices' row / column sums against r and c |
| `centrality_exposure` | 12b | exploratory: Katz–Bonacich exposure W̃ (I − 0.5 W̃)⁻¹ dose_pre (direct plus indirect counterparties, α = 0.5 fixed in advance), corr with direct and row-normalised exposure, Holm within the check only |
| `written_off_zero` | 11b | loans written off with blanks read as missing vs as 0 where other P&L lines were reported (main: `blank_as_zero`): per-bank counts, Eq. A′ and total effect side by side |
| `paid_up_dose` | 1 | alternative dose (required − paid-up capital) / size: matched labels, per-bank pre-window table, Eq. A′ and total effect; then total-effect robustness: (a) size term vs paid-up term entered jointly, (c) frozen pre-window size, (d) leave-one-out (CR1), (f) first stage (paid-up / size_pre and log paid-up stocks), (g) asset growth (log size), (h) horse race with the leverage dose |
| `paid_up_path` | 1 | per bank: paid-up at each Asar and at the pre-window end, first month at the requirement vs the Asar 2074 deadline, share of the increase already in place by the pre-window end (anticipation) |
| `paid_up_threats` | 1 | threats to the paid-up result: (1) stock event study by fiscal-year bin + total effect with dose and log size_pre x linear trend; (2) SBL / RBB mergers dated from the bulletins (jumps vs the clean-month p99), drop-the-bank and SBL + Lumbini pro-forma refits of (f)–(h); (3) floor vs catch-up: bunching at 8,000 vs placebo thresholds, raise vs shortfall (permutation p), raises by fiscal year |
| `paid_up_trend` | 1 | dose and log size_pre trends fitted on the pre quarters only and extrapolated: event study by bin (no trend vs pre-only trend), pooled total effect (none / whole-panel / pre-only trend) on paid-up / size_pre, log size, NIM, net profit, provisions; smoothness bound for a post slope break (CI(M), breakdown M*, relative to the pre slope; simplified Rambachan–Roth) |
| `label_continuity` | 2 | bank label pairs (relabel or not, from overlap, gap, size and capital-fund ratios), exits with candidate absorbers, entrants; data continuity only, not merger verification |
| `multiple_tests` | 5 | test counts, expected false positives, Holm and BH, over the step 4 JSONs, plus the joint wild-bootstrap Wald tests over the post bins |

`--quarterly` swaps the monthly outcomes for the quarterly ones in every check, e.g.
`--quarterly --only leave_one_out`; the output file gets a `_quarterly` suffix. `pre_trend`
always uses quarterly outcomes.

Wild-bootstrap p-values use `--n-boot` (default: the config's `n_bootstrap`, 9999, so a check
that reproduces a step 4 fit gives the step 4 p-value; `--n-boot 999` for a quick pass).
Output: `src/open_nrb_causal/results/reviewer_checks_<config>.md`. The joint Wald tests
themselves are computed in step 4 (`fit_event_study`, saved in the run JSON), so
`multiple_tests` needs a step 4 run made after 2026-11-04.

## Reproducing the paper

The specification is frozen: no new analysis, and every number in `paper2_draft.md` comes from
one of the files below. One command runs the whole chain, Paper 1 artifact → Paper 2 outputs
(steps 1, 2, 4, 5, 6 above plus the interface check), and stops at the first failure:

```bash
bash src/open_nrb_causal/reproduce_paper.sh
```

- Console logs: `results/logs/<step>.log`. Input hashes (configs, artifact, both CSVs, Paper 1's
  config): `results/logs/inputs.sha256`. Diff these between runs to confirm the inputs did not change.
- Every wild-bootstrap p-value uses 9,999 draws (`n_bootstrap`, `bootstrap_seed: 0`), main runs
  and reviewer checks alike.
- The three step 4 runs and the three `reviewer_checks` runs each go in parallel. The
  reviewer checks at 9,999 draws are the long step.

**Paper 1 interface** (`evaluation/check_paper1_interface`, step 3 of the script). `build_inputs`,
shared by every script, refuses to run unless:
- the CSV's banks and periods cover the artifact's (`align_to_artifact`);
- the CSV's r, c and K equal the artifact's saved marginals and cap (`check_marginals`);
- no month inside the config's window has a NaN W_hat (Paper 1's non-finite / degenerate
  months).

The interface report adds Paper 1's config (kappa, lambda_mi, seed, mtime vs the artifact), the
cap-binding share and corr(exposure W_hat, exposure RAS). Output:
`results/paper1_interface_<config>.md`.

**Manifest: draft → output.** `rc_<c>` = `results/reviewer_checks_<c>.md`, `<c>_seed0` =
`results/<c>_seed0.json` (also printed in `logs/4_run_<c>.log`).

| Draft item | Source (section of the output) |
|---|---|
| §4.1 sample (bulletins, sheets, institutions, N = 16, T = 133, missing months) | `bs_extraction_audit.csv` / `logs/2_audit.log`; `logs/4_run_baseline.log` (panel line); `main_seed0` `total_effect_panel` |
| §4.2, App. C, Figure 1 | `logs/2_audit.log`; `figures/extraction_audit.png` |
| §4.3 W_hat properties, §8 interface checks (exposure corr, cap share, missing month, artifact provenance) | `paper1_interface_<c>.md` |
| §8 fit of each matrix to r and c | `rc_main` `row_normalised` (marginal table) |
| §8 absent month (Kartik 2078), zero-cell fit (465/465, 249/398), 1.5e-5 W_hat difference; §4.3; §7.4 21 vs 16 window-complete labels | `panel_window_checks_main.md` (`evaluation/panel_window_checks`, not in `reproduce_paper.sh`) |
| §8 marginal balance (Σr vs Σc, infeasible months, Spearman); abstract, §9 lesson 3 | `rc_main` `marginal_balance` (written by `reviewer_checks` with no `--only`; `--only marginal_balance` writes the same `rc_main` file, not a separate one) |
| §4.4 doses (banks below, range, SD), §3.3 dose corr | `rc_main` `paid_up_dose` (pre-window table); `logs/4_run_main.log` (dose_pre line) |
| §4.4 YTD ratio 0.087, corr(dose × post, exposure × post) | `logs/4_run_<c>.log` (YTD and corr lines) |
| Table 1, §6.1 | `main_seed0` `outcomes.*.total_effect`; 1-SD / MDE: `power_mde.md`, `rc_main` `pre_trend` |
| Table 2, §6.2 first stage, trends, M* | `rc_main` `paid_up_dose` (f)(g), `paid_up_threats` (1), `paid_up_trend` |
| §6.2 bank-level compliance (Figure 2) | `rc_main` `paid_up_path`, `paid_up_threats` (3); `figures/paid_up_compliance.png`, `figures/paid_up_figures.md` |
| Table 3, §6.3 (ii), horse race, leave-one-out | `rc_main` `paid_up_dose` (Eq. A′, (c), (d), (h)); Holm / BH over the four: `multiple_tests` |
| §6.3 (i) event study (Figure 3), (iii) trends | `rc_main` `paid_up_trend`; `figures/paid_up_event_study.png`, `figures/paid_up_figures.md` |
| §6.4 γ, own NIM dose, paid-up exposure | `main_seed0` `outcomes.*.did`, `did_ras_baseline`; `rc_main` `paid_up_dose` (Eq. A′) |
| Table 4, Holm / BH | `main_seed0` `outcomes.*.event_study.joint`; `rc_main` `multiple_tests` |
| §6.4 MDE, Table 6 | `power_mde.md` (rows with and without "(paid-up dose)"; raw-SD and residual-SD standardised columns) |
| Table 5, §7.1 | `rc_main` `quarterly`, `same_quarter`, `pre_trend`, `leave_one_out` |
| §7.2 | `rc_main` `paid_up_dose` (a), (c) |
| §7.3, App. B | `rc_baseline`, `rc_clean_window` `weights`, `gps_period` |
| §7.4 mergers, survivorship, relabels | `rc_main` `paid_up_threats` (2), `survivorship`, `label_continuity`; merger dates: @nrblistbfi (external) |
| §7.5, App. E | `rc_main` `written_off_zero`, `row_normalised`, `centrality_exposure`; `rc_baseline` `frozen_size` |
| App. A | `rc_baseline`, `rc_clean_window` `negative_flows`, `quarterly` |
| App. D | `logs/4_run_<c>.log` (RDD lines) |
| App. F | `baseline_seed0`, `clean_window_seed0`; `figures/did_table.md`, `event_study.png`; `rc_baseline` `multiple_tests`, `leave_one_out` |

## Configuration (`experiments/configs/*.yaml`)

Every field is required (no defaults, same convention as Paper 1's configs).
`clean_window.yaml` differs from `baseline.yaml` only in `post_window_end`; `main.yaml` differs
from `clean_window.yaml` only in `outcome_construction: quarterly`.

| Field | Baseline value | Meaning |
|---|---|---|
| `artifact_path`, `csv_path`, `pnl_csv_path` | Paper 1 paths | inputs (see Data) |
| `paid_up_capital_requirement` | `8000` | NPR mn, required paid-up capital for the alternative dose (`reviewer_checks --only paid_up_dose`): Rs 800 crore, Class "क", Unified Directive 2073 dir. 23 pt. 4, due Asar end 2074; `null` = skip the estimates |
| `capital_floor` | `0.04` | NRB Basel III leverage-ratio minimum (CAF 2015 §2.7.B.a); see `variables.CAPITAL_FLOOR` for the approximation caveats |
| `pre_window_end` | `[2072, "Mangsir"]` | last month of the dose-freezing window (inclusive) |
| `effective_period` | `[2073, "Asar"]` | mandate effective; post = strictly after; Poush 2072 - Asar 2073 = transition |
| `post_window_end` | `null` | last month kept; `null` = whole panel (`clean_window.yaml`, `main.yaml`: `[2075, "Falgun"]`) |
| `pnl_ytd_cumulative` | `true` | the P&L is fiscal-YTD (set from step 4's YTD diagnostic) |
| `blank_as_zero` | `[loan_written_off]` | outcome fields whose blank row reads as 0 where the bank reported other P&L lines that month (no P&L row stays missing); user decision 2026-11-04, §5b item 11b; `[]` = all blanks missing |
| `outcome_construction` | `monthly` | `monthly` = de-cumulated monthly flows; `quarterly` = quarter-end YTD only (`main.yaml`; needs `pnl_ytd_cumulative: true`) |
| `weighting` | `unweighted` | `unweighted` TWFE (main) or `ht` = stabilised HT/GPS weights; `reviewer_checks --only weights` reports the other one |
| `n_bootstrap`, `bootstrap_seed` | `9999`, `0` | WCR wild-cluster bootstrap draws; 0 = CR1 only |
| `rdd_bandwidth`, `rdd_window` | `0.02`, Shrawan 2073 - Asar 2075 | pooled RDD around the floor |

## Status (2026-11-05)

- 2026-11-05: `reproduce_paper.sh` ran end to end on real data: 44 tests pass, the interface
  check passes for all three configs, and every p-value uses 9,999 draws. Headline unchanged
  (all null; written-off p 0.280). Paid-up: NIM −0.057 (p 0.002), net profit −0.041 (p 0.021);
  both still lose significance with the pre-only trend and the frozen-size denominator.
  Baseline paid-up values are corrupted after FY2075 (e.g. EBL 8 at Asar 2076): don't report
  baseline paid-up results.
- 2026-11-05: `row_normalised` (main) and the new `centrality_exposure` (exploratory) have
  been run: exposure null under both (smallest p 0.161 / 0.163; Holm 1.00). The marginal table
  shows the published r and c are 0 in 398 / 465 of 720 main-window bank-months; RAS fits r
  (max deviation 41) but not c (8,290), W_hat fits c more closely (438) but not r (2,311). The
  two networks target the same marginals but do not fit them the same way (draft §8).
- 2026-11-05: full `reviewer_checks`, `power_mde` (now with "(paid-up dose)" rows: dose and
  exposure rebuilt from the paid-up dose) and `plot_paid_up_figures` have been run; Figures 2–3
  match the draft. Paid-up exposure (Eq. A′) is null under both networks (wild p 0.29–0.99) and
  is now the primary spillover test in draft §6.4 (Table 6). In the main specification the
  standardised exposure MDE is 1.97–5.05 (paid-up) and 3.85–7.83 (leverage) within-bank SDs per
  SD of exposure, so the spillover null is weak. `plot_paid_up_figures` runs as the
  `5b_paid_up_figures` step of `reproduce_paper.sh` (on `main.yaml`).

## Status (2026-11-03)

- Steps 1-4a have been run on real data: tests pass, and the run completes with no crash on
  bad cells. N = 16 banks x T = 133 months (Asar 2072 - Shrawan 2083).
- Pooled DiD: null for every outcome under both W_hat and RAS. The event-study patterns are
  flat across all bins, which suggests an artefact of the short (5-month) reference bin. The
  RDD is infeasible because the 4% floor doesn't bind (0 cells right of the cutoff).
- Step 4b (clean window, T = 45) has also been run: null again under both networks, with
  smaller SEs than the full panel, so the null isn't a power loss. The full-panel event-study
  patterns don't survive (see `paper2_causal_spillover.md` §5a).
- Step 5 has been run: all four outputs are produced and `did_table.md` matches the step 4
  printouts. In the full-panel event study, the written-off dose path is ≈ -0.052 even in bins
  0-1, which contain only clean months, whereas the clean window gives ≈ 0 for the same bins.
  So the corrupted later months shift the earlier coefficients through the bank fixed effects.
  The clean window is the event study to report.
- Decisions 2026-11-03, after the §5b weight and quarterly checks: the main estimator is
  unweighted TWFE (the HT/GPS weights can't achieve overlap with N = 16), and the main
  specification is the clean window with quarterly outcomes (`main.yaml`). The step 4 runs
  above used the HT weights. Since `baseline.yaml` and `clean_window.yaml` are now unweighted,
  re-running them changes their JSONs.
- Decision 2026-11-03: the headline β is the dose-only total effect from Baisakh 2070
  (≈ 0.020, p 0.57 for NIM in `pre_trend`), and Eq. A′ is kept for γ. `main.yaml` now reports
  it, and `did_forest.png` / `did_table.md` show it as network "dose-only from 2070".
- 2026-11-04: joint Wald over the post bins, the written-off blank = 0 version, the paid-up
  capital dose and the label-continuity check have been run (`main.yaml`). No joint test
  survives Holm/BH, and loans written off is null under both codings. No label change touches
  the 16 banks in the clean window. Details in `paper2_causal_spillover.md` §5b items 1, 2, 5
  and 11b.
- 2026-11-04: `paid_up_capital_requirement` set to 8000 (NPR 8,000 mn by Asar end 2074, Unified
  Directive 2073 dir. 23 pt. 4; design doc §4a). Total effect with this dose: NIM −0.057 (wild
  p 0.003) and net profit −0.041 (p 0.023), flat leads; Eq. A′ null. The dose is mostly
  inverse bank size, so `paid_up_dose` now adds the decomposition, frozen-size and
  leave-one-out checks, plus `paid_up_path`. Not yet a causal claim (§5b item 1).
- 2026-11-04: loans written off now reads blanks as 0 in the main spec (`blank_as_zero`, user
  decision, §5b item 11b). Steps 4 and 5 have been re-run with it: the headline written-off β
  is −0.0020 (p 0.28, 16 clusters), and all other headline numbers are unchanged.
- 2026-11-04: paid-up dose checks (f)–(h). The first stage holds (paid-up / size_pre β 0.36,
  p 0.005, flat lead). Asset growth is borderline (log size β 3.0, p 0.053; lead p 0.066). In
  the horse race the paid-up dose keeps NIM (−0.057, p 0.013) and the leverage dose is null.
  Reading: the paid-up floor bound, and its NIM effect is mostly asset growth (denominator),
  not lower income. Net profit is not robust. Details in §5b item 1.
- 2026-11-04: `paid_up_threats`. The first stage holds and is a floor, not catch-up: it
  survives dose and size × trend (p 0.022), dropping SBL or RBB, and the SBL + Lumbini
  pro-forma. At Asar 2075, 10/16 banks sit in [8,000, 8,400) vs ≤ 1 at the placebo thresholds;
  Spearman(start, end) 0.01; raise / shortfall ≈ 1.01. NIM does not survive the trends
  (−0.017, p 0.53), and log size has negative pre bins (p 0.06–0.08) and loses significance
  with trends. The bulletins don't show a full Lumbini absorption into SBL in Asar 2073
  (+0.144 log vs ≈ +0.34 expected). Details in §5b item 1.
- 2026-11-04: `paid_up_trend` (pre-only trend; reproduces the `paid_up_threats` rows). The
  first stage survives (0.89, p 0.013; magnitude inflated by extrapolating a noisy pre slope).
  NIM: the no-trend event study has flat pre bins and post bins −0.064 / −0.060 / −0.040
  (p ≤ 0.006); with the pre-only trend it is +0.020 (p 0.65, SE ×4.6). Net profit and
  provisions are null with the trend. Reading: the floor bound (robust); the NIM effect is
  suggestive, and the short, unbalanced pre period can't separate it from a dose-specific
  trend. Details in §5b item 1.

## Deferred

- Bootstrap-replicate uncertainty for W_hat (needs a retrain or a Paper 1 artifact change;
  both out of scope).
- §4e (NRB Statement of Credit, 2-year-coverage file).
- Repairing the PDF extraction: Paper 1's `pdf_extract.py`, out of scope (see Data & Known
  Limitations).
