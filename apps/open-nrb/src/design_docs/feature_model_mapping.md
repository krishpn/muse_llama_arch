# NRB Feature Inventory → Research Model Mapping (full variable-level pass)

> Rebuilt from an actual extraction run against `Shrawan_2083_Publish.pdf` (not reconstructed
> from memory this time) — every unique `line_item`/`item_code` the parser currently produces,
> mapped individually to Paper 1 (SBM/GNN/RNN/Bayesian net reconstruction) or Paper 2
> (NRB-shock econometric causal design), with `[Aronow-Samii]` noting where the observational
> vs. randomized distinction we discussed applies.

## Known data-structure caveat (found during this pass)

**`item_code` is not unique across the Liabilities and Assets sides of the balance sheet** — both
sides restart numbering at 1 (e.g. `1 CAPITAL FUND` on liabilities, `1 LIQUID FUNDS` on assets;
same for `2`, `3`, lettered sub-items `a.`–`g.`, etc.). The parser currently has no column
disambiguating "which side" a row is on — that has to be inferred from which `TOTAL` block
(`TOTAL Liabilities` vs `TOTAL ASSETS`) the row falls between on the same page. If any
downstream step joins on `item_code` alone, it will silently collide liabilities and assets
rows that happen to share a code. Worth adding a `side` column (`liabilities`/`assets`) to the
extractor before this goes into the model pipeline.

---

## Balance Sheet — Liabilities side

| Code | Line item | Model role |
|---|---|---|
| 1 | CAPITAL FUND | Node feature (capital) — parent rollup |
| a. | Paid-up Capital | component of CAPITAL FUND |
| b. | Calls in Advance | component (minor) |
| c. | Proposed Bonus Share | component |
| d. | General Reserves | component |
| e. | Share Premium | component |
| f. | Retained Earning | component — links to P&L Net Profit via Bayesian structural edge |
| g. | Others Reserves Fund | component |
| 2 | BORROWINGS | Node feature (funding mix) — parent rollup |
| a. | NRB | funding component; candidate Paper 2 control (central-bank liquidity dependency) |
| b. | **Interbank Borrowing** | **Paper 1 edge-reconstruction margin — liability-side marginal for SBM/entropy inference of the bilateral interbank matrix** |
| c. | Foreign Banks and Fin. Ins. | funding component |
| d. | Other Financial Ins. | funding component |
| 3 | DEPOSITS | Node feature (funding mix) — parent rollup; Paper 2 control (funding stability) |
| a. | Current | component |
| b. | Savings | component |
| c. | Fixed | component |
| d. | Call Deposits | component |
| — | Domestic / Foreign / by depositor type (Govt. Enterprises, Private Sector, Financial Institutions, Non Residents) | granular funding-composition detail; aggregate unless depositor-concentration becomes its own regressor |
| 4 | Bills Payable | minor liability — not currently used in either model |
| 5 | Other Liabilities | residual — not used |
| 6 / 10 | Reconcillation A/c, Reconcillation Account | accounting plug/balancing entry — **not economically meaningful, exclude from model** |
| 7 / 11 | Profit & Loss A/c | carrying balance — cross-check against P&L Net Profit for the structural-constraint edge |
| — | TOTAL Liabilities | **Bayesian-net structural constraint**: `TOTAL Liabilities = TOTAL ASSETS` |

## Balance Sheet — Assets side

| Code | Line item | Model role |
|---|---|---|
| 1 | LIQUID FUNDS | Node feature (liquidity); Paper 2 control (liquidity ratio numerator) |
| a. | Cash Balance (Nepalese Notes & Coins / Foreign Currency) | component |
| b. | Bank Balance (Domestic / Foreign Currency) | component |
| c. | Money at Call | component |
| 2 | INVESTMENT IN SECURITIES | Node feature (asset mix) |
| a. | Govt. Securities | component |
| b. | NRB Bond | component |
| c./d./e. | Foreign Banks & Fin. Ins. / Other Financial / Others | component |
| 3 | SHARE & OTHER INVESTMENT | parent rollup |
| a. | **Interbank Lending** | **Paper 1 edge-reconstruction margin — asset-side marginal; paired with Interbank Borrowing above, this is the actual payoff variable for the network-reconstruction method** |
| b./c. | Financial Institutions / Others | component |
| 4 | LOANS & ADVANCES (incl. 4.1 LOANS & ADVANCES, 4.2 BILL PURCHASED) | Node feature (asset mix); **Paper 2 outcome candidate** (credit growth) |
| a. | Private Sector | component; also the natural join key to Statement-of-Credit sector detail |
| b. | Staff Loans / Adv. | component |
| c. | Government Organizations | component |
| d. | Other Financial Ins. | component |
| 5 | LOANS AGAINST COLLECTED BILLS | component of loan book |
| a./b./c. | Against Domestic Bills / Against Foreign Bills / Import Bills & Imports | component |
| 6 | FIXED ASSETS | Node feature (minor) — size control |
| 7 | OTHER ASSETS | residual, parent rollup |
| a. | Accrued Interest | links to P&L Interest Income recognition |
| b./c./d./e. | Sundry Debtors / Cash In Transit / Others | component |
| 8 | Expenses not Written off | legacy accounting item — not used |
| 9 | Non Banking Assets | minor; candidate distress proxy (foreclosed-asset holdings) for Paper 2 if NPA signal is wanted |
| 10 / 11 | Reconciliation / P&L carrying (asset side) | plug — not used |
| — | TOTAL ASSETS | structural constraint anchor (see above) |

## Profit & Loss

| Code | Line item | Model role |
|---|---|---|
| 1. | Interest Expense | Paper 2 outcome component (interest margin denominator) |
| 1.1 | On Deposit Liabilities | component |
| 1.1.1–1.1.4 | Saving / Fixed (by tenor 1.1.2.1–1.1.2.4) / Call / Certificate of Deposit | granular cost-of-funds detail |
| 1.2 | On Borrowing | links to BORROWINGS node feature — cost of interbank/wholesale funding |
| 1.3–1.5 | On Agency Balance / On Call Deposit / On Others | component |
| 1. | Interest Income | Paper 2 outcome component (interest margin numerator) |
| 1.1 | On Loans and Advances | links directly to LOANS & ADVANCES asset |
| 1.2 | On Investment | links to INVESTMENT IN SECURITIES asset |
| 2. | Commission and Discount / Commission Expense (2.1–2.3) | fee-income diversification control |
| 3. | Exchange Fluctuation Income / Staff Expense (3.1–3.2) | operating-cost control |
| 4. | Other Operating Income / Office Operating Expenses | operating-cost control |
| 5. | Non Operating Income / Foreign Exchange Loss (5.1–5.2) | minor flow — not currently used |
| 6. | Write Back from Provisions for loss / Non Operating Expense | minor flow — not currently used |
| 7. | Recovery of written-off Loan / **Provision for Risk** | **Paper 2 outcome** (loan-loss provisioning) |
| 7.1–7.1.3 | Loan loss Provision → General (Pass/Watch List) / Special / Additional | granular provisioning detail — useful for a provisioning-intensity regressor |
| 7.2/7.3/7.5 | Provision for Non-Banking Assets / Loss on Investment / Loss of Other Assets | minor provisioning components |
| 8. | Income from Extra Ordinary transactions / **Loan Written Off** | **Paper 2 outcome** (write-off rate) |
| 9. | Net Loss / Provision for Staff Bonus | not currently used |
| 10. | Provision for Income Tax | control (effective tax rate) |
| 11. | Others | residual |
| 12. | **Net Profit** | **Paper 2 primary outcome** |
| — | Total Expenses / Total Income | **Bayesian-net structural constraint**: `Total Income − Total Expenses = Net Profit` |

## Statement of Credit (no `item_code` — string-matched labels only, per the known gap noted earlier)

| Group | Line items | Model role |
|---|---|---|
| Sector-wise (12 sectors + Others + TOTAL) | Agricultural & Forest, Fishery, Mining, Food Production*, Construction, Electricity/Gas/Water, Metal Products/Mach., Transport/Comm./Public Utilities, Wholesaler & Retailer, Finance/Insurance/Real Estate, Tourism Service**, Other Services, Consumption Loans, Local Government, Others | **Paper 1 common-exposure/contagion channel** (shared sector concentration = correlated-shock edge signal independent of interbank margins); also **Paper 2 outcome** (sector credit growth) |
| Product-wise | Term Loan, Overdraft, Cash Credit Loan, Trust Receipt/Import Loan, Short Term Working Capital/Demand Loan, Residential Personal Home Loan#, Real Estate Loan, Margin Nature Loan, Hire Purchase Loan, Deprived Sector Loan, Bills Purchased, Other Product, Total | secondary concentration channel; Paper 2 control (loan-product mix) |
| Collateral-wise | Gold/Silver, Govt./Non-Govt. Securities, Fixed Deposit Receipts, Own/Other Licensed Institutions guarantee, Collateral of Properties, Fixed/Current Assets, Against security of Bill (Domestic/Foreign), Against Guarantee (Government/Institutional/Personal/Collective/Int'l/Other), Credit Card | asset-quality / risk-mitigant control candidate for Paper 2 — not currently used in Paper 1 |

Footnote markers (`*`, `**`, `#`) on some sector/product labels are inline suffixes, not bracketed
footnotes, so `clean_line_item_label`'s trailing-bracket strip correctly leaves them intact —
worth stripping separately if they're not wanted downstream.

## Observational-design caveat for Paper 2 `[Aronow-Samii]`

Every "Paper 2 outcome"/"Paper 2 control" tag above assumes the Aronow–Samii-style exposure
machinery is adapted the way we discussed: since the NRB regulatory shock isn't a randomized
assignment, `π_i(d_k)` (the probability of a bank being exposed) has to be *estimated* from
observables (size, capital ratio, BFI class) under a selection-on-observables assumption, rather
than known by design. That weakens the unbiasedness/conservative-variance guarantees from
"guaranteed" to "conditional on the propensity model being right" — same caveat applies to every
regressor built from this inventory, not just the spillover term.
