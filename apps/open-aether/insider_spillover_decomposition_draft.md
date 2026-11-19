# Homophily or Contagion? Board-Interlock Diffusion of 10b5-1 Cooling-Off Plans

*Working draft — skeleton with embedded results. Flat third-person academic style. [TO-FILL] marks items that still require a box run, a citation, or author input.*

---

## Abstract

In December 2022 the Securities and Exchange Commission amended Rule 10b5-1 to require a mandatory cooling-off period between the adoption of an insider trading plan and the first trade under it; the amendment took effect for most issuers in the second quarter of 2023. Adoption of cooling-off plans spread rapidly and unevenly across firms, and adopting firms cluster on the board-interlock network formed by shared directors. This paper asks whether that clustering reflects peer influence (contagion through interlocks) or selection (homophily among similar firms that adopt for common reasons). Using Securities and Exchange Commission Form 345 filings for 2006 through 2026 and a fixed pre-treatment interlock graph, the paper applies a design-based randomization-inference decomposition that permutes adoption timing within strata of interlock degree and pre-mandate trading activity. Between 88 and 94 percent of the observed association between neighbor adoption and own adoption is attributable to homophily. The residual contagion component is small — about 7 percent of the association under the primary specification — and attenuates to marginal significance as homophily controls tighten. The paper further shows that two standard peer-effect estimators overstate contagion on this network: a friends-of-friends instrument is invalid because the interlock graph is dense and transitive, and a within-director fixed-effects hazard is mechanically biased by immortal time, returning a large coefficient even under randomized adoption timing. The results bound board-network contagion in a governance-practice setting and caution against naive peer-effect estimation on dense interlock networks.

**Keywords:** Rule 10b5-1; insider trading; board interlocks; peer effects; homophily; network interference; randomization inference.
**JEL:** G34, G38, G14, C21. [TO-FILL: confirm codes]

---

## 1. Introduction

[TO-FILL: motivation paragraph — the 2023 Rule 10b5-1 amendment as a discrete governance shock; why board interlocks are the natural diffusion channel for governance practices; the policy stakes of distinguishing influence from selection.]



[TO-FILL: roadmap paragraph.]

## 2. Institutional background and related work

[TO-FILL: Rule 10b5-1 history, the 2022 amendment, cooling-off requirement, effective dates — full prose.] The Securities and Exchange Commission adopted the amendment in December 2022 (Securities and Exchange Commission, 2022, Release Nos. 33-11138; 34-96492); it took effect for most issuers on February 27, 2023, and imposes a mandatory cooling-off period between the adoption of a Rule 10b5-1 trading plan and the first trade under it (90 to 120 days for directors and officers [verify day counts]).

## 3. Data and network construction


## 4. Econometric framework

### 4.4 Results

Table 2 reports the decomposition across a ladder of strata granularities.

**Table 2. Homophily-versus-contagion decomposition (randomization inference, B = 2000).**

| Strata (degree × activity) | Null mean | Residual | Homophily share | Contagion share | One-sided *p* |
|---|---|---|---|---|---|
| none (1×1) | 0.02613 | 0.00365 | 87.7% | 12.3% | 0.0005 |
| 2×2 | 0.02726 | 0.00252 | 91.5% | 8.5% | 0.014 |
| 4×4 (primary) | 0.02772 | 0.00206 | 93.1% | 6.9% | 0.028 |
| 8×8 | 0.02808 | 0.00170 | 94.3% | 5.7% | 0.053 |

*Observed slope = 0.02978. Homophily share = null mean / observed; contagion share = residual / observed.*

The contagion residual declines monotonically and loses conventional significance as homophily controls tighten, from 12.3 percent of the association (p = 0.0005) with no stratification to 5.7 percent (p = 0.053) under the finest stratification. The 4×4 specification is the primary one because its cells retain adequate sample size; there the decomposition is roughly 93 percent homophily and 7 percent contagion, with the residual marginally significant (p = 0.028).

Two readings are ruled out. The clustering is not purely contagion: even minimal homophily control attributes the overwhelming share of the association to selection. It is also not cleanly interpretable as a robust causal peer effect: the residual is sensitive to the strength of homophily controls. The honest statement is an upper bound — board-network contagion accounts for at most about one-eighth of the observed co-adoption, and plausibly far less once firm composition is held fixed.

[Caveat to retain in the text: the drift toward 100 percent homophily at the finest strata is partly mechanical, because as strata approach singletons the permutation loses freedom and the null approaches the observed value by construction. The 8×8 p-value is therefore not evidence of exactly zero contagion; the informative object is the whole ladder, which shows the contagion estimate shrinking as composition is controlled.]

## 5. Why naive peer-effect estimators overstate contagion

A reader might expect an instrumental-variables or fixed-effects estimator to recover a cleaner peer effect. On this network both fail, in known and diagnosable ways.

### 5.1 Friends-of-friends instrument is invalid on a dense, transitive graph


### 5.2 Within-director fixed-effects hazard is mechanically biased by immortal time



[Table 3 — placebo distribution; figure with observed marker over the null histogram. TO-FILL figure.]

## 6. Secondary result: no volume spillover


## 7. Conclusion


---

## Appendix A. Robustness
[TO-FILL: alternative strata definitions; alternative exposure (fraction vs indicator); cloglog hazard vs linear probability; permutation of status as well as timing.]

### A.x Degree distribution
The interlock degree distribution is right-skewed and heavy-tailed (Figure 1a), but the paper makes no scale-free claim. A Clauset–Shalizi–Newman discrete power-law fit to the degree sequence (Clauset, Shalizi, and Newman, 2009), computed with the `powerlaw` package (Alstott, Bullmore, and Plenz, 2014), gives a scaling exponent α = 2.90 above a lower cutoff xmin = 6, covering 42.3 percent of nodes (Kolmogorov–Smirnov distance 0.11). Although α falls in the range often associated with scale-free networks, a likelihood-ratio comparison decisively rejects the power law in favor of a lognormal alternative (normalized log-likelihood ratio R = −10.1, p < 0.001) and in favor of an exponential alternative (R = −11.7, p < 0.001). The degree distribution is therefore best described as heavy-tailed and lognormal, not scale-free; scale-free structure is in any case rarer than often assumed (Broido and Clauset, 2019). The distributional form is not used anywhere in the identification argument; the structural feature that matters for Section 5.1 is the graph's high clustering (global clustering coefficient 0.273, against an Erdős–Rényi benchmark of 0.00132), not the tail exponent.

## Appendix B. Data construction
[TO-FILL: parser details; `AFF10B5ONE` cleaning; 2023Q1 exclusion; interlock projection from the director–firm bipartite graph.]

## References

*Every page range, volume/issue, DOI, and release number below is tagged `[verify]` and must be checked against the source before submission — these were assembled from memory and have not been verified against a database. References are grouped by the role each plays in the paper; entries are alphabetical within each group.*

### Design-based inference and randomization-inference foundations

*The paper runs randomization inference rather than a parametric model; these are its lineage.*

Fisher, R. A. (1935). *The Design of Experiments.* Edinburgh: Oliver and Boyd. [verify publisher, year, edition]

Imbens, G. W., and D. B. Rubin (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction.* Cambridge: Cambridge University Press. [verify publisher, year]

Rosenbaum, P. R. (2002). Covariance adjustment in randomized experiments and observational studies. *Statistical Science* 17(3), 286–327. [verify pp., vol/issue, DOI]

### Causal inference under network interference

*Exposure mappings and design-based estimation/inference when one unit's treatment affects another — the machinery behind Sections 4 and 6.*

Aronow, P. M., and C. Samii (2017). Estimating average causal effects under general interference, with application to a social network experiment. *Annals of Applied Statistics* 11(4), 1912–1947. [verify pp., vol/issue, DOI]

Athey, S., D. Eckles, and G. W. Imbens (2018). Exact p-values for network interference. *Journal of the American Statistical Association* 113(521), 230–240. [verify pp., vol/issue, DOI]

Eckles, D., B. Karrer, and J. Ugander (2017). Design and analysis of experiments in networks: Reducing bias from interference. *Journal of Causal Inference* 5(1), 20150021. [verify pp./article no., vol/issue, DOI]

Forastiere, L., E. M. Airoldi, and F. Mealli (2021). Identification and estimation of treatment and interference effects in observational studies on networks. *Journal of the American Statistical Association* 116(534), 901–918. [verify pp., vol/issue, DOI]

Hudgens, M. G., and M. E. Halloran (2008). Toward causal inference with interference. *Journal of the American Statistical Association* 103(482), 832–842. [verify pp., vol/issue, DOI]

Leung, M. P. (2020). Treatment and spillover effects under network interference. *Review of Economics and Statistics* 102(2), 368–380. [verify pp., vol/issue, DOI]

Sävje, F., P. M. Aronow, and M. G. Hudgens (2021). Average treatment effects in the presence of unknown interference. *Annals of Statistics* 49(2), 673–701. [verify pp., vol/issue, DOI]

### Peer effects and the reflection problem

*Why co-adoption cannot by itself identify influence, and when it can on a network — the basis for the Section 5.1 instrument and its failure.*

Angrist, J. D. (2014). The perils of peer effects. *Labour Economics* 30, 98–108. [verify pp., vol, DOI]

Bramoullé, Y., H. Djebbari, and B. Fortin (2009). Identification of peer effects through social networks. *Journal of Econometrics* 150(1), 41–55. [verify pp., vol/issue, DOI]

Bramoullé, Y., H. Djebbari, and B. Fortin (2020). Peer effects in networks: A survey. *Annual Review of Economics* 12, 603–629. [verify pp., vol, DOI]

Manski, C. F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies* 60(3), 531–542. [verify pp., vol/issue]

Manski, C. F. (2013). Identification of treatment response with social interactions. *Econometrics Journal* 16(1), S1–S23. [verify pp., vol/issue, DOI]

### Homophily versus contagion

*The core confounding the paper decomposes.*

Aral, S., L. Muchnik, and A. Sundararajan (2009). Distinguishing influence-based contagion from homophily-driven diffusion in dynamic networks. *Proceedings of the National Academy of Sciences* 106(51), 21544–21549. [verify pp., vol/issue, DOI]

Goldsmith-Pinkham, P., and G. W. Imbens (2013). Social networks and the identification of peer effects. *Journal of Business & Economic Statistics* 31(3), 253–264. [verify pp., vol/issue, DOI]

Shalizi, C. R., and A. C. Thomas (2011). Homophily and contagion are generically confounded in observational social network studies. *Sociological Methods & Research* 40(2), 211–239. [verify pp., vol/issue, DOI]

### Board interlocks and the diffusion of corporate practices

*The institutional literature establishing interlocks as a diffusion channel for governance practices.*

Bizjak, J., M. Lemmon, and R. Whitby (2009). Option backdating and board interlocks. *Review of Financial Studies* 22(11), 4821–4847. [verify pp., vol/issue, DOI]

Bouwman, C. H. S. (2011). Corporate governance propagation through overlapping directors. *Review of Financial Studies* 24(7), 2358–2394. [verify pp., vol/issue, DOI]

Davis, G. F. (1991). Agents without principles? The spread of the poison pill through the intercorporate network. *Administrative Science Quarterly* 36(4), 583–613. [verify pp., vol/issue]

Davis, G. F., and H. R. Greve (1997). Corporate elite networks and governance changes in the 1980s. *American Journal of Sociology* 103(1), 1–37. [verify pp., vol/issue, DOI]

Fracassi, C. (2017). Corporate finance policies and social networks. *Management Science* 63(8), 2420–2438. [verify pp., vol/issue, DOI]

Haunschild, P. R. (1993). Interorganizational imitation: The impact of interlocks on corporate acquisition activity. *Administrative Science Quarterly* 38(4), 564–592. [verify pp., vol/issue]

Mizruchi, M. S. (1996). What do interlocks do? An analysis, critique, and assessment of research on interlocking directorates. *Annual Review of Sociology* 22, 271–298. [verify pp., vol, DOI]

### Insider trading and Rule 10b5-1

*The governance practice under study and the 2022 amendment that is the shock.*

Cohen, L., C. Malloy, and L. Pomorski (2012). Decoding inside information. *Journal of Finance* 67(3), 1009–1043. [verify pp., vol/issue, DOI]

Jagolinzer, A. D. (2009). SEC Rule 10b5-1 and insiders' strategic trade. *Management Science* 55(2), 224–239. [verify pp., vol/issue, DOI]

Securities and Exchange Commission (2022). *Insider Trading Arrangements and Related Disclosures.* Final Rule, Release Nos. 33-11138; 34-96492. Adopted December 14, 2022; effective February 27, 2023. [verify release nos. and dates]

### Network structure: spatial autocorrelation and degree distributions

*Descriptive statistics for Section 3 and Appendix A (clustering of adoption, degree-distribution fit).*

Alstott, J., E. Bullmore, and D. Plenz (2014). powerlaw: A Python package for analysis of heavy-tailed distributions. *PLoS ONE* 9(1), e85777. [verify pp./article no., vol/issue, DOI]

Broido, A. D., and A. Clauset (2019). Scale-free networks are rare. *Nature Communications* 10, 1017. [verify pp./article no., vol, DOI]

Clauset, A., C. R. Shalizi, and M. E. J. Newman (2009). Power-law distributions in empirical data. *SIAM Review* 51(4), 661–703. [verify pp., vol/issue, DOI]

Cliff, A. D., and J. K. Ord (1981). *Spatial Processes: Models and Applications.* London: Pion. [verify publisher, year, edition]

Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika* 37(1/2), 17–23. [verify pp., vol/issue, DOI]

---

### Open empirical to-dos (box runs)
1. ~~Network descriptives for §3 and the motivating figure: adoption Moran's I / join-count on the interlock graph by quarter; adoption S-curve; exposed-versus-unexposed adoption gap by quarter.~~ DONE 2026-11-15 (`descriptives/network_desc.py`): Moran's I 0.14–0.19, join-count z≈18–22 p≈0.001 all quarters; 2023Q2 hazard gap 0.289 vs 0.120; Fig 1 saved.
2. ~~Re-estimate the Bramoullé IV first stage and 2SLS on the canonical 4,392-node graph; report mean two-hop degree there (§5.1).~~ DONE 2026-11-15: deg2 37.3, first stage t=15.1, 2SLS +0.003 (p=0.71).
3. ~~Re-estimate the Horvitz–Thompson volume spillover on the canonical graph (§6).~~ DONE 2026-11-15: τ̂=+1.02, RI two-sided p=0.35 (null).
4. ~~Table 1 summary statistics and degree distribution.~~ DONE 2026-11-15 (`descriptives/table1.py`): four panels embedded in §3.
5. ~~Re-run `director_placebo` on the box to confirm the §5.2 immortal-time placebo null still sits near −0.17 under the refactor.~~ DONE 2026-11-15: observed −0.181, placebo null mean −0.1705 (sd 0.003), z=−3.5 — artifact confirmed.
