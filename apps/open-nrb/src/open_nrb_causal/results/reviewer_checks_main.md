# Reviewer checks: main (N=16 x T=45, (2072, 'Asar') .. (2075, 'Falgun'), n_boot=9999, outcomes quarterly, main = unweighted TWFE)

## Survivorship (item 2): included vs excluded commercial banks
33 commercial banks from (2072, 'Asar'); 16 in the estimation panel (Paper 1 complete-case filter). Pre window = (2072, 'Asar')..(2072, 'Mangsir').

| bank | in panel | months with complete data | first | last | dose_pre | pre size (NPR mn) |
|---|---|---|---|---|---|---|
| ADBNL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0809 | 120,083 |
| BOK | no | 89/133 | Asar 2072 | Mangsir 2079 | -0.0398 | 48,351 |
| CBL | no | 43/133 | Magh 2075 | Magh 2079 | n/a | n/a |
| Century | no | 41/133 | Shrawan 2074 | Mangsir 2079 | n/a | n/a |
| Citizen | no | 90/133 | Magh 2075 | Shrawan 2083 | n/a | n/a |
| Citizens | no | 43/133 | Asar 2072 | Poush 2075 | -0.0436 | 45,824 |
| Civil | no | 31/133 | Asar 2073 | Poush 2075 | n/a | n/a |
| EBL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0344 | 91,391 |
| Global | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0573 | 73,660 |
| HBL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0400 | 89,690 |
| Kumari | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0405 | 41,610 |
| Laxmi | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0443 | 49,876 |
| Lumbini | no | 12/133 | Asar 2072 | Jestha 2073 | -0.0765 | 24,311 |
| MBL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0394 | 51,715 |
| Mega | no | 86/133 | Asar 2072 | Mangsir 2079 | -0.0679 | 28,018 |
| NABIL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0203 | 130,822 |
| NBB | no | 43/133 | Asar 2072 | Poush 2075 | -0.0699 | 44,508 |
| NBBL | no | 40/133 | Magh 2075 | Jestha 2079 | n/a | n/a |
| NBL | yes | 133/133 | Asar 2072 | Shrawan 2083 | 0.0003 | 90,513 |
| NCC | no | 89/133 | Asar 2072 | Mangsir 2079 | -0.0568 | 31,770 |
| NIBL | no | 90/133 | Asar 2072 | Poush 2079 | -0.0424 | 115,496 |
| NIC | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0484 | 64,001 |
| NIMB | no | 43/133 | Magh 2079 | Shrawan 2083 | n/a | n/a |
| NMB | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0493 | 47,823 |
| NSBI | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0438 | 64,912 |
| PRABHU | no | 43/133 | Asar 2072 | Poush 2075 | -0.0265 | 52,687 |
| Prabhu | no | 88/133 | Magh 2075 | Shrawan 2083 | n/a | n/a |
| Prime | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0492 | 49,098 |
| RBB | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0008 | 152,077 |
| SBL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0281 | 56,342 |
| SCBNL | yes | 133/133 | Asar 2072 | Shrawan 2083 | -0.0501 | 68,359 |
| Sanima | no | 59/133 | Chaitra 2074 | Shrawan 2083 | n/a | n/a |
| Sunrise | no | 95/133 | Asar 2072 | Jestha 2080 | -0.0438 | 41,272 |

dose_pre: included median -0.04376 (n=16) vs excluded median -0.0438 (n=9), Mann-Whitney p=0.246
pre-window size: included median 6.491e+04 (n=16) vs excluded median 4.451e+04 (n=9), Mann-Whitney p=0.005
Excluded banks whose last complete month falls inside the post window are candidate merger exits; check names/dates against NRB merger records before writing it up.

## HT/GPS weights (item 8)
| network | cells w>0 | mean | p99 / mean | max / mean | Kish ESS | ESS / cells |
|---|---|---|---|---|---|---|
| W_hat | 608 | 1.02 | 73.84 | 78.39 | 9 | 0.01 |
| RAS | 608 | 0.948 | 64.06 | 66.56 | 10 | 0.02 |

Where the weight sits: share of total HT weight per bank (top 5), and per period block.
- W_hat: NBL 0.45, NMB 0.28, ADBNL 0.26, Global 0.00, SCBNL 0.00; pre-window cells 0.00, post cells 1.00
- RAS: NBL 0.46, ADBNL 0.31, NMB 0.23, SCBNL 0.00, Global 0.00; pre-window cells 0.00, post cells 1.00

Trimmed HT weights (cap at the q-th percentile of positive weights, a standard IPW fix) vs main:

q = 0.95:
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.0332 | 0.022 | 0.158 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001436 | 8.94e-05 | 0.198 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.03563 | 0.0235 | 0.143 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0001217 | 7.38e-05 | 0.162 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.0009072 | 0.0263 | 0.972 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 9.422e-05 | 9.27e-05 | 0.320 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | -0.003937 | 0.0289 | 0.892 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 8.291e-05 | 9.64e-05 | 0.513 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | -6.057e-05 | 0.000672 | 0.939 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -1.326e-06 | 2.67e-06 | 0.660 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 2.408e-05 | 0.000649 | 0.974 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -2.881e-06 | 2.64e-06 | 0.348 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08448 | 0.0241 | 0.036 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002299 | 0.000104 | 0.159 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08759 | 0.0217 | 0.012 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002143 | 7.1e-05 | 0.093 | 167 |

q = 0.9:
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.02876 | 0.0215 | 0.191 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001239 | 9.12e-05 | 0.257 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.0303 | 0.0225 | 0.191 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0001211 | 7.03e-05 | 0.142 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.002466 | 0.024 | 0.910 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 6.834e-05 | 8.8e-05 | 0.544 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.0004626 | 0.0244 | 0.981 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 7.512e-05 | 7.88e-05 | 0.439 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | -0.0002849 | 0.000657 | 0.724 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -8.475e-07 | 2.81e-06 | 0.797 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | -6.354e-05 | 0.000693 | 0.936 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -2.705e-06 | 2.83e-06 | 0.418 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08295 | 0.0226 | 0.033 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002213 | 0.0001 | 0.160 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08741 | 0.0218 | 0.011 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002205 | 7.02e-05 | 0.086 | 167 |

The other weighting (ht) vs main (unweighted), same cells:
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.02704 | 0.0189 | 0.209 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001324 | 7.96e-05 | 0.196 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.03104 | 0.0199 | 0.162 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0001187 | 5.72e-05 | 0.103 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.008843 | 0.0227 | 0.675 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 6.565e-05 | 7.87e-05 | 0.454 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.008447 | 0.021 | 0.662 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 3.479e-05 | 6.66e-05 | 0.626 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 0.0004791 | 0.000609 | 0.439 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -2.792e-06 | 2.55e-06 | 0.301 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 0.0003148 | 0.000516 | 0.559 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -3.543e-06 | 2.49e-06 | 0.212 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08176 | 0.0233 | 0.042 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002205 | 0.000101 | 0.161 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08728 | 0.0196 | 0.009 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002143 | 6.49e-05 | 0.078 | 167 |

## GPS with period intercepts + per-period sigma (item 8 remedy)
The main GPS conditions only on frozen pre-mandate covariates, so its fitted mean is constant over time while exposure trends with W's nominal level; whole periods land in the density tails. Same regressors here, only the HT weights change.
| network | cells w>0 | mean | p99 / mean | max / mean | Kish ESS | ESS / cells |
|---|---|---|---|---|---|---|
| W_hat | 608 | 0.884 | 34.35 | 69.71 | 18 | 0.03 |
| RAS | 608 | 0.913 | 25.05 | 74.57 | 18 | 0.03 |

Where the weight sits: share of total HT weight per bank (top 5), and per period block.
- W_hat: NBL 0.50, Global 0.23, ADBNL 0.13, NMB 0.08, SCBNL 0.01; pre-window cells 0.23, post cells 0.77
- RAS: NBL 0.58, ADBNL 0.15, Global 0.10, NMB 0.08, SCBNL 0.01; pre-window cells 0.11, post cells 0.89

| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.01685 | 0.0211 | 0.422 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001106 | 7.76e-05 | 0.210 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.01097 | 0.0199 | 0.583 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -6.607e-05 | 5.36e-05 | 0.254 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.02395 | 0.0121 | 0.208 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 5.68e-05 | 4.32e-05 | 0.245 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.02659 | 0.0115 | 0.261 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 2.976e-05 | 3.76e-05 | 0.502 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 1.008e-05 | 0.000753 | 0.991 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -4.521e-06 | 3.91e-06 | 0.325 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 0.0003482 | 0.000892 | 0.741 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -6.405e-06 | 3.73e-06 | 0.118 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08973 | 0.0215 | 0.031 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002482 | 0.000101 | 0.126 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08866 | 0.0191 | 0.011 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002379 | 6.42e-05 | 0.060 | 167 |

## Negative de-cumulated monthly flows (item 11a)
pnl_ytd_cumulative=True. NIM and loans written off should be non-negative; provisions can be legitimately negative (write-backs), net profit can be either.
| outcome | finite cells | negative | share | worst FY (share) |
|---|---|---|---|---|
| net_profit_loss | 210 | 1 | 0.005 | FY2074 (0.02) |
| loan_loss_provision | 194 | 33 | 0.170 | FY2072 (0.20) |
| loan_written_off | 210 | 4 | 0.019 | FY2074 (0.05) |
| nim | 206 | 0 | 0.000 | FY2072 (0.00) |

Negative share by calendar month (Shrawan values are single months, never differenced):
| outcome | Shrawan | Bhadra | Ashwin | Kartik | Mangsir | Poush | Magh | Falgun | Chaitra | Baisakh | Jestha | Asar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| net_profit_loss | n/a | n/a | 0.00 | n/a | n/a | 0.00 | n/a | n/a | 0.02 | n/a | n/a | 0.00 |
| loan_loss_provision | n/a | n/a | 0.00 | n/a | n/a | 0.20 | n/a | n/a | 0.19 | n/a | n/a | 0.33 |
| loan_written_off | n/a | n/a | 0.00 | n/a | n/a | 0.02 | n/a | n/a | 0.02 | n/a | n/a | 0.05 |
| nim | n/a | n/a | 0.00 | n/a | n/a | 0.00 | n/a | n/a | 0.00 | n/a | n/a | 0.00 |

## Quarterly outcomes from quarter-end YTD only (item 11a remedy)
Flow at Ashwin, Poush, Chaitra, Asar = YTD minus the previous quarter end of the same fiscal year (Ashwin = YTD itself), divided by quarter-end size. Non-quarter-end months are never read. Regressors unchanged; non-quarter cells drop out as NaN outcomes.

| outcome | finite quarter cells | in pre window | negative share |
|---|---|---|---|
| net_profit_loss | 210 | 15 | 0.005 |
| loan_loss_provision | 194 | 12 | 0.170 |
| loan_written_off | 210 | 15 | 0.019 |
| nim | 206 | 13 | 0.000 |

Quarterly DiD, pooled-GPS HT weights (main = quarterly outcome, unweighted):
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.02704 | 0.0189 | 0.209 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001324 | 7.96e-05 | 0.196 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.03104 | 0.0199 | 0.162 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0001187 | 5.72e-05 | 0.103 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.008843 | 0.0227 | 0.675 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 6.565e-05 | 7.87e-05 | 0.454 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.008447 | 0.021 | 0.662 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 3.479e-05 | 6.66e-05 | 0.626 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 0.0004791 | 0.000609 | 0.439 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -2.792e-06 | 2.55e-06 | 0.301 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 0.0003148 | 0.000516 | 0.559 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -3.543e-06 | 2.49e-06 | 0.212 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08176 | 0.0233 | 0.042 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002205 | 0.000101 | 0.161 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08728 | 0.0196 | 0.009 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002143 | 6.49e-05 | 0.078 | 167 |

Quarterly DiD, unweighted (main = quarterly outcome, unweighted):
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.0247 | 0.0197 | 0.231 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -9.359e-05 | 8.73e-05 | 0.364 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.02807 | 0.0224 | 0.226 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0001136 | 7.21e-05 | 0.172 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.007646 | 0.0199 | 0.675 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 1.749e-05 | 7.24e-05 | 0.840 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.007518 | 0.0191 | 0.666 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 2.327e-05 | 6.29e-05 | 0.738 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | -0.0004529 | 0.000622 | 0.565 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -1.337e-06 | 3.11e-06 | 0.715 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | -0.0001336 | 0.000697 | 0.858 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -2.828e-06 | 2.96e-06 | 0.430 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08217 | 0.0237 | 0.040 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002041 | 0.000103 | 0.183 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08799 | 0.0219 | 0.011 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002184 | 7.16e-05 | 0.084 | 167 |

Quarterly DiD, period-effects GPS weights (main = quarterly outcome, unweighted):
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.01685 | 0.0211 | 0.422 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001106 | 7.76e-05 | 0.210 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.01097 | 0.0199 | 0.583 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -6.607e-05 | 5.36e-05 | 0.254 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.02395 | 0.0121 | 0.208 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | 5.68e-05 | 4.32e-05 | 0.245 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.02659 | 0.0115 | 0.261 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | 2.976e-05 | 3.76e-05 | 0.502 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 1.008e-05 | 0.000753 | 0.991 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -4.521e-06 | 3.91e-06 | 0.325 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 0.0003482 | 0.000892 | 0.741 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -6.405e-06 | 3.73e-06 | 0.118 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08973 | 0.0215 | 0.031 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002482 | 0.000101 | 0.126 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08866 | 0.0191 | 0.011 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002379 | 6.42e-05 | 0.060 | 167 |

## Same-quarter DiD: Ashwin (Q1) quarterly cells only (item 11a remedy, seasonality)
The quarterly pre window holds only Ashwin 2072 (Q1), while post cells span all four quarters, so a dose-related seasonal pattern would load onto dose x post. Keeping Q1 only compares like with like (one pre and a few post cells per bank).
Q1 columns: Ashwin 2072, Ashwin 2073, Ashwin 2074, Ashwin 2075

Q1-only DiD, unweighted:
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.06509 | 0.0421 | 0.075 | 63 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0001543 | 0.000137 | 0.305 | 63 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.09298 | 0.0493 | 0.035 | 63 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0002566 | 0.000143 | 0.099 | 63 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.05021 | 0.0127 | 0.035 | 56 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | -0.0001708 | 6.26e-05 | 0.126 | 56 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.05486 | 0.0108 | 0.016 | 56 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | -0.0001807 | 4.35e-05 | 0.041 | 56 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 0.0002249 | 0.000766 | 0.781 | 63 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -6.127e-06 | 6.54e-06 | 0.438 | 63 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 0.0003319 | 0.000708 | 0.650 | 63 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -6.125e-06 | 5.34e-06 | 0.367 | 63 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.05468 | 0.0289 | 0.144 | 61 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0001502 | 0.000126 | 0.326 | 61 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.06526 | 0.0282 | 0.060 | 61 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.000181 | 0.000104 | 0.147 | 61 |

Q1-only DiD, period-effects GPS weights:
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.1426 | 0.0633 | 0.153 | 63 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -0.0003933 | 0.000191 | 0.177 | 63 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.1573 | 0.0635 | 0.058 | 63 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.0004445 | 0.000155 | 0.036 | 63 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.07175 | 0.0172 | 0.072 | 56 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | -0.000226 | 7.7e-05 | 0.118 | 56 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.07314 | 0.0165 | 0.056 | 56 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | -0.0002262 | 6.25e-05 | 0.065 | 56 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 3.971e-05 | 0.000715 | 0.949 | 63 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -6.661e-06 | 7.1e-06 | 0.438 | 63 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | -2.436e-05 | 0.00079 | 0.973 | 63 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -6.29e-06 | 5.99e-06 | 0.395 | 63 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.08942 | 0.0311 | 0.144 | 61 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -0.0002493 | 0.000127 | 0.171 | 61 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.08861 | 0.0276 | 0.073 | 61 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0002436 | 9.26e-05 | 0.060 | 61 |

## Dose-only quarterly pre-trend from the clean 2070-2072 bulletins (item 3)
W_hat starts at Asar 2072, so exposure can't be built earlier; this is the dose-only event study on quarterly outcomes from Baisakh 2070. Same 16 banks, same frozen dose_pre (Asar-Mangsir 2072 mean) and Z_pre x post controls as the main run; bank + period FE, unweighted. Reference bin -1 = the pre-window fiscal year.
Extended panel (2070, 'Baisakh') .. (2075, 'Falgun'), T=68; months absent from the extraction: Chaitra 2071, Baisakh 2072, Jestha 2072. A quarter whose own or previous quarter-end month is absent is NaN.

Finite pre-mandate quarterly cells per bank (bins <= -1):
| bank | net_profit_loss | loan_loss_provision | loan_written_off | nim |
|---|---|---|---|---|
| ADBNL | 7 | 7 | 7 | 7 |
| EBL | 7 | 6 | 7 | 7 |
| Global | 0 | 0 | 0 | 0 |
| HBL | 7 | 7 | 7 | 7 |
| Kumari | 7 | 7 | 7 | 7 |
| Laxmi | 7 | 5 | 7 | 7 |
| MBL | 7 | 6 | 7 | 7 |
| NABIL | 7 | 7 | 7 | 7 |
| NBL | 7 | 7 | 7 | 7 |
| NIC | 7 | 4 | 7 | 7 |
| NMB | 2 | 0 | 2 | 0 |
| NSBI | 7 | 7 | 7 | 7 |
| Prime | 2 | 2 | 2 | 0 |
| RBB | 7 | 7 | 7 | 7 |
| SBL | 7 | 7 | 7 | 7 |
| SCBNL | 7 | 7 | 7 | 7 |

### net_profit_loss
n_obs=290, clusters=16
| bin | quarters | cells | dose beta | SE | wild p |
|---|---|---|---|---|---|
| -3 | Ash 2070, Pou 2070, Cha 2070, Asa 2071 | 52 | -0.01825 | 0.0161 | 0.376 |
| -2 | Ash 2071, Pou 2071 | 28 | 0.05767 | 0.0704 | 0.567 |
| -1 (ref) | Ash 2072 | 15 | 0 | - | - |
| transition | Pou 2072, Cha 2072, Asa 2073 | 41 | -0.005027 | 0.0203 | 0.810 |
| 0 | Ash 2073, Pou 2073, Cha 2073, Asa 2074 | 58 | -0.01283 | 0.0192 | 0.567 |
| 1 | Ash 2074, Pou 2074, Cha 2074, Asa 2075 | 64 | -0.006072 | 0.0229 | 0.829 |
| 2 | Ash 2075, Pou 2075 | 32 | -0.001967 | 0.0119 | 0.839 |

### loan_loss_provision
n_obs=268, clusters=16
| bin | quarters | cells | dose beta | SE | wild p |
|---|---|---|---|---|---|
| -3 | Ash 2070, Pou 2070, Cha 2070, Asa 2071 | 51 | 0.005771 | 0.00694 | 0.632 |
| -2 | Ash 2071, Pou 2071 | 23 | -0.006185 | 0.00569 | 0.307 |
| -1 (ref) | Ash 2072 | 12 | 0 | - | - |
| transition | Pou 2072, Cha 2072, Asa 2073 | 38 | 0.01968 | 0.00617 | 0.105 |
| 0 | Ash 2073, Pou 2073, Cha 2073, Asa 2074 | 54 | 0.008256 | 0.00898 | 0.411 |
| 1 | Ash 2074, Pou 2074, Cha 2074, Asa 2075 | 62 | 0.01385 | 0.0119 | 0.337 |
| 2 | Ash 2075, Pou 2075 | 28 | 0.01809 | 0.014 | 0.324 |

### loan_written_off
n_obs=290, clusters=16
| bin | quarters | cells | dose beta | SE | wild p |
|---|---|---|---|---|---|
| -3 | Ash 2070, Pou 2070, Cha 2070, Asa 2071 | 52 | -0.0006601 | 0.000794 | 0.443 |
| -2 | Ash 2071, Pou 2071 | 28 | -0.000247 | 0.0002 | 0.161 |
| -1 (ref) | Ash 2072 | 15 | 0 | - | - |
| transition | Pou 2072, Cha 2072, Asa 2073 | 41 | 0.000333 | 0.000389 | 0.474 |
| 0 | Ash 2073, Pou 2073, Cha 2073, Asa 2074 | 58 | -0.001944 | 0.00146 | 0.281 |
| 1 | Ash 2074, Pou 2074, Cha 2074, Asa 2075 | 64 | -0.002062 | 0.00154 | 0.273 |
| 2 | Ash 2075, Pou 2075 | 32 | -0.002023 | 0.00158 | 0.288 |

### nim
n_obs=284, clusters=16
| bin | quarters | cells | dose beta | SE | wild p |
|---|---|---|---|---|---|
| -3 | Ash 2070, Pou 2070, Cha 2070, Asa 2071 | 52 | -0.006975 | 0.0108 | 0.517 |
| -2 | Ash 2071, Pou 2071 | 26 | -0.005957 | 0.00615 | 0.425 |
| -1 (ref) | Ash 2072 | 13 | 0 | - | - |
| transition | Pou 2072, Cha 2072, Asa 2073 | 39 | 0.006957 | 0.0312 | 0.738 |
| 0 | Ash 2073, Pou 2073, Cha 2073, Asa 2074 | 58 | 0.007426 | 0.0346 | 0.856 |
| 1 | Ash 2074, Pou 2074, Cha 2074, Asa 2075 | 64 | 0.0339 | 0.028 | 0.340 |
| 2 | Ash 2075, Pou 2075 | 32 | 0.01518 | 0.0204 | 0.509 |

### Pooled: lead (bins < -1) and post, reference bin -1
1-SD effect = post beta x SD(dose_pre); pre mean = mean quarterly outcome over pre cells.
| outcome | sample | lead beta | lead SE | lead p | post beta | post SE | post p | 1-SD effect | pre mean | effect / pre mean | n_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| net_profit_loss | all quarters | 0.007819 | 0.0319 | 0.693 | -0.007274 | 0.0178 | 0.714 | -0.000146 | 0.00417 | -0.04 | 290 |
| net_profit_loss | Ashwin (Q1) only | -0.02197 | 0.0314 | 0.648 | 0.03002 | 0.0319 | 0.499 | 0.000604 | 0.00346 | 0.17 | 91 |
| net_profit_loss | from Asar 2072, all quarters | n/a | n/a | n/a | 0.006009 | 0.0206 | 0.797 | 0.000121 | 0.00354 | 0.03 | 169 |
| net_profit_loss | from Asar 2072, Q1 only | n/a | n/a | n/a | 0.03564 | 0.0258 | 0.252 | 0.000717 | 0.00354 | 0.20 | 63 |
| net_profit_loss | from Asar 2072, all quarters, + exposure | n/a | n/a | n/a | 0.0247 | 0.0197 | 0.231 | 0.000497 | 0.00354 | 0.14 | 169 |
| net_profit_loss | from Asar 2072, Q1 only, + exposure | n/a | n/a | n/a | 0.06509 | 0.0421 | 0.075 | 0.00131 | 0.00354 | 0.37 | 63 |
| loan_loss_provision | all quarters | 0.001852 | 0.00533 | 0.758 | 0.01192 | 0.0103 | 0.334 | 0.00024 | 0.000911 | 0.26 | 268 |
| loan_loss_provision | Ashwin (Q1) only | -0.01681 | 0.00856 | 0.121 | 0.009019 | 0.0138 | 0.555 | 0.000181 | 0.00143 | 0.13 | 81 |
| loan_loss_provision | from Asar 2072, all quarters | n/a | n/a | n/a | 0.01621 | 0.00869 | 0.126 | 0.000326 | 0.0011 | 0.30 | 156 |
| loan_loss_provision | from Asar 2072, Q1 only | n/a | n/a | n/a | 0.006777 | 0.0102 | 0.565 | 0.000136 | 0.0011 | 0.12 | 56 |
| loan_loss_provision | from Asar 2072, all quarters, + exposure | n/a | n/a | n/a | 0.007646 | 0.0199 | 0.675 | 0.000154 | 0.0011 | 0.14 | 156 |
| loan_loss_provision | from Asar 2072, Q1 only, + exposure | n/a | n/a | n/a | 0.05021 | 0.0127 | 0.035 | 0.00101 | 0.0011 | 0.92 | 56 |
| loan_written_off | all quarters | -0.0005192 | 0.000563 | 0.395 | -0.002003 | 0.0015 | 0.280 | -4.03e-05 | 3.08e-05 | -1.31 | 290 |
| loan_written_off | Ashwin (Q1) only | -0.0003314 | 0.000328 | 0.279 | -0.001409 | 0.00139 | 0.457 | -2.83e-05 | 1.02e-05 | -2.78 | 91 |
| loan_written_off | from Asar 2072, all quarters | n/a | n/a | n/a | -0.0006793 | 0.0008 | 0.555 | -1.37e-05 | 3.64e-06 | -3.75 | 169 |
| loan_written_off | from Asar 2072, Q1 only | n/a | n/a | n/a | -0.001382 | 0.00134 | 0.494 | -2.78e-05 | 3.64e-06 | -7.62 | 63 |
| loan_written_off | from Asar 2072, all quarters, + exposure | n/a | n/a | n/a | -0.0004529 | 0.000622 | 0.565 | -9.11e-06 | 3.64e-06 | -2.50 | 169 |
| loan_written_off | from Asar 2072, Q1 only, + exposure | n/a | n/a | n/a | 0.0002249 | 0.000766 | 0.781 | 4.52e-06 | 3.64e-06 | 1.24 | 63 |
| nim | all quarters | -0.006636 | 0.00858 | 0.437 | 0.01957 | 0.027 | 0.566 | 0.000394 | 0.00801 | 0.05 | 284 |
| nim | Ashwin (Q1) only | -0.02456 | 0.0131 | 0.158 | 0.008133 | 0.0314 | 0.820 | 0.000164 | 0.00721 | 0.02 | 87 |
| nim | from Asar 2072, all quarters | n/a | n/a | n/a | 0.04282 | 0.0272 | 0.210 | 0.000861 | 0.00684 | 0.13 | 167 |
| nim | from Asar 2072, Q1 only | n/a | n/a | n/a | 0.02411 | 0.0261 | 0.427 | 0.000485 | 0.00684 | 0.07 | 61 |
| nim | from Asar 2072, all quarters, + exposure | n/a | n/a | n/a | 0.08217 | 0.0237 | 0.040 | 0.00165 | 0.00684 | 0.24 | 167 |
| nim | from Asar 2072, Q1 only, + exposure | n/a | n/a | n/a | 0.05468 | 0.0289 | 0.144 | 0.0011 | 0.00684 | 0.16 | 61 |
SD(dose_pre) = 0.0201. A lead near 0 next to a post effect of similar size to the main run supports parallel pre-trends; a lead of the post's size and sign points to a pre-existing trend or mean reversion.

## Frozen pre-window size as denominator (item 11c)
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | 0.05012 | 0.0257 | 0.039 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -8.379e-05 | 9.08e-05 | 0.379 | 169 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | 0.06441 | 0.0383 | 0.053 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | -0.000167 | 0.000133 | 0.289 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.01822 | 0.0255 | 0.533 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | -9.156e-06 | 8.62e-05 | 0.916 | 156 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.01879 | 0.0243 | 0.505 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | -3.961e-06 | 7.71e-05 | 0.951 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | -0.0006078 | 0.00102 | 0.626 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | -2.291e-06 | 5e-06 | 0.696 | 169 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 1.906e-05 | 0.00122 | 0.990 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | -5.224e-06 | 4.94e-06 | 0.365 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.1053 | 0.0536 | 0.264 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -9.946e-05 | 0.000175 | 0.670 | 167 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.1087 | 0.052 | 0.256 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -0.0001055 | 0.000154 | 0.591 | 167 |

## Leave-one-bank-out (item 6), CR1 only; design and weights held fixed
| outcome | network | coef | full beta | LOO min | LOO max | sign flips | max abs t | most influential bank |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.00905 | 0.0415 | 0/16 | 1.79 | HBL (0.0415) |
| net_profit_loss | W_hat | exposure | -9.359e-05 | -0.0001778 | 0.0002103 | 1/16 | 2.34 | ADBNL (0.0002103) |
| net_profit_loss | RAS | dose | 0.02807 | 0.01766 | 0.04491 | 0/16 | 1.73 | HBL (0.04491) |
| net_profit_loss | RAS | exposure | -0.0001136 | -0.0001541 | -2.903e-05 | 0/16 | 2.28 | ADBNL (-2.903e-05) |
| loan_loss_provision | W_hat | dose | 0.007646 | -0.01351 | 0.05443 | 2/16 | 5.93 | NBL (0.05443) |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | -0.0001388 | 9.57e-05 | 3/16 | 3.84 | ADBNL (-0.0001388) |
| loan_loss_provision | RAS | dose | 0.007518 | -0.01427 | 0.05193 | 2/16 | 6.89 | NBL (0.05193) |
| loan_loss_provision | RAS | exposure | 2.327e-05 | -0.0001911 | 9.822e-05 | 3/16 | 3.89 | ADBNL (-0.0001911) |
| loan_written_off | W_hat | dose | -0.0004529 | -0.0008225 | 0.0001823 | 2/16 | 1.21 | NBL (0.0001823) |
| loan_written_off | W_hat | exposure | -1.337e-06 | -4.03e-06 | 2.087e-06 | 2/16 | 1.57 | ADBNL (2.087e-06) |
| loan_written_off | RAS | dose | -0.0001336 | -0.0003954 | 0.0007421 | 3/16 | 0.98 | NBL (0.0007421) |
| loan_written_off | RAS | exposure | -2.828e-06 | -4.916e-06 | 1.372e-07 | 1/16 | 1.67 | Laxmi (1.372e-07) |
| nim | W_hat | dose | 0.08217 | 0.04193 | 0.09329 | 0/16 | 5.03 | NBL (0.04193) |
| nim | W_hat | exposure | -0.0002041 | -0.0003169 | 0.0004316 | 1/16 | 9.13 | ADBNL (0.0004316) |
| nim | RAS | dose | 0.08799 | 0.05382 | 0.1001 | 0/16 | 5.16 | NBL (0.05382) |
| nim | RAS | exposure | -0.0002184 | -0.0002749 | 0.0005673 | 1/16 | 7.06 | ADBNL (0.0005673) |

## Marginal balance (Sec 8 follow-up): can any matrix match both published marginals?
Within the 16-bank closed system a matrix with zero diagonal can match r (lending) and c (borrowing) only if sum_i r_i = sum_j c_j in that month and no bank lends more than all others borrow (r_i <= sum_{j != i} c_j) or borrows more than all others lend. Where either fails, no reconstruction can be exact and a projection returns a compromise. The per-entry cap is a further constraint, not checked here.
- months: 45; |sum r - sum c| / max > 1%: 43, > 10%: 37; median relative gap 0.324, max 0.928
- sum r > sum c in 30 months, < in 15; max |gap| 13,296 (marginal units)
- bank-months with r_i > sum_(j!=i) c_j: 16; with c_i > sum_(j!=i) r_j: 2
- months failing at least one condition: 43 of 45
- W_hat: max |marginal dev| per month, median in failing months 200.8 (n=43), in the others 22.28 (n=2); Spearman(|gap|, dev) = 0.926
- RAS: max |marginal dev| per month, median in failing months 1,136 (n=43), in the others 13.51 (n=2); Spearman(|gap|, dev) = 0.927

Ten months with the largest |gap|:
| month | sum r | sum c | gap | rel | banks over | W_hat row dev | W_hat col dev | RAS row dev | RAS col dev |
|---|---|---|---|---|---|---|---|---|---|
| Poush 2075 | 6,904 | 20,200 | -13,296 | 0.658 | 1 | 833.4 | 5.077 | 0.0004883 | 6,055 |
| Mangsir 2072 | 10,322 | 743 | +9,579 | 0.928 | 4 | 598.7 | 437.8 | 0.0002441 | 8,290 |
| Kartik 2072 | 9,807 | 1,435 | +8,371 | 0.854 | 4 | 523.2 | 332.6 | 0.0002441 | 3,706 |
| Falgun 2072 | 13,224 | 5,221 | +8,003 | 0.605 | 0 | 2,170 | 261.8 | 0.0004883 | 2,614 |
| Magh 2072 | 7,726 | 1,356 | +6,370 | 0.825 | 2 | 398.4 | 284.1 | 41.35 | 3,289 |
| Poush 2072 | 10,616 | 5,533 | +5,083 | 0.479 | 1 | 380.4 | 323.6 | 0.0004883 | 2,261 |
| Kartik 2075 | 15,139 | 10,404 | +4,735 | 0.313 | 0 | 323.9 | 309.2 | 0.0004883 | 3,109 |
| Ashwin 2075 | 14,014 | 9,517 | +4,497 | 0.321 | 0 | 281.6 | 148.6 | 0.0004883 | 1,606 |
| Poush 2073 | 5,187 | 9,616 | -4,429 | 0.461 | 0 | 278.9 | 4.006 | 6.104e-05 | 967.2 |
| Bhadra 2074 | 9,079 | 4,862 | +4,217 | 0.464 | 0 | 263.6 | 148.6 | 0.0009766 | 1,735 |
If the deviations sit in the failing months and track |gap|, the misfit is the data (an unbalanced closed system), not the projection; if not, look at the cap or the solver.

## Row-normalised exposure mapping (item 12): exposure_i = sum_j (W_ij / sum_k W_ik) dose_pre_j
Weights: unweighted (HT ones re-estimated from the row-normalised exposure); banks with no interbank row in a month get exposure 0.
- published marginals: r <= 0 on 398, c <= 0 on 465 of 720 bank-months; median r 0, median c 0
| matrix | sum over | zero sums | max abs dev vs r | max abs dev vs c |
|---|---|---|---|---|
| W_hat | rows (j) | 169 | 2311 | 8367 |
| W_hat | cols (i) | 0 | 9200 | 437.8 |
| RAS | rows (j) | 398 | 41.35 | 9200 |
| RAS | cols (i) | 465 | 9939 | 8290 |
- on the 398 bank-months where the RAS row is 0: median r 0, median c 0, median W_hat row sum 27.1
- rows set to 0 before row-normalising: those with r <= 0 (398)
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | -0.01632 | 0.0181 | 0.529 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | 0.062 | 0.0204 | 0.161 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.02064 | 0.00819 | 0.035 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | -0.009288 | 0.00808 | 0.389 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | -0.001051 | 0.000912 | 0.421 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | 0.001346 | 0.00115 | 0.359 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.03378 | 0.0255 | 0.317 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | 0.01291 | 0.0294 | 0.872 | 167 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | -0.01367 | 0.0186 | 0.602 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | 0.06475 | 0.0293 | 0.267 | 169 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.02111 | 0.0078 | 0.023 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | -0.01067 | 0.0103 | 0.432 | 156 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | -0.000975 | 0.000909 | 0.474 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | 0.000859 | 0.00112 | 0.539 | 169 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.03558 | 0.0249 | 0.299 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | 0.007643 | 0.0436 | 0.955 | 167 |

## Centrality-based exposure (item 12b, exploratory): exposure_i = [W~ (I - 0.5 W~)^-1 dose_pre]_i, W~ row-normalised
Katz-Bonacich exposure, alpha = 0.5 fixed in advance. Leverage dose, Eq. (2), weights unweighted. Not part of any main-run test family; Holm below is within this check's exposure coefficients only.
- published marginals: r <= 0 on 398, c <= 0 on 465 of 720 bank-months; median r 0, median c 0
| matrix | sum over | zero sums | max abs dev vs r | max abs dev vs c |
|---|---|---|---|---|
| W_hat | rows (j) | 169 | 2311 | 8367 |
| W_hat | cols (i) | 0 | 9200 | 437.8 |
| RAS | rows (j) | 398 | 41.35 | 9200 |
| RAS | cols (i) | 465 | 9939 | 8290 |
- on the 398 bank-months where the RAS row is 0: median r 0, median c 0, median W_hat row sum 27.1
- rows set to 0 before row-normalising: those with r <= 0 (398)
- W_hat: corr(centrality exposure, direct exposure) on post cells = 0.516 (512 cells)
- W_hat: corr(centrality exposure, row-normalised direct exposure) on post cells = 0.965 (512 cells)
- RAS: corr(centrality exposure, direct exposure) on post cells = 0.548 (512 cells)
- RAS: corr(centrality exposure, row-normalised direct exposure) on post cells = 0.978 (512 cells)

| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | -0.01624 | 0.019 | 0.549 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | 0.06387 | 0.02 | 0.163 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.02039 | 0.00881 | 0.085 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | -0.01188 | 0.00849 | 0.369 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | -0.001064 | 0.000911 | 0.414 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | 0.001236 | 0.00102 | 0.340 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | 0.03453 | 0.0274 | 0.341 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | 0.01746 | 0.0315 | 0.896 | 167 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | -0.01226 | 0.0193 | 0.653 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | 0.06449 | 0.0277 | 0.282 | 169 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.02089 | 0.00786 | 0.019 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | -0.01055 | 0.00972 | 0.381 | 156 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | -0.0009519 | 0.000904 | 0.485 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | 0.0008594 | 0.00106 | 0.525 | 169 |
| nim | RAS | dose | 0.08799 | 0.011 | 0.03711 | 0.0266 | 0.312 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | 0.01087 | 0.0473 | 0.953 | 167 |

| outcome | network | exposure wild p | Holm p |
|---|---|---|---|
| net_profit_loss | W_hat | 0.163 | 1.000 |
| loan_loss_provision | W_hat | 0.369 | 1.000 |
| loan_written_off | W_hat | 0.340 | 1.000 |
| nim | W_hat | 0.896 | 1.000 |
| net_profit_loss | RAS | 0.282 | 1.000 |
| loan_loss_provision | RAS | 0.381 | 1.000 |
| loan_written_off | RAS | 0.525 | 1.000 |
| nim | RAS | 0.953 | 1.000 |

## Loans written off: blank read as missing vs blank read as 0 (item 11b)
The bulletin often prints no written-off row. 'Blank = missing' leaves it NaN; 'blank = 0' reads it as 0 wherever the bank reported some other P&L line that month (nothing written off yet this fiscal year); with no P&L row at all it stays missing. Zero-filling happens on the extracted (fiscal-YTD) figure, before differencing; a 0 after a positive YTD in the same fiscal year then shows up as a negative flow. This config's main version (blank_as_zero): blank = 0 (user decision 2026-11-04).

| bank | written-off reported | of which exactly 0 | blank, other P&L present (-> 0) | no P&L row (stays missing) |
|---|---|---|---|---|
| ADBNL | 23 | 8 | 22 | 0 |
| EBL | 1 | 0 | 44 | 0 |
| Global | 0 | 0 | 33 | 12 |
| HBL | 0 | 0 | 45 | 0 |
| Kumari | 0 | 0 | 45 | 0 |
| Laxmi | 28 | 0 | 17 | 0 |
| MBL | 6 | 0 | 39 | 0 |
| NABIL | 42 | 0 | 3 | 0 |
| NBL | 0 | 0 | 45 | 0 |
| NIC | 1 | 0 | 44 | 0 |
| NMB | 8 | 0 | 26 | 11 |
| NSBI | 0 | 0 | 45 | 0 |
| Prime | 0 | 0 | 34 | 11 |
| RBB | 0 | 0 | 45 | 0 |
| SBL | 2 | 0 | 43 | 0 |
| SCBNL | 44 | 0 | 1 | 0 |
| total | 155 | 8 | 531 | 34 |
Explicit zeros in the extraction would suggest a blank means 'not reported', not 0.

Regression outcome (quarterly, / size):
| version | finite cells | banks with any | negative share |
|---|---|---|---|
| blank = missing | 45 | 6 | 0.044 |
| blank = 0 | 210 | 16 | 0.019 |

Eq. A' (unweighted); 'main' columns = blank as missing, 'alt' = blank as 0:
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| loan_written_off | W_hat | dose | 9.722e-07 | 0.562 | -0.0004529 | 0.000622 | 0.565 | 169 |
| loan_written_off | W_hat | exposure | 1.241e-06 | 0.540 | -1.337e-06 | 3.11e-06 | 0.715 | 169 |
| loan_written_off | RAS | dose | 1.07e-06 | 0.555 | -0.0001336 | 0.000697 | 0.858 | 169 |
| loan_written_off | RAS | exposure | 1.148e-06 | 0.543 | -2.828e-06 | 2.96e-06 | 0.430 | 169 |
Clusters (missing -> 0): W_hat 6 -> 16, RAS 6 -> 16

Headline total effect (dose-only, quarterly, from Baisakh 2070):
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| blank = missing | 0.07709 | 2.65 | 0.969 | 0.008362 | 0.109 | 0.00155 | 6.54e-05 | 59 | 7 | - |
| blank = 0 | -0.002003 | 0.0015 | 0.280 | -0.0005192 | 0.395 | -4.03e-05 | 3.08e-05 | 290 | 16 | - |

## Paid-up capital as an alternative dose
dose_pu = (required paid-up capital - paid-up capital) / size, frozen at its pre-window mean like the main dose; same pre window, transition and post as the leverage dose (if the paid-up directive has its own dates, that needs its own timing). The balance-sheet row is matched on keywords 'PAID' + 'CAPITAL', not confirmed against a printed bulletin: check the matched labels below.

Matched labels (panel cells filled): 'A. PAID-UP CAPITAL' (2128)
Cells with a second, different matched value (ignored, first kept): 0
Other labels containing CAPITAL (not used): 1 CAPITAL FUND

Per bank, pre window Asar 2072 .. Mangsir 2072 (NPR mn):
| bank | pre cells | paid-up (pre mean) | size (pre mean) | paid-up / size | last paid-up / pre | main dose_pre |
|---|---|---|---|---|---|---|
| ADBNL | 6 | 9,861 | 120,083 | 0.0821 | 1.41 | -0.0809 |
| EBL | 6 | 2,137 | 91,391 | 0.0234 | 3.79 | -0.0344 |
| Global | 6 | 5,204 | 73,660 | 0.0706 | 1.98 | -0.0573 |
| HBL | 6 | 3,333 | 89,690 | 0.0372 | 2.56 | -0.0400 |
| Kumari | 6 | 2,432 | 41,610 | 0.0584 | 2.95 | -0.0405 |
| Laxmi | 6 | 2,777 | 49,876 | 0.0557 | 2.96 | -0.0443 |
| MBL | 6 | 2,776 | 51,715 | 0.0537 | 2.90 | -0.0394 |
| NABIL | 6 | 3,658 | 130,822 | 0.0280 | 2.20 | -0.0203 |
| NBL | 6 | 6,465 | 90,513 | 0.0714 | 1.52 | 0.0003 |
| NIC | 6 | 2,658 | 64,001 | 0.0415 | 3.32 | -0.0484 |
| NMB | 6 | 2,985 | 47,823 | 0.0624 | 3.22 | -0.0493 |
| NSBI | 6 | 3,058 | 64,912 | 0.0471 | 2.76 | -0.0438 |
| Prime | 6 | 3,140 | 49,098 | 0.0640 | 2.56 | -0.0492 |
| RBB | 6 | 8,589 | 152,077 | 0.0565 | 1.05 | -0.0008 |
| SBL | 6 | 2,031 | 56,342 | 0.0361 | 4.38 | -0.0281 |
| SCBNL | 6 | 2,248 | 68,359 | 0.0329 | 3.56 | -0.0501 |
Pre-window paid-up capital: median 2,985, min 2,031, max 9,861 (n=16); corr(paid-up / size, main dose_pre) = -0.261
'last paid-up / pre' uses the last month in this run's window (Falgun 2075).

Requirement 8,000 NPR mn: 14/16 banks below it pre-mandate, dose_pu_pre range [-0.0155, 0.1339], SD 0.0438 (main dose SD 0.0201). Betas are per unit of each dose; compare the 1-SD effects.

Eq. A' (unweighted); main = leverage dose, alt = paid-up dose (exposure rebuilt from it):
| outcome | network | coef | main beta | main p | alt beta | alt SE | alt p | alt n_obs |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | W_hat | dose | 0.0247 | 0.231 | -0.02582 | 0.0207 | 0.285 | 169 |
| net_profit_loss | W_hat | exposure | -9.359e-05 | 0.364 | -6.836e-06 | 3.27e-05 | 0.853 | 169 |
| loan_loss_provision | W_hat | dose | 0.007646 | 0.675 | 0.01285 | 0.0176 | 0.679 | 156 |
| loan_loss_provision | W_hat | exposure | 1.749e-05 | 0.840 | -1.378e-07 | 1.67e-05 | 0.994 | 156 |
| loan_written_off | W_hat | dose | -0.0004529 | 0.565 | 0.0003258 | 0.000473 | 0.607 | 169 |
| loan_written_off | W_hat | exposure | -1.337e-06 | 0.715 | 8.444e-07 | 1e-06 | 0.516 | 169 |
| nim | W_hat | dose | 0.08217 | 0.040 | -0.0801 | 0.0174 | 0.121 | 167 |
| nim | W_hat | exposure | -0.0002041 | 0.183 | -4.587e-05 | 3.27e-05 | 0.405 | 167 |
| net_profit_loss | RAS | dose | 0.02807 | 0.226 | -0.01134 | 0.0171 | 0.557 | 169 |
| net_profit_loss | RAS | exposure | -0.0001136 | 0.172 | 1.359e-05 | 2.69e-05 | 0.673 | 169 |
| loan_loss_provision | RAS | dose | 0.007518 | 0.666 | 0.007582 | 0.0134 | 0.770 | 156 |
| loan_loss_provision | RAS | exposure | 2.327e-05 | 0.738 | -5.436e-06 | 1.34e-05 | 0.711 | 156 |
| loan_written_off | RAS | dose | -0.0001336 | 0.858 | 0.0006176 | 0.000424 | 0.352 | 169 |
| loan_written_off | RAS | exposure | -2.828e-06 | 0.430 | 1.317e-06 | 9e-07 | 0.294 | 169 |
| nim | RAS | dose | 0.08799 | 0.011 | -0.07409 | 0.0179 | 0.175 | 167 |
| nim | RAS | exposure | -0.0002184 | 0.084 | -3.205e-05 | 2.94e-05 | 0.439 | 167 |

Headline total effect (dose-only, quarterly), leverage dose vs paid-up dose:

net_profit_loss:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| leverage dose | -0.007274 | 0.0178 | 0.714 | 0.007819 | 0.693 | -0.000146 | 0.00417 | 290 | 16 | - |
| paid-up dose | -0.04135 | 0.0109 | 0.021 | -0.002295 | 0.872 | -0.00181 | 0.00417 | 290 | 16 | - |

loan_loss_provision:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| leverage dose | 0.01192 | 0.0103 | 0.334 | 0.001852 | 0.758 | 0.00024 | 0.000911 | 268 | 16 | - |
| paid-up dose | -0.001022 | 0.00786 | 0.908 | 0.000219 | 0.946 | -4.48e-05 | 0.000911 | 268 | 16 | - |

loan_written_off:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| leverage dose | -0.002003 | 0.0015 | 0.280 | -0.0005192 | 0.395 | -4.03e-05 | 3.08e-05 | 290 | 16 | - |
| paid-up dose | 4.195e-05 | 0.000694 | 0.948 | 0.0005412 | 0.249 | 1.84e-06 | 3.08e-05 | 290 | 16 | - |

nim:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| leverage dose | 0.01957 | 0.027 | 0.566 | -0.006636 | 0.437 | 0.000394 | 0.00801 | 284 | 16 | - |
| paid-up dose | -0.05721 | 0.0078 | 0.002 | 0.0008399 | 0.839 | -0.00251 | 0.00801 | 284 | 16 | - |

### Paid-up dose robustness (total effect)
dose_pu_pre = size term (req / size, frozen) + paid-up term (-paid-up / size, frozen); max |sum - dose_pu_pre| = 2.98e-08. Size term range [0.0526, 0.1923], paid-up term [-0.0821, -0.0234]; corr(size term, dose_pu) = 0.924, corr(paid-up term, dose_pu) = 0.263, corr(size term, log size_pre) = -0.984. Every fit below already has log size_pre x post among its controls.

(a) Decomposition: both terms entered together (each with its own lead / transition / post). A binding paid-up floor predicts the same post beta on both (same units); a pure size effect puts it on the size term only. The size term is close to a function of log size_pre, so expect it to be imprecise.
| outcome | combined beta (p) | size term beta | SE | wild p | 1-SD | paid-up term beta | SE | wild p | 1-SD |
|---|---|---|---|---|---|---|---|---|---|
| net_profit_loss | -0.04135 (0.021) | -0.1056 | 0.0439 | 0.177 | -0.0045 | -0.04097 | 0.0124 | 0.016 | -0.000694 |
| loan_loss_provision | -0.001022 (0.908) | 0.03923 | 0.0179 | 0.210 | 0.00167 | 0.009876 | 0.00703 | 0.296 | 0.000167 |
| loan_written_off | 4.195e-05 (0.948) | -0.0001813 | 0.00309 | 0.944 | -7.72e-06 | -8.119e-05 | 0.000648 | 0.898 | -1.37e-06 |
| nim | -0.05721 (0.002) | 0.01063 | 0.018 | 0.677 | 0.000453 | -0.0622 | 0.0126 | 0.012 | -0.00105 |

(c) Outcomes divided by the frozen pre-window size (item 11c) instead of current size, so asset growth from capital raising can't shrink the ratio:

net_profit_loss:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| current size | -0.04135 | 0.0109 | 0.021 | -0.002295 | 0.872 | -0.00181 | 0.00417 | 290 | 16 | - |
| frozen pre size | -0.01974 | 0.0152 | 0.561 | 0.0009223 | 0.949 | -0.000865 | 0.00351 | 290 | 16 | - |

loan_loss_provision:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| current size | -0.001022 | 0.00786 | 0.908 | 0.000219 | 0.946 | -4.48e-05 | 0.000911 | 268 | 16 | - |
| frozen pre size | -0.002236 | 0.0112 | 0.865 | 8.072e-06 | 0.997 | -9.8e-05 | 0.000767 | 268 | 16 | - |

loan_written_off:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| current size | 4.195e-05 | 0.000694 | 0.948 | 0.0005412 | 0.249 | 1.84e-06 | 3.08e-05 | 290 | 16 | - |
| frozen pre size | -0.0001745 | 0.000801 | 0.826 | 0.0004542 | 0.237 | -7.64e-06 | 2.49e-05 | 290 | 16 | - |

nim:
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| current size | -0.05721 | 0.0078 | 0.002 | 0.0008399 | 0.839 | -0.00251 | 0.00801 | 284 | 16 | - |
| frozen pre size | -0.02122 | 0.0158 | 0.374 | -0.0005472 | 0.888 | -0.00093 | 0.00675 | 284 | 16 | - |

(d) Leave-one-bank-out, CR1 only: post beta (t) with that bank's outcomes dropped.
| dropped | net_profit_loss | loan_loss_provision | loan_written_off | nim |
|---|---|---|---|---|
| none | -0.04135 (-3.81) | -0.001022 (-0.13) | 4.195e-05 (0.06) | -0.05721 (-7.34) |
| ADBNL | -0.02968 (-2.07) | 0.001872 (0.15) | 0.001201 (0.89) | -0.0576 (-5.09) |
| EBL | -0.04388 (-3.29) | 0.0006687 (0.07) | -0.0004769 (-0.55) | -0.057 (-5.92) |
| Global | -0.04134 (-3.80) | -0.001106 (-0.14) | 4.204e-05 (0.06) | -0.05721 (-7.32) |
| HBL | -0.04363 (-3.47) | 0.0005285 (0.06) | 2.707e-05 (0.04) | -0.05651 (-6.60) |
| Kumari | -0.04574 (-4.68) | 0.0005803 (0.07) | 0.0001596 (0.21) | -0.06013 (-7.52) |
| Laxmi | -0.04116 (-3.45) | 0.002069 (0.29) | 0.000412 (0.75) | -0.05646 (-7.23) |
| MBL | -0.041 (-3.88) | -0.0017 (-0.21) | -7.49e-05 (-0.13) | -0.05696 (-6.32) |
| NABIL | -0.05252 (-3.30) | 0.005536 (0.65) | 0.0005843 (0.76) | -0.04912 (-5.17) |
| NBL | -0.01975 (-1.28) | -0.01585 (-1.25) | -0.0009341 (-0.56) | -0.07093 (-6.09) |
| NIC | -0.04012 (-3.71) | -0.0005152 (-0.07) | 4.533e-05 (0.07) | -0.0554 (-7.30) |
| NMB | -0.04181 (-3.81) | -0.001057 (-0.13) | 4.603e-05 (0.07) | -0.05721 (-7.32) |
| NSBI | -0.04127 (-3.95) | -0.001147 (-0.14) | 6.091e-05 (0.08) | -0.05561 (-7.54) |
| Prime | -0.04316 (-3.83) | -0.002797 (-0.35) | 3.994e-05 (0.06) | -0.05721 (-7.32) |
| RBB | -0.03629 (-5.71) | -0.0008329 (-0.19) | 3.705e-05 (0.05) | -0.05831 (-11.88) |
| SBL | -0.03989 (-3.24) | -0.003937 (-0.45) | 0.0001068 (0.15) | -0.06073 (-6.79) |
| SCBNL | -0.03873 (-3.05) | -0.001122 (-0.12) | -0.0002289 (-0.29) | -0.06353 (-5.94) |
net_profit_loss: LOO range [-0.05252, -0.01975], sign flips 0/16, most influential NBL (-0.01975)
loan_loss_provision: LOO range [-0.01585, 0.005536], sign flips 6/16, most influential NBL (-0.01585)
loan_written_off: LOO range [-0.0009341, 0.001201], sign flips 4/16, most influential ADBNL (0.001201)
nim: LOO range [-0.07093, -0.04912], sign flips 0/16, most influential NBL (-0.07093)

(f) First stage and (g) asset growth: the same total-effect regression on balance-sheet stocks at quarter-end months (no differencing, no size scaling). 'paid_up / size_pre' is in the dose's units: beta = post-period paid-up raised per unit of pre-window shortfall, averaged over all post quarters (so below 1 even under full compliance, since post starts a year before the deadline). 'log size' tests the denominator channel behind (c).
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| paid_up / size_pre | 0.3575 | 0.086 | 0.005 | -0.03744 | 0.279 | 0.0157 | 0.0444 | 352 | 16 | - |
| log paid_up | 9.991 | 2.21 | 0.009 | -0.8303 | 0.252 | 0.438 | 8.01 | 352 | 16 | - |
| log size | 3.037 | 1.2 | 0.062 | -0.8168 | 0.088 | 0.133 | 11 | 352 | 16 | - |

(h) Horse race: paid-up dose and leverage dose entered together (each with its own lead / transition / post).
| outcome | paid-up beta | SE | wild p | 1-SD | leverage beta | SE | wild p | 1-SD |
|---|---|---|---|---|---|---|---|---|
| net_profit_loss | -0.04166 | 0.0123 | 0.127 | -0.00182 | -0.01015 | 0.0132 | 0.502 | -0.000204 |
| loan_loss_provision | -0.0002355 | 0.0083 | 0.986 | -1.03e-05 | 0.01201 | 0.0106 | 0.424 | 0.000242 |
| loan_written_off | -4.413e-05 | 0.000817 | 0.964 | -1.93e-06 | -0.001894 | 0.00144 | 0.326 | -3.81e-05 |
| nim | -0.05666 | 0.00964 | 0.017 | -0.00248 | 0.01505 | 0.0102 | 0.159 | 0.000303 |
corr(dose_pu_pre, leverage dose_pre) = -0.180

## Paid-up capital path: when each bank reaches the requirement
Requirement 8,000 NPR mn, deadline Asar end 2074 (Unified Directive 2073, dir. 23 pt. 4). Panel (2070, 'Baisakh') .. (2075, 'Falgun') (paid-up after Falgun 2075 is in the corrupted bulletins, Sec 5a). 'first month >=' is the first bulletin month at or above the requirement; 'vs deadline' = months after the deadline (negative = early); 'months below after' counts later months back under it.

| bank | Asar 2070 | Asar 2071 | Asar 2072 | Mangsir 2072 | Asar 2073 | Asar 2074 | Asar 2075 | Falgun 2075 | first month >= | vs deadline | months below after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ADBNL | 9,637 | 9,637 | 9,861 | 9,861 | 10,374 | 12,520 | 13,938 | 13,938 | Baisakh 2070 | -50 | 0 |
| EBL | 1,761 | 1,921 | 2,137 | 2,137 | 2,743 | 4,606 | 8,107 | 8,107 | Poush 2074 | +6 | 0 |
| Global | 2,418 | 4,107 | 5,012 | 6,164 | 6,164 | 8,080 | 8,888 | 10,311 | Asar 2074 | +0 | 0 |
| HBL | 2,760 | 2,898 | 3,333 | 3,333 | 4,499 | 6,492 | 8,115 | 8,520 | Poush 2074 | +6 | 0 |
| Kumari | 1,604 | 1,828 | 2,432 | 2,432 | 2,699 | 6,208 | 7,163 | 7,163 | never | n/a | n/a |
| Laxmi | 1,694 | 1,948 | 2,338 | 3,039 | 3,039 | 6,318 | 8,222 | 8,222 | Kartik 2074 | +4 | 0 |
| MBL | 2,479 | 2,479 | 2,776 | 2,776 | 3,865 | 6,599 | 8,056 | 8,056 | Mangsir 2074 | +5 | 0 |
| NABIL | 2,437 | 3,047 | 3,658 | 3,658 | 4,757 | 6,186 | 8,043 | 8,043 | Ashwin 2074 | +3 | 0 |
| NBL | 3,716 | 6,465 | 6,465 | 6,465 | 6,465 | 8,043 | 8,043 | 9,811 | Asar 2074 | +0 | 0 |
| NIC | 2,312 | 2,312 | 2,658 | 2,658 | 4,575 | 6,693 | 8,031 | 8,834 | Bhadra 2074 | +2 | 0 |
| NMB | 2,000 | 2,000 | 2,400 | 4,155 | 4,487 | 6,462 | 7,603 | 9,618 | Ashwin 2075 | +15 | 0 |
| NSBI | 2,356 | 2,650 | 3,058 | 3,058 | 3,884 | 6,925 | 8,047 | 8,449 | Poush 2074 | +6 | 0 |
| Prime | 2,340 | 2,639 | 3,140 | 3,140 | 3,705 | 6,325 | 8,033 | 8,033 | Chaitra 2074 | +9 | 0 |
| RBB | 8,589 | 8,589 | 8,589 | 8,589 | 8,589 | 8,589 | 9,005 | 9,005 | Baisakh 2070 | -50 | 0 |
| SBL | 1,619 | 1,814 | 2,031 | 2,031 | 3,022 | 5,251 | 8,464 | 8,888 | Baisakh 2075 | +10 | 0 |
| SCBNL | 1,854 | 2,042 | 2,248 | 2,248 | 2,812 | 4,006 | 8,011 | 8,011 | Poush 2074 | +6 | 0 |
Reached 8,000 by the deadline: 4/16; within the panel at all: 15/16.
Anticipation (Sec 4a timing caveat): share of each bank's Asar 2072 .. Falgun 2075 paid-up increase already in place by Mangsir 2072: median 0.00, range [0.00, 0.24] (n=16); banks with any increase inside the pre window: Global, Laxmi, NMB.

## Threats to the paid-up result (Sec 5b item 1): pre-trend, mergers, catch-up

### 1. Pre-trend in the balance-sheet stocks
dose_pu x fiscal-year bin on quarter-end stocks; reference bin -1 = the pre-window fiscal year (its quarter-ends up to Mangsir 2072); Z_pre x post controls, bank + period FE, unweighted. A negative pre bin means high-dose banks were lower then relative to the reference, i.e. they were growing faster into it. Cells: beta (SE, wild p).
| bin | months | paid_up / size_pre | log paid_up | log size |
|---|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | -0.0256 (0.064, p 0.696) | -0.58 (1.3, p 0.669) | -0.716 (0.63, p 0.267) |
| -3 | Shr 2070 - Asa 2071 | -0.0462 (0.045, p 0.334) | -1.07 (0.85, p 0.268) | -1.12 (0.5, p 0.081) |
| -2 | Shr 2071 - Asa 2072 | -0.0297 (0.022, p 0.151) | -0.592 (0.42, p 0.156) | -0.448 (0.22, p 0.061) |
| -1 | Shr 2072 - Man 2072 | 0 (ref, 16 cells) | 0 (ref, 16 cells) | 0 (ref, 16 cells) |
| transition | Pou 2072 - Asa 2073 | 0.0859 (0.055, p 0.146) | 1.29 (0.88, p 0.186) | 0.598 (0.67, p 0.435) |
| 0 | Shr 2073 - Asa 2074 | 0.107 (0.11, p 0.405) | 7.91 (2.4, p 0.027) | 2.03 (1.3, p 0.186) |
| 1 | Shr 2074 - Asa 2075 | 0.49 (0.09, p 0.000) | 11.2 (2.1, p 0.002) | 3.54 (1.2, p 0.033) |
| 2 | Shr 2075 - Fal 2075 | 0.595 (0.091, p 0.000) | 11.7 (2.3, p 0.001) | 4.05 (1.2, p 0.016) |

Pooled total effect, without and with linear trends: dose_pu x trend and log size_pre x trend (years since the pre-window end, whole panel). With trends, post beta is the shift beyond each bank's dose- and size-predicted linear path. Conservative: a trend fitted on the whole panel also soaks up an effect that ramps up towards the deadline.
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| paid_up / size_pre | 0.3575 | 0.086 | 0.005 | -0.03744 | 0.279 | 0.0157 | 0.0444 | 352 | 16 | - |
| paid_up / size_pre, + trends | 0.3929 | 0.118 | 0.018 | 0.2203 | 0.001 | 0.0172 | 0.0444 | 352 | 16 | - |
| log paid_up | 9.991 | 2.21 | 0.009 | -0.8303 | 0.252 | 0.438 | 8.01 | 352 | 16 | - |
| log paid_up, + trends | 7.869 | 1.81 | 0.007 | 1.346 | 0.028 | 0.345 | 8.01 | 352 | 16 | - |
| log size | 3.037 | 1.2 | 0.062 | -0.8168 | 0.088 | 0.133 | 11 | 352 | 16 | - |
| log size, + trends | 1.673 | 1.36 | 0.207 | 0.425 | 0.427 | 0.0733 | 11 | 352 | 16 | - |
| nim | -0.05721 | 0.0078 | 0.002 | 0.0008399 | 0.839 | -0.00251 | 0.00801 | 284 | 16 | - |
| nim, + trends | -0.01718 | 0.021 | 0.519 | 0.005869 | 0.207 | -0.000753 | 0.00801 | 284 | 16 | - |
| net_profit_loss | -0.04135 | 0.0109 | 0.021 | -0.002295 | 0.872 | -0.00181 | 0.00417 | 290 | 16 | - |
| net_profit_loss, + trends | 0.02816 | 0.0162 | 0.100 | -0.01382 | 0.675 | 0.00123 | 0.00417 | 290 | 16 | - |
Not done here: a Rambachan-Roth sensitivity bound (how large a trend violation would explain the post beta); with about 7 pre quarters per bank the event study above is the input it needs.

### 2. Mergers inside the window
Hinted dates are approximate (NRB 'List of BFIs' '*' joint-operation dates, hand-converted AD -> BS). The tables date them from the bulletins: a merger shows as a size and paid-up jump beyond the 99th percentile of month-to-month |log size change| in the clean bulletins (up to Falgun 2075, adjacent panel months only).
Clean-month |log size change|: median 0.021, 99th pct 0.146 (n=1679).

SBL, hinted Asar 2073, absorbing Lumbini:
| month | size | dlog size | paid-up | d paid-up | Lumbini size | Lumbini paid-up |
|---|---|---|---|---|---|---|
| Falgun 2072 | 59,952 | -0.012 | 2,437 | +0 | 25,965 | 2,001 |
| Chaitra 2072 | 65,491 | +0.088 | 2,437 | +0 | 27,158 | 2,001 |
| Baisakh 2073 | 67,725 | +0.034 | 2,437 | +0 | 27,485 | 2,301 |
| Jestha 2073 | 69,966 | +0.033 | 2,437 | +0 | 27,817 | 2,301 |
| Asar 2073 | 80,826 | +0.144 | 3,022 | +585 | n/a | n/a |
| Shrawan 2073 | 80,126 | -0.009 | 3,022 | +0 | n/a | n/a |
| Bhadra 2073 | 83,664 | +0.043 | 3,022 | +0 | n/a | n/a |
| Ashwin 2073 | 85,344 | +0.020 | 3,022 | +0 | n/a | n/a |
| Kartik 2073 | 86,435 | +0.013 | 3,022 | +0 | n/a | n/a |
SBL: size jumps > p99 in the window: none
Lumbini: last month with a finite size: Jestha 2073

RBB, hinted Baisakh 2075:
| month | size | dlog size | paid-up | d paid-up |
|---|---|---|---|---|
| Poush 2074 | 188,809 | +0.074 | 8,589 | +0 |
| Magh 2074 | 185,365 | -0.018 | 8,589 | +0 |
| Falgun 2074 | 179,574 | -0.032 | 8,589 | +0 |
| Chaitra 2074 | 190,189 | +0.057 | 8,589 | +0 |
| Baisakh 2075 | 184,527 | -0.030 | 9,005 | +416 |
| Jestha 2075 | 203,775 | +0.099 | 9,005 | +0 |
| Asar 2075 | 212,942 | +0.044 | 9,005 | +0 |
| Shrawan 2075 | 203,620 | -0.045 | 9,005 | +0 |
| Bhadra 2075 | 204,040 | +0.002 | 9,005 | +0 |
RBB: size jumps > p99 in the window: none

Total effect of dose_pu on the stocks and the NIM / net-profit horse race (paid-up and leverage dose jointly), all banks vs dropping a merging bank vs pro-forma. Pro-forma: before the absorbed bank's last bulletin month, its paid-up and size are added to the acquirer's (a cell where it is missing stays missing); the acquirer's dose_pu, log size control and size_pre are rebuilt from the sums. Its P&L outcomes and leverage dose are left as reported (the absorbed bank's P&L is not added).
pro-forma SBL + Lumbini: added through Jestha 2073; SBL dose_pu_pre 0.1060 -> 0.0492, size_pre 56,342 -> 80,653; acquirer cells made missing by a missing partner cell: 0

| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| all banks: paid_up / size_pre | 0.3575 | 0.086 | 0.005 | -0.03744 | 0.279 | 0.0157 | 0.0444 | 352 | 16 | - |
| all banks: log paid_up | 9.991 | 2.21 | 0.009 | -0.8303 | 0.252 | 0.438 | 8.01 | 352 | 16 | - |
| all banks: log size | 3.037 | 1.2 | 0.062 | -0.8168 | 0.088 | 0.133 | 11 | 352 | 16 | - |
| drop SBL: paid_up / size_pre | 0.359 | 0.0942 | 0.008 | -0.04388 | 0.224 | 0.0157 | 0.0452 | 330 | 15 | - |
| drop SBL: log paid_up | 9.496 | 2.47 | 0.025 | -0.923 | 0.219 | 0.416 | 8.04 | 330 | 15 | - |
| drop SBL: log size | 2.659 | 1.48 | 0.144 | -0.7297 | 0.114 | 0.116 | 11 | 330 | 15 | - |
| drop RBB: paid_up / size_pre | 0.3645 | 0.0897 | 0.005 | -0.02305 | 0.558 | 0.016 | 0.0436 | 330 | 15 | - |
| drop RBB: log paid_up | 10.14 | 1.83 | 0.000 | -0.5074 | 0.541 | 0.444 | 7.94 | 330 | 15 | - |
| drop RBB: log size | 3.064 | 1.31 | 0.071 | -0.8091 | 0.194 | 0.134 | 10.9 | 330 | 15 | - |
| SBL pro-forma: paid_up / size_pre | 0.3356 | 0.104 | 0.038 | -0.04463 | 0.214 | 0.0144 | 0.0452 | 352 | 16 | - |
| SBL pro-forma: log paid_up | 9.273 | 2.63 | 0.038 | -0.9405 | 0.208 | 0.397 | 8.05 | 352 | 16 | - |
| SBL pro-forma: log size | 2.588 | 1.5 | 0.159 | -0.6739 | 0.156 | 0.111 | 11 | 352 | 16 | - |

| version | outcome | paid-up beta | SE | wild p | leverage beta | SE | wild p |
|---|---|---|---|---|---|---|---|
| all banks | nim | -0.05666 | 0.00964 | 0.017 | 0.01505 | 0.0102 | 0.159 |
| all banks | net_profit_loss | -0.04166 | 0.0123 | 0.127 | -0.01015 | 0.0132 | 0.502 |
| drop SBL | nim | -0.05745 | 0.0105 | 0.018 | 0.01422 | 0.0106 | 0.234 |
| drop SBL | net_profit_loss | -0.04218 | 0.0138 | 0.156 | -0.01114 | 0.0154 | 0.503 |
| drop RBB | nim | -0.05815 | 0.00508 | 0.001 | 0.00912 | 0.00848 | 0.322 |
| drop RBB | net_profit_loss | -0.03596 | 0.005 | 0.011 | -0.009402 | 0.0131 | 0.598 |
| SBL pro-forma | nim | -0.06074 | 0.00933 | 0.010 | 0.00223 | 0.0144 | 0.897 |
| SBL pro-forma | net_profit_loss | -0.04492 | 0.0162 | 0.195 | -0.0198 | 0.0169 | 0.369 |
A merger is also a way to comply with the floor, so dropping or pro-forma-ing the bank removes part of the effect as well as the mechanical jump; report both next to the main fit.

### 3. Floor or catch-up?
Plain catch-up (low-capital banks raising anyway) and an absolute floor both make high-dose banks raise more. They differ in where the banks end up: a floor piles them up just above 8,000 whatever their start, catch-up keeps their order and spread.

(a) Bunching: banks with paid-up in [T, 1.05 T) / [T, 1.10 T), at each snapshot. A floor predicts a spike at T = 8,000 after the deadline and none at the placebo thresholds.
| T | Mangsir 2072 | Asar 2074 | Asar 2075 |
|---|---|---|---|
| 4,000 | 1 / 1 | 1 / 1 | 0 / 0 |
| 5,000 | 0 / 0 | 0 / 1 | 0 / 0 |
| 6,000 | 1 / 2 | 2 / 7 | 0 / 0 |
| 7,000 | 0 / 0 | 0 / 0 | 1 / 2 |
| 8,000 (requirement) | 0 / 1 | 2 / 3 | 10 / 11 |
| 9,000 | 0 / 1 | 0 / 0 | 1 / 1 |
| 10,000 | 0 / 0 | 0 / 0 | 0 / 0 |
| 12,000 | 0 / 0 | 1 / 1 | 0 / 0 |

(b) Where the banks below 8,000 at Mangsir 2072 end up at Asar 2075 (n=14). Floor: rank correlation start vs end near 0 and a much smaller spread at the end; catch-up: rank correlation > 0.
Spearman(start, end) = 0.011 (p 0.970); coefficient of variation 0.408 -> 0.047; end level median 8,043, range [7,163, 8,888]

(c) Raise vs shortfall, cross-section: raise = paid-up Asar 2075 - Mangsir 2072; shortfall = max(0, 8,000 - paid-up at Mangsir 2072). A floor predicts slope near 1 and intercept near 0 (raise exactly to the floor, plus whatever overshoot); the slope's p is a permutation test (9999 draws), not a t-test.
| bank | paid-up start | shortfall | raise | raise / shortfall | end - requirement |
|---|---|---|---|---|---|
| ADBNL | 9,861 | 0 | 4,077 | n/a | +5,938 |
| EBL | 2,137 | 5,863 | 5,969 | 1.02 | +107 |
| Global | 6,164 | 1,836 | 2,724 | 1.48 | +888 |
| HBL | 3,333 | 4,667 | 4,782 | 1.02 | +115 |
| Kumari | 2,432 | 5,568 | 4,731 | 0.85 | -837 |
| Laxmi | 3,039 | 4,961 | 5,182 | 1.04 | +222 |
| MBL | 2,776 | 5,224 | 5,279 | 1.01 | +56 |
| NABIL | 3,658 | 4,342 | 4,386 | 1.01 | +43 |
| NBL | 6,465 | 1,535 | 1,578 | 1.03 | +43 |
| NIC | 2,658 | 5,342 | 5,373 | 1.01 | +31 |
| NMB | 4,155 | 3,845 | 3,449 | 0.90 | -397 |
| NSBI | 3,058 | 4,942 | 4,989 | 1.01 | +47 |
| Prime | 3,140 | 4,860 | 4,893 | 1.01 | +33 |
| RBB | 8,589 | 0 | 416 | n/a | +1,005 |
| SBL | 2,031 | 5,969 | 6,433 | 1.08 | +464 |
| SCBNL | 2,248 | 5,752 | 5,763 | 1.00 | +11 |
OLS (n=16): raise = 1,627 + 0.680 x shortfall, R^2 0.730, permutation p (slope) 0.0004

(d) Timing: paid-up increases by fiscal year (adjacent bulletin months only; bonus-share issues are annual, so every year has some). A floor concentrates them before and just after the deadline.
| fiscal year | banks raising | total increase (NPR mn) |
|---|---|---|
| 2069/70 | 5 | 3,358 |
| 2070/71 | 11 | 6,799 |
| 2071/72 | 13 | 5,883 |
| 2072/73 | 14 | 13,544 |
| 2073/74 | 15 | 33,623 |
| 2074/75 | 15 | 26,707 |
| 2075/76 | 8 | 7,240 |
Panel Baisakh 2070 .. Falgun 2075, so the first and last fiscal years are partial.

## Paid-up dose: pre-period trend extrapolation and a smoothness bound (Sec 5b item 1)
whole-panel trend (paid_up_threats) = one dose slope over every quarter, so a post effect that ramps up also tilts the slope. Pre-only trend = the dose x trend and log size_pre x trend slopes are identified from the pre quarters (Baisakh 2070 - Mangsir 2072) only: the transition and post periods get their own level and slope, and dose_post is the deviation from the extrapolated pre line at the mean post date. Same cells, controls and FE as the headline total effect; unweighted.

### Event study by fiscal-year bin, dose_pu coefficient: beta (SE, wild p)
No trend: reference bin -1. Pre-only trend: the pre bins are the trend fit (no dummies); each later bin is that year's deviation from the extrapolated pre line at the bin's mid-date.

paid_up / size_pre (n_obs 352 / 352):
| bin | months | no trend | pre-only trend |
|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | -0.0256 (0.064, p 0.696) | trend fit |
| -3 | Shr 2070 - Asa 2071 | -0.0462 (0.045, p 0.334) | trend fit |
| -2 | Shr 2071 - Asa 2072 | -0.0297 (0.022, p 0.151) | trend fit |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | trend fit |
| transition | Pou 2072 - Asa 2073 | 0.0859 (0.055, p 0.146) | 0.184 (0.12, p 0.155) |
| 0 | Shr 2073 - Asa 2074 | 0.107 (0.11, p 0.405) | 0.596 (0.19, p 0.012) |
| 1 | Shr 2074 - Asa 2075 | 0.49 (0.09, p 0.000) | 1.05 (0.36, p 0.012) |
| 2 | Shr 2075 - Fal 2075 | 0.595 (0.091, p 0.000) | 1.11 (0.43, p 0.022) |
pre slope, dose x trend: -0.168 per year (SE 0.11); dropped: none

log size (n_obs 352 / 352):
| bin | months | no trend | pre-only trend |
|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | -0.716 (0.63, p 0.267) | trend fit |
| -3 | Shr 2070 - Asa 2071 | -1.12 (0.5, p 0.081) | trend fit |
| -2 | Shr 2071 - Asa 2072 | -0.448 (0.22, p 0.061) | trend fit |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | trend fit |
| transition | Pou 2072 - Asa 2073 | 0.598 (0.67, p 0.435) | -0.254 (1.2, p 0.837) |
| 0 | Shr 2073 - Asa 2074 | 2.03 (1.3, p 0.186) | 1.59 (2, p 0.484) |
| 1 | Shr 2074 - Asa 2075 | 3.54 (1.2, p 0.033) | 1.1 (2.8, p 0.689) |
| 2 | Shr 2075 - Fal 2075 | 4.05 (1.2, p 0.016) | 1.6 (4.3, p 0.731) |
pre slope, dose x trend: 0.69 per year (SE 1.1); dropped: none

nim (n_obs 284 / 284):
| bin | months | no trend | pre-only trend |
|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | n/a (n/a, p n/a) | trend fit |
| -3 | Shr 2070 - Asa 2071 | 0.00139 (0.005, p 0.779) | trend fit |
| -2 | Shr 2071 - Asa 2072 | -0.000267 (0.0038, p 0.943) | trend fit |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | trend fit |
| transition | Pou 2072 - Asa 2073 | -0.0108 (0.0099, p 0.375) | -0.0358 (0.014, p 0.253) |
| 0 | Shr 2073 - Asa 2074 | -0.0638 (0.0096, p 0.002) | -0.0274 (0.033, p 0.800) |
| 1 | Shr 2074 - Asa 2075 | -0.0595 (0.0076, p 0.001) | 0.0414 (0.034, p 0.278) |
| 2 | Shr 2075 - Fal 2075 | -0.0395 (0.009, p 0.006) | 0.0775 (0.046, p 0.175) |
pre slope, dose x trend: -0.0304 per year (SE 0.012); dropped: none

net_profit_loss (n_obs 290 / 290):
| bin | months | no trend | pre-only trend |
|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | n/a (n/a, p n/a) | trend fit |
| -3 | Shr 2070 - Asa 2071 | 0.00711 (0.0066, p 0.299) | trend fit |
| -2 | Shr 2071 - Asa 2072 | -0.0193 (0.028, p 0.814) | trend fit |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | trend fit |
| transition | Pou 2072 - Asa 2073 | -1.26e-06 (0.01, p 1.000) | -0.0388 (0.038, p 0.525) |
| 0 | Shr 2073 - Asa 2074 | -0.0385 (0.011, p 0.047) | -0.00759 (0.05, p 0.921) |
| 1 | Shr 2074 - Asa 2075 | -0.0466 (0.011, p 0.013) | 0.031 (0.071, p 0.741) |
| 2 | Shr 2075 - Fal 2075 | -0.0331 (0.011, p 0.019) | 0.0712 (0.081, p 0.438) |
pre slope, dose x trend: -0.0264 per year (SE 0.02); dropped: none

loan_loss_provision (n_obs 268 / 268):
| bin | months | no trend | pre-only trend |
|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | n/a (n/a, p n/a) | trend fit |
| -3 | Shr 2070 - Asa 2071 | 0.00107 (0.0036, p 0.750) | trend fit |
| -2 | Shr 2071 - Asa 2072 | -0.00161 (0.0041, p 0.712) | trend fit |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | trend fit |
| transition | Pou 2072 - Asa 2073 | -0.00162 (0.0067, p 0.792) | 0.0144 (0.012, p 0.390) |
| 0 | Shr 2073 - Asa 2074 | -0.00244 (0.0075, p 0.775) | -0.000516 (0.013, p 0.972) |
| 1 | Shr 2074 - Asa 2075 | -0.000267 (0.0085, p 0.979) | 0.0118 (0.018, p 0.556) |
| 2 | Shr 2075 - Fal 2075 | 0.00137 (0.0091, p 0.904) | 0.0277 (0.02, p 0.198) |
pre slope, dose x trend: -0.00476 per year (SE 0.0054); dropped: none

### Pooled total effect: no trend vs whole-panel trend vs pre-only trend
| version | post beta | SE | wild p | lead beta | lead p | 1-SD effect | pre mean | n_obs | clusters | note |
|---|---|---|---|---|---|---|---|---|---|---|
| paid_up / size_pre, no trend | 0.3575 | 0.086 | 0.005 | -0.03744 | 0.279 | 0.0157 | 0.0444 | 352 | 16 | - |
| paid_up / size_pre, whole-panel trend | 0.3929 | 0.118 | 0.018 | 0.2203 | 0.001 | 0.0172 | 0.0444 | 352 | 16 | - |
| paid_up / size_pre, pre-only trend | 0.8934 | 0.311 | 0.010 | n/a | n/a | 0.0391 | 0.0444 | 352 | 16 | - |
| log size, no trend | 3.037 | 1.2 | 0.062 | -0.8168 | 0.088 | 0.133 | 11 | 352 | 16 | - |
| log size, whole-panel trend | 1.673 | 1.36 | 0.207 | 0.425 | 0.427 | 0.0733 | 11 | 352 | 16 | - |
| log size, pre-only trend | 1.37 | 2.76 | 0.669 | n/a | n/a | 0.06 | 11 | 352 | 16 | - |
| nim, no trend | -0.05721 | 0.0078 | 0.002 | 0.0008399 | 0.839 | -0.00251 | 0.00801 | 284 | 16 | - |
| nim, whole-panel trend | -0.01718 | 0.021 | 0.519 | 0.005869 | 0.207 | -0.000753 | 0.00801 | 284 | 16 | - |
| nim, pre-only trend | 0.02001 | 0.0358 | 0.658 | n/a | n/a | 0.000877 | 0.00801 | 284 | 16 | - |
| net_profit_loss, no trend | -0.04135 | 0.0109 | 0.021 | -0.002295 | 0.872 | -0.00181 | 0.00417 | 290 | 16 | - |
| net_profit_loss, whole-panel trend | 0.02816 | 0.0162 | 0.100 | -0.01382 | 0.675 | 0.00123 | 0.00417 | 290 | 16 | - |
| net_profit_loss, pre-only trend | 0.0226 | 0.0669 | 0.806 | n/a | n/a | 0.00099 | 0.00417 | 290 | 16 | - |
| loan_loss_provision, no trend | -0.001022 | 0.00786 | 0.908 | 0.000219 | 0.946 | -4.48e-05 | 0.000911 | 268 | 16 | - |
| loan_loss_provision, whole-panel trend | -0.03469 | 0.0239 | 0.319 | 0.004999 | 0.798 | -0.00152 | 0.000911 | 268 | 16 | - |
| loan_loss_provision, pre-only trend | 0.01595 | 0.0144 | 0.275 | n/a | n/a | 0.000699 | 0.000911 | 268 | 16 | - |
lead = n/a under the pre-only trend by construction (the trend replaces the lead test).

### Smoothness bound on the pre-only-trend estimate
Suppose the dose slope changes by M per year right at the pre-window end (a trend break the pre data can't see). Post has its own level and slope, so dose_post, evaluated at the mean post date tau, is biased by exactly M x tau. CI(M) = beta +/- (c x SE + M x tau), c = t(0.975, G-1), CR1 SE. Breakdown M* = the smallest M whose CI includes 0, also shown as a multiple of the estimated pre slope |s|. A simplified version of Rambachan-Roth's smoothness restriction (one slope break, CR1), not their optimal FLCI.
| outcome | beta | SE | pre slope s (SE) | tau (yr) | CI at M = 0 | CI at M = 0.25|s| | CI at M = 0.5|s| | CI at M = 1|s| | M* | M* / |s| |
|---|---|---|---|---|---|---|---|---|---|---|
| paid_up / size_pre | 0.8934 | 0.311 | -0.168 (0.1) | 1.96 | [0.2315, 1.555] | [0.1492, 1.638] | [0.06703, 1.72] | [-0.09742, 1.884] | 0.118 | 0.70 |
| log size | 1.37 | 2.76 | 0.69 (1.1) | 1.96 | [-4.516, 7.255] | [-4.854, 7.593] | [-5.192, 7.931] | [-5.867, 8.607] | 0 | 0.00 |
| nim | 0.02001 | 0.0358 | -0.0304 (0.012) | 1.96 | [-0.05625, 0.09627] | [-0.07115, 0.1112] | [-0.08605, 0.1261] | [-0.1159, 0.1559] | 0 | 0.00 |
| net_profit_loss | 0.0226 | 0.0669 | -0.0265 (0.02) | 1.96 | [-0.1201, 0.1653] | [-0.1331, 0.1783] | [-0.146, 0.1912] | [-0.172, 0.2172] | 0 | 0.00 |
| loan_loss_provision | 0.01595 | 0.0144 | -0.00473 (0.0053) | 1.96 | [-0.01471, 0.04662] | [-0.01703, 0.04894] | [-0.01935, 0.05125] | [-0.02398, 0.05589] | 0 | 0.00 |
M* = 0: the pre-only-trend CI already includes 0. M* / |s| > 1: only a slope change larger than the whole pre-period dose slope would explain the estimate away.

## Bank label splits and exits (item 2 follow-up): continuity in the data only
Checks whether two labels look like one bank renamed in the bulletin (never both present, short gap, size and capital fund carried over), and which banks' size jumps when another exits. This is NOT a verification of mergers or renames: that needs NRB's records.
33 commercial-bank labels, Baisakh 2070 .. Shrawan 2083. Reference: |log size change| between a bank's consecutive complete months, median 0.026, 99th pct 2.170 (n=3713).

| old label | new label | in estimation panel | old last | new first | overlap months | gap (panel cols) | size new/old | |log ratio| vs p99 | capital fund new/old |
|---|---|---|---|---|---|---|---|---|---|
| Citizens | Citizen | no/no | Poush 2075 | Magh 2075 | 0 | 1 | 1.033 | 0.032 <= 2.170 | 0.969 |
| NBB | NBBL | no/no | Poush 2075 | Magh 2075 | 0 | 1 | 1.042 | 0.041 <= 2.170 | 1.000 |
| PRABHU | Prabhu | no/no | Poush 2075 | Magh 2075 | 0 | 1 | 1.008 | 0.008 <= 2.170 | 0.971 |
| Civil | CBL | no/no | Poush 2075 | Magh 2075 | 0 | 1 | 1.003 | 0.003 <= 2.170 | 1.000 |
Consistent with a relabel: 0 overlap months, gap 1, size and capital-fund ratios near 1.

Exits (last complete month before the panel end, not part of a label pair) and the banks whose size jumps most at the next month (|log change| > p99), as candidate absorbers:
| bank | in estimation panel | last month | size then | next month | size jumps > p99 (bank: +NPR mn) |
|---|---|---|---|---|---|
| BOK | no | Mangsir 2079 | 147,631 | Poush 2079 | none |
| Century | no | Mangsir 2079 | 118,512 | Poush 2079 | none |
| Lumbini | no | Jestha 2073 | 27,817 | Asar 2073 | none |
| Mega | no | Mangsir 2079 | 219,482 | Poush 2079 | none |
| NCC | no | Mangsir 2079 | 159,690 | Poush 2079 | none |
| NIBL | no | Poush 2079 | 501,325 | Magh 2079 | NABIL: -415,114 |
| Sunrise | no | Jestha 2080 | 66,331 | Asar 2080 | NABIL: +350,310, Kumari: +282,706, HBL: +247,199, RBB: +228,264 |

Entrants (first complete month after the panel start, not part of a label pair):
| bank | first month | size then |
|---|---|---|
| Century | Chaitra 2070 | 19,803 |
| NIMB | Magh 2079 | 51,629 |
| Sanima | Kartik 2071 | 32,106 |
Corrupted bulletins (Chaitra 2075 onward, Sec 5a) also move size, so a jump there is weak evidence.

## Multiple testing (item 5), over the saved run JSON
| family | tests | p<0.05 | expected under H0 | Holm rejections | BH (FDR 5%) rejections | smallest Holm p |
|---|---|---|---|---|---|---|
| pooled DiD (W_hat + RAS) | 16 | 2 | 0.8 | 0 | 0 | 0.178 (nim RAS dose) |
| event study (W_hat) | 32 | 4 | 1.6 | 0 | 0 | 0.595 (nim exposure bin 0) |
| event study joint, all post bins = 0 (W_hat) | 8 | 1 | 0.4 | 0 | 0 | 0.302 (nim exposure_post) |
| headline total effect (dose-only) | 4 | 0 | 0.2 | 0 | 0 | 1.000 (net_profit_loss dose) |

Joint wild-bootstrap Wald tests from the event study (estimator.fit_twfe `joint`; H0: every bin in the family = 0; the transition bin is in no family):
| outcome | family | bins tested | Wald (CR1) | wild p |
|---|---|---|---|---|
| net_profit_loss | dose_post | 3 (0, 1, 2) | 5.05 | 0.498 |
| net_profit_loss | exposure_post | 3 (0, 1, 2) | 19 | 0.548 |
| loan_loss_provision | dose_post | 3 (0, 1, 2) | 7 | 0.477 |
| loan_loss_provision | exposure_post | 3 (0, 1, 2) | 26.3 | 0.124 |
| loan_written_off | dose_post | 3 (0, 1, 2) | 0.408 | 0.970 |
| loan_written_off | exposure_post | 3 (0, 1, 2) | 4.08 | 0.300 |
| nim | dose_post | 3 (0, 1, 2) | 26.9 | 0.137 |
| nim | exposure_post | 3 (0, 1, 2) | 30.6 | 0.038 |
