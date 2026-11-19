MDE at alpha=0.05 two-sided, power=0.8, t(G-1) reference; see power_mde.py. Rows marked (paid-up dose) use the paid-up dose and the exposure rebuilt from it (reviewer_checks paid_up_dose, Eq. A'); betas are per unit of each dose. std. MDE (resid) scales by the regressor's SD after FE, controls and the other main regressor are partialled out (weighted).

| config | outcome | network | coef | beta | CR1 SE | G | MDE | |beta|/MDE | SD(regressor) | within SD(y) | std. MDE (SD of y per SD of regressor) | resid SD(regressor) | std. MDE (resid) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | net_profit_loss | W_hat | dose x post | 0.0247 | 0.0197 | 16 | 0.059 | 0.42 | 0.0201 | 0.00137 | 0.868 | 0.0024 | 0.104 |
| main | net_profit_loss | W_hat | exposure x post | -9.359e-05 | 8.73e-05 | 16 | 0.000262 | 0.36 | 36.8 | 0.00137 | 7.039 | 0.738 | 0.141 |
| main | net_profit_loss | RAS | dose x post | 0.02807 | 0.0224 | 16 | 0.067 | 0.42 | 0.0201 | 0.00137 | 0.986 | 0.00247 | 0.121 |
| main | net_profit_loss | RAS | exposure x post | -0.0001136 | 7.21e-05 | 16 | 0.000216 | 0.53 | 39 | 0.00137 | 6.158 | 0.874 | 0.138 |
| main | loan_loss_provision | W_hat | dose x post | 0.007646 | 0.0199 | 16 | 0.0595 | 0.13 | 0.0201 | 0.00152 | 0.786 | 0.00227 | 0.089 |
| main | loan_loss_provision | W_hat | exposure x post | 1.749e-05 | 7.24e-05 | 16 | 0.000217 | 0.08 | 36.8 | 0.00152 | 5.238 | 0.605 | 0.086 |
| main | loan_loss_provision | RAS | dose x post | 0.007518 | 0.0191 | 16 | 0.0573 | 0.13 | 0.0201 | 0.00152 | 0.757 | 0.00229 | 0.086 |
| main | loan_loss_provision | RAS | exposure x post | 2.327e-05 | 6.29e-05 | 16 | 0.000189 | 0.12 | 39 | 0.00152 | 4.825 | 0.658 | 0.081 |
| main | loan_written_off | W_hat | dose x post | -0.0004529 | 0.000622 | 16 | 0.00186 | 0.24 | 0.0201 | 8.88e-05 | 0.422 | 0.0024 | 0.050 |
| main | loan_written_off | W_hat | exposure x post | -1.337e-06 | 3.11e-06 | 16 | 9.31e-06 | 0.14 | 36.8 | 8.88e-05 | 3.854 | 0.738 | 0.077 |
| main | loan_written_off | RAS | dose x post | -0.0001336 | 0.000697 | 16 | 0.00209 | 0.06 | 0.0201 | 8.88e-05 | 0.473 | 0.00247 | 0.058 |
| main | loan_written_off | RAS | exposure x post | -2.828e-06 | 2.96e-06 | 16 | 8.88e-06 | 0.32 | 39 | 8.88e-05 | 3.894 | 0.874 | 0.087 |
| main | nim | W_hat | dose x post | 0.08217 | 0.0237 | 16 | 0.0709 | 1.16 | 0.0201 | 0.00144 | 0.987 | 0.00241 | 0.118 |
| main | nim | W_hat | exposure x post | -0.0002041 | 0.000103 | 16 | 0.000307 | 0.66 | 36.8 | 0.00144 | 7.826 | 0.738 | 0.157 |
| main | nim | RAS | dose x post | 0.08799 | 0.0219 | 16 | 0.0657 | 1.34 | 0.0201 | 0.00144 | 0.914 | 0.00242 | 0.110 |
| main | nim | RAS | exposure x post | -0.0002184 | 7.16e-05 | 16 | 0.000215 | 1.02 | 39 | 0.00144 | 5.793 | 0.814 | 0.121 |
| main | net_profit_loss | W_hat (paid-up dose) | dose x post | -0.02582 | 0.0207 | 16 | 0.062 | 0.42 | 0.0438 | 0.00137 | 1.987 | 0.00284 | 0.129 |
| main | net_profit_loss | W_hat (paid-up dose) | exposure x post | -6.836e-06 | 3.27e-05 | 16 | 9.81e-05 | 0.07 | 70.3 | 0.00137 | 5.046 | 2.41 | 0.173 |
| main | net_profit_loss | RAS (paid-up dose) | dose x post | -0.01134 | 0.0171 | 16 | 0.0511 | 0.22 | 0.0438 | 0.00137 | 1.638 | 0.00283 | 0.106 |
| main | net_profit_loss | RAS (paid-up dose) | exposure x post | 1.359e-05 | 2.69e-05 | 16 | 8.07e-05 | 0.17 | 74.8 | 0.00137 | 4.411 | 2.69 | 0.159 |
| main | loan_loss_provision | W_hat (paid-up dose) | dose x post | 0.01285 | 0.0176 | 16 | 0.0527 | 0.24 | 0.0438 | 0.00152 | 1.516 | 0.00289 | 0.100 |
| main | loan_loss_provision | W_hat (paid-up dose) | exposure x post | -1.378e-07 | 1.67e-05 | 16 | 5e-05 | 0.00 | 70.3 | 0.00152 | 2.307 | 2.24 | 0.074 |
| main | loan_loss_provision | RAS (paid-up dose) | dose x post | 0.007582 | 0.0134 | 16 | 0.0402 | 0.19 | 0.0438 | 0.00152 | 1.157 | 0.00285 | 0.075 |
| main | loan_loss_provision | RAS (paid-up dose) | exposure x post | -5.436e-06 | 1.34e-05 | 16 | 4.01e-05 | 0.14 | 74.8 | 0.00152 | 1.968 | 2.32 | 0.061 |
| main | loan_written_off | W_hat (paid-up dose) | dose x post | 0.0003258 | 0.000473 | 16 | 0.00142 | 0.23 | 0.0438 | 8.88e-05 | 0.700 | 0.00284 | 0.045 |
| main | loan_written_off | W_hat (paid-up dose) | exposure x post | 8.444e-07 | 1e-06 | 16 | 3e-06 | 0.28 | 70.3 | 8.88e-05 | 2.379 | 2.41 | 0.081 |
| main | loan_written_off | RAS (paid-up dose) | dose x post | 0.0006176 | 0.000424 | 16 | 0.00127 | 0.49 | 0.0438 | 8.88e-05 | 0.627 | 0.00283 | 0.041 |
| main | loan_written_off | RAS (paid-up dose) | exposure x post | 1.317e-06 | 9e-07 | 16 | 2.7e-06 | 0.49 | 74.8 | 8.88e-05 | 2.270 | 2.69 | 0.082 |
| main | nim | W_hat (paid-up dose) | dose x post | -0.0801 | 0.0174 | 16 | 0.052 | 1.54 | 0.0438 | 0.00144 | 1.578 | 0.00279 | 0.100 |
| main | nim | W_hat (paid-up dose) | exposure x post | -4.587e-05 | 3.27e-05 | 16 | 9.82e-05 | 0.47 | 70.3 | 0.00144 | 4.781 | 2.37 | 0.161 |
| main | nim | RAS (paid-up dose) | dose x post | -0.07409 | 0.0179 | 16 | 0.0536 | 1.38 | 0.0438 | 0.00144 | 1.627 | 0.00273 | 0.101 |
| main | nim | RAS (paid-up dose) | exposure x post | -3.205e-05 | 2.94e-05 | 16 | 8.81e-05 | 0.36 | 74.8 | 0.00144 | 4.561 | 2.48 | 0.151 |
| baseline | net_profit_loss | W_hat | dose x post | -0.02061 | 0.0172 | 16 | 0.0516 | 0.40 | 0.0201 | 0.00745 | 0.139 | 0.00197 | 0.014 |
| baseline | net_profit_loss | W_hat | exposure x post | -3.41e-05 | 1.42e-05 | 16 | 4.25e-05 | 0.80 | 46.8 | 0.00745 | 0.267 | 1.88 | 0.011 |
| baseline | net_profit_loss | RAS | dose x post | -0.02076 | 0.0165 | 16 | 0.0494 | 0.42 | 0.0201 | 0.00745 | 0.133 | 0.00197 | 0.013 |
| baseline | net_profit_loss | RAS | exposure x post | -4.989e-05 | 2.34e-05 | 16 | 7e-05 | 0.71 | 51 | 0.00745 | 0.479 | 1.42 | 0.013 |
| baseline | loan_loss_provision | W_hat | dose x post | 0.01371 | 0.0124 | 16 | 0.0373 | 0.37 | 0.0201 | 0.00691 | 0.109 | 0.00151 | 0.008 |
| baseline | loan_loss_provision | W_hat | exposure x post | 5.215e-06 | 1.02e-05 | 16 | 3.07e-05 | 0.17 | 46.8 | 0.00691 | 0.208 | 1.36 | 0.006 |
| baseline | loan_loss_provision | RAS | dose x post | 0.01324 | 0.0118 | 16 | 0.0353 | 0.38 | 0.0201 | 0.00691 | 0.103 | 0.0015 | 0.008 |
| baseline | loan_loss_provision | RAS | exposure x post | -2.901e-06 | 9.85e-06 | 16 | 2.95e-05 | 0.10 | 51 | 0.00691 | 0.218 | 1.25 | 0.005 |
| baseline | loan_written_off | W_hat | dose x post | -0.002157 | 0.00142 | 16 | 0.00424 | 0.51 | 0.0201 | 0.000525 | 0.163 | 0.00197 | 0.016 |
| baseline | loan_written_off | W_hat | exposure x post | 1.382e-06 | 6.73e-07 | 16 | 2.02e-06 | 0.69 | 46.8 | 0.000525 | 0.180 | 1.88 | 0.007 |
| baseline | loan_written_off | RAS | dose x post | -0.002149 | 0.00141 | 16 | 0.00423 | 0.51 | 0.0201 | 0.000525 | 0.162 | 0.00197 | 0.016 |
| baseline | loan_written_off | RAS | exposure x post | 1.916e-06 | 9.22e-07 | 16 | 2.76e-06 | 0.69 | 51 | 0.000525 | 0.269 | 1.42 | 0.007 |
| baseline | nim | W_hat | dose x post | -0.009635 | 0.0351 | 16 | 0.105 | 0.09 | 0.0201 | 0.0305 | 0.069 | 0.00198 | 0.007 |
| baseline | nim | W_hat | exposure x post | -2.562e-05 | 3.18e-05 | 16 | 9.53e-05 | 0.27 | 46.8 | 0.0305 | 0.146 | 1.69 | 0.005 |
| baseline | nim | RAS | dose x post | -0.008984 | 0.033 | 16 | 0.0989 | 0.09 | 0.0201 | 0.0305 | 0.065 | 0.00198 | 0.006 |
| baseline | nim | RAS | exposure x post | -5.63e-05 | 3.63e-05 | 16 | 0.000109 | 0.52 | 51 | 0.0305 | 0.182 | 1.35 | 0.005 |
| baseline | net_profit_loss | W_hat (paid-up dose) | dose x post | -0.0146 | 0.016 | 16 | 0.0481 | 0.30 | 0.0438 | 0.00745 | 0.283 | 0.00229 | 0.015 |
| baseline | net_profit_loss | W_hat (paid-up dose) | exposure x post | 1.715e-05 | 8.43e-06 | 16 | 2.53e-05 | 0.68 | 84.7 | 0.00745 | 0.288 | 3.31 | 0.011 |
| baseline | net_profit_loss | RAS (paid-up dose) | dose x post | -0.01141 | 0.0151 | 16 | 0.0453 | 0.25 | 0.0438 | 0.00745 | 0.266 | 0.00223 | 0.014 |
| baseline | net_profit_loss | RAS (paid-up dose) | exposure x post | 1.917e-05 | 1.11e-05 | 16 | 3.33e-05 | 0.58 | 107 | 0.00745 | 0.477 | 2.84 | 0.013 |
| baseline | loan_loss_provision | W_hat (paid-up dose) | dose x post | -0.02788 | 0.00503 | 16 | 0.0151 | 1.85 | 0.0438 | 0.00691 | 0.096 | 0.0018 | 0.004 |
| baseline | loan_loss_provision | W_hat (paid-up dose) | exposure x post | -1.077e-05 | 4.17e-06 | 16 | 1.25e-05 | 0.86 | 84.7 | 0.00691 | 0.153 | 2.46 | 0.004 |
| baseline | loan_loss_provision | RAS (paid-up dose) | dose x post | -0.02555 | 0.0049 | 16 | 0.0147 | 1.74 | 0.0438 | 0.00691 | 0.093 | 0.00181 | 0.004 |
| baseline | loan_loss_provision | RAS (paid-up dose) | exposure x post | -6.27e-06 | 3.99e-06 | 16 | 1.2e-05 | 0.52 | 107 | 0.00691 | 0.185 | 2.52 | 0.004 |
| baseline | loan_written_off | W_hat (paid-up dose) | dose x post | 0.0002751 | 0.00119 | 16 | 0.00357 | 0.08 | 0.0438 | 0.000525 | 0.298 | 0.00229 | 0.016 |
| baseline | loan_written_off | W_hat (paid-up dose) | exposure x post | -6.563e-07 | 3.92e-07 | 16 | 1.18e-06 | 0.56 | 84.7 | 0.000525 | 0.190 | 3.31 | 0.007 |
| baseline | loan_written_off | RAS (paid-up dose) | dose x post | 0.000117 | 0.00122 | 16 | 0.00365 | 0.03 | 0.0438 | 0.000525 | 0.305 | 0.00223 | 0.016 |
| baseline | loan_written_off | RAS (paid-up dose) | exposure x post | -7.175e-07 | 4.97e-07 | 16 | 1.49e-06 | 0.48 | 107 | 0.000525 | 0.303 | 2.84 | 0.008 |
| baseline | nim | W_hat (paid-up dose) | dose x post | -0.005008 | 0.0484 | 16 | 0.145 | 0.03 | 0.0438 | 0.0305 | 0.209 | 0.00222 | 0.011 |
| baseline | nim | W_hat (paid-up dose) | exposure x post | 1.262e-05 | 2.65e-05 | 16 | 7.94e-05 | 0.16 | 84.7 | 0.0305 | 0.221 | 2.96 | 0.008 |
| baseline | nim | RAS (paid-up dose) | dose x post | 0.01103 | 0.0353 | 16 | 0.106 | 0.10 | 0.0438 | 0.0305 | 0.152 | 0.00219 | 0.008 |
| baseline | nim | RAS (paid-up dose) | exposure x post | 2.693e-05 | 1.64e-05 | 16 | 4.92e-05 | 0.55 | 107 | 0.0305 | 0.173 | 2.7 | 0.004 |
| clean_window | net_profit_loss | W_hat | dose x post | -0.004075 | 0.00859 | 16 | 0.0258 | 0.16 | 0.0201 | 0.00274 | 0.189 | 0.00346 | 0.033 |
| clean_window | net_profit_loss | W_hat | exposure x post | -2.603e-05 | 1.67e-05 | 16 | 5e-05 | 0.52 | 36.8 | 0.00274 | 0.671 | 3.4 | 0.062 |
| clean_window | net_profit_loss | RAS | dose x post | -0.004035 | 0.00917 | 16 | 0.0275 | 0.15 | 0.0201 | 0.00274 | 0.202 | 0.00346 | 0.035 |
| clean_window | net_profit_loss | RAS | exposure x post | -3.178e-05 | 2.62e-05 | 16 | 7.87e-05 | 0.40 | 39 | 0.00274 | 1.120 | 2.58 | 0.074 |
| clean_window | loan_loss_provision | W_hat | dose x post | 0.004903 | 0.00534 | 16 | 0.016 | 0.31 | 0.0201 | 0.000885 | 0.364 | 0.0029 | 0.053 |
| clean_window | loan_loss_provision | W_hat | exposure x post | 4.15e-06 | 4.64e-06 | 16 | 1.39e-05 | 0.30 | 36.8 | 0.000885 | 0.578 | 2.7 | 0.042 |
| clean_window | loan_loss_provision | RAS | dose x post | 0.004453 | 0.00516 | 16 | 0.0155 | 0.29 | 0.0201 | 0.000885 | 0.352 | 0.00289 | 0.051 |
| clean_window | loan_loss_provision | RAS | exposure x post | -2.326e-06 | 5.17e-06 | 16 | 1.55e-05 | 0.15 | 39 | 0.000885 | 0.683 | 2.49 | 0.044 |
| clean_window | loan_written_off | W_hat | dose x post | -0.0001309 | 0.000283 | 16 | 0.00085 | 0.15 | 0.0201 | 6.36e-05 | 0.269 | 0.00346 | 0.046 |
| clean_window | loan_written_off | W_hat | exposure x post | 4.147e-07 | 3.88e-07 | 16 | 1.16e-06 | 0.36 | 36.8 | 6.36e-05 | 0.672 | 3.4 | 0.062 |
| clean_window | loan_written_off | RAS | dose x post | -0.0001221 | 0.000275 | 16 | 0.000824 | 0.15 | 0.0201 | 6.36e-05 | 0.261 | 0.00346 | 0.045 |
| clean_window | loan_written_off | RAS | exposure x post | 2.134e-07 | 3.23e-07 | 16 | 9.69e-07 | 0.22 | 39 | 6.36e-05 | 0.594 | 2.58 | 0.039 |
| clean_window | nim | W_hat | dose x post | 0.003059 | 0.0078 | 16 | 0.0234 | 0.13 | 0.0201 | 0.00377 | 0.125 | 0.0035 | 0.022 |
| clean_window | nim | W_hat | exposure x post | -3.96e-05 | 2.56e-05 | 16 | 7.67e-05 | 0.52 | 36.8 | 0.00377 | 0.748 | 3.12 | 0.064 |
| clean_window | nim | RAS | dose x post | 0.003376 | 0.00805 | 16 | 0.0241 | 0.14 | 0.0201 | 0.00377 | 0.129 | 0.0035 | 0.022 |
| clean_window | nim | RAS | exposure x post | -5.018e-05 | 3.55e-05 | 16 | 0.000106 | 0.47 | 39 | 0.00377 | 1.101 | 2.48 | 0.070 |
| clean_window | net_profit_loss | W_hat (paid-up dose) | dose x post | 0.002229 | 0.00772 | 16 | 0.0232 | 0.10 | 0.0438 | 0.00274 | 0.370 | 0.00389 | 0.033 |
| clean_window | net_profit_loss | W_hat (paid-up dose) | exposure x post | 1.579e-05 | 1.01e-05 | 16 | 3.01e-05 | 0.52 | 70.3 | 0.00274 | 0.774 | 5.96 | 0.066 |
| clean_window | net_profit_loss | RAS (paid-up dose) | dose x post | 0.002966 | 0.00941 | 16 | 0.0282 | 0.11 | 0.0438 | 0.00274 | 0.451 | 0.00383 | 0.040 |
| clean_window | net_profit_loss | RAS (paid-up dose) | exposure x post | 1.458e-05 | 1.34e-05 | 16 | 4.02e-05 | 0.36 | 74.8 | 0.00274 | 1.098 | 5.19 | 0.076 |
| clean_window | loan_loss_provision | W_hat (paid-up dose) | dose x post | -0.003724 | 0.00325 | 16 | 0.00973 | 0.38 | 0.0438 | 0.000885 | 0.482 | 0.00336 | 0.037 |
| clean_window | loan_loss_provision | W_hat (paid-up dose) | exposure x post | -4.692e-06 | 3.25e-06 | 16 | 9.74e-06 | 0.48 | 70.3 | 0.000885 | 0.774 | 4.89 | 0.054 |
| clean_window | loan_loss_provision | RAS (paid-up dose) | dose x post | -0.002876 | 0.00323 | 16 | 0.0097 | 0.30 | 0.0438 | 0.000885 | 0.480 | 0.0034 | 0.037 |
| clean_window | loan_loss_provision | RAS (paid-up dose) | exposure x post | -1.876e-06 | 3.24e-06 | 16 | 9.72e-06 | 0.19 | 74.8 | 0.000885 | 0.821 | 5.04 | 0.055 |
| clean_window | loan_written_off | W_hat (paid-up dose) | dose x post | -0.0003184 | 0.000308 | 16 | 0.000924 | 0.34 | 0.0438 | 6.36e-05 | 0.636 | 0.00389 | 0.057 |
| clean_window | loan_written_off | W_hat (paid-up dose) | exposure x post | -2.577e-07 | 2.23e-07 | 16 | 6.7e-07 | 0.38 | 70.3 | 6.36e-05 | 0.741 | 5.96 | 0.063 |
| clean_window | loan_written_off | RAS (paid-up dose) | dose x post | -0.0002591 | 0.000275 | 16 | 0.000825 | 0.31 | 0.0438 | 6.36e-05 | 0.569 | 0.00383 | 0.050 |
| clean_window | loan_written_off | RAS (paid-up dose) | exposure x post | -1.091e-07 | 1.29e-07 | 16 | 3.86e-07 | 0.28 | 74.8 | 6.36e-05 | 0.454 | 5.19 | 0.032 |
| clean_window | nim | W_hat (paid-up dose) | dose x post | -0.007422 | 0.0092 | 16 | 0.0276 | 0.27 | 0.0438 | 0.00377 | 0.320 | 0.00383 | 0.028 |
| clean_window | nim | W_hat (paid-up dose) | exposure x post | 1.858e-05 | 1.46e-05 | 16 | 4.38e-05 | 0.42 | 70.3 | 0.00377 | 0.817 | 5.46 | 0.063 |
| clean_window | nim | RAS (paid-up dose) | dose x post | -0.007462 | 0.00876 | 16 | 0.0263 | 0.28 | 0.0438 | 0.00377 | 0.305 | 0.0038 | 0.026 |
| clean_window | nim | RAS (paid-up dose) | exposure x post | 1.773e-05 | 1.43e-05 | 16 | 4.29e-05 | 0.41 | 74.8 | 0.00377 | 0.850 | 5 | 0.057 |

Where each regressor's variation goes (weighted SDs on the fit's cells, partialling out step by step; the last step is the resid SD used above):

| config | outcome | network | regressor | SD, same cells, no FE | after bank + period FE | + other main regressor | ... + exposure level only | ... + Z_pre x post only | + all controls (= resid SD) | share left (resid / no FE) |
|---|---|---|---|---|---|---|---|---|---|---|
| main | net_profit_loss | W_hat | dose x post | 0.0218 | 0.00554 | 0.00554 | 0.00422 | 0.00293 | 0.0024 | 0.110 |
| main | net_profit_loss | W_hat | exposure x post | 33.4 | 21.8 | 21.8 | 1.47 | 21.7 | 0.738 | 0.022 |
| main | net_profit_loss | RAS | dose x post | 0.0218 | 0.00554 | 0.00554 | 0.00422 | 0.00293 | 0.00247 | 0.113 |
| main | net_profit_loss | RAS | exposure x post | 33.5 | 23.6 | 23.6 | 1.78 | 23.5 | 0.874 | 0.026 |
| main | loan_loss_provision | W_hat | dose x post | 0.0219 | 0.00562 | 0.00562 | 0.00415 | 0.00283 | 0.00227 | 0.103 |
| main | loan_loss_provision | W_hat | exposure x post | 31.1 | 20.7 | 20.7 | 1.46 | 20.6 | 0.605 | 0.019 |
| main | loan_loss_provision | RAS | dose x post | 0.0219 | 0.00562 | 0.00562 | 0.00419 | 0.00283 | 0.00229 | 0.104 |
| main | loan_loss_provision | RAS | exposure x post | 32 | 22.5 | 22.5 | 1.74 | 22.4 | 0.658 | 0.021 |
| main | loan_written_off | W_hat | dose x post | 0.0218 | 0.00554 | 0.00554 | 0.00422 | 0.00293 | 0.0024 | 0.110 |
| main | loan_written_off | W_hat | exposure x post | 33.4 | 21.8 | 21.8 | 1.47 | 21.7 | 0.738 | 0.022 |
| main | loan_written_off | RAS | dose x post | 0.0218 | 0.00554 | 0.00554 | 0.00422 | 0.00293 | 0.00247 | 0.113 |
| main | loan_written_off | RAS | exposure x post | 33.5 | 23.6 | 23.6 | 1.78 | 23.5 | 0.874 | 0.026 |
| main | nim | W_hat | dose x post | 0.0216 | 0.00543 | 0.00543 | 0.00404 | 0.00294 | 0.00241 | 0.111 |
| main | nim | W_hat | exposure x post | 33.6 | 21.6 | 21.6 | 1.44 | 21.5 | 0.738 | 0.022 |
| main | nim | RAS | dose x post | 0.0216 | 0.00543 | 0.00543 | 0.00407 | 0.00294 | 0.00242 | 0.112 |
| main | nim | RAS | exposure x post | 33.6 | 23.5 | 23.4 | 1.7 | 23.4 | 0.814 | 0.024 |
| main | net_profit_loss | W_hat (paid-up dose) | dose x post | 0.0448 | 0.0122 | 0.0122 | 0.0113 | 0.00348 | 0.00284 | 0.063 |
| main | net_profit_loss | W_hat (paid-up dose) | exposure x post | 66.3 | 44 | 43.8 | 5.72 | 42.9 | 2.41 | 0.036 |
| main | net_profit_loss | RAS (paid-up dose) | dose x post | 0.0448 | 0.0122 | 0.0122 | 0.0116 | 0.00348 | 0.00283 | 0.063 |
| main | net_profit_loss | RAS (paid-up dose) | exposure x post | 70.9 | 51.6 | 51.5 | 7.03 | 50.5 | 2.69 | 0.038 |
| main | loan_loss_provision | W_hat (paid-up dose) | dose x post | 0.0449 | 0.0122 | 0.0121 | 0.011 | 0.00322 | 0.00289 | 0.064 |
| main | loan_loss_provision | W_hat (paid-up dose) | exposure x post | 62.2 | 42.3 | 42 | 5.72 | 41.3 | 2.24 | 0.036 |
| main | loan_loss_provision | RAS (paid-up dose) | dose x post | 0.0449 | 0.0122 | 0.0121 | 0.0111 | 0.00322 | 0.00285 | 0.064 |
| main | loan_loss_provision | RAS (paid-up dose) | exposure x post | 68.4 | 50.2 | 49.9 | 6.71 | 49.1 | 2.32 | 0.034 |
| main | loan_written_off | W_hat (paid-up dose) | dose x post | 0.0448 | 0.0122 | 0.0122 | 0.0113 | 0.00348 | 0.00284 | 0.063 |
| main | loan_written_off | W_hat (paid-up dose) | exposure x post | 66.3 | 44 | 43.8 | 5.72 | 42.9 | 2.41 | 0.036 |
| main | loan_written_off | RAS (paid-up dose) | dose x post | 0.0448 | 0.0122 | 0.0122 | 0.0116 | 0.00348 | 0.00283 | 0.063 |
| main | loan_written_off | RAS (paid-up dose) | exposure x post | 70.9 | 51.6 | 51.5 | 7.03 | 50.5 | 2.69 | 0.038 |
| main | nim | W_hat (paid-up dose) | dose x post | 0.0446 | 0.0116 | 0.0115 | 0.0105 | 0.00347 | 0.00279 | 0.063 |
| main | nim | W_hat (paid-up dose) | exposure x post | 66.6 | 43.6 | 43.3 | 5.6 | 42.6 | 2.37 | 0.036 |
| main | nim | RAS (paid-up dose) | dose x post | 0.0446 | 0.0116 | 0.0116 | 0.0106 | 0.00348 | 0.00273 | 0.061 |
| main | nim | RAS (paid-up dose) | exposure x post | 71.2 | 51.4 | 51.1 | 6.57 | 50.4 | 2.48 | 0.035 |
| baseline | net_profit_loss | W_hat | dose x post | 0.0206 | 0.00373 | 0.00373 | 0.00358 | 0.00197 | 0.00197 | 0.096 |
| baseline | net_profit_loss | W_hat | exposure x post | 46.2 | 35.4 | 35.4 | 2.44 | 35.3 | 1.88 | 0.041 |
| baseline | net_profit_loss | RAS | dose x post | 0.0206 | 0.00373 | 0.00373 | 0.00354 | 0.00197 | 0.00197 | 0.096 |
| baseline | net_profit_loss | RAS | exposure x post | 50.2 | 41.9 | 41.8 | 1.85 | 41.8 | 1.42 | 0.028 |
| baseline | loan_loss_provision | W_hat | dose x post | 0.0205 | 0.00367 | 0.00366 | 0.00345 | 0.00151 | 0.00151 | 0.074 |
| baseline | loan_loss_provision | W_hat | exposure x post | 47.1 | 36.1 | 36.1 | 2.2 | 36.1 | 1.36 | 0.029 |
| baseline | loan_loss_provision | RAS | dose x post | 0.0205 | 0.00367 | 0.00366 | 0.00342 | 0.00151 | 0.0015 | 0.073 |
| baseline | loan_loss_provision | RAS | exposure x post | 51.5 | 42.7 | 42.7 | 1.69 | 42.7 | 1.25 | 0.024 |
| baseline | loan_written_off | W_hat | dose x post | 0.0206 | 0.00373 | 0.00373 | 0.00358 | 0.00197 | 0.00197 | 0.096 |
| baseline | loan_written_off | W_hat | exposure x post | 46.2 | 35.4 | 35.4 | 2.44 | 35.3 | 1.88 | 0.041 |
| baseline | loan_written_off | RAS | dose x post | 0.0206 | 0.00373 | 0.00373 | 0.00354 | 0.00197 | 0.00197 | 0.096 |
| baseline | loan_written_off | RAS | exposure x post | 50.2 | 41.9 | 41.8 | 1.85 | 41.8 | 1.42 | 0.028 |
| baseline | nim | W_hat | dose x post | 0.0204 | 0.00365 | 0.00365 | 0.00345 | 0.00198 | 0.00198 | 0.097 |
| baseline | nim | W_hat | exposure x post | 46.2 | 35.4 | 35.3 | 2.37 | 35.3 | 1.69 | 0.037 |
| baseline | nim | RAS | dose x post | 0.0204 | 0.00365 | 0.00365 | 0.00343 | 0.00198 | 0.00198 | 0.097 |
| baseline | nim | RAS | exposure x post | 50.3 | 41.8 | 41.8 | 1.81 | 41.8 | 1.35 | 0.027 |
| baseline | net_profit_loss | W_hat (paid-up dose) | dose x post | 0.0436 | 0.00824 | 0.00824 | 0.00822 | 0.00239 | 0.00229 | 0.052 |
| baseline | net_profit_loss | W_hat (paid-up dose) | exposure x post | 83.5 | 63.3 | 63.3 | 4.8 | 63.1 | 3.31 | 0.040 |
| baseline | net_profit_loss | RAS (paid-up dose) | dose x post | 0.0436 | 0.00824 | 0.00824 | 0.00816 | 0.00239 | 0.00223 | 0.051 |
| baseline | net_profit_loss | RAS (paid-up dose) | exposure x post | 105 | 87.2 | 87.2 | 4.13 | 87 | 2.84 | 0.027 |
| baseline | loan_loss_provision | W_hat (paid-up dose) | dose x post | 0.0433 | 0.00666 | 0.00666 | 0.00659 | 0.00191 | 0.0018 | 0.041 |
| baseline | loan_loss_provision | W_hat (paid-up dose) | exposure x post | 85 | 64.5 | 64.5 | 4.48 | 64.4 | 2.46 | 0.029 |
| baseline | loan_loss_provision | RAS (paid-up dose) | dose x post | 0.0433 | 0.00666 | 0.00666 | 0.00647 | 0.00191 | 0.00181 | 0.042 |
| baseline | loan_loss_provision | RAS (paid-up dose) | exposure x post | 108 | 89.1 | 89.1 | 3.75 | 88.9 | 2.52 | 0.023 |
| baseline | loan_written_off | W_hat (paid-up dose) | dose x post | 0.0436 | 0.00824 | 0.00824 | 0.00822 | 0.00239 | 0.00229 | 0.052 |
| baseline | loan_written_off | W_hat (paid-up dose) | exposure x post | 83.5 | 63.3 | 63.3 | 4.8 | 63.1 | 3.31 | 0.040 |
| baseline | loan_written_off | RAS (paid-up dose) | dose x post | 0.0436 | 0.00824 | 0.00824 | 0.00816 | 0.00239 | 0.00223 | 0.051 |
| baseline | loan_written_off | RAS (paid-up dose) | exposure x post | 105 | 87.2 | 87.2 | 4.13 | 87 | 2.84 | 0.027 |
| baseline | nim | W_hat (paid-up dose) | dose x post | 0.0435 | 0.00781 | 0.00781 | 0.0078 | 0.00238 | 0.00222 | 0.051 |
| baseline | nim | W_hat (paid-up dose) | exposure x post | 83.7 | 63.3 | 63.3 | 4.75 | 63.2 | 2.96 | 0.035 |
| baseline | nim | RAS (paid-up dose) | dose x post | 0.0435 | 0.00781 | 0.00781 | 0.00772 | 0.00238 | 0.00219 | 0.050 |
| baseline | nim | RAS (paid-up dose) | exposure x post | 106 | 87.2 | 87.2 | 4.05 | 87.1 | 2.7 | 0.026 |
| clean_window | net_profit_loss | W_hat | dose x post | 0.0225 | 0.00655 | 0.00654 | 0.00629 | 0.00346 | 0.00346 | 0.154 |
| clean_window | net_profit_loss | W_hat | exposure x post | 35.1 | 23.6 | 23.6 | 4.37 | 23.3 | 3.4 | 0.097 |
| clean_window | net_profit_loss | RAS | dose x post | 0.0225 | 0.00655 | 0.00654 | 0.00623 | 0.00346 | 0.00346 | 0.154 |
| clean_window | net_profit_loss | RAS | exposure x post | 37 | 25.8 | 25.8 | 3.32 | 25.5 | 2.58 | 0.070 |
| clean_window | loan_loss_provision | W_hat | dose x post | 0.0228 | 0.00699 | 0.00698 | 0.00661 | 0.00291 | 0.0029 | 0.128 |
| clean_window | loan_loss_provision | W_hat | exposure x post | 37.2 | 24.8 | 24.8 | 4.3 | 24.4 | 2.7 | 0.073 |
| clean_window | loan_loss_provision | RAS | dose x post | 0.0228 | 0.00699 | 0.00698 | 0.00655 | 0.00291 | 0.00289 | 0.127 |
| clean_window | loan_loss_provision | RAS | exposure x post | 39 | 27 | 26.9 | 3.31 | 26.5 | 2.49 | 0.064 |
| clean_window | loan_written_off | W_hat | dose x post | 0.0225 | 0.00655 | 0.00654 | 0.00629 | 0.00346 | 0.00346 | 0.154 |
| clean_window | loan_written_off | W_hat | exposure x post | 35.1 | 23.6 | 23.6 | 4.37 | 23.3 | 3.4 | 0.097 |
| clean_window | loan_written_off | RAS | dose x post | 0.0225 | 0.00655 | 0.00654 | 0.00623 | 0.00346 | 0.00346 | 0.154 |
| clean_window | loan_written_off | RAS | exposure x post | 37 | 25.8 | 25.8 | 3.32 | 25.5 | 2.58 | 0.070 |
| clean_window | nim | W_hat | dose x post | 0.0223 | 0.00645 | 0.00645 | 0.00611 | 0.0035 | 0.0035 | 0.157 |
| clean_window | nim | W_hat | exposure x post | 35.4 | 23.5 | 23.5 | 4.28 | 23.2 | 3.12 | 0.088 |
| clean_window | nim | RAS | dose x post | 0.0223 | 0.00645 | 0.00645 | 0.00608 | 0.0035 | 0.0035 | 0.157 |
| clean_window | nim | RAS | exposure x post | 37.3 | 25.7 | 25.7 | 3.27 | 25.4 | 2.48 | 0.067 |
| clean_window | net_profit_loss | W_hat (paid-up dose) | dose x post | 0.0456 | 0.0144 | 0.0143 | 0.0143 | 0.00406 | 0.00389 | 0.085 |
| clean_window | net_profit_loss | W_hat (paid-up dose) | exposure x post | 67.4 | 46 | 45.6 | 8.58 | 43.9 | 5.96 | 0.088 |
| clean_window | net_profit_loss | RAS (paid-up dose) | dose x post | 0.0456 | 0.0144 | 0.0143 | 0.0142 | 0.00407 | 0.00383 | 0.084 |
| clean_window | net_profit_loss | RAS (paid-up dose) | exposure x post | 71.4 | 50.8 | 50.4 | 7.42 | 48.6 | 5.19 | 0.073 |
| clean_window | loan_loss_provision | W_hat (paid-up dose) | dose x post | 0.045 | 0.0128 | 0.0127 | 0.0126 | 0.00354 | 0.00336 | 0.075 |
| clean_window | loan_loss_provision | W_hat (paid-up dose) | exposure x post | 70.9 | 48.1 | 47.6 | 8.71 | 46.1 | 4.89 | 0.069 |
| clean_window | loan_loss_provision | RAS (paid-up dose) | dose x post | 0.045 | 0.0128 | 0.0127 | 0.0124 | 0.00355 | 0.0034 | 0.076 |
| clean_window | loan_loss_provision | RAS (paid-up dose) | exposure x post | 74.9 | 52.9 | 52.4 | 7.32 | 50.8 | 5.04 | 0.067 |
| clean_window | loan_written_off | W_hat (paid-up dose) | dose x post | 0.0456 | 0.0144 | 0.0143 | 0.0143 | 0.00406 | 0.00389 | 0.085 |
| clean_window | loan_written_off | W_hat (paid-up dose) | exposure x post | 67.4 | 46 | 45.6 | 8.58 | 43.9 | 5.96 | 0.088 |
| clean_window | loan_written_off | RAS (paid-up dose) | dose x post | 0.0456 | 0.0144 | 0.0143 | 0.0142 | 0.00407 | 0.00383 | 0.084 |
| clean_window | loan_written_off | RAS (paid-up dose) | exposure x post | 71.4 | 50.8 | 50.4 | 7.42 | 48.6 | 5.19 | 0.073 |
| clean_window | nim | W_hat (paid-up dose) | dose x post | 0.0453 | 0.0138 | 0.0136 | 0.0136 | 0.00407 | 0.00383 | 0.085 |
| clean_window | nim | W_hat (paid-up dose) | exposure x post | 68 | 45.8 | 45 | 8.55 | 43.8 | 5.46 | 0.080 |
| clean_window | nim | RAS (paid-up dose) | dose x post | 0.0453 | 0.0138 | 0.0136 | 0.0135 | 0.00408 | 0.0038 | 0.084 |
| clean_window | nim | RAS (paid-up dose) | exposure x post | 72 | 50.7 | 49.9 | 7.32 | 48.6 | 5 | 0.069 |
