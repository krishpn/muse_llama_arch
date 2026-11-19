# Paid-up figures: main

## paid_up_compliance.png

Requirement 8,000 NPR mn; panel Baisakh 2070 .. Falgun 2075; value at the last month with a finite cell at or before each date.

| bank | paid-up Mangsir 2072 | shortfall | paid-up Asar 2075 | raise | in band |
|---|---|---|---|---|---|
| ADBNL | 9,861 | 0 | 13,938 | 4,077 |  |
| EBL | 2,137 | 5,863 | 8,107 | 5,969 | yes |
| Global | 6,164 | 1,836 | 8,888 | 2,724 |  |
| HBL | 3,333 | 4,667 | 8,115 | 4,782 | yes |
| Kumari | 2,432 | 5,568 | 7,163 | 4,731 |  |
| Laxmi | 3,039 | 4,961 | 8,222 | 5,182 | yes |
| MBL | 2,776 | 5,224 | 8,056 | 5,279 | yes |
| NABIL | 3,658 | 4,342 | 8,043 | 4,386 | yes |
| NBL | 6,465 | 1,535 | 8,043 | 1,578 | yes |
| NIC | 2,658 | 5,342 | 8,031 | 5,373 | yes |
| NMB | 4,155 | 3,845 | 7,603 | 3,449 |  |
| NSBI | 3,058 | 4,942 | 8,047 | 4,989 | yes |
| Prime | 3,140 | 4,860 | 8,033 | 4,893 | yes |
| RBB | 8,589 | 0 | 9,005 | 416 |  |
| SBL | 2,031 | 5,969 | 8,464 | 6,433 |  |
| SCBNL | 2,248 | 5,752 | 8,011 | 5,763 | yes |

## paid_up_event_study.png

Paid-up dose, no trend, reference bin -1; 95% CI = beta +/- 1.96 x CR1 SE; wild p from 9999 WCR draws (seed 0). Same fit as the 'no trend' column of reviewer_checks paid_up_trend.

Net interest income / size (n_obs 284):

| bin | months | beta | CR1 SE | wild p |
|---|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | n/a | | |
| -3 | Shr 2070 - Asa 2071 | 0.001393 | 0.005 | 0.779 |
| -2 | Shr 2071 - Asa 2072 | -0.0002669 | 0.00382 | 0.943 |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | | |
| transition | Pou 2072 - Asa 2073 | -0.01085 | 0.00989 | 0.375 |
| 0 | Shr 2073 - Asa 2074 | -0.06382 | 0.00965 | 0.002 |
| 1 | Shr 2074 - Asa 2075 | -0.05945 | 0.00759 | 0.001 |
| 2 | Shr 2075 - Fal 2075 | -0.0395 | 0.00896 | 0.006 |

Net profit / size (n_obs 290):

| bin | months | beta | CR1 SE | wild p |
|---|---|---|---|---|
| -4 | Bai 2070 - Asa 2070 | n/a | | |
| -3 | Shr 2070 - Asa 2071 | 0.007107 | 0.00665 | 0.299 |
| -2 | Shr 2071 - Asa 2072 | -0.01935 | 0.0284 | 0.814 |
| -1 | Shr 2072 - Man 2072 | 0 (ref) | | |
| transition | Pou 2072 - Asa 2073 | -1.264e-06 | 0.0102 | 1.000 |
| 0 | Shr 2073 - Asa 2074 | -0.03846 | 0.0111 | 0.047 |
| 1 | Shr 2074 - Asa 2075 | -0.04657 | 0.0113 | 0.013 |
| 2 | Shr 2075 - Fal 2075 | -0.03315 | 0.0107 | 0.019 |

