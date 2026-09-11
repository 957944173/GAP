# Paired per-prompt comparison (E1-A vs matched baseline)

- paired prompts: **49,947**

| metric | E1-A | baseline | delta |
|---|---|---|---|
| search rounds | 1.5999 | 1.6332 | -0.0334 (-2.04%) |
| search queries | 2.1085 | 2.1346 | -0.0260 |
| response chars | 5526.5 | 5573.8 | -47.4 |
| EM | 0.4328 | 0.4336 | -0.0008 |

- rounds **decreased** on 5,947 prompts (11.91%), **increased** on 4,868 (9.75%), unchanged on 39,132
- sign test on round changes: p = 3.21e-25
- EM up on 2,480 / down on 2,521 prompts (sign test p = 0.562)

## per benchmark

| benchmark | n | rounds E1-A | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.951 | 2.004 | -0.052 | 13.9% | 10.8% | -0.0041 |
| bamboogle | 125 | 1.856 | 1.872 | -0.016 | 8.8% | 6.4% | -0.0160 |
| hotpotqa | 7,404 | 1.615 | 1.645 | -0.030 | 10.8% | 8.5% | +0.0030 |
| musique | 2,411 | 2.212 | 2.252 | -0.039 | 18.4% | 14.5% | -0.0008 |
| nq | 3,610 | 1.264 | 1.299 | -0.035 | 9.4% | 7.8% | +0.0050 |
| popqa | 12,509 | 1.499 | 1.528 | -0.029 | 12.8% | 11.2% | -0.0010 |
| triviaqa | 11,312 | 1.284 | 1.302 | -0.018 | 8.9% | 7.5% | -0.0011 |
