# Paired per-prompt comparison (E1-D vs matched baseline)

- paired prompts: **49,947**

| metric | E1-D | baseline | delta |
|---|---|---|---|
| search rounds | 1.7270 | 1.2972 | +0.4298 (+33.13%) |
| search queries | 2.2382 | 1.8793 | +0.3590 |
| response chars | 5863.9 | 5088.7 | +775.2 |
| EM | 0.4392 | 0.4332 | +0.0061 |

- rounds **decreased** on 1,345 prompts (2.69%), **increased** on 15,141 (30.31%), unchanged on 33,461
- sign test on round changes: p = 0
- EM up on 3,601 / down on 3,297 prompts (sign test p = 0.000252)

## per benchmark

| benchmark | n | rounds E1-D | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 2.155 | 1.533 | +0.622 | 2.6% | 39.6% | +0.0009 |
| bamboogle | 125 | 2.064 | 1.680 | +0.384 | 5.6% | 34.4% | +0.0320 |
| hotpotqa | 7,404 | 1.749 | 1.374 | +0.375 | 2.7% | 29.8% | +0.0132 |
| musique | 2,411 | 2.428 | 1.807 | +0.622 | 5.1% | 44.0% | +0.0133 |
| nq | 3,610 | 1.340 | 1.063 | +0.277 | 1.7% | 21.7% | +0.0061 |
| popqa | 12,509 | 1.585 | 1.172 | +0.414 | 3.3% | 28.7% | +0.0049 |
| triviaqa | 11,312 | 1.364 | 1.086 | +0.278 | 1.9% | 21.9% | +0.0067 |
