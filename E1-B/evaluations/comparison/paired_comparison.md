# Paired per-prompt comparison (E1-B vs matched baseline)

- paired prompts: **49,947**

| metric | E1-B | baseline | delta |
|---|---|---|---|
| search rounds | 1.2972 | 1.5999 | -0.3026 (-18.92%) |
| search queries | 1.8793 | 2.1085 | -0.2293 |
| response chars | 5088.7 | 5526.5 | -437.7 |
| EM | 0.4332 | 0.4328 | +0.0004 |

- rounds **decreased** on 11,902 prompts (23.83%), **increased** on 2,105 (4.21%), unchanged on 35,940
- sign test on round changes: p = 0
- EM up on 3,498 / down on 3,480 prompts (sign test p = 0.829)

## per benchmark

| benchmark | n | rounds E1-B | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.533 | 1.951 | -0.419 | 29.3% | 5.1% | +0.0018 |
| bamboogle | 125 | 1.680 | 1.856 | -0.176 | 27.2% | 11.2% | -0.0240 |
| hotpotqa | 7,404 | 1.374 | 1.615 | -0.241 | 22.9% | 4.6% | -0.0073 |
| musique | 2,411 | 1.807 | 2.212 | -0.406 | 35.2% | 8.8% | -0.0046 |
| nq | 3,610 | 1.063 | 1.264 | -0.201 | 17.8% | 2.2% | +0.0030 |
| popqa | 12,509 | 1.172 | 1.499 | -0.328 | 24.1% | 4.4% | +0.0055 |
| triviaqa | 11,312 | 1.086 | 1.284 | -0.198 | 17.5% | 2.4% | -0.0015 |
