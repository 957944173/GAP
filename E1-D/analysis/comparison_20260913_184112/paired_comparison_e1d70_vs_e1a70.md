# Paired per-prompt comparison (E1-D vs matched baseline)

- paired prompts: **49,947**

| metric | E1-D | baseline | delta |
|---|---|---|---|
| search rounds | 1.6256 | 1.5999 | +0.0257 (+1.61%) |
| search queries | 2.1320 | 2.1085 | +0.0234 |
| response chars | 5570.0 | 5526.5 | +43.5 |
| EM | 0.4340 | 0.4328 | +0.0012 |

- rounds **decreased** on 4,883 prompts (9.78%), **increased** on 5,624 (11.26%), unchanged on 39,440
- sign test on round changes: p = 4.87e-13
- EM up on 2,555 / down on 2,497 prompts (sign test p = 0.414)

## per benchmark

| benchmark | n | rounds E1-D | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.990 | 1.951 | +0.038 | 10.7% | 13.0% | +0.0011 |
| bamboogle | 125 | 1.864 | 1.856 | +0.008 | 7.2% | 8.0% | -0.0080 |
| hotpotqa | 7,404 | 1.634 | 1.615 | +0.019 | 8.7% | 10.0% | +0.0016 |
| musique | 2,411 | 2.236 | 2.212 | +0.023 | 16.3% | 16.5% | -0.0050 |
| nq | 3,610 | 1.287 | 1.264 | +0.023 | 7.2% | 8.8% | -0.0028 |
| popqa | 12,509 | 1.525 | 1.499 | +0.026 | 11.0% | 12.3% | +0.0012 |
| triviaqa | 11,312 | 1.301 | 1.284 | +0.018 | 7.5% | 8.7% | +0.0035 |
