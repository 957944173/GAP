# Paired per-prompt comparison (E1-D vs matched baseline)

- paired prompts: **49,947**

| metric | E1-D | baseline | delta |
|---|---|---|---|
| search rounds | 1.6256 | 1.5912 | +0.0344 (+2.16%) |
| search queries | 2.1320 | 2.0970 | +0.0350 |
| response chars | 5570.0 | 5484.5 | +85.5 |
| EM | 0.4340 | 0.4358 | -0.0019 |

- rounds **decreased** on 4,861 prompts (9.73%), **increased** on 5,830 (11.67%), unchanged on 39,256
- sign test on round changes: p = 7.14e-21
- EM up on 2,463 / down on 2,557 prompts (sign test p = 0.185)

## per benchmark

| benchmark | n | rounds E1-D | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.990 | 1.937 | +0.053 | 11.0% | 14.0% | -0.0076 |
| bamboogle | 125 | 1.864 | 1.808 | +0.056 | 5.6% | 9.6% | -0.0080 |
| hotpotqa | 7,404 | 1.634 | 1.604 | +0.030 | 8.3% | 10.2% | +0.0018 |
| musique | 2,411 | 2.236 | 2.197 | +0.039 | 15.3% | 16.9% | -0.0120 |
| nq | 3,610 | 1.287 | 1.263 | +0.024 | 7.6% | 9.2% | +0.0017 |
| popqa | 12,509 | 1.525 | 1.492 | +0.033 | 11.1% | 12.9% | -0.0021 |
| triviaqa | 11,312 | 1.301 | 1.281 | +0.020 | 7.3% | 8.4% | +0.0034 |
