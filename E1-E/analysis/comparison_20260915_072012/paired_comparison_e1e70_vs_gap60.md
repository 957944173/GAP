# Paired per-prompt comparison (E1-E vs matched baseline)

- paired prompts: **49,947**

| metric | E1-E | baseline | delta |
|---|---|---|---|
| search rounds | 1.6186 | 1.5912 | +0.0273 (+1.72%) |
| search queries | 2.1211 | 2.0970 | +0.0242 |
| response chars | 5550.0 | 5484.5 | +65.5 |
| EM | 0.4348 | 0.4358 | -0.0010 |

- rounds **decreased** on 4,891 prompts (9.79%), **increased** on 5,870 (11.75%), unchanged on 39,186
- sign test on round changes: p = 3.82e-21
- EM up on 2,531 / down on 2,582 prompts (sign test p = 0.476)

## per benchmark

| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.980 | 1.937 | +0.043 | 11.4% | 14.0% | -0.0045 |
| bamboogle | 125 | 1.880 | 1.808 | +0.072 | 7.2% | 9.6% | -0.0240 |
| hotpotqa | 7,404 | 1.640 | 1.604 | +0.036 | 8.0% | 11.0% | +0.0011 |
| musique | 2,411 | 2.238 | 2.197 | +0.041 | 14.8% | 17.3% | -0.0050 |
| nq | 3,610 | 1.282 | 1.263 | +0.019 | 7.6% | 8.7% | +0.0042 |
| popqa | 12,509 | 1.513 | 1.492 | +0.021 | 11.2% | 13.0% | -0.0016 |
| triviaqa | 11,312 | 1.292 | 1.281 | +0.011 | 7.4% | 8.1% | +0.0015 |
