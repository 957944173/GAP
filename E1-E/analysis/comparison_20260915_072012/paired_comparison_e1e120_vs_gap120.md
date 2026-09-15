# Paired per-prompt comparison (E1-E vs matched baseline)

- paired prompts: **49,947**

| metric | E1-E | baseline | delta |
|---|---|---|---|
| search rounds | 1.5223 | 1.8568 | -0.3345 (-18.01%) |
| search queries | 2.0246 | 2.3473 | -0.3226 |
| response chars | 5405.6 | 6091.2 | -685.6 |
| EM | 0.4401 | 0.4450 | -0.0048 |

- rounds **decreased** on 13,269 prompts (26.57%), **increased** on 2,980 (5.97%), unchanged on 33,698
- sign test on round changes: p = 0
- EM up on 2,974 / down on 3,215 prompts (sign test p = 0.00219)

## per benchmark

| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.836 | 2.330 | -0.494 | 34.3% | 6.1% | +0.0029 |
| bamboogle | 125 | 1.864 | 1.960 | -0.096 | 15.2% | 5.6% | -0.0080 |
| hotpotqa | 7,404 | 1.581 | 1.835 | -0.255 | 23.4% | 5.7% | -0.0097 |
| musique | 2,411 | 2.153 | 2.627 | -0.474 | 36.8% | 9.0% | -0.0112 |
| nq | 3,610 | 1.206 | 1.473 | -0.266 | 22.0% | 4.2% | -0.0033 |
| popqa | 12,509 | 1.395 | 1.720 | -0.326 | 26.2% | 7.0% | -0.0030 |
| triviaqa | 11,312 | 1.239 | 1.454 | -0.214 | 19.8% | 4.7% | -0.0113 |
