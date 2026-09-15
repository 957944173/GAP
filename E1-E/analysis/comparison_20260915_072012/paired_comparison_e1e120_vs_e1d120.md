# Paired per-prompt comparison (E1-E vs matched baseline)

- paired prompts: **49,947**

| metric | E1-E | baseline | delta |
|---|---|---|---|
| search rounds | 1.5223 | 1.7270 | -0.2047 (-11.85%) |
| search queries | 2.0246 | 2.2382 | -0.2136 |
| response chars | 5405.6 | 5863.9 | -458.3 |
| EM | 0.4401 | 0.4392 | +0.0009 |

- rounds **decreased** on 10,285 prompts (20.59%), **increased** on 3,913 (7.83%), unchanged on 35,749
- sign test on round changes: p = 0
- EM up on 3,127 / down on 3,082 prompts (sign test p = 0.568)

## per benchmark

| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.836 | 2.155 | -0.319 | 26.8% | 7.8% | +0.0076 |
| bamboogle | 125 | 1.864 | 2.064 | -0.200 | 19.2% | 4.8% | +0.0000 |
| hotpotqa | 7,404 | 1.581 | 1.749 | -0.168 | 18.5% | 7.1% | -0.0022 |
| musique | 2,411 | 2.153 | 2.428 | -0.275 | 29.0% | 12.1% | -0.0100 |
| nq | 3,610 | 1.206 | 1.340 | -0.134 | 15.5% | 6.5% | -0.0014 |
| popqa | 12,509 | 1.395 | 1.585 | -0.191 | 20.4% | 9.5% | +0.0018 |
| triviaqa | 11,312 | 1.239 | 1.364 | -0.125 | 15.0% | 6.1% | -0.0025 |
