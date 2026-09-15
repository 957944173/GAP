# Paired per-prompt comparison (E1-E vs matched baseline)

- paired prompts: **49,947**

| metric | E1-E | baseline | delta |
|---|---|---|---|
| search rounds | 1.5223 | 1.6186 | -0.0962 (-5.94%) |
| search queries | 2.0246 | 2.1211 | -0.0965 |
| response chars | 5405.6 | 5550.0 | -144.4 |
| EM | 0.4401 | 0.4348 | +0.0053 |

- rounds **decreased** on 7,746 prompts (15.51%), **increased** on 5,318 (10.65%), unchanged on 36,883
- sign test on round changes: p = 3.85e-100
- EM up on 3,299 / down on 3,033 prompts (sign test p = 0.000829)

## per benchmark

| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.836 | 1.980 | -0.144 | 18.6% | 12.6% | +0.0060 |
| bamboogle | 125 | 1.864 | 1.880 | -0.016 | 14.4% | 12.8% | +0.0320 |
| hotpotqa | 7,404 | 1.581 | 1.640 | -0.059 | 13.3% | 9.7% | +0.0028 |
| musique | 2,411 | 2.153 | 2.238 | -0.085 | 21.2% | 17.8% | -0.0033 |
| nq | 3,610 | 1.206 | 1.282 | -0.076 | 12.6% | 7.4% | +0.0080 |
| popqa | 12,509 | 1.395 | 1.513 | -0.119 | 17.2% | 11.4% | +0.0106 |
| triviaqa | 11,312 | 1.239 | 1.292 | -0.053 | 11.3% | 7.8% | +0.0011 |
