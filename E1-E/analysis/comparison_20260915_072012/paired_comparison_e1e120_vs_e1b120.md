# Paired per-prompt comparison (E1-E vs matched baseline)

- paired prompts: **49,947**

| metric | E1-E | baseline | delta |
|---|---|---|---|
| search rounds | 1.5223 | 1.2972 | +0.2251 (+17.35%) |
| search queries | 2.0246 | 1.8793 | +0.1454 |
| response chars | 5405.6 | 5088.7 | +316.8 |
| EM | 0.4401 | 0.4332 | +0.0070 |

- rounds **decreased** on 1,867 prompts (3.74%), **increased** on 10,684 (21.39%), unchanged on 37,396
- sign test on round changes: p = 0
- EM up on 3,411 / down on 3,062 prompts (sign test p = 1.44e-05)

## per benchmark

| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.836 | 1.533 | +0.303 | 4.1% | 26.9% | +0.0084 |
| bamboogle | 125 | 1.864 | 1.680 | +0.184 | 6.4% | 22.4% | +0.0320 |
| hotpotqa | 7,404 | 1.581 | 1.374 | +0.207 | 3.6% | 21.4% | +0.0111 |
| musique | 2,411 | 2.153 | 1.807 | +0.346 | 6.5% | 33.4% | +0.0033 |
| nq | 3,610 | 1.206 | 1.063 | +0.143 | 2.2% | 14.1% | +0.0047 |
| popqa | 12,509 | 1.395 | 1.172 | +0.223 | 4.6% | 21.2% | +0.0067 |
| triviaqa | 11,312 | 1.239 | 1.086 | +0.153 | 2.3% | 15.2% | +0.0042 |
