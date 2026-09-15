# Paired per-prompt comparison (E1-E vs matched baseline)

- paired prompts: **49,947**

| metric | E1-E | baseline | delta |
|---|---|---|---|
| search rounds | 1.6186 | 1.5999 | +0.0187 (+1.17%) |
| search queries | 2.1211 | 2.1085 | +0.0126 |
| response chars | 5550.0 | 5526.5 | +23.5 |
| EM | 0.4348 | 0.4328 | +0.0020 |

- rounds **decreased** on 4,907 prompts (9.82%), **increased** on 5,646 (11.30%), unchanged on 39,394
- sign test on round changes: p = 6.3e-13
- EM up on 2,567 / down on 2,466 prompts (sign test p = 0.155)

## per benchmark

| benchmark | n | rounds E1-E | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.980 | 1.951 | +0.028 | 10.9% | 12.8% | +0.0042 |
| bamboogle | 125 | 1.880 | 1.856 | +0.024 | 8.8% | 10.4% | -0.0240 |
| hotpotqa | 7,404 | 1.640 | 1.615 | +0.025 | 8.4% | 10.7% | +0.0009 |
| musique | 2,411 | 2.238 | 2.212 | +0.025 | 15.1% | 17.4% | +0.0021 |
| nq | 3,610 | 1.282 | 1.264 | +0.018 | 7.7% | 8.7% | -0.0003 |
| popqa | 12,509 | 1.513 | 1.499 | +0.014 | 11.3% | 12.6% | +0.0017 |
| triviaqa | 11,312 | 1.292 | 1.284 | +0.008 | 7.5% | 8.2% | +0.0017 |
