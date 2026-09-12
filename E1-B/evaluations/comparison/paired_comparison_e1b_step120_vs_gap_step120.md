# Paired per-prompt comparison (E1-B vs matched baseline)

- paired prompts: **49,947**

| metric | E1-B | baseline | delta |
|---|---|---|---|
| search rounds | 1.2972 | 1.8568 | -0.5596 (-30.14%) |
| search queries | 1.8793 | 2.3473 | -0.4680 |
| response chars | 5088.7 | 6091.2 | -1002.5 |
| EM | 0.4332 | 0.4450 | -0.0118 |

- rounds **decreased** on 17,986 prompts (36.01%), **increased** on 1,037 (2.08%), unchanged on 30,924
- sign test on round changes: p = 0
- EM up on 3,229 / down on 3,819 prompts (sign test p = 2.1e-12)

## per benchmark

| benchmark | n | rounds E1-B | rounds base | Δrounds | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.533 | 2.330 | -0.797 | 46.5% | 1.9% | -0.0055 |
| bamboogle | 125 | 1.680 | 1.960 | -0.280 | 28.8% | 5.6% | -0.0400 |
| hotpotqa | 7,404 | 1.374 | 1.835 | -0.461 | 34.0% | 2.2% | -0.0208 |
| musique | 2,411 | 1.807 | 2.627 | -0.820 | 51.1% | 3.5% | -0.0145 |
| nq | 3,610 | 1.063 | 1.473 | -0.410 | 28.0% | 1.2% | -0.0080 |
| popqa | 12,509 | 1.172 | 1.720 | -0.549 | 34.7% | 2.7% | -0.0098 |
| triviaqa | 11,312 | 1.086 | 1.454 | -0.367 | 26.5% | 1.4% | -0.0156 |
