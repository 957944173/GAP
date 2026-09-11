# E1-A vs baseline (GAP step60->70 original reward): 7 benchmarks

- E1-A (shaped reward): `e1a_step70`
- baseline (original EM): `gap_step60_reference`

## EM

| benchmark | E1-A EM | baseline EM | delta |
|---|---|---|---|
| nq | 0.3780 | 0.3740 | +0.0040 |
| triviaqa | 0.5630 | 0.5630 | +0.0000 |
| popqa | 0.3970 | 0.4010 | -0.0040 |
| hotpotqa | 0.3970 | 0.3970 | +0.0000 |
| 2wikimultihopqa | 0.4380 | 0.4470 | -0.0090 |
| musique | 0.1680 | 0.1750 | -0.0070 |
| bamboogle | 0.4480 | 0.4480 | +0.0000 |
| **macro (mean of benchmarks)** | 0.3984 | 0.4007 | -0.0023 |
| **micro (mean of samples)** | 0.4317 | 0.4351 | -0.0033 |

## Trajectory / search metrics (macro over benchmarks)

| metric | E1-A | baseline | delta |
|---|---|---|---|
| search rounds | 1.6691 | 1.6546 | +0.0145 |
| search queries | 2.1312 | 2.1120 | +0.0193 |
| turns (all assistant msgs) | 2.6631 | 2.6480 | +0.0151 |
| response tokens | 322.1766 | 318.2871 | +3.8894 |
| parallel factor | 1.4032 | 1.4001 | +0.0031 |
| parallel sample rate | 0.4063 | 0.4017 | +0.0046 |
| multi-query-round rate | 0.4063 | 0.4017 | +0.0046 |
| zero-search rate | 0.0002 | 0.0002 | -0.0000 |
| no <answer> rate | 0.0042 | 0.0043 | -0.0001 |

- EM improved on: nq
- EM regressed on: popqa, 2wikimultihopqa, musique
