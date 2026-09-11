# E1-A vs baseline (GAP step60->70 original reward): 7 benchmarks

- E1-A (shaped reward): `e1a_step70`
- baseline (original EM): `baseline_e10_step70`

## EM

| benchmark | E1-A EM | baseline EM | delta |
|---|---|---|---|
| nq | 0.3780 | 0.3730 | +0.0050 |
| triviaqa | 0.5630 | 0.5640 | -0.0010 |
| popqa | 0.3970 | 0.3980 | -0.0010 |
| hotpotqa | 0.3970 | 0.3940 | +0.0030 |
| 2wikimultihopqa | 0.4380 | 0.4420 | -0.0040 |
| musique | 0.1680 | 0.1690 | -0.0010 |
| bamboogle | 0.4480 | 0.4640 | -0.0160 |
| **macro (mean of benchmarks)** | 0.3984 | 0.4006 | -0.0021 |
| **micro (mean of samples)** | 0.4317 | 0.4325 | -0.0008 |

## Trajectory / search metrics (macro over benchmarks)

| metric | E1-A | baseline | delta |
|---|---|---|---|
| search rounds | 1.6691 | 1.6997 | -0.0306 |
| search queries | 2.1312 | 2.1459 | -0.0147 |
| turns (all assistant msgs) | 2.6631 | 2.6917 | -0.0286 |
| response tokens | 322.1766 | 322.7029 | -0.5263 |
| parallel factor | 1.4032 | 1.3884 | +0.0148 |
| parallel sample rate | 0.4063 | 0.3944 | +0.0119 |
| multi-query-round rate | 0.4063 | 0.3944 | +0.0119 |
| zero-search rate | 0.0002 | 0.0002 | -0.0000 |
| no <answer> rate | 0.0042 | 0.0056 | -0.0014 |

- EM improved on: nq, hotpotqa
- EM regressed on: triviaqa, popqa, 2wikimultihopqa, musique, bamboogle
