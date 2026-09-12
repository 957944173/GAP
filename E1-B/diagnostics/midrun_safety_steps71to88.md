# E1-B mid-run safety investigation — steps 71–88 (224,000 rollouts, 28,000 groups)

Recorded 2026-09-11 ~15:45 while training continues (step ~89/120). Source:
`e1b_analyze_training.py --run-dir <E1-B run>` → `diagnostics/analysis_midrun/`.

## Run-level aggregate

| metric | E1-B steps71–88 | E1-0 baseline steps61–70 (original EM reward) | delta |
|---|---|---|---|
| rollouts / groups | 224,000 / 28,000 | 147,200 / 18,400 | — |
| **mean EM** | **0.45645** | 0.45391 | **+0.25 pp** |
| mean shaped reward | 0.45655 | — | — |
| mean efficiency bonus E | 0.002158 | — | — |
| efficiency > 0 rate | 0.216 % | — | — |
| **mean logical_search_batches** | **1.41294** | 1.44073 | **−1.93 %** |
| **mean search queries** | **1.97306** | 1.98976 | **−0.84 %** |
| mean parallel factor | 1.52306 | 1.50911 | +0.93 % |
| parallel sample rate | 0.49323 | 0.48291 | +1.03 pp |
| mean assistant turns | 2.41287 | 2.44059 | −1.14 % |
| mean response tokens | 1273.06 | 1277.76 | −0.37 % |
| zero-search rate | 0.0286 % | 0.0163 % | +0.012 pp |
| **zero-search-and-correct** | **0** | 0 | — |
| metadata_invalid | 0 | 0 | — |
| retained groups, baseline-EM filter | 493 | 340 | — |
| retained groups, shaped filter | 607 | — | — |
| groups newly retained by shaping | 114 | — | — |
| active efficiency groups | 125 | 95 | — |
| efficiency supervision ratio | 20.6 % | 16.6 % (E1-A) | — |

**Verdict: no accuracy loss, no query inflation, no zero-search shortcut, and the depth
reduction is genuine (rounds −1.9 % AND queries −0.8 %).**

## Query-packing check (§7 item 5) — investigated in depth

The per-step monitor raised 11 `PACKING_SUSPECT_RATIO` alerts. Because only 3–11 groups per
step are "active" (≥ 2 correct rollouts with differing cost), a single suspect group gives
1/6 = 16.7 %, so the per-step flag is statistically meaningless on its own.

Run-level (the meaningful aggregate):

| quantity | E1-B | E1-0 baseline | E1-A steps61–70 |
|---|---|---|---|
| active groups | 125 | 95 | 58 |
| packing-suspect groups | 10 | 4 | 2 |
| **packing-suspect ratio** | **8.0 %** | 4.2 % | 3.4 % |
| efficient correct rollouts: mean rounds | 1.376 | 1.488 | — |
| inefficient correct rollouts: mean rounds | 2.468 | 2.492 | — |
| **efficient correct rollouts: mean queries** | **1.896** | 2.040 | — |
| **inefficient correct rollouts: mean queries** | **2.959** | 2.932 | — |
| efficient mean parallel factor | 1.379 | 1.371 | — |
| inefficient mean parallel factor | 1.199 | 1.177 | — |
| efficient mean tokens | 1248.8 | 1343.2 | — |
| inefficient mean tokens | 1841.9 | 1828.8 | — |

Interpretation:

* the aggregate shows the fewer-round rollouts also use **0.9–1.1 fewer queries** on average
  (1.90 vs 2.96) and ~590 fewer tokens, i.e. the shaping is shortening the *whole* search
  trajectory, not re-bundling the same queries into fewer rounds;
* the suspect ratio is a small-count tail statistic: 10 of 125 active groups, and the
  windowed ratio moves between 5.5 % (steps 71–80) and 11.8 % (steps 81–88) on 51–73 groups;
* it is nonetheless **higher than E1-0 (4.2 %) and E1-A (3.4 %) and must keep being
  tracked**; the E1-A stop criterion was "packing-suspect > 10 %", which the second window
  touches. It is a *watch item*, not a stop: the same data shows queries falling, not rising,
  and the definitive test is the 51,201-sample step120 evaluation
  (`search_queries_mean` up while rounds fall would be the real failure signature).

## Threshold status vs the E1-A success/stop criteria

| criterion | threshold | current | status |
|---|---|---|---|
| exit success: EM drop | ≤ 0.5 pp | +0.25 pp (improvement) | OK |
| exit success: rounds | ≤ −1.5 % | −1.93 % | OK |
| stop: EM | −1 pp | +0.25 pp | not triggered |
| stop: packing-suspect | > 10 % | 8.0 % run-level / 11.8 % in one window | **watch** |
| stop: parallel factor | +5 % | +0.93 % | not triggered |
