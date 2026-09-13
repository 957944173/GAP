# E1-D live filter-decoupling observation (D1, first 9 generation batches)

Recorded 2026-09-13T00:17:28+08:00 from `/data01/wyy/Graph-Agent-Planning/E1-D/runs/D1_20260912_235330/diagnostics/groups_pid*.jsonl` while D1 was still running.

| quantity | value |
|---|---|
| candidate groups | 1440 |
| retained by the actual EM filter | 26 |
| retained by the counterfactual shaped filter | 29 |
| actual != EM-metric keep (mismatches) | 0 |
| retained: all-correct / mixed / all-wrong | 0 / 26 / 0 |
| efficiency-active groups excluded by the EM filter | 3 |
| efficiency-active groups actually retained | 0 |
| efficiency supervision density (retained active / retained) | 0.0000 |

Interpretation: the filter channel is closed exactly as designed (0 all-correct groups
survive, 0 mismatches between the actual keep decisions and the EM-metric rule) while the
shaped reward is still what the reward manager writes into token_level_rewards.  The early
window also shows the *cost* of decoupling that E1-C predicted: no efficiency-active group
has been retained yet, i.e. the efficiency supervision is far sparser than under the shaped
filter (E1-C: 291/1685 = 17.3% there).
