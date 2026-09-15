# E1-E pre-training offline selftest

**ALL PASS: True**

## PART A - planner replay vs E1-E0 (all 50 optimizer steps)
- q=0 exact steps: 50/50
- q=2 exact steps: 50/50

## PART B - runtime manager replay vs E1-E0
- steps exercised: [71, 72, 73]
- same batch size: True, same group set: True
## PART D - empty-keep neutrality
- forced insertion neutral: True
- forced insertion still 32 groups: True
- real stream has empty-keep batches: True
- real empty-keep steps match E1-E0: True
- kept_nothing calls recorded: 18

## PART C - invariants

- optimizer_batch_size_all_32: True
- quota_actual_le_2: True
- inserted_groups_k8: True
- inserted_groups_cost_variation_positive: True
- stopping_invariant_across_q_batch_sizes: True
- stopping_batch_independent_of_q: True
- mixed_only_stopping: True
- mixed_group_reward_equals_em: True
- efficiency_group_reward_in_1_1.05: True
- wrong_rollout_reward_zero: True
- group_reward_std_finite: True
- no_nan_inf_rewards: True
- efficiency_ranking_correct: True

## Section 11 requirement -> check

| E1-E_task.md section 11 requirement | check | result |
|---|---|---|
| 1. every optimizer batch = 32 groups | `optimizer_batch_size_all_32` | PASS |
| 2. q <= 2 | `quota_actual_le_2` | PASS |
| 3. mixed-only stopping | `mixed_only_stopping` | PASS |
| 4. quota does not change generation stopping | `stopping_invariant_across_q_batch_sizes` | PASS |
| 4b. stopping generation batch identical for q=0..4 | `stopping_batch_independent_of_q` | PASS |
| 5. inserted groups have k = 8 | `inserted_groups_k8` | PASS |
| 6. inserted groups cost variation > 0 | `inserted_groups_cost_variation_positive` | PASS |
| 7. mixed group reward == EM | `mixed_group_reward_equals_em` | PASS |
| 8. inserted group reward in [1, 1.05] | `efficiency_group_reward_in_1_1.05` | PASS |
| 9. wrong rollout reward == 0 | `wrong_rollout_reward_zero` | PASS |
| 10. every group reward std finite | `group_reward_std_finite` | PASS |
| 11. efficiency rollout ranking correct | `efficiency_ranking_correct` | PASS |
| 12. no NaN / Inf | `no_nan_inf_rewards` | PASS |

```json
{
  "part_a_planner_matches_e1e0": true,
  "part_b_runtime_matches_e1e0": true,
  "part_d_empty_keep_neutral": true,
  "part_c_invariants": true
}
```
