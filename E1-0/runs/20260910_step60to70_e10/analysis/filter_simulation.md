# E1-0 offline simulation of the ORIGINAL GAP group filter (trainer untouched)

| metric | value |
|---|---|
| groups_before_filter | 18400 |
| groups_expected_kept_by_original_reward | 340 |
| groups_expected_kept_rate_pct | 1.8478260869565217 |
| groups_expected_removed | 18060 |
| all_correct_groups_removed | 8184 |
| all_wrong_groups_removed | 9876 |
| rule | verl ray_trainer.py:1780-1781 with metric=seq_reward/seq_final_reward and EM 0/1 rewards: kept iff np.std(group rewards) > 0 or group size == 1, i.e. iff 0 < correct_count < group_size |
| newly_retained_groups_shaped_cost_xml | {"lambda_0.05": 84, "lambda_0.10": 84, "lambda_0.20": 84, "explanation": "A_i=1 for every rollout of an all-correct group, so R_i=1+lambda*E_i varies iff the correct-costs vary; all-wrong groups stay 0 and are never revived."} |
| newly_retained_groups_shaped_cost_structured | {"lambda_0.05": 84, "lambda_0.10": 84, "lambda_0.20": 84, "explanation": "A_i=1 for every rollout of an all-correct group, so R_i=1+lambda*E_i varies iff the correct-costs vary; all-wrong groups stay 0 and are never revived."} |
