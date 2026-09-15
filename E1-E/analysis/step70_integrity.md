# E1-E D1 integrity check (step60 -> step70)

- experiment dir: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu`
- checkpoint: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_70`
- tracker: 120 (expected 70)
- data.pt present: True
- steps 60..70 all present: True (missing: [])
- 'Setting global step to 60': True
- metrics parsed: 1240, non-finite: {}
- full-state resume (lr_scheduler/RNG): True

## error scan before the final-validation marker

- cuda_oom: 0
- nccl_error: 0
- nccl_warn: 12
- ray_worker_died: 0
- wiki_search_error: 0
- wiki_tool_exec_failed: 0
- traceback: 0
- cuda_error: 0

## runtime quota semantics (E1-E_task.md section 11/12)

```json
{
  "available": true,
  "n_step_records": 10,
  "n_generation_calls": 121,
  "n_group_records": 320,
  "n_rollout_records": 8960,
  "quota_error_lines": 0,
  "quota_error_examples": [],
  "steps_covered": [
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70
  ],
  "steps_missing": [],
  "all_steps_recorded": true,
  "steps_with_wrong_batch_size": [],
  "batch_size_definition": "step-level groups = mixed_kept_cum + quota_actual (must be 32); `stopping_call_rollouts` is only that call's materialised rollouts",
  "calls_with_quota_actual_gt_2": 0,
  "calls_with_wrong_quota_target": 0,
  "steps_without_exactly_one_stopping_call": [],
  "calls_where_stopping_disagrees_with_mixed_count": 0,
  "non_stopping_calls_with_insertions": 0,
  "quota_formula_holds_all_steps": true,
  "quota_formula_violations": [],
  "steps_short_of_design_quota": [],
  "n_steps_short_of_design_quota": 0,
  "efficiency_groups": 20,
  "mixed_groups": 300,
  "efficiency_groups_with_k_not_8": 0,
  "efficiency_groups_without_cost_variation": 0,
  "efficiency_groups_reward_out_of_1_1.05": 0,
  "efficiency_groups_em_mean_not_1": 0,
  "mixed_groups_reward_out_of_0_1": 0,
  "mixed_groups_reward_mean_not_equal_em": 0,
  "rollouts_mixed_reward_not_equal_em": 0,
  "rollouts_wrong_with_nonzero_reward": 0,
  "rollouts_efficiency_reward_or_em_wrong": 0,
  "rollouts_published_metric_inconsistent": 0,
  "rollouts_reward_tensor_mismatch": 0,
  "wrong_rollouts": 1182,
  "zero_search_rollouts": 0,
  "zero_search_share": 0.0,
  "malformed_metadata_rollouts": 0,
  "malformed_metadata_share": 0.0,
  "rollouts_skipped_neutralised_metric": 6400,
  "rollouts_checked_for_metric_consistency": 2560,
  "advantage_nan_total": 0,
  "advantage_inf_total": 0,
  "advantage_nan_total_manager_field": 0,
  "advantage_nan_total_recomputed": 0,
  "advantage_inf_total_recomputed": 0,
  "advantage_recomputed_per_step": {
    "61": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 5.204170427930421e-18,
      "std": 0.9372424642932687,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "62": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": -2.8912057932946786e-18,
      "std": 0.3820038102751492,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "63": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -6.765421556309548e-17,
      "std": 0.9372417363323738,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "64": {
      "n": 1536,
      "groups": 191,
      "nan": 0,
      "inf": 0,
      "mean": 1.3877787807814457e-17,
      "std": 0.3945861195366121,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "65": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 1.1796119636642288e-16,
      "std": 0.9372423253232471,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "66": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": -4.669297356170906e-17,
      "std": 0.38200373605179877,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "67": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": 2.6020852139652106e-17,
      "std": 0.38792608365169307,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "68": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 1.1102230246251565e-16,
      "std": 0.9372422653238279,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "69": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": -4.85722573273506e-17,
      "std": 0.38200349923690646,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "70": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -1.717376241217039e-16,
      "std": 0.9372417545745501,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    }
  },
  "manager_calls_served": 121,
  "manager_calls_recorded": 121,
  "unrecorded_calls": [],
  "n_unrecorded_calls": 0,
  "oversized_returned_batches": [],
  "n_oversized_returned_batches": 0,
  "kept_nothing_calls": 5,
  "all_manager_calls_recorded": true,
  "steps_with_generation_batch_gaps": {},
  "calls_returning_unexpected_rollout_counts": [],
  "n_calls_returning_unexpected_rollout_counts": 0,
  "derived_inserted_ids_per_step": {
    "61": [
      "61:158926",
      "61:38894"
    ],
    "62": [
      "62:87334",
      "62:81395"
    ],
    "63": [
      "63:151066",
      "63:44677"
    ],
    "64": [
      "64:1502",
      "64:142018"
    ],
    "65": [
      "65:111687",
      "65:162331"
    ],
    "66": [
      "66:102841",
      "66:116730"
    ],
    "67": [
      "67:122020",
      "67:135360"
    ],
    "68": [
      "68:93312",
      "68:165073"
    ],
    "69": [
      "69:118891",
      "69:145928"
    ],
    "70": [
      "70:93945",
      "70:11671"
    ]
  },
  "steps_with_inconsistent_inserted_ids": [],
  "id_list_note": "inserted ids are re-derived from quota_groups records of the stopping call; the D1 manager field repeated one id due to a leaked loop variable (fixed for D2)",
  "quota_fill_rate": 1.0,
  "steps_full_quota": 10,
  "steps_used": 10,
  "total_inserted_efficiency_groups": 20,
  "inserted_ids_distinct_every_step": true,
  "total_mixed_dropped": 31,
  "total_displaced_mixed_groups": 30,
  "total_eff_shortfall": 0,
  "steps_needing_injection": 10,
  "total_generation_batches": 121,
  "mean_generation_batches_per_step": 12.1,
  "quota_semantics_ok": true
}
```

## trainer log vs manager diagnostics

```json
{
  "available": true,
  "steps_cross_checked": 10,
  "score_max_mismatches": 0,
  "examples": [],
  "log_score_max_consistent_with_quota": true
}
```

### per optimizer step

| step | gen batches | quota | mixed seen | eff seen | cache | shortfall | inserted | displaced | batch |
|---|---|---|---|---|---|---|---|---|---|
| 61 | 12 | 2 | 33 | 7 | 2 | 0 | 2 | 3 | 32 |
| 62 | 11 | 2 | 33 | 2 | 2 | 0 | 2 | 3 | 32 |
| 63 | 10 | 2 | 32 | 3 | 2 | 0 | 2 | 2 | 32 |
| 64 | 12 | 2 | 33 | 10 | 2 | 0 | 2 | 3 | 32 |
| 65 | 12 | 2 | 33 | 12 | 2 | 0 | 2 | 3 | 32 |
| 66 | 14 | 2 | 33 | 11 | 2 | 0 | 2 | 2 | 32 |
| 67 | 14 | 2 | 34 | 5 | 2 | 0 | 2 | 4 | 32 |
| 68 | 9 | 2 | 33 | 5 | 2 | 0 | 2 | 3 | 32 |
| 69 | 14 | 2 | 32 | 9 | 2 | 0 | 2 | 2 | 32 |
| 70 | 13 | 2 | 35 | 11 | 2 | 0 | 2 | 5 | 32 |

**VERDICT: PASS**

