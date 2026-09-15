# E1-E D2 integrity check (step70 -> step120)

- experiment dir: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu`
- checkpoint: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_120`
- tracker: 120 (expected 120)
- data.pt present: True
- steps 70..120 all present: True (missing: [])
- 'Setting global step to 70': True
- metrics parsed: 6165, non-finite: {}
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
  "n_step_records": 50,
  "n_generation_calls": 536,
  "n_group_records": 1600,
  "n_rollout_records": 43520,
  "quota_error_lines": 0,
  "quota_error_examples": [],
  "steps_covered": [
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120
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
  "steps_short_of_design_quota": [
    {
      "step": 72,
      "quota_actual": 1,
      "design_m": 2,
      "efficiency_cache_size": 2,
      "recorded_eff_shortfall": 1,
      "mixed_kept_cum": 31
    }
  ],
  "n_steps_short_of_design_quota": 1,
  "efficiency_groups": 97,
  "mixed_groups": 1503,
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
  "wrong_rollouts": 5938,
  "zero_search_rollouts": 0,
  "zero_search_share": 0.0,
  "malformed_metadata_rollouts": 0,
  "malformed_metadata_share": 0.0,
  "rollouts_skipped_neutralised_metric": 30720,
  "rollouts_checked_for_metric_consistency": 12800,
  "advantage_nan_total": 0,
  "advantage_inf_total": 0,
  "advantage_nan_total_manager_field": 0,
  "advantage_nan_total_recomputed": 0,
  "advantage_inf_total_recomputed": 0,
  "advantage_recomputed_per_step": {
    "71": {
      "n": 2816,
      "groups": 352,
      "nan": 0,
      "inf": 0,
      "mean": -3.591666105942889e-17,
      "std": 0.28646022608255656,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "72": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -1.5439038936193583e-16,
      "std": 0.9372428970060299,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "73": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 2.5500435096859064e-16,
      "std": 0.9372424867519535,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "74": {
      "n": 2816,
      "groups": 352,
      "nan": 0,
      "inf": 0,
      "mean": 1.6006766619240537e-17,
      "std": 0.2864605297990731,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "75": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": -3.715199444383662e-17,
      "std": 0.3820037159113843,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "76": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": 1.879283765641541e-17,
      "std": 0.38200365887363263,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "77": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -1.8041124150158794e-16,
      "std": 0.9372417149650933,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "78": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -1.1709383462843448e-16,
      "std": 0.9372421246198999,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "79": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 1.4224732503009818e-16,
      "std": 0.93724194183337,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "80": {
      "n": 1536,
      "groups": 192,
      "nan": 0,
      "inf": 0,
      "mean": -6.924437874940755e-17,
      "std": 0.39950792325025847,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "81": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 9.367506770274758e-17,
      "std": 0.9372418529579237,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "82": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 9.020562075079397e-17,
      "std": 0.9372418534268592,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "83": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 3.7470027081099033e-16,
      "std": 0.93724193924657,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "84": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -2.220446049250313e-16,
      "std": 0.937241780887684,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "85": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": 5.551115123125783e-17,
      "std": 0.937242377744278,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "86": {
      "n": 256,
      "groups": 32,
      "nan": 0,
      "inf": 0,
      "mean": -2.0469737016526324e-16,
      "std": 0.9372417818018238,
      "min": -2.474866734172715,
      "max": 2.474866734172715
    },
    "87":
```

## trainer log vs manager diagnostics

```json
{
  "available": true,
  "steps_cross_checked": 50,
  "score_max_mismatches": 0,
  "examples": [],
  "log_score_max_consistent_with_quota": true
}
```

### per optimizer step

| step | gen batches | quota | mixed seen | eff seen | cache | shortfall | inserted | displaced | batch |
|---|---|---|---|---|---|---|---|---|---|
| 71 | 15 | 2 | 35 | 14 | 2 | 0 | 2 | 4 | 32 |
| 72 | 9 | 1 | 33 | 3 | 2 | 1 | 1 | 2 | 32 |
| 73 | 9 | 2 | 35 | 7 | 2 | 0 | 2 | 5 | 32 |
| 74 | 13 | 2 | 32 | 4 | 2 | 0 | 2 | 1 | 32 |
| 75 | 10 | 2 | 32 | 4 | 2 | 0 | 2 | 2 | 32 |
| 76 | 11 | 2 | 32 | 11 | 2 | 0 | 2 | 2 | 32 |
| 77 | 10 | 2 | 35 | 6 | 2 | 0 | 2 | 5 | 32 |
| 78 | 10 | 2 | 36 | 6 | 2 | 0 | 2 | 6 | 32 |
| 79 | 10 | 2 | 34 | 12 | 2 | 0 | 2 | 4 | 32 |
| 80 | 10 | 2 | 37 | 4 | 2 | 0 | 2 | 6 | 32 |
| 81 | 11 | 2 | 32 | 10 | 2 | 0 | 2 | 2 | 32 |
| 82 | 13 | 2 | 35 | 8 | 2 | 0 | 2 | 4 | 32 |
| 83 | 13 | 2 | 34 | 8 | 2 | 0 | 2 | 3 | 32 |
| 84 | 10 | 2 | 38 | 6 | 2 | 0 | 2 | 7 | 32 |
| 85 | 9 | 2 | 34 | 10 | 2 | 0 | 2 | 3 | 32 |
| 86 | 10 | 2 | 32 | 6 | 2 | 0 | 2 | 2 | 32 |
| 87 | 13 | 2 | 36 | 14 | 2 | 0 | 2 | 5 | 32 |
| 88 | 8 | 2 | 35 | 6 | 2 | 0 | 2 | 4 | 32 |
| 89 | 10 | 2 | 32 | 11 | 2 | 0 | 2 | 1 | 32 |
| 90 | 12 | 2 | 34 | 6 | 2 | 0 | 2 | 4 | 32 |
| 91 | 12 | 2 | 35 | 10 | 2 | 0 | 2 | 4 | 32 |
| 92 | 14 | 2 | 37 | 11 | 2 | 0 | 2 | 7 | 32 |
| 93 | 12 | 2 | 35 | 13 | 2 | 0 | 2 | 4 | 32 |
| 94 | 11 | 2 | 32 | 8 | 2 | 0 | 2 | 1 | 32 |
| 95 | 15 | 2 | 33 | 9 | 2 | 0 | 2 | 3 | 32 |
| 96 | 14 | 2 | 33 | 9 | 2 | 0 | 2 | 3 | 32 |
| 97 | 8 | 2 | 33 | 9 | 2 | 0 | 2 | 3 | 32 |
| 98 | 12 | 2 | 32 | 9 | 2 | 0 | 2 | 1 | 32 |
| 99 | 10 | 2 | 34 | 9 | 2 | 0 | 2 | 3 | 32 |
| 100 | 11 | 2 | 35 | 9 | 2 | 0 | 2 | 4 | 32 |
| 101 | 8 | 2 | 33 | 8 | 2 | 0 | 2 | 3 | 32 |
| 102 | 12 | 2 | 36 | 4 | 2 | 0 | 2 | 5 | 32 |
| 103 | 10 | 2 | 32 | 9 | 2 | 0 | 2 | 2 | 32 |
| 104 | 14 | 2 | 33 | 13 | 2 | 0 | 2 | 2 | 32 |
| 105 | 8 | 2 | 32 | 2 | 2 | 0 | 2 | 2 | 32 |
| 106 | 10 | 2 | 33 | 6 | 2 | 0 | 2 | 2 | 32 |
| 107 | 13 | 2 | 34 | 10 | 2 | 0 | 2 | 3 | 32 |
| 108 | 8 | 2 | 33 | 4 | 2 | 0 | 2 | 3 | 32 |
| 109 | 10 | 2 | 35 | 5 | 2 | 0 | 2 | 4 | 32 |
| 110 | 8 | 1 | 33 | 1 | 1 | 0 | 1 | 2 | 32 |
| 111 | 10 | 2 | 33 | 7 | 2 | 0 | 2 | 3 | 32 |
| 112 | 10 | 2 | 34 | 10 | 2 | 0 | 2 | 3 | 32 |
| 113 | 9 | 2 | 34 | 5 | 2 | 0 | 2 | 4 | 32 |
| 114 | 13 | 2 | 33 | 10 | 2 | 0 | 2 | 3 | 32 |
| 115 | 9 | 2 | 33 | 3 | 2 | 0 | 2 | 3 | 32 |
| 116 | 12 | 2 | 33 | 13 | 2 | 0 | 2 | 3 | 32 |
| 117 | 10 | 2 | 32 | 7 | 2 | 0 | 2 | 2 | 32 |
| 118 | 9 | 2 | 33 | 9 | 2 | 0 | 2 | 3 | 32 |
| 119 | 8 | 1 | 32 | 1 | 1 | 0 | 1 | 1 | 32 |
| 120 | 10 | 2 | 33 | 11 | 2 | 0 | 2 | 3 | 32 |

**VERDICT: PASS**

