# E1-D D2 integrity check (step70 -> step120)

- experiment dir: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu`
- checkpoint: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu/global_step_120`
- tracker: 120 (expected 120)
- data.pt present: True
- steps 70..120 all present: True (missing: [])
- 'Setting global step to 70': True
- metrics parsed: 6164, non-finite: {}
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

## filter decoupling (E1-D_task.md §11 / §19 Q2)

```json
{
  "available": true,
  "candidate_groups": 83680,
  "actual_retained": 1679,
  "em_metric_keep_equals_actual_keep_mismatches": 0,
  "filter_decoupling_confirmed": true,
  "counterfactual_shaped_retained": 2113,
  "would_be_revived_by_shaped_filter": 434,
  "retained_all_correct": 0,
  "retained_mixed": 1679,
  "retained_all_wrong": 0,
  "efficiency_active_excluded_by_em_filter": 434,
  "efficiency_active_retained_actual": 86,
  "efficiency_active_retained_shaped_filter": 520,
  "mixed_efficiency_active_retained": 86,
  "efficiency_supervision_density_actual": 0.05122096486003574,
  "generation_batches": 523
}
```

**VERDICT: PASS**

