# E1-D D1 integrity check (step60 -> step70)

- experiment dir: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu`
- checkpoint: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu/global_step_70`
- tracker: 120 (expected 70)
- data.pt present: True
- steps 60..70 all present: True (missing: [])
- 'Setting global step to 60': True
- metrics parsed: 1241, non-finite: {}
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
  "candidate_groups": 18880,
  "actual_retained": 331,
  "em_metric_keep_equals_actual_keep_mismatches": 0,
  "filter_decoupling_confirmed": true,
  "counterfactual_shaped_retained": 400,
  "would_be_revived_by_shaped_filter": 69,
  "retained_all_correct": 0,
  "retained_mixed": 331,
  "retained_all_wrong": 0,
  "efficiency_active_excluded_by_em_filter": 69,
  "efficiency_active_retained_actual": 8,
  "efficiency_active_retained_shaped_filter": 77,
  "mixed_efficiency_active_retained": 8,
  "efficiency_supervision_density_actual": 0.02416918429003021,
  "generation_batches": 118
}
```

**VERDICT: PASS**

