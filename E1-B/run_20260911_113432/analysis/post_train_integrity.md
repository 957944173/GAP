# E1-B post-training integrity check

- experiment dir: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu`
- checkpoint dir: `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_120`
- training log: `/data01/wyy/Graph-Agent-Planning/verl/logs/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu.log` (3.3MB)

## Checkpoint

- exists: True
- global step (from dir name): 120
- tracker value: 120
- data.pt present: True
- model_rank0: 3.2GB
- optimizer_rank0: 5.7GB
- extra_state_rank0: 14.3KB
- model_rank1: 3.2GB
- optimizer_rank1: 5.7GB
- extra_state_rank1: 14.4KB
- model_rank2: 3.2GB
- optimizer_rank2: 5.7GB
- extra_state_rank2: 14.4KB
- model_rank3: 3.2GB
- optimizer_rank3: 5.7GB
- extra_state_rank3: 14.4KB

## Training log

- steps seen: [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]
- all of step71..step120 present: True (missing: [])
- 'Setting global step to 70': True
- (the upstream 'Loaded model/optimizer/lr_scheduler from ...' INFO lines are not emitted to stdout in this run; state residency is proven directly below instead)
- metrics parsed: 3085, non-finite: {}
- final validation metrics present: True

## Optimizer / lr_scheduler / RNG full-state resume evidence

- step70_extra_state: {'path': '/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/global_step_70/actor/extra_state_world_size_4_rank_0.pt', 'keys': ['lr_scheduler', 'rng'], 'lr_scheduler_last_epoch': 70, 'rng_present': True, 'rng_substates': ['cpu', 'cuda', 'numpy', 'random']}
- step120_extra_state: {'path': '/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_120/actor/extra_state_world_size_4_rank_0.pt', 'keys': ['lr_scheduler', 'rng'], 'lr_scheduler_last_epoch': 120, 'rng_present': True, 'rng_substates': ['cpu', 'cuda', 'numpy', 'random']}
- state_resume_ok: True

### Error-pattern scan BEFORE the final-validation marker (real training window)

- cuda_oom: 0
- nccl_error: 0
- nccl_warn: 6
- ray_worker_died: 0
- nccl_unhandled: 0
- wiki_connection: 0
- wiki_http_error: 0
- wiki_search_error: 0
- wiki_tool_exec_failed: 0
- wiki_tool_bad_params: 0
- tool_error: 0
- traceback: 0
- cuda_error: 0

### Error-pattern scan AFTER the final-validation marker (expected teardown)

- cuda_oom: 0
- nccl_error: 0
- nccl_warn: 0
- ray_worker_died: 0
- nccl_unhandled: 0
- wiki_connection: 0
- wiki_http_error: 0
- wiki_search_error: 0
- wiki_tool_exec_failed: 0
- wiki_tool_bad_params: 0
- tool_error: 0
- traceback: 0
- cuda_error: 0

## VERDICT: PASS

