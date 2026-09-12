# E1-B training complete — final training-side summary

run dir: `/data01/wyy/Graph-Agent-Planning/E1-B/run_20260911_113432`
training wall time: 12:00:46 (11:34 -> 23:50, 2026-09-11), exit code 0
steps: 71..120 (50 steps), 501 reward-manager calls (1280 rollouts each), 641,280 rollouts, 80,160 groups

## Integrity (analysis/post_train_integrity.md)
- VERDICT: PASS; integrity_ok=True; state_resume_ok=True
- all of steps 71..120 present in the log; 3085 metric values parsed, 0 non-finite
- full-state resume: E1-A step70 extra_state lr_scheduler.last_epoch=70 + RNG(cpu/cuda/numpy/random)
  -> E1-B step120 extra_state lr_scheduler.last_epoch=120 + RNG present
- error scan before the final-validation marker: only nccl_warn=6 (benign c10d warnings);
  0 CUDA OOM, 0 NCCL errors, 0 tracebacks, 0 wiki/tool failures
- 22 original GAP/E1-0/E1-A files: sha256 identical before and after

## Checkpoints (all verified complete: 4x model 3.40 GB, 4x optim 6.17 GB,
   4x extra_state, data.pt StatefulDataLoader state, tracker=120)

| checkpoint | window mean EM | window rounds | window queries |
|---|---|---|---|
| global_step_80 (steps 71-80) | 0.4583 | 1.4214 | 1.9727 |
| global_step_90 (steps 81-90) | 0.4559 | 1.4085 | 1.9703 |
| global_step_100 (steps 91-100) | 0.4562 | 1.3836 | 1.9556 |
| global_step_110 (steps 101-110) | 0.4564 | 1.3559 | 1.9376 |
| global_step_120 (steps 111-120) | 0.4571 | 1.3302 | 1.9180 |

(per-checkpoint tables incl. mean reward, mean efficiency bonus, active efficiency groups,
retained groups and parallel factor: E1-B/checkpoints/checkpoint_stats.{md,csv,json})

## Reference levels
- E1-A step70 (E1-B start): EM 0.4528, rounds 1.4329, queries 1.9838
- E1-0 step70 (original-reward baseline): EM 0.4539, rounds 1.4407, queries 1.9898

Training-side read: mean EM is flat (0.4559-0.4583) while search depth falls monotonically
(1.4214 -> 1.3302, -6.4%) and queries fall too (1.9727 -> 1.9180, -2.8%); the number of
groups newly retained by shaping grows (baseline-EM filter 273 -> 1412 kept), and the
efficiency supervision ratio stays ~16-21%.
